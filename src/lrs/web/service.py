"""Cached recommendation service for the Flask UI."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from lrs.config import FEATURES_DIR, MODELS_DIR, NEETCODE150_PATH, PROCESSED_DIR
from lrs.data.preprocessor import hash_user_id
from lrs.models.baseline.als import ALSRecommender
from lrs.models.baseline.content_based import ContentBasedRecommender
from lrs.models.ensemble import EnsembleRecommender
from lrs.recommendation.graph_reranker import SimilarityGraphReranker
from lrs.recommendation.peer_stats import compute_peer_tag_scores
from lrs.recommendation.pipeline import generate_for_user, slug_to_user_id
from lrs.recommendation.tiers import TierAssigner
from lrs.web.explanations import _user_tag_scores, build_signals, slug_to_title
from lrs.web.neetcode150 import neetcode_cards

_store: "RecommendationStore | None" = None


@dataclass
class ProblemCard:
    slug: str
    title: str
    tags: list[str]
    tier: str
    p_solve: float
    signals: list[str]
    signal_count: int


@dataclass
class RecommendResponse:
    user_slug: str
    user_id: str
    warm_start: bool
    cold_start: bool = False
    contests: int = 0
    attempted: int = 0
    solved_count: int = 0
    problems: list[ProblemCard] = field(default_factory=list)
    error: str | None = None
    message: str | None = None


class RecommendationStore:
    """Loads artifacts once and serves recommendations."""

    def __init__(self) -> None:
        self.interactions: pd.DataFrame | None = None
        self.problems: pd.DataFrame | None = None
        self.user_features: pd.DataFrame | None = None
        self.ensemble: EnsembleRecommender | None = None
        self.graph: SimilarityGraphReranker | None = None
        self.title_map: dict[str, str] = {}
        self.slug_neighbors: dict[str, set[str]] = {}
        self._known_user_ids: set[str] | None = None
        self._ready = False

    def _load_known_user_ids(self) -> set[str]:
        if self._known_user_ids is not None:
            return self._known_user_ids

        path = PROCESSED_DIR / "interactions.parquet"
        if not path.exists():
            self._known_user_ids = set()
            return self._known_user_ids

        df = pd.read_parquet(path, columns=["user_id"])
        self._known_user_ids = set(df["user_id"].unique())
        return self._known_user_ids

    def _check_data(self) -> str | None:
        if not NEETCODE150_PATH.exists():
            return f"NeetCode 150 list not found at {NEETCODE150_PATH}"
        return None

    def _check_warm_data(self) -> str | None:
        if not (PROCESSED_DIR / "interactions.parquet").exists():
            return (
                "Missing processed data for personalized recommendations. "
                "Run: .venv/bin/python scripts/build_dataset.py --input combined_contest_data.jsonl"
            )
        if not (MODELS_DIR / "ensemble.pkl").exists() and not (MODELS_DIR / "als").exists():
            return "Missing trained models. Run: .venv/bin/python scripts/train_baseline.py --model all"
        return None

    def ensure_loaded(self) -> str | None:
        err = self._check_warm_data()
        if err:
            return err
        if self._ready:
            return None

        self.interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
        self.problems = pd.read_parquet(PROCESSED_DIR / "problems_clean.parquet")
        uf_path = FEATURES_DIR / "user_features.parquet"
        self.user_features = pd.read_parquet(uf_path) if uf_path.exists() else pd.DataFrame()

        self.title_map = dict(
            zip(self.problems["problem_id"], self.problems["title"], strict=False)
        )

        self.ensemble = self._load_ensemble()
        self.ensemble.set_user_history(self.interactions)
        self.graph = SimilarityGraphReranker()

        for slug, neighbors in self.graph._slug_to_neighbors.items():
            self.slug_neighbors[slug] = neighbors

        self._known_user_ids = set(self.interactions["user_id"].unique())
        self._ready = True
        return None

    def _load_ensemble(self) -> EnsembleRecommender:
        ensemble_path = MODELS_DIR / "ensemble.pkl"
        if ensemble_path.exists():
            with open(ensemble_path, "rb") as f:
                return pickle.load(f)

        als = ALSRecommender.load(MODELS_DIR / "als")
        content = ContentBasedRecommender.load(MODELS_DIR / "content")
        graph = SimilarityGraphReranker()
        ensemble = EnsembleRecommender()
        ensemble.add_models(als, content, graph)
        return ensemble

    def _cold_start_response(self, user_slug: str, user_id: str, count: int) -> RecommendResponse:
        raw_cards = neetcode_cards(count, str(NEETCODE150_PATH))
        cards = [
            ProblemCard(
                slug=c["slug"],
                title=c["title"],
                tags=c["tags"],
                tier=c["tier"],
                p_solve=c["p_solve"],
                signals=c["signals"],
                signal_count=c["signal_count"],
            )
            for c in raw_cards
        ]
        return RecommendResponse(
            user_slug=user_slug,
            user_id=user_id,
            warm_start=False,
            cold_start=True,
            problems=cards,
            message=(
                f"'{user_slug}' is not in contest data yet. "
                "Showing the NeetCode 150 curated path as a starting roadmap."
            ),
        )

    def recommend(self, user_slug: str, count: int = 10) -> RecommendResponse:
        err = self._check_data()
        if err:
            return RecommendResponse(
                user_slug=user_slug,
                user_id="",
                warm_start=False,
                error=err,
            )

        user_slug = user_slug.strip()
        user_id = slug_to_user_id(user_slug)
        warm = user_id in self._load_known_user_ids()

        if not warm:
            return self._cold_start_response(user_slug, user_id, count)

        load_err = self.ensure_loaded()
        if load_err:
            return RecommendResponse(
                user_slug=user_slug,
                user_id=user_id,
                warm_start=False,
                error=load_err,
            )

        assert self.interactions is not None
        assert self.problems is not None
        assert self.ensemble is not None
        assert self.graph is not None

        user_hist = self.interactions[self.interactions["user_id"] == user_id]
        contests = int(user_hist["contest_id"].nunique())
        attempted = int(user_hist["problem_id"].nunique())
        solved_slugs = set(user_hist[user_hist["solved"]]["problem_id"].unique())

        tier_size = max(count // 3, 3)
        assigner = TierAssigner(tier_size=tier_size)
        raw = generate_for_user(
            user_id,
            self.ensemble,
            self.interactions,
            self.problems,
            self.user_features if self.user_features is not None else pd.DataFrame(),
            tier_assigner=assigner,
        )

        uf = (
            self.user_features[self.user_features["user_id"] == user_id]
            if self.user_features is not None and not self.user_features.empty
            else pd.DataFrame()
        )
        user_feat = uf.iloc[0] if not uf.empty else None
        user_tag_scores = _user_tag_scores(user_feat)
        peer_tag_scores = compute_peer_tag_scores(self.interactions, user_id)

        history = user_hist.to_dict("records")
        struggle_slugs = self.graph.struggle_problems(history)

        ordered: list[tuple[str, dict]] = []
        for tier in ("edge_of_competence", "blind_spots", "confidence_builders"):
            for item in raw.get(tier, []):
                ordered.append((tier, item))

        cards: list[ProblemCard] = []
        for tier, item in ordered[:count]:
            slug = item["slug"]
            tags = item.get("tags") or []
            p_solve = float(item.get("p_solve", 0))
            signals = build_signals(
                slug=slug,
                tags=tags,
                tier=tier,
                p_solve=p_solve,
                user_tag_scores=user_tag_scores,
                peer_tag_scores=peer_tag_scores,
                slug_neighbors=self.slug_neighbors,
                solved_slugs=solved_slugs,
                struggle_slugs=struggle_slugs,
                title_map=self.title_map,
            )
            cards.append(
                ProblemCard(
                    slug=slug,
                    title=slug_to_title(slug, self.title_map),
                    tags=tags,
                    tier=tier,
                    p_solve=p_solve,
                    signals=signals,
                    signal_count=len(signals),
                )
            )

        return RecommendResponse(
            user_slug=user_slug,
            user_id=user_id,
            warm_start=True,
            cold_start=False,
            contests=contests,
            attempted=attempted,
            solved_count=len(solved_slugs),
            problems=cards,
        )


def get_store() -> RecommendationStore:
    global _store
    if _store is None:
        _store = RecommendationStore()
    return _store
