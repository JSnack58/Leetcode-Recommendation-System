"""End-to-end recommendation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lrs.data.preprocessor import hash_user_id
from lrs.models.ensemble import EnsembleRecommender
from lrs.recommendation.filters import apply_filters
from lrs.recommendation.peer_stats import compute_peer_tag_scores
from lrs.recommendation.ranker import RecommendationRanker
from lrs.recommendation.tiers import TierAssigner


def _as_tag_list(tags) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        import ast

        try:
            tags = ast.literal_eval(tags)
        except (ValueError, SyntaxError):
            return []
    if hasattr(tags, "__iter__") and not isinstance(tags, str):
        return [str(t) for t in tags]
    return []


def build_problem_tags(problems_df: pd.DataFrame) -> dict[str, list[str]]:
    tags_map: dict[str, list[str]] = {}
    for _, row in problems_df.iterrows():
        tags_map[row["problem_id"]] = _as_tag_list(row.get("tags"))
    return tags_map


def generate_for_user(
    user_id: str,
    ensemble: EnsembleRecommender,
    interactions: pd.DataFrame,
    problems_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    tier_assigner: TierAssigner | None = None,
    top_candidates: int = 500,
) -> dict:
    """Generate tiered recommendations for one user."""
    tier_assigner = tier_assigner or TierAssigner()
    ranker = RecommendationRanker(ranking_strategy="diversity")

    problem_tags = build_problem_tags(problems_df)
    all_problems = list(problems_df["problem_id"].unique())

    user_history = interactions[interactions["user_id"] == user_id]
    solved = set(user_history[user_history["solved"]]["problem_id"].unique())
    unsolved = [p for p in all_problems if p not in solved]

    if not unsolved:
        return {
            "user_id": user_id,
            "edge_of_competence": [],
            "blind_spots": [],
            "confidence_builders": [],
        }

    # Score pool (cap for speed)
    if len(unsolved) > top_candidates:
        # Prioritize problems in tags user has attempted
        attempted_tags: set[str] = set()
        for tags in user_history["tags"].dropna():
            if isinstance(tags, list):
                attempted_tags.update(tags)
        scored_pref = [
            p
            for p in unsolved
            if attempted_tags & set(problem_tags.get(p, []))
        ]
        pool = scored_pref[:top_candidates]
        if len(pool) < top_candidates:
            rest = [p for p in unsolved if p not in pool]
            pool.extend(rest[: top_candidates - len(pool)])
        candidates = pool
    else:
        candidates = unsolved

    scores = ensemble.predict(user_id, candidates)
    candidates, scores = apply_filters(candidates, scores, solved=solved)

    user_feat = None
    uf = user_features_df[user_features_df["user_id"] == user_id]
    if not uf.empty:
        user_feat = uf.iloc[0]

    peer_tag_scores = compute_peer_tag_scores(interactions, user_id)
    median_finish = None
    if user_feat is not None and pd.notna(user_feat.get("median_finish_time")):
        median_finish = float(user_feat["median_finish_time"])

    prob_finish: dict[str, float] = {}
    if "median_finish_time" in problems_df.columns:
        for _, row in problems_df.iterrows():
            if pd.notna(row.get("median_finish_time")):
                prob_finish[row["problem_id"]] = float(row["median_finish_time"])

    tiers = tier_assigner.assign_tiers(
        user_id,
        candidates,
        scores,
        user_features=user_feat,
        peer_tag_scores=peer_tag_scores,
        problem_tags=problem_tags,
        user_median_finish=median_finish,
        problem_median_finish=prob_finish,
    )

    def enrich(pids: list[str], tier_scores: np.ndarray | None = None) -> list[dict]:
        out = []
        for pid in pids:
            idx = candidates.index(pid) if pid in candidates else -1
            p_score = float(scores[idx]) if idx >= 0 else 0.0
            out.append(
                {
                    "slug": pid,
                    "p_solve": round(p_score, 4),
                    "tags": problem_tags.get(pid, []),
                }
            )
        return out

    # Re-rank each tier with diversity
    def tier_list(pids: list[str]) -> list[str]:
        if not pids:
            return []
        idxs = [candidates.index(p) for p in pids if p in candidates]
        tier_scores = scores[idxs]
        ranked = ranker.rank_by_diversity(pids, tier_scores, problem_tags)
        return [r.problem_id for r in ranked]

    edge_ids = tier_list(tiers.edge_of_competence)
    blind_ids = tier_list(tiers.blind_spots)
    conf_ids = tier_list(tiers.confidence_builders)

    return {
        "user_id": user_id,
        "edge_of_competence": enrich(edge_ids),
        "blind_spots": enrich(blind_ids),
        "confidence_builders": enrich(conf_ids),
    }


def slug_to_user_id(user_slug: str, salt: str = "lrs_v1") -> str:
    return hash_user_id(user_slug, salt)


def save_recommendations(result: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
