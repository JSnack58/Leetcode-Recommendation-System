"""Ensemble recommender combining ALS, content, and graph boosts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lrs.config import ENSEMBLE_W_ALS, ENSEMBLE_W_CONTENT, GRAPH_BOOST_ALPHA
from lrs.models.base import BaseRecommender
from lrs.models.baseline.als import ALSRecommender
from lrs.models.baseline.content_based import ContentBasedRecommender
from lrs.recommendation.graph_reranker import SimilarityGraphReranker
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleRecommender(BaseRecommender):
    """Weighted ALS + content with optional graph reranking."""

    def __init__(
        self,
        w_als: float = ENSEMBLE_W_ALS,
        w_content: float = ENSEMBLE_W_CONTENT,
        graph_alpha: float = GRAPH_BOOST_ALPHA,
    ):
        self.w_als = w_als
        self.w_content = w_content
        self.als_model: ALSRecommender | None = None
        self.content_model: ContentBasedRecommender | None = None
        self.graph_reranker: SimilarityGraphReranker | None = None
        self.graph_alpha = graph_alpha
        self._user_history: dict[str, list[dict]] = {}
        self._fitted = False

    def add_models(
        self,
        als: ALSRecommender,
        content: ContentBasedRecommender,
        graph: SimilarityGraphReranker | None = None,
    ) -> "EnsembleRecommender":
        self.als_model = als
        self.content_model = content
        self.graph_reranker = graph
        return self

    def set_user_history(self, interactions: pd.DataFrame) -> None:
        for user_id, grp in interactions.groupby("user_id"):
            self._user_history[user_id] = grp.to_dict("records")

    def fit(self, interactions: pd.DataFrame) -> "EnsembleRecommender":
        if self.als_model:
            self.als_model.fit(interactions)
        if self.content_model:
            problems = (
                interactions[["problem_id", "tags"]]
                .drop_duplicates("problem_id")
            )
            self.content_model.fit(interactions, problems)
        self.set_user_history(interactions)
        if self.graph_reranker is None:
            self.graph_reranker = SimilarityGraphReranker(alpha=self.graph_alpha)
        self._fitted = True
        return self

    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        if not self.als_model or not self.content_model:
            raise RuntimeError("Ensemble models not configured")

        p_als = self.als_model.predict(user_id, problem_ids)
        p_content = self.content_model.predict(user_id, problem_ids)
        total_w = self.w_als + self.w_content
        scores = (self.w_als * p_als + self.w_content * p_content) / total_w

        if self.graph_reranker:
            history = self._user_history.get(user_id, [])
            struggles = self.graph_reranker.struggle_problems(history)
            boosts = self.graph_reranker.graph_boost(problem_ids, struggles)
            scores = scores + boosts

        return np.clip(scores, 0.0, 1.0)

    def predict_components(
        self, user_id: str, problem_ids: list[str]
    ) -> dict[str, np.ndarray]:
        return {
            "als": self.als_model.predict(user_id, problem_ids) if self.als_model else np.array([]),
            "content": self.content_model.predict(user_id, problem_ids)
            if self.content_model
            else np.array([]),
            "ensemble": self.predict(user_id, problem_ids),
        }
