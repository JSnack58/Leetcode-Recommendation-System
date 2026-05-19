"""ALS collaborative filtering baseline using implicit library."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from lrs.models.base import BaseRecommender
from lrs.models.calibration import ScoreCalibrator
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def _interaction_weight(row: pd.Series) -> float:
    """Implicit feedback weight from solve quality."""
    if row.get("solved"):
        return 1.0 / (1.0 + float(row.get("penalty_count", 0)))
    return 0.1


class ALSRecommender(BaseRecommender):
    """ALS-based collaborative filtering recommender."""

    def __init__(
        self,
        latent_dim: int = 64,
        regularization: float = 0.1,
        iterations: int = 15,
    ):
        self.latent_dim = latent_dim
        self.regularization = regularization
        self.iterations = iterations

        self.model: AlternatingLeastSquares | None = None
        self.calibrator: ScoreCalibrator | None = None

        self.user_id_map: dict[str, int] = {}
        self.item_id_map: dict[str, int] = {}
        self.id_to_user: dict[int, str] = {}
        self.id_to_item: dict[int, str] = {}
        self._matrix: sparse.csr_matrix | None = None
        self._fitted = False

    def _build_matrix(self, interactions: pd.DataFrame) -> sparse.csr_matrix:
        users = interactions["user_id"].unique()
        items = interactions["problem_id"].unique()
        self.user_id_map = {u: i for i, u in enumerate(users)}
        self.item_id_map = {p: i for i, p in enumerate(items)}
        self.id_to_user = {i: u for u, i in self.user_id_map.items()}
        self.id_to_item = {i: p for p, i in self.item_id_map.items()}

        rows, cols, data = [], [], []
        agg_spec: dict = {
            "solved": ("solved", "max"),
            "penalty_count": ("penalty_count", "sum"),
        }
        if "normalized_score" in interactions.columns:
            agg_spec["normalized_score"] = ("normalized_score", "max")
        grouped = interactions.groupby(["user_id", "problem_id"], as_index=False).agg(
            **{k: v for k, v in agg_spec.items()}
        )
        for _, row in grouped.iterrows():
            uid = self.user_id_map[row["user_id"]]
            pid = self.item_id_map[row["problem_id"]]
            w = _interaction_weight(row)
            rows.append(uid)
            cols.append(pid)
            data.append(w)

        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(items)),
            dtype=np.float32,
        )
        return mat

    def fit(
        self,
        interactions: pd.DataFrame,
        calibrator: ScoreCalibrator | None = None,
    ) -> "ALSRecommender":
        self._matrix = self._build_matrix(interactions)
        # implicit ALS: CSR rows = users, cols = items
        self.model = AlternatingLeastSquares(
            factors=self.latent_dim,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )
        self.model.fit(self._matrix)
        self._fitted = True
        self.calibrator = calibrator
        logger.info(
            f"Trained ALS on {self._matrix.shape[0]} users, {self._matrix.shape[1]} items"
        )
        return self

    def _raw_scores(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not fitted")

        scores = np.zeros(len(problem_ids), dtype=np.float32)
        if user_id not in self.user_id_map:
            return scores + 0.25  # cold-start prior

        uidx = self.user_id_map[user_id]
        user_factors = self.model.user_factors[uidx]

        for i, pid in enumerate(problem_ids):
            if pid in self.item_id_map:
                iidx = self.item_id_map[pid]
                item_factors = self.model.item_factors[iidx]
                scores[i] = float(np.dot(user_factors, item_factors))

        return scores

    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        raw = self._raw_scores(user_id, problem_ids)
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        # Sigmoid fallback
        return 1.0 / (1.0 + np.exp(-raw))

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "als_model.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "user_id_map": self.user_id_map,
                    "item_id_map": self.item_id_map,
                    "calibrator": self.calibrator,
                    "latent_dim": self.latent_dim,
                    "regularization": self.regularization,
                    "iterations": self.iterations,
                },
                f,
            )

    @classmethod
    def load(cls, model_dir: str | Path) -> "ALSRecommender":
        model_dir = Path(model_dir)
        with open(model_dir / "als_model.pkl", "rb") as f:
            data = pickle.load(f)
        rec = cls(
            latent_dim=data["latent_dim"],
            regularization=data["regularization"],
            iterations=data["iterations"],
        )
        rec.model = data["model"]
        rec.user_id_map = data["user_id_map"]
        rec.item_id_map = data["item_id_map"]
        rec.id_to_user = {i: u for u, i in rec.user_id_map.items()}
        rec.id_to_item = {i: p for p, i in rec.item_id_map.items()}
        rec.calibrator = data.get("calibrator")
        rec._fitted = True
        return rec
