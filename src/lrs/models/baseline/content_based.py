"""Content-based filtering using problem tag vectors."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from lrs.features.problem_features import compute_problem_tag_vectors
from lrs.models.base import BaseRecommender
from lrs.models.calibration import ScoreCalibrator
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Tag cosine similarity between user profile and candidate problems."""

    def __init__(self):
        self.problem_vectors: dict[str, np.ndarray] = {}
        self.tag_list: list[str] = []
        self.user_profiles: dict[str, np.ndarray] = {}
        self.calibrator: ScoreCalibrator | None = None
        self._fitted = False

    def fit(
        self,
        interactions: pd.DataFrame,
        problems_df: pd.DataFrame | None = None,
        calibrator: ScoreCalibrator | None = None,
    ) -> "ContentBasedRecommender":
        if problems_df is None:
            problems_df = (
                interactions[["problem_id", "tags"]]
                .drop_duplicates("problem_id")
                .rename(columns={"problem_id": "problem_id"})
            )

        self.problem_vectors, self.tag_list = compute_problem_tag_vectors(problems_df)
        self.calibrator = calibrator

        for user_id, user_data in interactions.groupby("user_id"):
            solved = user_data[user_data["solved"]]
            if solved.empty:
                self.user_profiles[user_id] = np.zeros(len(self.tag_list), dtype=np.float32)
                continue

            vectors = []
            weights = []
            for _, row in solved.iterrows():
                pid = row["problem_id"]
                if pid in self.problem_vectors:
                    vectors.append(self.problem_vectors[pid])
                    weights.append(float(row.get("normalized_score", 1.0)))

            if vectors:
                w = np.array(weights, dtype=np.float32)
                w = w / w.sum()
                profile = np.average(np.stack(vectors), axis=0, weights=w)
            else:
                profile = np.zeros(len(self.tag_list), dtype=np.float32)
            self.user_profiles[user_id] = profile

        self._fitted = True
        logger.info(f"Trained content model for {len(self.user_profiles)} users")
        return self

    def _raw_scores(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        profile = self.user_profiles.get(user_id)
        if profile is None:
            profile = np.zeros(len(self.tag_list), dtype=np.float32)

        scores = np.zeros(len(problem_ids), dtype=np.float32)
        for i, pid in enumerate(problem_ids):
            vec = self.problem_vectors.get(pid)
            if vec is not None and profile.sum() > 0:
                scores[i] = float(
                    cosine_similarity(profile.reshape(1, -1), vec.reshape(1, -1))[0, 0]
                )
        return scores

    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        raw = self._raw_scores(user_id, problem_ids)
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        # Map cosine [-1,1] to [0,1]
        return (raw + 1.0) / 2.0

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "content_model.pkl", "wb") as f:
            pickle.dump(
                {
                    "problem_vectors": self.problem_vectors,
                    "tag_list": self.tag_list,
                    "user_profiles": self.user_profiles,
                    "calibrator": self.calibrator,
                },
                f,
            )

    @classmethod
    def load(cls, model_dir: str | Path) -> "ContentBasedRecommender":
        with open(Path(model_dir) / "content_model.pkl", "rb") as f:
            data = pickle.load(f)
        rec = cls()
        rec.problem_vectors = data["problem_vectors"]
        rec.tag_list = data["tag_list"]
        rec.user_profiles = data["user_profiles"]
        rec.calibrator = data.get("calibrator")
        rec._fitted = True
        return rec
