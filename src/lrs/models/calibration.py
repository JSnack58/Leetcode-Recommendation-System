"""Score calibration utilities for mapping model scores to P(solve)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    """Maps raw model scores to calibrated probabilities."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._model: IsotonicRegression | LogisticRegression | None = None
        self._minmax: tuple[float, float] | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScoreCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels, dtype=np.float64)

        if self.method == "platt":
            self._model = LogisticRegression(max_iter=1000)
            self._model.fit(scores, labels)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(scores.ravel(), labels)
        else:
            lo, hi = float(scores.min()), float(scores.max())
            self._minmax = (lo, hi if hi > lo else lo + 1e-6)

        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if self._model is not None:
            if isinstance(self._model, LogisticRegression):
                return self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
            return self._model.predict(scores)
        if self._minmax is not None:
            lo, hi = self._minmax
            return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)
        return np.clip(scores, 0.0, 1.0)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"method": self.method, "model": self._model, "minmax": self._minmax},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "ScoreCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        cal = cls(method=data["method"])
        cal._model = data["model"]
        cal._minmax = data["minmax"]
        return cal
