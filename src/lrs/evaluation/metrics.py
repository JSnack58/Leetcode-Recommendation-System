"""Offline evaluation metrics."""

from __future__ import annotations

import numpy as np


def precision_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rec = recommended[:k]
    if not rec:
        return 0.0
    return len(relevant & set(rec)) / k


def recall_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    rec = recommended[:k]
    return len(relevant & set(rec)) / len(relevant)


def ndcg_at_k(relevant: set[str], recommended: list[str], k: int) -> float:
    rec = recommended[:k]
    if not rec or not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, pid in enumerate(rec) if pid in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def mean_reciprocal_rank(relevant: set[str], recommended: list[str]) -> float:
    for i, pid in enumerate(recommended):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))
