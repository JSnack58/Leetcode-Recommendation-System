"""Score normalization helpers to avoid collapsed identical P(solve) values."""

from __future__ import annotations

import numpy as np


def minmax_spread(
    scores: np.ndarray,
    low: float = 0.2,
    high: float = 0.9,
) -> np.ndarray:
    """Map scores to [low, high] preserving sort order."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return scores
    if len(scores) == 1:
        return np.array([(low + high) / 2.0])

    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.linspace(low, high, len(scores))

    return low + (high - low) * (scores - lo) / (hi - lo)


def has_score_variance(scores: np.ndarray, min_std: float = 0.015) -> bool:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) <= 1:
        return False
    return float(scores.std()) >= min_std and len(np.unique(np.round(scores, 3))) > 2


def calibrate_or_spread_raw(
    raw: np.ndarray,
    calibrator,
    *,
    fallback_low: float = 0.25,
    fallback_high: float = 0.85,
) -> tuple[np.ndarray, bool]:
    """Apply calibrator; if output collapses, use per-batch min-max on raw scores.

    Returns:
        (scores, used_rank_spread) — used_rank_spread True if fallback was applied.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if calibrator is not None:
        calibrated = calibrator.predict(raw)
        if has_score_variance(calibrated):
            return np.clip(calibrated, 0.0, 1.0), False

    # Cosine-like raw in [-1, 1] vs ALS dot products — pick sensible mapping
    if raw.min() >= -1.1 and raw.max() <= 1.1 and raw.min() < 0:
        mapped = (raw + 1.0) / 2.0
    else:
        mapped = minmax_spread(raw, low=fallback_low, high=fallback_high)

    if has_score_variance(mapped):
        return np.clip(mapped, 0.0, 1.0), True

    return minmax_spread(mapped, low=fallback_low, high=fallback_high), True
