"""Temporal train/test evaluation harness."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from lrs.evaluation.metrics import (
    brier_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from lrs.models.baseline.als import ALSRecommender
from lrs.models.calibration import ScoreCalibrator
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def extract_contest_number(contest_id) -> int:
    """Extract numeric suffix from contest id for ordering."""
    s = str(contest_id)
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else int(contest_id) if str(contest_id).isdigit() else 0


def temporal_split(
    interactions: pd.DataFrame,
    n_test_contests: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by contest number — last N contests for test."""
    interactions = interactions.copy()
    interactions["_contest_num"] = interactions["contest_id"].map(extract_contest_number)
    contest_nums = sorted(interactions["_contest_num"].unique())
    if len(contest_nums) <= n_test_contests:
        split_at = contest_nums[len(contest_nums) // 2]
    else:
        split_at = contest_nums[-n_test_contests]

    train = interactions[interactions["_contest_num"] < split_at].drop(columns=["_contest_num"])
    test = interactions[interactions["_contest_num"] >= split_at].drop(columns=["_contest_num"])
    logger.info(f"Temporal split: train={len(train):,} test={len(test):,}")
    return train, test


def fit_calibrator(
    model: ALSRecommender,
    val_interactions: pd.DataFrame,
    sample_size: int = 50000,
) -> ScoreCalibrator:
    """Fit isotonic calibrator on validation interactions."""
    val = val_interactions
    if len(val) > sample_size:
        val = val.sample(sample_size, random_state=42)

    raw_scores = []
    labels = []
    for _, row in val.iterrows():
        uid, pid = row["user_id"], row["problem_id"]
        raw = model._raw_scores(uid, [pid])[0]
        raw_scores.append(raw)
        labels.append(float(row["solved"]))

    cal = ScoreCalibrator(method="isotonic")
    cal.fit(np.array(raw_scores), np.array(labels))
    return cal


def run_backtest(
    interactions: pd.DataFrame,
    model: ALSRecommender | None = None,
    n_test_contests: int = 10,
    k: int = 10,
) -> dict[str, float]:
    """Run temporal backtest with ALS model."""
    train, test = temporal_split(interactions, n_test_contests)

    if model is None:
        model = ALSRecommender(latent_dim=32, iterations=10)
    model.fit(train)
    cal = fit_calibrator(model, test.head(min(50000, len(test))))
    model.calibrator = cal

    test_users = test["user_id"].unique()[:200]
    precisions, recalls, ndcgs, mrrs, briers = [], [], [], [], []

    all_problems = interactions["problem_id"].unique()

    for user_id in test_users:
        train_user = train[train["user_id"] == user_id]
        test_user = test[test["user_id"] == user_id]
        if test_user.empty:
            continue

        relevant = set(test_user[test_user["solved"]]["problem_id"])
        if not relevant:
            continue

        solved_train = set(train_user[train_user["solved"]]["problem_id"])
        candidates = [p for p in all_problems if p not in solved_train]
        if len(candidates) > 1000:
            candidates = list(
                np.random.default_rng(42).choice(candidates, size=1000, replace=False)
            )

        scores = model.predict(user_id, candidates)
        top_idx = np.argsort(scores)[::-1][:k]
        recommended = [candidates[i] for i in top_idx]

        precisions.append(precision_at_k(relevant, recommended, k))
        recalls.append(recall_at_k(relevant, recommended, k))
        ndcgs.append(ndcg_at_k(relevant, recommended, k))
        mrrs.append(mean_reciprocal_rank(relevant, recommended))

        for pid in relevant:
            idx = candidates.index(pid) if pid in candidates else -1
            if idx >= 0:
                briers.append(brier_score(np.array([scores[idx]]), np.array([1.0])))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "brier": float(np.mean(briers)) if briers else 0.0,
        "n_users_evaluated": len(precisions),
    }
