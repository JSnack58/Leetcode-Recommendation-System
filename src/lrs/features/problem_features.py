"""Problem-level feature engineering.

Reads from data/processed/problems_clean.parquet and interaction data,
outputs to data/features/problem_features.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lrs.config import FEATURES_DIR


def build_problem_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-problem aggregate features from the interactions table.

    Parameters
    ----------
    interactions : pd.DataFrame
        The cleaned interactions DataFrame (from interactions.parquet).

    Returns
    -------
    pd.DataFrame
        One row per question_id with columns:
        total_attempts, total_solves, solve_rate, avg_fail_count,
        avg_solve_time, num_contests_appeared.
    """
    problem_df = (
        interactions.groupby("question_id")
        .agg(
            total_attempts=("solved", "count"),
            total_solves=("solved", "sum"),
            avg_fail_count=("fail_count", "mean"),
            avg_solve_time=(
                "solve_time_minutes",
                lambda s: s.dropna().mean() if s.notna().any() else np.nan,
            ),
            num_contests_appeared=("contest_slug", "nunique"),
        )
        .reset_index()
    )

    problem_df["solve_rate"] = (
        problem_df["total_solves"] / problem_df["total_attempts"]
    )

    return problem_df


def save_problem_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """Build problem features and persist to parquet. Returns the built DataFrame."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    problem_df = build_problem_features(interactions)
    problem_df.to_parquet(
        FEATURES_DIR / "problem_features.parquet", index=False
    )
    print(
        f"Saved {len(problem_df):,} problem feature rows to "
        f"{FEATURES_DIR / 'problem_features.parquet'}"
    )
    return problem_df
