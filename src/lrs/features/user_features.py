"""User-level feature engineering.

Reads from data/processed/interactions.parquet and outputs per-user
aggregated statistics to data/features/user_features.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lrs.config import FEATURES_DIR


def build_user_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user aggregate features from the interactions table.

    Parameters
    ----------
    interactions : pd.DataFrame
        The cleaned interactions DataFrame (from interactions.parquet).

    Returns
    -------
    pd.DataFrame
        One row per user_slug with columns:
        total_contests, total_solved, total_attempted, overall_solve_rate,
        avg_rank, avg_fail_count, avg_solve_time, num_languages_used.
    """
    user_df = (
        interactions.groupby("user_slug")
        .agg(
            total_contests=("contest_slug", "nunique"),
            total_solved=("solved", "sum"),
            total_attempted=("solved", "count"),
            avg_rank=("rank", "mean"),
            avg_fail_count=("fail_count", "mean"),
            avg_solve_time=(
                "solve_time_minutes",
                lambda s: s.dropna().mean() if s.notna().any() else np.nan,
            ),
            num_languages_used=(
                "lang",
                lambda s: s[s != ""].nunique(),
            ),
        )
        .reset_index()
    )

    user_df["overall_solve_rate"] = (
        user_df["total_solved"] / user_df["total_attempted"]
    )

    return user_df


def save_user_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """Build user features and persist to parquet. Returns the built DataFrame."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    user_df = build_user_features(interactions)
    user_df.to_parquet(FEATURES_DIR / "user_features.parquet", index=False)
    print(f"Saved {len(user_df):,} user feature rows to {FEATURES_DIR / 'user_features.parquet'}")
    return user_df
