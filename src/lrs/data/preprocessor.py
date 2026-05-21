"""Cleaning and joining pipeline for raw contest data.

Reads from data/raw/, outputs to data/processed/.
Immutability rule: never read from processed/ as input to this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lrs.config import PROCESSED_DIR, FEATURES_DIR
from lrs.data.loader import load_raw_contests


def preprocess() -> None:
    """Run the full preprocessing pipeline.

    Steps:
      1. Load raw contest JSONL data via loader.
      2. Drop duplicate user-problem rows within the same contest.
      3. Normalize solve_time_seconds to minutes; cap outliers at 90th pctl.
      4. Compute per-problem statistics (difficulty_proxy from solve rate,
         avg solve time, etc.) and save as problems_clean.parquet.
      5. Save the cleaned interactions as interactions.parquet.
    """
    # Ensure output directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    df = load_raw_contests()

    # 2. Drop duplicates (keep first occurrence per user+contest+problem)
    df = df.drop_duplicates(
        subset=["user_slug", "contest_slug", "question_id"], keep="first"
    )

    # 3. Normalise solve time to minutes and cap outliers
    df["solve_time_minutes"] = df["solve_time_seconds"] / 60.0

    solved_mask = (df["solved"] == 1) & df["solve_time_minutes"].notna()
    if solved_mask.any():
        p90 = df.loc[solved_mask, "solve_time_minutes"].quantile(0.90)
        df.loc[solved_mask, "solve_time_minutes"] = df.loc[
            solved_mask, "solve_time_minutes"
        ].clip(upper=p90)

    # 4. Compute per-problem statistics -> problems_clean.parquet
    problem_stats = (
        df.groupby("question_id")
        .agg(
            total_attempts=("solved", "count"),
            total_solves=("solved", "sum"),
            avg_fail_count=("fail_count", "mean"),
            avg_solve_time_minutes=(
                "solve_time_minutes",
                lambda s: s.dropna().mean() if s.notna().any() else np.nan,
            ),
            num_contests_appeared=("contest_slug", "nunique"),
        )
        .reset_index()
    )

    problem_stats["solve_rate"] = (
        problem_stats["total_solves"] / problem_stats["total_attempts"]
    )
    problem_stats["difficulty_proxy"] = 1.0 - problem_stats["solve_rate"]

    problem_stats.to_parquet(
        PROCESSED_DIR / "problems_clean.parquet", index=False
    )

    # 5. Save cleaned interactions
    df.to_parquet(PROCESSED_DIR / "interactions.parquet", index=False)

    print(
        f"Preprocessing complete.\n"
        f"  Interactions: {len(df):,} rows  ->  {PROCESSED_DIR / 'interactions.parquet'}\n"
        f"  Problems:     {len(problem_stats):,} rows  ->  {PROCESSED_DIR / 'problems_clean.parquet'}"
    )


if __name__ == "__main__":
    preprocess()
