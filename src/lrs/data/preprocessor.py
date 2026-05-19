"""Cleaning and joining pipeline for raw contest data.

Reads from data/raw/, outputs to data/processed/.
Immutability rule: never read from processed/ as input to this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from lrs.data.contest_parser import parse_contest_jsonl, parse_contest_jsonl_file
from lrs.data.problem_catalog import join_interactions_to_catalog, load_problem_catalog
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def hash_user_id(user_slug: str, salt: str = "lrs_v1") -> str:
    """Hash user slug to preserve privacy while maintaining consistency."""
    return hashlib.sha256(f"{salt}:{user_slug}".encode()).hexdigest()[:16]


def compute_difficulty_proxy(
    finish_time_min: float | None,
    penalty_count: int,
    global_median_time: float,
    global_median_penalty: float,
) -> float:
    """Compute difficulty proxy from solver behavior (higher = harder)."""
    if finish_time_min is None or (isinstance(finish_time_min, float) and pd.isna(finish_time_min)):
        return 2.0

    time_ratio = float(finish_time_min) / max(global_median_time, 1.0)
    penalty_ratio = penalty_count / max(global_median_penalty, 1.0)
    difficulty = (time_ratio + penalty_ratio) / 2.0
    return float(min(max(difficulty, 0.1), 3.0))


def preprocess_contest_data(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    user_salt: str = "lrs_v1",
    catalog_gml: Path | None = None,
    sample: int | None = None,
    raw_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main preprocessing pipeline for contest data.

    Args:
        raw_dir: Path to directory of per-contest ``*.jsonl`` files
        output_dir: Path to data/processed/ directory
        user_salt: Salt for user ID hashing
        catalog_gml: Optional path to bipartite GML for problem catalog
        sample: Optional limit on source contest rows (dev mode)
        raw_file: Single combined JSONL (e.g. combined_contest_data.jsonl)

    Returns:
        Tuple of (interactions_df, problems_df)
    """
    if raw_file is None and raw_dir is None:
        raise ValueError("Provide raw_dir or raw_file")

    output_dir = Path(output_dir or "data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw_file is not None:
        df = parse_contest_jsonl_file(Path(raw_file), sample=sample)
    else:
        df = parse_contest_jsonl(Path(raw_dir), sample=sample)
    catalog = load_problem_catalog(catalog_gml)
    df, problems_clean = join_interactions_to_catalog(df, catalog)

    logger.info("Starting preprocessing...")

    duplicate_cols = ["user_slug", "problem_id", "contest_id"]
    duplicates = df.duplicated(subset=duplicate_cols, keep=False)
    if duplicates.any():
        logger.warning(f"Dropping {duplicates.sum()} duplicate records")
        df = df[~duplicates]

    df["user_id"] = df["user_slug"].apply(lambda x: hash_user_id(x, user_salt))
    df["user_rating"] = df["rank_percentile"] * 3000  # scale to pseudo-rating

    finish_times = df["finish_time_min"].dropna()
    if len(finish_times) > 0:
        threshold = finish_times.quantile(0.90)
        df.loc[df["finish_time_min"] > threshold, "finish_time_min"] = threshold
        logger.info(f"Capped finish time outliers at {threshold:.2f} minutes")

    global_median_time = float(df["finish_time_min"].dropna().median() or 30.0)
    global_median_penalty = float(df["penalty_count"].median() or 1.0)
    df["difficulty_proxy"] = df.apply(
        lambda row: compute_difficulty_proxy(
            row["finish_time_min"],
            int(row["penalty_count"]),
            global_median_time,
            global_median_penalty,
        ),
        axis=1,
    )

    required_cols = ["user_id", "contest_id", "problem_id", "solved"]
    df = df.dropna(subset=required_cols)

    interactions_cols = [
        "user_id",
        "contest_id",
        "problem_id",
        "solved",
        "finish_time_min",
        "penalty_count",
        "language",
        "user_rating",
        "rank_percentile",
        "difficulty_proxy",
        "normalized_score",
        "tags",
    ]
    df = df[[c for c in interactions_cols if c in df.columns]]
    df = df.sort_values(["user_id", "contest_id", "problem_id"])

    logger.info(f"Preprocessed {len(df):,} interaction records")

    interactions_path = output_dir / "interactions.parquet"
    df.to_parquet(interactions_path, index=False)
    logger.info(f"Saved interactions to {interactions_path}")

    problems_path = output_dir / "problems_clean.parquet"
    problems_clean.to_parquet(problems_path, index=False)
    logger.info(f"Saved problems to {problems_path}")

    return df, problems_clean


def main() -> None:
    """Run preprocessing pipeline."""
    import os

    from dotenv import load_dotenv

    load_dotenv()

    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw/contests"))
    output_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    sample = os.getenv("LRS_SAMPLE_ROWS")
    sample_n = int(sample) if sample else None

    preprocess_contest_data(raw_dir, output_dir, sample=sample_n)


if __name__ == "__main__":
    main()
