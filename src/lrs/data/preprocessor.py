"""Cleaning and joining pipeline for raw contest data.

Reads from data/raw/, outputs to data/processed/.
Immutability rule: never read from processed/ as input to this module.
"""

import hashlib
from pathlib import Path
from typing import Generator

import pandas as pd
from loguru import logger

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def hash_user_id(user_slug: str, salt: str = "lrs_v1") -> str:
    """Hash user slug to preserve privacy while maintaining consistency.
    
    Args:
        user_slug: Original LeetCode user slug
        salt: Salt for hashing (change to invalidate all hashes)
    
    Returns:
        SHA256 hash of user slug
    """
    return hashlib.sha256(f"{salt}:{user_slug}".encode()).hexdigest()[:16]


def load_raw_contest_data(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all raw contest JSONL files.
    
    Args:
        raw_dir: Path to data/raw/contests/ directory
    
    Returns:
        DataFrame with all contest records
    """
    jsonl_files = list(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {raw_dir}")
    
    logger.info(f"Loading {len(jsonl_files)} contest files from {raw_dir}")
    
    records = []
    for file_path in jsonl_files:
        logger.debug(f"Processing {file_path.name}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    record = pd.read_json(line, typ="series")
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num} in {file_path.name}: {e}")
    
    if not records:
        raise ValueError("No valid records found in contest files")
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df):,} total records from {len(jsonl_files)} files")
    
    return df


def load_raw_problem_data(raw_dir: Path) -> pd.DataFrame:
    """Load problem metadata from raw data.
    
    Args:
        raw_dir: Path to data/raw/ directory
    
    Returns:
        DataFrame with problem metadata
    """
    problems_file = raw_dir / "problems.jsonl"
    if not problems_file.exists():
        logger.warning(f"Problem metadata file not found at {problems_file}")
        return pd.DataFrame()
    
    records = []
    with open(problems_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = pd.read_json(line, typ="series")
                records.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse problem record: {e}")
    
    if records:
        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df):,} problem records")
    else:
        df = pd.DataFrame()
    
    return df


def normalize_finish_time(finish_time_ms: float | None) -> float | None:
    """Normalize finish time from milliseconds to minutes.
    
    Args:
        finish_time_ms: Finish time in milliseconds (or None if unsolved)
    
    Returns:
        Finish time in minutes (or None)
    """
    if finish_time_ms is None:
        return None
    return round(finish_time_ms / 60000, 2)


def cap_outliers(series: pd.Series, percentile: float = 90) -> pd.Series:
    """Cap values at specified percentile to handle outliers.
    
    Args:
        series: Input series
        percentile: Percentile to cap at
    
    Returns:
        Series with outliers capped
    """
    threshold = series.quantile(percentile / 100)
    return series.clip(upper=threshold)


def compute_difficulty_proxy(
    finish_time_min: float | None,
    penalty_count: int,
    global_median_time: float,
    global_median_penalty: int
) -> float:
    """Compute difficulty proxy from solver behavior.
    
    Higher values indicate harder problems for the user.
    
    Args:
        finish_time_min: User's finish time in minutes
        penalty_count: Number of wrong submissions
        global_median_time: Median finish time across all users
        global_median_penalty: Median penalty count across all users
    
    Returns:
        Difficulty proxy score (higher = harder)
    """
    if finish_time_min is None:
        # Unsolved problems are considered very hard
        return 2.0
    
    # Normalize finish time relative to global median
    time_ratio = finish_time_min / max(global_median_time, 1)
    
    # Normalize penalty count relative to global median
    penalty_ratio = penalty_count / max(global_median_penalty, 1)
    
    # Combine into difficulty proxy
    difficulty = (time_ratio + penalty_ratio) / 2
    
    # Cap at reasonable range
    return min(max(difficulty, 0.1), 3.0)


def preprocess_contest_data(
    raw_dir: Path,
    output_dir: Path,
    user_salt: str = "lrs_v1"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main preprocessing pipeline for contest data.
    
    Args:
        raw_dir: Path to data/raw/contests/ directory
        output_dir: Path to data/processed/ directory
        user_salt: Salt for user ID hashing
    
    Returns:
        Tuple of (interactions_df, problems_df)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    df = load_raw_contest_data(raw_dir)
    
    # Load problem metadata if available
    problems_df = pd.DataFrame()
    problems_file = raw_dir.parent / "problems.jsonl"
    if problems_file.exists():
        problems_df = load_raw_problem_data(raw_dir.parent)
    
    logger.info("Starting preprocessing...")
    
    # Drop duplicates (same user + problem in same contest)
    duplicate_cols = ["user_slug", "problem_id", "contest_id"]
    duplicates = df.duplicated(subset=duplicate_cols, keep=False)
    if duplicates.any():
        logger.warning(f"Dropping {duplicates.sum()} duplicate records")
        df = df[~duplicates]
    
    # Hash user IDs for privacy
    df["user_id"] = df["user_slug"].apply(lambda x: hash_user_id(x, user_salt))
    df = df.drop(columns=["user_slug"])
    
    # Normalize finish time to minutes
    df["finish_time_min"] = df["finish_time_ms"].apply(normalize_finish_time)
    df = df.drop(columns=["finish_time_ms"])
    
    # Cap finish time outliers at 90th percentile
    finish_times = df["finish_time_min"].dropna()
    if len(finish_times) > 0:
        threshold = finish_times.quantile(0.90)
        df.loc[df["finish_time_min"] > threshold, "finish_time_min"] = threshold
        logger.info(f"Capped finish time outliers at {threshold:.2f} minutes")
    
    # Compute difficulty proxy
    global_median_time = df["finish_time_min"].dropna().median()
    global_median_penalty = df["penalty_count"].median()
    df["difficulty_proxy"] = df.apply(
        lambda row: compute_difficulty_proxy(
            row["finish_time_min"],
            row["penalty_count"],
            global_median_time,
            global_median_penalty
        ),
        axis=1
    )
    
    # Normalize score to [0, 1] range (assuming max score is 23)
    df["normalized_score"] = df["score"] / 23.0
    df["normalized_score"] = df["normalized_score"].clip(upper=1.0)
    
    # Compute solved status
    df["solved"] = df["score"] > 0
    
    # Drop rows with missing required fields
    required_cols = ["user_id", "contest_id", "problem_id", "solved"]
    df = df.dropna(subset=required_cols)
    
    # Join problem metadata if available
    if not problems_df.empty:
        df = df.merge(
            problems_df[["problem_id", "difficulty_label", "tags", "acceptance_rate"]],
            on="problem_id",
            how="left"
        )
    
    # Select final columns
    interactions_cols = [
        "user_id", "contest_id", "problem_id", "solved",
        "finish_time_min", "penalty_count", "language",
        "user_rating", "difficulty_proxy", "normalized_score",
        "difficulty_label", "tags", "acceptance_rate"
    ]
    df = df[[col for col in interactions_cols if col in df.columns]]
    
    # Sort by user and contest for easier processing
    df = df.sort_values(["user_id", "contest_id", "problem_id"])
    
    logger.info(f"Preprocessed {len(df):,} interaction records")
    
    # Save to parquet
    interactions_path = output_dir / "interactions.parquet"
    df.to_parquet(interactions_path, index=False)
    logger.info(f"Saved interactions to {interactions_path}")
    
    # Save cleaned problems separately
    if not problems_df.empty:
        problems_cols = ["problem_id", "difficulty_label", "tags", "acceptance_rate"]
        problems_clean = problems_df[[col for col in problems_cols if col in problems_df.columns]]
        problems_path = output_dir / "problems_clean.parquet"
        problems_clean.to_parquet(problems_path, index=False)
        logger.info(f"Saved problems to {problems_path}")
    
    return df, problems_clean if not problems_df.empty else pd.DataFrame()


def main():
    """Run preprocessing pipeline."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw/contests"))
    output_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    
    preprocess_contest_data(raw_dir, output_dir)


if __name__ == "__main__":
    main()
