"""Cleaning and joining pipeline for raw contest data.

Reads from data/raw/, outputs to data/processed/.
Immutability rule: never read from processed/ as input to this module.
"""

import hashlib
import json
from pathlib import Path
from typing import Generator

import pandas as pd
from loguru import logger


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
    logger.info(f"Found {len(jsonl_files)} contest files in {raw_dir}")
    
    records = []
    for file_path in jsonl_files:
        logger.debug(f"Loading {file_path.name}")
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line in {file_path.name}: {e}")
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df):,} total contest records")
    return df


def preprocess_contest_data(raw_dir: Path, processed_dir: Path) -> None:
    """Main preprocessing pipeline.
    
    Args:
        raw_dir: Path to data/raw/contests/
        processed_dir: Path to data/processed/
    """
    logger.info("Starting preprocessing pipeline")
    
    # Load raw data
    raw_df = load_raw_contest_data(raw_dir)
    
    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate interactions
    interactions = generate_interactions(raw_df)
    interactions.to_parquet(processed_dir / "interactions.parquet")
    logger.info(f"Saved {len(interactions):,} interaction records")
    
    # Generate user features
    user_features = generate_user_features(raw_df)
    user_features.to_parquet(processed_dir / "user_features.parquet")
    logger.info(f"Saved {len(user_features):,} user feature records")
    
    # Generate problem features
    problem_features = generate_problem_features(raw_df)
    problem_features.to_parquet(processed_dir / "problem_features.parquet")
    logger.info(f"Saved {len(problem_features):,} problem feature records")
    
    logger.info("Preprocessing pipeline completed")


def generate_interactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Generate interaction records from raw contest data.
    
    Args:
        raw_df: Raw contest data DataFrame
    
    Returns:
        DataFrame with interaction records
    """
    # Hash user IDs for privacy
    raw_df = raw_df.copy()
    raw_df["user_id"] = raw_df["user_slug"].apply(hash_user_id)
    
    # Flatten submissions into individual interaction records
    interactions = []
    
    for _, row in raw_df.iterrows():
        user_id = row["user_id"]
        contest_id = row["contest_id"]
        submissions = row.get("submissions", {})
        
        for problem_id, submission_info in submissions.items():
            # Determine if solved (no failures)
            solved = submission_info.get("fail_count", 0) == 0
            
            # Create interaction record
            interactions.append({
                "user_id": user_id,
                "problem_id": problem_id,
                "contest_id": contest_id,
                "solved": solved,
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Remove duplicates (user may have solved same problem multiple times)
    interactions_df = interactions_df.drop_duplicates(subset=["user_id", "problem_id"])
    
    # Keep only solved problems for ALS training
    interactions_df = interactions_df[interactions_df["solved"] == True].copy()
    
    return interactions_df


def generate_user_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-user aggregated features.
    
    Args:
        raw_df: Raw contest data DataFrame
    
    Returns:
        DataFrame with user features
    """
    # Hash user IDs
    raw_df = raw_df.copy()
    raw_df["user_id"] = raw_df["user_slug"].apply(hash_user_id)
    
    # Aggregate user statistics
    user_stats = raw_df.groupby("user_id").agg(
        total_contests=("contest_id", "count"),
        total_problems_solved=("problem_id", "count"),
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        first_contest_date=("contest_id", "min"),
        last_contest_date=("contest_id", "max"),
    ).reset_index()
    
    # Add user ID index for matrix alignment
    user_ids = user_stats["user_id"].unique()
    user_stats["user_id_idx"] = user_stats["user_id"].map({uid: idx for idx, uid in enumerate(user_ids)})
    
    return user_stats


def generate_problem_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-problem aggregated features.
    
    Args:
        raw_df: Raw contest data DataFrame
    
    Returns:
        DataFrame with problem features
    """
    # Flatten submissions to get problem-level statistics
    problem_stats = []
    
    for _, row in raw_df.iterrows():
        contest_id = row["contest_id"]
        submissions = row.get("submissions", {})
        
        for problem_id, submission_info in submissions.items():
            solved = submission_info.get("fail_count", 0) == 0
            problem_stats.append({
                "problem_id": problem_id,
                "contest_id": contest_id,
                "solved": solved,
            })
    
    problem_df = pd.DataFrame(problem_stats)
    
    # Aggregate problem statistics
    problem_agg = problem_df.groupby("problem_id").agg(
        total_attempts=("contest_id", "count"),
        total_solved=("solved", "sum"),
    ).reset_index()
    
    # Calculate solve rate
    problem_agg["solve_rate"] = problem_agg["total_solved"] / problem_agg["total_attempts"]
    
    # Add problem ID index for matrix alignment
    problem_ids = problem_agg["problem_id"].unique()
    problem_agg["problem_id_idx"] = problem_agg["problem_id"].map({pid: idx for idx, pid in enumerate(problem_ids)})
    
    return problem_agg
