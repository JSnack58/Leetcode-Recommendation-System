#!/usr/bin/env python3
"""Preprocess contest data and split into train/test sets.

Reads from data/raw/contests/combined_contest_data.jsonl
Outputs to data/processed/train/ and data/processed/test/
"""

import hashlib
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from loguru import logger


def hash_user_id(user_slug: str, salt: str = "lrs_v1") -> str:
    """Hash user slug to preserve privacy while maintaining consistency."""
    return hashlib.sha256(f"{salt}:{user_slug}".encode()).hexdigest()[:16]


def main():
    """Main preprocessing pipeline."""
    raw_dir = Path("data/raw/contests")
    train_dir = Path("data/processed/train")
    test_dir = Path("data/processed/test")
    
    # Create output directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Load combined data
    combined_file = raw_dir / "combined_contest_data.jsonl"
    logger.info(f"Loading from {combined_file}")
    
    records = []
    with open(combined_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    
    logger.info(f"Loaded {len(records):,} contest records")
    
    # Hash user IDs and flatten submissions
    interactions = []
    user_stats = defaultdict(lambda: {"contests": 0, "problems_solved": 0, "scores": []})
    problem_stats = defaultdict(lambda: {"attempts": 0, "solved": 0})
    
    for record in records:
        user_slug = record.get("user_slug")
        if not user_slug:
            continue
        
        user_id = hash_user_id(user_slug)
        contest_id = record.get("contest_id")
        score = record.get("score", 0)
        submissions = record.get("submissions", {})
        
        # Update user stats
        user_stats[user_id]["contests"] += 1
        user_stats[user_id]["scores"].append(score)
        
        for problem_id, submission_info in submissions.items():
            solved = submission_info.get("fail_count", 0) == 0
            
            # Update problem stats
            problem_stats[problem_id]["attempts"] += 1
            if solved:
                problem_stats[problem_id]["solved"] += 1
            
            # Create interaction record
            interactions.append({
                "user_id": user_id,
                "problem_id": problem_id,
                "contest_id": contest_id,
                "solved": solved,
            })
    
    # Remove duplicates (user may have solved same problem multiple times)
    seen = set()
    unique_interactions = []
    for interaction in interactions:
        key = (interaction["user_id"], interaction["problem_id"])
        if key not in seen:
            seen.add(key)
            unique_interactions.append(interaction)
    
    interactions = unique_interactions
    logger.info(f"Unique interactions: {len(interactions):,}")
    
    # Split by user (80/20)
    user_interactions = defaultdict(list)
    for interaction in interactions:
        user_interactions[interaction["user_id"]].append(interaction)
    
    user_ids = list(user_interactions.keys())
    import random
    random.seed(42)
    random.shuffle(user_ids)
    
    split_idx = int(len(user_ids) * 0.8)
    train_user_ids = set(user_ids[:split_idx])
    test_user_ids = set(user_ids[split_idx:])
    
    train_interactions = [i for i in interactions if i["user_id"] in train_user_ids]
    test_interactions = [i for i in interactions if i["user_id"] in test_user_ids]
    
    logger.info(f"Train interactions: {len(train_interactions):,}")
    logger.info(f"Test interactions: {len(test_interactions):,}")
    
    # Save train interactions
    train_df = pd.DataFrame(train_interactions)
    train_df.to_parquet(train_dir / "interactions.parquet")
    logger.info(f"Saved {len(train_df):,} train interaction records")
    
    # Save test interactions
    test_df = pd.DataFrame(test_interactions)
    test_df.to_parquet(test_dir / "interactions.parquet")
    logger.info(f"Saved {len(test_df):,} test interaction records")
    
    # Save train user features
    train_user_features = []
    for user_id in train_user_ids:
        stats = user_stats[user_id]
        train_user_features.append({
            "user_id": user_id,
            "total_contests": stats["contests"],
            "total_problems_solved": stats["problems_solved"],
            "avg_score": sum(stats["scores"]) / len(stats["scores"]),
            "max_score": max(stats["scores"]),
            "user_id_idx": list(train_user_ids).index(user_id),
        })
    
    train_user_df = pd.DataFrame(train_user_features)
    train_user_df.to_parquet(train_dir / "user_features.parquet")
    logger.info(f"Saved {len(train_user_df):,} train user feature records")
    
    # Save test user features
    test_user_features = []
    for user_id in test_user_ids:
        stats = user_stats[user_id]
        test_user_features.append({
            "user_id": user_id,
            "total_contests": stats["contests"],
            "total_problems_solved": stats["problems_solved"],
            "avg_score": sum(stats["scores"]) / len(stats["scores"]),
            "max_score": max(stats["scores"]),
            "user_id_idx": list(test_user_ids).index(user_id),
        })
    
    test_user_df = pd.DataFrame(test_user_features)
    test_user_df.to_parquet(test_dir / "user_features.parquet")
    logger.info(f"Saved {len(test_user_df):,} test user feature records")
    
    # Save problem features (shared between train and test)
    problem_features = []
    for problem_id, stats in problem_stats.items():
        solve_rate = stats["solved"] / stats["attempts"] if stats["attempts"] > 0 else 0
        problem_features.append({
            "problem_id": problem_id,
            "total_attempts": stats["attempts"],
            "total_solved": stats["solved"],
            "solve_rate": solve_rate,
            "problem_id_idx": list(problem_stats.keys()).index(problem_id),
        })
    
    problem_df = pd.DataFrame(problem_features)
    problem_df.to_parquet(train_dir / "problem_features.parquet")
    problem_df.to_parquet(test_dir / "problem_features.parquet")
    logger.info(f"Saved {len(problem_df):,} problem feature records")
    
    logger.info("Preprocessing pipeline completed")


if __name__ == "__main__":
    import pandas as pd
    main()
