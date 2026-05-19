"""Problem-level feature engineering.

Reads from data/processed/problems_clean.parquet and interaction data,
outputs to data/features/problem_features.parquet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def compute_problem_features(
    interactions_df: pd.DataFrame,
    problems_df: pd.DataFrame,
    output_dir: Path | None = None
) -> pd.DataFrame:
    """Compute per-problem aggregated features.
    
    Args:
        interactions_df: Core interaction data
        problems_df: Problem metadata
        output_dir: Optional path to save features
    
    Returns:
        DataFrame with problem features
    """
    logger.info("Computing problem features...")
    
    problem_features_list = []
    
    for _, problem_row in problems_df.iterrows():
        problem_id = problem_row["problem_id"]
        
        # Get all interactions for this problem
        problem_data = interactions_df[interactions_df["problem_id"] == problem_id]
        
        # Basic statistics
        n_attempts = len(problem_data)
        n_solved = problem_data["solved"].sum()
        solve_rate = n_solved / max(n_attempts, 1)
        
        # Score statistics
        scores = problem_data["normalized_score"].dropna()
        avg_score = scores.mean() if len(scores) > 0 else 0.0
        score_std = scores.std() if len(scores) > 1 else 0.0
        
        # Finish time statistics
        finish_times = problem_data["finish_time_min"].dropna()
        avg_finish_time = finish_times.mean() if len(finish_times) > 0 else None
        median_finish_time = finish_times.median() if len(finish_times) > 0 else None
        min_finish_time = finish_times.min() if len(finish_times) > 0 else None
        max_finish_time = finish_times.max() if len(finish_times) > 0 else None
        
        # Penalty statistics
        penalty_counts = problem_data["penalty_count"]
        total_penalties = penalty_counts.sum()
        avg_penalty = penalty_counts.mean()
        penalty_rate = total_penalties / max(n_attempts, 1)
        
        # Difficulty label
        difficulty_label = problem_row.get("difficulty_label", "unknown")
        
        # Tags
        tags = problem_row.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = eval(tags)  # Handle stringified lists
            except:
                tags = []
        
        # Acceptance rate
        acceptance_rate = problem_row.get("acceptance_rate", 0.0)
        
        # Difficulty proxy (average across all solvers)
        difficulty_proxies = problem_data["difficulty_proxy"].dropna()
        avg_difficulty_proxy = difficulty_proxies.mean() if len(difficulty_proxies) > 0 else 1.0
        
        # Difficulty bucket (based on difficulty_proxy)
        if avg_difficulty_proxy < 0.8:
            difficulty_bucket = "easy"
        elif avg_difficulty_proxy < 1.5:
            difficulty_bucket = "medium"
        else:
            difficulty_bucket = "hard"
        
        problem_features = {
            "problem_id": problem_id,
            "difficulty_label": difficulty_label,
            "difficulty_bucket": difficulty_bucket,
            "tags": tags,
            "acceptance_rate": acceptance_rate,
            "n_attempts": n_attempts,
            "n_solved": n_solved,
            "solve_rate": solve_rate,
            "avg_score": avg_score,
            "score_std": score_std,
            "avg_finish_time": avg_finish_time,
            "median_finish_time": median_finish_time,
            "min_finish_time": min_finish_time,
            "max_finish_time": max_finish_time,
            "avg_penalty": avg_penalty,
            "penalty_rate": penalty_rate,
            "avg_difficulty_proxy": avg_difficulty_proxy
        }
        
        problem_features_list.append(problem_features)
    
    df = pd.DataFrame(problem_features_list)
    
    logger.info(f"Computed features for {len(df):,} problems")
    
    if output_dir:
        output_path = output_dir / "problem_features.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved problem features to {output_path}")
    
    return df


def compute_problem_tag_vectors(
    problems_df: pd.DataFrame
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build one-hot tag vectors for all problems.
    
    Args:
        problems_df: Problem metadata
    
    Returns:
        Tuple of (problem_tag_vectors, all_tags)
    """
    # Collect all tags
    all_tags = set()
    for tags in problems_df["tags"]:
        if isinstance(tags, (list, tuple)):
            all_tags.update(tags)
        elif hasattr(tags, "__iter__") and not isinstance(tags, str):
            all_tags.update(list(tags))
    
    tag_list = sorted(all_tags)
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}
    
    # Build vectors
    problem_tag_vectors = {}
    for _, row in problems_df.iterrows():
        problem_id = row["problem_id"]
        tags = row.get("tags", [])
        if hasattr(tags, "__iter__") and not isinstance(tags, str):
            tags = list(tags)
        elif isinstance(tags, str):
            try:
                tags = eval(tags)
            except Exception:
                tags = []
        else:
            tags = []
        
        vector = np.zeros(len(tag_list), dtype=np.float32)
        for tag in tags:
            if tag in tag_to_idx:
                vector[tag_to_idx[tag]] = 1.0
        
        problem_tag_vectors[problem_id] = vector
    
    logger.info(f"Built tag vectors for {len(problem_tag_vectors)} problems with {len(tag_list)} tags")
    
    return problem_tag_vectors, tag_list


def compute_problem_similarity_matrix(
    problem_tag_vectors: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compute tag-based similarity between problems.
    
    Args:
        problem_tag_vectors: Dictionary of problem_id -> tag vector
    
    Returns:
        DataFrame with problem similarities
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Build matrix
    problem_ids = list(problem_tag_vectors.keys())
    vectors = np.array([problem_tag_vectors[pid] for pid in problem_ids])
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    
    # Convert to long format
    similarities = []
    for i, pid1 in enumerate(problem_ids):
        for j, pid2 in enumerate(problem_ids):
            if i != j:
                similarities.append({
                    "problem_id_1": pid1,
                    "problem_id_2": pid2,
                    "similarity": similarity_matrix[i, j]
                })
    
    df = pd.DataFrame(similarities)
    logger.info(f"Computed {len(df):,} problem similarities")
    
    return df


def main():
    """Run problem feature engineering pipeline."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    from lrs.data.feature_store import FeatureStore
    
    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    feature_dir = Path(os.getenv("DATA_FEATURES_DIR", "data/features"))
    
    # Load data
    feature_store = FeatureStore(processed_dir)
    interactions_df = feature_store.interactions
    problems_df = feature_store.problem_features
    
    # Compute problem features
    problem_features_df = compute_problem_features(interactions_df, problems_df, feature_dir)
    
    # Compute tag vectors
    problem_tag_vectors, tag_list = compute_problem_tag_vectors(problems_df)
    
    # Save tag vectors
    tag_vectors_path = feature_dir / "problem_tag_vectors.parquet"
    tag_vectors_data = {
        "problem_id": list(problem_tag_vectors.keys()),
        "tag_vector": [str(v) for v in problem_tag_vectors.values()]
    }
    pd.DataFrame(tag_vectors_data).to_parquet(tag_vectors_path)
    logger.info(f"Saved tag vectors to {tag_vectors_path}")


if __name__ == "__main__":
    main()
