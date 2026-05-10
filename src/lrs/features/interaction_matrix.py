"""Build sparse user-problem interaction matrix for collaborative filtering.

The composite "rating" is derived from solved status + finish_time percentile.
Outputs a scipy CSR matrix saved to data/features/interaction_matrix/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from loguru import logger

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def build_interaction_matrix(
    interactions_df: pd.DataFrame,
    output_dir: Path | None = None,
    score_column: str = "normalized_score",
    include_unsolved: bool = False
) -> sparse.csr_matrix:
    """Build sparse user × problem interaction matrix.
    
    Args:
        interactions_df: Core interaction data
        output_dir: Optional path to save matrix
        score_column: Column to use for matrix values
        include_unsolved: Whether to include unsolved attempts (as 0)
    
    Returns:
        Sparse CSR matrix of shape (n_users, n_problems)
    """
    logger.info("Building interaction matrix...")
    
    # Get unique IDs and create mappings
    user_ids = interactions_df["user_id"].unique()
    problem_ids = interactions_df["problem_id"].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    problem_to_idx = {pid: idx for idx, pid in enumerate(problem_ids)}
    
    n_users = len(user_ids)
    n_problems = len(problem_ids)
    
    logger.info(f"Matrix dimensions: {n_users} users × {n_problems} problems")
    
    # Build matrix data
    rows = []
    cols = []
    data = []
    
    for _, row in interactions_df.iterrows():
        user_idx = user_to_idx[row["user_id"]]
        problem_idx = problem_to_idx[row["problem_id"]]
        
        # Use specified score column
        score = row[score_column]
        
        # Handle unsolved attempts
        if not row["solved"] and not include_unsolved:
            continue
        
        rows.append(user_idx)
        cols.append(problem_idx)
        data.append(score)
    
    # Create sparse matrix
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_problems),
        dtype=np.float32
    )
    
    density = 100 * matrix.nnz / (n_users * n_problems)
    logger.info(f"Built matrix with {matrix.nnz:,} non-zero entries ({density:.2f}% density)")
    
    if output_dir:
        output_path = output_dir / "interaction_matrix.npz"
        sparse.save_npz(output_path, matrix)
        logger.info(f"Saved interaction matrix to {output_path}")
    
    return matrix


def build_bias_terms(
    interactions_df: pd.DataFrame,
    score_column: str = "normalized_score"
) -> tuple[dict[str, float], dict[str, float], float]:
    """Compute user and item bias terms for matrix factorization.
    
    Args:
        interactions_df: Core interaction data
        score_column: Column to use for bias computation
    
    Returns:
        Tuple of (user_biases, item_biases, global_mean)
    """
    logger.info("Computing bias terms...")
    
    # Global mean
    scores = interactions_df[score_column].dropna()
    global_mean = scores.mean()
    logger.info(f"Global mean score: {global_mean:.4f}")
    
    # User biases (deviation from global mean)
    user_scores = interactions_df.groupby("user_id")[score_column].mean()
    user_biases = {uid: score - global_mean for uid, score in user_scores.items()}
    
    # Item biases (deviation from global mean)
    item_scores = interactions_df.groupby("problem_id")[score_column].mean()
    item_biases = {pid: score - global_mean for pid, score in item_scores.items()}
    
    logger.info(f"Computed biases for {len(user_biases)} users and {len(item_biases)} items")
    
    return user_biases, item_biases, global_mean


def build_adjacency_list(
    interactions_df: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build bipartite graph adjacency lists for graph-based models.
    
    Args:
        interactions_df: Core interaction data
    
    Returns:
        Tuple of (user_to_problems, problem_to_users)
    """
    logger.info("Building adjacency lists...")
    
    user_to_problems = {}
    problem_to_users = {}
    
    for _, row in interactions_df.iterrows():
        user_id = row["user_id"]
        problem_id = row["problem_id"]
        
        if user_id not in user_to_problems:
            user_to_problems[user_id] = []
        user_to_problems[user_id].append(problem_id)
        
        if problem_id not in problem_to_users:
            problem_to_users[problem_id] = []
        problem_to_users[problem_id].append(user_id)
    
    logger.info(f"Built adjacency lists: {len(user_to_problems)} users, {len(problem_to_users)} problems")
    
    return user_to_problems, problem_to_users


def compute_user_problem_interactions(
    interactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute aggregated user-problem interaction statistics.
    
    Args:
        interactions_df: Core interaction data
    
    Returns:
        DataFrame with aggregated interactions
    """
    logger.info("Computing user-problem interactions...")
    
    # Group by user-problem pairs
    interactions = interactions_df.groupby(["user_id", "problem_id"]).agg(
        n_attempts=("contest_id", "count"),
        n_solved=("solved", "sum"),
        avg_score=("normalized_score", "mean"),
        avg_finish_time=("finish_time_min", "mean"),
        avg_penalty=("penalty_count", "mean"),
        max_score=("normalized_score", "max")
    ).reset_index()
    
    interactions["solve_rate"] = interactions["n_solved"] / interactions["n_attempts"]
    
    logger.info(f"Computed interactions for {len(interactions):,} user-problem pairs")
    
    return interactions


def main():
    """Run interaction matrix building pipeline."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    from lrs.data.feature_store import FeatureStore
    
    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    feature_dir = Path(os.getenv("DATA_FEATURES_DIR", "data/features"))
    
    # Load data
    feature_store = FeatureStore(processed_dir)
    interactions_df = feature_store.interactions
    
    # Build interaction matrix
    matrix = build_interaction_matrix(interactions_df, feature_dir)
    
    # Compute bias terms
    user_biases, item_biases, global_mean = build_bias_terms(interactions_df)
    
    # Save bias terms
    bias_data = {
        "user_id": list(user_biases.keys()),
        "bias": list(user_biases.values()),
        "global_mean": global_mean
    }
    pd.DataFrame(bias_data).to_parquet(feature_dir / "user_biases.parquet")
    logger.info("Saved user biases")
    
    item_bias_data = {
        "problem_id": list(item_biases.keys()),
        "bias": list(item_biases.values())
    }
    pd.DataFrame(item_bias_data).to_parquet(feature_dir / "item_biases.parquet")
    logger.info("Saved item biases")
    
    # Build adjacency lists
    user_to_problems, problem_to_users = build_adjacency_list(interactions_df)
    
    # Save adjacency lists
    import json
    with open(feature_dir / "user_adjacency.json", "w") as f:
        json.dump(user_to_problems, f)
    with open(feature_dir / "problem_adjacency.json", "w") as f:
        json.dump(problem_to_users, f)
    logger.info("Saved adjacency lists")


if __name__ == "__main__":
    main()
