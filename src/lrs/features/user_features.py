"""User-level feature engineering.

Reads from data/processed/interactions.parquet and outputs per-user
aggregated statistics to data/features/user_features.parquet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


def compute_user_features(
    interactions_df: pd.DataFrame,
    output_dir: Path | None = None
) -> pd.DataFrame:
    """Compute per-user aggregated features.
    
    Args:
        interactions_df: Core interaction data
        output_dir: Optional path to save features
    
    Returns:
        DataFrame with user features
    """
    logger.info("Computing user features...")
    
    user_features_list = []
    
    for user_id in interactions_df["user_id"].unique():
        user_data = interactions_df[interactions_df["user_id"] == user_id]
        
        # Basic statistics
        n_contests = user_data["contest_id"].nunique()
        n_problems = user_data["problem_id"].nunique()
        n_solved = user_data["solved"].sum()
        solve_rate = n_solved / max(n_problems, 1)
        
        # Score statistics
        scores = user_data["normalized_score"].dropna()
        avg_score = scores.mean() if len(scores) > 0 else 0.0
        score_std = scores.std() if len(scores) > 1 else 0.0
        
        # Rating
        user_rating = user_data["user_rating"].dropna()
        avg_rating = user_rating.mean() if len(user_rating) > 0 else 0.0
        
        # Finish time statistics
        finish_times = user_data["finish_time_min"].dropna()
        avg_finish_time = finish_times.mean() if len(finish_times) > 0 else None
        median_finish_time = finish_times.median() if len(finish_times) > 0 else None
        
        # Penalty statistics
        penalty_counts = user_data["penalty_count"]
        n_attempts = len(penalty_counts)
        total_penalties = penalty_counts.sum()
        penalty_rate = total_penalties / max(n_attempts, 1)
        
        # Language preference
        lang_counts = user_data["language"].value_counts()
        top_language = lang_counts.index[0] if len(lang_counts) > 0 else "unknown"
        top_language_pct = lang_counts.iloc[0] / max(n_attempts, 1)
        
        # Per-tag performance
        tag_scores = {}
        for tag in user_data["tags"].dropna():
            if isinstance(tag, list):
                for t in tag:
                    if t not in tag_scores:
                        tag_scores[t] = {"total_score": 0, "count": 0}
                    tag_scores[t]["total_score"] += user_data.loc[
                        user_data["tags"].apply(lambda x: t in x if isinstance(x, list) else False),
                        "normalized_score"
                    ].sum()
                    tag_scores[t]["count"] += 1
        
        # Normalize tag scores
        tag_avg_scores = {}
        for tag, stats in tag_scores.items():
            tag_avg_scores[tag] = stats["total_score"] / max(stats["count"], 1)
        
        # Convert to fixed-length vector (sorted tags)
        all_tags = sorted(tag_avg_scores.keys())
        tag_vector = np.array([tag_avg_scores.get(tag, 0.0) for tag in all_tags], dtype=np.float32)
        
        # Per-difficulty performance
        difficulty_levels = ["easy", "medium", "hard"]
        diff_performance = {}
        for diff in difficulty_levels:
            diff_data = user_data[user_data["difficulty_label"].str.lower() == diff]
            if len(diff_data) > 0:
                diff_scores = diff_data["normalized_score"].dropna()
                diff_performance[diff] = diff_scores.mean() if len(diff_scores) > 0 else 0.0
            else:
                diff_performance[diff] = 0.0
        
        # Difficulty vector
        diff_vector = np.array([
            diff_performance.get("easy", 0.0),
            diff_performance.get("medium", 0.0),
            diff_performance.get("hard", 0.0)
        ], dtype=np.float32)
        
        # Region
        region = user_data["data_region"].dropna().mode()
        region = region.iloc[0] if len(region) > 0 else "unknown"
        
        user_features = {
            "user_id": user_id,
            "total_contests": n_contests,
            "total_problems": n_problems,
            "n_solved": n_solved,
            "solve_rate": solve_rate,
            "avg_score": avg_score,
            "score_std": score_std,
            "avg_rating": avg_rating,
            "avg_finish_time": avg_finish_time,
            "median_finish_time": median_finish_time,
            "penalty_rate": penalty_rate,
            "top_language": top_language,
            "top_language_pct": top_language_pct,
            "region": region,
            "tag_vector": tag_vector,
            "difficulty_vector": diff_vector
        }
        
        user_features_list.append(user_features)
    
    df = pd.DataFrame(user_features_list)
    
    logger.info(f"Computed features for {len(df):,} users")
    
    if output_dir:
        output_path = output_dir / "user_features.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved user features to {output_path}")
    
    return df


def compute_tag_performance(
    interactions_df: pd.DataFrame
) -> pd.Series:
    """Compute per-tag solve rates and average scores.
    
    Args:
        interactions_df: Core interaction data
    
    Returns:
        Series with tag statistics
    """
    tag_stats = {}
    
    for tag in interactions_df["tags"].dropna():
        if isinstance(tag, list):
            for t in tag:
                if t not in tag_stats:
                    tag_stats[t] = {"total_score": 0, "count": 0, "solved": 0}
                
                tag_stats[t]["count"] += 1
                tag_stats[t]["total_score"] += interactions_df.loc[
                    interactions_df["tags"].apply(lambda x: t in x if isinstance(x, list) else False),
                    "normalized_score"
                ].sum()
                
                solved_mask = interactions_df["tags"].apply(
                    lambda x: t in x if isinstance(x, list) else False
                ) & (interactions_df["solved"] == True)
                tag_stats[t]["solved"] += solved_mask.sum()
    
    # Compute averages
    tag_results = {}
    for tag, stats in tag_stats.items():
        tag_results[tag] = {
            "avg_score": stats["total_score"] / max(stats["count"], 1),
            "solve_rate": stats["solved"] / max(stats["count"], 1),
            "n_interactions": stats["count"]
        }
    
    return pd.Series(tag_results)


def main():
    """Run user feature engineering pipeline."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    from lrs.data.feature_store import FeatureStore
    
    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    feature_dir = Path(os.getenv("DATA_FEATURES_DIR", "data/features"))
    
    # Load interactions
    feature_store = FeatureStore(processed_dir)
    interactions_df = feature_store.interactions
    
    # Compute user features
    user_features_df = compute_user_features(interactions_df, feature_dir)
    
    # Compute tag performance
    tag_performance = compute_tag_performance(interactions_df)
    tag_performance_path = feature_dir / "tag_performance.parquet"
    tag_performance.to_frame("stats").reset_index().to_parquet(tag_performance_path)
    logger.info(f"Saved tag performance to {tag_performance_path}")


if __name__ == "__main__":
    main()
