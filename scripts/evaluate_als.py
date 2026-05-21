#!/usr/bin/env python3
"""Evaluate the ALS model on test data with RMSE and Ranking Metrics.

This script:
1. Loads train/test data from data/processed/train/ and data/processed/test/
2. Trains the ALS model on the training set
3. Evaluates the model on the test set using:
   - RMSE (Root Mean Square Error)
   - NDCG@K (Normalized Discounted Cumulative Gain)
   - Precision@K
   - Recall@K
   - MRR (Mean Reciprocal Rank)
   - Hit Rate@K
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from lrs.models.baseline.als import ALSRecommender
from lrs.utils.logging import logger as app_logger
from lrs.evaluation.metrics import (
    calculate_rmse,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mean_reciprocal_rank,
    calculate_hit_rate,
    EvaluationMetrics,
)

logger = app_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ALS collaborative filtering model"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.getenv("DATA_PROCESSED_DIR", "data/processed/train"),
        help="Path to the training data directory"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.getenv("DATA_PROCESSED_DIR", "data/processed/test"),
        help="Path to the test data directory"
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=os.getenv("MODEL_OUTPUT_DIR", "models/als"),
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Number of latent factors"
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.1,
        help="Regularization parameter (alpha)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of ALS iterations"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K value for ranking metrics (NDCG@K, Precision@K, etc.)"
    )
    parser.add_argument(
        "--use-bpr",
        action="store_true",
        default=False,
        help="Use BPR loss for implicit feedback"
    )
    parser.add_argument(
        "--solved-only",
        action="store_true",
        default=True,
        help="Only use solved problems (positive feedback)"
    )
    return parser.parse_args()


def load_train_data(train_dir: Path) -> pd.DataFrame:
    """Load training interactions."""
    interactions_path = train_dir / "interactions.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Training interactions not found: {interactions_path}")
    
    logger.info(f"Loading training data from {interactions_path}")
    df = pd.read_parquet(interactions_path)
    
    # Transform to ALS format
    # Check if normalized_score column exists, otherwise create rating from solved status
    if "normalized_score" in df.columns:
        als_data = df[["user_id", "problem_id", "normalized_score"]].copy()
        als_data.columns = ["user_id", "item_id", "rating"]
    else:
        # For implicit feedback (solved only), use binary rating
        als_data = df[["user_id", "problem_id"]].copy()
        als_data["item_id"] = als_data["problem_id"]
        als_data["rating"] = 1.0  # Binary rating for solved problems
    
    logger.info(f"Loaded {len(als_data):,} training interactions")
    return als_data


def load_test_data(test_dir: Path) -> pd.DataFrame:
    """Load test interactions."""
    interactions_path = test_dir / "interactions.parquet"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Test interactions not found: {interactions_path}")
    
    logger.info(f"Loading test data from {interactions_path}")
    df = pd.read_parquet(interactions_path)
    
    # Transform to ALS format
    # Check if normalized_score column exists, otherwise create rating from solved status
    if "normalized_score" in df.columns:
        als_data = df[["user_id", "problem_id", "normalized_score"]].copy()
        als_data.columns = ["user_id", "item_id", "rating"]
    else:
        # For implicit feedback (solved only), use binary rating
        als_data = df[["user_id", "problem_id"]].copy()
        als_data["item_id"] = als_data["problem_id"]
        als_data["rating"] = 1.0  # Binary rating for solved problems
    
    logger.info(f"Loaded {len(als_data):,} test interactions")
    return als_data


def prepare_evaluation_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> Dict[str, any]:
    """Prepare data structures for evaluation.
    
    Returns:
        Dictionary containing:
        - train_interactions: DataFrame for training
        - test_user_interactions: Dict mapping user_id to their test items
        - all_train_items: Set of all items seen during training
        - all_items: Set of all unique items
    """
    # Build user -> items mapping for test data
    test_user_interactions: Dict[str, Set[str]] = {}
    for _, row in test_data.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]
        if user_id not in test_user_interactions:
            test_user_interactions[user_id] = set()
        test_user_interactions[user_id].add(item_id)
    
    # Get all items seen during training
    all_train_items = set(train_data["item_id"].unique())
    
    # Get all unique items
    all_items = set(train_data["item_id"].unique()).union(set(test_data["item_id"].unique()))
    
    return {
        "train_interactions": train_data,
        "test_user_interactions": test_user_interactions,
        "all_train_items": all_train_items,
        "all_items": all_items,
    }


def evaluate_model(
    model: ALSRecommender,
    train_data: pd.DataFrame,
    test_user_interactions: Dict[str, Set[str]],
    all_train_items: Set[str],
    k: int = 10
) -> EvaluationMetrics:
    """Evaluate the model on test data.
    
    Args:
        model: Trained ALS model
        train_data: Training interactions DataFrame
        test_user_interactions: Dict mapping user_id to their test items
        all_train_items: Set of all items seen during training
        k: K value for ranking metrics
    
    Returns:
        EvaluationMetrics object with aggregated results
    """
    metrics = EvaluationMetrics()
    
    # Build user -> seen items mapping from training data
    user_seen_items: Dict[str, Set[str]] = {}
    for _, row in train_data.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]
        if user_id not in user_seen_items:
            user_seen_items[user_id] = set()
        user_seen_items[user_id].add(item_id)
    
    # Evaluate each user in test set
    users_evaluated = 0
    for user_id, test_items in test_user_interactions.items():
        # Get user's seen items from training
        seen_items = user_seen_items.get(user_id, set())
        
        # Get all items except seen items (for recommendation)
        available_items = all_train_items - seen_items
        
        if len(available_items) == 0:
            continue
        
        # Get recommendations
        recommendations = model.recommend(user_id, n=k, exclude_seen=True, seen_problems=list(seen_items))
        recommended_items = [item for item, score in recommendations]
        
        # Ground truth: test items the user has interacted with
        ground_truth = test_items
        
        # Calculate metrics
        # For RMSE, we need to predict ratings for test items
        if len(ground_truth) > 0 and len(recommended_items) > 0:
            # Get ratings for test items
            test_item_list = list(ground_truth)
            predictions = model.predict(user_id, test_item_list)
            
            # Calculate RMSE (using predicted vs actual rating of 1.0 for solved items)
            rmse = calculate_rmse(
                actual=[1.0] * len(test_item_list),
                predicted=predictions
            )
            
            # Calculate ranking metrics
            ndcg = calculate_ndcg_at_k(ground_truth, recommended_items, k)
            precision = calculate_precision_at_k(ground_truth, recommended_items, k)
            recall = calculate_recall_at_k(ground_truth, recommended_items, k)
            mrr = calculate_mean_reciprocal_rank(ground_truth, recommended_items)
            hit_rate = calculate_hit_rate(ground_truth, recommended_items, k)
            
            metrics.add_sample(rmse, ndcg, precision, recall, mrr, hit_rate)
            users_evaluated += 1
    
    logger.info(f"Evaluated {users_evaluated} users")
    return metrics


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    load_dotenv()
    
    # Paths
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    model_output_dir = Path(args.model_output_dir)
    
    logger.info(f"Training directory: {train_dir}")
    logger.info(f"Test directory: {test_dir}")
    logger.info(f"Model output directory: {model_output_dir}")
    
    # Load data
    train_data = load_train_data(train_dir)
    test_data = load_test_data(test_dir)
    
    # Prepare evaluation data
    eval_data = prepare_evaluation_data(train_data, test_data)
    
    # Train model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING ALS MODEL")
    logger.info("=" * 60)
    
    model = ALSRecommender(
        latent_dim=args.latent_dim,
        regularization=args.regularization,
        iterations=args.iterations,
        use_bpr=args.use_bpr
    )
    
    model.fit(eval_data["train_interactions"])
    
    # Save model
    model.save(model_output_dir)
    logger.info(f"Model saved to {model_output_dir}")
    
    # Evaluate model
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 60)
    
    metrics = evaluate_model(
        model=model,
        train_data=eval_data["train_interactions"],
        test_user_interactions=eval_data["test_user_interactions"],
        all_train_items=eval_data["all_train_items"],
        k=args.k
    )
    
    # Print results
    metrics.print_results()
    
    # Print detailed per-user statistics
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED STATISTICS")
    logger.info("=" * 60)
    averages = metrics.compute_averages()
    logger.info(f"Users evaluated: {len(metrics.rmse_values)}")
    logger.info(f"RMSE - Mean: {averages['rmse']:.6f}, Std: {np.std(metrics.rmse_values):.6f}")
    logger.info(f"NDCG@{args.k} - Mean: {averages['ndcg_at_10']:.6f}, Std: {np.std(metrics.ndcg_values):.6f}")
    logger.info(f"Precision@{args.k} - Mean: {averages['precision_at_10']:.6f}, Std: {np.std(metrics.precision_values):.6f}")
    logger.info(f"Recall@{args.k} - Mean: {averages['recall_at_10']:.6f}, Std: {np.std(metrics.recall_values):.6f}")
    logger.info(f"MRR - Mean: {averages['mrr']:.6f}, Std: {np.std(metrics.mrr_values):.6f}")
    logger.info(f"Hit Rate@{args.k} - Mean: {averages['hit_rate_at_10']:.6f}, Std: {np.std(metrics.hit_rate_values):.6f}")
    
    logger.info("\n=== ALS evaluation completed successfully ===")
    
    return model, metrics


if __name__ == "__main__":
    main()
