#!/usr/bin/env python3
"""Train the ALS collaborative filtering model.

This script loads interaction data from the feature store, trains an ALS model,
and saves the trained model to disk.

Supports both full data loading and chunked loading for memory efficiency.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from lrs.data.loader import load_als_interactions
from lrs.models.baseline.als import ALSRecommender
from lrs.utils.logging import logger as app_logger

logger = app_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the ALS collaborative filtering model"
    )
    parser.add_argument(
        "--feature-store-path",
        type=str,
        default=os.getenv("DATA_PROCESSED_DIR", "data/processed"),
        help="Path to the feature store directory"
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
        "--use-bpr",
        action="store_true",
        default=False,
        help="Use BPR loss for implicit feedback"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for memory-efficient training. If set, data will be loaded in chunks."
    )
    parser.add_argument(
        "--solved-only",
        action="store_true",
        default=True,
        help="Only use solved problems (positive feedback)"
    )
    return parser.parse_args()


def main():
    """Train the ALS model and save it."""
    args = parse_args()
    load_dotenv()
    
    # Configuration
    feature_store_path = Path(args.feature_store_path)
    model_output_dir = Path(args.model_output_dir)
    
    # Model hyperparameters
    latent_dim = args.latent_dim
    regularization = args.regularization
    iterations = args.iterations
    use_bpr = args.use_bpr
    chunk_size = args.chunk_size
    solved_only = args.solved_only
    
    logger.info(f"Feature store path: {feature_store_path}")
    logger.info(f"Model output directory: {model_output_dir}")
    logger.info(f"Model configuration: latent_dim={latent_dim}, regularization={regularization}, iterations={iterations}, use_bpr={use_bpr}")
    
    # Determine training mode
    use_chunked = chunk_size is not None
    
    if use_chunked:
        logger.info(f"Using chunked data loading with chunk_size={chunk_size:,}")
        # Load interactions as a generator for chunked training
        interaction_generator = load_als_interactions(
            feature_store_path,
            chunk_size=chunk_size,
            solved_only=solved_only
        )
        
        # Train with chunked data
        logger.info("Training ALS model with chunked data loading...")
        model = ALSRecommender(
            latent_dim=latent_dim,
            regularization=regularization,
            iterations=iterations,
            use_bpr=use_bpr
        )
        model.fit_chunked(interaction_generator)
    else:
        # Load all interactions at once (original behavior)
        logger.info("Loading all interactions at once...")
        als_data = load_als_interactions(
            feature_store_path,
            chunk_size=None,
            solved_only=solved_only
        )
        
        logger.info(f"Prepared {len(als_data):,} records for ALS training")
        
        # Train ALS model
        logger.info("Training ALS model...")
        model = ALSRecommender(
            latent_dim=latent_dim,
            regularization=regularization,
            iterations=iterations,
            use_bpr=use_bpr
        )
        model.fit(als_data)
    
    # Save model
    model.save(model_output_dir)
    logger.info(f"Model saved to {model_output_dir}")
    
    # Test predictions
    logger.info("\n--- Testing predictions ---")
    
    if use_chunked:
        # For chunked training, we need to reload data for testing
        test_data = load_als_interactions(
            feature_store_path,
            chunk_size=None,
            solved_only=solved_only
        )
    else:
        test_data = als_data
    
    test_users = list(test_data["user_id"].unique())[:5]
    
    for user_id in test_users:
        user_problems = test_data[test_data["user_id"] == user_id]["item_id"].unique()
        if len(user_problems) > 0:
            # Predict for a few problems
            sample_problems = list(user_problems[:5])
            predictions = model.predict(user_id, sample_problems)
            logger.info(f"User {user_id}: predictions = {predictions.round(3)}")
    
    # Test similar problems
    logger.info("\n--- Testing similar problems ---")
    test_problem = test_data["item_id"].unique()[0]
    similar = model.get_similar_problems(test_problem, n=5)
    logger.info(f"Problems similar to {test_problem}: {similar}")
    
    # Test similar users
    logger.info("\n--- Testing similar users ---")
    test_user = test_users[0]
    similar_users = model.get_similar_users(test_user, n=5)
    logger.info(f"Users similar to {test_user}: {similar_users}")
    
    # Test cold start
    logger.info("\n--- Testing cold start ---")
    new_user = "new_user_123"
    sample_problems = list(test_data["item_id"].unique())[:5]
    predictions = model.predict(new_user, sample_problems)
    logger.info(f"New user {new_user} predictions: {predictions.round(3)}")
    
    logger.info("\n=== ALS training completed successfully ===")
    
    return model


if __name__ == "__main__":
    main()
