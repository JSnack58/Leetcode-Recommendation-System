"""ALS (Alternating Least Squares) collaborative filtering baseline.

Uses the `implicit` library for efficient CPU-based ALS on sparse matrices.
Implements the BaseRecommender interface.

Reference:
    Rendle, S., et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback."
    Rendle, S., et al. "Factorization Machines."
"""

from pathlib import Path
from typing import Generator, Optional

import json
import numpy as np
import pandas as pd
from scipy import sparse
from loguru import logger

from lrs.models.base import BaseRecommender


class ALSRecommender(BaseRecommender):
    """ALS-based collaborative filtering recommender with bias terms.
    
    This model implements the classic matrix factorization approach:
        r̂_ui = μ + b_u + b_i + q_u^T p_i
    
    Where:
        - μ = global mean rating
        - b_u = user bias
        - b_i = item bias
        - q_u^T p_i = latent factor interaction
    
    Attributes:
        model: Trained ALS model from implicit library
        user_biases: Dictionary of user bias terms
        item_biases: Dictionary of item bias terms
        global_mean: Global mean rating
        user_id_map: Mapping from user_id to matrix index
        item_id_map: Mapping from item_id to matrix index
        id_to_user: Reverse mapping from index to user_id
        id_to_item: Reverse mapping from index to item_id
        latent_dim: Number of latent factors
        regularization: Regularization parameter (alpha)
        iterations: Number of ALS iterations
        use_bpr: Whether to use BPR loss (implicit feedback)
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        regularization: float = 0.1,
        iterations: int = 15,
        use_bpr: bool = False
    ):
        """Initialize ALS recommender.
        
        Args:
            latent_dim: Number of latent factors
            regularization: Regularization parameter (alpha)
            iterations: Number of ALS iterations
            use_bpr: Whether to use BPR loss (implicit feedback)
        """
        self.latent_dim = latent_dim
        self.regularization = regularization
        self.iterations = iterations
        self.use_bpr = use_bpr
        
        # Model attributes (set after training)
        self.model = None
        self.user_biases = {}
        self.item_biases = {}
        self.global_mean = 0.0
        self.user_id_map = {}
        self.item_id_map = {}
        self.id_to_user = {}
        self.id_to_item = {}
        
        logger.info(f"Initialized ALSRecommender: latent_dim={latent_dim}, "
                   f"regularization={regularization}, iterations={iterations}, use_bpr={use_bpr}")
    
    def fit(self, interactions: pd.DataFrame) -> None:
        """Train the ALS model.
        
        Args:
            interactions: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        logger.info("Training ALS model...")
        
        # Create user and item ID mappings
        users = interactions["user_id"].unique()
        items = interactions["item_id"].unique()
        
        self.user_id_map = {user: idx for idx, user in enumerate(users)}
        self.item_id_map = {item: idx for idx, item in enumerate(items)}
        self.id_to_user = {idx: user for user, idx in self.user_id_map.items()}
        self.id_to_item = {idx: item for item, idx in self.item_id_map.items()}
        
        # Build sparse matrix
        n_users = len(users)
        n_items = len(items)
        
        rows = interactions["user_id"].map(self.user_id_map).values
        cols = interactions["item_id"].map(self.item_id_map).values
        ratings = interactions["rating"].values
        
        # Create sparse matrix
        R = sparse.csr_matrix(
            (ratings, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32
        )
        
        logger.info(f"Built interaction matrix: {n_users} users × {n_items} items, "
                   f"{R.nnz} non-zero entries")
        
        # Compute biases
        self.global_mean = ratings.mean()
        user_ratings = R.toarray().mean(axis=1)
        item_ratings = np.array(R.mean(axis=0)).flatten()
        
        self.user_biases = {
            self.id_to_user[idx]: bias - self.global_mean
            for idx, bias in enumerate(user_ratings)
        }
        self.item_biases = {
            self.id_to_item[idx]: bias - self.global_mean
            for idx, bias in enumerate(item_ratings)
        }
        
        # Train ALS model
        if self.use_bpr:
            from implicit.bpr import BayesianPersonalizedRanking
            self.model = BayesianPersonalizedRanking(
                n_factors=self.latent_dim,
                regularization=self.regularization,
                iterations=self.iterations,
                num_threads=4,
                random_state=42
            )
            self.model.fit(R)
        else:
            from implicit.als import AlternatingLeastSquares
            self.model = AlternatingLeastSquares(
                factors=self.latent_dim,
                regularization=self.regularization,
                iterations=self.iterations,
                use_native=True,
                use_cg=True,
                num_threads=4,
                random_state=42
            )
            self.model.fit(R)
        
        logger.info("ALS training completed")
    
    def fit_chunked(
        self,
        interaction_chunks: pd.DataFrame | Generator[pd.DataFrame, None, None],
        chunk_size: Optional[int] = None
    ) -> None:
        """Train the ALS model using chunked data loading for memory efficiency.
        
        This method supports both a single DataFrame and a generator of DataFrames.
        When using a generator, data is processed in chunks to avoid loading
        all interactions into memory at once.
        
        Args:
            interaction_chunks: Either a DataFrame with ALS-formatted interactions,
                               or a generator yielding DataFrames with columns
                               ['user_id', 'item_id', 'rating']
            chunk_size: If interaction_chunks is a DataFrame, split it into chunks
                       of this size. Ignored if interaction_chunks is a generator.
        """
        logger.info("Training ALS model with chunked data loading...")
        
        # Collect all unique users and items across all chunks
        all_users = set()
        all_items = set()
        
        if isinstance(interaction_chunks, pd.DataFrame):
            # Single DataFrame - split into chunks if needed
            if chunk_size:
                logger.info(f"Splitting DataFrame into chunks of {chunk_size:,} rows")
                for i in range(0, len(interaction_chunks), chunk_size):
                    chunk = interaction_chunks.iloc[i:i + chunk_size]
                    all_users.update(chunk["user_id"].unique())
                    all_items.update(chunk["item_id"].unique())
            else:
                all_users.update(interaction_chunks["user_id"].unique())
                all_items.update(interaction_chunks["item_id"].unique())
        else:
            # Generator - iterate through all chunks
            logger.info("Iterating through generator to collect ID mappings...")
            for chunk in interaction_chunks:
                all_users.update(chunk["user_id"].unique())
                all_items.update(chunk["item_id"].unique())
        
        # Create ID mappings
        self.user_id_map = {user: idx for idx, user in enumerate(sorted(all_users))}
        self.item_id_map = {item: idx for idx, item in enumerate(sorted(all_items))}
        self.id_to_user = {idx: user for user, idx in self.user_id_map.items()}
        self.id_to_item = {idx: item for item, idx in self.item_id_map.items()}
        
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        logger.info(f"Total unique users: {n_users}, Total unique items: {n_items}")
        
        # Build sparse matrix incrementally using COO format (efficient for construction)
        logger.info("Building sparse matrix incrementally...")
        
        row_list = []
        col_list = []
        rating_list = []
        total_rows = 0
        
        if isinstance(interaction_chunks, pd.DataFrame):
            chunks_to_process = []
            if chunk_size:
                for i in range(0, len(interaction_chunks), chunk_size):
                    chunks_to_process.append(interaction_chunks.iloc[i:i + chunk_size])
            else:
                chunks_to_process = [interaction_chunks]
        else:
            chunks_to_process = interaction_chunks
        
        for chunk_idx, chunk in enumerate(chunks_to_process):
            rows = chunk["user_id"].map(self.user_id_map).values
            cols = chunk["item_id"].map(self.item_id_map).values
            ratings = chunk["rating"].values
            
            row_list.extend(rows)
            col_list.extend(cols)
            rating_list.extend(ratings)
            total_rows += len(chunk)
            
            if (chunk_idx + 1) % 10 == 0:
                logger.info(f"Processed {chunk_idx + 1} chunks, {total_rows:,} rows so far")
        
        logger.info(f"Total rows collected: {total_rows:,}")
        
        # Create sparse matrix from collected data
        R = sparse.coo_matrix(
            (rating_list, (row_list, col_list)),
            shape=(n_users, n_items),
            dtype=np.float32
        ).tocsr()
        
        logger.info(f"Built interaction matrix: {n_users} users × {n_items} items, "
                   f"{R.nnz} non-zero entries")
        
        # Compute biases from the final sparse matrix
        self.global_mean = R.data.mean() if R.nnz > 0 else 0.0
        
        # Compute user and item biases
        user_means = np.array(R.mean(axis=1)).flatten()
        item_means = np.array(R.mean(axis=0)).flatten()
        
        # Handle zero-mean rows/cols (users/items with no ratings)
        user_means[user_means == 0] = self.global_mean
        item_means[item_means == 0] = self.global_mean
        
        self.user_biases = {
            self.id_to_user[idx]: bias - self.global_mean
            for idx, bias in enumerate(user_means)
        }
        self.item_biases = {
            self.id_to_item[idx]: bias - self.global_mean
            for idx, bias in enumerate(item_means)
        }
        
        # Train ALS model
        if self.use_bpr:
            from implicit.bpr import BayesianPersonalizedRanking
            self.model = BayesianPersonalizedRanking(
                n_factors=self.latent_dim,
                regularization=self.regularization,
                iterations=self.iterations,
                num_threads=4,
                random_state=42
            )
            self.model.fit(R)
        else:
            from implicit.als import AlternatingLeastSquares
            self.model = AlternatingLeastSquares(
                factors=self.latent_dim,
                regularization=self.regularization,
                iterations=self.iterations,
                use_native=True,
                use_cg=True,
                num_threads=4,
                random_state=42
            )
            self.model.fit(R)
        
        logger.info("ALS training completed with chunked data loading")
    
    def predict(self, user_id: str, item_ids: list[str]) -> list[float]:
        """Predict ratings for user-item pairs.
        
        Args:
            user_id: User ID
            item_ids: List of item IDs
        
        Returns:
            List of predicted ratings
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle cold start users
        if user_id not in self.user_id_map:
            return [
                self.global_mean + self.item_biases.get(item, 0.0)
                for item in item_ids
            ]
        
        user_idx = self.user_id_map[user_id]
        item_indices = [
            self.item_id_map[item] for item in item_ids
            if item in self.item_id_map
        ]
        
        if not item_indices:
            return [
                self.global_mean + self.user_biases.get(user_id, 0.0)
                for _ in item_ids
            ]
        
        # Get predictions from model
        if self.use_bpr:
            # BPR returns scores, not ratings
            scores = self.model.user_item_score(user_idx, item_indices)
            predictions = scores.tolist()
        else:
            # ALS returns ratings
            ratings = self.model.user_item_score(user_idx, item_indices)
            predictions = ratings.tolist()
        
        # Add biases
        user_bias = self.user_biases.get(user_id, 0.0)
        final_predictions = []
        for i, item in enumerate(item_ids):
            if item in self.item_id_map:
                item_bias = self.item_biases.get(item, 0.0)
                final_predictions.append(
                    self.global_mean + user_bias + item_bias + predictions[i]
                )
            else:
                final_predictions.append(
                    self.global_mean + user_bias
                )
        
        return final_predictions
    
    def get_similar_problems(self, problem_id: str, n: int = 10) -> list[tuple[str, float]]:
        """Get similar problems based on user interaction patterns.
        
        Args:
            problem_id: Problem ID to find similar problems for
            n: Number of similar problems to return
        
        Returns:
            List of (problem_id, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if problem_id not in self.item_id_map:
            return []
        
        item_idx = self.item_id_map[problem_id]
        
        # Get similar items
        similar_indices = self.model.similar_items(item_idx, n=n)
        
        results = []
        for idx, score in zip(similar_indices[0], similar_indices[1]):
            problem = self.id_to_item[idx]
            results.append((problem, float(score)))
        
        return results
    
    def get_similar_users(self, user_id: str, n: int = 10) -> list[tuple[str, float]]:
        """Get similar users based on interaction patterns.
        
        Args:
            user_id: User ID to find similar users for
            n: Number of similar users to return
        
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Get similar users
        similar_indices = self.model.similar_users(user_idx, n=n)
        
        results = []
        for idx, score in zip(similar_indices[0], similar_indices[1]):
            user = self.id_to_user[idx]
            results.append((user, float(score)))
        
        return results
    
    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
        seen_problems: Optional[list[str]] = None
    ) -> list[tuple[str, float]]:
        """Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude problems the user has already solved
            seen_problems: List of problem IDs to exclude
        
        Returns:
            List of (problem_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle cold start users
        if user_id not in self.user_id_map:
            # Recommend most popular problems
            return self._recommend_popular(n)
        
        user_idx = self.user_id_map[user_id]
        
        # Get all problem scores
        scores = self.model.user_item_score(user_idx)
        
        # Filter out seen problems
        if exclude_seen and seen_problems:
            seen_indices = [
                idx for idx, item in enumerate(self.id_to_item.values())
                if item in seen_problems
            ]
            scores = np.delete(scores, seen_indices)
        
        # Get top-N
        top_indices = np.argsort(scores)[-n:][::-1]
        
        results = []
        for idx in top_indices:
            problem = self.id_to_item[idx]
            score = scores[idx]
            results.append((problem, float(score)))
        
        return results
    
    def _recommend_popular(self, n: int) -> list[tuple[str, float]]:
        """Recommend most popular problems (for cold start users).
        
        Args:
            n: Number of recommendations
        
        Returns:
            List of (problem_id, popularity_score) tuples
        """
        # Use item biases as popularity proxy
        popularity = [
            (item, bias)
            for item, bias in self.item_biases.items()
        ]
        popularity.sort(key=lambda x: x[1], reverse=True)
        
        return popularity[:n]
    
    def save(self, output_dir: str | Path) -> None:
        """Save the trained model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model metadata
        metadata = {
            "latent_dim": self.latent_dim,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "use_bpr": self.use_bpr,
            "global_mean": float(self.global_mean),
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "user_id_map": self.user_id_map,
            "item_id_map": self.item_id_map,
            "id_to_user": self.id_to_user,
            "id_to_item": self.id_to_item,
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save latent factors
        if self.model is not None:
            user_factors = self.model.user_factors
            item_factors = self.model.item_factors
            
            np.save(output_dir / "user_factors.npy", user_factors)
            np.save(output_dir / "item_factors.npy", item_factors)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load(self, input_dir: str | Path) -> None:
        """Load a trained model from disk.
        
        Args:
            input_dir: Directory containing the saved model
        """
        input_dir = Path(input_dir)
        
        # Load metadata
        metadata_path = input_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.latent_dim = metadata["latent_dim"]
        self.regularization = metadata["regularization"]
        self.iterations = metadata["iterations"]
        self.use_bpr = metadata["use_bpr"]
        self.global_mean = metadata["global_mean"]
        self.user_biases = metadata["user_biases"]
        self.item_biases = metadata["item_biases"]
        self.user_id_map = metadata["user_id_map"]
        self.item_id_map = metadata["item_id_map"]
        self.id_to_user = metadata["id_to_user"]
        self.id_to_item = metadata["id_to_item"]
        
        # Load latent factors
        user_factors = np.load(input_dir / "user_factors.npy")
        item_factors = np.load(input_dir / "item_factors.npy")
        
        # Reconstruct model
        if self.use_bpr:
            from implicit.bpr import BayesianPersonalizedRanking
            self.model = BayesianPersonalizedRanking(
                n_factors=self.latent_dim,
                regularization=self.regularization,
                iterations=self.iterations,
                num_threads=4,
                random_state=42
            )
        else:
            from implicit.als import AlternatingLeastSquares
            self.model = AlternatingLeastSquares(
                factors=self.latent_dim,
                regularization=self.regularization,
                iterations=self.iterations,
                use_native=True,
                use_cg=True,
                num_threads=4,
                random_state=42
            )
        
        self.model.user_factors = user_factors
        self.model.item_factors = item_factors
        
        logger.info(f"Model loaded from {input_dir}")
