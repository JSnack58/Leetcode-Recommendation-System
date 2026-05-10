"""ALS (Alternating Least Squares) collaborative filtering baseline.

Uses the `implicit` library for efficient CPU-based ALS on sparse matrices.
Implements the BaseRecommender interface.

Reference:
    Rendle, S., et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback."
    Rendle, S., et al. "Factorization Machines."
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from loguru import logger

from lrs.models.base import BaseRecommender
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


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
        regularization: Regularization parameter
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        regularization: float = 0.1,
        iterations: int = 15,
        use_bpr: bool = False
    ):
        """Initialize the ALS recommender.
        
        Args:
            latent_dim: Number of latent factors
            regularization: Regularization parameter (λ)
            iterations: Number of ALS iterations
            use_bpr: Whether to use Bayesian Personalized Ranking (implicit feedback)
        """
        self.latent_dim = latent_dim
        self.regularization = regularization
        self.iterations = iterations
        self.use_bpr = use_bpr
        
        self.model: Optional[object] = None
        self.user_biases: dict[str, float] = {}
        self.item_biases: dict[str, float] = {}
        self.global_mean: float = 0.0
        
        self.user_id_map: dict[str, int] = {}
        self.item_id_map: dict[str, int] = {}
        self.id_to_user: dict[int, str] = {}
        self.id_to_item: dict[int, str] = {}
        
        self._fitted = False
    
    def fit(self, interactions: pd.DataFrame) -> "ALSRecommender":
        """Train the ALS model on interaction data.
        
        Args:
            interactions: DataFrame with columns [user_id, problem_id, score]
        
        Returns:
            self for method chaining
        """
        # TODO: Implement ALS training using implicit.als.AlternatingLeastSquares
        # Steps:
        # 1. Build user_id_map and item_id_map
        # 2. Construct sparse interaction matrix
        # 3. Compute bias terms (user biases, item biases, global mean)
        # 4. Train ALS model with implicit library
        # 5. Store latent factors and bias terms
        # 6. Set _fitted = True
        
        logger.info("TODO: Implement ALS training")
        return self
    
    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        """Predict scores for user-problem pairs.
        
        Args:
            user_id: User ID to predict for
            problem_ids: List of problem IDs to score
        
        Returns:
            Array of predicted scores
        """
        # TODO: Implement prediction
        # Steps:
        # 1. Check if user exists in model (cold start handling)
        # 2. Get user latent factors
        # 3. Get item latent factors for requested problems
        # 4. Compute dot product of latent factors
        # 5. Add bias terms: global_mean + user_bias + item_bias + latent_interaction
        # 6. Return predictions
        
        logger.info("TODO: Implement ALS prediction")
        return np.array([])
    
    def recommend(
        self,
        user_id: str,
        candidate_problem_ids: list[str],
        k: int = 10
    ) -> list[str]:
        """Return top-k recommendations for a user.
        
        Args:
            user_id: User ID to recommend for
            candidate_problem_ids: Pool of candidate problem IDs
            k: Number of recommendations to return
        
        Returns:
            List of top-k problem IDs
        """
        # Uses base class implementation: score all candidates and return top-k
        scores = self.predict(user_id, candidate_problem_ids)
        top_indices = np.argsort(scores)[::-1][:k]
        return [candidate_problem_ids[i] for i in top_indices]
    
    def get_similar_problems(self, problem_id: str, n: int = 10) -> list[tuple[str, float]]:
        """Get problems similar to the given problem.
        
        Args:
            problem_id: Problem ID to find similar problems for
            n: Number of similar problems to return
        
        Returns:
            List of (problem_id, similarity) tuples
        """
        # TODO: Implement similarity search using model.similar_items()
        logger.info("TODO: Implement similar problems search")
        return []
    
    def get_similar_users(self, user_id: str, n: int = 10) -> list[tuple[str, float]]:
        """Get users similar to the given user.
        
        Args:
            user_id: User ID to find similar users for
            n: Number of similar users to return
        
        Returns:
            List of (user_id, similarity) tuples
        """
        # TODO: Implement similarity search using model.similar_users()
        logger.info("TODO: Implement similar users search")
        return []
    
    def save(self, output_dir: str | Path) -> None:
        """Save model artifacts to disk.
        
        Args:
            output_dir: Directory to save artifacts
        """
        # TODO: Implement model serialization
        # Save:
        # - Model weights
        # - ID mappings
        # - Bias terms
        # - Metadata (latent_dim, regularization, etc.)
        logger.info("TODO: Implement model saving")
    
    @classmethod
    def load(cls, model_dir: str | Path) -> "ALSRecommender":
        """Load a trained model from disk.
        
        Args:
            model_dir: Directory containing saved model artifacts
        
        Returns:
            Loaded ALSRecommender instance
        """
        # TODO: Implement model deserialization
        # Load:
        # - Model weights
        # - ID mappings
        # - Bias terms
        # - Metadata
        logger.info("TODO: Implement model loading")
        return cls()


class ContentBasedRecommender(BaseRecommender):
    """Content-based filtering using problem tags and features.
    
    This model recommends problems based on tag similarity and
    difficulty matching, useful for cold-start scenarios.
    """
    
    def __init__(self):
        """Initialize the content-based recommender."""
        self.problem_features: dict[str, dict] = {}
        self.user_profile: dict[str, dict] = {}
        self._fitted = False
    
    def fit(
        self,
        interactions: pd.DataFrame,
        problem_features_df: Optional[pd.DataFrame] = None
    ) -> "ContentBasedRecommender":
        """Train the content-based model.
        
        Args:
            interactions: Interaction data
            problem_features_df: DataFrame with problem features
        
        Returns:
            self for method chaining
        """
        # TODO: Implement content-based training
        # Steps:
        # 1. Build problem feature vectors from problem_features_df
        # 2. Build user profiles from interaction history
        # 3. Compute tag preferences and difficulty preferences
        # 4. Store profiles and features
        # 5. Set _fitted = True
        
        logger.info("TODO: Implement content-based training")
        return self
    
    def _compute_tag_similarity(self, tags1: set, tags2: set) -> float:
        """Compute Jaccard similarity between two tag sets.
        
        Args:
            tags1: First tag set
            tags2: Second tag set
        
        Returns:
            Jaccard similarity score
        """
        # TODO: Implement Jaccard similarity
        logger.info("TODO: Implement tag similarity")
        return 0.0
    
    def _compute_difficulty_match(self, user_diff: dict, problem_diff: str) -> float:
        """Compute how well a problem difficulty matches user preference.
        
        Args:
            user_diff: User's difficulty preference dictionary
            problem_diff: Problem difficulty label
        
        Returns:
            Match score
        """
        # TODO: Implement difficulty matching
        logger.info("TODO: Implement difficulty matching")
        return 0.5
    
    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        """Predict scores based on content similarity.
        
        Args:
            user_id: User ID to predict for
            problem_ids: List of problem IDs to score
        
        Returns:
            Array of predicted scores
        """
        # TODO: Implement content-based prediction
        # Steps:
        # 1. Get user profile
        # 2. For each problem, compute:
        #    - Tag similarity (Jaccard)
        #    - Difficulty match
        #    - Acceptance rate factor
        # 3. Combine into final score
        # 4. Return predictions
        
        logger.info("TODO: Implement content-based prediction")
        return np.array([])
    
    def recommend(
        self,
        user_id: str,
        candidate_problem_ids: list[str],
        k: int = 10
    ) -> list[str]:
        """Return top-k recommendations based on content similarity.
        
        Args:
            user_id: User ID to recommend for
            candidate_problem_ids: Pool of candidate problem IDs
            k: Number of recommendations to return
        
        Returns:
            List of top-k problem IDs
        """
        # Uses base class implementation
        scores = self.predict(user_id, candidate_problem_ids)
        top_indices = np.argsort(scores)[::-1][:k]
        return [candidate_problem_ids[i] for i in top_indices]
