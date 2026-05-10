"""LightGCN (Light Graph Convolutional Network) for collaborative filtering.

A simplified GCN that only learns user and item embeddings through
neighbor aggregation, without using complex neural network components.

Reference:
    Wang, X., et al. "LightGCN: Simplifying and Powering Graph Convolution
    Network for Recommendation." SIGIR 2020.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from lrs.models.base import BaseRecommender
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


class LightGCNRecommender(BaseRecommender):
    """LightGCN-based collaborative filtering recommender.
    
    This model uses graph convolutional networks to learn user and item
    embeddings through neighbor aggregation on the user-item interaction graph.
    
    Architecture:
        - User and item nodes form a bipartite graph
        - Embeddings are propagated through K layers of GCN
        - Final predictions use dot product of aggregated embeddings
    
    Attributes:
        model: Trained LightGCN model
        user_embeddings: Learned user embeddings
        item_embeddings: Learned item embeddings
        user_id_map: Mapping from user_id to matrix index
        item_id_map: Mapping from item_id to matrix index
        id_to_user: Reverse mapping from index to user_id
        id_to_item: Reverse mapping from index to item_id
        embedding_dim: Dimension of embeddings
        n_layers: Number of GCN layers
        regularization: L2 regularization parameter
        learning_rate: Learning rate for training
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        n_layers: int = 3,
        regularization: float = 1e-5,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 4096,
        dropout: float = 0.0
    ):
        """Initialize the LightGCN recommender.
        
        Args:
            embedding_dim: Dimension of user/item embeddings
            n_layers: Number of GCN layers
            regularization: L2 regularization strength
            learning_rate: Learning rate for Adam optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            dropout: Dropout rate for regularization
        """
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        
        self.model: Optional[object] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        
        self.user_id_map: dict[str, int] = {}
        self.item_id_map: dict[str, int] = {}
        self.id_to_user: dict[int, str] = {}
        self.id_to_item: dict[int, str] = {}
        
        self._fitted = False
    
    def fit(self, interactions: pd.DataFrame) -> "LightGCNRecommender":
        """Train the LightGCN model on interaction data.
        
        Args:
            interactions: DataFrame with columns [user_id, problem_id, score]
        
        Returns:
            self for method chaining
        """
        # TODO: Implement LightGCN training
        # Steps:
        # 1. Build user_id_map and item_id_map
        # 2. Construct bipartite adjacency matrix
        # 3. Initialize user and item embeddings randomly
        # 4. Build training data (positive samples + negative sampling)
        # 5. Train using PyTorch:
        #    - Forward pass: K layers of neighbor aggregation
        #    - Loss: BPR loss or MSE loss
        #    - Backward pass: Adam optimizer
        # 6. Store learned embeddings
        # 7. Set _fitted = True
        
        logger.info("TODO: Implement LightGCN training")
        return self
    
    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        """Predict scores for user-problem pairs.
        
        Args:
            user_id: User ID to predict for
            problem_ids: List of problem IDs to score
        
        Returns:
            Array of predicted scores
        """
        # TODO: Implement LightGCN prediction
        # Steps:
        # 1. Check if user exists in model (cold start handling)
        # 2. Get user embedding
        # 3. Get item embeddings for requested problems
        # 4. Compute dot products: score = user_emb · item_emb
        # 5. Return predictions
        
        logger.info("TODO: Implement LightGCN prediction")
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
        # Uses base class implementation
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
        # TODO: Implement similarity search
        # Compute cosine similarity between item embeddings
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
        # TODO: Implement similarity search
        # Compute cosine similarity between user embeddings
        logger.info("TODO: Implement similar users search")
        return []
    
    def save(self, output_dir: str | Path) -> None:
        """Save model artifacts to disk.
        
        Args:
            output_dir: Directory to save artifacts
        """
        # TODO: Implement model serialization
        # Save:
        # - User embeddings
        # - Item embeddings
        # - ID mappings
        # - Metadata (embedding_dim, n_layers, etc.)
        logger.info("TODO: Implement model saving")
    
    @classmethod
    def load(cls, model_dir: str | Path) -> "LightGCNRecommender":
        """Load a trained model from disk.
        
        Args:
            model_dir: Directory containing saved model artifacts
        
        Returns:
            Loaded LightGCNRecommender instance
        """
        # TODO: Implement model deserialization
        # Load:
        # - User embeddings
        # - Item embeddings
        # - ID mappings
        # - Metadata
        logger.info("TODO: Implement model loading")
        return cls()


class NCFRecommender(BaseRecommender):
    """Neural Collaborative Filtering (NCF) recommender.
    
    Deep learning approach that captures non-linear user-item interactions
    using a Multi-Layer Perceptron (MLP) combined with Matrix Factorization.
    
    Architecture:
        - User and item embeddings concatenated
        - MLP layers process the concatenated embeddings
        - Output layer produces predicted score
        - Can be combined with MF predictions (GMF-MLP architecture)
    
    Attributes:
        model: Trained NCF model (PyTorch)
        user_embeddings: Learned user embeddings
        item_embeddings: Learned item embeddings
        mlp_layers: MLP architecture
        user_id_map: Mapping from user_id to matrix index
        item_id_map: Mapping from item_id to matrix index
        id_to_user: Reverse mapping from index to user_id
        id_to_item: Reverse mapping from index to item_id
        embedding_dim: Dimension of embeddings
        mlp_dims: List of MLP layer dimensions
        learning_rate: Learning rate for training
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        mlp_dims: Optional[list[int]] = None,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 4096,
        dropout: float = 0.0,
        use_mf: bool = True
    ):
        """Initialize the NCF recommender.
        
        Args:
            embedding_dim: Dimension of user/item embeddings
            mlp_dims: List of MLP layer dimensions (e.g., [128, 64, 32])
            learning_rate: Learning rate for Adam optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            dropout: Dropout rate for regularization
            use_mf: Whether to use GMF (Generalized MF) component
        """
        self.embedding_dim = embedding_dim
        self.mlp_dims = mlp_dims or [128, 64, 32]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_mf = use_mf
        
        self.model: Optional[object] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        
        self.user_id_map: dict[str, int] = {}
        self.item_id_map: dict[str, int] = {}
        self.id_to_user: dict[int, str] = {}
        self.id_to_item: dict[int, str] = {}
        
        self._fitted = False
    
    def fit(self, interactions: pd.DataFrame) -> "NCFRecommender":
        """Train the NCF model on interaction data.
        
        Args:
            interactions: DataFrame with columns [user_id, problem_id, score]
        
        Returns:
            self for method chaining
        """
        # TODO: Implement NCF training
        # Steps:
        # 1. Build user_id_map and item_id_map
        # 2. Initialize user and item embeddings
        # 3. Build MLP layers
        # 4. Build training data (user-item-score triplets)
        # 5. Train using PyTorch:
        #    - Forward pass: embeddings → concat → MLP → sigmoid
        #    - Loss: MSE loss for regression or BCE for binary
        #    - Backward pass: Adam optimizer
        # 6. Store learned embeddings and model
        # 7. Set _fitted = True
        
        logger.info("TODO: Implement NCF training")
        return self
    
    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        """Predict scores for user-problem pairs.
        
        Args:
            user_id: User ID to predict for
            problem_ids: List of problem IDs to score
        
        Returns:
            Array of predicted scores
        """
        # TODO: Implement NCF prediction
        # Steps:
        # 1. Check if user exists in model (cold start handling)
        # 2. Get user embedding
        # 3. Get item embeddings for requested problems
        # 4. Concatenate embeddings and pass through MLP
        # 5. Apply sigmoid activation
        # 6. Return predictions
        
        logger.info("TODO: Implement NCF prediction")
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
        # Uses base class implementation
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
        # TODO: Implement similarity search
        # Compute cosine similarity between item embeddings
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
        # TODO: Implement similarity search
        # Compute cosine similarity between user embeddings
        logger.info("TODO: Implement similar users search")
        return []
    
    def save(self, output_dir: str | Path) -> None:
        """Save model artifacts to disk.
        
        Args:
            output_dir: Directory to save artifacts
        """
        # TODO: Implement model serialization
        # Save:
        # - Model state dict
        # - User embeddings
        # - Item embeddings
        # - ID mappings
        # - Metadata
        logger.info("TODO: Implement model saving")
    
    @classmethod
    def load(cls, model_dir: str | Path) -> "NCFRecommender":
        """Load a trained model from disk.
        
        Args:
            model_dir: Directory containing saved model artifacts
        
        Returns:
            Loaded NCFRecommender instance
        """
        # TODO: Implement model deserialization
        # Load:
        # - Model state dict
        # - User embeddings
        # - Item embeddings
        # - ID mappings
        # - Metadata
        logger.info("TODO: Implement model loading")
        return cls()
