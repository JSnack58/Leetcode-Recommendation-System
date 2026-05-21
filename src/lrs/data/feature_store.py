"""Unified Feature Store for the recommendation system.

This module provides a centralized feature store that all models (ALS, LightGCN,
Content-Based) consume from. It ensures data consistency and efficiency by
preprocessing data once and serving it to all consumers.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from loguru import logger


class FeatureStore:
    """Unified feature store for all recommendation models.
    
    This class provides a centralized interface for loading and managing
    features used by ALS, LightGCN, and Content-Based models.
    
    Attributes:
        store_path: Path to the feature store directory
        interactions: Core interaction data (shared by all models)
        user_features: Per-user aggregated features
        problem_features: Per-problem features
    """
    
    def __init__(self, store_path: str | Path):
        """Initialize the feature store.
        
        Args:
            store_path: Path to the feature store directory
        """
        self.store_path = Path(store_path)
        self._interactions: Optional[pd.DataFrame] = None
        self._user_features: Optional[pd.DataFrame] = None
        self._problem_features: Optional[pd.DataFrame] = None
        self._user_id_map: Optional[dict[str, int]] = None
        self._problem_id_map: Optional[dict[str, int]] = None
        self._id_to_user: Optional[dict[int, str]] = None
        self._id_to_problem: Optional[dict[int, str]] = None
    
    @property
    def interactions(self) -> pd.DataFrame:
        """Load and return core interaction data."""
        if self._interactions is None:
            self._interactions = self._load_interactions()
        return self._interactions
    
    @property
    def user_features(self) -> pd.DataFrame:
        """Load and return user features."""
        if self._user_features is None:
            self._user_features = self._load_user_features()
        return self._user_features
    
    @property
    def problem_features(self) -> pd.DataFrame:
        """Load and return problem features."""
        if self._problem_features is None:
            self._problem_features = self._load_problem_features()
        return self._problem_features
    
    def _load_interactions(self) -> pd.DataFrame:
        """Load core interaction data from parquet."""
        path = self.store_path / "interactions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Interactions file not found: {path}")
        
        logger.debug(f"Loading interactions from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} interaction records")
        return df
    
    def _load_user_features(self) -> pd.DataFrame:
        """Load user features from parquet."""
        path = self.store_path / "user_features.parquet"
        if not path.exists():
            logger.warning(f"User features file not found: {path}")
            return pd.DataFrame()
        
        logger.debug(f"Loading user features from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} user feature records")
        return df
    
    def _load_problem_features(self) -> pd.DataFrame:
        """Load problem features from parquet."""
        path = self.store_path / "problem_features.parquet"
        if not path.exists():
            logger.warning(f"Problem features file not found: {path}")
            return pd.DataFrame()
        
        logger.debug(f"Loading problem features from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} problem feature records")
        return df
    
    def get_user_problem_map(self) -> tuple[dict[str, int], dict[int, str]]:
        """Get bidirectional mapping between user IDs and indices.
        
        Returns:
            Tuple of (user_id_to_idx, idx_to_user_id)
        """
        if self._user_id_map is None:
            user_ids = self.interactions["user_id"].unique()
            self._user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
            self._id_to_user = {idx: uid for uid, idx in self._user_id_map.items()}
            logger.info(f"Built user ID map with {len(self._user_id_map)} users")
        
        return self._user_id_map, self._id_to_user
    
    def get_problem_id_map(self) -> tuple[dict[str, int], dict[int, str]]:
        """Get bidirectional mapping between problem IDs and indices.
        
        Returns:
            Tuple of (problem_id_to_idx, idx_to_problem_id)
        """
        if self._problem_id_map is None:
            problem_ids = self.interactions["problem_id"].unique()
            self._problem_id_map = {pid: idx for idx, pid in enumerate(problem_ids)}
            self._id_to_problem = {idx: pid for pid, idx in self._problem_id_map.items()}
            logger.info(f"Built problem ID map with {len(self._problem_id_map)} problems")
        
        return self._problem_id_map, self._id_to_problem
    
    def get_sparse_interaction_matrix(
        self,
        score_column: str = "normalized_score",
        include_unsolved: bool = False
    ) -> sparse.csr_matrix:
        """Build sparse user × problem interaction matrix.
        
        Args:
            score_column: Column to use for matrix values
            include_unsolved: Whether to include unsolved interactions
        
        Returns:
            Sparse CSR matrix of shape (n_users, n_problems)
        """
        interactions = self.interactions.copy()
        
        if not include_unsolved:
            interactions = interactions[interactions["solved"] == True]
        
        user_map, _ = self.get_user_problem_map()
        problem_map, _ = self.get_problem_id_map()
        
        rows = interactions["user_id"].map(user_map).values
        cols = interactions["problem_id"].map(problem_map).values
        values = interactions[score_column].values
        
        n_users = len(user_map)
        n_problems = len(problem_map)
        
        matrix = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(n_users, n_problems),
            dtype=np.float32
        )
        
        logger.info(f"Built interaction matrix: {n_users} × {n_problems}, "
                   f"{matrix.nnz} non-zero entries")
        
        return matrix
    
    def get_user_features_matrix(self) -> tuple[np.ndarray, dict[str, int]]:
        """Get user features matrix for models that need it.
        
        Returns:
            Tuple of (features_matrix, user_id_to_idx)
        """
        user_map, _ = self.get_user_problem_map()
        
        # Ensure user features are loaded
        _ = self.user_features
        
        # Create feature matrix aligned with user IDs
        feature_cols = [
            col for col in self.user_features.columns
            if col not in ["user_id", "user_id_idx"]
        ]
        
        if not feature_cols:
            logger.warning("No feature columns found in user_features")
            return np.array([]), user_map
        
        # Create matrix with users in order
        n_users = len(user_map)
        features = np.zeros((n_users, len(feature_cols)), dtype=np.float32)
        
        for user_id, idx in user_map.items():
            if user_id in self.user_features["user_id"].values:
                user_row = self.user_features[self.user_features["user_id"] == user_id].iloc[0]
                features[idx] = user_row[feature_cols].values.astype(np.float32)
        
        return features, user_map
    
    def get_problem_features_matrix(self) -> tuple[np.ndarray, dict[str, int]]:
        """Get problem features matrix for models that need it.
        
        Returns:
            Tuple of (features_matrix, problem_id_to_idx)
        """
        problem_map, _ = self.get_problem_id_map()
        
        # Ensure problem features are loaded
        _ = self.problem_features
        
        # Create feature matrix aligned with problem IDs
        feature_cols = [
            col for col in self.problem_features.columns
            if col not in ["problem_id", "problem_id_idx"]
        ]
        
        if not feature_cols:
            logger.warning("No feature columns found in problem_features")
            return np.array([]), problem_map
        
        # Create matrix with problems in order
        n_problems = len(problem_map)
        features = np.zeros((n_problems, len(feature_cols)), dtype=np.float32)
        
        for problem_id, idx in problem_map.items():
            if problem_id in self.problem_features["problem_id"].values:
                problem_row = self.problem_features[
                    self.problem_features["problem_id"] == problem_id
                ].iloc[0]
                features[idx] = problem_row[feature_cols].values.astype(np.float32)
        
        return features, problem_map
