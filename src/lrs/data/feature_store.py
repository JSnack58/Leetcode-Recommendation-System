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

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


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
            include_unsolved: Whether to include unsolved attempts
        
        Returns:
            Sparse CSR matrix of shape (n_users, n_problems)
        """
        user_map, _ = self.get_user_problem_map()
        problem_map, _ = self.get_problem_id_map()
        
        df = self.interactions.copy()
        if not include_unsolved:
            df = df[df["solved"] == True]
        
        rows = df["user_id"].map(user_map).astype(int).values
        cols = df["problem_id"].map(problem_map).astype(int).values
        data = df[score_column].values
        
        n_users = len(user_map)
        n_problems = len(problem_map)
        
        logger.debug(f"Building {n_users} × {n_problems} interaction matrix")
        
        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_problems),
            dtype=np.float32
        )
        
        logger.info(f"Built interaction matrix with {matrix.nnz:,} non-zero entries "
                   f"({100*matrix.nnz/(n_users*n_problems):.2f}% density)")
        
        return matrix
    
    def get_user_features_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Get dense matrix of user features.
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if self.user_features.empty:
            logger.warning("No user features available")
            return np.array([]), []
        
        # Get numeric columns only
        numeric_cols = self.user_features.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric user features available")
            return np.array([]), []
        
        feature_matrix = self.user_features[numeric_cols].values
        logger.info(f"Loaded user features matrix with shape {feature_matrix.shape}")
        
        return feature_matrix, numeric_cols
    
    def get_problem_features_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Get dense matrix of problem features.
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if self.problem_features.empty:
            logger.warning("No problem features available")
            return np.array([]), []
        
        # Get numeric columns only
        numeric_cols = self.problem_features.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.warning("No numeric problem features available")
            return np.array([]), []
        
        feature_matrix = self.problem_features[numeric_cols].values
        logger.info(f"Loaded problem features matrix with shape {feature_matrix.shape}")
        
        return feature_matrix, numeric_cols
    
    def get_tag_vectors(
        self,
        tag_column: str = "tags"
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Build tag vectors for content-based filtering.
        
        Args:
            tag_column: Column containing tags
        
        Returns:
            Tuple of (user_tag_vectors, problem_tag_vectors)
        """
        # Build problem tag vectors (one-hot encoding)
        all_tags = set()
        for tags in self.problem_features[tag_column]:
            if isinstance(tags, list):
                all_tags.update(tags)
        
        tag_list = sorted(all_tags)
        tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}
        
        # Problem tag vectors
        problem_tag_vectors = {}
        for _, row in self.problem_features.iterrows():
            pid = row["problem_id"]
            tags = row.get(tag_column, [])
            if isinstance(tags, list):
                vector = np.zeros(len(tag_list), dtype=np.float32)
                for tag in tags:
                    if tag in tag_to_idx:
                        vector[tag_to_idx[tag]] = 1.0
                problem_tag_vectors[pid] = vector
        
        # User tag preference vectors (average of solved problems)
        user_tag_vectors = {}
        for user_id in self.user_features["user_id"].unique():
            user_solved = self.interactions[
                (self.interactions["user_id"] == user_id) & 
                (self.interactions["solved"] == True)
            ]
            
            if user_solved.empty:
                user_tag_vectors[user_id] = np.zeros(len(tag_list), dtype=np.float32)
                continue
            
            tag_counts = np.zeros(len(tag_list), dtype=np.float32)
            for _, row in user_solved.iterrows():
                tags = row.get("tags", [])
                if isinstance(tags, list):
                    for tag in tags:
                        if tag in tag_to_idx:
                            tag_counts[tag_to_idx[tag]] += 1
            
            # Normalize by number of solved problems
            n_solved = len(user_solved)
            user_tag_vectors[user_id] = tag_counts / max(n_solved, 1)
        
        logger.info(f"Built tag vectors for {len(tag_list)} tags")
        logger.info(f"Built tag vectors for {len(user_tag_vectors)} users and "
                   f"{len(problem_tag_vectors)} problems")
        
        return user_tag_vectors, problem_tag_vectors
    
    def get_temporal_split(
        self,
        n_train: int = 250,
        n_val: int = 30,
        n_test: int = 61
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions into temporal train/val/test sets.
        
        Args:
            n_train: Number of contests for training
            n_val: Number of contests for validation
            n_test: Number of contests for testing
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Get unique contests sorted by ID
        contests = sorted(self.interactions["contest_id"].unique())
        n_contests = len(contests)
        
        if n_contests < n_train + n_val + n_test:
            logger.warning(
                f"Only {n_contests} contests available, adjusting split ratios"
            )
            n_train = max(1, int(n_contests * 0.7))
            n_val = max(1, int(n_contests * 0.1))
            n_test = n_contests - n_train - n_val
        
        train_contests = contests[:n_train]
        val_contests = contests[n_train:n_train + n_val]
        test_contests = contests[n_train + n_val:]
        
        train_df = self.interactions[self.interactions["contest_id"].isin(train_contests)]
        val_df = self.interactions[self.interactions["contest_id"].isin(val_contests)]
        test_df = self.interactions[self.interactions["contest_id"].isin(test_contests)]
        
        logger.info(f"Temporal split: {len(train_df):,} train, "
                   f"{len(val_df):,} val, {len(test_df):,} test")
        
        return train_df, val_df, test_df
    
    def get_user_rating_percentile(self, user_id: str) -> float:
        """Get user's rating percentile among all users.
        
        Args:
            user_id: User ID to get percentile for
        
        Returns:
            Percentile rank (0-1)
        """
        ratings = self.user_features["user_rating"].dropna()
        if ratings.empty:
            return 0.5
        
        user_rating = self.user_features[self.user_features["user_id"] == user_id]["user_rating"]
        if user_rating.empty:
            return 0.5
        
        rating = user_rating.iloc[0]
        percentile = (ratings < rating).sum() / len(ratings)
        return percentile
    
    def get_peer_group(
        self,
        user_id: str,
        rating_window: float = 100.0
    ) -> pd.DataFrame:
        """Get peer group of users within rating window.
        
        Args:
            user_id: User ID to find peers for
            rating_window: Rating window in points (±)
        
        Returns:
            DataFrame of peer user features
        """
        user_rating = self.user_features[self.user_features["user_id"] == user_id]["user_rating"]
        if user_rating.empty:
            return pd.DataFrame()
        
        rating = user_rating.iloc[0]
        peers = self.user_features[
            (self.user_features["user_rating"] >= rating - rating_window) &
            (self.user_features["user_rating"] <= rating + rating_window) &
            (self.user_features["user_id"] != user_id)
        ]
        
        return peers


def create_feature_store(store_path: str | Path) -> FeatureStore:
    """Factory function to create a FeatureStore instance.
    
    Args:
        store_path: Path to the feature store directory
    
    Returns:
        Configured FeatureStore instance
    """
    return FeatureStore(store_path)
