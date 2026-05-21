"""Data loading utilities for the recommendation system.

This module provides data loading utilities including chunked loading
for memory-efficient training of models like ALS.
"""

from pathlib import Path
from typing import Generator, Optional

import pandas as pd
from loguru import logger


def load_interactions(
    feature_store_path: str | Path,
    chunk_size: Optional[int] = None
) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
    """Load interaction data from the feature store.
    
    Args:
        feature_store_path: Path to the feature store directory
        chunk_size: If provided, returns a generator that yields chunks of data.
                   If None, loads all data at once.
    
    Returns:
        DataFrame with all interactions, or a generator yielding DataFrames
    """
    path = Path(feature_store_path) / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Interactions file not found: {path}")
    
    logger.debug(f"Loading interactions from {path}")
    
    if chunk_size:
        logger.info(f"Loading interactions in chunks of {chunk_size:,} rows")
        return _iter_chunks(path, chunk_size)
    else:
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} interaction records")
        return df


def _iter_chunks(
    path: Path,
    chunk_size: int
) -> Generator[pd.DataFrame, None, None]:
    """Iterate over parquet file in chunks.
    
    Args:
        path: Path to the parquet file
        chunk_size: Number of rows per chunk
    
    Yields:
        DataFrames with chunk_size rows each
    """
    reader = pd.read_parquet(path, engine="pyarrow")
    for i in range(0, len(reader), chunk_size):
        chunk = reader.iloc[i:i + chunk_size]
        yield chunk


def load_als_interactions(
    feature_store_path: str | Path,
    chunk_size: Optional[int] = None,
    solved_only: bool = True
) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
    """Load interactions formatted for ALS training.
    
    This function loads and transforms interaction data into the format
    expected by ALS models: ['user_id', 'item_id', 'rating'].
    
    Args:
        feature_store_path: Path to the feature store directory
        chunk_size: If provided, returns a generator that yields chunks.
                   If None, loads all data at once.
        solved_only: If True, only include solved problems (positive feedback)
    
    Returns:
        DataFrame or generator with columns ['user_id', 'item_id', 'rating']
    """
    path = Path(feature_store_path) / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Interactions file not found: {path}")
    
    if chunk_size:
        logger.info(f"Loading ALS interactions in chunks of {chunk_size:,} rows")
        return _iter_als_chunks(path, chunk_size, solved_only)
    else:
        logger.info("Loading all ALS interactions")
        return _load_als_interactions_all(path, solved_only)


def _load_als_interactions_all(
    path: Path,
    solved_only: bool
) -> pd.DataFrame:
    """Load all ALS interactions at once.
    
    Args:
        path: Path to the parquet file
        solved_only: Whether to filter to solved problems only
    
    Returns:
        DataFrame with ALS-formatted interactions
    """
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} raw interaction records")
    
    if solved_only:
        df = df[df["solved"] == True]
        logger.info(f"Filtered to {len(df):,} solved interactions")
    
    # Transform to ALS format
    als_data = df[["user_id", "problem_id", "normalized_score"]].copy()
    als_data.columns = ["user_id", "item_id", "rating"]
    
    logger.info(f"Prepared {len(als_data):,} records for ALS training")
    return als_data


def _iter_als_chunks(
    path: Path,
    chunk_size: int,
    solved_only: bool
) -> Generator[pd.DataFrame, None, None]:
    """Iterate over ALS-formatted interactions in chunks.
    
    Args:
        path: Path to the parquet file
        chunk_size: Number of rows per chunk
        solved_only: Whether to filter to solved problems only
    
    Yields:
        DataFrames with ALS-formatted interactions
    """
    reader = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Processing {len(reader):,} raw records in chunks of {chunk_size:,}")
    
    for i in range(0, len(reader), chunk_size):
        chunk = reader.iloc[i:i + chunk_size]
        
        if solved_only:
            chunk = chunk[chunk["solved"] == True]
        
        if len(chunk) > 0:
            als_chunk = chunk[["user_id", "problem_id", "normalized_score"]].copy()
            als_chunk.columns = ["user_id", "item_id", "rating"]
            yield als_chunk


def load_problems(
    feature_store_path: str | Path,
    chunk_size: Optional[int] = None
) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
    """Load problem data from the feature store.
    
    Args:
        feature_store_path: Path to the feature store directory
        chunk_size: If provided, returns a generator that yields chunks.
                   If None, loads all data at once.
    
    Returns:
        DataFrame with problem data, or a generator yielding DataFrames
    """
    path = Path(feature_store_path) / "problems_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Problems file not found: {path}")
    
    logger.debug(f"Loading problems from {path}")
    
    if chunk_size:
        logger.info(f"Loading problems in chunks of {chunk_size:,} rows")
        return _iter_chunks(path, chunk_size)
    else:
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} problem records")
        return df
