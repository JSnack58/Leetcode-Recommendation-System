"""Ranking logic for recommendation system.

This module provides ranking functions to order recommendations
within each tier based on various criteria.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from lrs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RankedRecommendation:
    """Container for a ranked recommendation."""
    
    problem_id: str
    score: float
    rank: int
    metadata: Optional[dict] = None


class RecommendationRanker:
    """Ranks recommendations within each tier.
    
    Attributes:
        ranking_strategy: Strategy for ranking (score, recency, diversity)
        diversity_weight: Weight for diversity in ranking
        recency_weight: Weight for recency in ranking
    """
    
    def __init__(
        self,
        ranking_strategy: str = "score",
        diversity_weight: float = 0.0,
        recency_weight: float = 0.0
    ):
        """Initialize the ranker.
        
        Args:
            ranking_strategy: Strategy for ranking (score, recency, diversity)
            diversity_weight: Weight for diversity in ranking
            recency_weight: Weight for recency in ranking
        """
        self.ranking_strategy = ranking_strategy
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        
        logger.info(
            f"Initialized RecommendationRanker:\n"
            f"  ranking_strategy: {ranking_strategy}\n"
            f"  diversity_weight: {diversity_weight}\n"
            f"  recency_weight: {recency_weight}"
        )
    
    def rank(
        self,
        problem_ids: list[str],
        scores: np.ndarray,
        problem_tags: Optional[dict[str, list[str]]] = None,
        recent_attempts: Optional[dict[str, int]] = None
    ) -> list[RankedRecommendation]:
        """Rank problems within a tier.
        
        Args:
            problem_ids: List of problem IDs
            scores: Predicted scores
            problem_tags: Dictionary of problem_id -> tags
            recent_attempts: Dictionary of problem_id -> contests since last attempt
        
        Returns:
            List of RankedRecommendation objects
        """
        # TODO: Implement ranking logic
        # Steps:
        # 1. Compute ranking scores based on strategy
        # 2. Sort by ranking score (descending)
        # 3. Assign ranks
        # 4. Return ranked recommendations
        
        logger.info("TODO: Implement ranking logic")
        
        # Default: rank by score
        ranked = []
        sorted_indices = np.argsort(scores)[::-1]
        
        for rank, idx in enumerate(sorted_indices, start=1):
            ranked.append(RankedRecommendation(
                problem_id=problem_ids[idx],
                score=float(scores[idx]),
                rank=rank,
                metadata=None
            ))
        
        return ranked
    
    def rank_by_score(
        self,
        problem_ids: list[str],
        scores: np.ndarray
    ) -> list[RankedRecommendation]:
        """Rank by predicted score (default strategy).
        
        Args:
            problem_ids: List of problem IDs
            scores: Predicted scores
        
        Returns:
            List of RankedRecommendation objects
        """
        ranked = []
        sorted_indices = np.argsort(scores)[::-1]
        
        for rank, idx in enumerate(sorted_indices, start=1):
            ranked.append(RankedRecommendation(
                problem_id=problem_ids[idx],
                score=float(scores[idx]),
                rank=rank,
                metadata=None
            ))
        
        return ranked
    
    def rank_by_diversity(
        self,
        problem_ids: list[str],
        scores: np.ndarray,
        problem_tags: dict[str, list[str]],
        max_per_tag: int = 2
    ) -> list[RankedRecommendation]:
        """Rank by score with tag diversity consideration.
        
        Args:
            problem_ids: List of problem IDs
            scores: Predicted scores
            problem_tags: Dictionary of problem_id -> tags
            max_per_tag: Maximum problems per tag
        
        Returns:
            List of RankedRecommendation objects
        """
        # TODO: Implement diversity-aware ranking
        # Steps:
        # 1. Sort by score
        # 2. Track tag counts
        # 3. Skip problems that would exceed tag limit
        # 4. Return ranked list
        
        ranked = []
        tag_counts = {}
        sorted_indices = np.argsort(scores)[::-1]
        
        for rank, idx in enumerate(sorted_indices, start=1):
            pid = problem_ids[idx]
            tags = problem_tags.get(pid, [])
            
            # Check if any tag is at max
            tag_overflow = False
            for tag in tags:
                if tag in tag_counts and tag_counts[tag] >= max_per_tag:
                    tag_overflow = True
                    break
            
            if not tag_overflow:
                ranked.append(RankedRecommendation(
                    problem_id=pid,
                    score=float(scores[idx]),
                    rank=rank,
                    metadata=None
                ))
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return ranked
    
    def rank_by_recency(
        self,
        problem_ids: list[str],
        scores: np.ndarray,
        recent_attempts: dict[str, int]
    ) -> list[RankedRecommendation]:
        """Rank by score with recency consideration.
        
        Args:
            problem_ids: List of problem IDs
            scores: Predicted scores
            recent_attempts: Dictionary of problem_id -> contests since last attempt
        
        Returns:
            List of RankedRecommendation objects
        """
        # TODO: Implement recency-aware ranking
        # Steps:
        # 1. Compute recency bonus for each problem
        # 2. Combine with score
        # 3. Sort by combined score
        # 4. Return ranked list
        
        logger.info("TODO: Implement recency-aware ranking")
        
        ranked = []
        sorted_indices = np.argsort(scores)[::-1]
        
        for rank, idx in enumerate(sorted_indices, start=1):
            pid = problem_ids[idx]
            recency = recent_attempts.get(pid, 0)
            
            ranked.append(RankedRecommendation(
                problem_id=pid,
                score=float(scores[idx]),
                rank=rank,
                metadata={"recency": recency}
            ))
        
        return ranked


def create_ranker(
    ranking_strategy: str = "score",
    diversity_weight: float = 0.0,
    recency_weight: float = 0.0
) -> RecommendationRanker:
    """Factory function to create a RecommendationRanker.
    
    Args:
        ranking_strategy: Strategy for ranking
        diversity_weight: Weight for diversity
        recency_weight: Weight for recency
    
    Returns:
        Configured RecommendationRanker
    """
    return RecommendationRanker(
        ranking_strategy=ranking_strategy,
        diversity_weight=diversity_weight,
        recency_weight=recency_weight
    )
