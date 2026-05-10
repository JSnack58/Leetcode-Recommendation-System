"""Tier assignment logic for recommendation system.

This module implements the three-tier recommendation strategy:
- Edge of Competence: Problems that challenge just above current level
- Blind Spots: Topics where user underperforms compared to peers
- Confidence Builders: High-success-rate problems for reinforcement
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from lrs.config import (
    EDGE_P_LOW,
    EDGE_P_HIGH,
    BLIND_SPOT_THRESHOLD,
    CONFIDENCE_P_MIN,
    TIER_SIZE
)
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TierRecommendation:
    """Container for tiered recommendations."""
    
    edge_of_competence: list[str]
    blind_spots: list[str]
    confidence_builders: list[str]
    
    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dictionary format."""
        return {
            "edge_of_competence": self.edge_of_competence,
            "blind_spots": self.blind_spots,
            "confidence_builders": self.confidence_builders
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"TierRecommendation(\n"
            f"  edge_of_competence: {self.edge_of_competence}\n"
            f"  blind_spots: {self.blind_spots}\n"
            f"  confidence_builders: {self.confidence_builders}\n"
            f")"
        )


class TierAssigner:
    """Assigns problems to recommendation tiers based on predicted scores.
    
    Attributes:
        edge_p_low: Lower bound for Edge of Competence probability
        edge_p_high: Upper bound for Edge of Competence probability
        blind_spot_threshold: Minimum divergence for Blind Spots
        confidence_p_min: Minimum probability for Confidence Builders
        tier_size: Number of problems per tier
    """
    
    def __init__(
        self,
        edge_p_low: float = EDGE_P_LOW,
        edge_p_high: float = EDGE_P_HIGH,
        blind_spot_threshold: float = BLIND_SPOT_THRESHOLD,
        confidence_p_min: float = CONFIDENCE_P_MIN,
        tier_size: int = TIER_SIZE
    ):
        """Initialize the tier assigner.
        
        Args:
            edge_p_low: Lower bound for Edge of Competence
            edge_p_high: Upper bound for Edge of Competence
            blind_spot_threshold: Minimum divergence for Blind Spots
            confidence_p_min: Minimum probability for Confidence Builders
            tier_size: Number of problems per tier
        """
        self.edge_p_low = edge_p_low
        self.edge_p_high = edge_p_high
        self.blind_spot_threshold = blind_spot_threshold
        self.confidence_p_min = confidence_p_min
        self.tier_size = tier_size
        
        logger.info(
            f"Initialized TierAssigner with thresholds:\n"
            f"  Edge of Competence: [{edge_p_low}, {edge_p_high}]\n"
            f"  Blind Spots: > {blind_spot_threshold} divergence\n"
            f"  Confidence Builders: > {confidence_p_min}"
        )
    
    def assign_tiers(
        self,
        user_id: str,
        problem_ids: list[str],
        scores: np.ndarray,
        user_features: Optional[pd.DataFrame] = None,
        peer_features: Optional[pd.DataFrame] = None
    ) -> TierRecommendation:
        """Assign problems to tiers based on predicted scores.
        
        Args:
            user_id: User ID
            problem_ids: List of problem IDs
            scores: Predicted scores for each problem
            user_features: Optional user features for Blind Spot detection
            peer_features: Optional peer features for comparison
        
        Returns:
            TierRecommendation with problems assigned to each tier
        """
        logger.debug(f"Assigning tiers for user {user_id} with {len(problem_ids)} candidates")
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_problem_ids = [problem_ids[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        # Initialize tier lists
        edge_of_competence = []
        blind_spots = []
        confidence_builders = []
        
        # Edge of Competence: [P_low, P_high]
        edge_mask = (sorted_scores >= self.edge_p_low) & (sorted_scores <= self.edge_p_high)
        edge_candidates = [
            pid for pid, score in zip(sorted_problem_ids, sorted_scores)
            if edge_mask[sorted_problem_ids.index(pid)]
        ]
        edge_of_competence = edge_candidates[:self.tier_size]
        
        # Confidence Builders: > P_min
        confidence_mask = sorted_scores >= self.confidence_p_min
        confidence_candidates = [
            pid for pid, score in zip(sorted_problem_ids, sorted_scores)
            if confidence_mask[sorted_problem_ids.index(pid)]
        ]
        confidence_builders = confidence_candidates[:self.tier_size]
        
        # Blind Spots: Topics where user underperforms
        if user_features is not None and peer_features is not None:
            blind_spot_problems = self._detect_blind_spots(
                user_id, problem_ids, scores, user_features, peer_features
            )
            blind_spots = blind_spot_problems[:self.tier_size]
        
        logger.info(
            f"Assigned tiers for user {user_id}:\n"
            f"  Edge of Competence: {len(edge_of_competence)} problems\n"
            f"  Blind Spots: {len(blind_spots)} problems\n"
            f"  Confidence Builders: {len(confidence_builders)} problems"
        )
        
        return TierRecommendation(
            edge_of_competence=edge_of_competence,
            blind_spots=blind_spots,
            confidence_builders=confidence_builders
        )
    
    def _detect_blind_spots(
        self,
        user_id: str,
        problem_ids: list[str],
        scores: np.ndarray,
        user_features: pd.DataFrame,
        peer_features: pd.DataFrame
    ) -> list[str]:
        """Detect problems in user's blind spot topics.
        
        Blind spots are topics where the user's performance is significantly
        below their peer group average.
        
        Args:
            user_id: User ID
            problem_ids: List of problem IDs
            scores: Predicted scores
            user_features: User feature DataFrame
            peer_features: Peer group feature DataFrame
        
        Returns:
            List of problem IDs in blind spots
        """
        # Get user's tag performance
        user_tag_scores = user_features.get("tag_scores", {})
        
        # Get peer group average tag performance
        peer_avg_tag_scores = peer_features.groupby("tag")["score"].mean().to_dict()
        
        # Find tags where user underperforms
        blind_spot_tags = []
        for tag, user_score in user_tag_scores.items():
            peer_score = peer_avg_tag_scores.get(tag, 0.5)
            if user_score < peer_score - self.blind_spot_threshold:
                blind_spot_tags.append(tag)
        
        logger.debug(f"Detected blind spot tags: {blind_spot_tags}")
        
        # Find problems with these tags
        blind_spot_problems = []
        for pid, score in zip(problem_ids, scores):
            # Check if problem has any blind spot tags
            # This would require access to problem features
            # For now, use score-based heuristic
            if score < 0.5:  # Low predicted score
                blind_spot_problems.append(pid)
        
        return blind_spot_problems
    
    def filter_solved_problems(
        self,
        problem_ids: list[str],
        solved_problems: set[str],
        scores: np.ndarray
    ) -> tuple[list[str], np.ndarray]:
        """Filter out already-solved problems.
        
        Args:
            problem_ids: List of problem IDs
            solved_problems: Set of already-solved problem IDs
            scores: Predicted scores
        
        Returns:
            Tuple of (filtered problem IDs, filtered scores)
        """
        mask = [pid not in solved_problems for pid in problem_ids]
        filtered_ids = [pid for pid, m in zip(problem_ids, mask) if m]
        filtered_scores = scores[mask]
        
        logger.debug(f"Filtered {len(problem_ids) - len(filtered_ids)} solved problems")
        
        return filtered_ids, filtered_scores
    
    def enforce_tag_diversity(
        self,
        problem_ids: list[str],
        scores: np.ndarray,
        problem_tags: dict[str, list[str]],
        max_per_tag: int = 2
    ) -> tuple[list[str], np.ndarray]:
        """Enforce tag diversity in recommendations.
        
        Args:
            problem_ids: List of problem IDs
            scores: Predicted scores
            problem_tags: Dictionary of problem_id -> tags
            max_per_tag: Maximum problems per tag
        
        Returns:
            Tuple of (diversified problem IDs, diversified scores)
        """
        tag_counts = {}
        selected_ids = []
        selected_scores = []
        
        for pid, score in zip(problem_ids, scores):
            tags = problem_tags.get(pid, [])
            
            # Check if any tag is at max
            tag_overflow = False
            for tag in tags:
                if tag in tag_counts and tag_counts[tag] >= max_per_tag:
                    tag_overflow = True
                    break
            
            if not tag_overflow:
                selected_ids.append(pid)
                selected_scores.append(score)
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        logger.debug(f"Enforced tag diversity: {len(problem_ids)} -> {len(selected_ids)} problems")
        
        return selected_ids, np.array(selected_scores)


def create_tier_assigner(
    edge_p_low: float = EDGE_P_LOW,
    edge_p_high: float = EDGE_P_HIGH,
    blind_spot_threshold: float = BLIND_SPOT_THRESHOLD,
    confidence_p_min: float = CONFIDENCE_P_MIN,
    tier_size: int = TIER_SIZE
) -> TierAssigner:
    """Factory function to create a TierAssigner.
    
    Args:
        edge_p_low: Lower bound for Edge of Competence
        edge_p_high: Upper bound for Edge of Competence
        blind_spot_threshold: Minimum divergence for Blind Spots
        confidence_p_min: Minimum probability for Confidence Builders
        tier_size: Number of problems per tier
    
    Returns:
        Configured TierAssigner
    """
    return TierAssigner(
        edge_p_low=edge_p_low,
        edge_p_high=edge_p_high,
        blind_spot_threshold=blind_spot_threshold,
        confidence_p_min=confidence_p_min,
        tier_size=tier_size
    )
