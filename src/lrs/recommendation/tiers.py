"""Tier assignment logic for recommendation system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lrs.config import (
    BLIND_SPOT_THRESHOLD,
    CONFIDENCE_P_MIN,
    EDGE_P_HIGH,
    EDGE_P_LOW,
    TIER_SIZE,
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
        return {
            "edge_of_competence": self.edge_of_competence,
            "blind_spots": self.blind_spots,
            "confidence_builders": self.confidence_builders,
        }


class TierAssigner:
    """Assigns problems to recommendation tiers based on predicted scores."""

    def __init__(
        self,
        edge_p_low: float = EDGE_P_LOW,
        edge_p_high: float = EDGE_P_HIGH,
        blind_spot_threshold: float = BLIND_SPOT_THRESHOLD,
        confidence_p_min: float = CONFIDENCE_P_MIN,
        tier_size: int = TIER_SIZE,
    ):
        self.edge_p_low = edge_p_low
        self.edge_p_high = edge_p_high
        self.blind_spot_threshold = blind_spot_threshold
        self.confidence_p_min = confidence_p_min
        self.tier_size = tier_size

    def assign_tiers(
        self,
        user_id: str,
        problem_ids: list[str],
        scores: np.ndarray,
        user_features: pd.Series | None = None,
        peer_tag_scores: dict[str, float] | None = None,
        problem_tags: dict[str, list[str]] | None = None,
        user_median_finish: float | None = None,
        problem_median_finish: dict[str, float] | None = None,
    ) -> TierRecommendation:
        """Assign problems to tiers based on predicted scores."""
        order = np.argsort(scores)[::-1]
        sorted_ids = [problem_ids[i] for i in order]
        sorted_scores = scores[order]

        edge_of_competence: list[str] = []
        confidence_builders: list[str] = []
        blind_spots: list[str] = []

        for pid, score in zip(sorted_ids, sorted_scores):
            if self.edge_p_low <= score <= self.edge_p_high and len(edge_of_competence) < self.tier_size:
                edge_of_competence.append(pid)

        if not edge_of_competence:
            target = (self.edge_p_low + self.edge_p_high) / 2.0
            dists = np.abs(sorted_scores - target)
            edge_order = np.argsort(dists)
            for idx in edge_order:
                if len(edge_of_competence) >= self.tier_size:
                    break
                pid = sorted_ids[idx]
                if pid not in edge_of_competence:
                    edge_of_competence.append(pid)

        for pid, score in zip(sorted_ids, sorted_scores):
            if score >= self.confidence_p_min and len(confidence_builders) < self.tier_size:
                if user_median_finish is not None and problem_median_finish:
                    p_med = problem_median_finish.get(pid)
                    if p_med is not None and p_med > user_median_finish * 1.2:
                        continue
                confidence_builders.append(pid)

        if user_features is not None and peer_tag_scores and problem_tags:
            blind_spot_tags = self._blind_spot_tags(user_features, peer_tag_scores)
            blind_spots = self._problems_for_tags(
                sorted_ids, sorted_scores, blind_spot_tags, problem_tags
            )[: self.tier_size]

        return TierRecommendation(
            edge_of_competence=edge_of_competence,
            blind_spots=blind_spots,
            confidence_builders=confidence_builders,
        )

    def _blind_spot_tags(
        self,
        user_features: pd.Series,
        peer_tag_scores: dict[str, float],
    ) -> set[str]:
        user_tag_scores = user_features.get("tag_scores") or {}
        if isinstance(user_tag_scores, str):
            import ast

            try:
                user_tag_scores = ast.literal_eval(user_tag_scores)
            except (ValueError, SyntaxError):
                user_tag_scores = {}

        blind: set[str] = set()
        for tag, user_score in user_tag_scores.items():
            peer_score = peer_tag_scores.get(tag, 0.5)
            if user_score < peer_score - self.blind_spot_threshold:
                blind.add(tag)
        return blind

    def _problems_for_tags(
        self,
        sorted_ids: list[str],
        sorted_scores: np.ndarray,
        blind_tags: set[str],
        problem_tags: dict[str, list[str]],
    ) -> list[str]:
        result: list[str] = []
        for pid, score in zip(sorted_ids, sorted_scores):
            tags = problem_tags.get(pid, [])
            if blind_tags & set(tags):
                result.append(pid)
        return result

    def filter_solved_problems(
        self,
        problem_ids: list[str],
        solved_problems: set[str],
        scores: np.ndarray,
    ) -> tuple[list[str], np.ndarray]:
        mask = np.array([pid not in solved_problems for pid in problem_ids])
        return [p for p, m in zip(problem_ids, mask) if m], scores[mask]

    def enforce_tag_diversity(
        self,
        problem_ids: list[str],
        scores: np.ndarray,
        problem_tags: dict[str, list[str]],
        max_per_tag: int = 2,
    ) -> tuple[list[str], np.ndarray]:
        tag_counts: dict[str, int] = {}
        selected_ids: list[str] = []
        selected_scores: list[float] = []

        for pid, score in zip(problem_ids, scores):
            tags = problem_tags.get(pid, [])
            if any(tag_counts.get(t, 0) >= max_per_tag for t in tags):
                continue
            selected_ids.append(pid)
            selected_scores.append(float(score))
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return selected_ids, np.array(selected_scores)
