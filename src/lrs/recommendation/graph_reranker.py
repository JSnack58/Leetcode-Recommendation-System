"""Similarity-graph boosts for recommendation reranking."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from lrs.config import SIMILARITY_GRAPH_GML
from lrs.utils.logging import get_logger

logger = get_logger(__name__)


class SimilarityGraphReranker:
    """Boost candidates similar to user's recent struggles."""

    def __init__(self, gml_path: Path | None = None, alpha: float = 0.05):
        self.alpha = alpha
        self.graph = nx.read_gml(Path(gml_path or SIMILARITY_GRAPH_GML))
        self._slug_to_neighbors: dict[str, set[str]] = {}
        self._build_index()
        logger.info(
            f"Loaded similarity graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def _build_index(self) -> None:
        for _nid, attrs in self.graph.nodes(data=True):
            slug = attrs.get("titleSlug")
            if not slug:
                continue
            neighbors = set()
            for nb in self.graph.neighbors(_nid):
                nb_slug = self.graph.nodes[nb].get("titleSlug")
                if nb_slug:
                    neighbors.add(nb_slug)
            self._slug_to_neighbors[slug] = neighbors

    def struggle_problems(
        self,
        interactions: list[dict],
        min_penalty: int = 1,
    ) -> set[str]:
        """Problems user struggled with (penalties or unsolved)."""
        struggles = set()
        for row in interactions:
            pid = row.get("problem_id")
            if not pid:
                continue
            if row.get("penalty_count", 0) >= min_penalty or not row.get("solved", True):
                struggles.add(pid)
        return struggles

    def graph_boost(
        self,
        problem_ids: list[str],
        struggle_slugs: set[str],
    ) -> np.ndarray:
        """Per-candidate boost if 1-hop similar to a struggle problem."""
        boosts = np.zeros(len(problem_ids), dtype=np.float32)
        struggle_neighbors: set[str] = set()
        for s in struggle_slugs:
            struggle_neighbors |= self._slug_to_neighbors.get(s, set())

        for i, pid in enumerate(problem_ids):
            if pid in struggle_neighbors:
                boosts[i] = self.alpha
            else:
                pid_neighbors = self._slug_to_neighbors.get(pid, set())
                if pid_neighbors & struggle_slugs:
                    boosts[i] = self.alpha * 0.5
        return boosts

    def clique_penalty(
        self,
        selected: list[str],
        candidate: str,
    ) -> float:
        """Penalize if candidate is similar to already-selected problems."""
        if not selected:
            return 0.0
        cand_n = self._slug_to_neighbors.get(candidate, set())
        for s in selected:
            if candidate == s or s in cand_n:
                return 0.02
        return 0.0
