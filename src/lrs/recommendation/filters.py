"""Post-scoring filters for recommendations."""

from __future__ import annotations

import numpy as np


def filter_solved(
    problem_ids: list[str],
    scores: np.ndarray,
    solved: set[str],
) -> tuple[list[str], np.ndarray]:
    """Remove already-solved problems."""
    mask = [pid not in solved for pid in problem_ids]
    return [p for p, m in zip(problem_ids, mask) if m], scores[np.array(mask)]


def dedup(problem_ids: list[str], scores: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Keep first occurrence of each problem_id."""
    seen: set[str] = set()
    ids: list[str] = []
    sc: list[float] = []
    for pid, s in zip(problem_ids, scores):
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
            sc.append(float(s))
    return ids, np.array(sc)


def apply_filters(
    problem_ids: list[str],
    scores: np.ndarray,
    solved: set[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Apply standard filter chain."""
    if solved:
        problem_ids, scores = filter_solved(problem_ids, scores, solved)
    return dedup(problem_ids, scores)
