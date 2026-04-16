"""Shared pytest fixtures for all LRS tests.

Provides small, in-memory DataFrames that represent the schema of real data
without requiring actual data files to be present.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """Minimal interaction DataFrame with the production schema."""
    return pd.DataFrame(
        {
            "user_id": ["alice", "alice", "bob", "bob", "carol"],
            "contest_id": ["weekly-400"] * 5,
            "problem_id": ["two-sum", "add-two-numbers", "two-sum", "longest-substring", "two-sum"],
            "solved": [True, False, True, True, False],
            "finish_time_min": [5.2, None, 8.1, 22.5, None],
            "penalty_count": [0, 2, 1, 0, 3],
            "language": ["python3"] * 5,
            "user_rating": [1500.0, 1500.0, 1800.0, 1800.0, 1200.0],
            "difficulty_proxy": [0.2, 0.6, 0.2, 0.7, 0.2],
        }
    )


@pytest.fixture
def sample_problems() -> pd.DataFrame:
    """Minimal problem catalog DataFrame."""
    return pd.DataFrame(
        {
            "problem_id": ["two-sum", "add-two-numbers", "longest-substring"],
            "title": [
                "Two Sum",
                "Add Two Numbers",
                "Longest Substring Without Repeating Characters",
            ],
            "tags": [["Array", "Hash Table"], ["Linked List"], ["Sliding Window", "String"]],
            "difficulty_label": ["Easy", "Medium", "Medium"],
            "acceptance_rate": [0.49, 0.39, 0.33],
        }
    )
