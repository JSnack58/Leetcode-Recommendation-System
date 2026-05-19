"""Tests for problem catalog loading and join."""

import pandas as pd

from lrs.data.problem_catalog import join_interactions_to_catalog


def test_join_interactions_to_catalog():
    interactions = pd.DataFrame(
        {
            "frontend_id": ["1", "999"],
            "user_slug": ["u1", "u1"],
            "solved": [True, True],
        }
    )
    catalog = pd.DataFrame(
        {
            "frontend_id": ["1"],
            "problem_id": ["two-sum"],
            "title": ["Two Sum"],
            "tags": [["Array"]],
        }
    )
    merged, problems = join_interactions_to_catalog(interactions, catalog)
    assert len(merged) == 1
    assert merged.iloc[0]["problem_id"] == "two-sum"
    assert len(problems) == 1
