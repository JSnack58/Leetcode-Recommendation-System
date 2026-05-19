"""Tests for tier assignment."""

import numpy as np
import pandas as pd

from lrs.recommendation.tiers import TierAssigner


def test_edge_of_competence_band():
    assigner = TierAssigner(edge_p_low=0.35, edge_p_high=0.65, tier_size=3)
    pids = ["a", "b", "c", "d", "e"]
    scores = np.array([0.9, 0.5, 0.4, 0.2, 0.6])
    tiers = assigner.assign_tiers("u1", pids, scores)
    assert "b" in tiers.edge_of_competence
    assert "c" in tiers.edge_of_competence
    assert "e" in tiers.edge_of_competence
    assert "a" in tiers.confidence_builders
    assert len(tiers.edge_of_competence) <= 3


def test_blind_spots_with_tags():
    assigner = TierAssigner(blind_spot_threshold=0.1, tier_size=2)
    user_feat = pd.Series({"tag_scores": {"DP": 0.2, "Graph": 0.8}})
    peer = {"DP": 0.6, "Graph": 0.75}
    tags = {"p1": ["DP"], "p2": ["Graph"], "p3": ["DP"]}
    pids = ["p1", "p2", "p3"]
    scores = np.array([0.4, 0.4, 0.3])
    tiers = assigner.assign_tiers(
        "u1", pids, scores, user_features=user_feat, peer_tag_scores=peer, problem_tags=tags
    )
    assert "p1" in tiers.blind_spots or "p3" in tiers.blind_spots
