"""Tests for ALS recommender on fixture data."""

import numpy as np

from lrs.models.baseline.als import ALSRecommender
def test_als_fit_predict(sample_interactions):
    model = ALSRecommender(latent_dim=8, iterations=5)
    model.fit(sample_interactions)
    pids = ["two-sum", "add-two-numbers", "longest-substring"]
    scores = model.predict("alice", pids)
    assert len(scores) == 3
    assert np.all((scores >= 0) & (scores <= 1))
