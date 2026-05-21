"""Tests for score spread utilities."""

import numpy as np

from lrs.models.score_utils import has_score_variance, minmax_spread


def test_minmax_spread_flat_input():
    scores = minmax_spread(np.array([5.0, 5.0, 5.0]), low=0.3, high=0.8)
    assert len(np.unique(np.round(scores, 2))) == 3
    assert scores.min() >= 0.3
    assert scores.max() <= 0.8


def test_has_score_variance():
    assert not has_score_variance(np.array([0.64, 0.64, 0.64]))
    assert has_score_variance(np.array([0.3, 0.5, 0.7]))
