"""Tests for evaluation metrics."""

from lrs.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision_recall_ndcg():
    relevant = {"a", "b"}
    recommended = ["a", "c", "b", "d"]
    assert precision_at_k(relevant, recommended, 3) == 2 / 3
    assert recall_at_k(relevant, recommended, 3) == 1.0
    assert ndcg_at_k(relevant, recommended, 3) > 0.9
