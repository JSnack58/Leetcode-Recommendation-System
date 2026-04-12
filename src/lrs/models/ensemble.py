"""Ensemble: combine scores from multiple models.

Aggregates raw scores from baseline and advanced models before passing
them to the tier assigner. Does not contain tier logic — that lives in
recommendation/tiers.py.
"""

# TODO: Implement ensemble
# Options: weighted average, rank fusion (RRF), learned meta-ranker
# Write ADR when the ensemble strategy is chosen
