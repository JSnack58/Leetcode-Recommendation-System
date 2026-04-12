"""Temporal train/test split evaluation harness.

Uses contest date as the temporal boundary:
  - Train: contests before cutoff date
  - Test:  most recent N contests

This simulates the real-world scenario where we predict on unseen future contests.
Standard random splits would leak future information into training.
"""

# TODO: Implement temporal backtest harness
# Key function: run_backtest(model, interactions, n_test_contests) -> MetricsDict
