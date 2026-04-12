"""User-level feature engineering.

Reads from data/processed/interactions.parquet and outputs per-user
aggregated statistics to data/features/user_features.parquet.
"""

# TODO: Implement user feature pipeline
# Key features:
#   avg_finish_time_by_tag    — per-tag average finish time (minutes)
#   solve_rate_by_difficulty  — fraction solved at each difficulty_proxy bucket
#   penalty_rate              — penalties / total attempts
#   rating_trajectory         — rating delta over last N contests
