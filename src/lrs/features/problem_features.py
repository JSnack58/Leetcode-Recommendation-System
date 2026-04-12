"""Problem-level feature engineering.

Reads from data/processed/problems_clean.parquet and interaction data,
outputs to data/features/problem_features.parquet.
"""

# TODO: Implement problem feature pipeline
# Key features:
#   tag_vector              — multi-hot encoding of problem tags
#   acceptance_rate         — global acceptance rate from LeetCode
#   difficulty_proxy        — contest-derived difficulty (vs static Easy/Medium/Hard)
#   avg_peer_finish_time    — average finish time across all users who attempted it
