"""Cleaning and joining pipeline for raw contest data.

Reads from data/raw/, outputs to data/processed/.
Immutability rule: never read from processed/ as input to this module.
"""

# TODO: Implement preprocessing pipeline
# Key transformations:
#   - Drop duplicate submissions (same user + problem in same contest)
#   - Normalize finish_time to minutes; cap outliers at 90th percentile
#   - Join problem metadata (tags, acceptance_rate, difficulty_label)
#   - Compute difficulty_proxy from finish_time + penalty_count
#   - Output: interactions.parquet, problems_clean.parquet
