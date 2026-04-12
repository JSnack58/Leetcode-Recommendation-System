"""Build sparse user-problem interaction matrix for collaborative filtering.

The composite "rating" is derived from solved status + finish_time percentile.
Outputs a scipy CSR matrix saved to data/features/interaction_matrix/.
"""

# TODO: Implement interaction matrix builder
# Key decisions to make (write ADR when decided):
#   - Rating formula: e.g., solved * (1 - finish_time_percentile) or similar
#   - How to encode unsolved attempts (penalty for trying and failing?)
#   - Sparse format: scipy CSR for implicit/surprise, COO for PyG
