"""LightGCN graph-based collaborative filtering.

Models users and problems as nodes in a bipartite graph.
Captures higher-order transitions: "users who struggled with X often find Y helpful."

Reference: He et al., "LightGCN: Simplifying and Powering GCN for Recommendation" (SIGIR 2020)
"""

# TODO: Implement LightGCN
# Framework options: PyTorch Geometric (PyG) or DGL — write ADR when decided
# Input: bipartite graph from interaction_matrix.py (COO format edges)
# Requires: torch, torch-geometric (in [advanced] dependency group)
