"""Content-based filtering baseline.

Recommends problems similar to those the user has already solved,
based on tag vectors and difficulty proximity.
Uses TF-IDF or cosine similarity over problem tag vectors.
"""

# TODO: Implement content-based model
# Input: problem tag_vector (from features/problem_features.py)
# Approach: compute cosine similarity between user's solved problem profile
#           and all unsolved candidate problems
