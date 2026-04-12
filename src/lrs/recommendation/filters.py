"""Post-scoring filters applied before final ranking.

Filters are applied after tier assignment to enforce quality constraints.
"""

# TODO: Implement filters
# Key filters:
#   already_solved_filter(candidates, user_id) -> remove problems user has solved
#   dedup_filter(candidates)                   -> remove duplicate problem_ids
#   tag_diversity_filter(candidates, n_per_tag) -> cap problems per tag to avoid
#                                                  over-indexing one topic
