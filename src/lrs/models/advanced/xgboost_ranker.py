"""XGBoost/LightGBM learning-to-rank model.

Incorporates metadata features that standard matrix factorization ignores:
penalty counts, programming language, finish times, and rating trajectory.

Uses a pairwise or listwise ranking objective.
"""

# TODO: Implement XGBoost ranker
# Feature inputs: user_features + problem_features joined on candidates
# Objective: xgb.train with objective="rank:pairwise" or lgbm LambdaRank
# Labels: derived from solved + finish_time (see interaction_matrix.py rating formula)
