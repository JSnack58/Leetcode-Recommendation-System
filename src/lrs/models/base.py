"""Abstract base class for all recommendation models.

Every model — baseline or advanced — must implement this interface.
Evaluation, serving, and tier-assignment code depend only on this contract,
never on model-specific internals.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Common interface for all LRS recommendation models."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "BaseRecommender":
        """Train the model on an interaction DataFrame.

        Args:
            interactions: DataFrame with at minimum columns
                          [user_id, problem_id, rating].

        Returns:
            self, to allow method chaining.
        """

    @abstractmethod
    def predict(self, user_id: str, problem_ids: list[str]) -> np.ndarray:
        """Return a predicted score for each (user, problem) pair.

        Args:
            user_id:     The user to score for.
            problem_ids: List of problem slugs to score.

        Returns:
            1-D numpy array of floats, same length as problem_ids.
            Higher score = stronger recommendation signal.
        """

    def recommend(
        self,
        user_id: str,
        candidate_problem_ids: list[str],
        k: int = 10,
    ) -> list[str]:
        """Return the top-k problem IDs for a user.

        Default implementation: score all candidates and return top-k by score.
        Override if the model has a more efficient top-k routine.

        Args:
            user_id:               The user to recommend for.
            candidate_problem_ids: Pool of problems to rank (unsolved, filtered).
            k:                     Number of recommendations to return.

        Returns:
            List of problem slugs, sorted by predicted score descending.
        """
        scores = self.predict(user_id, candidate_problem_ids)
        top_indices = np.argsort(scores)[::-1][:k]
        return [candidate_problem_ids[i] for i in top_indices]
