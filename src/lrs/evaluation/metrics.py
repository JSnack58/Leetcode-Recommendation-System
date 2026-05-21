"""Offline evaluation metrics for recommendation quality."""

import numpy as np
from typing import List, Set, Dict


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Square Error.
    
    Args:
        actual: Array of actual ratings
        predicted: Array of predicted ratings
    
    Returns:
        RMSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length")
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def calculate_ndcg_at_k(relevant: Set[str], recommended: List[str], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        relevant: Set of relevant (ground truth) item IDs
        recommended: List of recommended item IDs (ordered by score)
        k: Number of recommendations to consider
    
    Returns:
        NDCG@k value (0.0 to 1.0)
    """
    relevant = set(relevant)
    k = min(k, len(recommended))
    recommended_k = recommended[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            relevance = 1.0  # Binary relevance
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    num_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    
    # Avoid division by zero
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return float(ndcg)


def calculate_precision_at_k(relevant: Set[str], recommended: List[str], k: int = 10) -> float:
    """Calculate Precision at k.
    
    Args:
        relevant: Set of relevant (ground truth) item IDs
        recommended: List of recommended item IDs (ordered by score)
        k: Number of recommendations to consider
    
    Returns:
        Precision@k value (0.0 to 1.0)
    """
    relevant = set(relevant)
    k = min(k, len(recommended))
    recommended_k = recommended[:k]
    
    hits = sum(1 for item in recommended_k if item in relevant)
    precision = hits / k if k > 0 else 0.0
    
    return float(precision)


def calculate_recall_at_k(relevant: Set[str], recommended: List[str], k: int = 10) -> float:
    """Calculate Recall at k.
    
    Args:
        relevant: Set of relevant (ground truth) item IDs
        recommended: List of recommended item IDs (ordered by score)
        k: Number of recommendations to consider
    
    Returns:
        Recall@k value (0.0 to 1.0)
    """
    relevant = set(relevant)
    k = min(k, len(recommended))
    recommended_k = recommended[:k]
    
    hits = sum(1 for item in recommended_k if item in relevant)
    recall = hits / len(relevant) if len(relevant) > 0 else 0.0
    
    return float(recall)


def calculate_mean_reciprocal_rank(relevant: Set[str], recommended: List[str]) -> float:
    """Calculate Mean Reciprocal Rank.
    
    Args:
        relevant: Set of relevant (ground truth) item IDs
        recommended: List of recommended item IDs (ordered by score)
    
    Returns:
        Reciprocal rank (1/rank of first relevant item, 0 if none found)
    """
    relevant = set(relevant)
    
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_coverage(recommended: List[Set[str]], all_items: Set[str]) -> float:
    """Calculate catalog coverage.
    
    Args:
        recommended: List of sets of recommended items (one set per user)
        all_items: Set of all available items in the catalog
    
    Returns:
        Coverage fraction (0.0 to 1.0)
    """
    recommended_items = set()
    for rec_set in recommended:
        recommended_items.update(rec_set)
    
    coverage = len(recommended_items) / len(all_items) if len(all_items) > 0 else 0.0
    
    return float(coverage)


def calculate_novelty(recommended: List[List[str]], item_popularity: Dict[str, float]) -> float:
    """Calculate average novelty (inverse popularity) of recommendations.
    
    Args:
        recommended: List of recommended item lists (one per user)
        item_popularity: Dictionary mapping item_id to popularity score
    
    Returns:
        Average novelty score
    """
    novelty_scores = []
    
    for rec_list in recommended:
        for item in rec_list:
            popularity = item_popularity.get(item, 0.0)
            # Novelty = 1 / (popularity + 1) to avoid division by zero
            novelty = 1.0 / (popularity + 1.0)
            novelty_scores.append(novelty)
    
    avg_novelty = np.mean(novelty_scores) if novelty_scores else 0.0
    
    return float(avg_novelty)


def calculate_hit_rate(relevant: Set[str], recommended: List[str], k: int = 10) -> float:
    """Calculate Hit Rate at k (fraction of users with at least one hit).
    
    Args:
        relevant: Set of relevant (ground truth) item IDs
        recommended: List of recommended item IDs (ordered by score)
        k: Number of recommendations to consider
    
    Returns:
        Hit rate (0.0 to 1.0)
    """
    relevant = set(relevant)
    k = min(k, len(recommended))
    recommended_k = recommended[:k]
    
    hit = any(item in relevant for item in recommended_k)
    
    return 1.0 if hit else 0.0


class EvaluationMetrics:
    """Container for aggregated evaluation metrics."""
    
    def __init__(self):
        self.rmse_values: List[float] = []
        self.ndcg_values: List[float] = []
        self.precision_values: List[float] = []
        self.recall_values: List[float] = []
        self.mrr_values: List[float] = []
        self.hit_rate_values: List[float] = []
    
    def add_sample(self, rmse: float, ndcg: float, precision: float, 
                   recall: float, mrr: float, hit_rate: float):
        """Add a sample of metrics."""
        self.rmse_values.append(rmse)
        self.ndcg_values.append(ndcg)
        self.precision_values.append(precision)
        self.recall_values.append(recall)
        self.mrr_values.append(mrr)
        self.hit_rate_values.append(hit_rate)
    
    def compute_averages(self) -> Dict[str, float]:
        """Compute average metrics across all samples."""
        return {
            "rmse": np.mean(self.rmse_values) if self.rmse_values else 0.0,
            "ndcg_at_10": np.mean(self.ndcg_values) if self.ndcg_values else 0.0,
            "precision_at_10": np.mean(self.precision_values) if self.precision_values else 0.0,
            "recall_at_10": np.mean(self.recall_values) if self.recall_values else 0.0,
            "mrr": np.mean(self.mrr_values) if self.mrr_values else 0.0,
            "hit_rate_at_10": np.mean(self.hit_rate_values) if self.hit_rate_values else 0.0,
        }
    
    def print_results(self):
        """Print formatted results to stdout."""
        averages = self.compute_averages()
        
        print("\n" + "=" * 60)
        print("ALS MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"{'Metric':<30} {'Value':>15}")
        print("-" * 60)
        print(f"{'RMSE':<30} {averages['rmse']:>15.6f}")
        print(f"{'NDCG@10':<30} {averages['ndcg_at_10']:>15.6f}")
        print(f"{'Precision@10':<30} {averages['precision_at_10']:>15.6f}")
        print(f"{'Recall@10':<30} {averages['recall_at_10']:>15.6f}")
        print(f"{'MRR':<30} {averages['mrr']:>15.6f}")
        print(f"{'Hit Rate@10':<30} {averages['hit_rate_at_10']:>15.6f}")
        print("=" * 60)
