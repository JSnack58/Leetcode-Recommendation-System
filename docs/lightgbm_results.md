# LightGBM Model Results

**Author:** Kayvaun  
**Date:** May 2026  
**Model:** LightGBM Binary Classifier (predict P(solved) per user-problem pair)

---

## Dataset

- **Source:** 60 biweekly LeetCode contests (contests 61-120), scraped from the public contest ranking API
- **Total interactions:** 4,651,228 rows (user-problem pairs, including negative samples for unattempted problems)
- **Unique problems:** 240
- **Train/validation split:** 80/20, stratified by solved label

## Model Configuration

- Objective: binary cross-entropy (binary_logloss)
- Boosting: GBDT
- Num leaves: 63
- Learning rate: 0.05
- Feature fraction: 0.8
- Bagging fraction: 0.8
- Rounds: 500 (no early stopping triggered)

## Training Results

| Metric | Train | Validation |
|--------|-------|------------|
| Binary Logloss | 0.2542 | 0.2567 |

The small gap between train and validation logloss indicates the model generalizes well without overfitting.

## Feature Importance (by Gain)

| Rank | Feature | Gain | Description |
|------|---------|------|-------------|
| 1 | overall_solve_rate | 9,409,006 | User's historical solve rate across all contests |
| 2 | solve_rate | 7,054,090 | Problem's global solve rate (proxy for difficulty) |
| 3 | avg_rank | 2,294,718 | User's average contest ranking |
| 4 | total_solves | 2,029,857 | Total number of solves for the problem |
| 5 | avg_solve_time_problem | 1,081,532 | Average time to solve the problem |
| 6 | total_solved | 845,426 | Total problems the user has solved |
| 7 | total_contests | 246,829 | Number of contests the user participated in |
| 8 | avg_solve_time | 244,611 | User's average solve time |
| 9 | total_attempts | 169,729 | Total attempts on the problem |
| 10 | avg_fail_count_problem | 155,526 | Average failed submissions per problem |
| 11 | avg_fail_count | 87,438 | User's average failed submissions |
| 12 | num_languages_used | 60,551 | Number of programming languages the user has used |
| 13 | total_attempted | 23,898 | Total problems the user has attempted |
| 14 | num_contests_appeared | 0 | Number of contests the problem appeared in |

## Key Insights

1. **User skill vs. problem difficulty** are the dominant signals. The top two features (overall_solve_rate and solve_rate) together account for ~68% of the total gain, confirming that the model primarily matches user ability level to problem difficulty.

2. **Side features matter.** Features like avg_solve_time_problem, total_contests, and avg_fail_count provide signal that pure collaborative filtering (ALS) cannot capture. This is the main advantage of the LightGBM approach.

3. **Interpretability.** Unlike ALS which uses latent factors, LightGBM can explain its predictions through feature importance. For example, a recommendation can be justified as: "This problem has a solve rate close to your historical ability level, and the average solve time is within your range."

4. **num_contests_appeared has zero importance** because each problem only appears in a single contest, making this feature constant and uninformative.

## Comparison with Team Models

| Aspect | Heuristic (Max) | ALS (Anthony) | LightGBM (Kayvaun) |
|--------|-----------------|---------------|---------------------|
| Approach | Rule-based + graph lookups | Collaborative filtering | Gradient-boosted classifier |
| Side features | N/A | No (latent factors only) | Yes (user + problem stats) |
| Cold start handling | Yes | Poor | Yes (uses median features) |
| Interpretability | High | Low | Medium (feature importance) |
| Personalization | Limited | Strong | Strong |

## How to Reproduce

```bash
# Step 1: Preprocess raw contest data
py -c "from lrs.data.preprocessor import preprocess; preprocess()"

# Step 2: Train the model
py scripts/train_advanced.py --model lightgbm
```

Model artifacts are saved to `models/lightgbm/`.
