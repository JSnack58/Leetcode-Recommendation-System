# LeetCode Recommendation System - Technical Specification

## Executive Summary

This document specifies a **hybrid ensemble recommendation system** that balances challenge and achievable success while driving skill improvement. The system will support **real-time recommendations** with a **2-week deployment timeline**.

---

## 1. System Overview

### 1.1 Core Objective

Build a recommendation system that:
- **Balances challenge and achievability**: Recommend problems with ~50-70% predicted solve probability
- **Drives skill improvement**: Surface problems in underperforming areas while reinforcing foundations
- **Supports real-time serving**: Sub-second latency for recommendation requests

### 1.2 Key Constraints

| Constraint | Value |
|---|---|
| Deployment Timeline | 2 weeks |
| Hardware | RTX 3090 + RTX 3080 GPUs |
| Latency Target | < 100ms per recommendation request |
| Users | 856,274 unique users |
| Problems | ~3,000 LeetCode problems |
| Interactions | 3.4M contest records |

---

## 2. Architecture Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Real-Time Serving Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │   User       │────▶│  Score       │────▶│   Tier       │       │
│   │   Request    │     │  Ensemble    │     │   Assigner   │       │
│   └──────────────┘     └──────────────┘     └──────────────┘       │
│                              │                    │                  │
│                              ▼                    ▼                  │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │              Pre-computed Candidate Pool                     │    │
│   │   (Top 500 problems per user, updated hourly)               │    │
│   └────────────────────────────────────────────────────────────┘    │
│                              │                    │                  │
│                              ▼                    ▼                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │   ALS        │     │   NCF        │     │   Content    │       │
│   │   (Bias SVD) │     │   (LightGCN) │     │   (Tags)     │       │
│   └──────────────┘     └──────────────┘     └──────────────┘       │
│                              │                    │                  │
│                              └────────┬───────────┘                  │
│                                       ▼                              │
│                              ┌──────────────┐                        │
│                              │   Ranker     │                        │
│                              │   (Final     │                        │
│                              │    ordering) │                        │
│                              └──────────────┘                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Offline Training Layer                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │   Data       │────▶│   Feature    │────▶│   Model      │       │
│   │   Ingestion  │     │   Engine     │     │   Training   │       │
│   └──────────────┘     └──────────────┘     └──────────────┘       │
│                              │                    │                  │
│                              ▼                    ▼                  │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │              Feature Store (Parquet)                         │    │
│   │   - user_features.parquet                                    │    │
│   │   - problem_features.parquet                                 │    │
│   │   - interaction_matrix.npz                                   │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Latency Budget |
|---|---|---|
| **ALS Model** | Fast baseline with bias terms | < 5ms |
| **LightGCN Model** | Graph-based embeddings | < 20ms |
| **Content-Based** | Tag similarity scoring | < 5ms |
| **Ensemble** | Weighted score combination | < 2ms |
| **Tier Assigner** | Classify into 3 tiers | < 5ms |
| **Ranker** | Final ordering | < 3ms |
| **Total** | End-to-end | < 100ms |

---

## 3. Data Model

### 3.1 Rating Signal Definition

**Primary Signal**: Normalized contest score (0-23)

```python
# Rating calculation
rating = (contest_score / max_contest_score) * 100
# Normalized to [0, 1] range for model training
normalized_rating = rating / 100.0
```

**Why this signal**:
- Captures both success (solved) and performance quality (score)
- Comparable across contests after normalization
- Richer than binary (solved/not solved)

### 3.2 Feature Engineering

#### User Features

| Feature | Type | Description |
|---|---|---|
| `user_rating` | float | Current contest rating |
| `rating_percentile` | float | User's rating percentile in their cohort |
| `lang_preference` | categorical | Most used submission language |
| `country_region` | categorical | US/CN/Other (Unknown excluded) |
| `total_contests` | int | Number of contests participated |
| `avg_score` | float | Mean contest score |
| `score_std` | float | Standard deviation of scores |
| `tag_scores` | vector | Per-tag average score (DP, Graph, etc.) |
| `difficulty_performance` | vector | Performance by difficulty level |

#### Problem Features

| Feature | Type | Description |
|---|---|---|
| `difficulty_label` | categorical | Easy/Medium/Hard |
| `difficulty_proxy` | float | Derived from solver behavior |
| `tags` | list | Problem tags (DP, Array, etc.) |
| `acceptance_rate` | float | Global acceptance rate |
| `avg_finish_time` | float | Average solver finish time |
| `penalty_rate` | float | Average penalty rate |

#### Interaction Features

| Feature | Type | Description |
|---|---|---|
| `user_problem_score` | float | Historical score on this problem |
| `user_problem_solved` | bool | Whether user solved this problem |
| `user_problem_attempts` | int | Number of attempts |
| `peer_avg_score` | float | Average score from similar users |
| `tag_overlap` | float | Tag overlap with user's strong areas |

---

## 4. Model Specifications

### 4.1 ALS (Bias Matrix Factorization)

**Purpose**: Fast baseline capturing core user-problem signal

**Mathematical Formulation**:
```
r̂_ui = μ + b_u + b_i + q_u^T p_i
```

Where:
- `μ` = global mean score
- `b_u` = user bias (learned per user)
- `b_i` = item bias (learned per problem)
- `q_u` = user latent factor vector (dim: 64)
- `p_i` = item latent factor vector (dim: 64)

**Training**:
- Algorithm: Alternating Least Squares with regularization
- Regularization: λ = 0.1
- Latent factors: 64 dimensions
- Bias terms: Yes
- Convergence: 15 iterations

**Inference**:
- Pre-compute user and item biases
- Dot product of latent factors
- **Latency**: < 5ms

**Cold Start**:
- New users: `μ + b_i` (global mean + item bias)
- New items: `μ + b_u` (global mean + user bias)

### 4.2 LightGCN (Graph Neural Network)

**Purpose**: Capture complex non-linear user-problem interactions

**Architecture**:
```
Input Layer:
  - User IDs (856K users)
  - Problem IDs (~3K problems)

Embedding Layer:
  - User embeddings: 64-dim
  - Item embeddings: 64-dim

GCN Layers (3 layers):
  Layer 1: User → Item, Item → User (message passing)
  Layer 2: Aggregated embeddings
  Layer 3: Final embeddings

Output Layer:
  - Concatenate user + item embeddings
  - Dot product → predicted score
  - Sigmoid activation → [0, 1]
```

**Training**:
- Loss: MSE on normalized scores
- Optimizer: Adam (lr=0.001)
- Batch size: 4096
- Epochs: 50
- Dropout: 0.2

**Inference**:
- Pre-compute all user and item embeddings
- Dot product for scoring
- **Latency**: < 20ms (GPU)

**Cold Start**:
- New users: Initialize with random embedding, fine-tune on first interactions
- New items: Use content-based initialization from problem features

### 4.3 Content-Based Filtering

**Purpose**: Handle cold start and ensure diversity

**Scoring**:
```python
content_score = similarity(user_tags, problem_tags) * difficulty_match(user_level, problem_difficulty)
```

**Tag Similarity**: Cosine similarity between user's tag preference vector and problem's tag vector

**Difficulty Match**:
- Easy: Score boost for users at rating < 1200
- Medium: Score boost for users at rating 1200-1800
- Hard: Score boost for users at rating > 1800

**Latency**: < 5ms (vectorized operations)

---

## 5. Ensemble Strategy

### 5.1 Score Combination

**Weighted Average**:
```
final_score = w_als * als_score + w_lightgcn * lightgcn_score + w_content * content_score
```

**Initial Weights** (tunable):
| Model | Weight | Rationale |
|---|---|---|
| ALS | 0.3 | Fast, reliable baseline |
| LightGCN | 0.5 | Captures complex patterns |
| Content-Based | 0.2 | Diversity and cold-start |

**Adaptive Weights** (future):
- Per-user weighting based on historical performance
- Meta-ranker trained to optimize for solve rate

### 5.2 Score Calibration

**Purpose**: Ensure scores are interpretable as probabilities

**Method**: Platt Scaling (logistic regression on validation set)

```python
# Calibrate raw scores to [0, 1] probability
calibrated_score = sigmoid(a * raw_score + b)
```

**Training**: Fit `a` and `b` on held-out validation data

---

## 6. Tier Assignment

### 6.1 Tier Definitions

| Tier | Goal | Probability Range | Selection Criteria |
|---|---|---|---|
| **Edge of Competence** | Challenge just above current level | 0.35 - 0.65 | Predicted solve probability in range |
| **Blind Spots** | Address underperforming areas | N/A | User tag score < peer avg - 0.15 |
| **Confidence Builders** | Reinforce foundations | > 0.75 | High solve probability + fast finish time |

### 6.2 Blind Spot Detection

**Algorithm**:
1. Define peer group: users within ±100 rating points
2. Compute user's average score per tag
3. Compute peer group average per tag
4. Flag tags where: `user_tag_score < peer_avg - threshold`
5. Recommend top problems from flagged tags

**Threshold**: 0.15 (15% below peer average)

### 6.3 Confidence Builder Selection

**Criteria**:
1. Predicted solve probability > 0.75
2. Predicted finish time < user's median for that difficulty
3. Problem not recently attempted (within last 5 contests)

---

## 7. Real-Time Serving

### 7.1 Serving Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                               │
│                    (FastAPI / uvicorn)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Recommendation Service                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  POST /recommend/{user_id}                            │   │
│  │  Response: {                                          │   │
│  │    "edge_of_competence": [...],                      │   │
│  │    "blind_spots": [...],                            │   │
│  │    "confidence_builders": [...]                      │   │
│  │  }                                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Serving Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   ALS        │  │   LightGCN   │  │   Content    │       │
│  │   (CPU)      │  │   (GPU)      │  │   (CPU)      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Candidate Pool Strategy

**Pre-computation**:
- Hourly batch job computes top 500 candidate problems per user
- Stored in Redis for fast retrieval
- Reduces scoring from 3K problems to 500 per request

**Refresh Strategy**:
- Full refresh: Every 6 hours
- Incremental: Every hour for active users
- Trigger: New contest results, model retraining

### 7.3 Caching Strategy

| Cache | TTL | Content |
|---|---|---|
| **User Embeddings** | 24 hours | LightGCN user vectors |
| **Problem Embeddings** | 7 days | LightGCN item vectors |
| **Candidate Pool** | 1 hour | Top 500 problems per user |
| **Tier Assignments** | 1 hour | Pre-computed tier classifications |

---

## 8. Training Pipeline

### 8.1 Data Pipeline

```
Raw Contest Data (JSONL)
         │
         ▼
Preprocessing (clean, normalize, join)
         │
         ▼
Feature Engineering
    ┌────┴────┐
    ▼         ▼
User    Problem
Features Features
    └────┬────┘
         ▼
Interaction Matrix (sparse)
         │
         ▼
Train/Validation/Test Split (temporal)
```

### 8.2 Training Schedule

| Task | Frequency | Duration |
|---|---|---|
| **Feature Engineering** | Daily | 30 min |
| **ALS Training** | Daily | 10 min |
| **LightGCN Training** | Weekly | 2 hours |
| **Model Evaluation** | Weekly | 30 min |
| **Candidate Pool Update** | Hourly | 5 min |

### 8.3 Hyperparameter Configuration

**ALS**:
```yaml
regularization: 0.1
n_factors: 64
n_iterations: 15
```

**LightGCN**:
```yaml
embedding_dim: 64
n_layers: 3
batch_size: 4096
learning_rate: 0.001
epochs: 50
dropout: 0.2
```

**Ensemble**:
```yaml
weights:
  als: 0.3
  lightgcn: 0.5
  content: 0.2
```

---

## 9. Evaluation Strategy

### 9.1 Offline Metrics

| Metric | Description | Target |
|---|---|---|
| **NDCG@10** | Normalized discounted cumulative gain | > 0.4 |
| **Precision@10** | Fraction of recommended problems solved | > 0.35 |
| **MRR** | Mean reciprocal rank | > 0.25 |
| **Coverage** | % of problems that can be recommended | > 0.8 |
| **Novelty** | Average distance from user's history | > 0.5 |

### 9.2 Temporal Backtest

**Split Strategy**:
- Training: Contests 1-250
- Validation: Contests 251-280
- Test: Contests 281-341

**Evaluation Procedure**:
1. Train on training set
2. Tune hyperparameters on validation set
3. Evaluate on test set (held-out contests)
4. Compare against baseline (ALS only)

### 9.3 A/B Testing Framework (Future)

**Metrics to track**:
- Click-through rate on recommendations
- Problem solve rate for recommended problems
- User engagement (time spent on platform)
- Long-term skill improvement (rating growth)

---

## 10. Deployment Plan

### 10.1 Week 1: Core Implementation

| Day | Task | Deliverable |
|---|---|---|
| 1-2 | Data preprocessing pipeline | Clean feature store |
| 3-4 | ALS model implementation | Trained ALS model |
| 5-7 | LightGCN model implementation | Trained LightGCN model |

### 10.2 Week 2: Integration & Deployment

| Day | Task | Deliverable |
|---|---|---|
| 8-9 | Ensemble and tier logic | Working recommendation engine |
| 10 | API service implementation | REST API endpoint |
| 11 | Evaluation and tuning | Performance metrics |
| 12 | Deployment and monitoring | Production system |

### 10.3 Infrastructure Requirements

| Component | Specification |
|---|---|
| **GPU** | RTX 3090 / 3080 (for LightGCN inference) |
| **CPU** | 8 cores (for ALS, content-based) |
| **Memory** | 32GB RAM |
| **Storage** | 100GB SSD (for model artifacts) |
| **Cache** | Redis (for candidate pool) |

### 10.4 Monitoring

**Metrics to monitor**:
- Recommendation latency (p50, p95, p99)
- Model prediction distribution
- User engagement metrics
- Error rates

---

## 11. Risk Mitigation

| Risk | Mitigation |
|---|---|
| **Model training fails** | Fallback to ALS-only mode |
| **High latency** | Reduce candidate pool size, optimize embeddings |
| **Cold start issues** | Content-based fallback for new users |
| **Data quality issues** | Validation checks, anomaly detection |

---

## 12. Open Questions

1. **Should we use adaptive ensemble weights per-user or global weights?**
   - Global: Simpler, faster
   - Per-user: More personalized, requires meta-model

2. **What's the optimal candidate pool size?**
   - Tradeoff: Larger pool = better recommendations, higher latency

3. **Should we incorporate temporal features (user improvement trend)?**
   - Pros: More personalized to current skill level
   - Cons: More complex, requires temporal modeling

4. **How frequently should we retrain the LightGCN model?**
   - Weekly: Balance between freshness and compute cost
   - Daily: More fresh, higher compute cost

---

## 13. Next Steps

1. **Review and approve this specification**
2. **Switch to Code mode for implementation**
3. **Begin Week 1 tasks (data preprocessing, ALS model)**
4. **Set up evaluation framework**
