# Project Roadmap

## Phase 0: Data Collection & EDA
**Goal**: Have a clean, analyzed dataset ready for modeling.

- [x] Implement LeetCode contest API scraper (`src/lrs/data/scraper.py`) — see [ADR-0002](../adr/0002-contest-scraping-approach.md)
- [ ] Scrape last 50 contests → target ~1M+ user-problem interactions
- [ ] Build preprocessing pipeline (cleaning, type enforcement, joins)
- [ ] EDA: user rating distribution, solve rates by tag, time-to-solve distributions
- [ ] Define interaction matrix schema and save to `data/processed/`

**Exit criteria**: `data/processed/interactions.parquet` exists and passes schema validation.

---

## Phase 1: Baseline Models
**Goal**: Establish measurable offline baselines before adding complexity.

- [ ] Implement SVD baseline (`src/lrs/models/baseline/svd.py`)
- [ ] Implement ALS baseline (`src/lrs/models/baseline/als.py`)
- [ ] Implement content-based filtering (`src/lrs/models/baseline/content_based.py`)
- [ ] Wire up evaluation harness (`src/lrs/evaluation/backtest.py`)
- [ ] Record NDCG@10, Precision@10, MRR for each baseline

**Exit criteria**: All three baselines evaluated; results recorded in `experiments-log.md`.

---

## Phase 2: Advanced Models
**Goal**: Beat baseline NDCG@10 with at least one advanced model.

- [ ] Implement NCF (`src/lrs/models/advanced/ncf.py`)
- [ ] Implement LightGCN (`src/lrs/models/advanced/lightgcn.py`)
- [ ] Implement XGBoost ranker (`src/lrs/models/advanced/xgboost_ranker.py`)
- [ ] Hyperparameter search for each model
- [ ] Ablation study: what does each feature contribute?

**Exit criteria**: At least one advanced model beats best baseline by ≥5% NDCG@10.

---

## Phase 3: Multi-Tier Recommendation Logic
**Goal**: Implement the three-tier recommendation strategy.

- [ ] Implement tier assignment logic (`src/lrs/recommendation/tiers.py`)
  - Edge of Competence: latent score just above user's current ceiling
  - Blind Spots: tags where user's score diverges from peer group average
  - Confidence Builders: high predicted success rate + fast solve time
- [ ] Implement dedup / already-solved filter (`src/lrs/recommendation/filters.py`)
- [ ] Implement final ranker (`src/lrs/recommendation/ranker.py`)
- [ ] End-to-end integration test

**Exit criteria**: `generate_recommendations.py` produces plausible, non-overlapping tiers for a test user.

---

## Phase 4: Evaluation & Iteration
**Goal**: Validate tier quality and iterate.

- [ ] Offline evaluation: per-tier precision, diversity, novelty
- [ ] User study or A/B test design
- [ ] Document final model selection in an ADR
- [ ] Write architecture docs for final system

**Exit criteria**: Final model selected, documented, and artifacts versioned.
