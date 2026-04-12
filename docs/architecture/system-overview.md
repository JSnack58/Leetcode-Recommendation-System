# System Overview

## High-Level Data Flow

```
LeetCode Contest API
        │
        ▼
  [ Scraper ]  ──────────────────────────────────────────────────────────────
  src/lrs/data/scraper.py                                                    │
        │                                                                    │
        ▼                                                               raw contest
  data/raw/contests/                                                    JSON/Parquet
  data/raw/problems/
        │
        ▼
  [ Preprocessor ]
  src/lrs/data/preprocessor.py
        │
        ▼
  data/processed/
  ├── interactions.parquet     (user × problem, with solve time, penalties)
  └── problems_clean.parquet   (deduplicated problem catalog + tags)
        │
        ▼
  [ Feature Engineering ]
  src/lrs/features/
  ├── user_features.py         (rating history, avg finish time, penalty rate)
  ├── problem_features.py      (tag vectors, difficulty proxy, acceptance rate)
  └── interaction_matrix.py   (sparse user-problem matrix for CF models)
        │
        ▼
  data/features/
  ├── user_features.parquet
  ├── problem_features.parquet
  └── interaction_matrix/
        │
        ├──────────────────────────────────────────────────────────────────
        │                    │                    │
        ▼                    ▼                    ▼
  [ Baseline Models ]  [ Advanced Models ]  [ XGBoost Ranker ]
  svd.py / als.py      ncf.py               xgboost_ranker.py
  content_based.py     lightgcn.py
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                             ▼
                    [ Ensemble + Tier Assigner ]
                    src/lrs/models/ensemble.py
                    src/lrs/recommendation/tiers.py
                             │
                             ▼
                    ┌────────────────────┐
                    │  Recommendation    │
                    │  Output            │
                    ├────────────────────┤
                    │ Edge of Competence │
                    │ Blind Spots        │
                    │ Confidence Builders│
                    └────────────────────┘
```

## Components

| Component | Location | Responsibility |
|---|---|---|
| Scraper | `src/lrs/data/scraper.py` | Fetch contest data from LeetCode API |
| Preprocessor | `src/lrs/data/preprocessor.py` | Clean, join, type-cast raw data |
| Feature engineering | `src/lrs/features/` | Build user/problem/interaction features |
| Baseline models | `src/lrs/models/baseline/` | SVD, ALS, content-based CF |
| Advanced models | `src/lrs/models/advanced/` | NCF, LightGCN, XGBoost ranker |
| Ensemble | `src/lrs/models/ensemble.py` | Combine model scores |
| Tier assigner | `src/lrs/recommendation/tiers.py` | Assign recommendations to tiers |
| Filters | `src/lrs/recommendation/filters.py` | Remove solved problems, dedup |
| Evaluation | `src/lrs/evaluation/` | Offline metrics, temporal backtest |

## Key Design Principles

- **`recommendation/` is separate from `models/`**: Tier assignment is a business rule, not a model property. This lets us swap model backends without touching tier logic.
- **Immutable raw data**: `data/raw/` is never modified after scraping. All transformations happen in `preprocessor.py`.
- **`src/` layout**: Prevents accidental imports of the working directory. All code imports from `lrs.*`.
