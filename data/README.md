# Data

All data directories are **gitignored**. Large files are managed outside git (DVC, S3, or GCS).

## Directory Structure

```
data/
├── raw/           ← Immutable original scrapes. Never modify after creation.
│   ├── contests/  ← Raw contest API responses (JSON or Parquet, one file per contest)
│   └── problems/  ← Raw problem metadata snapshot
├── processed/     ← Cleaned, joined, type-cast data
│   ├── interactions.parquet
│   └── problems_clean.parquet
└── features/      ← Feature-engineered outputs consumed by models
    ├── user_features.parquet
    ├── problem_features.parquet
    └── interaction_matrix/
```

## Data Provenance

| File | Source | Script |
|---|---|---|
| `raw/contests/` | LeetCode Contest API | `src/lrs/data/scraper.py` |
| `raw/problems/` | LeetCode Problem API | `src/lrs/data/scraper.py` |
| `processed/interactions.parquet` | Derived from `raw/contests/` | `src/lrs/data/preprocessor.py` |
| `processed/problems_clean.parquet` | Derived from `raw/problems/` | `src/lrs/data/preprocessor.py` |
| `features/` | Derived from `processed/` | `src/lrs/features/*.py` |

## Accessing Data

If you are a new team member:

1. Ask a maintainer for access to the shared storage bucket / DVC remote.
2. Run `dvc pull` (or follow the manual download instructions provided separately).

## Schema

See [docs/architecture/data-pipeline.md](../docs/architecture/data-pipeline.md) for full column-level schema documentation.
