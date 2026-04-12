# Notebooks

Exploratory analysis and model prototyping. Notebooks are **not** the source of truth for production code — once an approach is validated here, it gets implemented in `src/lrs/`.

## Naming Convention

```
{phase_number}_{descriptive_name}.ipynb
```

Examples: `01_data_overview.ipynb`, `02_svd_baseline.ipynb`

## Directories

| Directory | Purpose |
|---|---|
| `01_eda/` | Exploratory data analysis — distributions, missing values, user/problem statistics |
| `02_baselines/` | Baseline model prototyping (SVD, ALS, content-based) |
| `03_advanced_models/` | NCF, LightGCN, XGBoost experiments |
| `04_evaluation/` | Metric analysis, recommendation quality inspection |

## Policy

- **Strip outputs before committing**: Run `make notebook-clean` or `nbstripout` before every commit. Output cells bloat the repo and create noisy diffs.
- **No production logic in notebooks**: Business logic belongs in `src/lrs/`. Notebooks call `lrs.*` functions; they don't define them.
- **Label experiments**: Include a markdown cell at the top of each notebook with: date, hypothesis, and outcome.
