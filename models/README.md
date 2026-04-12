# Model Artifacts

Serialized model artifacts are **gitignored**. Do not commit large binary files here.

## Versioning

Artifacts are versioned externally. Ask a maintainer for access, then pull:

```bash
# If using DVC:
dvc pull

# If using manual S3/GCS download:
# aws s3 sync s3://<bucket>/lrs-models/ models/
```

## Directory Structure

```
models/
├── baseline/
│   ├── svd_v1.pkl
│   └── als_v1.pkl
└── advanced/
    ├── ncf_v1.pt
    └── lightgcn_v1.pt
```

## Naming Convention

`{model_name}_v{version}.{ext}` — bump version when retraining with different hyperparameters or data.
