# Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=lrs --cov-report=term-missing

# Run only unit tests (fast)
uv run pytest tests/unit/ -v
```

## Structure

| Directory | Purpose |
|---|---|
| `unit/` | Fast, isolated tests. No real data files required. Use fixtures from `conftest.py`. |
| `integration/` | End-to-end pipeline tests. May require processed data to be present. |

## Coverage Targets

- `src/lrs/recommendation/` — 90%+ (tier logic is a business rule; test thoroughly)
- `src/lrs/evaluation/metrics.py` — 90%+ (metrics must be numerically correct)
- `src/lrs/models/` — 70%+ (stubs will raise NotImplementedError until implemented)
