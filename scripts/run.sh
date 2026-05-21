#!/usr/bin/env bash
# Run any project script with .venv/bin/python
# Usage: ./scripts/run.sh scripts/build_dataset.py [--args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing $VENV_PY — run: python3 -m venv .venv && .venv/bin/pip install -e \".[dev]\""
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <script.py> [args...]"
  exit 1
fi

SCRIPT="$1"
shift
if [[ "$SCRIPT" != /* ]]; then
  SCRIPT="$ROOT/$SCRIPT"
fi

exec "$VENV_PY" "$SCRIPT" "$@"
