#!/usr/bin/env bash
# Run the Flask UI strictly inside the project .venv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing $VENV_PY"
  echo "Create the venv:"
  echo "  cd $ROOT"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -e \".[dev]\""
  exit 1
fi

exec "$VENV_PY" "$ROOT/web/app.py" "$@"
