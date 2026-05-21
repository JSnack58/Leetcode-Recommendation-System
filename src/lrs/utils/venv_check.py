"""Ensure commands run inside the project virtual environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def venv_python() -> Path:
    return repo_root() / ".venv" / "bin" / "python"


def in_project_venv() -> bool:
    """True if the current interpreter is the repo .venv or VIRTUAL_ENV points there."""
    expected = venv_python()
    if expected.exists() and Path(sys.executable).resolve() == expected.resolve():
        return True
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and Path(venv).resolve() == (repo_root() / ".venv").resolve():
        return True
    return False


def require_venv(command_hint: str | None = None) -> None:
    """Exit with instructions if not running in .venv."""
    if in_project_venv():
        return
    root = repo_root()
    py = venv_python()
    hint = command_hint or "your command"
    if not py.parent.exists():
        msg = (
            f"No project venv at {root / '.venv'}.\n"
            f"Create it:\n"
            f"  cd {root}\n"
            f"  python3 -m venv .venv\n"
            f"  .venv/bin/pip install -e \".[dev]\"\n"
            f"Then run:\n"
            f"  .venv/bin/python {hint}"
        )
    else:
        msg = (
            f"Run this command with the project venv, not system Python:\n"
            f"  .venv/bin/python {hint}\n"
            f"Or activate first:\n"
            f"  source .venv/bin/activate"
        )
    print(msg, file=sys.stderr)
    sys.exit(1)
