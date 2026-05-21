#!/usr/bin/env python3
"""Simple Flask UI for LeetCode problem recommendations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lrs.utils.venv_check import require_venv

require_venv("web/app.py")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from flask import Flask, render_template, request

from lrs.web.service import get_store

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)


@app.route("/", methods=["GET", "POST"])
def index():
    slug = ""
    count = 10
    result = None

    if request.method == "POST":
        slug = (request.form.get("user_slug") or "").strip()
        try:
            count = max(1, min(30, int(request.form.get("count") or 10)))
        except ValueError:
            count = 10

        if slug:
            result = get_store().recommend(slug, count=count)

    return render_template(
        "index.html",
        slug=slug,
        count=count,
        result=result,
    )


if __name__ == "__main__":
    import os

    port = int(os.getenv("FLASK_PORT", "5001"))
    app.run(debug=True, host="127.0.0.1", port=port)
