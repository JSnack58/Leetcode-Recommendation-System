"""Tests for NeetCode 150 cold-start path."""

from pathlib import Path

from lrs.web.neetcode150 import load_neetcode150, neetcode_cards


def test_load_neetcode150():
    root = Path(__file__).resolve().parents[2]
    problems = load_neetcode150(str(root / "neetcode150.json"))
    assert len(problems) == 150
    assert problems[0]["slug"] == "contains-duplicate"


def test_neetcode_cards_count():
    root = Path(__file__).resolve().parents[2]
    cards = neetcode_cards(5, str(root / "neetcode150.json"))
    assert len(cards) == 5
    assert cards[0]["signals"]
    assert "NeetCode 150" in cards[0]["signals"][0]
