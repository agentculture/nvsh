"""Tests for scripts/lfm-finetune/requirements-train.txt."""

from __future__ import annotations

import re
from pathlib import Path
from unittest import TestCase


def _lines() -> list[str]:
    """Return non-blank, non-comment lines from requirements-train.txt."""
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "lfm-finetune" / "requirements-train.txt"
    return [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _normalise(name: str) -> str:
    """Lowercase, replace underscores with hyphens."""
    return name.lower().replace("_", "-")


class TestRequirementsTrain(TestCase):
    """Each bullet from the spec is its own test method."""

    def test_pinned_format(self) -> None:
        """Every non-empty, non-comment line matches name==version."""
        pattern = re.compile(r"^[A-Za-z0-9._-]+==[A-Za-z0-9.+_-]+$")
        for ln in _lines():
            self.assertRegex(ln, pattern, f"unpinned line: {ln!r}")

    def test_required_packages(self) -> None:
        """The set of package names includes all required ones."""
        required = {
            "torch",
            "transformers",
            "peft",
            "trl",
            "unsloth",
            "accelerate",
            "huggingface-hub",
            "tokenizers",
        }
        present = {_normalise(ln.split("==")[0]) for ln in _lines()}
        missing = required - present
        self.assertFalse(missing, f"missing packages: {missing}")

    def test_no_duplicates(self) -> None:
        """No package name appears twice."""
        names = [_normalise(ln.split("==")[0]) for ln in _lines()]
        seen: set[str] = set()
        for n in names:
            self.assertNotIn(n, seen, f"duplicate package: {n}")
            seen.add(n)
