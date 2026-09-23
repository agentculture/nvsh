"""Tests for scripts/needle-finetune/try_table.py's verdict() (finding 4, PR #47).

Uses importlib.util to load the script from disk so the hyphenated directory
name does not interfere with normal import (same pattern as
test_needle_finetune_dataset.py).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from nvsh.tiers.base import TierDecision

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "needle-finetune" / "try_table.py"
_spec = importlib.util.spec_from_file_location("needle_finetune_try_table", _PATH)
_try_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_try_table)

verdict = _try_table.verdict


def _decision(operation: str = "gpu_stats") -> TierDecision:
    return TierDecision(operation=operation, args={}, confidence=0.9, read_only=True)


# ---------------------------------------------------------------------------
# escalate entries (should-decline) -- unchanged behaviour
# ---------------------------------------------------------------------------


def test_escalate_entry_is_correct_when_tier1_declines():
    entry = {"expect": {"escalate": True}}
    assert verdict(entry, "not-a-tier-decision") is True


def test_escalate_entry_is_wrong_when_tier1_proposes():
    entry = {"expect": {"escalate": True}}
    assert verdict(entry, _decision()) is False


# ---------------------------------------------------------------------------
# explain entries (should-decline, added by this PR's corpus change) -- the
# fix under test: no KeyError, and a proposal is scored wrong, not a crash.
# ---------------------------------------------------------------------------


def test_explain_entry_is_correct_when_tier1_declines():
    entry = {"expect": {"explain": True, "answer": "some explanation"}}
    assert verdict(entry, "not-a-tier-decision") is True


def test_explain_entry_with_a_proposal_is_wrong_not_a_crash():
    entry = {"expect": {"explain": True, "answer": "some explanation"}}
    # Before the fix this raised KeyError("operation") instead of returning False.
    assert verdict(entry, _decision()) is False


# ---------------------------------------------------------------------------
# ordinary operation entries -- unchanged behaviour
# ---------------------------------------------------------------------------


def test_operation_entry_is_correct_on_matching_pick():
    entry = {"expect": {"operation": "gpu_stats"}}
    assert verdict(entry, _decision("gpu_stats")) is True


def test_operation_entry_is_wrong_on_mismatched_pick():
    entry = {"expect": {"operation": "gpu_stats"}}
    assert verdict(entry, _decision("thermal_stats")) is False


def test_operation_entry_is_wrong_when_tier1_declines():
    entry = {"expect": {"operation": "gpu_stats"}}
    assert verdict(entry, "not-a-tier-decision") is False
