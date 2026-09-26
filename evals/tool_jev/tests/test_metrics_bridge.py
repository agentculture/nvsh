"""Parity tests for ``evals/tool_jev/metrics_bridge.py`` (task t7, issue #64).

The bridge never reimplements a formula that already exists in
``scripts/lfm-finetune/metrics.py``/``gate.py``; every test below loads
those two modules by path exactly as ``tests/test_lfm_finetune_metrics.py``
and ``tests/test_lfm_finetune_gate.py`` do, and checks that the bridge's
figures equal what those modules compute directly on the same fixture file.
Real operation names below (``memory_stats``, ``service_status``,
``service_logs``, ``service_restart``, ``container_restart``) are the same
ones ``tests/test_lfm_finetune_metrics.py`` already uses as fixture data;
this is not held-out or corpus data.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from evals.tool_jev import metrics_bridge as bridge

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METRICS_PATH = _REPO_ROOT / "scripts" / "lfm-finetune" / "metrics.py"
_GATE_PATH = _REPO_ROOT / "scripts" / "lfm-finetune" / "gate.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def metrics():
    """A direct load of metrics.py, independent of the bridge's own copy."""
    return _load("_test_metrics_bridge_direct_lfm_metrics", _METRICS_PATH)


@pytest.fixture(scope="module")
def gate():
    """A direct load of gate.py, independent of the bridge's own copy."""
    return _load("_test_metrics_bridge_direct_lfm_gate", _GATE_PATH)


ESC = "(escalate)"
EXP = "(explain)"


def _op(name: str, **args) -> dict:
    return {"operation": name, "args": dict(args)}


def _rec(
    entry_id: str,
    expected: dict,
    outcome: str,
    operation: str | None = None,
    arguments: dict | None = None,
    candidates: dict | None = None,
    **extra,
) -> dict:
    return {
        "id": entry_id,
        "expected": expected,
        "outcome": outcome,
        "operation": operation,
        "arguments": arguments,
        "candidates": candidates,
        "tokens": 5,
        "ttfd_ms": 40.0,
        "latency_ms": 50.0,
        **extra,
    }


def _fixture_rows() -> list[dict]:
    """Six lines: right pick, wrong read-only pick, wrong mutating pick,
    a correct escalate, a missing-candidate line, and a line with no
    candidate distribution at all."""
    return [
        # right: proposes exactly the expected read-only operation.
        _rec(
            "right",
            _op("memory_stats"),
            "propose",
            "memory_stats",
            {},
            {"memory_stats": 0.9, ESC: 0.1},
        ),
        # wrong read-only pick: a different read-only operation proposed.
        _rec(
            "wrong-read-only",
            _op("service_status", service="docker"),
            "propose",
            "service_logs",
            {"service": "docker"},
            {"service_logs": 0.7, "service_status": 0.3},
        ),
        # wrong mutating pick: a different mutating operation proposed.
        _rec(
            "wrong-mutating",
            _op("service_restart", service="docker"),
            "propose",
            "container_restart",
            {"container": "docker"},
            {"container_restart": 0.6, "service_restart": 0.4},
        ),
        # escalate expected, correctly escalated.
        _rec("escalate-right", {"escalate": True}, "escalate", candidates={ESC: 1.0}),
        # missing-candidate: id ends -nocand, gold (escalate) never offered.
        _rec(
            "missing-nocand",
            {"escalate": True},
            "escalate",
            candidates={"memory_stats": 0.5, "service_status": 0.5},
        ),
        # no distribution at all: an unparseable output.
        _rec(
            "no-distribution",
            _op("memory_stats"),
            "invalid",
            candidates=None,
            invalid_reason="malformed",
        ),
    ]


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _fixture_rows()), encoding="utf-8")
    return path


@pytest.fixture
def predictions(metrics, tmp_path):
    return metrics.read_predictions(_write(tmp_path))


@pytest.fixture
def result(predictions, metrics, gate):
    return bridge.compute(predictions, metrics_mod=metrics, gate_mod=gate)


def _by_id(predictions, entry_id):
    return next(p for p in predictions if p.id == entry_id)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_metrics_module_returns_the_real_module():
    module = bridge.load_metrics_module()
    assert hasattr(module, "compute")
    assert hasattr(module, "brier_one")


def test_load_gate_module_returns_the_real_module():
    module = bridge.load_gate_module()
    assert hasattr(module, "decide")
    assert hasattr(module, "normalized_entropy")


# ---------------------------------------------------------------------------
# Acceptance criterion 1: parity against metrics.compute / gate helpers.
# ---------------------------------------------------------------------------


def test_metrics_compute_passthrough_is_exactly_metrics_compute(result, predictions, metrics):
    assert result["metrics_compute"] == metrics.compute(predictions)


def test_abstain_alias_matches_metrics_compute_abstention(result):
    assert result["abstain"] == result["metrics_compute"]["abstention"]


def test_wrong_mutating_alias_matches_metrics_compute(result):
    assert result["wrong_mutating"] == result["metrics_compute"]["wrong_mutating"]
    # One wrong-mutating pick in the fixture (a different mutating operation).
    assert result["wrong_mutating"]["total"] == 1


def test_slices_alias_matches_metrics_compute(result):
    assert result["slices"] == result["metrics_compute"]["slices"]


def test_missing_candidate_summary_matches_is_missing_candidate(result, predictions, metrics):
    expected_n = sum(1 for p in predictions if metrics.is_missing_candidate(p))
    assert result["missing_candidate"]["n"] == expected_n
    assert result["missing_candidate"]["N"] == len(predictions)
    # The -nocand row is the only one caught by the id-suffix rule.
    assert expected_n == 1


@pytest.mark.parametrize(
    "entry_id, expect",
    [
        ("right", True),
        ("wrong-read-only", False),
        ("wrong-mutating", False),
        ("escalate-right", True),
    ],
)
def test_top1_correct_matches_top_candidate_against_gold(
    predictions, metrics, gate, entry_id, expect
):
    prediction = _by_id(predictions, entry_id)
    row = bridge.row_metrics(metrics, gate, prediction)
    rolled = metrics.rollup_escalate_candidates(prediction.candidates)
    label, _ = metrics.top_candidate(rolled)
    gold = metrics.expected_label(prediction.expected)
    assert row["top1_correct"] == (label == gold) == expect


def test_top1_correct_is_not_measurable_with_no_distribution(predictions, metrics, gate):
    prediction = _by_id(predictions, "no-distribution")
    row = bridge.row_metrics(metrics, gate, prediction)
    assert row["top1_correct"] == bridge.NOT_MEASURABLE


def test_brier_matches_brier_one_on_the_rolled_distribution(predictions, metrics, gate):
    prediction = _by_id(predictions, "wrong-mutating")
    rolled = metrics.rollup_escalate_candidates(prediction.candidates)
    gold = metrics.expected_label(prediction.expected)
    expected = metrics.brier_one(rolled, gold)
    assert bridge.brier(metrics, prediction) == expected


def test_entropy_matches_gate_normalized_entropy(predictions, metrics, gate):
    prediction = _by_id(predictions, "wrong-read-only")
    rolled = metrics.rollup_escalate_candidates(prediction.candidates)
    n_offered = len({metrics.canonical_label(o) for o in prediction.candidates})
    expected = gate.normalized_entropy(rolled, n_offered)
    assert bridge.entropy(metrics, gate, prediction) == expected


def test_margin_matches_gate_argmax_and_second_place(predictions, metrics, gate):
    prediction = _by_id(predictions, "right")
    rolled = metrics.rollup_escalate_candidates(prediction.candidates)
    offered = list(prediction.candidates.keys())
    top1_label, p_top1 = gate._argmax(rolled, offered)
    expected = p_top1 - gate._second_place(rolled, top1_label)
    assert bridge.margin(metrics, gate, prediction) == expected


def test_log_loss_matches_negative_log_of_gold_probability(predictions, metrics, gate):
    prediction = _by_id(predictions, "wrong-read-only")
    rolled = metrics.rollup_escalate_candidates(prediction.candidates)
    gold = metrics.expected_label(prediction.expected)
    expected = -math.log(rolled.get(gold, 0.0) or bridge._LOG_LOSS_EPSILON)
    assert bridge.log_loss_one(metrics, prediction) == pytest.approx(expected)


def test_top_k_accuracy_matches_manual_ranking(predictions, metrics, gate):
    # Operation-expected rows with a distribution: right, wrong-read-only,
    # wrong-mutating. Gold is in the top-1 for "right" only, but every one
    # of them has gold within its top-2 (each fixture candidate dict has
    # exactly two entries).
    result = bridge.top_k_accuracy(metrics, predictions, top_k=(1, 2))
    assert result[1]["N"] == 3
    assert result[1]["n"] == 1
    assert result[2]["n"] == 3
    assert result[1]["rate"] == pytest.approx(1 / 3)
    assert result[2]["rate"] == pytest.approx(1.0)


def test_topk_correct_agrees_with_manual_sort_for_k_equals_one(predictions, metrics):
    for prediction in predictions:
        if prediction.candidates is None:
            assert bridge.topk_correct(metrics, prediction, 1) is None
            continue
        rolled = metrics.rollup_escalate_candidates(prediction.candidates)
        ranked = sorted(rolled.items(), key=lambda item: (-item[1], item[0]))
        gold = metrics.expected_label(prediction.expected)
        expected = gold in [label for label, _ in ranked[:1]]
        assert bridge.topk_correct(metrics, prediction, 1) == expected


# ---------------------------------------------------------------------------
# Acceptance criterion 2: no distribution -> not_measurable, never a number.
# ---------------------------------------------------------------------------


def test_no_distribution_row_reports_not_measurable_everywhere(predictions, metrics, gate):
    prediction = _by_id(predictions, "no-distribution")
    row = bridge.row_metrics(metrics, gate, prediction)
    assert row["has_distribution"] is False
    assert row["top1_correct"] == bridge.NOT_MEASURABLE
    assert row["brier"] == bridge.NOT_MEASURABLE
    assert row["log_loss"] == bridge.NOT_MEASURABLE
    assert row["entropy"] == bridge.NOT_MEASURABLE
    assert row["margin"] == bridge.NOT_MEASURABLE
    for value in row["topk_correct"].values():
        assert value == bridge.NOT_MEASURABLE
    # missing_candidate/wrong_mutating/slice are always defined regardless.
    assert row["missing_candidate"] is False
    assert row["wrong_mutating"] is False
    assert row["slice"] == "read_only"


def test_no_distribution_functions_never_return_a_number(predictions, metrics, gate):
    prediction = _by_id(predictions, "no-distribution")
    for value in (
        bridge.brier(metrics, prediction),
        bridge.log_loss_one(metrics, prediction),
        bridge.entropy(metrics, gate, prediction),
        bridge.margin(metrics, gate, prediction),
    ):
        assert value == bridge.NOT_MEASURABLE
        assert not isinstance(value, (int, float))


def test_no_distribution_row_excluded_from_top_k_accuracy_denominator(predictions, metrics):
    result = bridge.top_k_accuracy(metrics, predictions, top_k=(1,))
    # 3 operation-expected rows total, but only 3 have distributions in this
    # fixture minus the no-distribution one which is also operation-expected.
    assert result[1]["N"] == 3  # right, wrong-read-only, wrong-mutating (has distributions)


def test_mean_log_loss_excludes_no_distribution_rows(predictions, metrics):
    losses = bridge.mean_log_loss(metrics, predictions)
    assert losses["without_distribution"] == 1
    assert losses["n"] == len(predictions) - 1


# ---------------------------------------------------------------------------
# Full compute() shape
# ---------------------------------------------------------------------------


def test_compute_rows_cover_every_prediction_by_id(result, predictions):
    assert [row["id"] for row in result["rows"]] == [p.id for p in predictions]


def test_compute_loads_its_own_modules_when_none_given(predictions):
    result = bridge.compute(predictions)
    assert result["metrics_compute"]["outcome_counts"]["escalate"] == 2
    assert len(result["rows"]) == len(predictions)
