"""Tests for ``evals/tool_jev/deepeval_layer.py`` (task t15, issue #64).

Fixtures below are synthetic (hand-built ``Trace`` objects over the real,
public operation names ``memory_stats``/``disk_stats``/``service_logs``/
``service_restart``/``container_restart`` from ``nvsh/ops/table.py`` — not
corpus or held-out data). Every "exact metric" assertion below is checked
against ``evals.tool_jev.metrics_bridge`` / the loaded ``metrics.py``
directly, so a metric's pass/fail is proven to equal the bridge's own
row-level verdict for the same case (acceptance criterion 2), not just
asserted by inspection.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evals.tool_jev import deepeval_layer, metrics_bridge
from evals.tool_jev.trace import RawRecord, Trace

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def metrics_mod():
    return metrics_bridge.load_metrics_module()


@pytest.fixture(scope="module")
def gate_mod():
    return metrics_bridge.load_gate_module()


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------


def _trace(
    case_id: str,
    expected: dict,
    outcome: str,
    *,
    operation: str | None = None,
    arguments: dict | None = None,
    candidates: dict | None = None,
    split: str = "test",
    subject: str = "fake-subject",
    decision: str | None = None,
    reason: str = "policy applied",
) -> Trace:
    raw = RawRecord(
        outcome=outcome,
        operation=operation,
        arguments=arguments,
        candidates=candidates,
        tokens=5,
        ttfd_ms=10.0,
        latency_ms=12.0,
    )
    trace = Trace(case_id=case_id, split=split, raw=raw, ground_truth=expected, subject=subject)
    return trace.with_policy("raw", decision if decision is not None else outcome, reason)


# ---------------------------------------------------------------------------
# build_test_case / prediction_from_trace
# ---------------------------------------------------------------------------


def test_prediction_from_trace_round_trips_raw_fields(metrics_mod):
    trace = _trace(
        "c1",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.9, "disk_stats": 0.1},
    )
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    assert prediction.id == "c1"
    assert prediction.outcome == "propose"
    assert prediction.operation == "memory_stats"
    assert prediction.candidates == {"memory_stats": 0.9, "disk_stats": 0.1}


def test_prediction_from_trace_requires_ground_truth(metrics_mod):
    raw = RawRecord(outcome="propose", operation="memory_stats", arguments={}, candidates=None)
    trace = Trace(case_id="c-no-gt", split="test", raw=raw, ground_truth=None)
    with pytest.raises(deepeval_layer.DeepevalLayerError):
        deepeval_layer.prediction_from_trace(trace, metrics_mod)


def test_build_test_case_never_carries_case_text(metrics_mod):
    trace = _trace(
        "c2",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    assert test_case.input == "case:c2"
    assert "text" not in (test_case.metadata or {})
    assert test_case.metadata["case_id"] == "c2"
    assert test_case.metadata["policy"] == "raw"
    assert test_case.actual_output == "propose"


def test_build_test_case_requires_a_final_decision_for_the_policy(metrics_mod):
    trace = _trace(
        "c3",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
    )
    with pytest.raises(deepeval_layer.DeepevalLayerError):
        deepeval_layer.build_test_case(trace, "some-other-policy", metrics_mod=metrics_mod)


# ---------------------------------------------------------------------------
# RightActionMetric
# ---------------------------------------------------------------------------


def test_right_action_matches_bridge_top1_correct_with_distribution(metrics_mod):
    expected = {"operation": "memory_stats", "args": {}}
    trace = _trace(
        "c-right-1",
        expected,
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.7, "disk_stats": 0.3},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    expected_verdict = metrics_bridge.top1_correct(metrics_mod, prediction)

    metric = deepeval_layer.RightActionMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is expected_verdict is True
    assert metric.__name__ == "right_action"


def test_right_action_matches_bridge_top1_correct_when_wrong(metrics_mod):
    expected = {"operation": "memory_stats", "args": {}}
    trace = _trace(
        "c-right-2",
        expected,
        "propose",
        operation="disk_stats",
        arguments={},
        candidates={"memory_stats": 0.2, "disk_stats": 0.8},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    expected_verdict = metrics_bridge.top1_correct(metrics_mod, prediction)

    metric = deepeval_layer.RightActionMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is expected_verdict is False


def test_right_action_falls_back_to_right_proposal_without_a_distribution(metrics_mod):
    expected = {"operation": "memory_stats", "args": {}}
    trace = _trace(
        "c-right-3",
        expected,
        "propose",
        operation="memory_stats",
        arguments={},
        candidates=None,
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    expected_verdict = metrics_mod._right_proposal(prediction)

    metric = deepeval_layer.RightActionMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is expected_verdict is True


def test_right_action_not_applicable_to_an_escalate_expected_case(metrics_mod):
    expected = {"escalate": True}
    trace = _trace("c-right-4", expected, "escalate")
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)

    metric = deepeval_layer.RightActionMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True
    assert "not applicable" in metric.reason


# ---------------------------------------------------------------------------
# WrongMutatingMetric
# ---------------------------------------------------------------------------


def test_wrong_mutating_matches_bridge_row_metrics(metrics_mod):
    expected = {"operation": "service_restart", "args": {"service": "rsyslog"}}
    trace = _trace(
        "c-wm-1",
        expected,
        "propose",
        operation="container_restart",
        arguments={"container": "web"},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    row = metrics_bridge.row_metrics(metrics_mod, metrics_bridge.load_gate_module(), prediction)
    assert row["wrong_mutating"] is True

    metric = deepeval_layer.WrongMutatingMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is False
    assert metric.__name__ == "wrong_mutating"


def test_wrong_mutating_passes_on_a_right_mutating_call(metrics_mod):
    expected = {"operation": "service_restart", "args": {"service": "rsyslog"}}
    trace = _trace(
        "c-wm-2",
        expected,
        "propose",
        operation="service_restart",
        arguments={"service": "rsyslog"},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    row = metrics_bridge.row_metrics(metrics_mod, metrics_bridge.load_gate_module(), prediction)
    assert row["wrong_mutating"] is False

    metric = deepeval_layer.WrongMutatingMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True


# ---------------------------------------------------------------------------
# CorrectAbstainEscalateMetric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "final_decision, expected_success",
    [("escalate", True), ("abstain_uncertain", True), ("propose", False), ("explain", False)],
)
def test_correct_abstain_escalate_on_an_escalate_expected_case(
    metrics_mod, final_decision, expected_success
):
    expected = {"escalate": True}
    trace = _trace("c-esc-1", expected, "escalate", decision=final_decision)
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)

    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    # Same vocabulary metrics.py's own _escalated() uses -- not re-derived.
    fake = type("Fake", (), {"outcome": final_decision})()
    assert metric.success is metrics_mod._escalated(fake) is expected_success


@pytest.mark.parametrize(
    "final_decision, expected_success",
    [("explain", True), ("escalate", False), ("propose", False)],
)
def test_correct_abstain_escalate_on_an_explain_expected_case(
    metrics_mod, final_decision, expected_success
):
    expected = {"explain": True}
    trace = _trace("c-exp-1", expected, "explain", decision=final_decision)
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)

    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is expected_success


def test_correct_abstain_escalate_not_applicable_to_an_operation_expected_case(metrics_mod):
    expected = {"operation": "memory_stats", "args": {}}
    trace = _trace(
        "c-op-1", expected, "propose", operation="memory_stats", arguments={}, decision="propose"
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)

    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True
    assert "not applicable" in metric.reason


# ---------------------------------------------------------------------------
# MissingCandidateHandledMetric
# ---------------------------------------------------------------------------


def test_missing_candidate_handled_matches_bridge_when_escalated(metrics_mod):
    expected = {"operation": "service_logs", "args": {}}
    # gold ("service_logs") is not among the offered candidates.
    trace = _trace(
        "c-mc-1-nocand",
        expected,
        "escalate",
        candidates={"memory_stats": 0.5, "disk_stats": 0.5},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    assert metrics_mod.is_missing_candidate(prediction) is True

    metric = deepeval_layer.MissingCandidateHandledMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True
    assert metric.__name__ == "missing_candidate_handled"


def test_missing_candidate_handled_matches_bridge_when_not_escalated(metrics_mod):
    expected = {"operation": "service_logs", "args": {}}
    trace = _trace(
        "c-mc-2-nocand",
        expected,
        "propose",
        operation="disk_stats",
        arguments={},
        candidates={"memory_stats": 0.5, "disk_stats": 0.5},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    assert metrics_mod.is_missing_candidate(prediction) is True

    metric = deepeval_layer.MissingCandidateHandledMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is False


def test_missing_candidate_handled_not_applicable_when_gold_offered(metrics_mod):
    expected = {"operation": "memory_stats", "args": {}}
    trace = _trace(
        "c-mc-3",
        expected,
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.9, "disk_stats": 0.1},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    assert metrics_mod.is_missing_candidate(prediction) is False

    metric = deepeval_layer.MissingCandidateHandledMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True
    assert "not applicable" in metric.reason


# ---------------------------------------------------------------------------
# evaluate_traces(): full DeepEval run, no network, results scoped to tmp_path
# ---------------------------------------------------------------------------


def _block_sockets(monkeypatch):
    def _refuse(*args, **kwargs):
        raise AssertionError("evaluate_traces() must never touch the network")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


@pytest.fixture
def traces():
    return [
        _trace(
            "c-int-right",
            {"operation": "memory_stats", "args": {}},
            "propose",
            operation="memory_stats",
            arguments={},
            candidates={"memory_stats": 0.9, "disk_stats": 0.1},
            decision="propose",
        ),
        _trace(
            "c-int-wrongmut",
            {"operation": "service_restart", "args": {"service": "rsyslog"}},
            "propose",
            operation="container_restart",
            arguments={"container": "web"},
            decision="propose",
        ),
        _trace(
            "c-int-escalate",
            {"escalate": True},
            "escalate",
            decision="escalate",
        ),
        _trace(
            "c-int-explain-miss",
            {"explain": True},
            "propose",
            operation="memory_stats",
            arguments={},
            decision="propose",
        ),
        _trace(
            "c-int-nocand",
            {"operation": "service_logs", "args": {}},
            "escalate",
            candidates={"memory_stats": 0.5, "disk_stats": 0.5},
            decision="escalate",
        ),
    ]


def test_evaluate_traces_runs_offline_and_writes_results_only_under_results_folder(
    monkeypatch, tmp_path, traces, metrics_mod, gate_mod
):
    monkeypatch.chdir(tmp_path)
    _block_sockets(monkeypatch)
    results_folder = tmp_path / "results"

    outcome = deepeval_layer.evaluate_traces(
        traces,
        "raw",
        results_folder=results_folder,
        metrics_mod=metrics_mod,
        gate_mod=gate_mod,
    )

    # -- criterion 1: a test_run_*.json landed in results_folder, and no
    # .deepeval cache/keystore leaked into this repository (cwd is tmp_path
    # for the duration of the call, via monkeypatch.chdir above).
    run_files = list(results_folder.glob("test_run_*.json"))
    assert run_files, f"no test_run_*.json written to {results_folder}"
    with open(run_files[0], encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload, "test_run_*.json is empty"
    assert not (REPO_ROOT / ".deepeval").exists(), (
        "evaluate_traces() must never create .deepeval/ inside the repo "
        "(DEEPEVAL_CACHE_FOLDER/HIDDEN_DIR resolves relative to cwd; the "
        "test chdirs into tmp_path so this checks the real repo stayed clean)"
    )

    # -- criterion 2: every exact metric's pass/fail equals the
    # metrics_bridge verdict for the same case.
    predictions_by_id = {
        trace.case_id: deepeval_layer.prediction_from_trace(trace, metrics_mod) for trace in traces
    }
    by_name = {
        result.metadata["case_id"]: result for result in outcome.deepeval_result.test_results
    }
    assert set(by_name) == {trace.case_id for trace in traces}

    for trace in traces:
        result = by_name[trace.case_id]
        prediction = predictions_by_id[trace.case_id]
        row = metrics_bridge.row_metrics(metrics_mod, gate_mod, prediction)
        metrics_by_name = {m.name: m for m in result.metrics_data}

        expected_kind = metrics_mod.expect_kind(prediction.expected)

        if expected_kind == "operation":
            expected_right_action = metrics_bridge.top1_correct(metrics_mod, prediction)
            if expected_right_action is None:
                expected_right_action = metrics_mod._right_proposal(prediction)
            assert metrics_by_name["right_action"].success is expected_right_action
            assert metrics_by_name["correct_abstain_escalate"].success is True
        else:
            assert metrics_by_name["right_action"].success is True

        assert metrics_by_name["wrong_mutating"].success is (not row["wrong_mutating"])

        if row["missing_candidate"]:
            assert metrics_by_name["missing_candidate_handled"].success is metrics_mod._escalated(
                prediction
            )
        else:
            assert metrics_by_name["missing_candidate_handled"].success is True

    # -- corpus metrics come from metrics_bridge.compute(), not DeepEval.
    direct_corpus = metrics_bridge.compute(
        list(predictions_by_id.values()), metrics_mod=metrics_mod, gate_mod=gate_mod
    )
    assert outcome.corpus_metrics == direct_corpus


def test_evaluate_traces_corpus_metrics_are_not_recomputed_by_deepeval(
    monkeypatch, tmp_path, traces, metrics_mod, gate_mod
):
    monkeypatch.chdir(tmp_path)
    _block_sockets(monkeypatch)
    outcome = deepeval_layer.evaluate_traces(
        traces,
        "raw",
        results_folder=tmp_path / "results2",
        metrics_mod=metrics_mod,
        gate_mod=gate_mod,
    )
    # metrics_bridge's own aggregate keys must be present, verbatim.
    assert "right_proposals" in outcome.corpus_metrics["metrics_compute"]
    assert "abstain" in outcome.corpus_metrics
    assert "wrong_mutating" in outcome.corpus_metrics
    assert "slices" in outcome.corpus_metrics
