"""Tests for ``evals/tool_jev/deepeval_layer.py`` (task t15, issue #64).

Fixtures below are synthetic (hand-built ``Trace`` objects over the real,
public operation names ``memory_stats``/``disk_stats``/``service_logs``/
``service_restart``/``container_restart`` from ``nvsh/ops/table.py`` — not
corpus or held-out data), replayed through this package's own real,
committed policy files (``evals/tool_jev/policies/*.json``) -- never a
policy this test invents its own gate/calibration math for.

Every "exact metric" assertion below is checked against the module's own
``apply_policy_to_prediction`` plus ``evals.tool_jev.metrics_bridge`` /
the loaded ``metrics.py`` directly, so a metric's pass/fail is proven to
equal the bridge's row-level verdict on the *same policy-applied
prediction* (acceptance criterion 2), not just asserted by inspection.

Grading is now per (case, policy) -- issue 64's core distinction between
"the model improved" and "the harness prevented the model's mistake":
right_action / wrong_mutating / missing_candidate_handled are graded on
the prediction a named policy's calibration+gate actually produces, not on
the raw model output alone. See the three scenarios this coordinates:

(a) a wrong mutating raw pick that ``mutating-strict-example`` abstains on
    (its mutating ``ThresholdSet`` has a real floor/margin/entropy ceiling)
    fails ``wrong_mutating`` under ``"raw"`` and passes under
    ``"mutating-strict-example"``.
(b) a correct read-only propose that ``scorer-r3b-shipped`` abstains on
    (its read_only margin threshold) passes ``right_action`` under
    ``"raw"`` and fails it under ``"scorer-r3b-shipped"``.
(c) for policy ``"raw"`` (no calibration, no gate -- bare argmax), every
    metric's verdict equals what the old raw-graded behaviour would have
    given, for a fixture whose raw record is already internally
    consistent with its own candidates' argmax.

Note on ``scorer-r3b-shipped``: its own JSON docstring says it carries no
mutating-side threshold ("a mutating proposal is never abstained on by
this policy") -- so scenario (a)'s abstain-on-a-wrong-mutating-pick uses
``mutating-strict-example`` instead, which does carry a mutating
``ThresholdSet``; ``scorer-r3b-shipped`` is used for scenario (b), which
only needs its read_only margin.
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
# Trace builder
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
    return Trace(case_id=case_id, split=split, raw=raw, ground_truth=expected, subject=subject)


# ---------------------------------------------------------------------------
# prediction_from_trace / build_test_case
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


def test_build_test_case_raises_for_an_unknown_policy_name(metrics_mod):
    trace = _trace(
        "c3",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
    )
    with pytest.raises(FileNotFoundError):
        deepeval_layer.build_test_case(trace, "totally-bogus-policy", metrics_mod=metrics_mod)


def test_build_test_case_accepts_an_in_memory_policy_dict(metrics_mod):
    trace = _trace(
        "c4",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
    )
    policy = {"name": "inline-test-policy", "version": "1", "calibration": None, "gate": None}
    test_case = deepeval_layer.build_test_case(trace, policy, metrics_mod=metrics_mod)
    assert test_case.metadata["policy"] == "inline-test-policy"


# ---------------------------------------------------------------------------
# apply_policy_to_prediction
# ---------------------------------------------------------------------------


def test_apply_policy_to_prediction_passthrough_with_no_distribution(metrics_mod):
    trace = _trace(
        "c-nogate",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
        candidates=None,
    )
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    for policy in ("raw", "scorer-r3b-shipped", "mutating-strict-example"):
        applied = deepeval_layer.apply_policy_to_prediction(
            policy, prediction, metrics_mod=metrics_mod
        )
        assert applied is prediction, f"policy {policy!r} must not touch a row with no distribution"


def test_apply_policy_to_prediction_raw_reproduces_bare_argmax(metrics_mod):
    trace = _trace(
        "c-raw-argmax",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.7, "disk_stats": 0.3},
    )
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
    applied = deepeval_layer.apply_policy_to_prediction("raw", prediction, metrics_mod=metrics_mod)
    assert applied.outcome == "propose"
    assert applied.operation == "memory_stats"
    assert applied.arguments == {}
    assert applied.candidates == {"memory_stats": 0.7, "disk_stats": 0.3}


def test_apply_policy_to_prediction_gate_abstains_a_wrong_mutating_pick(metrics_mod):
    """Scenario (a): a wrong mutating pick that mutating-strict-example abstains on."""
    trace = _trace(
        "c-scenario-a",
        {"operation": "service_restart", "args": {"service": "rsyslog"}},
        "propose",
        operation="container_restart",
        arguments={"container": "web"},
        candidates={"container_restart": 0.55, "service_restart": 0.45},
    )
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)

    raw_applied = deepeval_layer.apply_policy_to_prediction(
        "raw", prediction, metrics_mod=metrics_mod
    )
    assert raw_applied.outcome == "propose"
    assert raw_applied.operation == "container_restart"

    gated_applied = deepeval_layer.apply_policy_to_prediction(
        "mutating-strict-example", prediction, metrics_mod=metrics_mod
    )
    assert gated_applied.outcome == "abstain_uncertain"
    assert gated_applied.operation is None
    assert gated_applied.arguments is None


def test_apply_policy_to_prediction_gate_abstains_a_correct_readonly_propose(metrics_mod):
    """Scenario (b): a correct read-only propose that scorer-r3b-shipped abstains on."""
    trace = _trace(
        "c-scenario-b",
        {"operation": "memory_stats", "args": {}},
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.55, "disk_stats": 0.45},
    )
    prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)

    raw_applied = deepeval_layer.apply_policy_to_prediction(
        "raw", prediction, metrics_mod=metrics_mod
    )
    assert raw_applied.outcome == "propose"
    assert raw_applied.operation == "memory_stats"

    gated_applied = deepeval_layer.apply_policy_to_prediction(
        "scorer-r3b-shipped", prediction, metrics_mod=metrics_mod
    )
    assert gated_applied.outcome == "abstain_uncertain"
    assert gated_applied.operation is None


# ---------------------------------------------------------------------------
# RightActionMetric
# ---------------------------------------------------------------------------


def test_right_action_matches_bridge_right_proposal_with_distribution(metrics_mod):
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
    policy_applied = deepeval_layer.apply_policy_to_prediction(
        "raw", prediction, metrics_mod=metrics_mod
    )
    expected_verdict = metrics_mod._right_proposal(policy_applied)

    metric = deepeval_layer.RightActionMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is expected_verdict is True
    assert metric.__name__ == "right_action"


def test_right_action_matches_bridge_right_proposal_when_wrong(metrics_mod):
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
    policy_applied = deepeval_layer.apply_policy_to_prediction(
        "raw", prediction, metrics_mod=metrics_mod
    )
    expected_verdict = metrics_mod._right_proposal(policy_applied)

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


def test_right_action_scenario_b_gated_policy_loses_credit_for_an_abstained_correct_pick(
    metrics_mod,
):
    """Scenario (b) at the metric level: raw passes, scorer-r3b-shipped fails."""
    expected = {"operation": "memory_stats", "args": {}}
    trace = _trace(
        "c-right-scenario-b",
        expected,
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.55, "disk_stats": 0.45},
    )
    metric = deepeval_layer.RightActionMetric(metrics_mod=metrics_mod)

    raw_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric.measure(raw_case)
    assert metric.success is True

    gated_case = deepeval_layer.build_test_case(
        trace, "scorer-r3b-shipped", metrics_mod=metrics_mod
    )
    metric.measure(gated_case)
    assert metric.success is False


# ---------------------------------------------------------------------------
# WrongMutatingMetric
# ---------------------------------------------------------------------------


def test_wrong_mutating_matches_bridge_row_metrics(metrics_mod, gate_mod):
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
    policy_applied = deepeval_layer.apply_policy_to_prediction(
        "raw", prediction, metrics_mod=metrics_mod
    )
    row = metrics_bridge.row_metrics(metrics_mod, gate_mod, policy_applied)
    assert row["wrong_mutating"] is True

    metric = deepeval_layer.WrongMutatingMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is False
    assert metric.__name__ == "wrong_mutating"


def test_wrong_mutating_passes_on_a_right_mutating_call(metrics_mod, gate_mod):
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
    policy_applied = deepeval_layer.apply_policy_to_prediction(
        "raw", prediction, metrics_mod=metrics_mod
    )
    row = metrics_bridge.row_metrics(metrics_mod, gate_mod, policy_applied)
    assert row["wrong_mutating"] is False

    metric = deepeval_layer.WrongMutatingMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True


def test_wrong_mutating_scenario_a_gated_policy_saves_a_wrong_mutating_pick(metrics_mod):
    """Scenario (a) at the metric level: raw fails, mutating-strict-example passes."""
    expected = {"operation": "service_restart", "args": {"service": "rsyslog"}}
    trace = _trace(
        "c-wm-scenario-a",
        expected,
        "propose",
        operation="container_restart",
        arguments={"container": "web"},
        candidates={"container_restart": 0.55, "service_restart": 0.45},
    )
    metric = deepeval_layer.WrongMutatingMetric(metrics_mod=metrics_mod)

    raw_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric.measure(raw_case)
    assert metric.success is False

    gated_case = deepeval_layer.build_test_case(
        trace, "mutating-strict-example", metrics_mod=metrics_mod
    )
    metric.measure(gated_case)
    assert metric.success is True


# ---------------------------------------------------------------------------
# CorrectAbstainEscalateMetric
# ---------------------------------------------------------------------------


def test_correct_abstain_escalate_passes_when_raw_argmax_is_escalate(metrics_mod):
    expected = {"escalate": True}
    trace = _trace(
        "c-esc-correct",
        expected,
        "escalate",
        candidates={"(escalate)": 0.9, "memory_stats": 0.1},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True
    assert metric.__name__ == "correct_abstain_escalate"


def test_correct_abstain_escalate_fails_when_raw_argmax_proposes(metrics_mod):
    expected = {"escalate": True}
    trace = _trace(
        "c-esc-wrong-propose",
        expected,
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.9, "(escalate)": 0.1},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is False


def test_correct_abstain_escalate_fails_when_raw_argmax_explains(metrics_mod):
    expected = {"escalate": True}
    trace = _trace(
        "c-esc-wrong-explain",
        expected,
        "explain",
        candidates={"(explain)": 0.9, "memory_stats": 0.1},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is False


def test_correct_abstain_escalate_passes_on_a_gated_abstain_uncertain(metrics_mod):
    """abstain_uncertain also counts as correctly declining an escalate-expected case."""
    expected = {"escalate": True}
    trace = _trace(
        "c-esc-abstain-uncertain",
        expected,
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.55, "disk_stats": 0.45},
    )
    test_case = deepeval_layer.build_test_case(trace, "scorer-r3b-shipped", metrics_mod=metrics_mod)
    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True


def test_correct_abstain_escalate_passes_when_raw_argmax_explains_an_explain_case(metrics_mod):
    expected = {"explain": True}
    trace = _trace(
        "c-exp-correct",
        expected,
        "explain",
        candidates={"(explain)": 0.9, "memory_stats": 0.1},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is True


def test_correct_abstain_escalate_fails_when_raw_argmax_proposes_on_an_explain_case(metrics_mod):
    expected = {"explain": True}
    trace = _trace(
        "c-exp-wrong-propose",
        expected,
        "propose",
        operation="memory_stats",
        arguments={},
        candidates={"memory_stats": 0.9, "(explain)": 0.1},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is False


def test_correct_abstain_escalate_fails_when_raw_argmax_escalates_on_an_explain_case(metrics_mod):
    expected = {"explain": True}
    trace = _trace(
        "c-exp-wrong-escalate",
        expected,
        "escalate",
        candidates={"(escalate)": 0.9, "memory_stats": 0.1},
    )
    test_case = deepeval_layer.build_test_case(trace, "raw", metrics_mod=metrics_mod)
    metric = deepeval_layer.CorrectAbstainEscalateMetric(metrics_mod=metrics_mod)
    metric.measure(test_case)
    assert metric.success is False


def test_correct_abstain_escalate_not_applicable_to_an_operation_expected_case(metrics_mod):
    expected = {"operation": "memory_stats", "args": {}}
    trace = _trace("c-op-1", expected, "propose", operation="memory_stats", arguments={})
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
    # gold ("service_logs") is not among the offered candidates, and the
    # raw argmax is the escalate label -- consistent with a raw record a
    # real Track B interface would actually produce.
    trace = _trace(
        "c-mc-1-nocand",
        expected,
        "escalate",
        candidates={"(escalate)": 0.9, "memory_stats": 0.1},
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
        operation="memory_stats",
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
# evaluate_traces(): full DeepEval run, no network, .deepeval kept out of
# the repo without the test itself having to chdir.
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
        ),
        _trace(
            "c-int-wrongmut",
            {"operation": "service_restart", "args": {"service": "rsyslog"}},
            "propose",
            operation="container_restart",
            arguments={"container": "web"},
            candidates={"container_restart": 0.55, "service_restart": 0.45},
        ),
        _trace(
            "c-int-escalate",
            {"escalate": True},
            "escalate",
            candidates={"(escalate)": 0.9, "memory_stats": 0.1},
        ),
        _trace(
            "c-int-explain-miss",
            {"explain": True},
            "propose",
            operation="memory_stats",
            arguments={},
            candidates={"memory_stats": 0.9, "(explain)": 0.1},
        ),
        _trace(
            "c-int-nocand",
            {"operation": "service_logs", "args": {}},
            "escalate",
            candidates={"(escalate)": 0.9, "memory_stats": 0.1},
        ),
    ]


def _expected_verdicts(traces, policy, metrics_mod, gate_mod):
    """The verdicts every metric must produce for *traces* under *policy*.

    Computed directly from ``apply_policy_to_prediction`` +
    ``metrics_bridge``/``metrics.py`` -- the same functions the metrics
    themselves call -- so this is a genuine cross-check, not a restatement.
    """
    out = {}
    for trace in traces:
        raw_prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
        applied = deepeval_layer.apply_policy_to_prediction(
            policy, raw_prediction, metrics_mod=metrics_mod
        )
        row = metrics_bridge.row_metrics(metrics_mod, gate_mod, applied)
        kind = metrics_mod.expect_kind(applied.expected)

        if kind == "operation":
            right_action = metrics_mod._right_proposal(applied)
            correct_abstain_escalate = True
        elif kind == "escalate":
            right_action = True
            correct_abstain_escalate = metrics_mod._escalated(applied)
        else:  # explain
            right_action = True
            correct_abstain_escalate = applied.outcome == "explain"

        if row["missing_candidate"]:
            missing_candidate_handled = metrics_mod._escalated(applied)
        else:
            missing_candidate_handled = True

        out[trace.case_id] = {
            "right_action": right_action,
            "wrong_mutating": not row["wrong_mutating"],
            "correct_abstain_escalate": correct_abstain_escalate,
            "missing_candidate_handled": missing_candidate_handled,
        }
    return out


@pytest.mark.parametrize("policy", ["raw", "scorer-r3b-shipped"])
def test_evaluate_traces_runs_offline_and_grades_per_policy(
    monkeypatch, tmp_path, traces, metrics_mod, gate_mod, policy
):
    _block_sockets(monkeypatch)
    results_folder = tmp_path / f"results-{policy}"

    outcome = deepeval_layer.evaluate_traces(
        traces,
        policy,
        results_folder=results_folder,
        metrics_mod=metrics_mod,
        gate_mod=gate_mod,
    )

    run_files = list(results_folder.glob("test_run_*.json"))
    assert run_files, f"no test_run_*.json written to {results_folder}"
    with open(run_files[0], encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload, "test_run_*.json is empty"

    expected = _expected_verdicts(traces, policy, metrics_mod, gate_mod)
    by_case = {r.metadata["case_id"]: r for r in outcome.deepeval_result.test_results}
    assert set(by_case) == {t.case_id for t in traces}

    for case_id, want in expected.items():
        got = {m.name: m.success for m in by_case[case_id].metrics_data}
        assert got == want, f"case {case_id!r} under policy {policy!r}: {got} != {want}"

    # corpus metrics come from metrics_bridge.compute() on the SAME
    # policy-applied predictions, not recomputed by DeepEval.
    policy_applied_predictions = [
        deepeval_layer.apply_policy_to_prediction(
            policy, deepeval_layer.prediction_from_trace(t, metrics_mod), metrics_mod=metrics_mod
        )
        for t in traces
    ]
    direct_corpus = metrics_bridge.compute(
        policy_applied_predictions, metrics_mod=metrics_mod, gate_mod=gate_mod
    )
    assert outcome.corpus_metrics == direct_corpus


def test_evaluate_traces_scenario_c_raw_policy_matches_ungated_grading(
    monkeypatch, tmp_path, traces, metrics_mod, gate_mod
):
    """Scenario (c): for policy "raw", verdicts equal the old raw-graded ones.

    Every fixture trace's raw record is already internally consistent with
    its own candidates' bare argmax (as a real Track B interface's record
    would be), so "raw" policy grading and grading the untouched raw
    prediction directly must agree case for case.
    """
    _block_sockets(monkeypatch)
    outcome = deepeval_layer.evaluate_traces(
        traces,
        "raw",
        results_folder=tmp_path / "results-raw-parity",
        metrics_mod=metrics_mod,
        gate_mod=gate_mod,
    )
    by_case = {r.metadata["case_id"]: r for r in outcome.deepeval_result.test_results}

    for trace in traces:
        raw_prediction = deepeval_layer.prediction_from_trace(trace, metrics_mod)
        row = metrics_bridge.row_metrics(metrics_mod, gate_mod, raw_prediction)
        kind = metrics_mod.expect_kind(raw_prediction.expected)

        if kind == "operation":
            old_right_action = metrics_mod._right_proposal(raw_prediction)
            old_correct_abstain_escalate = True
        elif kind == "escalate":
            old_right_action = True
            old_correct_abstain_escalate = metrics_mod._escalated(raw_prediction)
        else:
            old_right_action = True
            old_correct_abstain_escalate = raw_prediction.outcome == "explain"
        old_missing_candidate_handled = (
            metrics_mod._escalated(raw_prediction) if row["missing_candidate"] else True
        )

        got = {m.name: m.success for m in by_case[trace.case_id].metrics_data}
        assert got["right_action"] is old_right_action
        assert got["wrong_mutating"] is (not row["wrong_mutating"])
        assert got["correct_abstain_escalate"] is old_correct_abstain_escalate
        assert got["missing_candidate_handled"] is old_missing_candidate_handled


def test_evaluate_traces_keeps_deepeval_state_out_of_the_repo_without_chdir_in_the_test(
    monkeypatch, tmp_path, traces, metrics_mod, gate_mod
):
    """Runs with cwd == wherever pytest started (the repo, typically) -- the
    library itself, not the test, must keep DeepEval's ".deepeval" state
    scoped to the run dir under results_folder."""
    _block_sockets(monkeypatch)
    cwd_before = Path.cwd()
    results_folder = tmp_path / "results-cwd-check"

    deepeval_layer.evaluate_traces(
        traces,
        "raw",
        results_folder=results_folder,
        metrics_mod=metrics_mod,
        gate_mod=gate_mod,
    )

    assert Path.cwd() == cwd_before, "evaluate_traces() must restore the caller's cwd"
    assert not (REPO_ROOT / ".deepeval").exists(), (
        "evaluate_traces() must never create .deepeval/ inside the repo, even when "
        "called with cwd == the repo root and the test itself never chdirs"
    )
    assert not (Path.cwd() / ".deepeval").exists()
    # If DeepEval created any local state at all, it may only be under this
    # call's own run dir inside results_folder -- nowhere else.
    stray = [
        p
        for p in tmp_path.rglob(".deepeval")
        if results_folder not in p.parents and p != results_folder
    ]
    assert stray == [], f".deepeval/ leaked outside the run dir: {stray}"


def test_evaluate_traces_corpus_metrics_are_not_recomputed_by_deepeval(
    monkeypatch, tmp_path, traces, metrics_mod, gate_mod
):
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


# ---------------------------------------------------------------------------
# Codex wave-2 review: every effective deepeval destination must be outside
# the repository, checked before anything is created.
# ---------------------------------------------------------------------------


def test_evaluate_traces_refuses_a_results_folder_inside_the_repo():
    target = REPO_ROOT / "evals-results-should-not-exist"
    with pytest.raises(deepeval_layer.DeepevalLayerError):
        deepeval_layer.evaluate_traces([], "raw", results_folder=target)
    assert not target.exists()


def test_evaluate_traces_refuses_a_display_config_folder_inside_the_repo(tmp_path):
    from deepeval.evaluate.configs import DisplayConfig

    inside = REPO_ROOT / "evals-display-should-not-exist"
    config = DisplayConfig(results_folder=str(inside), print_results=False, show_indicator=False)
    with pytest.raises(deepeval_layer.DeepevalLayerError):
        deepeval_layer.evaluate_traces(
            [], "raw", results_folder=tmp_path / "run", display_config=config
        )
    assert not inside.exists()
    assert not (tmp_path / "run").exists()
