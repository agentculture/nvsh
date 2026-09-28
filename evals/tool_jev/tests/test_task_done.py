"""Track A 'task done' beside the strict 'right proposal' (deviation d9, 2026-09-29).

In the LfmTier loop a read-only operation runs as an inspection; a subject
that runs the expected read-only operation and then explains the result has
done what was asked, although it never *proposed* the operation.
"""

from __future__ import annotations

import json

from evals.tool_jev import metrics_bridge, report
from evals.tool_jev.tests.test_track_a_loop import SERVICE, STATUS, _call, _finish
from evals.tool_jev.trace import RawRecord

EXPLAIN = _call("explain", {"text": "it is running"})


# -- the record carries the inspections the loop ran ---------------------------------


def test_a_track_a_record_lists_the_read_only_inspections_it_ran(tmp_path):
    record, rounds, _ = _finish(tmp_path, [("answer", STATUS), ("answer", EXPLAIN)])
    assert (record.outcome, rounds) == ("explain", 2)
    assert record.inspections == ({"operation": "service_status", "arguments": SERVICE},)


def test_a_record_that_inspected_nothing_has_no_inspections(tmp_path):
    record, _, _ = _finish(tmp_path, [("answer", EXPLAIN)])
    assert record.inspections == ()


def test_inspections_round_trip_and_old_traces_still_load():
    record = RawRecord(
        outcome="explain",
        operation=None,
        arguments=None,
        candidates=None,
        inspections=({"operation": "service_status", "arguments": SERVICE},),
    )
    again = RawRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert again == record
    old = {k: v for k, v in record.to_dict().items() if k != "inspections"}
    assert RawRecord.from_dict(old).inspections is None
    assert "inspections" not in RawRecord.from_dict(old).to_dict()


# -- the metric ----------------------------------------------------------------------------


def _prediction(metrics_mod, case_id, expected, outcome, operation=None, arguments=None):
    return metrics_mod.Prediction.from_dict(
        {
            "id": case_id,
            "expected": expected,
            "outcome": outcome,
            "operation": operation,
            "arguments": arguments,
            "candidates": None,
            "tokens": 0,
            "ttfd_ms": 0.0,
            "latency_ms": 0.0,
        }
    )


def test_task_done_counts_an_explain_after_the_expected_read_only_inspection():
    metrics_mod = metrics_bridge.load_metrics_module()
    status = {"operation": "service_status", "args": SERVICE}
    restart = {"operation": "service_restart", "args": SERVICE}
    predictions = [
        _prediction(metrics_mod, "ran-it", status, "explain"),  # done: inspected, explained
        _prediction(metrics_mod, "wrong-args", status, "explain"),  # not done: other service
        _prediction(metrics_mod, "proposed", status, "propose", "service_status", SERVICE),
        _prediction(metrics_mod, "mutating", restart, "explain"),  # not done: mutating expected
        _prediction(metrics_mod, "nothing", status, "explain"),  # not done: no inspection
    ]
    inspections = {
        "ran-it": ({"operation": "service_status", "arguments": SERVICE},),
        "wrong-args": ({"operation": "service_status", "arguments": {"service": "other"}},),
        "mutating": ({"operation": "service_status", "arguments": SERVICE},),
        "nothing": (),
    }
    result = metrics_bridge.compute(predictions, inspections=inspections)
    assert result["metrics_compute"]["right_proposals"]["n"] == 1  # strict: unchanged
    assert result["task_done"] == {"n": 2, "N": 5, "inspections_recorded": True}


def test_task_done_equals_strict_when_no_inspections_were_recorded():
    metrics_mod = metrics_bridge.load_metrics_module()
    status = {"operation": "service_status", "args": SERVICE}
    predictions = [
        _prediction(metrics_mod, "a", status, "propose", "service_status", SERVICE),
        _prediction(metrics_mod, "b", status, "explain"),
    ]
    result = metrics_bridge.compute(predictions)
    assert result["task_done"] == {"n": 1, "N": 2, "inspections_recorded": False}


# -- the page ------------------------------------------------------------------------------


def test_main_table_shows_right_and_task_done_columns():
    row = {
        "subject": "ref",
        "variant": "model-only",
        "kind": "reference",
        "harness_policy": None,
        "measurable": False,
        "top1": {"rate": None},
        "ece": None,
        "brier": None,
        "coverage": {"rate": 0.5},
        "abstain_precision": 1.0,
        "abstain_recall": 0.5,
        "missing_candidate": {"rate": 0.0},
        "wrong_mutations": 0,
        "right": {"n": 5, "N": 83},
        "task_done": {"n": 66, "N": 83, "inspections_recorded": True},
    }
    lines = report._render_main_table([row])
    assert "| Right | Task done |" in lines[0]
    assert "| 5/83 | 66/83 |" in lines[2]
