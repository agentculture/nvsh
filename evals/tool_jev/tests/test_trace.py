"""Tests for evals/tool_jev/trace.py (task t6, issue #64).

Acceptance criteria covered:

1. a fixture predictions line round-trips into RawRecord and back without loss
2. applying two policies yields two final entries and the raw record is
   byte-identical before and after
3. the trace writer refuses a path inside the git worktree

All fixtures here are synthetic (made up for this test), never real request
text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tool_jev import trace as trace_mod
from evals.tool_jev.trace import (
    PolicyResult,
    Prediction,
    RawRecord,
    Trace,
    TraceWriteError,
    append_trace,
    read_traces,
    write_traces,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fixture_line(**overrides) -> dict:
    row = {
        "id": "case-001",
        "expected": {"operation": "inspect-disk", "args": {}},
        "outcome": "propose",
        "operation": "inspect-disk",
        "arguments": {},
        "candidates": {"inspect-disk": 0.9, "(escalate)": 0.1},
        "tokens": 12,
        "ttfd_ms": 45.5,
        "latency_ms": 120.25,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Criterion 1: predictions line round-trips into RawRecord and back.
# ---------------------------------------------------------------------------


def test_prediction_line_round_trips_through_raw_record_without_loss():
    line = _fixture_line()
    prediction1 = Prediction.from_dict(line)

    raw = RawRecord.from_prediction(prediction1)

    rebuilt_line = raw.to_prediction_dict(prediction1.id, prediction1.expected)
    prediction2 = Prediction.from_dict(rebuilt_line)

    assert prediction1 == prediction2


def test_prediction_line_round_trips_with_invalid_reason_and_null_fields():
    line = _fixture_line(
        outcome="invalid",
        operation=None,
        arguments=None,
        candidates=None,
        invalid_reason="unparseable",
    )
    prediction1 = Prediction.from_dict(line)

    raw = RawRecord.from_prediction(prediction1)
    rebuilt_line = raw.to_prediction_dict(prediction1.id, prediction1.expected)
    prediction2 = Prediction.from_dict(rebuilt_line)

    assert prediction1 == prediction2
    assert raw.invalid_reason == "unparseable"
    assert raw.operation is None
    assert raw.arguments is None
    assert raw.candidates is None


def test_raw_record_from_prediction_line_helper_matches_manual_build():
    line = _fixture_line()
    via_helper = RawRecord.from_prediction_line(line)
    via_manual = RawRecord.from_prediction(Prediction.from_dict(line))
    assert via_helper == via_manual


def test_raw_record_to_dict_and_from_dict_round_trip():
    line = _fixture_line()
    raw = RawRecord.from_prediction(Prediction.from_dict(line))
    rebuilt = RawRecord.from_dict(raw.to_dict())
    assert rebuilt == raw


def test_raw_record_from_provider_answer_fits_the_same_schema():
    raw = RawRecord.from_provider_answer(
        provider="acme",
        model="acme-large",
        returned_model="acme-large-2026-09-01",
        interface="tool_call",
        outcome="propose",
        operation="inspect-disk",
        arguments={},
        candidates=None,
    )
    assert raw.provider == "acme"
    assert raw.model == "acme-large"
    assert raw.returned_model == "acme-large-2026-09-01"
    assert raw.interface == "tool_call"
    assert raw.tokens is None
    assert raw.ttfd_ms is None
    assert raw.candidates is None


def test_raw_record_from_provider_answer_rejects_bad_interface():
    with pytest.raises(ValueError):
        RawRecord.from_provider_answer(
            provider="acme",
            model="acme-large",
            returned_model=None,
            interface="not-a-real-interface",
            outcome="propose",
        )


# ---------------------------------------------------------------------------
# Criterion 2: two policies -> two final entries, raw untouched.
# ---------------------------------------------------------------------------


def test_two_policies_yield_two_final_entries_and_raw_is_byte_identical():
    line = _fixture_line()
    trace0 = Trace.from_prediction_line(line, split="val", subject="stock")

    before_raw_json = json.dumps(trace0.raw.to_dict(), sort_keys=True)

    trace1 = trace0.with_policy("policy-a", "propose", "matched top candidate")
    trace2 = trace1.with_policy("policy-b", "escalate", "confidence below gate threshold")

    after_raw_json = json.dumps(trace2.raw.to_dict(), sort_keys=True)

    assert len(trace2.final) == 2
    assert trace2.final["policy-a"] == PolicyResult("propose", "matched top candidate")
    assert trace2.final["policy-b"] == PolicyResult("escalate", "confidence below gate threshold")
    assert before_raw_json == after_raw_json
    assert trace2.raw == trace0.raw
    # the original traces are untouched by later with_policy calls
    assert trace0.final == {}
    assert len(trace1.final) == 1


def test_with_policy_never_mutates_the_original_trace_object():
    line = _fixture_line()
    trace0 = Trace.from_prediction_line(line, split="test")
    original_final = trace0.final

    trace0.with_policy("policy-a", "propose", "reason a")

    assert trace0.final == {}
    assert trace0.final is original_final


def test_trace_to_dict_and_from_dict_round_trip():
    line = _fixture_line()
    trace = Trace.from_prediction_line(line, split="val", subject="stock")
    trace = trace.with_policy("policy-a", "propose", "reason a")

    rebuilt = Trace.from_dict(trace.to_dict())

    assert rebuilt == trace


# ---------------------------------------------------------------------------
# Criterion 3: the writer refuses a path inside the git worktree.
# ---------------------------------------------------------------------------


def _sample_trace() -> Trace:
    return Trace.from_prediction_line(_fixture_line(), split="val", subject="stock")


def test_append_trace_refuses_a_path_inside_the_git_worktree(tmp_path):
    inside_repo_path = REPO_ROOT / "evals" / "tool_jev" / "tests" / "_should_not_exist.jsonl"
    assert not inside_repo_path.exists()

    with pytest.raises(TraceWriteError):
        append_trace(inside_repo_path, _sample_trace())

    assert not inside_repo_path.exists()


def test_write_traces_refuses_a_path_inside_the_git_worktree():
    inside_repo_path = REPO_ROOT / "evals" / "tool_jev" / "tests" / "_should_not_exist_batch.jsonl"
    assert not inside_repo_path.exists()

    with pytest.raises(TraceWriteError):
        write_traces(inside_repo_path, [_sample_trace()])

    assert not inside_repo_path.exists()


def test_append_trace_writes_jsonl_outside_the_worktree(tmp_path):
    run_dir = tmp_path / "run-outside-repo"
    out_path = run_dir / "traces.jsonl"

    trace1 = _sample_trace()
    trace2 = trace1.with_policy("policy-a", "propose", "reason a")

    append_trace(out_path, trace1)
    append_trace(out_path, trace2)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    read_back = [json.loads(line) for line in lines]
    assert read_back[0]["final"] == {}
    assert read_back[1]["final"]["policy-a"] == {"decision": "propose", "reason": "reason a"}


def test_write_traces_truncates_and_read_traces_round_trips(tmp_path):
    run_dir = tmp_path / "another-run"
    out_path = run_dir / "traces.jsonl"

    trace1 = _sample_trace()
    trace2 = trace1.with_policy("policy-a", "escalate", "low confidence")

    write_traces(out_path, [trace1, trace2])
    read_back = read_traces(out_path)

    assert read_back == [trace1, trace2]

    # truncation: writing again with fewer traces replaces the file's contents
    write_traces(out_path, [trace1])
    assert read_traces(out_path) == [trace1]


# ---------------------------------------------------------------------------
# Reviewer finding (codex wave 1): serialization must not reorder
# ``raw.candidates`` -- gate.decide's tie-break depends on offered order,
# and metrics_bridge derives ``offered`` from ``list(candidates.keys())``.
# ---------------------------------------------------------------------------


def test_serialization_preserves_candidate_order_for_policy_decisions(tmp_path):
    from evals.tool_jev import policies

    raw = RawRecord(
        outcome="propose",
        operation="disk_usage",
        arguments={},
        candidates={"disk_usage": 0.5, "(explain)": 0.5},
    )
    trace0 = Trace(case_id="c1", split="test", raw=raw)
    policy = policies.load_policy(policies.builtin_policy_path("raw"))

    offered_before = list(trace0.raw.candidates.keys())
    decision_before = policies.apply(policy, trace0.raw.to_dict(), offered_before)

    out_path = tmp_path / "run-outside-repo" / "traces.jsonl"
    write_traces(out_path, [trace0])
    (reloaded,) = read_traces(out_path)

    offered_after = list(reloaded.raw.candidates.keys())
    decision_after = policies.apply(policy, reloaded.raw.to_dict(), offered_after)

    # The bug this guards against: json.dumps(..., sort_keys=True) sorted
    # "(explain)" ahead of "disk_usage" (ASCII "(" < "d"), flipping the
    # decision from "propose" to "explain" purely from a round trip.
    assert offered_after == offered_before
    assert decision_before[0] == "propose"
    assert decision_after[0] == "propose"
    assert decision_before == decision_after


def test_append_trace_also_preserves_candidate_order(tmp_path):
    raw = RawRecord(
        outcome="propose",
        operation="disk_usage",
        arguments={},
        candidates={"disk_usage": 0.5, "(explain)": 0.5},
    )
    trace0 = Trace(case_id="c1", split="test", raw=raw)

    out_path = tmp_path / "run-outside-repo" / "traces.jsonl"
    append_trace(out_path, trace0)
    (reloaded,) = read_traces(out_path)

    assert list(reloaded.raw.candidates.keys()) == ["disk_usage", "(explain)"]


def test_is_inside_git_worktree_true_for_repo_paths_false_for_tmp_path(tmp_path):
    assert trace_mod._is_inside_git_worktree(REPO_ROOT / "evals" / "tool_jev" / "trace.py")
    assert trace_mod._is_inside_git_worktree(REPO_ROOT)
    assert not trace_mod._is_inside_git_worktree(tmp_path / "some" / "nested" / "path.jsonl")
