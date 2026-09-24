"""The shared issue-46 metrics module: scripts/lfm-finetune/metrics.py.

Every figure is checked against a small hand-computed fixture (the arithmetic
is written out next to each assertion), so a change to a definition shows up
here rather than in a report. Operation names appear only as fixture data;
the module under test decides mutating from the operation table's
``read_only`` flag.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from nvsh.tiers import bench as tier_bench
from nvsh.tiers.router import AGENT, TierOutcome

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/metrics.py"

ESC = tier_bench.ESCALATE_LABEL
EXP = tier_bench.EXPLAIN_LABEL


@pytest.fixture(scope="module")
def metrics():
    spec = importlib.util.spec_from_file_location("lfm_metrics", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lfm_metrics"] = module
    spec.loader.exec_module(module)
    return module


def _op(name: str, **args) -> dict:
    return {"operation": name, "args": dict(args)}


def _rec(
    entry_id: str,
    expected: dict,
    outcome: str,
    operation: str | None = None,
    arguments: dict | None = None,
    candidates: dict | None = None,
    tokens: int = 0,
    ttfd_ms: float = 0.0,
    latency_ms: float = 0.0,
    **extra,
) -> dict:
    return {
        "id": entry_id,
        "expected": expected,
        "outcome": outcome,
        "operation": operation,
        "arguments": arguments,
        "candidates": candidates,
        "tokens": tokens,
        "ttfd_ms": ttfd_ms,
        "latency_ms": latency_ms,
        **extra,
    }


TOKENS = [10, 12, 8, 9, 30, 4, 11, 6, 20, 5, 7, 3, 15]
TTFD = [500.0] + [40.0 + 2 * i for i in range(12)]  # cold 500, warm 40..62


def _fixture_rows() -> list[dict]:
    """Thirteen predictions; the expected figures are worked out in each test."""
    rows = [
        # p1: right proposal
        _rec(
            "p1",
            _op("memory_stats"),
            "propose",
            "memory_stats",
            {},
            {"memory_stats": 0.9, ESC: 0.1},
        ),
        # p2: right read-only operation, wrong arguments
        _rec(
            "p2",
            _op("service_status", service="docker"),
            "propose",
            "service_status",
            {"service": "nginx"},
            {"service_status": 0.6, "service_logs": 0.4},
        ),
        # p3: a different mutating operation -> wrong mutating (wrong operation)
        _rec(
            "p3",
            _op("service_restart", service="docker"),
            "propose",
            "container_restart",
            {"container": "docker"},
            {"container_restart": 0.7, "service_restart": 0.3},
        ),
        # p4: the expected mutating operation with wrong arguments
        _rec(
            "p4",
            _op("container_restart", container="trainer"),
            "propose",
            "container_restart",
            {"container": "inference"},
            {"container_restart": 0.8, ESC: 0.2},
        ),
        # p5: the model's own output was unparseable
        _rec(
            "p5",
            _op("memory_stats"),
            "invalid",
            candidates={"memory_stats": 0.55, ESC: 0.45},
            invalid_reason="malformed",
        ),
        # p6: escalate expected, escalated (TP)
        _rec("p6", {"escalate": True}, "escalate", candidates={ESC: 1.0}),
        # p7: escalate expected, a mutating proposal (FN, FP tool call, wrong mutating)
        _rec(
            "p7",
            {"escalate": True},
            "propose",
            "service_restart",
            {"service": "x"},
            {"service_restart": 0.5, ESC: 0.3, EXP: 0.2},
        ),
        # p8: escalate expected, explained (FN)
        _rec("p8", {"escalate": True}, "explain", candidates={EXP: 0.6, ESC: 0.4}),
        # p9: explain expected, explained
        _rec("p9", {"explain": True}, "explain", candidates={EXP: 0.9, ESC: 0.1}),
        # p10: explain expected, escalated (left out of precision, like bench)
        _rec("p10", {"explain": True}, "escalate", candidates={ESC: 0.7, EXP: 0.3}),
        # p11: explain expected, a read-only proposal (FP tool call, not mutating)
        _rec(
            "p11",
            {"explain": True},
            "propose",
            "memory_stats",
            {},
            {"memory_stats": 0.95, EXP: 0.05},
        ),
        # p12: operation expected, escalated (FP escalation)
        _rec("p12", _op("memory_stats"), "escalate", candidates={ESC: 0.6, "memory_stats": 0.4}),
        # p13: escalate expected, an operation not in the table -> invalid; no distribution
        _rec("p13", {"escalate": True}, "propose", "not_an_operation", {}, None),
    ]
    for row, tokens, ttfd in zip(rows, TOKENS, TTFD):
        row["tokens"] = tokens
        row["ttfd_ms"] = ttfd
        row["latency_ms"] = ttfd + 10.0
    return rows


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def result(metrics, tmp_path):
    return metrics.compute(metrics.read_predictions(_write(tmp_path, _fixture_rows())))


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def test_schema_fields_are_the_documented_ones(metrics):
    assert metrics.FIELDS == (
        "id",
        "expected",
        "outcome",
        "operation",
        "arguments",
        "candidates",
        "tokens",
        "ttfd_ms",
        "latency_ms",
    )
    assert metrics.OUTCOMES == ("propose", "explain", "escalate", "invalid")


def test_read_predictions_round_trips_the_fixture(metrics, tmp_path):
    predictions = metrics.read_predictions(_write(tmp_path, _fixture_rows()))
    assert [p.id for p in predictions] == [f"p{i}" for i in range(1, 14)]
    assert predictions[1].arguments == {"service": "nginx"}
    assert predictions[12].candidates is None


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda row: row.pop("tokens"), "tokens"),
        (lambda row: row.update(outcome="abstain"), "outcome"),
        (lambda row: row.update(expected={"escalate": True, "explain": True}), "expected"),
        (lambda row: row.update(candidates={"memory_stats": 0.5, ESC: 0.2}), "sum"),
        (lambda row: row.update(candidates={"memory_stats": 1.2, ESC: -0.2}), "candidates"),
        (lambda row: row.update(tokens=-1), "tokens"),
        (lambda row: row.update(operation=None), "operation"),
    ],
)
def test_read_predictions_rejects_bad_records_with_the_line(metrics, tmp_path, mutate, fragment):
    rows = _fixture_rows()
    mutate(rows[0])
    path = _write(tmp_path, rows)
    with pytest.raises(metrics.MetricsError) as info:
        metrics.read_predictions(path)
    assert "line 1" in str(info.value)
    assert fragment in str(info.value)


def test_read_predictions_rejects_unparseable_json_and_duplicate_ids(metrics, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_fixture_rows()[0]) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(metrics.MetricsError, match="line 2"):
        metrics.read_predictions(path)
    rows = _fixture_rows()
    rows[1]["id"] = "p1"
    duplicated = _write(tmp_path, rows)
    with pytest.raises(metrics.MetricsError, match="duplicate"):
        metrics.read_predictions(duplicated)


# ---------------------------------------------------------------------------
# Decision metrics on the fixture
# ---------------------------------------------------------------------------


def test_right_proposals_n_of_n_and_percent(result):
    # operation-expected: p1 p2 p3 p4 p5 p12 -> N=6; only p1 is right
    assert result["right_proposals"] == {"n": 1, "N": 6, "percent": pytest.approx(100 / 6)}


def test_escalation_recall_and_precision(result):
    # escalate-expected: p6 p7 p8 p13 -> TP=1 (p6), FN=3; FP=1 (p12, operation-expected);
    # p10 escalated on an explain entry and is reported separately, as in bench.
    escalation = result["escalation"]
    assert (escalation["tp"], escalation["fn"], escalation["fp"]) == (1, 3, 1)
    assert escalation["recall"] == pytest.approx(0.25)
    assert escalation["precision"] == pytest.approx(0.5)
    assert escalation["escalated_on_explain"] == 1


def test_abstain_is_the_escalate_outcome_with_strict_precision(result):
    # c25: issue 46's abstain is nvsh's escalate. Deviation d2 (operator, 2026-09-23):
    # abstention precision is strict -- an escalation on an explain entry is a false
    # abstention too -- so p6 / (p6 + p12 + p10) = 1/3; bench's figure stays in "escalation".
    assert result["abstention"]["recall"] == result["escalation"]["recall"]
    assert result["abstention"]["precision"] == pytest.approx(1 / 3)
    assert result["escalation"]["precision_strict"] == pytest.approx(1 / 3)
    assert result["escalation"]["precision"] == pytest.approx(0.5)


def test_escalation_agrees_with_bench_compute_escalation(metrics, tmp_path):
    predictions = metrics.read_predictions(_write(tmp_path, _fixture_rows()))
    items = []
    for p in predictions:
        entry = tier_bench.CorpusEntry(
            id=p.id, kind="explicit", text="", expect=p.expected, source="fixture"
        )
        escalated = p.outcome == "escalate"
        outcome = TierOutcome(
            handled_by=None if escalated else "tier2",
            escalated_to=AGENT if escalated else None,
            explanation="" if p.outcome == "explain" else None,
            operation=p.operation if p.outcome == "propose" else None,
            args=dict(p.arguments or {}),
        )
        items.append(tier_bench.ItemResult(entry=entry, outcome=outcome, latency_ms=0.0))
    bench = tier_bench.compute_escalation(items)
    ours = metrics.compute(predictions)["escalation"]
    assert (ours["tp"], ours["fn"], ours["fp"]) == (bench["tp"], bench["fn"], bench["fp"])


def test_false_positive_tool_calls_over_explain_and_escalate_items(result):
    # explain/escalate-expected: p6 p7 p8 p13 p9 p10 p11 -> N=7; proposals: p7, p11
    # (p13's unknown operation is an invalid output, counted there instead)
    assert result["false_positive_tool_calls"] == {"n": 2, "N": 7, "rate": pytest.approx(2 / 7)}


def test_wrong_mutating_splits_operation_and_arguments(result):
    # wrong operation (bench's definition): p3 (other mutating op), p7 (on escalate);
    # wrong arguments of the expected mutating operation: p4.  p2/p11 are read-only.
    assert result["wrong_mutating"] == {"wrong_operation": 2, "wrong_arguments": 1, "total": 3}


def test_invalid_outputs_count_malformed_and_out_of_table(result):
    # p5 recorded invalid (malformed), p13 proposed an operation not in the table
    assert result["invalid"] == {
        "n": 2,
        "N": 13,
        "rate": pytest.approx(2 / 13),
        "by_reason": {"malformed": 1, "unknown_operation": 1},
    }


def test_proposal_with_arguments_failing_the_schema_is_invalid(metrics):
    predictions = [
        metrics.Prediction.from_dict(
            _rec("x", _op("power_set", mode="balanced"), "propose", "power_set", {"mode": "turbo"})
        )
    ]
    got = metrics.compute(predictions)
    assert got["invalid"]["n"] == 1
    assert list(got["invalid"]["by_reason"]) != ["unknown_operation"]
    assert got["wrong_mutating"]["total"] == 0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_ece_ten_equal_width_bins_on_the_fixture(result):
    # Top candidate vs the expected label, 12 records with a distribution (p13 has none):
    #   bin 5: p5 .55 right, p7 .50 wrong            -> |0.5  - 0.525 | * 2/12
    #   bin 6: p2 .6 right, p8 .6 wrong, p12 .6 wrong -> |1/3  - 0.6   | * 3/12
    #   bin 7: p3 .7 wrong, p10 .7 wrong             -> |0    - 0.7   | * 2/12
    #   bin 8: p4 .8 right                           -> |1    - 0.8   | * 1/12
    #   bin 9: p1 .9 p6 1.0 p9 .9 right, p11 .95 wrong -> |0.75 - 0.9375| * 4/12
    #   sum = (0.05 + 0.8 + 1.4 + 0.2 + 0.75) / 12 = 3.2 / 12
    calibration = result["calibration"]
    assert calibration["n"] == 12
    assert calibration["without_distribution"] == 1
    assert calibration["ece"] == pytest.approx(3.2 / 12)
    assert [b["n"] for b in calibration["bins"]] == [0, 0, 0, 0, 0, 2, 3, 2, 1, 4]


def test_brier_on_the_fixture(result):
    # Multi-class Brier per record, sum over labels of (p - onehot)^2, then the mean:
    #   .02 .32 .98 .08 .405 0 .78 .72 .02 .98 1.805 .72 -> 6.83 / 12
    assert result["calibration"]["brier"] == pytest.approx(6.83 / 12)


def test_ece_is_zero_when_confidence_matches_accuracy(metrics):
    # four at 0.75, three right: bin 7, |0.75 - 0.75| = 0
    pairs = [(0.75, True), (0.75, True), (0.75, True), (0.75, False)]
    assert metrics.ece(pairs) == pytest.approx(0.0)


def test_ece_puts_confidence_one_in_the_last_bin_and_edges_up(metrics):
    assert metrics.bin_index(1.0) == 9
    assert metrics.bin_index(0.0) == 0
    assert metrics.bin_index(0.1) == 1
    assert metrics.bin_index(0.3) == 3
    assert metrics.ece([(1.0, False)]) == pytest.approx(1.0)


def test_brier_counts_a_gold_label_missing_from_the_candidates(metrics):
    # {a: 1.0}, gold b: (1-0)^2 for a plus (0-1)^2 for b = 2
    assert metrics.brier_one({"a": 1.0}, "b") == pytest.approx(2.0)
    assert metrics.brier_one({"a": 0.5, "b": 0.5}, "b") == pytest.approx(0.5)


def test_top_candidate_tie_is_broken_by_label_order(metrics):
    assert metrics.top_candidate({"b": 0.5, "a": 0.5}) == ("a", 0.5)


def test_calibration_empty_when_no_record_has_a_distribution(metrics):
    predictions = [metrics.Prediction.from_dict(_rec("x", {"escalate": True}, "escalate"))]
    calibration = metrics.compute(predictions)["calibration"]
    assert (calibration["n"], calibration["ece"], calibration["brier"]) == (0, None, None)


# ---------------------------------------------------------------------------
# Tokens and timing
# ---------------------------------------------------------------------------


def test_tokens_generated_per_decision(result):
    # total 140 over 13; sorted 3 4 5 6 7 8 [9] 10 11 12 15 20 30
    assert result["tokens"] == {"total": 140, "mean": pytest.approx(140 / 13), "median": 9}


def test_time_to_first_decision_and_latency_cold_and_warm(result):
    # cold = the first record; warm 40..62 step 2 -> median (50+52)/2, nearest-rank p95 = 62
    assert result["time_to_first_decision"] == {
        "cold_ms": 500.0,
        "warm_median_ms": 51.0,
        "warm_p95_ms": 62.0,
    }
    assert result["latency"] == {"cold_ms": 510.0, "warm_median_ms": 61.0, "warm_p95_ms": 72.0}


# ---------------------------------------------------------------------------
# The issue-46 mapping (reporting only)
# ---------------------------------------------------------------------------


def test_issue46_json_maps_each_outcome(metrics):
    def as46(row):
        return metrics.issue46_json(metrics.Prediction.from_dict(row))

    assert as46(
        _rec("a", _op("service_status", service="d"), "propose", "service_status", {"service": "d"})
    ) == {"action": "tool", "tool": "service_status", "arguments": {"service": "d"}}
    assert as46(_rec("b", {"escalate": True}, "escalate")) == {"action": "abstain"}
    assert as46(_rec("c", {"explain": True}, "explain")) == {"action": "no_action"}
    assert as46(_rec("d", {"explain": True}, "invalid")) == {"action": "invalid"}


def test_mapping_is_emitted_with_the_metrics(metrics, result):
    mapping = result["issue46_mapping"]
    assert [row["nvsh"] for row in mapping["rows"]] == list(metrics.OUTCOMES)
    assert '{"action": "abstain"}' in mapping["markdown"]
    assert "escalate" in mapping["markdown"]
    assert mapping["note"]


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_prints_json(metrics, tmp_path, capsys):
    assert metrics.main([str(_write(tmp_path, _fixture_rows()))]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["right_proposals"]["N"] == 6
    assert out["issue46_mapping"]["rows"]


def test_main_reports_a_bad_file_on_stderr(metrics, tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text("{nope\n", encoding="utf-8")
    assert metrics.main([str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "line 1" in captured.err
