#!/usr/bin/env python3
"""Issue-46 metrics from a predictions JSONL file, shared by every model measured.

A development-machine tool for the Qwen3.5-0.8B tool-decision experiment
(part of #46). It is NEVER imported by the nvsh package -- nothing under nvsh/
may depend on it. The stock baseline, Track A (generative) and Track B
(candidate scoring) all write the same predictions file, and this module
turns any one of them into the same figures, so the comparison table compares
models rather than scorers.

Usage::

    python scripts/lfm-finetune/metrics.py out/predictions-stock.jsonl

prints one JSON object with every metric plus the issue-46 mapping below.

The predictions file
--------------------

One JSON object per line, one line per corpus entry, in the order the entries
were run (the first line is the cold decision). The fields, all required:

``id``
    The corpus entry's id (unique within the file).
``expected``
    The entry's ``expect`` block, exactly as in the corpus:
    ``{"operation": name, "args": {...}}``, ``{"escalate": true}`` or
    ``{"explain": true}``.
``outcome``
    What the model decided: ``"propose"``, ``"explain"``, ``"escalate"``, or
    ``"invalid"`` when its output could not be read as any of the three.
``operation`` / ``arguments``
    The proposed operation and its arguments; ``null`` unless ``outcome`` is
    ``"propose"``.
``candidates``
    The model's own normalised probability over the candidate set, recorded
    before any threshold: ``{label: probability}``, where a label is an
    operation name or ``nvsh.tiers.bench``'s ``"(escalate)"`` /
    ``"(explain)"``. The values lie in [0, 1] and sum to 1; ``null`` when the
    run recorded none (such a line is left out of calibration and counted).
``tokens``
    Tokens the model generated for this decision (0 for a scorer that
    generates none).
``ttfd_ms`` / ``latency_ms``
    Time to first decision and total decision time, in milliseconds.

An optional ``invalid_reason`` names why an ``"invalid"`` line was invalid
(``"unparseable"`` when absent). A ``"propose"`` line whose operation is not
in ``nvsh/ops/table.py`` or whose arguments fail its schema is counted as
invalid too, under the table's own validation code, never as a tool call.

The metrics
-----------

* **Right proposals**: operation-expected entries answered with that
  operation and exactly those arguments (``bench._is_correct``), n/N and %.
* **Escalation (abstain) recall and precision**: ``bench.compute_escalation``'s
  definition -- recall over escalate-expected entries; precision's false
  positives are escalations of operation-expected entries. Explain entries are
  left out of both, as in bench, and escalations of them are reported
  separately as ``escalated_on_explain``.
* **False-positive tool calls**: proposals on explain- or escalate-expected
  entries, over all of those entries.
* **Wrong mutating**: bench's count (a mutating operation proposed where
  something else was expected) plus measure.py's separate count (the expected
  mutating operation with wrong arguments); the bar is judged on the total.
  Mutating is the operation table's ``read_only`` flag, never a name.
* **Invalid outputs**: as above, over every line.
* **ECE** (10 equal-width bins over the top candidate's probability, a
  probability of exactly 1.0 in the last bin) and **Brier** (multi-class: the
  sum over labels of (p - one-hot)^2, a gold label missing from the
  candidates counting as p = 0, averaged over lines). The top candidate is the
  highest probability, a tie going to the label that sorts first; it is right
  when its label is the expected one (operation name, or the escalate/explain
  label) -- arguments are not part of calibration.
* **Tokens generated** per decision (total, mean, median) and **time to first
  decision** / **latency** as bench reports latency: the first line cold, the
  rest warm (median and nearest-rank p95).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.ops import table as ops_table  # noqa: E402
from nvsh.tiers import bench as tier_bench  # noqa: E402

FIELDS = (
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
OUTCOMES = ("propose", "explain", "escalate", "invalid")

#: Equal-width confidence bins for ECE.
ECE_BINS = 10
#: How far a candidate distribution's sum may stray from 1 (JSON float round trip).
SUM_TOLERANCE = 1e-4
#: ``invalid_reason`` when an invalid line gives none.
DEFAULT_INVALID_REASON = "unparseable"

#: nvsh outcome -> issue 46's decision JSON, for the report only (decision c25:
#: models are trained and scored on nvsh's own tools; abstain is escalate).
ISSUE46_MAPPING = (
    {
        "nvsh": "propose",
        "issue46": '{"action": "tool", "tool": <operation>, "arguments": <arguments>}',
    },
    {"nvsh": "explain", "issue46": '{"action": "no_action"}'},
    {"nvsh": "escalate", "issue46": '{"action": "abstain"}'},
    {"nvsh": "invalid", "issue46": '{"action": "invalid"}'},
)
ISSUE46_NOTE = (
    "Reporting only: every model is trained and scored on nvsh's propose/explain/escalate "
    "tools. Issue 46's abstain is nvsh's escalate, so abstention precision and recall are "
    "the escalation figures. Explain (answer in words, no tool) has no counterpart in "
    "issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for "
    "Track B and is never counted as an abstention. An invalid output is not a decision."
)


class MetricsError(ValueError):
    """A predictions file that does not follow the schema above."""


@dataclass(frozen=True)
class Prediction:
    """One line of a predictions file."""

    id: str
    expected: dict
    outcome: str
    operation: str | None
    arguments: dict | None
    candidates: dict | None
    tokens: int
    ttfd_ms: float
    latency_ms: float
    invalid_reason: str | None = None

    @classmethod
    def from_dict(cls, row: object) -> "Prediction":
        """Validate one decoded line; raises :class:`MetricsError` naming the problem."""
        if not isinstance(row, dict):
            raise MetricsError("a line must be a JSON object")
        missing = [name for name in FIELDS if name not in row]
        if missing:
            raise MetricsError(f"missing field(s): {', '.join(missing)}")
        if not isinstance(row["id"], str) or not row["id"]:
            raise MetricsError("id must be a non-empty string")
        _check_expected(row["expected"])
        outcome = row["outcome"]
        if outcome not in OUTCOMES:
            raise MetricsError(f"outcome {outcome!r} is not one of {', '.join(OUTCOMES)}")
        _check_proposal(outcome, row["operation"], row["arguments"])
        _check_candidates(row["candidates"])
        tokens = row["tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise MetricsError("tokens must be a non-negative integer")
        for name in ("ttfd_ms", "latency_ms"):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise MetricsError(f"{name} must be a non-negative number")
        reason = row.get("invalid_reason")
        if reason is not None and not isinstance(reason, str):
            raise MetricsError("invalid_reason must be a string")
        return cls(
            id=row["id"],
            expected=dict(row["expected"]),
            outcome=outcome,
            operation=row["operation"],
            arguments=None if row["arguments"] is None else dict(row["arguments"]),
            candidates=None if row["candidates"] is None else dict(row["candidates"]),
            tokens=tokens,
            ttfd_ms=float(row["ttfd_ms"]),
            latency_ms=float(row["latency_ms"]),
            invalid_reason=reason,
        )


def _check_expected(expected: object) -> None:
    if not isinstance(expected, dict):
        raise MetricsError("expected must be an object")
    kinds = [key for key in ("operation", "escalate", "explain") if expected.get(key)]
    if len(kinds) != 1:
        raise MetricsError("expected must name exactly one of operation, escalate, explain")
    if kinds == ["operation"]:
        if not isinstance(expected["operation"], str):
            raise MetricsError("expected operation must be a string")
        if not isinstance(expected.get("args", {}), dict):
            raise MetricsError("expected args must be an object")


def _check_proposal(outcome: str, operation: object, arguments: object) -> None:
    if outcome == "propose":
        if not isinstance(operation, str) or not operation:
            raise MetricsError("a propose line needs an operation string")
        if not isinstance(arguments, dict):
            raise MetricsError("a propose line needs an arguments object")
    elif operation is not None or arguments is not None:
        raise MetricsError(f"operation and arguments must be null for outcome {outcome!r}")


def _check_candidates(candidates: object) -> None:
    if candidates is None:
        return
    if not isinstance(candidates, dict) or not candidates:
        raise MetricsError("candidates must be a non-empty object or null")
    for label, value in candidates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricsError(f"candidates[{label!r}] must be a number")
        if not 0.0 <= value <= 1.0:
            raise MetricsError(f"candidates[{label!r}] = {value} is outside [0, 1]")
    total = math.fsum(candidates.values())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise MetricsError(f"candidates sum to {total:.6f}, not 1")


def read_predictions(path: str | Path) -> list[Prediction]:
    """Read and validate a predictions file; blank lines are skipped."""
    predictions: list[Prediction] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                prediction = Prediction.from_dict(json.loads(line))
            except json.JSONDecodeError as exc:
                raise MetricsError(f"{path}: line {number}: not JSON ({exc.msg})") from None
            except MetricsError as exc:
                raise MetricsError(f"{path}: line {number}: {exc}") from None
            if prediction.id in seen:
                raise MetricsError(f"{path}: line {number}: duplicate id {prediction.id!r}")
            seen.add(prediction.id)
            predictions.append(prediction)
    return predictions


# ---------------------------------------------------------------------------
# Per-line classification
# ---------------------------------------------------------------------------


def expect_kind(expected: Mapping[str, object]) -> str:
    """``"escalate"``, ``"explain"`` or ``"operation"`` -- measure.py's three kinds."""
    if expected.get("escalate"):
        return "escalate"
    if expected.get("explain"):
        return "explain"
    return "operation"


def expected_label(expected: Mapping[str, object]) -> str:
    """The calibration label: bench's escalate/explain label, or the operation name."""
    kind = expect_kind(expected)
    if kind == "escalate":
        return tier_bench.ESCALATE_LABEL
    if kind == "explain":
        return tier_bench.EXPLAIN_LABEL
    return str(expected["operation"])


def invalid_reason(prediction: Prediction) -> str | None:
    """Why this line is an invalid output, or ``None`` when it is a valid decision."""
    if prediction.outcome == "invalid":
        return prediction.invalid_reason or DEFAULT_INVALID_REASON
    if prediction.outcome == "propose":
        error = ops_table.validate(prediction.operation, prediction.arguments)
        if error is not None:
            return error.code
    return None


def _proposed(prediction: Prediction) -> bool:
    """A valid tool call: a proposal the operation table accepts."""
    return prediction.outcome == "propose" and invalid_reason(prediction) is None


def _is_mutating_proposal(prediction: Prediction) -> bool:
    if not _proposed(prediction):
        return False
    operation = ops_table.get(prediction.operation)  # type: ignore[arg-type]
    return operation is not None and not operation.read_only


def _right_proposal(prediction: Prediction) -> bool:
    expected = prediction.expected
    return (
        _proposed(prediction)
        and prediction.operation == expected.get("operation")
        and prediction.arguments == expected.get("args", {})
    )


def _wrong_operation_mutating(prediction: Prediction) -> bool:
    """``bench._is_false_mutating``: a mutating operation where something else was expected."""
    return _is_mutating_proposal(prediction) and (
        prediction.operation != prediction.expected.get("operation")
    )


def _wrong_arguments_mutating(prediction: Prediction) -> bool:
    """measure.py's separate row: the expected mutating operation, wrong arguments."""
    return (
        _is_mutating_proposal(prediction)
        and prediction.operation == prediction.expected.get("operation")
        and not _right_proposal(prediction)
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def top_candidate(candidates: Mapping[str, float]) -> tuple[str, float]:
    """The highest-probability label; a tie goes to the label that sorts first."""
    label = min(candidates, key=lambda name: (-candidates[name], name))
    return label, float(candidates[label])


def bin_index(confidence: float, bins: int = ECE_BINS) -> int:
    """Equal-width bin of *confidence*: [0, 0.1) is 0, ..., [0.9, 1.0] is the last."""
    # Rounded first so a recorded 0.3 (0.30000000000000004 after * 10) cannot slip a bin.
    index = math.floor(round(confidence * bins, 9))
    return min(max(index, 0), bins - 1)


def ece_bins(pairs: Sequence[tuple[float, bool]], bins: int = ECE_BINS) -> list[dict]:
    """Per bin: count, mean confidence and accuracy (``None`` for an empty bin)."""
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in pairs:
        grouped[bin_index(confidence, bins)].append((confidence, correct))
    rows = []
    for index, group in enumerate(grouped):
        rows.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": len(group),
                "confidence": (math.fsum(c for c, _ in group) / len(group)) if group else None,
                "accuracy": (sum(1 for _, ok in group if ok) / len(group)) if group else None,
            }
        )
    return rows


def ece(pairs: Sequence[tuple[float, bool]], bins: int = ECE_BINS) -> float | None:
    """Expected calibration error: sum over bins of (n_b / N) * |accuracy_b - confidence_b|."""
    if not pairs:
        return None
    total = len(pairs)
    return math.fsum(
        row["n"] / total * abs(row["accuracy"] - row["confidence"])
        for row in ece_bins(pairs, bins)
        if row["n"]
    )


def brier_one(candidates: Mapping[str, float], gold: str) -> float:
    """Multi-class Brier for one line; a gold label absent from *candidates* counts as p = 0."""
    labels = set(candidates) | {gold}
    return math.fsum(
        (float(candidates.get(label, 0.0)) - (1.0 if label == gold else 0.0)) ** 2
        for label in labels
    )


def compute_calibration(predictions: Sequence[Prediction]) -> dict:
    with_distribution = [p for p in predictions if p.candidates is not None]
    pairs = []
    briers = []
    for prediction in with_distribution:
        gold = expected_label(prediction.expected)
        label, confidence = top_candidate(prediction.candidates)  # type: ignore[arg-type]
        pairs.append((confidence, label == gold))
        briers.append(brier_one(prediction.candidates, gold))  # type: ignore[arg-type]
    return {
        "n": len(with_distribution),
        "without_distribution": len(predictions) - len(with_distribution),
        "ece": ece(pairs),
        "brier": (math.fsum(briers) / len(briers)) if briers else None,
        "bins": ece_bins(pairs),
    }


# ---------------------------------------------------------------------------
# The whole file
# ---------------------------------------------------------------------------


def _ratio(n: int, total: int) -> float | None:
    return (n / total) if total else None


def _timing(values: Sequence[float]) -> dict:
    """bench.compute_latency's shape: the first value cold, the rest warm."""
    if not values:
        return {"cold_ms": None, "warm_median_ms": None, "warm_p95_ms": None}
    warm = list(values[1:])
    return {
        "cold_ms": values[0],
        "warm_median_ms": tier_bench._median(warm) if warm else None,
        "warm_p95_ms": tier_bench._percentile_95(sorted(warm)) if warm else None,
    }


def _median(values: Sequence[float]) -> float | None:
    return tier_bench._median(values) if values else None


def compute(predictions: Sequence[Prediction]) -> dict:
    """Every issue-46 metric for one predictions file, plus the issue-46 mapping."""
    kinds: dict[str, list[Prediction]] = {"operation": [], "escalate": [], "explain": []}
    for prediction in predictions:
        kinds[expect_kind(prediction.expected)].append(prediction)

    tp = sum(1 for p in kinds["escalate"] if p.outcome == "escalate")
    fn = len(kinds["escalate"]) - tp
    fp = sum(1 for p in kinds["operation"] if p.outcome == "escalate")
    recall = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)

    declines = kinds["explain"] + kinds["escalate"]
    right = sum(1 for p in kinds["operation"] if _right_proposal(p))
    false_calls = sum(1 for p in declines if _proposed(p))

    reasons: dict[str, int] = {}
    for prediction in predictions:
        reason = invalid_reason(prediction)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
    invalid = sum(reasons.values())

    wrong_operation = sum(1 for p in predictions if _wrong_operation_mutating(p))
    wrong_arguments = sum(1 for p in predictions if _wrong_arguments_mutating(p))

    tokens = [p.tokens for p in predictions]
    right_ratio = _ratio(right, len(kinds["operation"]))
    return {
        "right_proposals": {
            "n": right,
            "N": len(kinds["operation"]),
            "percent": None if right_ratio is None else 100 * right_ratio,
        },
        "escalation": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall": recall,
            "precision": precision,
            "escalated_on_explain": sum(1 for p in kinds["explain"] if p.outcome == "escalate"),
        },
        "abstention": {"recall": recall, "precision": precision},
        "false_positive_tool_calls": {
            "n": false_calls,
            "N": len(declines),
            "rate": _ratio(false_calls, len(declines)),
        },
        "wrong_mutating": {
            "wrong_operation": wrong_operation,
            "wrong_arguments": wrong_arguments,
            "total": wrong_operation + wrong_arguments,
        },
        "invalid": {
            "n": invalid,
            "N": len(predictions),
            "rate": _ratio(invalid, len(predictions)),
            "by_reason": dict(sorted(reasons.items())),
        },
        "calibration": compute_calibration(predictions),
        "tokens": {
            "total": sum(tokens),
            "mean": (sum(tokens) / len(tokens)) if tokens else None,
            "median": _median(tokens),
        },
        "time_to_first_decision": _timing([p.ttfd_ms for p in predictions]),
        "latency": _timing([p.latency_ms for p in predictions]),
        "issue46_mapping": issue46_mapping(),
    }


# ---------------------------------------------------------------------------
# The issue-46 mapping
# ---------------------------------------------------------------------------


def issue46_json(prediction: Prediction) -> dict:
    """One line as issue 46's decision JSON (see :data:`ISSUE46_MAPPING`)."""
    if invalid_reason(prediction) is not None:
        return {"action": "invalid"}
    if prediction.outcome == "propose":
        return {
            "action": "tool",
            "tool": prediction.operation,
            "arguments": dict(prediction.arguments or {}),
        }
    if prediction.outcome == "escalate":
        return {"action": "abstain"}
    return {"action": "no_action"}


def issue46_mapping() -> dict:
    """The fixed mapping, as rows and as a markdown table for the report."""
    lines = ["| nvsh outcome | issue 46 JSON |", "|---|---|"]
    lines += [f"| {row['nvsh']} | `{row['issue46']}` |" for row in ISSUE46_MAPPING]
    return {
        "rows": [dict(row) for row in ISSUE46_MAPPING],
        "markdown": "\n".join(lines),
        "note": ISSUE46_NOTE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("predictions", type=Path, help="predictions JSONL file")
    args = parser.parse_args(argv)
    try:
        result = compute(read_predictions(args.predictions))
    except (OSError, MetricsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
