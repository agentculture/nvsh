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
  separately as ``escalated_on_explain``. ``precision_strict`` also counts
  those as false escalations; issue 46's **abstention** precision is the
  strict one (deviation d2), so the c34 bar is judged on it.
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
* **Per-slice calibration** (issue 53): the same ECE/Brier/reliability bins,
  broken down by the *gold* label's slice -- ``read_only`` and ``mutating``
  (the operation table's ``read_only`` flag; never an operation name in this
  module's own code), with escalate/explain-expected entries forming their
  own ``escalate_or_explain`` slice. Each slice also reports its own offered
  candidate count (mean/median of ``len(candidates)``) and its own
  missing-candidate rate -- a line is missing-candidate when its id ends in
  ``-nocand`` (see ``eval_slices.py``) or its gold label isn't in the offered
  candidate set. :func:`reliability_markdown` renders one reliability-bin
  table per slice; ``measure.py``'s report page is the one that calls it.
* **Escalation reason roll-up**: a candidate label of the form
  ``escalate:<reason>`` (e.g. ``escalate:low_confidence``, a gate's specific
  decline reason) is folded into bench's bare ``(escalate)`` label wherever
  metrics group by outcome -- top-1 accuracy, Brier attribution, missing-
  candidate detection, per-slice calibration. The reason itself is preserved
  separately, as a tally of the top candidate's reason suffix, in
  ``escalation_reasons``.
* **abstain_uncertain**: a distinct prediction outcome (a confidence gate
  declining because it wasn't sure of the scorer, as opposed to a semantic
  escalate decision). It counts toward escalation recall/precision and the
  escalation bars exactly like ``escalate`` (``fp``/``fn``/``tp`` are the
  union of the two), but is tallied separately in ``escalation.
  abstain_uncertain`` and in the top-level ``outcome_counts`` so a report can
  still tell the two apart.
* **Bootstrap confidence intervals**: every rate (right proposals, escalation
  recall/precision/precision_strict, false-positive tool calls, invalid) and
  every ECE/Brier figure (top-level and per slice) carries its ``n`` and a
  seeded, percentile 95% bootstrap CI (``{"n", "value", "ci_low",
  "ci_high"}``), resampling the underlying lines with replacement. The seed
  and resample count are fixed defaults, both parameterised on
  :func:`compute`, and reported back under ``bootstrap``.
* **Tokens generated** per decision (total, mean, median) and **time to first
  decision** / **latency** as bench reports latency: the first line cold, the
  rest warm (median and nearest-rank p95).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

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
OUTCOMES = ("propose", "explain", "escalate", "abstain_uncertain", "invalid")

#: Equal-width confidence bins for ECE.
ECE_BINS = 10
#: How far a candidate distribution's sum may stray from 1 (JSON float round trip).
SUM_TOLERANCE = 1e-4
#: ``invalid_reason`` when an invalid line gives none.
DEFAULT_INVALID_REASON = "unparseable"

#: A candidate label ``escalate:<reason>`` rolls up to bench's bare escalate
#: label everywhere metrics group by outcome (see module docstring).
ESCALATE_REASON_PREFIX = "escalate:"
#: Bootstrap defaults for every rate/ECE/Brier confidence interval; both are
#: parameters on :func:`compute` too.
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
#: The per-slice breakdown of :func:`compute_slices`, in report order.
SLICE_NAMES = ("read_only", "mutating", "escalate_or_explain")

#: nvsh outcome -> issue 46's decision JSON, for the report only (decision c25:
#: models are trained and scored on nvsh's own tools; abstain is escalate).
ISSUE46_MAPPING = (
    {
        "nvsh": "propose",
        "issue46": '{"action": "tool", "tool": <operation>, "arguments": <arguments>}',
    },
    {"nvsh": "explain", "issue46": '{"action": "no_action"}'},
    {"nvsh": "escalate", "issue46": '{"action": "abstain"}'},
    {"nvsh": "abstain_uncertain", "issue46": '{"action": "abstain"}'},
    {"nvsh": "invalid", "issue46": '{"action": "invalid"}'},
)
ISSUE46_NOTE = (
    "Reporting only: every model is trained and scored on nvsh's propose/explain/escalate "
    "tools. Issue 46's abstain is nvsh's escalate, so abstention recall is the escalation "
    "recall and abstention precision the strict escalation precision (an escalation on an "
    "explain entry counts against it). Explain (answer in words, no tool) has no counterpart in "
    "issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for "
    "Track B and is never counted as an abstention. abstain_uncertain (issue 53: a confidence "
    "gate, not a semantic escalate decision) maps to the same issue-46 abstain action as "
    "escalate, and counts the same way in every escalation bar, but is tallied separately in "
    "nvsh's own escalation/outcome_counts. An invalid output is not a decision."
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


def _escalated(prediction: Prediction) -> bool:
    """True for either escalation outcome.

    ``escalate`` (a semantic decision) and ``abstain_uncertain`` (a confidence
    gate declining because it wasn't sure of the scorer) both count toward
    every escalation bar (tp/fn/fp, recall, precision); :func:`compute` also
    tallies ``abstain_uncertain`` on its own.
    """
    return prediction.outcome in ("escalate", "abstain_uncertain")


def _is_mutating_proposal(prediction: Prediction) -> bool:
    if not _proposed(prediction):
        return False
    operation = ops_table.get(prediction.operation)  # type: ignore[arg-type]
    return operation is not None and not operation.read_only


#: The unit suffix ``nvsh.ops.ground`` adds when it matches a service by name.
_UNIT_SUFFIX = ".service"


def _canonical_argument(name: str, value: object) -> object:
    """*value* as grounding compares it (issue 53, deviation d5).

    ``nvsh.ops.ground`` matches a service case-insensitively with the unit
    suffix optional ('rsyslog' finds 'rsyslog.service') and a container
    case-insensitively, so gold written either way names the same target.
    Every other argument compares exactly.
    """
    if not isinstance(value, str) or name not in ("service", "container"):
        return value
    value = value.casefold()
    if name == "service" and value.endswith(_UNIT_SUFFIX):
        value = value[: -len(_UNIT_SUFFIX)]
    return value


def _same_arguments(got: object, expected: object) -> bool:
    if not isinstance(got, dict) or not isinstance(expected, dict):
        return got == expected
    if got.keys() != expected.keys():
        return False
    return all(
        _canonical_argument(key, got[key]) == _canonical_argument(key, expected[key]) for key in got
    )


def _right_proposal(prediction: Prediction) -> bool:
    expected = prediction.expected
    return (
        _proposed(prediction)
        and prediction.operation == expected.get("operation")
        and _same_arguments(prediction.arguments, expected.get("args", {}))
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
# Escalation-reason roll-up (issue 53)
# ---------------------------------------------------------------------------


def _is_escalate_label(label: str) -> bool:
    """True for bench's bare ``(escalate)`` label or an ``escalate:<reason>`` one."""
    return label == tier_bench.ESCALATE_LABEL or label.startswith(ESCALATE_REASON_PREFIX)


def canonical_label(label: str) -> str:
    """*label*, or bench's bare escalate label when *label* is escalate-family."""
    return tier_bench.ESCALATE_LABEL if _is_escalate_label(label) else label


def escalate_reason(label: str) -> str | None:
    """The ``<reason>`` of an ``escalate:<reason>`` label, or ``None``."""
    if label.startswith(ESCALATE_REASON_PREFIX):
        return label[len(ESCALATE_REASON_PREFIX) :] or None
    return None


def rollup_escalate_candidates(candidates: Mapping[str, float]) -> dict[str, float]:
    """*candidates* with every escalate-family label's mass summed into one entry.

    A scorer that splits its escalate mass across several ``escalate:<reason>``
    labels should not be penalised (nor rewarded) relative to one that reports
    a single bare ``(escalate)``; every calibration figure compares against
    this rolled-up distribution.
    """
    rolled: dict[str, float] = {}
    for label, value in candidates.items():
        key = canonical_label(label)
        rolled[key] = rolled.get(key, 0.0) + value
    return rolled


def escalation_reason_counts(predictions: Sequence[Prediction]) -> dict[str, int]:
    """How often each ``escalate:<reason>`` label was a line's (raw) top candidate."""
    counts: dict[str, int] = {}
    for prediction in predictions:
        if prediction.candidates is None:
            continue
        label, _ = top_candidate(prediction.candidates)
        reason = escalate_reason(label)
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _resample(items: Sequence, rng: random.Random) -> list:
    n = len(items)
    return [items[rng.randrange(n)] for _ in range(n)]


def _mean_stat(values: Sequence[float]) -> float | None:
    return (math.fsum(values) / len(values)) if values else None


def bootstrap_ci(
    items: Sequence,
    stat_fn: Callable[[Sequence], float | None],
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """Percentile 95% CI of ``stat_fn(items)``, resampling *items* with replacement.

    ``n`` is ``len(items)``; a ``stat_fn`` that cannot be computed on a given
    resample (e.g. a ratio with a zero denominator) may return ``None`` and
    that resample is skipped. Seeded (``random.Random(seed)``) so a report is
    reproducible.
    """
    n = len(items)
    if n == 0:
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}
    value = stat_fn(items)
    if n == 1:
        return {"n": 1, "value": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)  # nosec B311 - bootstrap resampling, not security
    stats = []
    for _ in range(resamples):
        sample_stat = stat_fn(_resample(items, rng))
        if sample_stat is not None:
            stats.append(sample_stat)
    if not stats:
        return {"n": n, "value": value, "ci_low": None, "ci_high": None}
    stats.sort()
    lo = stats[int(round(0.025 * (len(stats) - 1)))]
    hi = stats[int(round(0.975 * (len(stats) - 1)))]
    return {"n": n, "value": value, "ci_low": lo, "ci_high": hi}


def _stratified_bootstrap_ci(
    strata: Mapping[str, Sequence],
    stat_fn: Callable[[Mapping[str, Sequence]], float | None],
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """Like :func:`bootstrap_ci`, but for a ratio drawn from more than one group.

    Each named stratum (e.g. escalate-expected vs operation-expected lines) is
    resampled independently, at its own size, every iteration; *stat_fn* sees
    the resampled strata and recomputes the ratio from them.
    """
    total_n = sum(len(items) for items in strata.values())
    value = stat_fn(strata)
    if total_n == 0:
        return {"n": 0, "value": value, "ci_low": None, "ci_high": None}
    if total_n == 1:
        return {"n": 1, "value": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)  # nosec B311 - bootstrap resampling, not security
    stats = []
    for _ in range(resamples):
        sample = {name: _resample(items, rng) for name, items in strata.items() if items}
        for name in strata:
            sample.setdefault(name, [])
        sample_stat = stat_fn(sample)
        if sample_stat is not None:
            stats.append(sample_stat)
    if not stats:
        return {"n": total_n, "value": value, "ci_low": None, "ci_high": None}
    stats.sort()
    lo = stats[int(round(0.025 * (len(stats) - 1)))]
    hi = stats[int(round(0.975 * (len(stats) - 1)))]
    return {"n": total_n, "value": value, "ci_low": lo, "ci_high": hi}


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


def compute_calibration(
    predictions: Sequence[Prediction],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """ECE/Brier/bins over *predictions*, each carrying a seeded bootstrap CI.

    The top candidate is taken from :func:`rollup_escalate_candidates`'
    output, so an ``escalate:<reason>`` split doesn't change accuracy or
    Brier relative to a bare ``(escalate)``.
    """
    with_distribution = [p for p in predictions if p.candidates is not None]
    pairs = []
    briers = []
    for prediction in with_distribution:
        gold = expected_label(prediction.expected)
        rolled = rollup_escalate_candidates(prediction.candidates)  # type: ignore[arg-type]
        label, confidence = top_candidate(rolled)
        pairs.append((confidence, label == gold))
        briers.append(brier_one(rolled, gold))
    return {
        "n": len(with_distribution),
        "without_distribution": len(predictions) - len(with_distribution),
        "ece": ece(pairs),
        "ece_ci": bootstrap_ci(pairs, ece, seed, resamples),
        "brier": (math.fsum(briers) / len(briers)) if briers else None,
        "brier_ci": bootstrap_ci(briers, _mean_stat, seed, resamples),
        "bins": ece_bins(pairs),
    }


# ---------------------------------------------------------------------------
# Slices (issue 53): read-only vs mutating vs escalate/explain
# ---------------------------------------------------------------------------


def slice_name(prediction: Prediction) -> str:
    """Which of :data:`SLICE_NAMES` a prediction's GOLD label belongs to.

    Decided from the operation table's ``read_only`` flag, never a name;
    an escalate- or explain-expected entry forms its own slice regardless of
    what was predicted. A gold operation absent from the table (should not
    happen for a real corpus) is conservatively bucketed as ``mutating``.
    """
    if expect_kind(prediction.expected) != "operation":
        return "escalate_or_explain"
    operation = ops_table.get(str(prediction.expected["operation"]))
    if operation is None:
        return "mutating"
    return "read_only" if operation.read_only else "mutating"


def is_missing_candidate(prediction: Prediction) -> bool:
    """A missing-candidate line: id ends ``-nocand``, or gold isn't offered.

    ``eval_slices.py`` builds explicit ``-nocand`` entries (candidates =
    every operation but the gold one, ``expect`` rewritten to escalate); this
    also catches any other line whose offered candidates never include the
    gold label, after the escalate-reason roll-up.
    """
    if prediction.id.endswith("-nocand"):
        return True
    if prediction.candidates is None:
        return False
    gold = canonical_label(expected_label(prediction.expected))
    offered = {canonical_label(label) for label in prediction.candidates}
    return gold not in offered


def _candidate_count_stats(predictions: Sequence[Prediction]) -> dict:
    counts = [len(p.candidates) for p in predictions if p.candidates is not None]
    return {
        "n": len(counts),
        "mean": (sum(counts) / len(counts)) if counts else None,
        "median": _median(counts),
    }


def compute_slice(
    predictions: Sequence[Prediction],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """One slice's calibration, offered-candidate count and missing-candidate rate."""
    missing = [1 if is_missing_candidate(p) else 0 for p in predictions]
    return {
        "n": len(predictions),
        "calibration": compute_calibration(predictions, seed=seed, resamples=resamples),
        "candidate_count": _candidate_count_stats(predictions),
        "missing_candidate": {
            "n": sum(missing),
            "N": len(predictions),
            "rate": bootstrap_ci(missing, _mean_stat, seed, resamples),
        },
    }


def compute_slices(
    predictions: Sequence[Prediction],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """:data:`SLICE_NAMES` -> :func:`compute_slice`, bucketed by :func:`slice_name`."""
    buckets: dict[str, list[Prediction]] = {name: [] for name in SLICE_NAMES}
    for prediction in predictions:
        buckets[slice_name(prediction)].append(prediction)
    return {
        name: compute_slice(items, seed=seed, resamples=resamples)
        for name, items in buckets.items()
    }


def reliability_markdown(slices: Mapping[str, dict]) -> str:
    """One markdown reliability-bin table per slice, from :func:`compute_slices`'s output.

    Not called anywhere in this module -- ``measure.py``'s report page is the
    caller (issue 53, task t8).
    """
    tables = []
    for name in SLICE_NAMES:
        data = slices.get(name)
        if data is None:
            continue
        bins = data["calibration"]["bins"]
        lines = [
            f"### {name}",
            "",
            "| bin | n | confidence | accuracy |",
            "| --- | --- | --- | --- |",
        ]
        for row in bins:
            confidence = "-" if row["confidence"] is None else f"{row['confidence']:.3f}"
            accuracy = "-" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
            lines.append(
                f"| [{row['lower']:.1f}, {row['upper']:.1f}) | {row['n']} | "
                f"{confidence} | {accuracy} |"
            )
        tables.append("\n".join(lines))
    return "\n\n".join(tables)


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


def compute(
    predictions: Sequence[Prediction],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict:
    """Every issue-46 metric for one predictions file, plus the issue-46 mapping.

    ``bootstrap_seed``/``bootstrap_resamples`` (issue 53) control every rate
    and ECE/Brier confidence interval in the result; both default to fixed
    constants so a report is reproducible without passing them.
    """
    seed, resamples = bootstrap_seed, bootstrap_resamples
    kinds: dict[str, list[Prediction]] = {"operation": [], "escalate": [], "explain": []}
    for prediction in predictions:
        kinds[expect_kind(prediction.expected)].append(prediction)

    tp = sum(1 for p in kinds["escalate"] if _escalated(p))
    fn = len(kinds["escalate"]) - tp
    fp = sum(1 for p in kinds["operation"] if _escalated(p))
    recall = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    # Deviation d2: an escalation on an explain entry is a false abstention too.
    on_explain = sum(1 for p in kinds["explain"] if _escalated(p))
    precision_strict = _ratio(tp, tp + fp + on_explain)

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

    def _precision_stat(strata: Mapping[str, Sequence[Prediction]]) -> float | None:
        t = sum(1 for p in strata["escalate"] if _escalated(p))
        f = sum(1 for p in strata["operation"] if _escalated(p))
        denom = t + f
        return (t / denom) if denom else None

    def _precision_strict_stat(strata: Mapping[str, Sequence[Prediction]]) -> float | None:
        t = sum(1 for p in strata["escalate"] if _escalated(p))
        f = sum(1 for p in strata["operation"] if _escalated(p))
        e = sum(1 for p in strata["explain"] if _escalated(p))
        denom = t + f + e
        return (t / denom) if denom else None

    abstain_uncertain = {
        "tp": sum(1 for p in kinds["escalate"] if p.outcome == "abstain_uncertain"),
        "fp": sum(1 for p in kinds["operation"] if p.outcome == "abstain_uncertain"),
        "escalated_on_explain": sum(
            1 for p in kinds["explain"] if p.outcome == "abstain_uncertain"
        ),
    }
    outcome_counts = {
        outcome: sum(1 for p in predictions if p.outcome == outcome) for outcome in OUTCOMES
    }

    return {
        "right_proposals": {
            "n": right,
            "N": len(kinds["operation"]),
            "percent": None if right_ratio is None else 100 * right_ratio,
            "ci": bootstrap_ci(
                [1 if _right_proposal(p) else 0 for p in kinds["operation"]],
                _mean_stat,
                seed,
                resamples,
            ),
        },
        "escalation": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall": recall,
            "recall_ci": bootstrap_ci(
                [1 if _escalated(p) else 0 for p in kinds["escalate"]], _mean_stat, seed, resamples
            ),
            "precision": precision,
            "precision_ci": _stratified_bootstrap_ci(
                {"escalate": kinds["escalate"], "operation": kinds["operation"]},
                _precision_stat,
                seed,
                resamples,
            ),
            "precision_strict": precision_strict,
            "precision_strict_ci": _stratified_bootstrap_ci(
                {
                    "escalate": kinds["escalate"],
                    "operation": kinds["operation"],
                    "explain": kinds["explain"],
                },
                _precision_strict_stat,
                seed,
                resamples,
            ),
            "escalated_on_explain": on_explain,
            # abstain_uncertain (issue 53) counts toward tp/fn/fp above like escalate,
            # but is broken out here so a report can tell the two decisions apart.
            "abstain_uncertain": abstain_uncertain,
        },
        "abstention": {"recall": recall, "precision": precision_strict},
        "false_positive_tool_calls": {
            "n": false_calls,
            "N": len(declines),
            "rate": _ratio(false_calls, len(declines)),
            "ci": bootstrap_ci(
                [1 if _proposed(p) else 0 for p in declines], _mean_stat, seed, resamples
            ),
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
            "ci": bootstrap_ci(
                [1 if invalid_reason(p) is not None else 0 for p in predictions],
                _mean_stat,
                seed,
                resamples,
            ),
            "by_reason": dict(sorted(reasons.items())),
        },
        "calibration": compute_calibration(predictions, seed=seed, resamples=resamples),
        "outcome_counts": outcome_counts,
        "escalation_reasons": escalation_reason_counts(predictions),
        "slices": compute_slices(predictions, seed=seed, resamples=resamples),
        "bootstrap": {"seed": seed, "resamples": resamples},
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
    if _escalated(prediction):
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
