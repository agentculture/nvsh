"""Benchmark corpus and runner: drives the REAL router so it measures what ships.

Task t22. ``nvsh/tiers/corpus/dev.json`` and ``held-out.json`` hold the
prompts and failure cases from issues #30/#31 and challenge probes s13/s16
(:func:`load_corpus`, :func:`dev_corpus_path`, :func:`held_out_corpus_path`);
:func:`bench` runs them through a real :class:`~nvsh.tiers.router.TierRouter`
-- built from whatever ``tier1``/``tier2`` the caller supplies, scripted or
real -- and reports accuracy, argument accuracy, a false-mutating-pick
count, escalation precision/recall, cold/warm latency, memory, image size
and a pass/miss line per success-signal target (spec c20).

Nothing here executes anything: a :class:`~nvsh.tiers.base.Tier` never does,
and the router's own chain (``decide()`` -> ``ground()`` -> ``render()``) is
unchanged. Records written during a run go to a throwaway
:class:`~nvsh.tiers.records.TierRecords` in a temp directory, never the
operator's real log.

Per assumption c38, held-out phrasings must not be authored alongside the
operation descriptions in the same sitting: ``held-out.json`` ships with a
header explaining that and an empty ``entries`` list, so a run against it
reports "0 entries" rather than a fabricated score.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..agent.base import AgentContext, AgentRequest, RequestKind
from ..ops import ground as ops_ground
from ..ops import table as ops_table
from ..platform._model import PATH, Platform, Value
from .base import Decline, DeclineReason, Tier
from .records import TierRecords
from .router import TierOutcome, TierRouter, Verifier

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
DEV_SPLIT = "dev"
HELD_OUT_SPLIT = "held-out"

#: c20's numeric targets. Kept here, not hard-coded into ``build_targets``'s
#: body, so a future spec revision changes one place.
_WARM_P95_MS = 150.0
_MIN_ACCURACY = 0.9
_MIN_ESCALATION_RECALL = 0.8
_MAX_ADDED_MIB = 1024.0
#: Tolerance for the "zero" targets, which are always integer counts.
_ZERO_COUNT_TOLERANCE = 0.5


def dev_corpus_path() -> Path:
    """``nvsh/tiers/corpus/dev.json``, tuned against (assumption c38)."""
    return CORPUS_DIR / "dev.json"


def held_out_corpus_path() -> Path:
    """``nvsh/tiers/corpus/held-out.json`` -- ships empty; see module docstring."""
    return CORPUS_DIR / "held-out.json"


ESCALATE_LABEL = "(escalate)"
NO_CLASS_LABEL = "(none)"

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusEntry:
    """One benchmark prompt: an expected operation+arguments, or an escalation.

    Keyed by operation NAME only, as data -- nothing in this module ever
    switches on a specific operation name; the table in ``nvsh.ops`` is the
    only place operations are named.
    """

    id: str
    kind: str  # "explicit" | "failure"
    text: str
    expect: dict  # {"operation": name, "args": {...}} or {"escalate": True}
    source: str
    #: How the request is phrased ("imperative", "question", "symptom",
    #: "terse", "jargon", ...). Free text, as data: the grid in
    #: docs/tiers-improving-accuracy.md names the classes, no code does.
    phrasing: str = ""


@dataclass(frozen=True)
class CorpusLoadResult:
    """What :func:`load_corpus` found: valid entries plus reported problems."""

    entries: tuple[CorpusEntry, ...]
    problems: tuple[str, ...]
    header: str | None = None


def load_corpus(path: str | Path) -> CorpusLoadResult:
    """Load and validate a corpus file. Never raises for a bad entry.

    Every entry's expected operation/args is checked against
    :func:`nvsh.ops.table.validate`; a bad entry is reported in
    ``problems`` and skipped, so the corpus cannot silently rot when the
    operation table changes.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    header = raw.get("header") if isinstance(raw, dict) else None
    raw_entries = raw.get("entries", []) if isinstance(raw, dict) else raw
    if not isinstance(raw_entries, list):
        raw_entries = []

    entries: list[CorpusEntry] = []
    problems: list[str] = []
    for index, item in enumerate(raw_entries):
        parsed = _parse_entry(index, item)
        if isinstance(parsed, str):
            problems.append(parsed)
            continue
        error = _validate_expect(parsed)
        if error is not None:
            problems.append(error)
            continue
        entries.append(parsed)
    return CorpusLoadResult(entries=tuple(entries), problems=tuple(problems), header=header)


def _parse_entry(index: int, item: object) -> CorpusEntry | str:
    if not isinstance(item, dict):
        return f"entry[{index}]: not an object"
    try:
        entry_id = str(item["id"])
        kind = str(item["kind"])
        text = str(item["text"])
        expect = item["expect"]
    except KeyError as exc:
        return f"entry[{index}]: missing field {exc}"
    source = str(item.get("source", ""))
    if kind not in ("explicit", "failure"):
        return f"{entry_id}: unknown kind {kind!r}"
    if not isinstance(expect, dict):
        return f"{entry_id}: expect must be an object"
    phrasing = str(item.get("class", ""))
    return CorpusEntry(
        id=entry_id, kind=kind, text=text, expect=expect, source=source, phrasing=phrasing
    )


def _validate_expect(entry: CorpusEntry) -> str | None:
    if entry.expect.get("escalate") is True:
        return None
    operation = entry.expect.get("operation")
    args = entry.expect.get("args", {})
    error = ops_table.validate(operation, args)
    if error is not None:
        return f"{entry.id}: expect {error.message}"
    return None


# ---------------------------------------------------------------------------
# Driving the real router
# ---------------------------------------------------------------------------


class UnavailableTier(Tier):
    """The CLI's ``--tier fixture`` default: declines every request.

    Lets ``nvsh tiers bench`` run end-to-end -- corpus loading, routing, a
    results file -- with no local model installed. A meaningful accuracy
    run needs a real tier (``--tier needle``) or a scripted one built
    through the Python API (see ``tests/test_tier_bench.py``), not this.
    """

    name = "fixture"

    def select(self, request: AgentRequest, context: AgentContext):
        return Decline(DeclineReason.TIER_UNAVAILABLE, "fixture tier: no local model configured")

    def close(self) -> None:
        # Nothing to release: this tier holds no connection or subprocess.
        pass


@dataclass(frozen=True)
class ItemResult:
    """One corpus entry's routed outcome plus this run's own latency reading."""

    entry: CorpusEntry
    outcome: TierOutcome | None
    latency_ms: float


def request_for(entry: CorpusEntry) -> AgentRequest:
    if entry.kind == "failure":
        return AgentRequest(kind=RequestKind.FAILURE, command=entry.text, exit_code=1)
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt=entry.text)


def context_for(entry: CorpusEntry) -> AgentContext:
    return AgentContext(output=entry.text if entry.kind == "failure" else "")


def run_items(
    entries: Sequence[CorpusEntry], router: TierRouter, clock: Callable[[], float]
) -> list[ItemResult]:
    """Route every entry through *router*, draining each :class:`Route`.

    ``clock`` is this run's own latency instrument (independent of whatever
    clock ``router`` itself was built with), so the caller can inject a
    deterministic one for a repeatable test.
    """
    results: list[ItemResult] = []
    for entry in entries:
        request = request_for(entry)
        context = context_for(entry)
        started = clock()
        route = router.route(request, context)
        for _event in route:
            # drain: a Route is a generator; exhausting it drives the decision to completion
            pass
        elapsed_ms = (clock() - started) * 1000.0
        results.append(ItemResult(entry=entry, outcome=route.outcome, latency_ms=elapsed_ms))
    return results


def load_world(path: str | Path) -> dict:
    """The corpus file's ``world`` object, or ``{}`` (never raises)."""
    try:
        with open(path, encoding="utf-8") as handle:
            world = json.load(handle).get("world")
    except (OSError, ValueError, AttributeError):
        return {}
    return world if isinstance(world, dict) else {}


def _names(world: Mapping[str, object], key: str) -> list[str]:
    raw = world.get(key)
    return [str(name) for name in raw] if isinstance(raw, list) else []


def world_runner(world: Mapping[str, object]) -> Callable[[list[str], float], tuple[int, str]]:
    """A grounding runner that answers from the corpus's fixture world.

    It runs nothing: the unit and container lookups get the world's names,
    anything else is "not found". Scores then measure the tier, not
    whichever services happen to exist on the host running the bench.
    """
    services = "".join(f"{name} loaded active running\n" for name in _names(world, "services"))
    containers = "".join(f"{name}\n" for name in _names(world, "containers"))

    def runner(argv: list[str], timeout: float) -> tuple[int, str]:
        del timeout
        if argv == ops_ground.SERVICE_LOOKUP_ARGV:
            return (0, services)
        if argv == ops_ground.CONTAINER_LOOKUP_ARGV:
            return (0, containers)
        return (127, "")

    return runner


def world_platform(world: Mapping[str, object]) -> Platform:
    """The fixture machine the corpus assumes: its kind and its device CLI."""
    kind = str(world.get("platform") or "unknown")
    cli = world.get("device_cli")
    if not isinstance(cli, str) or not cli:
        return Platform(kind=kind)
    value = Value(name=f"{cli}_cli", text=cli, source=cli, method=PATH, present=True)
    return Platform(kind=kind, values=(value,))


def item_rows(items: Sequence[ItemResult]) -> list[dict]:
    """One row per corpus entry: what was expected and what the tiers did.

    The aggregate scores say how good a tier is; these rows say *which*
    requests it got wrong and why (the decline reasons), which is what a
    corpus fix, a description rewrite or a fine-tune is built from.
    """
    rows = []
    for item in items:
        outcome = item.outcome
        rows.append(
            {
                "id": item.entry.id,
                "kind": item.entry.kind,
                "expect": item.entry.expect,
                "handled_by": outcome.handled_by if outcome else None,
                "operation": outcome.operation if outcome else None,
                "args": dict(outcome.args) if outcome else {},
                "declines": (
                    [[tier, reason.value] for tier, reason in outcome.declines] if outcome else []
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Accuracy / escalation / false-mutating-pick metrics
# ---------------------------------------------------------------------------


def _explicit_target_items(items: Sequence[ItemResult]) -> list[ItemResult]:
    """Items whose ``expect`` names an operation, not an escalation."""
    return [item for item in items if not item.entry.expect.get("escalate")]


def accuracy_by_kind(items: Sequence[ItemResult]) -> dict:
    """:func:`compute_operation_accuracy` per request kind.

    A failure never reaches Tier 1, so a Tier-1-only run scores zero on every
    failure item by design. The overall figure is the c20 target; this split
    is what shows which tier is responsible for a miss.
    """
    kinds = sorted({item.entry.kind for item in items})
    return {
        kind: compute_operation_accuracy([item for item in items if item.entry.kind == kind])
        for kind in kinds
    }


def _expected_label(entry: CorpusEntry) -> str:
    """The expected operation's name, or ``"(escalate)"`` for a should-decline."""
    if entry.expect.get("escalate"):
        return ESCALATE_LABEL
    return str(entry.expect.get("operation"))


def _is_correct(item: ItemResult) -> bool:
    """Right operation and arguments, or an escalation where one was expected."""
    outcome = item.outcome
    if item.entry.expect.get("escalate"):
        return outcome is not None and outcome.escalated_to is not None
    if outcome is None or outcome.operation != item.entry.expect.get("operation"):
        return False
    return outcome.args == item.entry.expect.get("args", {})


def _breakdown(items: Sequence[ItemResult], label: Callable[[CorpusEntry], str]) -> dict:
    """Correct/total per label, with the ids that missed -- what to write next."""
    groups: dict[str, list[ItemResult]] = {}
    for item in items:
        groups.setdefault(label(item.entry), []).append(item)
    return {
        name: {
            "total": len(group),
            "correct": sum(1 for item in group if _is_correct(item)),
            "missed": [item.entry.id for item in group if not _is_correct(item)],
        }
        for name, group in sorted(groups.items())
    }


def accuracy_by_operation(items: Sequence[ItemResult]) -> dict:
    """Per expected operation (should-decline entries under one label)."""
    return _breakdown(items, _expected_label)


def accuracy_by_class(items: Sequence[ItemResult]) -> dict:
    """Per phrasing class; entries without one are grouped as ``"(none)"``."""
    return _breakdown(items, lambda entry: entry.phrasing or NO_CLASS_LABEL)


def compute_operation_accuracy(items: Sequence[ItemResult]) -> dict:
    """Correct-operation and correct-operation-and-arguments rates.

    Covers both ``explicit`` and ``failure`` items with a named expected
    operation -- a failure case answered by Tier 2 (or escalated straight
    through when Tier 2 is absent) is measured the same way.
    """
    targets = _explicit_target_items(items)
    total = len(targets)
    op_correct = 0
    args_correct = 0
    for item in targets:
        outcome = item.outcome
        if outcome is None or outcome.operation != item.entry.expect.get("operation"):
            continue
        op_correct += 1
        if outcome.args == item.entry.expect.get("args", {}):
            args_correct += 1
    return {
        "total": total,
        "operation_correct": op_correct,
        "argument_correct": args_correct,
        "accuracy": (op_correct / total) if total else None,
        "argument_accuracy": (args_correct / total) if total else None,
    }


def _is_false_mutating(entry: CorpusEntry, outcome: TierOutcome | None) -> bool:
    """A mutating operation proposed where the expectation was different."""
    if outcome is None or outcome.operation is None:
        return False
    operation = ops_table.get(outcome.operation)
    if operation is None or operation.read_only:
        return False
    expected_escalate = bool(entry.expect.get("escalate"))
    expected_operation = entry.expect.get("operation")
    return expected_escalate or expected_operation != outcome.operation


def _shows_interpretation(outcome: TierOutcome | None) -> bool:
    if outcome is None or outcome.proposal is None or outcome.operation is None:
        return False
    return outcome.operation in outcome.proposal.rationale


def compute_false_mutating(items: Sequence[ItemResult]) -> dict:
    """False-mutating picks, and whether every one of them showed its
    interpreted operation (the router always builds the rationale this way;
    this is the honesty check, not a trust assumption)."""
    flagged = [item for item in items if _is_false_mutating(item.entry, item.outcome)]
    without_interpretation = [
        item.entry.id for item in flagged if not _shows_interpretation(item.outcome)
    ]
    return {
        "count": len(flagged),
        "ids": [item.entry.id for item in flagged],
        "without_interpretation_shown": without_interpretation,
    }


def compute_escalation(items: Sequence[ItemResult]) -> dict:
    """Precision/recall of "this request should escalate" over the whole corpus."""
    tp = fp = fn = tn = 0
    for item in items:
        expected = bool(item.entry.expect.get("escalate"))
        got = item.outcome is not None and item.outcome.escalated_to is not None
        if expected and got:
            tp += 1
        elif expected and not got:
            fn += 1
        elif got:
            fp += 1
        else:
            tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / (tp + fn)) if (tp + fn) else None,
    }


# ---------------------------------------------------------------------------
# Deviation d1: verifier calibration metrics
# ---------------------------------------------------------------------------


def _pick_is_correct(entry: CorpusEntry, outcome: TierOutcome) -> bool:
    """Same "correct" as :func:`compute_operation_accuracy`: operation AND args match.

    Matching the operation name alone would count a pick that targets the
    wrong service or container as a positive calibration/threshold sample,
    even though the main accuracy score correctly marks it wrong.
    """
    if entry.expect.get("escalate"):
        return False
    if outcome.operation != entry.expect.get("operation"):
        return False
    return outcome.args == entry.expect.get("args", {})


def _calibration_samples(items: Sequence[ItemResult]) -> list[tuple[float, bool]]:
    samples: list[tuple[float, bool]] = []
    for item in items:
        outcome = item.outcome
        if outcome is None or outcome.verifier is None or outcome.verifier.calibrated is None:
            continue
        samples.append((outcome.verifier.calibrated, _pick_is_correct(item.entry, outcome)))
    return samples


def _average_ranks(sorted_values: Sequence[float]) -> list[float]:
    """Average (tie-safe) 1-based ranks over an already-sorted sequence."""
    n = len(sorted_values)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    return ranks


def _auc(samples: Sequence[tuple[float, bool]]) -> float | None:
    """Rank-based (Mann-Whitney U) AUC, tie-safe. ``None`` with one class only."""
    positives = sum(1 for _, correct in samples if correct)
    negatives = len(samples) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(samples, key=lambda pair: pair[0])
    ranks = _average_ranks([score for score, _ in ordered])
    rank_sum_positive = sum(rank for (_, correct), rank in zip(ordered, ranks) if correct)
    u_statistic = rank_sum_positive - positives * (positives + 1) / 2.0
    return u_statistic / (positives * negatives)


def compute_calibration(items: Sequence[ItemResult]) -> dict:
    """AUC of the verifier's calibrated score against "was the pick correct".

    Comparison hook (d1): the signal being measured is whichever
    :class:`~nvsh.tiers.router.Verifier` the caller wired in --
    :class:`~nvsh.tiers.router.LogprobVerifier`'s yes/no read today, a raw
    Needle-confidence verifier or a multiple-choice one tomorrow -- since
    ``compute_calibration`` only reads ``outcome.verifier.calibrated``, not
    any verifier internals. A ``Verifier`` protocol instance is the
    extension point; no multiple-choice implementation ships here.
    """
    samples = _calibration_samples(items)
    if len(samples) < 2:
        return {"samples": len(samples), "auc": None, "note": "not enough scored picks"}
    return {"samples": len(samples), "auc": _auc(samples), "note": ""}


def _rates_at(samples: Sequence[tuple[float, bool]], cutoff: float) -> tuple[float, float]:
    """(true-positive rate, false-positive rate) for "decline below cutoff",
    where a true positive is a wrong pick correctly flagged for decline."""
    tp = sum(1 for score, correct in samples if not correct and score < cutoff)
    fn = sum(1 for score, correct in samples if not correct and score >= cutoff)
    fp = sum(1 for score, correct in samples if correct and score < cutoff)
    tn = sum(1 for score, correct in samples if correct and score >= cutoff)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr, fpr


def suggest_thresholds(items: Sequence[ItemResult]) -> dict:
    """A dev-split-only threshold sweep suggesting ``escalate_below``/``ask_below``.

    Sweeps every distinct calibrated score seen in *items* as a candidate
    cutoff and picks the one maximising Youden's J (tpr - fpr) for
    "decline a wrong pick" as ``escalate_below``; ``ask_below`` is one step
    looser, so a pick that scores between the two still reaches the
    operator as an uncertain proposal instead of being escalated outright.
    A suggestion only -- the router's own thresholds are unmeasured
    placeholders until an operator adopts one.
    """
    samples = _calibration_samples(items)
    if len(samples) < 2:
        return {
            "escalate_below": None,
            "ask_below": None,
            "youden_j": None,
            "note": "not enough dev samples",
        }
    candidates = sorted({score for score, _ in samples})
    best_j, best_cutoff = None, candidates[0]
    for cutoff in candidates:
        tpr, fpr = _rates_at(samples, cutoff)
        j_statistic = tpr - fpr
        if best_j is None or j_statistic > best_j:
            best_j, best_cutoff = j_statistic, cutoff
    index = candidates.index(best_cutoff)
    ask_below = candidates[index + 1] if index + 1 < len(candidates) else best_cutoff
    return {
        "escalate_below": best_cutoff,
        "ask_below": ask_below,
        "youden_j": best_j,
        "note": "dev-split suggestion; not adopted automatically",
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile_95(sorted_values: Sequence[float]) -> float:
    """Nearest-rank p95 over an already-sorted sequence."""
    n = len(sorted_values)
    rank = max(1, min(n, -(-95 * n // 100)))  # ceil(95/100 * n), clamped
    return float(sorted_values[rank - 1])


def compute_latency(items: Sequence[ItemResult]) -> dict:
    """Cold (first call) vs warm (median / p95 of the rest) latency."""
    if not items:
        return {"cold_ms": None, "warm_median_ms": None, "warm_p95_ms": None}
    cold = items[0].latency_ms
    warm = [item.latency_ms for item in items[1:]]
    if not warm:
        return {"cold_ms": cold, "warm_median_ms": None, "warm_p95_ms": None}
    return {
        "cold_ms": cold,
        "warm_median_ms": _median(warm),
        "warm_p95_ms": _percentile_95(sorted(warm)),
    }


# ---------------------------------------------------------------------------
# Memory: /proc/<pid>/status and docker stats, both injectable and optional
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryReading:
    """One point-in-time memory reading. Every field is optional."""

    vm_rss_kb: int | None = None
    vm_hwm_kb: int | None = None
    docker_used_mib: float | None = None
    docker_limit_mib: float | None = None


def _default_proc_status_text(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/status")
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _parse_kb_field(line: str) -> int | None:
    parts = line.split()
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def read_proc_status(
    pid: int, *, read_text: Callable[[int], str | None] | None = None
) -> MemoryReading | None:
    """``VmRSS``/``VmHWM`` from ``/proc/<pid>/status``. Never raises; ``None`` on any problem."""
    read_text = read_text if read_text is not None else _default_proc_status_text
    try:
        text = read_text(pid)
    except Exception:  # noqa: BLE001 -- a measurement must never break the bench
        return None
    if not text:
        return None
    rss = hwm = None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            rss = _parse_kb_field(line)
        elif line.startswith("VmHWM:"):
            hwm = _parse_kb_field(line)
    if rss is None and hwm is None:
        return None
    return MemoryReading(vm_rss_kb=rss, vm_hwm_kb=hwm)


_MEM_UNIT_TO_MIB = {"b": 1.0 / (1024 * 1024), "kib": 1.0 / 1024, "mib": 1.0, "gib": 1024.0}


def _split_number_unit(token: str) -> tuple[float, str] | None:
    """Split ``"12.34MiB"`` into ``(12.34, "MiB")``. No regex: a hand-rolled
    scan avoids the super-linear backtracking a ``[\\d.]+\\s*[A-Za-z]+``
    pattern risks on adversarial input (S8786)."""
    token = token.strip()
    index = 0
    length = len(token)
    while index < length and (token[index].isdigit() or token[index] == "."):
        index += 1
    number_part, unit_part = token[:index], token[index:].strip()
    if not number_part or not unit_part:
        return None
    try:
        return float(number_part), unit_part
    except ValueError:
        return None


def _parse_docker_mem_usage(raw: str) -> tuple[float, float] | None:
    parts = raw.strip().split("/")
    if len(parts) != 2:
        return None
    used = _split_number_unit(parts[0])
    limit = _split_number_unit(parts[1])
    if used is None or limit is None:
        return None
    used_value, used_unit = used
    limit_value, limit_unit = limit
    used_factor = _MEM_UNIT_TO_MIB.get(used_unit.lower())
    limit_factor = _MEM_UNIT_TO_MIB.get(limit_unit.lower())
    if used_factor is None or limit_factor is None:
        return None
    return used_value * used_factor, limit_value * limit_factor


def _default_docker_stats(container: str) -> str | None:
    import subprocess  # nosec B404 - fixed argv, no shell=True

    completed = subprocess.run(  # nosec B603 B607 - fixed argv list, docker on PATH by design
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def read_docker_stats(
    container: str, *, run_stats: Callable[[str], str | None] | None = None
) -> MemoryReading | None:
    """``docker stats --no-stream`` memory usage for *container*.

    Never raises; ``None`` for a missing container, a missing ``docker``, or
    output that does not parse as ``used / limit``.
    """
    run_stats = run_stats if run_stats is not None else _default_docker_stats
    try:
        raw = run_stats(container)
    except Exception:  # noqa: BLE001 -- a measurement must never break the bench
        return None
    if not raw:
        return None
    parsed = _parse_docker_mem_usage(raw)
    if parsed is None:
        return None
    used_mib, limit_mib = parsed
    return MemoryReading(docker_used_mib=used_mib, docker_limit_mib=limit_mib)


def _reading_mib(reading: MemoryReading | None) -> float | None:
    if reading is None:
        return None
    if reading.vm_rss_kb is not None:
        return reading.vm_rss_kb / 1024.0
    return reading.docker_used_mib


def _reserved_mib(reading: MemoryReading | None) -> float | None:
    if reading is None:
        return None
    if reading.vm_hwm_kb is not None:
        return reading.vm_hwm_kb / 1024.0
    return reading.docker_limit_mib


def compute_memory(idle: MemoryReading | None, peak: MemoryReading | None) -> dict:
    """Idle/peak/reserved/added memory from two optional readings. Never ``None`` -> crash."""
    idle_mib = _reading_mib(idle)
    peak_mib = _reading_mib(peak)
    added_mib = (
        max(peak_mib - idle_mib, 0.0) if idle_mib is not None and peak_mib is not None else None
    )
    return {
        "idle_mib": idle_mib,
        "peak_mib": peak_mib,
        "reserved_mib": _reserved_mib(peak) if peak is not None else _reserved_mib(idle),
        "added_mib": added_mib,
    }


# ---------------------------------------------------------------------------
# Pass/miss per c20 target
# ---------------------------------------------------------------------------


def _check(
    name: str, value: float | None, predicate: Callable[[float], bool], requirement: str
) -> dict:
    if value is None:
        return {"target": name, "requirement": requirement, "status": "not measured", "value": None}
    status = "pass" if predicate(value) else "miss"
    return {"target": name, "requirement": requirement, "status": status, "value": value}


def build_targets(
    accuracy: dict, latency: dict, escalation: dict, false_mutating: dict, memory: dict
) -> list[dict]:
    """One pass/miss/not-measured line per c20 success-signal target."""
    without_interpretation = len(false_mutating["without_interpretation_shown"])
    return [
        _check(
            "tier1_warm_p95_under_150ms",
            latency.get("warm_p95_ms"),
            lambda v: v < _WARM_P95_MS,
            "< 150 ms",
        ),
        _check(
            "correct_operation_and_arguments_ge_90pct",
            accuracy.get("argument_accuracy"),
            lambda v: v >= _MIN_ACCURACY,
            ">= 90%",
        ),
        _check(
            "zero_wrong_mutating_without_interpretation",
            float(without_interpretation),
            # `without_interpretation` is a non-negative integer count carried as a
            # float only to share `_check`'s signature -- compare with a tolerance
            # rather than `== 0.0` (S1244).
            lambda v: v < _ZERO_COUNT_TOLERANCE,
            "== 0",
        ),
        _check(
            "should_escalate_recall_ge_80pct",
            escalation.get("recall"),
            lambda v: v >= _MIN_ESCALATION_RECALL,
            ">= 80%",
        ),
        _check(
            "added_memory_under_1gb",
            memory.get("added_mib"),
            lambda v: v < _MAX_ADDED_MIB,
            "< 1024 MiB",
        ),
    ]


# ---------------------------------------------------------------------------
# Provenance: model hashes, image size
# ---------------------------------------------------------------------------


def model_hashes(pins: Mapping[str, object] | None, *, platform_tag: str = "") -> dict:
    """Pinned weights/engine sha256s (from ``pins.json``), ``None`` if absent."""
    if not pins:
        return {"weights_sha256": None, "engine_sha256": None}
    needle3 = pins.get("needle3") if isinstance(pins, Mapping) else None
    needle3 = needle3 if isinstance(needle3, Mapping) else {}
    weights = needle3.get("weights") if isinstance(needle3.get("weights"), Mapping) else {}
    engines = needle3.get("engines") if isinstance(needle3.get("engines"), Mapping) else {}
    engine_pin = engines.get(platform_tag) if isinstance(engines.get(platform_tag), Mapping) else {}
    return {"weights_sha256": weights.get("sha256"), "engine_sha256": engine_pin.get("sha256")}


def image_size_bytes(pins: Mapping[str, object] | None) -> int | None:
    """The first pinned Tier 2 image's size, or ``None`` when nothing is pinned yet."""
    if not pins:
        return None
    images = pins.get("images") if isinstance(pins, Mapping) else None
    if not images:
        return None
    first = images[0]
    return first.get("size_bytes") if isinstance(first, Mapping) else None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierThresholds:
    """The router's verifier thresholds this run was made with. Reported, not measured."""

    ask_below: float = 0.0
    escalate_below: float = -2.0
    min_mass: float = 0.05


@dataclass(frozen=True)
class BenchOptions:
    """Everything :func:`bench` needs besides "what to run and against what".

    Grouped out of ``bench``'s own signature (S107: 21 parameters was over
    the 13-parameter limit) -- ``tier2``/``verifier``/``min_confidence``/
    ``thresholds``/``clock``/``runner`` wire up the router the same way a
    caller building one directly would; the rest
    (``idle_memory``..``corpus_problems``) is reported provenance only, never
    read to make a routing decision.
    """

    tier2: Tier | None = None
    verifier: Verifier | None = None
    min_confidence: float = 0.0
    thresholds: VerifierThresholds = field(default_factory=VerifierThresholds)
    clock: Callable[[], float] = time.monotonic
    runner: ops_ground.Runner = ops_ground.default_runner
    idle_memory: MemoryReading | None = None
    peak_memory: MemoryReading | None = None
    pins: Mapping[str, object] | None = None
    platform_tag: str = ""
    nvsh_version: str = ""
    engine: str = "none"
    mode: str = "unknown"
    grounding: str = "unknown"
    concurrent_load: tuple[float, float, float] | None = None
    timestamp: str = ""
    corpus_problems: Sequence[str] = ()


def bench(
    entries: Sequence[CorpusEntry],
    *,
    split: str,
    tier1: Tier | None,
    platform: Platform,
    options: BenchOptions = BenchOptions(),
) -> dict:
    """Run *entries* through a real :class:`TierRouter` and return the results dict.

    ``tier1``/``options.tier2`` are whatever the caller built -- a scripted
    :class:`~nvsh.tiers.fake.FakeTier` for a repeatable test, or a real tier
    for a device measurement. Everything but ``provenance.timestamp`` is
    reproducible when *entries*, the tiers' scripted output and
    ``options.clock`` are held fixed (criterion 2).
    """
    clock = options.clock
    with tempfile.TemporaryDirectory(prefix="nvsh-tiers-bench-") as tmp_dir:
        records = TierRecords(path=Path(tmp_dir) / "bench-records.jsonl")
        router = TierRouter(
            tier1,
            options.tier2,
            records,
            platform,
            runner=options.runner,
            verifier=options.verifier,
            min_confidence=options.min_confidence,
            clock=clock,
        )
        items = run_items(entries, router, clock)

    accuracy = compute_operation_accuracy(items)
    escalation = compute_escalation(items)
    false_mutating = compute_false_mutating(items)
    calibration = compute_calibration(items)
    threshold_suggestion = (
        suggest_thresholds(items)
        if split == DEV_SPLIT
        else {
            "escalate_below": None,
            "ask_below": None,
            "youden_j": None,
            "note": "not the dev split",
        }
    )
    latency = compute_latency(items)
    memory = compute_memory(options.idle_memory, options.peak_memory)
    targets = build_targets(accuracy, latency, escalation, false_mutating, memory)

    return {
        "corpus": {
            "split": split,
            "count": len(entries),
            "problems": list(options.corpus_problems),
        },
        "accuracy": accuracy,
        "accuracy_by_kind": accuracy_by_kind(items),
        "accuracy_by_operation": accuracy_by_operation(items),
        "accuracy_by_class": accuracy_by_class(items),
        "items": item_rows(items),
        "escalation": escalation,
        "false_mutating_pick": false_mutating,
        "calibration": calibration,
        "threshold_suggestion": threshold_suggestion,
        "latency": latency,
        "memory": memory,
        "targets": targets,
        "provenance": {
            "nvsh_version": options.nvsh_version,
            "device": platform.to_dict(),
            "engine": options.engine,
            "mode": options.mode,
            "grounding": options.grounding,
            "concurrent_load": (
                list(options.concurrent_load) if options.concurrent_load is not None else None
            ),
            "model_hashes": model_hashes(options.pins, platform_tag=options.platform_tag),
            "image_size_bytes": image_size_bytes(options.pins),
            "thresholds_in_force": {
                "min_confidence": options.min_confidence,
                "ask_below": options.thresholds.ask_below,
                "escalate_below": options.thresholds.escalate_below,
                "min_mass": options.thresholds.min_mass,
            },
            "timestamp": options.timestamp,
        },
    }
