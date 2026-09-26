"""Trace schema: raw model output and per-policy final decisions, side by side.

A :class:`Trace` is one evaluation case: the model's raw, unpolicied answer
(:class:`RawRecord`) plus zero or more policies' *final* decisions layered on
top of it. Policies never see or touch ``raw`` in place -- each
:meth:`Trace.with_policy` call returns a new, otherwise-identical ``Trace``
(both are frozen dataclasses), so the raw record a report attributes to a
model is provably the same object a later policy comparison used.

``RawRecord`` is built from either of two sources that this evals suite
scores side by side (see the module docstring of
``scripts/lfm-finetune/metrics.py``, which this module loads by path -- see
below -- rather than copying):

* a **predictions line** written by ``nvsh.tiers.bench`` / the LFM
  fine-tuning harness, one JSON object per corpus entry
  (``outcome``, ``operation``, ``arguments``, ``candidates`` or ``None``,
  ``tokens``, ``ttfd_ms``/``latency_ms``, optional ``invalid_reason``);
* a **provider answer** -- a hosted or local model's tool-call/choice
  response, parsed into the same ``outcome``/``operation``/``arguments``/
  ``candidates``/``invalid_reason`` vocabulary, plus which provider, model,
  the model id the provider actually returned, and whether the answer came
  back as a tool call or a chosen label (``interface``).

``RawRecord`` carries both sets of fields (the provider-only ones default to
``None`` when built from a predictions line, and the predictions-only
``tokens``/``ttfd_ms``/``latency_ms`` default to ``None`` when built from a
provider answer), so one schema fits both sources.

This module is never imported by anything under ``nvsh/`` (evals may import
nvsh, not the other way around), and it never spawns a subprocess or touches
the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

# ---------------------------------------------------------------------------
# Load scripts/lfm-finetune/metrics.py by path (never copy its code -- see
# COMMON.md / this task's brief). This gives us Prediction.from_dict, which
# is the one place the predictions-line schema is validated.
# ---------------------------------------------------------------------------

_METRICS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "lfm-finetune" / "metrics.py"


def _load_metrics_module():
    spec = importlib.util.spec_from_file_location("lfm_metrics_for_trace", _METRICS_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load metrics module from {_METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_metrics = _load_metrics_module()
Prediction = _metrics.Prediction
MetricsError = _metrics.MetricsError


# ---------------------------------------------------------------------------
# RawRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRecord:
    """The unpolicied model output for one case.

    ``outcome``/``operation``/``arguments``/``candidates``/``invalid_reason``
    follow ``scripts/lfm-finetune/metrics.py``'s ``Prediction`` schema and are
    shared by both sources this record can come from.

    ``tokens``/``ttfd_ms``/``latency_ms`` come from a predictions line; they
    are ``None`` when this record was built from a provider answer that did
    not report them.

    ``provider``/``model``/``returned_model``/``interface`` come from a
    provider answer (``interface`` is ``"tool_call"`` or ``"choice"``); they
    are ``None`` when this record was built from a predictions line.
    """

    outcome: str
    operation: str | None
    arguments: dict | None
    candidates: dict | None
    invalid_reason: str | None = None
    tokens: int | None = None
    ttfd_ms: float | None = None
    latency_ms: float | None = None
    provider: str | None = None
    model: str | None = None
    returned_model: str | None = None
    interface: str | None = None

    @classmethod
    def from_prediction(cls, prediction: "Prediction") -> "RawRecord":
        """Build a ``RawRecord`` from a validated ``metrics.Prediction``."""
        return cls(
            outcome=prediction.outcome,
            operation=prediction.operation,
            arguments=None if prediction.arguments is None else dict(prediction.arguments),
            candidates=None if prediction.candidates is None else dict(prediction.candidates),
            invalid_reason=prediction.invalid_reason,
            tokens=prediction.tokens,
            ttfd_ms=prediction.ttfd_ms,
            latency_ms=prediction.latency_ms,
        )

    @classmethod
    def from_prediction_line(cls, row: Mapping[str, Any]) -> "RawRecord":
        """Validate and build a ``RawRecord`` straight from a decoded predictions line."""
        return cls.from_prediction(Prediction.from_dict(dict(row)))

    @classmethod
    def from_provider_answer(
        cls,
        *,
        provider: str,
        model: str,
        returned_model: str | None,
        interface: str,
        outcome: str,
        operation: str | None = None,
        arguments: dict | None = None,
        candidates: dict | None = None,
        invalid_reason: str | None = None,
    ) -> "RawRecord":
        """Build a ``RawRecord`` from a provider's parsed tool-call/choice answer.

        ``candidates`` is ``None`` when the provider returned no logprobs to
        normalise into a distribution.
        """
        if interface not in ("tool_call", "choice"):
            raise ValueError(f"interface must be 'tool_call' or 'choice', got {interface!r}")
        return cls(
            outcome=outcome,
            operation=operation,
            arguments=None if arguments is None else dict(arguments),
            candidates=None if candidates is None else dict(candidates),
            invalid_reason=invalid_reason,
            provider=provider,
            model=model,
            returned_model=returned_model,
            interface=interface,
        )

    def to_dict(self) -> dict:
        """The full record, every field, for JSONL serialisation."""
        return {
            "outcome": self.outcome,
            "operation": self.operation,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "candidates": None if self.candidates is None else dict(self.candidates),
            "invalid_reason": self.invalid_reason,
            "tokens": self.tokens,
            "ttfd_ms": self.ttfd_ms,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "model": self.model,
            "returned_model": self.returned_model,
            "interface": self.interface,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RawRecord":
        """Inverse of :meth:`to_dict`, for reading a written trace back."""
        return cls(**dict(data))

    def to_prediction_dict(self, case_id: str, expected: Mapping[str, Any]) -> dict:
        """Reconstruct a metrics.py-schema predictions line from this record.

        ``case_id``/``expected`` are supplied by the caller (they live on the
        :class:`Trace`, not on the raw record itself) so the result is a
        complete predictions line: ``Prediction.from_dict`` on the result
        re-parses to the same ``Prediction`` this record was built from.
        """
        result: dict[str, Any] = {
            "id": case_id,
            "expected": dict(expected),
            "outcome": self.outcome,
            "operation": self.operation,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "candidates": None if self.candidates is None else dict(self.candidates),
            "tokens": self.tokens,
            "ttfd_ms": self.ttfd_ms,
            "latency_ms": self.latency_ms,
        }
        if self.invalid_reason is not None:
            result["invalid_reason"] = self.invalid_reason
        return result


# ---------------------------------------------------------------------------
# Per-policy final decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    """One policy's final decision on a case."""

    decision: str
    reason: str

    def to_dict(self) -> dict:
        return {"decision": self.decision, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyResult":
        return cls(decision=data["decision"], reason=data["reason"])


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """One evaluation case: raw model output plus each policy's final call on it.

    ``final`` maps a policy name to its :class:`PolicyResult`. ``raw`` is
    never mutated by a policy: :meth:`with_policy` returns a new ``Trace``
    whose ``raw`` is the same, untouched ``RawRecord``.
    """

    case_id: str
    split: str
    raw: RawRecord
    final: Mapping[str, PolicyResult] = field(default_factory=dict)
    ground_truth: Mapping[str, Any] | None = None
    subject: str | None = None

    @classmethod
    def from_prediction_line(
        cls,
        row: Mapping[str, Any],
        *,
        split: str,
        subject: str | None = None,
    ) -> "Trace":
        """Build a ``Trace`` straight from a decoded metrics.py predictions line."""
        prediction = Prediction.from_dict(dict(row))
        return cls(
            case_id=prediction.id,
            split=split,
            raw=RawRecord.from_prediction(prediction),
            ground_truth=dict(prediction.expected),
            subject=subject,
        )

    def with_policy(self, policy: str, decision: str, reason: str) -> "Trace":
        """Return a new ``Trace`` with one more policy's final decision recorded.

        ``self`` (and its ``raw``) is untouched; the returned ``Trace`` is a
        distinct object carrying every previous policy's result plus this one.
        """
        new_final = dict(self.final)
        new_final[policy] = PolicyResult(decision=decision, reason=reason)
        return replace(self, final=new_final)

    def to_dict(self) -> dict:
        """The full trace, for JSONL serialisation."""
        return {
            "case_id": self.case_id,
            "split": self.split,
            "raw": self.raw.to_dict(),
            "final": {name: result.to_dict() for name, result in self.final.items()},
            "ground_truth": None if self.ground_truth is None else dict(self.ground_truth),
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trace":
        """Inverse of :meth:`to_dict`, for reading a written trace back."""
        return cls(
            case_id=data["case_id"],
            split=data["split"],
            raw=RawRecord.from_dict(data["raw"]),
            final={
                name: PolicyResult.from_dict(result)
                for name, result in data.get("final", {}).items()
            },
            ground_truth=data.get("ground_truth"),
            subject=data.get("subject"),
        )


# ---------------------------------------------------------------------------
# JSONL writer -- private run dir only, never inside this git worktree
# ---------------------------------------------------------------------------


class TraceWriteError(ValueError):
    """Raised when asked to write trace output to a path inside a git worktree."""


def _is_inside_git_worktree(path: Path) -> bool:
    """True if ``path`` (or any of its existing ancestors) sits inside a git
    working tree -- detected by walking up looking for a ``.git`` entry,
    which is a directory in an ordinary clone and a gitlink *file* in a
    linked worktree (both must be refused)."""
    resolved = path.resolve()
    start = resolved if resolved.is_dir() else resolved.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return True
    return False


def _dump_trace(trace: Trace) -> str:
    """Serialize *trace* to one JSON line, never reordering its keys.

    ``json.dumps(..., sort_keys=True)`` looked deterministic but is not
    order-preserving: it recursively sorts every nested mapping, including
    ``raw.candidates``. ``candidates`` order is semantic --
    ``metrics_bridge._rolled_and_offered`` derives the ``offered`` sequence
    ``gate.decide`` uses for tie-breaking straight from
    ``list(candidates.keys())`` -- so alphabetically resorting it (e.g.
    ``"(explain)"`` sorting before ``"disk_usage"`` on ASCII ``(`` < ``d``)
    can silently flip a policy's decision between the pre- and
    post-serialization record. A plain ``dict`` already serializes in
    insertion order (Python 3.7+ / JSON), which is exactly the order the
    caller built ``candidates`` in, so omitting ``sort_keys`` alone keeps
    output fully deterministic (the same ``Trace`` always serializes
    identically) while preserving offered order.
    """
    return json.dumps(trace.to_dict())


def append_trace(path: str | Path, trace: Trace) -> None:
    """Append one ``Trace`` as a JSONL line to ``path``.

    Refuses (``TraceWriteError``) when ``path`` is inside a git worktree:
    trace output belongs in the operator's private run dir, outside any repo.
    """
    target = Path(path)
    if _is_inside_git_worktree(target):
        raise TraceWriteError(f"refusing to write a trace file inside a git worktree: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(_dump_trace(trace))
        handle.write("\n")


def write_traces(path: str | Path, traces: Iterable[Trace]) -> None:
    """Write ``traces`` to a fresh JSONL file at ``path`` (truncating any existing file).

    Refuses (``TraceWriteError``) when ``path`` is inside a git worktree.
    """
    target = Path(path)
    if _is_inside_git_worktree(target):
        raise TraceWriteError(f"refusing to write a trace file inside a git worktree: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(_dump_trace(trace))
            handle.write("\n")


def read_traces(path: str | Path) -> list[Trace]:
    """Read a JSONL trace file back into a list of :class:`Trace` objects."""
    traces: list[Trace] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            traces.append(Trace.from_dict(json.loads(line)))
    return traces
