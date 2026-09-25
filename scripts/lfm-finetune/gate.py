#!/usr/bin/env python3
"""Uncertainty gate (issue 53): turn a scorer's distribution into a decision.

A development-machine tool. It is NEVER imported by the nvsh package --
nothing under ``nvsh/`` may depend on it. Stdlib only.

``scorer.py`` (Track B) already produces, per request, a normalised
probability distribution over the offered candidates (operation names, plus
``nvsh.tiers.bench``'s ``"(escalate)"`` / ``"(explain)"`` controls) and an
argmax choice. This module adds a confidence gate on top of that argmax: a
scorer that is *right on average* can still be wrong on a specific request,
and a mutating operation proposed with low confidence is a worse outcome
than asking the operator to look. :func:`decide` never generates a new
choice -- it only decides whether to trust the argmax, override it with a
semantic escalation, or decline it as ``abstain_uncertain`` (a confidence
decision, as opposed to ``escalate``, a semantic one the scorer itself
picked -- see ``metrics.py``'s module docstring).

The decision, in order:

1. Every ``escalate:<reason>`` candidate label is rolled into the bare
   ``(escalate)`` label first (``metrics.rollup_escalate_candidates``), so a
   scorer that splits its escalate mass across reasons is judged the same
   as one that reports a single bare label.
2. **Semantic escalate**: if the rolled escalate mass is at or above
   ``thresholds.escalate`` (when that threshold is set), or escalate is
   itself the argmax, the decision is ``escalate`` regardless of anything
   else below.
3. Otherwise take the argmax (``top1``). If it is the explain control, the
   decision is ``explain``.
4. Otherwise top1 names an operation. Which one of the two threshold sets
   applies -- :attr:`Thresholds.read_only` or :attr:`Thresholds.mutating`
   -- is decided from ``nvsh.ops.table``'s ``read_only`` flag for that
   operation, **never from the operation's name**: this module names no
   operation anywhere in its own code, so a new table entry needs no
   change here. The decision is ``abstain_uncertain`` when top1's own
   probability is below the set's ``floor``, or its margin over the
   second-place candidate is below the set's ``margin``, or the rolled
   distribution's normalised entropy is above the set's ``max_entropy``;
   otherwise it is ``propose``. A threshold left as ``None`` never fires
   (an all-``None`` :class:`Thresholds` reproduces the bare argmax).

Normalised entropy is the rolled distribution's Shannon entropy divided by
``log(n_offered)``, where ``n_offered`` is the number of labels actually
offered to the scorer (*before* the escalate-reason roll-up, since that is
how many slots the model chose among) -- so it is 0 for a certain answer,
1 for a uniform one over everything offered, and stays comparable across
requests that offered different numbers of candidates. A request with at
most one offered candidate has normalised entropy 0 (there is no
uncertainty to measure).

Ties (an exact probability tie between two candidates after the roll-up)
go to whichever candidate appears earlier in *offered*, matching
``scorer.score``'s own ``max(labels, key=...)`` tie-break (Python's
``max`` keeps the first maximum it sees), so :func:`decide` reproduces a
stored prediction's argmax exactly when it is given the same
``candidates`` dict and the same offered order.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # runnable from any directory, for nvsh.*

from nvsh.ops import table as ops_table  # noqa: E402
from nvsh.tiers import bench as tier_bench  # noqa: E402


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


metrics = _sibling("metrics")

#: The two candidates :func:`decide` never treats as an operation.
ESCALATE_LABEL = tier_bench.ESCALATE_LABEL
EXPLAIN_LABEL = tier_bench.EXPLAIN_LABEL

#: Decision outcomes, in the order :func:`decide` can produce them.
OUTCOMES = ("propose", "explain", "escalate", "abstain_uncertain")


class GateError(ValueError):
    """A malformed threshold payload or an empty distribution."""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdSet:
    """One operation class's abstention thresholds. A ``None`` field never fires.

    ``floor``: minimum top1 probability to propose.
    ``margin``: minimum top1-minus-top2 probability gap to propose.
    ``max_entropy``: maximum normalised entropy (0..1) to propose.
    """

    floor: float | None = None
    margin: float | None = None
    max_entropy: float | None = None

    def to_json(self) -> dict:
        return {"floor": self.floor, "margin": self.margin, "max_entropy": self.max_entropy}

    @classmethod
    def from_json(cls, data: Mapping) -> "ThresholdSet":
        if not isinstance(data, Mapping):
            raise GateError(f"a threshold set must be an object, got {data!r}")
        return cls(
            floor=_optional_float(data.get("floor"), "floor"),
            margin=_optional_float(data.get("margin"), "margin"),
            max_entropy=_optional_float(data.get("max_entropy"), "max_entropy"),
        )


@dataclass(frozen=True)
class Thresholds:
    """The gate's full configuration: one semantic-escalate floor, two class sets.

    ``read_only`` and ``mutating`` are looked up from ``nvsh.ops.table``'s
    ``read_only`` flag for the argmax operation -- never from its name (see
    module docstring). ``Thresholds()`` (every field ``None``) disables
    every check: :func:`decide` then reproduces the bare argmax.
    """

    escalate: float | None = None
    read_only: ThresholdSet = field(default_factory=ThresholdSet)
    mutating: ThresholdSet = field(default_factory=ThresholdSet)

    def to_json(self) -> dict:
        return {
            "escalate": self.escalate,
            "read_only": self.read_only.to_json(),
            "mutating": self.mutating.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping) -> "Thresholds":
        if not isinstance(data, Mapping):
            raise GateError(f"thresholds must be an object, got {data!r}")
        return cls(
            escalate=_optional_float(data.get("escalate"), "escalate"),
            read_only=ThresholdSet.from_json(data.get("read_only", {})),
            mutating=ThresholdSet.from_json(data.get("mutating", {})),
        )


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{name} must be a number or null, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """One :func:`decide` result.

    ``outcome`` is one of :data:`OUTCOMES`. ``label`` is the candidate name
    the decision is about: the argmax operation for ``propose`` and
    ``abstain_uncertain``, :data:`ESCALATE_LABEL` / :data:`EXPLAIN_LABEL`
    for the two controls. ``reason`` names which check fired for
    ``abstain_uncertain`` (``"floor"``, ``"margin"`` or ``"entropy"``), or
    ``"threshold"`` / ``"argmax"`` for ``escalate``; ``None`` for the other
    outcomes.
    """

    outcome: str
    label: str
    reason: str | None = None


def _canonical_order(offered: Sequence[str], rolled: Mapping[str, float]) -> list[str]:
    """*offered*'s labels, canonicalised and de-duplicated, in first-seen order.

    Matches ``scorer.score``'s own iteration order (the offered sequence),
    so tie-breaking here agrees with its ``max(labels, key=...)``. Any
    label present only in *rolled* (should not normally happen) is
    appended at the end so it can still be found.
    """
    order: list[str] = []
    seen: set[str] = set()
    for name in offered:
        canon = metrics.canonical_label(name)
        if canon not in seen:
            seen.add(canon)
            order.append(canon)
    for name in rolled:
        if name not in seen:
            seen.add(name)
            order.append(name)
    return order


def _argmax(rolled: Mapping[str, float], offered: Sequence[str]) -> tuple[str, float]:
    """The highest-probability label in *rolled*; ties go to the earlier *offered* entry."""
    best_label: str | None = None
    best_p = -1.0
    for name in _canonical_order(offered, rolled):
        p = rolled.get(name, 0.0)
        if p > best_p:
            best_p = p
            best_label = name
    assert best_label is not None  # rolled is non-empty (checked by decide)
    return best_label, best_p


def _second_place(rolled: Mapping[str, float], top_label: str) -> float:
    """The highest probability among *rolled*'s labels other than *top_label*."""
    rest = [p for name, p in rolled.items() if name != top_label]
    return max(rest) if rest else 0.0


def normalized_entropy(rolled: Mapping[str, float], n_offered: int) -> float:
    """Shannon entropy of *rolled*, divided by ``log(n_offered)``; 0 when ``n_offered <= 1``."""
    if n_offered <= 1:
        return 0.0
    entropy = -math.fsum(p * math.log(p) for p in rolled.values() if p > 0.0)
    return entropy / math.log(n_offered)


def decide(
    distribution: Mapping[str, float], offered: Sequence[str], thresholds: Thresholds
) -> Decision:
    """Gate one request's scored distribution into a :class:`Decision`.

    *distribution* is a candidate -> probability mapping (``metrics.py``'s
    ``candidates`` schema: non-empty, summing to ~1), typically a
    prediction line's own ``candidates``. *offered* is the sequence of
    labels that were actually offered to the scorer, in the order they
    were offered (a prediction's ``candidates`` dict keys, in their
    original JSON order, are exactly this). See the module docstring for
    the decision order.
    """
    if not distribution:
        raise GateError("distribution must not be empty")
    rolled = metrics.rollup_escalate_candidates(distribution)
    top1_label, p_top1 = _argmax(rolled, offered)

    p_escalate = rolled.get(ESCALATE_LABEL, 0.0)
    if top1_label == ESCALATE_LABEL:
        return Decision("escalate", ESCALATE_LABEL, "argmax")
    if thresholds.escalate is not None and p_escalate >= thresholds.escalate:
        return Decision("escalate", ESCALATE_LABEL, "threshold")

    if top1_label == EXPLAIN_LABEL:
        return Decision("explain", EXPLAIN_LABEL)

    operation = ops_table.get(top1_label)
    read_only = True if operation is None else operation.read_only
    thresholds_for_class = thresholds.read_only if read_only else thresholds.mutating

    margin = p_top1 - _second_place(rolled, top1_label)
    entropy = normalized_entropy(rolled, len(offered))

    if thresholds_for_class.floor is not None and p_top1 < thresholds_for_class.floor:
        return Decision("abstain_uncertain", top1_label, "floor")
    if thresholds_for_class.margin is not None and margin < thresholds_for_class.margin:
        return Decision("abstain_uncertain", top1_label, "margin")
    if thresholds_for_class.max_entropy is not None and entropy > thresholds_for_class.max_entropy:
        return Decision("abstain_uncertain", top1_label, "entropy")
    return Decision("propose", top1_label)
