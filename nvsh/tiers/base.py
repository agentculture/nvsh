"""The tier contract: decisions, declines, and validation of untrusted tier output.

A tier (a tiny tool-selecting local model, e.g. Needle3, or a small local
agent) returns "function calls" that nvsh must never trust directly. This
module turns that raw output into either a validated :class:`TierDecision`
or a :class:`Decline` carrying a reason code -- never anything a caller
could run as-is. A :class:`Tier` therefore only ever *selects*; it has no
``execute``/``run``/``exec`` of any kind, on this class or any subclass.

Nothing here imports subprocess-heavy or third-party modules: this module
must stay importable on the hot success path.
"""

from __future__ import annotations

import abc
import enum
import math
from dataclasses import dataclass

from ..agent.base import AgentContext, AgentRequest
from ..ops import table as ops_table

#: Names that must never be treated as an operation, even if they happen to
#: validate against the operation table (they never will, since none of the
#: 16 registered operations use these names) -- a defense-in-depth check for
#: a tier that invents a free-form "run a shell command" tool.
_RAW_SHELL_NAMES = frozenset({"bash", "shell", "run", "sh", "exec", "eval", "command"})


class DeclineReason(enum.Enum):
    """Why a tier's output (or a decision built from it) was declined."""

    NO_CALL = "no_call"
    MULTIPLE_CALLS = "multiple_calls"
    UNKNOWN_OPERATION = "unknown_operation"
    BAD_ARGUMENT = "bad_argument"
    RAW_SHELL = "raw_shell"
    MALFORMED_OUTPUT = "malformed_output"
    TIER_UNAVAILABLE = "tier_unavailable"
    TIER_ERROR = "tier_error"
    LOW_CONFIDENCE = "low_confidence"
    MEMORY_FLOOR = "memory_floor"
    NOT_GROUNDED = "not_grounded"
    LOOP_LIMIT = "loop_limit"
    #: The tier itself asked for the full agent: Tier 2's ``escalate`` tool,
    #: a turn that produced nothing usable, or a loop that ran out of rounds.
    #: Not a fault -- escalation is one of Tier 2's three normal outcomes.
    ESCALATED = "escalated"
    #: The operation validated and grounded, but no single non-shell command
    #: renders it on this platform (``nvsh.ops.render`` returned ``None``).
    NOT_RENDERABLE = "not_renderable"


@dataclass(frozen=True)
class TierDecision:
    """One validated call a tier proposed, ready for grounding/approval.

    ``args`` is always a ``dict[str, str]`` -- table validation already
    guarantees every value is a non-empty string. ``read_only`` mirrors the
    resolved :class:`~nvsh.ops._model.Operation`'s own flag so a caller
    never has to look the operation back up just to branch on it.
    """

    operation: str
    args: dict[str, str]
    confidence: float | None
    read_only: bool


@dataclass(frozen=True)
class Decline:
    """A tier's output could not become a decision.

    ``inspections`` carries what the tier already found out before giving up
    -- pairs of ``(operation, result excerpt)`` -- so the next tier, or the
    full agent, does not repeat the same read-only work. It is empty for a
    tier that inspects nothing (Tier 1 never does). The excerpts are the
    tier's own text; the router redacts and bounds them before they leave
    the process.
    """

    reason: DeclineReason
    detail: str = ""
    inspections: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Explanation:
    """A tier answered in plain words instead of proposing a command.

    Tier 2's third outcome (propose / explain / escalate): ``text`` is the
    explanation to show the operator, ``inspections`` the read-only work it
    did to get there, in the same shape as :attr:`Decline.inspections`.
    """

    text: str
    inspections: tuple[tuple[str, str], ...] = ()


def _coerce_confidence(confidence: object) -> float | None:
    """Return *confidence* as a float, or ``None`` if it isn't numeric.

    ``None`` is passed through as-is (a fine-tuned Needle model reporting no
    confidence must never be penalised). Anything else that isn't an
    ``int``/``float`` (a stray string, for instance) is treated the same as
    "absent" rather than raising or silently miscomparing.
    """
    if confidence is None:
        return None
    if isinstance(confidence, bool):
        # bool is an int subclass; a tier has no business reporting True/False.
        return None
    if isinstance(confidence, (int, float)):
        value = float(confidence)
        # NaN compares False against any floor and is not valid JSON; an
        # out-of-range score is not a probability. Both count as absent.
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            return value
    return None


def _single_call_or_decline(raw_calls: object) -> tuple[object, object] | Decline:
    """Reduce *raw_calls* to exactly one ``(name, arguments)`` pair, or a Decline.

    Never raises: any shape that isn't "a list with exactly one dict-shaped
    call" becomes a :class:`Decline` with a reason code, not an exception.
    """
    if not isinstance(raw_calls, list):
        return Decline(
            reason=DeclineReason.MALFORMED_OUTPUT,
            detail=f"tier output was {type(raw_calls).__name__}, not a list of calls",
        )

    if len(raw_calls) == 0:
        return Decline(reason=DeclineReason.NO_CALL, detail="tier returned no calls")

    if len(raw_calls) > 1:
        return Decline(
            reason=DeclineReason.MULTIPLE_CALLS,
            detail=f"tier returned {len(raw_calls)} calls, expected exactly one",
        )

    (call,) = raw_calls

    if isinstance(call, str):
        return Decline(
            reason=DeclineReason.RAW_SHELL,
            detail="tier returned a raw string instead of a typed call",
        )

    if not isinstance(call, dict):
        return Decline(
            reason=DeclineReason.MALFORMED_OUTPUT,
            detail=f"tier call was {type(call).__name__}, not an object",
        )

    if "name" not in call or "arguments" not in call:
        return Decline(
            reason=DeclineReason.MALFORMED_OUTPUT,
            detail="tier call is missing 'name' or 'arguments'",
        )

    name = call["name"]
    arguments = call["arguments"]

    if isinstance(name, str) and name.strip().lower() in _RAW_SHELL_NAMES:
        return Decline(
            reason=DeclineReason.RAW_SHELL,
            detail=f"tier proposed {name!r}, a free-form shell escape hatch",
        )

    return (name, arguments)


def decide(
    raw_calls: object,
    confidence: object = None,
    *,
    min_confidence: float = 0.0,
) -> TierDecision | Decline:
    """Turn a tier's raw output into a validated decision or a decline.

    Never raises, no matter what *raw_calls* or *confidence* contain --
    the tier is untrusted input. ``raw_calls`` is expected to be a list of
    ``{"name": str, "arguments": dict}`` call objects (per the measured
    real-model output shape), possibly empty, possibly with several calls.
    ``confidence`` of ``None`` never causes a decline on its own; a numeric
    confidence below ``min_confidence`` does.
    """
    try:
        reduced = _single_call_or_decline(raw_calls)
        if isinstance(reduced, Decline):
            return reduced

        name, arguments = reduced

        error = ops_table.validate(name, arguments)
        if error is not None:
            return Decline(
                reason=(
                    DeclineReason.UNKNOWN_OPERATION
                    if error.code == "unknown_operation"
                    else DeclineReason.BAD_ARGUMENT
                ),
                detail=error.message,
            )

        op = ops_table.get(name)
        if op is None:
            # validate() already confirmed this, but stay defensive: never
            # trust that two calls into the same table agree.
            return Decline(
                reason=DeclineReason.UNKNOWN_OPERATION,
                detail=f"unknown operation {name!r}",
            )

        coerced_confidence = _coerce_confidence(confidence)
        if coerced_confidence is not None and coerced_confidence < min_confidence:
            return Decline(
                reason=DeclineReason.LOW_CONFIDENCE,
                detail=f"confidence {coerced_confidence!r} below floor {min_confidence!r}",
            )

        return TierDecision(
            operation=name,
            args=dict(arguments),
            confidence=coerced_confidence,
            read_only=op.read_only,
        )
    except Exception as exc:  # decide() must never raise on untrusted input
        return Decline(
            reason=DeclineReason.MALFORMED_OUTPUT,
            detail=f"unexpected error validating tier output: {exc}",
        )


class Tier(abc.ABC):
    """A local response tier: proposes a decision, never executes one.

    What a tier *returns* is never run by the tier. Tier 2 does run read-only
    operations from the table while it looks into a request (grounded,
    rendered by ``nvsh.ops``, with a timeout); it never runs a mutating one.

    ``select()`` is the only way a tier speaks: it returns a
    :class:`TierDecision` or a :class:`Decline`, both inert data. No ``Tier``
    subclass may define ``execute``/``run``/``exec`` -- turning a decision
    into an action is strictly the caller's job, after grounding and
    operator approval.
    """

    name: str

    @abc.abstractmethod
    def select(
        self, request: AgentRequest, context: AgentContext
    ) -> TierDecision | Decline | Explanation:
        """Propose a decision for *request* given *context*. Never executes it.

        A tier that can answer in plain words (Tier 2) may return an
        :class:`Explanation` instead; a tier that only selects operations
        (Tier 1) returns a :class:`TierDecision` or a :class:`Decline`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources (connections, subprocesses) the tier holds."""
        raise NotImplementedError
