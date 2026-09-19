"""The tier router: which tier answers a request, and what it may propose.

The router is the only place that turns a tier's pick into something the
operator can see. It streams :class:`~nvsh.agent.base.AgentEvent` objects
exactly as a backend adapter's ``run()`` does, so the client can render a
tier answer and a full-agent answer through the same code path.

**It executes nothing.** Every decision travels one fixed chain --
``decide()`` (already run inside the tier) -> :func:`nvsh.ops.ground.ground`
-> :func:`nvsh.ops.render.render` -- and the resulting
:class:`~nvsh.agent.base.Proposal` carries *only* what ``render()`` produced,
joined with :func:`shlex.join`. A tier's text never reaches a shell, and
approval stays exactly where it was: the caller's ``approve`` callback in
``nvsh.agent.loop.run_loop``.

**No operation name is special-cased here.** The router asks the table what
an operation is (``read_only``, its arguments, its description); adding an
operation to ``nvsh/ops/table.py`` needs no change in this module.

**``request.target`` is not the router's concern.** An ``@target`` request
names a harness explicitly, and the caller bypasses the tiers entirely for
it (spec: "an explicit ``@target`` naming another harness always bypasses
the tiers"). The router does not inspect ``target``.

Per-call state lives on the :class:`Route` object that :meth:`TierRouter.route`
returns, never on the router: one router is shared by every handler thread in
the per-user daemon.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, Protocol, Sequence

from ..agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    EventKind,
    Proposal,
    ProposalKind,
    RequestKind,
)
from ..ops import ground as ops_ground
from ..ops import table as ops_table
from ..ops._model import Operation
from ..ops.render import render as render_argv
from ..platform._model import Platform
from ..redact import redact
from ._bounded import bounded_cut
from .base import Decline, DeclineReason, Explanation, Tier, TierDecision
from .records import TierRecord, TierRecords
from .toolchat import calibrated_logit, yes_no_probability

#: The name recorded (and reported in :attr:`TierOutcome.escalated_to`) for
#: "the full agent" -- Tier 3, which is not a :class:`Tier` at all.
AGENT = "agent"

#: What a :class:`Verifier` may ask the router to do with a Tier 1 pick.
PROPOSE = "propose"
ASK = "ask"
ESCALATE = "escalate"

#: How much of one inspection result travels to the next tier / the agent.
EXCERPT_CHARS = 2048
#: Extra characters handed to the redactor beyond the excerpt limit.
REDACT_SLACK = 512

#: The content-free request the per-operation verifier baselines are measured
#: against. Not a real question: it holds the question's *shape* constant so
#: the lift for a real request is comparable across operations.
BASELINE_REQUEST = "N/A"


# ---------------------------------------------------------------------------
# The optional verifier (deviation d1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierVerdict:
    """One verifier's reading of a Tier 1 pick.

    ``p_yes`` and ``calibrated`` are measurements for the bench and the
    records; ``action`` is what the router does with them.
    """

    p_yes: float | None = None
    calibrated: float | None = None
    action: str = PROPOSE


class Verifier(Protocol):
    """Second-opinion check on a Tier 1 pick. **Not a safety mechanism.**

    Approval is the safety mechanism: whatever a verifier says, a mutating
    operation still reaches the operator as a proposal that shows the
    interpreted operation and arguments, and nothing runs until the operator
    approves it. A verifier only decides whether Tier 1's answer is worth
    showing at all.

    ``verify`` returns ``None`` when it cannot form an opinion (no server, a
    malformed reply, an operation it has no baseline for); it never raises.
    """

    def verify(self, request_text: str, decision: TierDecision) -> VerifierVerdict | None:
        """Read *decision* against *request_text*. Never raises."""


class _Scorer(Protocol):
    """The one method :class:`LogprobVerifier` needs from a ToolChat."""

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        """Token -> log-probability for the next token after *prompt*."""


def describe(operation: Operation, args: Mapping[str, str]) -> str:
    """The interpreted operation and arguments, as one line of plain text.

    ``service_restart service=nginx.service`` -- what the operator is shown
    in a proposal's rationale, and what the verifier is asked about.
    """
    rendered = " ".join(f"{name}={args[name]}" for name in sorted(args))
    return f"{operation.name} {rendered}".strip()


def _question(request_text: str, operation: Operation, args: Mapping[str, str]) -> str:
    """The fixed yes/no question the verifier scores.

    One shape for every operation: nothing here is keyed on a specific
    operation name, so an operation added to the table is verified the same
    way as every other one.
    """
    solution = f"{operation.description} ({describe(operation, args)})"
    return f"For {request_text} the solution is {solution}. Correct? Answer yes or no:"


class LogprobVerifier:
    """Asks a local model one yes/no question about a Tier 1 pick.

    **Not a safety mechanism** -- see :class:`Verifier`. It exists to catch
    the case Needle3 is known for: a confident pick of the wrong operation.
    The model's answer is read as the probability mass on "yes" versus "no"
    for the single next token (:func:`~nvsh.tiers.toolchat.yes_no_probability`),
    calibrated against the same question asked about a content-free request
    (:func:`~nvsh.tiers.toolchat.calibrated_logit`), so an operation the model
    simply likes saying yes to does not score well by default.

    The baselines are measured lazily, once per operation, from the operation
    table -- never hard-coded, never keyed on a particular operation's name.
    ``operation_names`` bounds which operations may be verified at all
    (default: every operation in the table).

    Every failure -- a server that is not there, a reply with no
    log-probabilities, an operation with no baseline -- returns ``None``,
    which the router treats as "no verifier ran".
    """

    def __init__(
        self,
        chat: _Scorer,
        *,
        operation_names: Sequence[str] | None = None,
        ask_below: float = 0.0,
        escalate_below: float = -2.0,
        min_mass: float = 0.05,
        baseline_request: str = BASELINE_REQUEST,
    ) -> None:
        self._chat = chat
        self._operation_names = frozenset(
            ops_table.names() if operation_names is None else operation_names
        )
        self._ask_below = ask_below
        self._escalate_below = escalate_below
        self._min_mass = min_mass
        self._baseline_request = baseline_request
        self._baselines: dict[str, float] = {}

    def verify(self, request_text: str, decision: TierDecision) -> VerifierVerdict | None:
        """Score *decision*. Returns ``None`` rather than raising, always."""
        operation = ops_table.get(decision.operation)
        if operation is None or operation.name not in self._operation_names:
            return None
        baseline = self._baseline(operation)
        if baseline is None:
            return None
        p_yes = self._p_yes(_question(request_text, operation, decision.args))
        if p_yes is None:
            return None
        calibrated = calibrated_logit(p_yes, baseline)
        return VerifierVerdict(p_yes=p_yes, calibrated=calibrated, action=self._action(calibrated))

    def _action(self, calibrated: float) -> str:
        if calibrated < self._escalate_below:
            return ESCALATE
        if calibrated < self._ask_below:
            return ASK
        return PROPOSE

    def _baseline(self, operation: Operation) -> float | None:
        """The content-free p(yes) for *operation*, measured once and kept."""
        known = self._baselines.get(operation.name)
        if known is not None:
            return known
        measured = self._p_yes(_question(self._baseline_request, operation, {}))
        if measured is not None:
            # A failed measurement is not kept: a server that was still
            # starting must not switch the check off for the daemon's life.
            self._baselines[operation.name] = measured
        return measured

    def _p_yes(self, question: str) -> float | None:
        try:
            scores = self._chat.score_next_token(question)
        except Exception:  # noqa: BLE001 -- a check must never break a request
            return None
        p_yes, mass = yes_no_probability(scores)
        if p_yes is None or mass < self._min_mass:
            return None
        return p_yes


# ---------------------------------------------------------------------------
# The per-call result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierOutcome:
    """How one routed request ended.

    Exactly one of ``handled_by`` and ``escalated_to`` is set: a tier
    answered (with a ``proposal`` or an ``explanation``), or the request
    needs the full agent and ``escalation_context`` carries what the tiers
    already found out.
    """

    handled_by: str | None = None
    escalated_to: str | None = None
    proposal: Proposal | None = None
    explanation: str | None = None
    declines: tuple[tuple[str, DeclineReason], ...] = ()
    escalation_context: tuple[tuple[str, str], ...] = ()
    verifier: VerifierVerdict | None = None
    operation: str | None = None
    args: dict[str, str] = field(default_factory=dict)


def _excerpt(text: object) -> str:
    """One inspection result, redacted and bounded, ready to leave the process."""
    raw = text if isinstance(text, str) else str(text)
    # Bound first: the redactor's cost grows much faster than its input, so a
    # tier handing over a megabyte must not stall the request. A secret cut
    # by the excerpt limit lies inside the slack and is still seen whole --
    # except a multi-line private-key block, which bounded_cut replaces with a
    # placeholder rather than leave the redactor half a block it cannot match.
    raw = bounded_cut(raw, EXCERPT_CHARS + REDACT_SLACK)
    cleaned = redact(raw.encode("utf-8", "replace")).decode("utf-8", "replace")
    if len(cleaned) > EXCERPT_CHARS:
        return cleaned[: EXCERPT_CHARS - 3] + "..."
    return cleaned


def _inspections(result: Decline | Explanation) -> tuple[tuple[str, str], ...]:
    raw = getattr(result, "inspections", ()) or ()
    return tuple((str(operation), _excerpt(excerpt)) for operation, excerpt in raw)


class Route:
    """One routed request: iterate it for events, then read :attr:`outcome`.

    Iterating yields :class:`~nvsh.agent.base.AgentEvent` objects in the same
    shapes an adapter yields, so a client streams a tier answer exactly as it
    streams the full agent's. Every event carries the tier's name in
    ``args["tier"]``, which is what lets the panel say who answered.

    A handled request ends with ``DONE``. An escalated one ends with a
    ``STATUS`` and **no** ``DONE``: the caller goes on to the full agent.
    """

    def __init__(self, router: "TierRouter", request: AgentRequest, context: AgentContext) -> None:
        self._router = router
        self._request = request
        self._context = context
        self._declines: list[tuple[str, DeclineReason]] = []
        self._inspected: list[tuple[str, str]] = []
        self._verdict: VerifierVerdict | None = None
        self._outcome: TierOutcome | None = None

    # -- public surface --

    def __iter__(self) -> Iterator[AgentEvent]:
        return self._run()

    @property
    def outcome(self) -> TierOutcome | None:
        """How the request ended, or ``None`` while the route is still running."""
        return self._outcome

    def record_decision(self, decision: str) -> None:
        """Write the follow-up record naming what the operator did.

        The router never learns the operator's answer -- the client does,
        after approval -- so this is how an ``approved``/``declined`` lands
        in the measurement log against the tier that proposed it.
        """
        outcome = self._outcome
        if outcome is None:
            return
        self._router.records.write(
            TierRecord(
                tier=outcome.handled_by or AGENT,
                request_kind=self._request.kind.value,
                operation=outcome.operation,
                args=dict(outcome.args),
                operator_decision=decision,
                request_text=self._request_text(),
            )
        )

    # -- the routing itself --

    def _run(self) -> Iterator[AgentEvent]:
        order = self._order()
        for position, tier in enumerate(order):
            following = order[position + 1].name if position + 1 < len(order) else AGENT
            handled = yield from self._consult(tier, following)
            if handled:
                return
        yield self._status("no local answer; asking the full agent", AGENT)
        self._outcome = TierOutcome(
            escalated_to=AGENT,
            declines=tuple(self._declines),
            escalation_context=tuple(self._inspected),
            verifier=self._verdict,
        )

    def _order(self) -> list[Tier]:
        """The tiers to consult, in order, for this request's kind.

        A FAILURE never reaches Tier 1: Needle selects an operation from a
        request phrased as an instruction, and a failed command is not one.
        """
        router = self._router
        if self._request.kind is RequestKind.FAILURE:
            candidates = [router.tier2]
        else:
            candidates = [router.tier1, router.tier2]
        return [tier for tier in candidates if tier is not None]

    def _consult(self, tier: Tier, following: str) -> Iterator[AgentEvent]:
        """Ask one tier. Yields its events; returns True when it answered."""
        started = self._router.clock()
        selection = self._select(tier)
        latency_ms = (self._router.clock() - started) * 1000.0

        if isinstance(selection, Explanation):
            yield from self._explained(tier, selection, latency_ms)
            return True

        decision = selection if isinstance(selection, TierDecision) else None
        if decision is not None:
            built = self._build(tier, decision)
            if isinstance(built, Proposal):
                yield from self._proposed(tier, decision, built, latency_ms)
                return True
            selection = built

        yield from self._declined(tier, decision, selection, latency_ms, following)
        return False

    def _select(self, tier: Tier) -> TierDecision | Decline | Explanation:
        """``tier.select()``, with any exception turned into a decline."""
        try:
            result = tier.select(self._request, self._context)
        except Exception as exc:  # noqa: BLE001 -- a tier crash costs one answer
            return Decline(DeclineReason.TIER_ERROR, f"{type(exc).__name__}: {exc}")
        if isinstance(result, (TierDecision, Decline, Explanation)):
            return result
        return Decline(
            DeclineReason.MALFORMED_OUTPUT,
            f"tier returned {type(result).__name__}, not a decision",
        )

    def _build(self, tier: Tier, decision: TierDecision) -> Proposal | Decline:
        """Chain a decision to a proposal: floor -> table -> ground -> render -> verifier."""
        floor = self._router.min_confidence
        if decision.confidence is not None and decision.confidence < floor:
            return Decline(
                DeclineReason.LOW_CONFIDENCE,
                f"confidence {decision.confidence} below floor {floor}",
            )

        operation = ops_table.get(decision.operation)
        if operation is None:
            return Decline(
                DeclineReason.UNKNOWN_OPERATION, f"unknown operation {decision.operation!r}"
            )

        grounded = ops_ground.ground(operation, dict(decision.args), self._router.runner)
        if isinstance(grounded, ops_ground.GroundDecline):
            return Decline(DeclineReason.NOT_GROUNDED, grounded.message)

        argv = render_argv(operation.name, dict(grounded.args), self._router.platform)
        if argv is None:
            return Decline(
                DeclineReason.NOT_RENDERABLE,
                f"no single command renders {operation.name} on this machine",
            )
        # The verifier runs last: the table, grounding and rendering are the
        # cheap, deterministic checks, so a pick they refuse never costs a
        # model call -- and the question is asked about the grounded
        # arguments, which are the ones the operator would be shown.
        if tier is self._router.tier1:
            checked = TierDecision(
                operation=operation.name,
                args=dict(grounded.args),
                confidence=decision.confidence,
                read_only=operation.read_only,
            )
            self._verdict = self._verify(checked)
            if self._verdict is not None and self._verdict.action == ESCALATE:
                return Decline(DeclineReason.LOW_CONFIDENCE, "the confidence check said no")
        return self._proposal(tier, operation, grounded.args, argv)

    def _verify(self, decision: TierDecision) -> VerifierVerdict | None:
        verifier = self._router.verifier
        if verifier is None:
            return None
        try:
            verdict = verifier.verify(self._request_text(), decision)
        except Exception:  # noqa: BLE001 -- an unavailable check is no check
            return None
        return verdict if isinstance(verdict, VerifierVerdict) else None

    def _proposal(
        self, tier: Tier, operation: Operation, args: dict[str, str], argv: list[str]
    ) -> Proposal:
        """The proposal for a grounded, rendered operation.

        ``command`` is built only from ``render()``'s argv. The rationale
        always names the interpreted operation and its arguments, at every
        confidence -- 1.0 and ``None`` read identically -- because that, not
        the confidence, is what lets the operator catch a wrong pick.
        ``kind`` comes from the *table's* ``read_only`` flag, never from the
        tier's own claim about it.
        """
        rationale = f"{tier.name} read this as: {describe(operation, args)}"
        if self._verdict is not None and self._verdict.action == ASK:
            rationale += " (the confidence check was unsure about it)"
        kind = ProposalKind.INSPECT if operation.read_only else ProposalKind.FIX
        return Proposal(command=shlex.join(argv), rationale=rationale, kind=kind)

    # -- the three endings --

    def _proposed(
        self, tier: Tier, decision: TierDecision, proposal: Proposal, latency_ms: float
    ) -> Iterator[AgentEvent]:
        self._write(tier, decision, None, latency_ms, None)
        yield self._status(f"{tier.name} is answering", tier.name, **self._verdict_args())
        yield AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal, args={"tier": tier.name})
        yield AgentEvent(kind=EventKind.DONE, args={"tier": tier.name})
        self._outcome = TierOutcome(
            handled_by=tier.name,
            proposal=proposal,
            declines=tuple(self._declines),
            escalation_context=tuple(self._inspected),
            verifier=self._verdict,
            operation=decision.operation,
            args=dict(decision.args),
        )

    def _explained(
        self, tier: Tier, explanation: Explanation, latency_ms: float
    ) -> Iterator[AgentEvent]:
        self._inspected.extend(_inspections(explanation))
        self._write(tier, None, None, latency_ms, None)
        yield self._status(f"{tier.name} is answering", tier.name)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=explanation.text, args={"tier": tier.name})
        yield AgentEvent(kind=EventKind.DONE, args={"tier": tier.name})
        self._outcome = TierOutcome(
            handled_by=tier.name,
            explanation=explanation.text,
            declines=tuple(self._declines),
            escalation_context=tuple(self._inspected),
            verifier=self._verdict,
        )

    def _declined(
        self,
        tier: Tier,
        decision: TierDecision | None,
        decline: Decline,
        latency_ms: float,
        following: str,
    ) -> Iterator[AgentEvent]:
        self._declines.append((tier.name, decline.reason))
        self._inspected.extend(_inspections(decline))
        self._write(tier, decision, decline, latency_ms, following)
        if decline.reason is DeclineReason.TIER_UNAVAILABLE:
            # The one status a missing or broken flavor is allowed to cost
            # (spec c17): one line, then the request moves up.
            detail = decline.detail or decline.reason.value
            yield self._status(f"{tier.name} is unavailable: {detail}", tier.name)

    # -- shared plumbing --

    def _write(
        self,
        tier: Tier,
        decision: TierDecision | None,
        decline: Decline | None,
        latency_ms: float,
        escalated_to: str | None,
    ) -> None:
        self._router.records.write(
            TierRecord(
                tier=tier.name,
                request_kind=self._request.kind.value,
                operation=decision.operation if decision is not None else None,
                args=dict(decision.args) if decision is not None else {},
                confidence=decision.confidence if decision is not None else None,
                latency_ms=latency_ms,
                decline_reason=decline.reason.value if decline is not None else None,
                escalated_to=escalated_to,
                request_text=self._request_text(),
                verifier=self._verdict_args() if tier is self._router.tier1 else {},
            )
        )

    def _verdict_args(self) -> dict[str, float | str]:
        """The verifier's numbers, for the bench and the panel. Empty when absent."""
        verdict = self._verdict
        if verdict is None:
            return {}
        args: dict[str, float | str] = {"verifier_action": verdict.action}
        if verdict.p_yes is not None:
            args["p_yes"] = verdict.p_yes
        if verdict.calibrated is not None:
            args["calibrated"] = verdict.calibrated
        return args

    def _status(self, text: str, tier: str, **extra: float | str) -> AgentEvent:
        return AgentEvent(kind=EventKind.STATUS, text=text, args={"tier": tier, **extra})

    def _request_text(self) -> str:
        return self._request.prompt or self._request.command


class TierRouter:
    """Routes one request through the configured tiers. Shared across threads.

    ``tier1``/``tier2`` may each be ``None`` (flavor not installed, or the
    operator has not enabled it): a missing tier is skipped without a status
    line, and a request with no tier at all goes straight to the full agent.
    """

    def __init__(
        self,
        tier1: Tier | None,
        tier2: Tier | None,
        records: TierRecords,
        platform: Platform,
        *,
        runner: ops_ground.Runner = ops_ground.default_runner,
        verifier: Verifier | None = None,
        min_confidence: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.tier1 = tier1
        self.tier2 = tier2
        self.records = records
        self.platform = platform
        self.runner = runner
        self.verifier = verifier
        self.min_confidence = min_confidence
        self.clock = clock

    def route(self, request: AgentRequest, context: AgentContext) -> Route:
        """Return a fresh :class:`Route` for *request*. Nothing runs until it is iterated."""
        return Route(self, request, context)
