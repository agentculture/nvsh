"""Tests for the tier router: which tier answers, what it records, what it proposes.

Covers spec targets c7 (a tier never executes and never emits shell), c10
(confidence is recorded but is not the safety mechanism), c17 (a broken
flavor reaches the full agent with one status line) and their honesty
conditions h6, h9, h16.

Every fixture tier here is inert data: the router is given ``FakeTier``
scripts and a fake ``runner``, and no test lets anything run a command --
the one place a command is even handed on is ``run_loop``, whose
``executor`` is a recorder.
"""

from __future__ import annotations

import shlex

import pytest

from nvsh.agent.audit import AuditLog
from nvsh.agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    EventKind,
    Proposal,
    ProposalKind,
    RequestKind,
)
from nvsh.agent.fake import FakeAgent
from nvsh.agent.loop import ExecResult, run_loop
from nvsh.approvals import command_refusal_reason
from nvsh.ops import ground as ops_ground
from nvsh.ops import table as ops_table
from nvsh.ops.render import render
from nvsh.platform._model import Platform
from nvsh.tiers.base import Decline, DeclineReason, Explanation, Tier, TierDecision
from nvsh.tiers.fake import FakeTier
from nvsh.tiers.records import TierRecords
from nvsh.tiers.router import AGENT, LogprobVerifier, TierRouter, VerifierVerdict

_PLATFORM = Platform(kind="jetson")

_UNITS = "nginx.service loaded active running\nvllm.service loaded active running\n"


def _runner(argv: list[str], timeout: float) -> tuple[int, str]:
    """Stand-in for ``nvsh.ops.ground.default_runner``; runs nothing."""
    if argv[:1] == ["systemctl"]:
        return (0, _UNITS)
    if argv[:1] == ["docker"]:
        return (0, "trainer\n")
    return (127, "")


def _pick(operation: str, args: dict | None = None, confidence: float | None = None):
    op = ops_table.get(operation)
    assert op is not None, operation
    return TierDecision(
        operation=operation,
        args=dict(args or {}),
        confidence=confidence,
        read_only=op.read_only,
    )


def _request(kind: RequestKind = RequestKind.EXPLICIT, prompt: str = "restart nginx"):
    return AgentRequest(kind=kind, prompt=prompt)


def _context() -> AgentContext:
    return AgentContext(platform="jetson")


class _ExplainingTier(Tier):
    """A Tier 2 stand-in that answers in plain words (its third outcome)."""

    def __init__(self, explanation: Explanation, name: str = "lfm") -> None:
        self.name = name
        self._explanation = explanation

    def select(self, request, context):
        return self._explanation

    def close(self) -> None:
        """Nothing to release."""


@pytest.fixture
def records(tmp_path) -> TierRecords:
    return TierRecords(path=tmp_path / "tiers.jsonl")


def _router(records: TierRecords, tier1=None, tier2=None, **kwargs) -> TierRouter:
    return TierRouter(tier1, tier2, records, _PLATFORM, runner=_runner, **kwargs)


def _proposals(events: list[AgentEvent]) -> list[Proposal]:
    return [e.proposal for e in events if e.kind == EventKind.PROPOSAL and e.proposal]


def _statuses(events: list[AgentEvent]) -> list[str]:
    return [e.text for e in events if e.kind == EventKind.STATUS]


def _kinds(events: list[AgentEvent]) -> list[EventKind]:
    return [e.kind for e in events]


# ---------------------------------------------------------------------------
# h6: one request through each path, with the record naming tier and reason
# ---------------------------------------------------------------------------


def test_answered_at_tier1_outcome_names_the_tier(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], name="needle")
    route = _router(records, tier1, FakeTier([], name="lfm")).route(_request(), _context())
    list(route)
    assert route.outcome.handled_by == "needle"


def test_answered_at_tier1_writes_one_record_naming_the_tier(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], name="needle")
    list(_router(records, tier1).route(_request(), _context()))
    assert [(r["tier"], r["operation"]) for r in records.read_all()] == [
        ("needle", "service_status")
    ]


def test_answered_at_tier1_never_consults_tier2(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], name="needle")
    tier2 = FakeTier([], name="lfm")
    list(_router(records, tier1, tier2).route(_request(), _context()))
    assert tier2.requests_seen == []


def test_declined_by_tier1_is_answered_by_tier2(records):
    tier1 = FakeTier([Decline(DeclineReason.NO_CALL, "nothing selected")], name="needle")
    tier2 = FakeTier([_pick("service_restart", {"service": "nginx"})], name="lfm")
    route = _router(records, tier1, tier2).route(_request(), _context())
    list(route)
    assert route.outcome.handled_by == "lfm"


def test_tier1_decline_record_names_the_reason_and_the_next_tier(records):
    tier1 = FakeTier([Decline(DeclineReason.NO_CALL, "nothing selected")], name="needle")
    tier2 = FakeTier([_pick("service_restart", {"service": "nginx"})], name="lfm")
    list(_router(records, tier1, tier2).route(_request(), _context()))
    first = records.read_all()[0]
    assert first["tier"] == "needle"
    assert first["decline_reason"] == "no_call"
    assert first["escalated_to"] == "lfm"


def test_declined_by_both_escalates_to_the_full_agent(records):
    tier1 = FakeTier([Decline(DeclineReason.NO_CALL)], name="needle")
    tier2 = FakeTier([Decline(DeclineReason.LOOP_LIMIT)], name="lfm")
    route = _router(records, tier1, tier2).route(_request(), _context())
    list(route)
    assert route.outcome.handled_by is None
    assert route.outcome.escalated_to == AGENT


def test_declined_by_both_records_each_tier_with_its_reason(records):
    tier1 = FakeTier([Decline(DeclineReason.NO_CALL)], name="needle")
    tier2 = FakeTier([Decline(DeclineReason.LOOP_LIMIT)], name="lfm")
    list(_router(records, tier1, tier2).route(_request(), _context()))
    assert [(r["tier"], r["decline_reason"], r["escalated_to"]) for r in records.read_all()] == [
        ("needle", "no_call", "lfm"),
        ("lfm", "loop_limit", AGENT),
    ]


def test_declined_by_both_reports_both_decline_reasons(records):
    tier1 = FakeTier([Decline(DeclineReason.NO_CALL)], name="needle")
    tier2 = FakeTier([Decline(DeclineReason.LOOP_LIMIT)], name="lfm")
    route = _router(records, tier1, tier2).route(_request(), _context())
    list(route)
    assert route.outcome.declines == (
        ("needle", DeclineReason.NO_CALL),
        ("lfm", DeclineReason.LOOP_LIMIT),
    )


def test_escalation_yields_a_final_status_and_no_done(records):
    tier1 = FakeTier([Decline(DeclineReason.NO_CALL)], name="needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert EventKind.DONE not in _kinds(events)


def test_a_handled_request_ends_with_done(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], name="needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert _kinds(events) == [EventKind.STATUS, EventKind.PROPOSAL, EventKind.DONE]


def test_status_events_name_the_answering_tier(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], name="needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert all(e.args.get("tier") == "needle" for e in events)


# ---------------------------------------------------------------------------
# c11 / Tier 2's three outcomes: escalation carries what was inspected
# ---------------------------------------------------------------------------


def test_escalation_context_carries_the_redacted_inspection(records):
    secret = "HF_" + "TOKEN=" + "x" * 20
    tier2 = FakeTier(
        [
            Decline(
                DeclineReason.LOOP_LIMIT,
                inspections=(("service_logs", f"boot ok\n{secret}"),),
            )
        ],
        name="lfm",
    )
    route = _router(records, None, tier2).route(_request(RequestKind.FAILURE), _context())
    list(route)
    assert route.outcome.escalation_context == (
        ("service_logs", "boot ok\nHF_TOKEN=<REDACTED:env_assignment>"),
    )


def test_escalation_context_excerpts_are_bounded(records):
    tier2 = FakeTier(
        [Decline(DeclineReason.LOOP_LIMIT, inspections=(("process_list", "y" * 9000),))],
        name="lfm",
    )
    route = _router(records, None, tier2).route(_request(RequestKind.FAILURE), _context())
    list(route)
    assert len(route.outcome.escalation_context[0][1]) <= 2048


def test_an_explanation_is_streamed_as_text_and_done(records):
    tier2 = _ExplainingTier(Explanation(text="the disk is full"))
    events = list(_router(records, None, tier2).route(_request(), _context()))
    assert [(e.kind, e.text) for e in events if e.kind == EventKind.TEXT_DELTA] == [
        (EventKind.TEXT_DELTA, "the disk is full")
    ]


def test_an_explanation_proposes_nothing(records):
    tier2 = _ExplainingTier(Explanation(text="the disk is full"))
    route = _router(records, None, tier2).route(_request(), _context())
    events = list(route)
    assert _proposals(events) == []
    assert route.outcome.handled_by == "lfm"


# ---------------------------------------------------------------------------
# Which tier a request starts at
# ---------------------------------------------------------------------------


def test_a_failure_request_never_reaches_tier1(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], name="needle")
    tier2 = FakeTier([Decline(DeclineReason.LOOP_LIMIT)], name="lfm")
    list(_router(records, tier1, tier2).route(_request(RequestKind.FAILURE), _context()))
    assert tier1.requests_seen == []


def test_a_failure_request_starts_at_tier2(records):
    tier1 = FakeTier([], name="needle")
    tier2 = FakeTier([_pick("service_status", {"service": "nginx"})], name="lfm")
    route = _router(records, tier1, tier2).route(_request(RequestKind.FAILURE), _context())
    list(route)
    assert route.outcome.handled_by == "lfm"


@pytest.mark.parametrize("kind", [RequestKind.EXPLICIT, RequestKind.SLASH])
def test_an_explicit_or_slash_request_tries_tier1_first(records, kind):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], name="needle")
    tier2 = FakeTier([], name="lfm")
    list(_router(records, tier1, tier2).route(_request(kind), _context()))
    assert len(tier1.requests_seen) == 1


# ---------------------------------------------------------------------------
# c17 / h16: an unavailable tier costs exactly one status line (owed by t10)
# ---------------------------------------------------------------------------


def test_an_unavailable_tier_escalates_to_the_agent(records):
    tier1 = FakeTier([Decline(DeclineReason.TIER_UNAVAILABLE, "needle import failed")], "needle")
    route = _router(records, tier1).route(_request(), _context())
    list(route)
    assert route.outcome.escalated_to == AGENT


def test_an_unavailable_tier_costs_exactly_one_status_line(records):
    tier1 = FakeTier([Decline(DeclineReason.TIER_UNAVAILABLE, "needle import failed")], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert len([text for text in _statuses(events) if "needle" in text]) == 1


def test_an_unconfigured_tier_is_skipped_silently(records):
    route = _router(records, None, None).route(_request(), _context())
    events = list(route)
    assert len(_statuses(events)) == 1


def test_a_tier_that_raises_is_recorded_as_a_tier_error(records):
    tier1 = FakeTier([RuntimeError("the child died")], name="needle")
    list(_router(records, tier1).route(_request(), _context()))
    assert records.read_all()[0]["decline_reason"] == DeclineReason.TIER_ERROR.value


# ---------------------------------------------------------------------------
# c10 / h9: a wrong mutating operation at confidence 1.0 -- and at None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [1.0, None])
def test_a_mutating_pick_becomes_a_fix_proposal(records, confidence):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"}, confidence)], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert _proposals(events)[0].kind == ProposalKind.FIX


@pytest.mark.parametrize("confidence", [1.0, None])
def test_a_mutating_pick_shows_the_interpreted_operation_and_arguments(records, confidence):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"}, confidence)], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert "service_restart service=nginx.service" in _proposals(events)[0].rationale


@pytest.mark.parametrize("confidence", [1.0, None])
def test_a_mutating_pick_executes_nothing_without_approval(records, tmp_path, confidence):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"}, confidence)], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    executed = _run_through_loop(events, tmp_path, approve=lambda proposal: False)
    assert executed == []


def test_a_read_only_pick_becomes_an_inspect_proposal(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert _proposals(events)[0].kind == ProposalKind.INSPECT


def test_the_recorded_confidence_is_the_tier_s_own(records):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"}, 1.0)], "needle")
    list(_router(records, tier1).route(_request(), _context()))
    assert records.read_all()[0]["confidence"] == 1.0


def test_a_confidence_below_the_router_floor_declines(records):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"}, 0.1)], "needle")
    route = _router(records, tier1, min_confidence=0.5).route(_request(), _context())
    list(route)
    assert route.outcome.declines == (("needle", DeclineReason.LOW_CONFIDENCE),)


# ---------------------------------------------------------------------------
# c7 / h6: the command comes from render(), grounding gates the arguments
# ---------------------------------------------------------------------------


def test_the_proposed_command_is_exactly_the_rendered_argv(records):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    expected = render("service_restart", {"service": "nginx.service"}, _PLATFORM)
    assert _proposals(events)[0].command == shlex.join(expected)


def test_an_ungroundable_argument_declines_and_falls_to_the_next_tier(records):
    tier1 = FakeTier([_pick("service_restart", {"service": "rm -rf /"})], "needle")
    tier2 = FakeTier([_pick("memory_stats")], name="lfm")
    route = _router(records, tier1, tier2).route(_request(), _context())
    list(route)
    assert route.outcome.declines == (("needle", DeclineReason.NOT_GROUNDED),)


def test_an_unrenderable_operation_declines(records):
    tier1 = FakeTier([_pick("machine_status")], "needle")
    route = _router(records, tier1).route(_request(), _context())
    list(route)
    assert route.outcome.declines == (("needle", DeclineReason.NOT_RENDERABLE),)


def test_an_operation_missing_from_the_table_declines(records):
    lie = TierDecision(operation="reboot_everything", args={}, confidence=1.0, read_only=False)
    route = _router(records, FakeTier([lie], "needle")).route(_request(), _context())
    list(route)
    assert route.outcome.declines == (("needle", DeclineReason.UNKNOWN_OPERATION),)


def test_a_mutating_operation_is_never_proposed_as_inspect(records):
    """A tier that lies about ``read_only`` cannot downgrade the proposal kind."""
    lie = TierDecision(
        operation="service_restart", args={"service": "nginx"}, confidence=1.0, read_only=True
    )
    events = list(_router(records, FakeTier([lie], "needle")).route(_request(), _context()))
    assert _proposals(events)[0].kind == ProposalKind.FIX


# ---------------------------------------------------------------------------
# h16: the existing approval / sudo / destructive-command policy is unchanged
# ---------------------------------------------------------------------------


def _run_through_loop(events, tmp_path, approve):
    """Drive router events through the existing run_loop, recording executions."""
    executed: list[str] = []

    def executor(command: str) -> ExecResult:
        executed.append(command)
        return ExecResult(exit_code=0)

    run_loop(
        FakeAgent(list(events)),
        _request(),
        _context(),
        approve,
        executor,
        lambda event: None,
        AuditLog(path=tmp_path / "audit.jsonl"),
    )
    return executed


def test_a_rendered_sudo_command_is_never_auto_approved(records):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    assert command_refusal_reason(_proposals(events)[0].command) is not None


def test_an_approved_router_proposal_runs_the_rendered_command(records, tmp_path):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    executed = _run_through_loop(events, tmp_path, approve=lambda proposal: True)
    assert executed == ["systemctl status --no-pager nginx.service"]


def test_a_router_proposal_is_audited_like_any_agent_proposal(records, tmp_path):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    events = list(_router(records, tier1).route(_request(), _context()))
    _run_through_loop(events, tmp_path, approve=lambda proposal: False)
    entries = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(entries) == 2  # one "proposal", one "decision", no "outcome"


def test_the_router_itself_runs_nothing(records):
    """The only subprocess-shaped callable the router gets is the ground runner."""
    seen: list[list[str]] = []

    def watching_runner(argv: list[str], timeout: float) -> tuple[int, str]:
        seen.append(list(argv))
        return _runner(argv, timeout)

    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    router = TierRouter(tier1, None, records, _PLATFORM, runner=watching_runner)
    list(router.route(_request(), _context()))
    assert seen == [list(ops_ground.SERVICE_LOOKUP_ARGV)]


# ---------------------------------------------------------------------------
# Deviation d1: the optional verifier after a Tier 1 pick
# ---------------------------------------------------------------------------


class _ScriptedVerifier:
    def __init__(self, verdict) -> None:
        self._verdict = verdict
        self.calls = 0

    def verify(self, request_text: str, decision: TierDecision):
        self.calls += 1
        if isinstance(self._verdict, BaseException):
            raise self._verdict
        return self._verdict


def test_a_verifier_that_escalates_makes_tier1_decline(records):
    verifier = _ScriptedVerifier(VerifierVerdict(p_yes=0.1, calibrated=-3.0, action="escalate"))
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    route = _router(records, tier1, verifier=verifier).route(_request(), _context())
    list(route)
    assert route.outcome.declines == (("needle", DeclineReason.LOW_CONFIDENCE),)


def test_a_verifier_that_is_unsure_still_proposes(records):
    verifier = _ScriptedVerifier(VerifierVerdict(p_yes=0.5, calibrated=-0.2, action="ask"))
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    events = list(_router(records, tier1, verifier=verifier).route(_request(), _context()))
    assert "unsure" in _proposals(events)[0].rationale


def test_a_verifier_that_raises_is_treated_as_absent(records):
    verifier = _ScriptedVerifier(RuntimeError("no server"))
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    route = _router(records, tier1, verifier=verifier).route(_request(), _context())
    list(route)
    assert route.outcome.verifier is None


def test_the_verifier_numbers_ride_the_status_event(records):
    verifier = _ScriptedVerifier(VerifierVerdict(p_yes=0.9, calibrated=1.5, action="propose"))
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    events = list(_router(records, tier1, verifier=verifier).route(_request(), _context()))
    status = [e for e in events if e.kind == EventKind.STATUS][0]
    assert status.args.get("p_yes") == 0.9
    assert status.args.get("calibrated") == 1.5


def test_the_verifier_is_not_consulted_for_tier2(records):
    verifier = _ScriptedVerifier(VerifierVerdict(p_yes=0.1, calibrated=-3.0, action="escalate"))
    tier2 = FakeTier([_pick("service_restart", {"service": "nginx"})], "lfm")
    list(_router(records, None, tier2, verifier=verifier).route(_request(), _context()))
    assert verifier.calls == 0


class _FakeChat:
    """A ToolChat-like stand-in: one score_next_token per distinct prompt."""

    def __init__(self, yes_logprob: float = -0.1, no_logprob: float = -2.0) -> None:
        self.prompts: list[str] = []
        self._yes = yes_logprob
        self._no = no_logprob

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        self.prompts.append(prompt)
        if "N/A" in prompt:
            return {"yes": -2.0, "no": -0.1}
        return {"yes": self._yes, "no": self._no}


def test_the_logprob_verifier_asks_the_fixed_yes_no_question():
    chat = _FakeChat()
    LogprobVerifier(chat).verify("restart nginx", _pick("service_restart", {"service": "nginx"}))
    assert chat.prompts[-1].endswith("Correct? Answer yes or no:")


def test_the_logprob_verifier_computes_a_baseline_only_once():
    chat = _FakeChat()
    verifier = LogprobVerifier(chat)
    decision = _pick("service_restart", {"service": "nginx"})
    verifier.verify("restart nginx", decision)
    verifier.verify("restart nginx please", decision)
    assert len([p for p in chat.prompts if "N/A" in p]) == 1


def test_the_logprob_verifier_reports_a_calibrated_lift():
    verdict = LogprobVerifier(_FakeChat()).verify(
        "restart nginx", _pick("service_restart", {"service": "nginx"})
    )
    assert verdict.calibrated > 0


def test_the_logprob_verifier_escalates_a_strong_no():
    chat = _FakeChat(yes_logprob=-5.0, no_logprob=-0.01)
    verdict = LogprobVerifier(chat).verify(
        "restart nginx", _pick("service_restart", {"service": "nginx"})
    )
    assert verdict.action == "escalate"


def test_the_logprob_verifier_returns_none_when_the_server_fails():
    class _BrokenChat:
        def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
            raise OSError("connection refused")

    verdict = LogprobVerifier(_BrokenChat()).verify("restart nginx", _pick("machine_status"))
    assert verdict is None


def test_the_logprob_verifier_declines_an_operation_outside_the_table():
    lie = TierDecision(operation="reboot_everything", args={}, confidence=None, read_only=False)
    assert LogprobVerifier(_FakeChat()).verify("restart nginx", lie) is None


# ---------------------------------------------------------------------------
# The follow-up record the client writes once the operator has decided
# ---------------------------------------------------------------------------


def test_record_decision_writes_a_follow_up_record(records):
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    route = _router(records, tier1).route(_request(), _context())
    list(route)
    route.record_decision("approved")
    assert [r.get("operator_decision") for r in records.read_all()] == [None, "approved"]


def test_record_decision_names_the_agent_when_the_request_escalated(records):
    tier1 = FakeTier([Decline(DeclineReason.NO_CALL)], "needle")
    route = _router(records, tier1).route(_request(), _context())
    list(route)
    route.record_decision("declined")
    assert records.read_all()[-1]["tier"] == AGENT


def test_the_outcome_is_none_until_the_route_is_drained(records):
    tier1 = FakeTier([_pick("service_status", {"service": "nginx"})], "needle")
    route = _router(records, tier1).route(_request(), _context())
    assert route.outcome is None


def test_a_wrong_mutating_pick_for_a_read_only_request_still_needs_approval(records, tmp_path):
    """The h9 case: high confidence, plainly wrong operation, nothing runs."""
    asked = _request(prompt="show me the nginx logs")
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"}, 1.0)], "needle")
    events = list(_router(records, tier1).route(asked, _context()))
    executed = _run_through_loop(events, tmp_path, approve=lambda proposal: False)
    assert executed == []
    assert "service_restart" in _proposals(events)[0].rationale


def test_a_pick_grounding_refuses_never_costs_a_verifier_call(records):
    verifier = _ScriptedVerifier(VerifierVerdict(action="propose"))
    tier1 = FakeTier([_pick("service_restart", {"service": "rm -rf /"})], "needle")
    list(_router(records, tier1, verifier=verifier).route(_request(), _context()))
    assert verifier.calls == 0


class _RecordingVerifier(_ScriptedVerifier):
    def verify(self, request_text: str, decision: TierDecision):
        self.seen = decision
        return super().verify(request_text, decision)


def test_the_verifier_is_asked_about_the_grounded_arguments(records):
    verifier = _RecordingVerifier(VerifierVerdict(action="propose"))
    tier1 = FakeTier([_pick("service_restart", {"service": "nginx"})], "needle")
    list(_router(records, tier1, verifier=verifier).route(_request(), _context()))
    assert verifier.seen.args == {"service": "nginx.service"}


class _LateChat(_FakeChat):
    """A server that is not up for the first call, then answers."""

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        if not self.prompts:
            self.prompts.append(prompt)
            raise OSError("connection refused")
        return super().score_next_token(prompt, top=top)


def test_a_failed_baseline_measurement_is_retried_on_the_next_request():
    verifier = LogprobVerifier(_LateChat())
    pick = _pick("service_restart", {"service": "nginx.service"})
    first = verifier.verify("restart nginx", pick)
    second = verifier.verify("restart nginx", pick)
    assert first is None
    assert second is not None


def test_the_verifier_numbers_land_in_the_tier1_record(records):
    verifier = _ScriptedVerifier(VerifierVerdict(p_yes=0.9, calibrated=1.5, action="propose"))
    tier1 = FakeTier([_pick("memory_stats")], "needle")
    list(_router(records, tier1, verifier=verifier).route(_request(), _context()))
    assert records.read_all()[0]["verifier"] == {
        "verifier_action": "propose",
        "p_yes": 0.9,
        "calibrated": 1.5,
    }


def test_fake_tier_passes_an_explanation_through():
    explanation = Explanation(text="disk is full")
    assert FakeTier([explanation]).select(_request(), _context()) is explanation
