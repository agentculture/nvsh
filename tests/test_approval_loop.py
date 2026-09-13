"""Drives FakeAgent through inspect -> dangerous proposal -> retry -> verify
via run_loop, and checks the audit trail and the "never via READLINE_LINE"
invariant.
"""

from __future__ import annotations

import os

from nvsh.agent import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    AuditLog,
    EventKind,
    ExecResult,
    FakeAgent,
    Proposal,
    ProposalKind,
    RequestKind,
    run_loop,
)


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="rm /x", exit_code=1)


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="rm: /x: Operation not permitted", cwd="/")


def _script() -> list[AgentEvent]:
    return [
        AgentEvent(kind=EventKind.STATUS, text="inspecting"),
        AgentEvent(
            kind=EventKind.PROPOSAL,
            proposal=Proposal(
                command="ls -la /x", rationale="see what is there", kind=ProposalKind.INSPECT
            ),
        ),
        AgentEvent(
            kind=EventKind.PROPOSAL,
            proposal=Proposal(
                command="sudo rm -rf /x", rationale="force removal", kind=ProposalKind.FIX
            ),
        ),
        AgentEvent(
            kind=EventKind.PROPOSAL,
            proposal=Proposal(command="rm /x", rationale="retry now", kind=ProposalKind.RETRY),
        ),
        AgentEvent(
            kind=EventKind.PROPOSAL,
            proposal=Proposal(
                command="test -e /x", rationale="confirm gone", kind=ProposalKind.VERIFY
            ),
        ),
        AgentEvent(kind=EventKind.DONE),
    ]


def test_dangerous_proposal_never_runs_until_approved(tmp_path):
    agent = FakeAgent(_script())
    audit = AuditLog(path=tmp_path / "audit.jsonl")

    order: list[str] = []
    rendered: list[Proposal] = []

    def on_event(event: AgentEvent) -> None:
        if event.kind == EventKind.PROPOSAL:
            rendered.append(event.proposal)
            order.append(f"render:{event.proposal.command}")

    def approve(proposal: Proposal) -> bool:
        order.append(f"approve-called:{proposal.command}")
        # At the moment approve() is invoked, this proposal has been
        # rendered but not yet executed.
        assert f"exec:{proposal.command}" not in order
        # Only the dangerous fix is ever refused in this scenario; every
        # other proposal (inspect/retry/verify) is approved. The loop
        # itself has no opinion on this -- it is this callback's policy.
        return True

    def executor(command: str) -> ExecResult:
        order.append(f"exec:{command}")
        if command == "test -e /x":
            return ExecResult(exit_code=1)  # /x no longer exists: verified
        return ExecResult(exit_code=0)

    results = run_loop(
        agent=agent,
        request=_request(),
        context=_context(),
        approve=approve,
        executor=executor,
        on_event=on_event,
        audit=audit,
    )

    # All four proposals were rendered, in script order.
    assert [p.command for p in rendered] == [
        "ls -la /x",
        "sudo rm -rf /x",
        "rm /x",
        "test -e /x",
    ]
    # Each proposal is rendered and approve() is called for it *before* its
    # command is ever executed -- the dangerous 'sudo rm -rf /x' included.
    assert order == [
        "render:ls -la /x",
        "approve-called:ls -la /x",
        "exec:ls -la /x",
        "render:sudo rm -rf /x",
        "approve-called:sudo rm -rf /x",
        "exec:sudo rm -rf /x",
        "render:rm /x",
        "approve-called:rm /x",
        "exec:rm /x",
        "render:test -e /x",
        "approve-called:test -e /x",
        "exec:test -e /x",
    ]
    assert [r.exit_code for r in results] == [0, 0, 0, 1]

    # The proposal never ran before approve() returned True: nothing in
    # 'order' has an exec: entry preceding its matching approve-called:
    # entry (already asserted inside approve() above, per-call).


def test_dangerous_proposal_withheld_until_approve_true(tmp_path):
    """A proposal approve() refuses is never handed to the executor."""
    agent = FakeAgent(_script())
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    executed: list[str] = []

    def approve(proposal: Proposal) -> bool:
        # Refuse only the dangerous fix; approve everything else.
        return proposal.command != "sudo rm -rf /x"

    def executor(command: str) -> ExecResult:
        executed.append(command)
        return ExecResult(exit_code=0)

    run_loop(
        agent=agent,
        request=_request(),
        context=_context(),
        approve=approve,
        executor=executor,
        on_event=lambda event: None,
        audit=audit,
    )

    assert "sudo rm -rf /x" not in executed
    assert executed == ["ls -la /x", "rm /x", "test -e /x"]


def test_audit_log_records_proposal_decision_outcome(tmp_path):
    agent = FakeAgent(_script())
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(path=audit_path)

    run_loop(
        agent=agent,
        request=_request(),
        context=_context(),
        approve=lambda proposal: True,
        executor=lambda command: ExecResult(exit_code=0),
        on_event=lambda event: None,
        audit=audit,
    )

    entries = audit.read_all()
    # proposal + decision + outcome for each of the 4 proposals = 12 lines.
    assert len(entries) == 12
    events_by_kind = [e["event"] for e in entries]
    assert (
        events_by_kind
        == [
            "proposal",
            "decision",
            "outcome",
        ]
        * 4
    )

    sudo_entries = [e for e in entries if e.get("proposal", {}).get("command") == "sudo rm -rf /x"]
    assert len(sudo_entries) == 3
    proposal_entry, decision_entry, outcome_entry = sudo_entries
    assert proposal_entry["event"] == "proposal"
    assert proposal_entry["proposal"]["kind"] == "fix"
    assert decision_entry["decision"] is True
    assert outcome_entry["outcome"]["exit_code"] == 0

    # Audit file permissions: 0600.
    mode = audit_path.stat().st_mode & 0o777
    assert mode == 0o600
    # Parent dir: 0700.
    parent_mode = audit_path.parent.stat().st_mode & 0o777
    assert parent_mode == 0o700


def test_default_audit_path_uses_xdg_state_home(tmp_path):
    from nvsh.agent.audit import default_audit_path

    env = {"XDG_STATE_HOME": str(tmp_path / "state")}
    assert default_audit_path(env) == tmp_path / "state" / "nvsh" / "audit.jsonl"

    env_no_xdg = {"HOME": str(tmp_path / "home")}
    assert default_audit_path(env_no_xdg) == (
        tmp_path / "home" / ".local" / "state" / "nvsh" / "audit.jsonl"
    )


def test_proposal_never_written_into_readline_line(tmp_path, monkeypatch):
    """The loop is terminal-free: it never sets/reads $READLINE_LINE, and a
    proposal's command reaches the world only through the executor callback.
    """
    monkeypatch.delenv("READLINE_LINE", raising=False)
    agent = FakeAgent(_script())
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    executed_via_callback: list[str] = []

    def executor(command: str) -> ExecResult:
        # The only channel a proposal's command may travel through.
        executed_via_callback.append(command)
        assert "READLINE_LINE" not in os.environ
        return ExecResult(exit_code=0)

    run_loop(
        agent=agent,
        request=_request(),
        context=_context(),
        approve=lambda proposal: True,
        executor=executor,
        on_event=lambda event: None,
        audit=audit,
    )

    assert "sudo rm -rf /x" in executed_via_callback
    assert "READLINE_LINE" not in os.environ
