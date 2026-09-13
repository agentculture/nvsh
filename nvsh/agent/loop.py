"""run_loop: drives one NvshAgent request to completion. UI-free.

The loop never touches readline or the terminal, and it never decides policy
on its own: every proposal -- ``inspect``, ``fix``, ``retry`` and ``verify``
alike -- goes through the caller-supplied ``approve`` callback. A caller that
wants to auto-approve read-only ``inspect`` proposals does that inside its
own ``approve`` function; the loop has no special case for any ``kind``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .audit import AuditLog
from .base import AgentContext, AgentEvent, AgentRequest, EventKind, NvshAgent, Proposal


@dataclass(frozen=True)
class ExecResult:
    """The result of running one approved proposal's command through ``executor``."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""


def run_loop(
    agent: NvshAgent,
    request: AgentRequest,
    context: AgentContext,
    approve: Callable[[Proposal], bool],
    executor: Callable[[str], ExecResult],
    on_event: Callable[[AgentEvent], None],
    audit: AuditLog,
) -> list[ExecResult]:
    """Run ``request`` against ``agent`` and return the ``ExecResult`` for each
    proposal that was approved and executed, in order.

    For every event: it is first handed to ``on_event`` (the caller's
    rendering hook). A ``PROPOSAL`` event is then always logged, always
    passed to ``approve``, and its decision is always logged; only when
    ``approve`` returns ``True`` is ``proposal.command`` passed to
    ``executor`` -- never anything else, never through an environment
    variable -- and the outcome is logged.
    """
    results: list[ExecResult] = []
    agent.start()
    try:
        for event in agent.run(request, context):
            on_event(event)
            if event.kind != EventKind.PROPOSAL:
                continue
            proposal = event.proposal
            audit.record(event="proposal", proposal=proposal)
            decision = approve(proposal)
            audit.record(event="decision", proposal=proposal, decision=decision)
            if not decision:
                continue
            outcome = executor(proposal.command)
            results.append(outcome)
            audit.record(event="outcome", proposal=proposal, decision=decision, outcome=outcome)
    finally:
        agent.close()
    return results
