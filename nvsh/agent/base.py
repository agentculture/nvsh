"""The NvshAgent contract: the abstract surface every backend adapter implements.

Nothing in this module imports pi (or any other concrete backend). Adapters
(fake, pi, stdlib-http, ...) live in their own modules and import *from*
here, never the other way around.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Mapping


class RequestKind(str, Enum):
    """Why the agent is being asked to run."""

    FAILURE = "failure"
    SLASH = "slash"
    EXPLICIT = "explicit"


class EventKind(str, Enum):
    """The kinds of events an adapter's ``run()`` may yield."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PROPOSAL = "proposal"
    STATUS = "status"
    DONE = "done"
    ERROR = "error"


class ProposalKind(str, Enum):
    """What a :class:`Proposal` is proposing to do."""

    INSPECT = "inspect"
    FIX = "fix"
    RETRY = "retry"
    VERIFY = "verify"


@dataclass(frozen=True)
class Proposal:
    """A single command an adapter proposes to run.

    ``kind`` classifies the proposal so a caller's ``approve`` policy can
    treat read-only inspection differently from a mutating fix -- but the
    loop itself stays policy-free: it calls ``approve`` for every kind and
    never special-cases ``inspect`` on its own (see ``loop.run_loop``).
    """

    command: str
    rationale: str
    kind: ProposalKind


@dataclass(frozen=True)
class AgentRequest:
    """What the caller is asking the agent to do."""

    kind: RequestKind
    prompt: str = ""
    command: str = ""
    exit_code: int | None = None
    failure_id: str = ""
    #: The operator's own sentence when the "failed command" was actually a
    #: plain-language question typed at the prompt (deviation d20). Empty
    #: for a real failure. Presentation-only: the prompt already carries the
    #: question, and this is what lets the panel say "asking the agent: ..."
    #: instead of "<line> failed (exit 127)".
    ask: str = ""


@dataclass(frozen=True)
class AgentContext:
    """Bounded, already-redacted context handed to the agent for one request."""

    platform: str = ""
    output: str = ""
    cwd: str = ""
    shell_pid: int = 0
    redaction_report: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentEvent:
    """One streamed event from an adapter's ``run()``.

    Only the fields relevant to ``kind`` are populated; the rest keep their
    defaults so a scripted event can be constructed with just the fields it
    needs.
    """

    kind: EventKind
    text: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    result: object = None
    proposal: Proposal | None = None
    error: str = ""


def event_to_dict(event: AgentEvent) -> dict:
    """Encode an :class:`AgentEvent` as a JSON-serializable dict.

    Default-valued fields are omitted so one line on the daemon socket
    (``nvsh.daemon``) stays small; :func:`event_from_dict` restores them.
    """
    data: dict[str, object] = {"kind": event.kind.value}
    if event.text:
        data["text"] = event.text
    if event.tool:
        data["tool"] = event.tool
    if event.args:
        data["args"] = dict(event.args)
    if event.result is not None:
        data["result"] = event.result
    if event.proposal is not None:
        data["proposal"] = {
            "command": event.proposal.command,
            "rationale": event.proposal.rationale,
            "kind": event.proposal.kind.value,
        }
    if event.error:
        data["error"] = event.error
    return data


def event_from_dict(data: Mapping[str, object]) -> AgentEvent:
    """Decode what :func:`event_to_dict` produced.

    Deliberately lenient about ``kind``: a kind this version does not know
    (a newer daemon talking to an older client) degrades to
    :attr:`EventKind.STATUS` rather than raising, so an unknown event is
    surfaced as noise instead of breaking the stream.
    """
    raw_kind = str(data.get("kind", EventKind.STATUS.value))
    try:
        kind = EventKind(raw_kind)
        text = str(data.get("text", ""))
    except ValueError:
        kind = EventKind.STATUS
        text = str(data.get("text", "") or raw_kind)

    raw_proposal = data.get("proposal")
    proposal = None
    if isinstance(raw_proposal, Mapping):
        try:
            proposal_kind = ProposalKind(str(raw_proposal.get("kind", ProposalKind.FIX.value)))
        except ValueError:
            proposal_kind = ProposalKind.FIX
        proposal = Proposal(
            command=str(raw_proposal.get("command", "")),
            rationale=str(raw_proposal.get("rationale", "")),
            kind=proposal_kind,
        )

    raw_args = data.get("args")
    return AgentEvent(
        kind=kind,
        text=text,
        tool=str(data.get("tool", "")),
        args=dict(raw_args) if isinstance(raw_args, Mapping) else {},
        result=data.get("result"),
        proposal=proposal,
        error=str(data.get("error", "")),
    )


@dataclass(frozen=True)
class Capabilities:
    """What an adapter can do, self-reported for the conformance suite and UI."""

    streaming: bool = True
    tool_calling: bool = False
    cancellation: bool = True
    persistent_session: bool = False
    local_model: bool = False


class NvshAgent(abc.ABC):
    """Abstract backend adapter contract.

    ``run()`` and everything else here is UI-free: no terminal I/O, no
    readline, nothing printed. Callers (``loop.run_loop`` and, later, the
    interactive shell) own presentation.
    """

    @abc.abstractmethod
    def start(self) -> None:
        """Prepare the adapter (e.g. spawn a subprocess). Idempotent."""

    @abc.abstractmethod
    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        """Stream events for one request. Must respect a pending ``cancel()``."""

    @abc.abstractmethod
    def cancel(self) -> None:
        """Ask the in-flight ``run()`` to stop yielding further events."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources (subprocess, sockets, ...). Idempotent."""

    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """Report what this adapter supports."""
