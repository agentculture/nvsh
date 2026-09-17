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
    THINKING = "thinking"
    DONE = "done"
    ERROR = "error"
    #: Daemon-only (t10): the agent is busy with a turn this request may
    #: steer, replace or leave; ``args`` carries ``owner``, ``elapsed``,
    #: ``steerable`` and ``choices``. Never yielded by an adapter.
    BUSY = "busy"


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
class Target:
    """Which backend/model/effort/alias a request should be routed to.

    All fields but ``backend`` default to ``None``, meaning "let the caller
    decide" (e.g. the adapter's configured default model). Kept as its own
    small immutable value so ``nvsh/daemon.py`` can adopt it without this
    module reaching into daemon internals.
    """

    backend: str
    model: str | None = None
    effort: str | None = None
    alias: str | None = None


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
    #: Which backend/model/effort/alias to route this request to. ``None``
    #: means "use whatever the caller is already configured with" -- the
    #: common case, and why the wire codec omits it entirely by default.
    target: Target | None = None


def target_to_dict(target: Target) -> dict:
    """Encode a :class:`Target` as a JSON-serializable dict.

    Default-valued (``None``) fields are omitted, matching
    :func:`event_to_dict`'s convention.
    """
    data: dict[str, object] = {"backend": target.backend}
    if target.model is not None:
        data["model"] = target.model
    if target.effort is not None:
        data["effort"] = target.effort
    if target.alias is not None:
        data["alias"] = target.alias
    return data


def target_from_dict(data: Mapping[str, object] | None) -> Target | None:
    """Decode what :func:`target_to_dict` produced.

    Returns ``None`` for ``None`` input or a dict without a usable
    ``backend`` -- a pre-change wire message never had a ``target`` key at
    all, so this makes "absent" and "malformed" behave the same way.
    """
    if not isinstance(data, Mapping):
        return None
    backend = data.get("backend")
    if not isinstance(backend, str) or not backend:
        return None
    return Target(
        backend=backend,
        model=data.get("model") if isinstance(data.get("model"), str) else None,
        effort=data.get("effort") if isinstance(data.get("effort"), str) else None,
        alias=data.get("alias") if isinstance(data.get("alias"), str) else None,
    )


def request_to_dict(request: AgentRequest) -> dict:
    """Encode an :class:`AgentRequest` as a JSON-serializable dict.

    Default-valued fields are omitted so a request predating ``target``
    round-trips identically, and old daemons/clients that don't know about
    ``target`` still see a familiar shape.
    """
    data: dict[str, object] = {"kind": request.kind.value}
    if request.prompt:
        data["prompt"] = request.prompt
    if request.command:
        data["command"] = request.command
    if request.exit_code is not None:
        data["exit_code"] = request.exit_code
    if request.failure_id:
        data["failure_id"] = request.failure_id
    if request.ask:
        data["ask"] = request.ask
    if request.target is not None:
        data["target"] = target_to_dict(request.target)
    return data


def request_from_dict(data: Mapping[str, object] | None) -> AgentRequest:
    """Decode what :func:`request_to_dict` produced, tolerating junk.

    A dict without a ``target`` key -- exactly what a pre-change caller
    sends -- decodes with ``target=None``.
    """
    data = data or {}
    try:
        kind = RequestKind(str(data.get("kind", RequestKind.EXPLICIT.value)))
    except ValueError:
        kind = RequestKind.EXPLICIT
    raw_exit = data.get("exit_code")
    exit_code = int(raw_exit) if isinstance(raw_exit, (int, float)) else None
    raw_target = data.get("target")
    target = target_from_dict(raw_target) if isinstance(raw_target, Mapping) else None
    return AgentRequest(
        kind=kind,
        prompt=str(data.get("prompt", "")),
        command=str(data.get("command", "")),
        exit_code=exit_code,
        failure_id=str(data.get("failure_id", "")),
        ask=str(data.get("ask", "")),
        target=target,
    )


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
    #: Whether the adapter can stream :attr:`EventKind.THINKING` events.
    thinking: bool = False
    #: Whether the adapter honors ``AgentRequest.target.effort``.
    effort: bool = False
    #: Protocol path the adapter speaks to its harness: ``"rpc"`` (pi),
    #: ``"stream-json"`` (claude, agy, qwen print mode), ``"app-server"``
    #: (codex), ``"acp"`` (qwen, kiro) or ``"http"`` (openai-compat); ``""``
    #: when the adapter has not declared one.
    path: str = ""
    #: Who mediates approval of a proposed command: ``"nvsh"`` (nvsh's own
    #: propose/approve loop decides), ``"harness"`` (the backend's own
    #: approval UI decides), or ``"none"`` (no approval gate at all).
    approval: str = "none"
    #: Whether the adapter can read/write files on its own, outside of
    #: nvsh's deterministic operator tools.
    unmediated_file_access: bool = False
    #: Whether the adapter can deliver text into the turn that is running
    #: right now (:meth:`NvshAgent.steer`) rather than only as the next
    #: request. Declared in principle -- the runtime answer is still
    #: ``steer()``'s return value, which can be ``False`` even when this is
    #: ``True`` (codex's exec fallback, no turn id yet, the turn already
    #: ended).
    steer: bool = False


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

    def steer(self, text: str) -> bool:
        """Deliver ``text`` into the turn that is running right now.

        Returns ``True`` when the backend really took it mid-turn, and
        ``False`` when it has no such channel -- the caller then sends the
        text as the next request in the conversation instead, with the
        prior turn as context (deviation d16). Deliberately *not* abstract:
        "no mid-turn channel" is a legitimate, complete answer, and an
        adapter that cannot steer should not have to say so in code.
        """
        return False

    @abc.abstractmethod
    def cancel(self) -> None:
        """Ask the in-flight ``run()`` to stop yielding further events."""

    def force_stop(self) -> None:
        """Stop the in-flight turn for certain, even if the harness ignores it.

        Deliberately *not* abstract: the default asks politely
        (:meth:`cancel`) and then releases everything (:meth:`close`, which
        for subprocess adapters ends in a process-group kill). Adapters with
        a protocol-level interrupt override it. A stop only ever sends
        signals and protocol messages; it never edits harness files.
        """
        try:
            self.cancel()
        finally:
            self.close()

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources (subprocess, sockets, ...). Idempotent."""

    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """Report what this adapter supports."""
