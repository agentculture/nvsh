"""ClaudeAgent: the official ``claude`` CLI driven over its stream-json protocol.

nvsh speaks to ``claude`` exactly the way the Claude Agent SDK's own process
transport does -- the argv below was read out of the SDK bundled inside the
``claude`` 2.1.270 binary, not guessed:

``claude -p --output-format stream-json --input-format stream-json --verbose
--include-partial-messages --permission-prompts host --permission-prompt-tool
stdio [--model M] [--effort E] [--session-id UUID | --resume ID]``

**How a permission prompt reaches nvsh (the t10 finding).** In print mode the
CLI only routes a permission decision to its host when it is started with
``--permission-prompt-tool stdio``. ``--permission-prompts host`` is the
*default* and by itself decides nothing: with no stdio permission tool the
CLI has nobody to ask, so anything that would prompt is refused with
``system/permission_denied`` carrying ``"This command requires approval"``.
With the flag, the CLI writes

``{"type": "control_request", "request_id": ..., "request":
{"subtype": "can_use_tool", "tool_name": ..., "input": {...},
"tool_use_id": ..., "decision_reason": ...}}``

on stdout and blocks until the host writes a ``control_response`` on stdin.
nvsh turns that request into an :attr:`EventKind.PROPOSAL` and answers it from
:meth:`respond_ui` -- the same method name ``pi`` exposes, so the daemon,
``client_transport`` and the panel drive both backends through one code path.
``--dangerously-skip-permissions`` is never passed: nvsh proposes, the
operator approves.

Line mapping (``_events_for``):

* ``system``                                     -> ``STATUS``
* ``stream_event`` ``text_delta``                -> ``TEXT_DELTA``
* ``stream_event`` ``thinking_delta``            -> ``THINKING``
* ``assistant`` content ``thinking``             -> ``THINKING``
* ``assistant`` content ``tool_use``             -> ``TOOL_CALL``
* ``user`` content ``tool_result``               -> ``TOOL_RESULT``
* ``control_request`` ``can_use_tool``           -> ``PROPOSAL``
* ``result`` ``success``                         -> ``DONE``; any other
  subtype -> ``ERROR``

A message whose partial deltas were already streamed is not replayed from its
aggregate ``assistant`` envelope (the envelope's ``message.id`` is remembered
from ``message_start``), so ``--include-partial-messages`` never doubles the
text or the thinking. Tool calls are always taken from the aggregate envelope:
that is the first place the tool's input is complete.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed argv list below, no shell=True
import threading
import uuid
from collections import deque
from typing import Iterator

from ._env import child_env
from ._subprocess import (
    _STDERR_JOIN_TIMEOUT,
    _STDERR_TAIL_LINES,
    SubprocessAgent,
    _drain,
    build_prompt,
    build_system_prompt,
    reject_bypass_args,
)
from .base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    Proposal,
    ProposalKind,
)

#: What ``capabilities().path`` reports: the protocol nvsh drives the CLI
#: over, not a filesystem path -- ``claude`` is resolved from ``PATH`` and the
#: interesting fact about this adapter is that it speaks stream-json both ways.
TRANSPORT_PATH = "stream-json"

#: Tool inputs whose value is the command line an operator would approve.
#: Anything else yields a ``Proposal`` with an empty ``command``: the loop
#: never runs a proposal that carries no command (see ``loop.run_loop``), so
#: an unrecognized tool can be shown and answered but never executed.
_COMMAND_FIELDS = ("command",)

#: How long a proposal's rationale may be before it is trimmed for a panel.
_RATIONALE_LIMIT = 400

#: Field names a caller may use to say "the operator allowed this".
_ALLOW_KEYS = ("confirmed", "allow", "allowed", "approved")

#: Values of a ``value=`` field that mean allow.
_ALLOW_VALUES = {"allow", "yes", "y", "true", "approve", "approved"}


def _trim(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _RATIONALE_LIMIT:
        return collapsed
    return collapsed[: _RATIONALE_LIMIT - 1] + "…"


def _opt_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _command_of(tool_input: dict) -> str:
    for field in _COMMAND_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _is_allow(fields: dict) -> bool:
    """Did the caller say yes? Anything unrecognized is a no."""
    if fields.get("cancelled"):
        return False
    for key in _ALLOW_KEYS:
        if key in fields:
            return bool(fields[key])
    value = fields.get("value")
    if isinstance(value, str):
        return value.strip().lower() in _ALLOW_VALUES
    return False


class ClaudeAgent(SubprocessAgent):
    """Adapter for the official ``claude`` CLI in print + stream-json mode."""

    binary = "claude"

    def __init__(
        self,
        config: dict | None = None,
        *,
        env: dict[str, str] | None = None,
        binary: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        extra_args: list[str] | None = None,
        approval: str = "nvsh",
        session_id: str | None = None,
    ) -> None:
        super().__init__(config, env=env)
        settings = self._config
        self.binary = binary or str(settings.get("binary") or type(self).binary)
        self._model = model if model is not None else _opt_str(settings.get("model"))
        self._effort = effort if effort is not None else _opt_str(settings.get("effort"))
        raw_extra = extra_args if extra_args is not None else settings.get("extra_args")
        self._extra_args = [str(item) for item in raw_extra] if raw_extra else []
        reject_bypass_args(self._extra_args, "claude")
        self._approval = str(settings.get("approval") or approval or "nvsh")
        self._session_id = session_id or _opt_str(settings.get("session_id")) or str(uuid.uuid4())
        #: Set once a turn has run against ``_session_id``: a warm adapter
        #: resumes that conversation instead of claiming the id again.
        self._warm = False
        #: ``request_id`` -> the tool input the CLI asked about, so
        #: :meth:`respond_ui` can echo it back as ``updatedInput`` on allow.
        self._pending: dict[str, dict] = {}
        #: ``tool_use_id`` -> tool name, so a ``tool_result`` can name its tool.
        self._tools: dict[str, str] = {}
        #: ``message.id`` of every message whose deltas were streamed already.
        self._streamed: set[str] = set()

    # -- argv -------------------------------------------------------------

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        argv = [
            self.binary,
            "-p",
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--append-system-prompt",
            build_system_prompt(context),
        ]
        if self._approval == "none":
            argv += ["--permission-prompts", "none"]
        else:
            # Both flags, in this order: "host" says a host answers, and the
            # stdio permission tool is what actually makes can_use_tool
            # control_requests arrive on stdout (see the module docstring).
            argv += ["--permission-prompts", "host", "--permission-prompt-tool", "stdio"]
        model = self._target_model(request)
        if model:
            argv += ["--model", model]
        effort = self._target_effort(request)
        if effort:
            argv += ["--effort", effort]
        if self._warm:
            argv += ["--resume", self._session_id]
        else:
            argv += ["--session-id", self._session_id]
        argv += self._extra_args
        return argv

    def _target_model(self, request: AgentRequest) -> str:
        target = request.target
        if target is not None and target.model:
            return target.model
        return self._model or ""

    def _target_effort(self, request: AgentRequest) -> str:
        target = request.target
        if target is not None and target.effort:
            return target.effort
        return self._effort or ""

    # -- run --------------------------------------------------------------

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        """Stream one turn. Unlike the shared base, stdin stays a live pipe:
        the prompt goes in as a stream-json user message and a permission
        prompt is answered on the same channel."""
        argv = self._argv(request, context)
        self._pending.clear()
        self._tools.clear()
        self._streamed.clear()
        try:
            self._proc = subprocess.Popen(  # nosec B603 - argv is a fixed list, no shell
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_env(self._env),
                # Its own process group, so a stop can kill the whole tree
                # (kill_tree, task t2) instead of leaving a tool grandchild
                # behind when this adapter overrides SubprocessAgent.run to
                # keep stdin a live pipe.
                start_new_session=True,
            )
        except OSError as exc:
            yield AgentEvent(kind=EventKind.ERROR, error=f"failed to start {argv[0]}: {exc}")
            return

        tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        drain: threading.Thread | None = None
        if self._proc.stderr is not None:
            drain = threading.Thread(target=_drain, args=(self._proc.stderr, tail), daemon=True)
            drain.start()

        self._send(
            {
                "type": "user",
                "message": {"role": "user", "content": build_prompt(request, context)},
            }
        )

        try:
            assert self._proc.stdout is not None
            for raw_line in self._proc.stdout:
                if self._cancelled:
                    return
                for event in self._events(raw_line.rstrip("\n")):
                    if self._cancelled:
                        return
                    yield event
                    if event.kind in (EventKind.ERROR, EventKind.DONE):
                        return
            if self._cancelled:
                return
            yield self._exit_event(argv, tail, drain)
        finally:
            self._warm = True
            self._terminate_if_running()
            if drain is not None:
                drain.join(timeout=_STDERR_JOIN_TIMEOUT)

    # -- line mapping -----------------------------------------------------

    def _events(self, line: str) -> list[AgentEvent]:
        if not line.strip():
            return []
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict):
            return []
        return self._events_for(obj)

    def _events_for(self, obj: dict) -> list[AgentEvent]:
        kind = obj.get("type")
        if kind == "system":
            return [AgentEvent(kind=EventKind.STATUS, text=self._system_text(obj))]
        if kind == "stream_event":
            return self._stream_events(obj.get("event") or {})
        if kind == "assistant":
            return self._assistant_events(obj.get("message") or {})
        if kind == "user":
            return self._user_events(obj.get("message") or {})
        if kind == "control_request":
            return self._control_request_events(obj)
        if kind == "result":
            if obj.get("subtype") == "success":
                return [AgentEvent(kind=EventKind.DONE)]
            return [AgentEvent(kind=EventKind.ERROR, error=self._result_error(obj))]
        return []

    @staticmethod
    def _system_text(obj: dict) -> str:
        for field in ("text", "message", "status", "subtype"):
            value = obj.get(field)
            if isinstance(value, str) and value:
                return value
        return ""

    @staticmethod
    def _result_error(obj: dict) -> str:
        for field in ("error", "result", "subtype"):
            value = obj.get(field)
            if isinstance(value, str) and value:
                return value
        return "claude reported an error"

    def _stream_events(self, event: dict) -> list[AgentEvent]:
        etype = event.get("type")
        if etype == "message_start":
            message_id = (event.get("message") or {}).get("id")
            if isinstance(message_id, str) and message_id:
                self._streamed.add(message_id)
            return []
        if etype != "content_block_delta":
            return []
        delta = event.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            text = str(delta.get("text", ""))
            return [AgentEvent(kind=EventKind.TEXT_DELTA, text=text)] if text else []
        if dtype == "thinking_delta":
            text = str(delta.get("thinking", ""))
            return [AgentEvent(kind=EventKind.THINKING, text=text)] if text else []
        return []

    def _assistant_events(self, message: dict) -> list[AgentEvent]:
        message_id = message.get("id")
        streamed = isinstance(message_id, str) and message_id in self._streamed
        events: list[AgentEvent] = []
        for part in message.get("content") or []:
            if not isinstance(part, dict):
                continue
            event = self._assistant_part_event(part, streamed)
            if event is not None:
                events.append(event)
        return events

    def _assistant_part_event(self, part: dict, streamed: bool) -> AgentEvent | None:
        """Map one aggregate ``assistant`` content part, or ``None`` for nothing."""
        ptype = part.get("type")
        if ptype == "tool_use":
            return self._tool_call_event(part)
        if streamed:
            # Already delivered delta by delta; do not say it twice.
            return None
        if ptype == "thinking":
            text = str(part.get("thinking", ""))
            return AgentEvent(kind=EventKind.THINKING, text=text) if text else None
        if ptype == "text":
            text = str(part.get("text", ""))
            return AgentEvent(kind=EventKind.TEXT_DELTA, text=text) if text else None
        return None

    def _tool_call_event(self, part: dict) -> AgentEvent:
        tool = str(part.get("name", ""))
        tool_use_id = str(part.get("id", ""))
        if tool_use_id:
            self._tools[tool_use_id] = tool
        args = part.get("input")
        return AgentEvent(
            kind=EventKind.TOOL_CALL,
            tool=tool,
            args=dict(args) if isinstance(args, dict) else {},
        )

    def _user_events(self, message: dict) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        for part in message.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "tool_result":
                continue
            tool_use_id = str(part.get("tool_use_id", ""))
            events.append(
                AgentEvent(
                    kind=EventKind.TOOL_RESULT,
                    tool=self._tools.get(tool_use_id, ""),
                    result=part.get("content"),
                    args={"tool_use_id": tool_use_id} if tool_use_id else {},
                )
            )
        return events

    def _control_request_events(self, obj: dict) -> list[AgentEvent]:
        request = obj.get("request") or {}
        request_id = str(obj.get("request_id", ""))
        if request.get("subtype") != "can_use_tool":
            # Not something nvsh can answer. Say so rather than leaving the
            # CLI blocked on a reply that is never coming.
            if request_id:
                self._send(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "error",
                            "request_id": request_id,
                            "error": f"nvsh does not handle {request.get('subtype')!r}",
                        },
                    }
                )
            return []
        raw_input = request.get("input")
        tool_input = dict(raw_input) if isinstance(raw_input, dict) else {}
        self._pending[request_id] = tool_input
        rationale = str(request.get("decision_reason") or request.get("description") or "")
        proposal = Proposal(
            command=_command_of(tool_input),
            rationale=_trim(rationale),
            kind=ProposalKind.FIX,
        )
        return [
            AgentEvent(
                kind=EventKind.PROPOSAL,
                proposal=proposal,
                tool=str(request.get("tool_name", "")),
                args={
                    "request_id": request_id,
                    "tool_use_id": str(request.get("tool_use_id", "")),
                },
            )
        ]

    # -- answering a permission prompt ------------------------------------

    def respond_ui(self, request_id: str, **fields: object) -> None:
        """Answer a pending ``can_use_tool`` prompt. Mirrors ``PiAgent``.

        The caller (panel, ``client_transport``, the daemon's abort path)
        passes the same fields it passes pi: ``confirmed=True`` to allow,
        ``cancelled=True`` to deny, or ``value="allow"``/``"deny"``. Anything
        unrecognized denies: a field nvsh cannot read as an approval is not
        an approval.
        """
        if not request_id:
            return
        tool_input = self._pending.pop(request_id, {})
        if _is_allow(fields):
            response: dict = {"behavior": "allow", "updatedInput": tool_input}
        else:
            reason = fields.get("reason") or fields.get("message")
            response = {
                "behavior": "deny",
                "message": str(reason) if reason else "denied by the nvsh operator",
            }
        self._send(
            {
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": response,
                },
            }
        )

    def _send(self, message: dict) -> bool:
        """Write one JSON line to the CLI's stdin. Never raises."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return False
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            return False
        return True

    # -- capabilities -----------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=True,
            cancellation=True,
            persistent_session=True,
            local_model=False,
            thinking=True,
            effort=True,
            path=TRANSPORT_PATH,
            approval="nvsh" if self._approval != "none" else "none",
            unmediated_file_access=True,
        )
