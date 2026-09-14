"""CodexAgent: drives ``codex app-server`` over stdio, with ``codex exec`` as fallback.

Two paths, one adapter:

* **app-server (preferred).** ``codex app-server`` speaks JSON-RPC over
  stdio: one ``initialize`` handshake, then ``thread/start`` (approval
  policy ``on-request``, sandbox ``read-only``), then one ``turn/start``
  per request. Notifications stream back as ``item/*`` events, and a
  command the model wants to run that the sandbox will not allow comes
  back as a *server request* --
  ``item/commandExecution/requestApproval`` -- which this adapter surfaces
  as an :class:`~nvsh.agent.base.EventKind.PROPOSAL` and answers only once
  the caller says so (:meth:`CodexAgent.respond_approval`). That is
  "propose, don't run" on codex's own wire: the sandbox is read-only and
  the policy is ``on-request``, so nothing mutating happens without an
  explicit answer. ``never``, ``danger-full-access`` and ``--full-auto``
  never appear anywhere in this module.
* **``codex exec --json`` (fallback).** Used when ``initialize`` fails --
  an older ``codex`` with no ``app-server`` subcommand, a binary that dies
  on launch, a handshake that never answers. The thin line-mapping
  behaviour this adapter had before app-server existed is kept verbatim
  (see :meth:`_parse_line`), so a machine with an older CLI keeps working.

Field names come from ``codex app-server generate-json-schema`` (codex-cli
0.147.0): ``ClientRequest``/``ServerRequest``/``ServerNotification`` and the
``CommandExecutionRequestApprovalParams`` / ``ExecCommandApprovalParams``
pair. ``model`` and ``effort`` travel as ``-c model=`` and
``-c model_reasoning_effort=`` global config overrides -- the same flags on
both paths -- never as a per-turn override, so what the operator configured
is what both paths send.

The long-lived client here is modelled on :mod:`nvsh.agent.pi`: plain
:mod:`subprocess` pipes, a reader thread draining stdout onto a
:class:`queue.Queue`, request/response correlation by id with everything
else held in order, and a terminate/kill escalation on close.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import shutil
import subprocess  # external `codex` CLI (bandit B404, allowed repo-wide)
import threading
import time
from typing import Any, Iterator, Mapping

from ._env import child_env
from ._subprocess import (
    SubprocessAgent,
    build_full_prompt,
    escalate_close,
    redacted_tail,
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

#: Name this client announces in ``initialize``'s ``clientInfo``.
CLIENT_NAME = "nvsh"

#: The approval policy every thread is started with. ``"never"`` is
#: deliberately not reachable from this module: nvsh proposes and the
#: operator approves, so codex must ask rather than decide.
APPROVAL_POLICY = "on-request"

#: The sandbox every thread is started with. ``"danger-full-access"`` is
#: likewise never used.
SANDBOX_MODE = "read-only"

#: Tokens that must never reach codex's argv or params. Named so the rule is
#: greppable from one place and testable as data.
BANNED_TOKENS = ("never", "danger-full-access", "--full-auto")

#: How long one JSON-RPC request may go unanswered before the client gives
#: up and says so. A liveness bound, not a thinking budget: ``turn/start``
#: answers as soon as the turn is *created*, and the turn's own thinking
#: time is bounded by the daemon's turn cap.
_ACK_TIMEOUT_SECONDS = 20.0

#: Environment override for that bound, in seconds.
ACK_TIMEOUT_ENV = "NVSH_CODEX_ACK_TIMEOUT"

#: How long a queue.get() waits before re-checking liveness/cancellation.
_POLL_INTERVAL_SECONDS = 0.2

#: How long close() waits at each rung of the terminate/kill escalation.
_CLOSE_WAIT_SECONDS = 2.0

#: How many stderr lines are kept to explain a launch failure.
_STDERR_TAIL_LINES = 40

#: How much assistant text preceding an approval request is kept as that
#: proposal's rationale when codex's own ``reason`` is empty.
_RATIONALE_LIMIT = 600

#: Server *requests* (they carry an ``id`` and expect a result) that mean
#: "may I run this command?". ``item/commandExecution/requestApproval`` is
#: the current method; ``execCommandApproval`` is the legacy one, still
#: emitted by older servers, and takes the older ``ReviewDecision`` reply
#: vocabulary (``approved``/``denied``) instead of ``accept``/``decline``.
APPROVAL_METHODS = ("item/commandExecution/requestApproval", "execCommandApproval")

#: Decisions per approval method: ``(approve, decline)``. Both vocabularies
#: come straight from the generated schema.
_DECISIONS = {
    "item/commandExecution/requestApproval": ("accept", "decline"),
    "execCommandApproval": ("approved", "denied"),
}

#: Notifications that are lifecycle/progress bookkeeping: true, uninteresting
#: to an operator at a failing prompt, and noisy enough to bury the panel.
#: Same judgement (and the same reason) as ``pi``'s ``_QUIET_EVENT_TYPES``.
_QUIET_NOTIFICATIONS = frozenset(
    {
        "account/rateLimits/updated",
        "item/commandExecution/outputDelta",
        "item/commandExecution/terminalInteraction",
        "item/fileChange/outputDelta",
        "item/fileChange/patchUpdated",
        "item/reasoning/summaryPartAdded",
        "mcpServer/startupStatus/updated",
        "remoteControl/status/changed",
        "serverRequest/resolved",
        "thread/started",
        "thread/status/changed",
        "thread/tokenUsage/updated",
        "turn/diff/updated",
        "turn/plan/updated",
        "turn/started",
    }
)

#: Notifications carrying streamed model reasoning. ``textDelta`` is raw
#: reasoning (only when the model and config emit it); ``summaryTextDelta``
#: is the summarized form, which is what a live gpt-5.6 turn actually sent.
_THINKING_NOTIFICATIONS = ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta")


class CodexRpcError(RuntimeError):
    """codex did not hold up its end of the app-server protocol.

    Raised rather than returned so no failure is silent: every caller turns
    it into an ``ERROR`` event, or -- for the ``initialize`` handshake only
    -- into the decision to use the ``codex exec --json`` fallback.
    """


def ack_timeout(env: Mapping[str, str] | None = None) -> float:
    """How long to wait for one JSON-RPC response, in seconds."""
    resolved = os.environ if env is None else env
    try:
        value = float(resolved.get(ACK_TIMEOUT_ENV, ""))
    except (TypeError, ValueError):
        return _ACK_TIMEOUT_SECONDS
    return value if value > 0 else _ACK_TIMEOUT_SECONDS


def approval_fields(params: Mapping[str, Any]) -> tuple[str, str]:
    """``(command, reason)`` for one approval server request.

    The current method carries ``command`` as a single string; the legacy
    ``execCommandApproval`` carries it as an argv list. Neither prompt text
    nor any other human-facing field is ever treated as a command -- an
    approval request without one yields ``""``, and approving it runs
    nothing (the same rule ``pi``'s ``_proposal_fields`` enforces).
    """
    raw = params.get("command")
    if isinstance(raw, list):
        command = " ".join(str(part) for part in raw)
    elif isinstance(raw, str):
        command = raw
    else:
        command = ""
    reason = params.get("reason")
    return command, str(reason) if isinstance(reason, str) else ""


def _client_version() -> str:
    """nvsh's own version for ``clientInfo``, resolved lazily."""
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - a handshake must not fail over a version string
        return "0"


class CodexAgent(SubprocessAgent):
    """Drives ``codex app-server``; falls back to ``codex exec --json``."""

    binary = "codex"

    def __init__(
        self,
        config: dict | None = None,
        *,
        binary: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        extra_args: list[str] | None = None,
        approval: str = "nvsh",
        env: dict[str, str] | None = None,
        app_server: bool = True,
    ) -> None:
        super().__init__(config, env=env)
        settings = self._config
        self.binary = binary or str(settings.get("binary") or "codex")
        resolved_model = model if model is not None else settings.get("model")
        self._model = str(resolved_model) if resolved_model else None
        resolved_effort = effort if effort is not None else settings.get("effort")
        self._effort = str(resolved_effort) if resolved_effort else None
        raw_extra = extra_args if extra_args is not None else settings.get("extra_args") or []
        self._extra_args = [str(item) for item in raw_extra]
        reject_bypass_args(self._extra_args, "codex")
        for item in self._extra_args:
            if item in BANNED_TOKENS or any(item.endswith(f"={token}") for token in BANNED_TOKENS):
                raise ValueError(f"codex: extra_args may not set an approval bypass ({item!r})")
        self._approval = str(settings.get("approval") or approval)
        self._use_app_server = app_server

        #: ``None`` until the handshake has been tried once; then True/False
        #: for the life of this adapter, so a machine without ``app-server``
        #: pays the probe once rather than on every failing command.
        self._app_server_ok: bool | None = None
        self._rpc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        self._queue: queue.Queue = queue.Queue()
        self._held: collections.deque = collections.deque()
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._streaming = False
        #: request id -> method, for approval server requests still unanswered.
        self._pending_approvals: dict[Any, str] = {}
        self._said: list[str] = []
        self._ack_timeout = ack_timeout(self._env)

    # -- argv --------------------------------------------------------------

    def _config_overrides(self) -> list[str]:
        """``-c`` overrides carrying model and reasoning effort, verbatim.

        Both values are opaque strings nvsh never validates (decision c24):
        codex decides what it accepts, and a rejected value is reported by
        codex rather than guessed at here.
        """
        argv: list[str] = []
        if self._model:
            argv += ["-c", f"model={self._model}"]
        if self._effort:
            argv += ["-c", f"model_reasoning_effort={self._effort}"]
        return argv

    def _base_argv(self) -> list[str]:
        return [self.binary, *self._config_overrides(), *self._extra_args]

    def app_server_argv(self) -> list[str]:
        """argv for the long-lived server. Never ``app-server daemon``.

        ``codex app-server`` reads JSON-RPC lines on the stdin *this*
        process hands it; the ``daemon`` subcommand instead manages a
        machine-wide background server, which nvsh has no business starting.
        """
        return [*self._base_argv(), "app-server"]

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        # Fallback path. ``codex exec`` has no --system-prompt flag (checked
        # against the installed CLI's --help), so the system brief leads the
        # prompt text itself -- the same bytes the app-server path sends.
        prompt = build_full_prompt(request, context)
        return [*self._base_argv(), "exec", "--json", prompt]

    def thread_start_params(self, context: AgentContext | None = None) -> dict:
        """Params for ``thread/start``: ask on request, read-only sandbox."""
        params: dict[str, Any] = {
            "approvalPolicy": APPROVAL_POLICY,
            "sandbox": SANDBOX_MODE,
        }
        if context is not None and context.cwd:
            params["cwd"] = context.cwd
        return params

    # -- app-server lifecycle ---------------------------------------------

    @property
    def thread_id(self) -> str | None:
        """The app-server thread this adapter is talking to, if any."""
        return self._thread_id

    def _spawn_app_server(self) -> None:
        # argv is a fixed list built by app_server_argv(), never shell=True
        # (bandit B603, allowed repo-wide).
        self._rpc = subprocess.Popen(
            self.app_server_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env(self._env),
        )
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()

    def ensure_app_server(self) -> bool:
        """Spawn and handshake with ``codex app-server``. Idempotent.

        Returns ``True`` when the server is up and ``initialize`` answered,
        ``False`` when this binary cannot give us one -- which is the whole
        fallback condition: ``run()`` then uses ``codex exec --json``.
        """
        if not self._use_app_server:
            return False
        if self._app_server_ok is False:
            return False
        proc = self._rpc
        if self._app_server_ok and proc is not None and proc.poll() is None:
            return True
        # Either the first attempt, or a close() followed by a fresh start():
        # the verdict was "yes" but the process is gone, so hand back a live
        # one rather than writing into a closed pipe.
        self._app_server_ok = False
        try:
            self._spawn_app_server()
        except OSError:
            self._teardown_app_server()
            return False
        try:
            self._request(
                "initialize",
                {"clientInfo": {"name": CLIENT_NAME, "version": _client_version()}},
            )
        except CodexRpcError:
            self._teardown_app_server()
            return False
        self._app_server_ok = True
        return True

    def _reader_loop(self) -> None:
        """Drain stdout line by line onto the queue, then post an EOF sentinel."""
        proc = self._rpc
        assert proc is not None and proc.stdout is not None
        try:
            for raw_line in proc.stdout:
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    # Not JSON-RPC at all. Before initialize has answered,
                    # this is what "this binary has no app-server" looks
                    # like; it is never a protocol message, so it is dropped
                    # rather than surfaced as an event.
                    continue
                self._queue.put(obj)
        except (OSError, ValueError):  # closed underneath us by close()/cancel()
            pass
        finally:
            self._queue.put(None)

    def _stderr_loop(self) -> None:
        proc = self._rpc
        if proc is None or proc.stderr is None:  # pragma: no cover - always piped
            return
        try:
            for line in proc.stderr:
                self._stderr_tail.append(line)
        except (OSError, ValueError):  # pragma: no cover - closed by teardown
            pass

    def _exit_detail(self) -> str:
        """How the app-server process is doing, for an error message tail."""
        proc = self._rpc
        if proc is None:
            return " (no codex app-server process)"
        code = proc.poll()
        state = "still running" if code is None else f"exited with code {code}"
        stderr = redacted_tail(self._stderr_tail)
        tail = " | ".join(line.strip() for line in stderr.splitlines() if line.strip())
        if tail:
            return f" (codex app-server {state}; stderr tail: {tail})"
        return f" (codex app-server {state}; no stderr)"

    # -- request / response ------------------------------------------------

    def _send(self, obj: dict) -> bool:
        """Write one JSON-RPC line. ``False`` when the server's stdin is gone."""
        proc = self._rpc
        if proc is None or proc.stdin is None:
            return False
        line = json.dumps(obj) + "\n"
        with self._write_lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (ValueError, OSError):  # OSError covers BrokenPipeError
                return False
        return True

    def _next_object(self, timeout: float) -> Any:
        """The next thing codex said: held objects first, then the live queue."""
        if self._held:
            return self._held.popleft()
        return self._queue.get(timeout=timeout)

    def _request(self, method: str, params: dict, timeout: float | None = None) -> dict:
        """Send one request and wait for *its* response, holding everything else.

        Objects that arrive while waiting (notifications, server requests)
        are kept in arrival order and replayed to the event loop afterwards,
        so a detour through a request never reorders the stream.
        """
        self._next_id += 1
        request_id = self._next_id
        if not self._send({"id": request_id, "method": method, "params": params}):
            raise CodexRpcError(f"could not send {method} to codex{self._exit_detail()}")
        wait = self._ack_timeout if timeout is None else timeout
        return self._await_response(method, request_id, wait)

    def _await_response(self, method: str, request_id: int, timeout: float) -> dict:
        carried: list = []
        deadline = time.monotonic() + timeout
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise CodexRpcError(
                        f"codex did not answer {method} within {timeout:g}s{self._exit_detail()}"
                    )
                try:
                    obj = self._next_object(min(_POLL_INTERVAL_SECONDS, left))
                except queue.Empty:
                    proc = self._rpc
                    if proc is not None and proc.poll() is not None:
                        raise CodexRpcError(
                            f"codex went away before answering {method}{self._exit_detail()}"
                        ) from None
                    continue
                if obj is None:  # EOF sentinel
                    carried.append(None)
                    raise CodexRpcError(
                        f"codex closed its output before answering {method}{self._exit_detail()}"
                    )
                answer = self._response_or_none(obj, method, request_id)
                if answer is not None:
                    return answer
                carried.append(obj)
        finally:
            self._held.extendleft(reversed(carried))

    @staticmethod
    def _response_or_none(obj: Any, method: str, request_id: int) -> dict | None:
        """*obj* as this request's result, or ``None`` if it is something else.

        Raises :class:`CodexRpcError` when it *is* the answer and codex
        rejected the request -- a rejection is never worth waiting through.
        """
        if not isinstance(obj, dict) or obj.get("id") != request_id or "method" in obj:
            return None
        error = obj.get("error")
        if error is not None:
            message = error.get("message") if isinstance(error, Mapping) else error
            raise CodexRpcError(f"codex rejected {method}: {message}")
        result = obj.get("result")
        return result if isinstance(result, dict) else {}

    # -- running -----------------------------------------------------------

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        if not self.ensure_app_server():
            yield from super().run(request, context)
            return

        self._cancelled = False
        self._said.clear()
        prompt = build_full_prompt(request, context)
        try:
            if self._thread_id is None:
                started = self._request("thread/start", self.thread_start_params(context))
                self._thread_id = str((started.get("thread") or {}).get("id") or "")
            turn = self._request(
                "turn/start",
                {"threadId": self._thread_id, "input": [{"type": "text", "text": prompt}]},
            )
            self._turn_id = str((turn.get("turn") or {}).get("id") or "")
        except CodexRpcError as exc:
            yield AgentEvent(kind=EventKind.ERROR, error=str(exc))
            return

        self._streaming = True
        try:
            yield from self._events()
        finally:
            self._streaming = False
            self._said.clear()

    def resume(self, thread_id: str) -> str:
        """Rejoin a stored thread with ``thread/resume``; returns its id.

        Lets a later request continue the conversation an earlier failure
        started, which is what ``persistent_session`` in
        :meth:`capabilities` claims.
        """
        if not self.ensure_app_server():
            raise CodexRpcError("codex app-server is not available; cannot resume a thread")
        params = dict(self.thread_start_params())
        params["threadId"] = str(thread_id)
        result = self._request("thread/resume", params)
        resumed = (result.get("thread") or {}).get("id") or thread_id
        self._thread_id = str(resumed)
        return self._thread_id

    def _events(self) -> Iterator[AgentEvent]:
        """The turn's event loop, from the started turn to DONE/ERROR."""
        while not self._cancelled:
            try:
                obj = self._next_object(_POLL_INTERVAL_SECONDS)
            except queue.Empty:
                proc = self._rpc
                if proc is not None and proc.poll() is not None:
                    yield AgentEvent(
                        kind=EventKind.ERROR,
                        error=f"codex app-server stopped{self._exit_detail()}",
                    )
                    return
                continue
            if obj is None:
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    error=f"codex app-server closed its output{self._exit_detail()}",
                )
                return
            event = self._map(obj)
            if event is None:
                continue
            yield event
            if self._cancelled or event.kind in (EventKind.DONE, EventKind.ERROR):
                return

    def _map(self, obj: Any) -> AgentEvent | None:
        """One JSON-RPC object as an :class:`AgentEvent`, or ``None`` to skip."""
        if not isinstance(obj, dict):
            return None
        method = obj.get("method")
        if method is None:
            return None  # a response to something nobody is waiting on any more
        raw_params = obj.get("params")
        params = raw_params if isinstance(raw_params, Mapping) else {}
        if obj.get("id") is not None:
            return self._server_request(obj, str(method), params)
        return self._notification(str(method), params)

    def _server_request(
        self, obj: dict, method: str, params: Mapping[str, Any]
    ) -> AgentEvent | None:
        """A request *from* the server. Approval requests become proposals."""
        if method not in APPROVAL_METHODS:
            # Answer it, or codex may block the turn waiting for a reply
            # nobody will ever send; the operator still sees it as STATUS.
            self._send(
                {
                    "id": obj.get("id"),
                    "error": {"code": -32601, "message": f"nvsh does not handle {method}"},
                }
            )
            return AgentEvent(kind=EventKind.STATUS, text=method)
        request_id = obj.get("id")
        self._pending_approvals[request_id] = method
        command, reason = approval_fields(params)
        return AgentEvent(
            kind=EventKind.PROPOSAL,
            proposal=Proposal(
                command=command,
                rationale=reason or self._rationale_text(),
                kind=ProposalKind.FIX,
            ),
            args={"request_id": request_id, "method": method},
        )

    def _notification(self, method: str, params: Mapping[str, Any]) -> AgentEvent | None:
        if method in _THINKING_NOTIFICATIONS:
            return AgentEvent(kind=EventKind.THINKING, text=str(params.get("delta", "")))
        if method == "item/agentMessage/delta":
            delta = str(params.get("delta", ""))
            self._said.append(delta)
            return AgentEvent(kind=EventKind.TEXT_DELTA, text=delta)
        if method in ("item/started", "item/completed"):
            return self._item_event(method, params)
        if method == "turn/completed":
            return self._turn_completed(params)
        if method == "error":
            error = params.get("error")
            message = error.get("message") if isinstance(error, Mapping) else error
            return AgentEvent(kind=EventKind.ERROR, error=str(message or "codex reported an error"))
        if method in _QUIET_NOTIFICATIONS:
            return None
        # Unknown notification types surface as STATUS so nothing new is
        # silently dropped.
        return AgentEvent(kind=EventKind.STATUS, text=method)

    def _item_event(self, method: str, params: Mapping[str, Any]) -> AgentEvent | None:
        """``item/started`` / ``item/completed`` for a command execution.

        Every other item type is already covered by its own delta stream
        (``agentMessage``, ``reasoning``) or is the echo of what we just
        sent (``userMessage``), so only command executions -- the thing an
        operator is watching for -- become tool events.
        """
        item = params.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "commandExecution":
            return None
        tool = "commandExecution"
        if method == "item/started":
            return AgentEvent(
                kind=EventKind.TOOL_CALL,
                tool=tool,
                args={"command": item.get("command", ""), "cwd": item.get("cwd", "")},
            )
        # A finished tool resets the rationale window: what the model says
        # next is about the next thing it wants to do.
        self._said.clear()
        return AgentEvent(
            kind=EventKind.TOOL_RESULT,
            tool=tool,
            result={
                "status": item.get("status", ""),
                "exitCode": item.get("exitCode"),
                "output": item.get("aggregatedOutput") or "",
            },
        )

    def _turn_completed(self, params: Mapping[str, Any]) -> AgentEvent:
        raw_turn = params.get("turn")
        turn = raw_turn if isinstance(raw_turn, Mapping) else {}
        if turn.get("status") == "failed":
            error = turn.get("error")
            message = error.get("message") if isinstance(error, Mapping) else error
            return AgentEvent(kind=EventKind.ERROR, error=str(message or "codex turn failed"))
        return AgentEvent(kind=EventKind.DONE)

    def _rationale_text(self) -> str:
        """Assistant text since the last tool result, trimmed for a panel."""
        text = " ".join("".join(self._said).split())
        if len(text) <= _RATIONALE_LIMIT:
            return text
        return text[: _RATIONALE_LIMIT - 1] + "…"

    # -- approvals / steering ---------------------------------------------

    def respond_approval(self, request_id: Any, approved: bool) -> bool:
        """Answer one pending approval server request.

        Not part of the :class:`~nvsh.agent.base.NvshAgent` contract: the
        caller that showed the ``PROPOSAL`` calls this once the operator has
        decided. The reply vocabulary follows the method that asked --
        ``accept``/``decline`` for ``item/commandExecution/requestApproval``,
        ``approved``/``denied`` for the legacy ``execCommandApproval``.
        """
        method = self._pending_approvals.pop(request_id, None)
        if method is None:
            return False
        approve, decline = _DECISIONS[method]
        return self._send(
            {"id": request_id, "result": {"decision": approve if approved else decline}}
        )

    def respond_ui(self, request_id: Any, **fields: object) -> None:
        """The shared responder entry point (mirrors ``PiAgent``/``ClaudeAgent``).

        The panel, ``client_transport`` and the daemon all answer a PROPOSAL
        through ``respond_ui``: ``confirmed=True`` or ``value="allow"`` means
        approve, anything else (including ``cancelled=True``) declines.
        """
        if fields.get("cancelled"):
            allow = False
        else:
            value = str(fields.get("value") or "").lower()
            allow = bool(fields.get("confirmed")) or value in {"allow", "approve", "yes", "y"}
        self.respond_approval(request_id, allow)

    def _decline_pending(self) -> None:
        """Answer every unanswered approval with a decline.

        A turn that is being cancelled or torn down must not leave codex
        blocked on a question nobody will answer, and declining is the only
        safe default: nvsh never runs a command the operator did not
        approve.
        """
        for request_id in list(self._pending_approvals):
            self.respond_approval(request_id, False)

    def steer(self, text: str) -> bool:
        """Deliver ``text`` into the turn running right now (``turn/steer``).

        Deliberately not response-waited: it is only ever sent mid-turn,
        when the event loop owns the queue, and two readers would race for
        the reply. A rejection is not lost -- it arrives as an ``error``
        notification, which the event loop surfaces as an ``ERROR`` event.
        """
        if not text or not self._streaming:
            return False
        if not self._thread_id or not self._turn_id:
            return False
        proc = self._rpc
        if proc is None or proc.poll() is not None:
            return False
        self._next_id += 1
        return self._send(
            {
                "id": self._next_id,
                "method": "turn/steer",
                "params": {
                    "threadId": self._thread_id,
                    "expectedTurnId": self._turn_id,
                    "input": [{"type": "text", "text": text}],
                },
            }
        )

    # -- cancel / close ----------------------------------------------------

    def cancel(self) -> None:
        """Stop the in-flight turn: ``turn/interrupt``, plus any fallback child."""
        self._cancelled = True
        proc = self._rpc
        if proc is not None and proc.poll() is None:
            self._decline_pending()
            if self._thread_id and self._turn_id:
                self._next_id += 1
                self._send(
                    {
                        "id": self._next_id,
                        "method": "turn/interrupt",
                        "params": {"threadId": self._thread_id, "turnId": self._turn_id},
                    }
                )
        super().cancel()

    def _teardown_app_server(self) -> None:
        # The escalation (stdin, wait, terminate, kill) is the shared
        # helper every adapter closes through (deviation d5).
        escalate_close(self._rpc, wait=_CLOSE_WAIT_SECONDS)
        for thread in (self._reader, self._stderr_reader):
            if thread is not None:
                thread.join(timeout=_CLOSE_WAIT_SECONDS)
        self._rpc = None
        self._reader = None
        self._stderr_reader = None
        self._thread_id = None
        self._turn_id = None
        self._streaming = False
        self._pending_approvals.clear()
        # Anything the dead server had said but nobody consumed belongs to a
        # session that no longer exists; carrying it into the next one (or
        # into the fallback path) would replay a stale stream.
        self._held.clear()
        self._queue = queue.Queue()

    def close(self) -> None:
        """Idempotent teardown of both the app-server and any fallback child."""
        if self._closed:
            return
        if self._rpc is not None and self._rpc.poll() is None:
            self._decline_pending()
        self._teardown_app_server()
        super().close()

    # -- fallback line mapping --------------------------------------------

    def _parse_line(self, line: str) -> AgentEvent | None:
        """One ``codex exec --json`` stdout line as an event (fallback path).

        Maps the ``{"msg": {"type": ...}}`` envelope: ``task_started`` ->
        ``STATUS``, ``agent_message_delta`` -> ``TEXT_DELTA``, ``error`` ->
        ``ERROR``, ``task_complete`` -> ``DONE``. Non-JSON lines and
        unrecognized ``msg`` types are skipped.
        """
        if not line.strip():
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        msg = obj.get("msg", {})
        kind = msg.get("type")
        if kind == "task_started":
            return AgentEvent(kind=EventKind.STATUS, text=msg.get("text", ""))
        if kind == "agent_message_delta":
            return AgentEvent(kind=EventKind.TEXT_DELTA, text=msg.get("delta", ""))
        if kind == "error":
            return AgentEvent(kind=EventKind.ERROR, error=msg.get("message", ""))
        if kind == "task_complete":
            return AgentEvent(kind=EventKind.DONE)
        return None

    # -- reporting ---------------------------------------------------------

    def binary_path(self) -> str:
        """Where this adapter's ``codex`` is, or ``""`` when it is not found.

        Resolved against the *adapter's own* ``PATH`` -- the one the child
        will be spawned with -- not the parent process's, so what is
        reported is what would actually run.
        """
        source = self._env if self._env is not None else os.environ
        return shutil.which(self.binary, path=source.get("PATH")) or ""

    def capabilities(self) -> Capabilities:
        """What this adapter can do.

        Reported for the app-server path, which is the one nvsh uses
        whenever the installed CLI has it. ``approval`` reports the
        configured mediator; codex's app-server has no approval UI of its
        own (the *client* is the UI), so nvsh mediates either way and the
        wire policy stays ``on-request``.
        """
        return Capabilities(
            streaming=True,
            tool_calling=True,
            cancellation=True,
            persistent_session=True,
            local_model=False,
            thinking=True,
            effort=True,
            path=self.binary_path(),
            approval=self._approval,
            unmediated_file_access=True,
        )
