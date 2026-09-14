"""AcpAgent: one adapter for every CLI that speaks the Agent Client Protocol.

ACP is JSON-RPC 2.0 over the child's stdio, one JSON object per line. nvsh
is the *client*: it drives ``initialize`` -> ``session/new`` ->
(``session/set_mode`` / ``session/set_config_option``) -> ``session/prompt``,
reads the agent's ``session/update`` notifications as the turn streams, and
answers the agent's one server-to-client request, ``session/request_permission``.

Because the protocol is the contract, a single class covers every ACP CLI;
what differs per harness is argv, which mode to sit in, and what it can do.
Those differences live in :data:`ENTRIES` (``qwen``, ``kiro``) and nowhere
else. Nothing here imports a vendor package: each harness is reached only
through argv.

Two rules this module enforces rather than merely documents:

* **nvsh advertises no filesystem and no terminal capability.** The
  ``initialize`` payload says ``fs.readTextFile = false`` and
  ``fs.writeTextFile = false`` and offers no ``terminal`` capability, so the
  agent uses its *own* tools rather than asking nvsh to read or write files
  on its behalf. That also means nvsh cannot see or mediate those reads and
  writes, which is exactly what
  :attr:`Capabilities.unmediated_file_access` reports (always ``True`` here).
* **No bypass mode, ever.** ``auto``, ``auto-edit`` and ``yolo`` (qwen's
  modes that approve tool calls without asking) and ``--trust-all-tools``
  (kiro's flag for the same) are refused at construction time, not merely
  left unset -- nvsh proposes and the operator approves (spec c7).

Wire shapes below were recorded from qwen-code 0.23.3 (``qwen --acp``) and
kiro-cli 2.21.4 (``kiro-cli acp``) on 2026-09-14; ``tests/fakes/acp``
replays that transcript.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import subprocess  # nosec B404 - fixed argv lists below, never shell=True
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ._env import child_env
from ._subprocess import escalate_close, redacted_tail
from .base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    NvshAgent,
    Proposal,
    ProposalKind,
)
from .prompt import build_full_prompt, build_prompt

#: The ACP revision nvsh speaks. Both probed CLIs answered ``initialize``
#: with ``protocolVersion: 1``.
PROTOCOL_VERSION = 1

#: The one server-to-client request nvsh answers: the agent asking
#: permission to run a tool call.
REQUEST_PERMISSION_METHOD = "session/request_permission"

#: How long a single queue get() waits before re-checking process liveness
#: and the cancel flag. Same bound, and same reason, as ``PiAgent``'s.
_POLL_INTERVAL_SECONDS = 0.2

#: How long ``initialize`` (and the session setup commands after it) may go
#: unanswered before the adapter gives up and says so out loud. An ACP CLI
#: answers ``initialize`` in well under a second once it has booted, so this
#: is a liveness bound, not a thinking budget: a turn's own thinking time is
#: bounded by the daemon's turn cap, never by this.
_INITIALIZE_TIMEOUT_SECONDS = 30.0

#: Environment override for that bound, in seconds (mirrors
#: ``NVSH_PI_ACK_TIMEOUT``).
INITIALIZE_TIMEOUT_ENV = "NVSH_ACP_INIT_TIMEOUT"

#: How long close() waits at each rung of its escalation.
_CLOSE_WAIT_SECONDS = 2.0

#: How much of the agent's assistant text is kept as a proposal's rationale
#: when the tool call carries no description of its own.
_RATIONALE_LIMIT = 600

#: Modes that approve tool calls without asking a human. Refused outright:
#: "propose, don't run" is not a default that may be configured away.
FORBIDDEN_MODES = frozenset({"auto", "auto-edit", "auto_edit", "autoedit", "yolo"})

#: Argv fragments that would do the same thing from the command line.
FORBIDDEN_ARGS = frozenset({"--trust-all-tools", "--yolo", "--dangerously-skip-permissions"})

#: ``session/update`` kinds that carry no news an operator at a failing
#: prompt can act on. Same judgement as ``PiAgent``'s ``_QUIET_EVENT_TYPES``
#: (deviation d11): genuinely unknown kinds still surface as STATUS, so
#: nothing new is dropped silently.
_QUIET_UPDATES = frozenset(
    {
        "available_commands_update",
        "usage_update",
        "current_mode_update",
        "user_message_chunk",
        "plan",
    }
)

#: Tool-call statuses that mean "still going" -- an update carrying one of
#: these is progress, not a result.
_PENDING_STATUSES = frozenset({"pending", "in_progress"})

#: "That poll came up empty." ``None`` already means EOF on the reader
#: queue, so waiting in vain needs a value of its own.
_NOTHING = object()


class AcpError(RuntimeError):
    """The ACP child did not hold up its end of the protocol.

    Raised rather than returned quietly: every caller turns an exception
    into an error the operator can read, and silence is the worse failure.
    """


@dataclass(frozen=True)
class AcpEntry:
    """One ACP harness: how to launch it and what it is allowed to do.

    ``default_mode`` is the mode nvsh puts the session in when it mediates
    approval itself; ``harness_mode`` is the mode it uses when the operator
    has handed approval to the harness (``[agents.<name>] approval =
    "harness"``). ``None`` means "do not send ``session/set_mode`` at all",
    which is right for kiro: its modes are the operator's own agent configs,
    not a permission ladder.
    """

    name: str
    command: tuple[str, ...]
    binary: str
    description: str
    thinking: bool = False
    local_model: bool = False
    effort: bool = False
    #: Whether this harness asks nvsh for permission before running a tool
    #: in ``default_mode``. When it does, nvsh's own approve loop is the gate
    #: and the adapter may report ``tool_calling``.
    approval_channel: bool = False
    default_mode: str | None = None
    harness_mode: str | None = None
    #: Flag that selects a model on the command line, when the harness has
    #: one. ``None`` means the model is chosen in-session instead
    #: (``session/set_config_option``).
    model_flag: str | None = None


#: The ACP harnesses nvsh knows about. Add a harness here, not a new class.
ENTRIES: dict[str, AcpEntry] = {
    "qwen": AcpEntry(
        name="qwen",
        command=("qwen", "--acp"),
        binary="qwen",
        description="Qwen Code over ACP (qwen --acp); plan mode unless approval = 'harness'.",
        thinking=True,
        local_model=True,
        effort=True,
        # In plan mode qwen analyses and does not execute, so there is no
        # approval round trip to report (decision c53).
        approval_channel=False,
        default_mode="plan",
        harness_mode="default",
        model_flag=None,  # qwen picks its model with session/set_config_option
    ),
    "kiro": AcpEntry(
        name="kiro",
        command=("kiro-cli", "acp"),
        binary="kiro-cli",
        description="Kiro CLI over ACP (kiro-cli acp); asks permission per tool call.",
        thinking=False,
        local_model=False,
        effort=False,
        approval_channel=True,
        # kiro's modes are the operator's own agent configs; nvsh does not
        # second-guess which one they picked.
        default_mode=None,
        harness_mode=None,
        model_flag="--model",
    ),
}


@dataclass(frozen=True)
class AcpPolicy:
    """The per-harness knobs that are neither argv nor the session's mode.

    Three of them are reported straight back out of
    :meth:`AcpAgent.capabilities`; ``model_flag`` is how -- or whether --
    the harness takes a model on the command line. They travel together
    because they are set together, once, from the harness's
    :class:`AcpEntry`, and keeping them in one frozen value stops
    :meth:`AcpAgent.__init__` from growing a parameter per capability.
    """

    local_model: bool = False
    effort_supported: bool = False
    tool_calling: bool = False
    model_flag: str | None = None


#: The policy an agent gets when the caller names none: a harness that
#: reports nothing extra and takes no model flag.
_DEFAULT_POLICY = AcpPolicy()


def initialize_timeout(env: Mapping[str, str] | None = None) -> float:
    """How long to wait for the ``initialize`` result, in seconds."""
    resolved = os.environ if env is None else env
    try:
        value = float(resolved.get(INITIALIZE_TIMEOUT_ENV, ""))
    except (TypeError, ValueError):
        return _INITIALIZE_TIMEOUT_SECONDS
    return value if value > 0 else _INITIALIZE_TIMEOUT_SECONDS


def _text_of(content: object) -> str:
    """The text inside an ACP ``content`` value, whatever shape it takes."""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        if isinstance(content.get("text"), str):
            return str(content["text"])
        inner = content.get("content")
        if inner is not None:
            return _text_of(inner)
        return ""
    if isinstance(content, Sequence):
        return "".join(_text_of(item) for item in content)
    return ""


def _tool_name(payload: Mapping[str, object]) -> str:
    """A tool call's name: the harness's own, else its human title."""
    meta = payload.get("_meta")
    if isinstance(meta, Mapping) and isinstance(meta.get("toolName"), str):
        return str(meta["toolName"])
    title = payload.get("title")
    return str(title) if isinstance(title, str) else ""


def _proposal_fields(tool_call: Mapping[str, object]) -> tuple[str, str]:
    """``(command, rationale)`` for one ``session/request_permission``.

    The command is only ever read from a field that *holds* a command --
    ``rawInput.command``, which both probed harnesses populate. A human
    title is never treated as a command (deviation d8 on the pi side): a
    permission request that carries no command yields ``""``, and the loop
    then runs nothing even if the operator approves it.
    """
    raw_input = tool_call.get("rawInput")
    command = ""
    rationale = ""
    if isinstance(raw_input, Mapping):
        if isinstance(raw_input.get("command"), str):
            command = str(raw_input["command"])
        for key in ("description", "__tool_use_purpose", "purpose", "reason"):
            value = raw_input.get(key)
            if isinstance(value, str) and value:
                rationale = value
                break
    return command, rationale


def _is_response_to(obj: object, request_id: int) -> bool:
    """Is this frame the response to the request nvsh is waiting on?"""
    return isinstance(obj, Mapping) and obj.get("id") == request_id and "method" not in obj


def _is_server_request(obj: object) -> bool:
    """Is this frame a request *from* the harness (a method with an id)?"""
    return isinstance(obj, Mapping) and bool(obj.get("method")) and obj.get("id") is not None


def _matching_value(values: object, wanted: str) -> str | None:
    """The first of one config option's advertised values that ``wanted`` names.

    Exact value first, then the option's display name, then a prefix -- qwen
    names a model ``"worker(openai)"`` while an operator writes ``"worker"``.
    ``None`` means the option advertised nothing usable, and the caller then
    leaves the setting alone.
    """
    if not isinstance(values, list):
        return None
    for candidate in values:
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get("value")
        if not isinstance(value, str):
            continue
        if wanted in (value, candidate.get("name")) or value.startswith(wanted):
            return value
    return None


def _selected_option(options: Sequence[object], allow: bool) -> tuple[str | None, str]:
    """Pick the option to answer a permission request with.

    ``allow_always`` / ``reject_always`` are never selected: an "always"
    answer would move the decision inside the harness's own store, where
    nvsh's approve loop can no longer see or revoke it. Returns
    ``(optionId, outcome)``; ``optionId`` is ``None`` when the harness
    offered nothing usable, and the caller then cancels the request outright.
    """
    wanted = "allow_once" if allow else "reject_once"
    for option in options:
        if isinstance(option, Mapping) and option.get("kind") == wanted:
            option_id = option.get("optionId")
            if isinstance(option_id, str):
                return option_id, "selected"
    return None, "cancelled"


class AcpAgent(NvshAgent):
    """Drives one ACP CLI as a persistent, session-holding subprocess."""

    def __init__(
        self,
        command: list[str],
        name: str,
        model: str | None = None,
        effort: str | None = None,
        extra_args: list[str] | None = None,
        approval: str = "nvsh",
        *,
        mode: str | None = None,
        thinking: bool = False,
        policy: AcpPolicy | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        initialize_timeout_seconds: float | None = None,
    ) -> None:
        if not command:
            raise ValueError("an ACP agent needs a command to launch")
        self._command = list(command)
        self._name = name
        self._model = model
        self._effort = effort
        self._extra_args = list(extra_args or [])
        self._approval = approval
        self._mode = mode
        self._thinking = thinking
        self._policy = policy if policy is not None else _DEFAULT_POLICY
        self._model_flag = self._policy.model_flag
        self._cwd = str(cwd) if cwd is not None else os.getcwd()
        #: An explicit ``cwd`` pins the session; otherwise the first request's
        #: ``context.cwd`` wins over the daemon's own launch directory.
        self._cwd_pinned = cwd is not None
        self._env = child_env(env)
        self._initialize_timeout = (
            initialize_timeout(env)
            if initialize_timeout_seconds is None
            else initialize_timeout_seconds
        )
        self._check_no_bypass()

        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        self._queue: queue.Queue = queue.Queue()
        #: Objects pulled off the queue while waiting for one response but
        #: belonging to the stream. Read before the queue so ordering holds.
        self._held: collections.deque = collections.deque()
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._session_id = ""
        self._session_modes: dict = {}
        self._config_options: list = []
        self._sent_system_prompt = False
        #: request_id (as a string, the way the daemon and panel pass it
        #: around) -> (raw JSON-RPC id, options) for every permission
        #: request still waiting for an answer.
        self._pending: dict[str, tuple[object, list]] = {}
        self._said: list[str] = []
        self._cancelled = False
        self._closed = False

    # -- construction-time guards ------------------------------------------

    def _check_no_bypass(self) -> None:
        """Refuse a configuration that would run tools without approval."""
        if self._mode is not None and self._mode.lower() in FORBIDDEN_MODES:
            raise ValueError(
                f"{self._name}: mode {self._mode!r} approves tool calls without asking; "
                "nvsh proposes and the operator approves"
            )
        for arg in self._command + self._extra_args:
            if arg.lower() in FORBIDDEN_ARGS:
                raise ValueError(f"{self._name}: {arg} bypasses approval; nvsh never passes it")

    # -- argv / lifecycle --------------------------------------------------

    def build_argv(self) -> list[str]:
        """The child's argv: the entry's command, its model flag, extra args."""
        argv = list(self._command)
        if self._model and self._model_flag:
            argv += [self._model_flag, str(self._model)]
        argv += self._extra_args
        return argv

    def start(self) -> None:
        """Spawn the child and set its session up. Idempotent.

        Raises :class:`AcpError` when the child dies on launch or never
        answers ``initialize`` within :func:`initialize_timeout` seconds --
        ``run`` turns that into a single ERROR event and closes, so a hung
        harness costs the operator one bounded wait instead of the whole
        turn cap.
        """
        if self._proc is not None:
            return
        self._closed = False
        self._cancelled = False
        self._held.clear()
        self._stderr_tail.clear()
        self._sent_system_prompt = False

        argv = self.build_argv()
        try:
            self._proc = subprocess.Popen(  # nosec B603 - fixed argv list, no shell
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self._env,
                cwd=self._cwd,
            )
        except OSError as exc:
            raise AcpError(f"failed to start {argv[0]}: {exc}") from exc

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()
        try:
            self._handshake()
        except AcpError:
            self.close()
            raise

    def _handshake(self) -> None:
        """initialize, then session/new, then mode / model / effort."""
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                # No filesystem, no terminal: the agent uses its own tools,
                # and nvsh does not pretend to mediate what it cannot see.
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}},
            },
        )
        session = self._request("session/new", {"cwd": self._cwd, "mcpServers": []})
        self._adopt_session(session)
        if self._mode:
            self._request("session/set_mode", {"sessionId": self._session_id, "modeId": self._mode})
        self._apply_config_option("model", self._model)
        self._apply_config_option("reasoning_effort", self._effort)

    def _adopt_session(self, result: Mapping[str, object]) -> None:
        session_id = result.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise AcpError(f"{self._name} returned no sessionId from session/new")
        self._session_id = session_id
        modes = result.get("modes")
        self._session_modes = dict(modes) if isinstance(modes, Mapping) else {}
        options = result.get("configOptions")
        self._config_options = list(options) if isinstance(options, list) else []

    def _config_value(self, config_id: str, wanted: str) -> str | None:
        """Match ``wanted`` against one advertised config option's values.

        The matching itself is :func:`_matching_value`. ``None`` means the
        harness advertised no such option, or no matching value, and nvsh
        then leaves the setting alone rather than sending something the
        harness would reject.
        """
        for option in self._config_options:
            if isinstance(option, Mapping) and option.get("id") == config_id:
                return _matching_value(option.get("options"), wanted)
        return None

    def _apply_config_option(self, config_id: str, wanted: str | None) -> None:
        """Best-effort ``session/set_config_option``; never fatal.

        A harness that does not advertise the option (kiro advertises none)
        or does not know the value simply keeps its own default: a model or
        effort that could not be selected is not a reason to refuse the
        turn, and the panel still shows what the harness actually is.
        """
        if not wanted:
            return
        value = self._config_value(config_id, str(wanted))
        if value is None:
            return
        try:
            self._request(
                "session/set_config_option",
                {"sessionId": self._session_id, "configId": config_id, "value": value},
            )
        except AcpError:
            return

    def resume(self, session_id: str) -> None:
        """Resume a stored session instead of the one ``start()`` created."""
        self.start()
        result = self._request(
            "session/resume",
            {"sessionId": session_id, "cwd": self._cwd, "mcpServers": []},
        )
        self._session_id = session_id
        modes = result.get("modes")
        if isinstance(modes, Mapping):
            self._session_modes = dict(modes)
        options = result.get("configOptions")
        if isinstance(options, list):
            self._config_options = list(options)

    @property
    def session_id(self) -> str:
        """The session this adapter is holding, or ``""`` before ``start()``."""
        return self._session_id

    # -- reading -----------------------------------------------------------

    def _reader_loop(self) -> None:
        """One JSON object per line onto the queue; ``None`` marks EOF."""
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for raw_line in self._proc.stdout:
                text = raw_line.strip()
                if not text:
                    continue
                try:
                    self._queue.put(json.loads(text))
                except json.JSONDecodeError:
                    # Some ACP CLIs print a banner before the first frame.
                    # Non-JSON is noise on this channel, not a protocol
                    # error, so it is kept for a failure message and
                    # otherwise ignored.
                    self._stderr_tail.append(text + "\n")
        except (OSError, ValueError):  # closed underneath us by close()/cancel()
            pass
        finally:
            self._queue.put(None)

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:  # pragma: no cover - always piped
            return
        try:
            for line in proc.stderr:
                self._stderr_tail.append(line)
        except (OSError, ValueError):
            pass

    def _exit_detail(self) -> str:
        """How the child is doing, for the tail of an error message."""
        proc = self._proc
        if proc is None:
            return f" (no {self._name} process)"
        code = proc.poll()
        state = "still running" if code is None else f"exited with code {code}"
        stderr = redacted_tail(self._stderr_tail)
        tail = " | ".join(line.strip() for line in stderr.splitlines() if line.strip())
        if tail:
            return f" ({self._name} {state}; stderr tail: {tail})"
        return f" ({self._name} {state}; no stderr)"

    # -- writing -----------------------------------------------------------

    def _send(self, obj: dict) -> bool:
        """Write one JSON-RPC frame. ``False`` when the child's stdin is gone."""
        proc = self._proc
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

    def _send_request(self, method: str, params: dict) -> int:
        """Send a request without waiting; returns its JSON-RPC id."""
        self._next_id += 1
        request_id = self._next_id
        frame = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        if not self._send(frame):
            raise AcpError(f"could not send {method} to {self._name}{self._exit_detail()}")
        return request_id

    def _request(self, method: str, params: dict, timeout: float | None = None) -> dict:
        """Send a request and wait for *its* response."""
        request_id = self._send_request(method, params)
        return self._await_response(
            method, request_id, self._initialize_timeout if timeout is None else timeout
        )

    def _next_object(self, timeout: float) -> Any:
        if self._held:
            return self._held.popleft()
        return self._queue.get(timeout=timeout)

    def _await_response(self, method: str, request_id: int, timeout: float) -> dict:
        """Wait for one response, holding everything else for the stream.

        Server-to-client *requests* that arrive meanwhile are answered right
        here (see :meth:`_answer_unknown_request`) rather than held: a
        harness that is blocked on an unanswered request of its own would
        never get around to answering ours.
        """
        carried: list = []
        deadline = time.monotonic() + timeout
        try:
            while True:
                obj = self._poll_frame(method, deadline, timeout)
                if obj is _NOTHING:
                    continue
                if obj is None:
                    carried.append(None)
                    raise AcpError(
                        f"{self._name} closed its output before answering {method}"
                        f"{self._exit_detail()}"
                    )
                if _is_response_to(obj, request_id):
                    return self._response_result(method, obj)
                if _is_server_request(obj):
                    if obj.get("method") != REQUEST_PERMISSION_METHOD:
                        self._answer_unknown_request(obj)
                        continue
                carried.append(obj)
        finally:
            self._held.extendleft(reversed(carried))

    def _poll_frame(self, method: str, deadline: float, timeout: float) -> Any:
        """One frame from the child, or :data:`_NOTHING` when none came.

        Raises :class:`AcpError` when ``deadline`` has passed or the child
        died without answering; ``_NOTHING`` means only that this poll came
        up empty and the caller should go round again.
        """
        left = deadline - time.monotonic()
        if left <= 0:
            raise AcpError(
                f"{self._name} did not answer {method} within {timeout:g}s{self._exit_detail()}"
            )
        try:
            return self._next_object(min(_POLL_INTERVAL_SECONDS, left))
        except queue.Empty:
            if self._proc is not None and self._proc.poll() is not None:
                raise AcpError(
                    f"{self._name} went away before answering {method}{self._exit_detail()}"
                ) from None
            return _NOTHING

    def _response_result(self, method: str, obj: Mapping[str, object]) -> dict:
        """The ``result`` of our own response frame; an error frame raises."""
        if "error" in obj:
            raise AcpError(f"{self._name} rejected {method}: {_error_text(obj)}")
        result = obj.get("result")
        return dict(result) if isinstance(result, Mapping) else {}

    def _answer_unknown_request(self, obj: Mapping[str, object]) -> None:
        """Tell the harness nvsh does not implement one of its requests.

        Both probed CLIs send client requests of their own beyond the ACP
        core (``craft/drainMidTurnQueue`` from qwen, ``_kiro.dev/*`` from
        kiro). Answering with the standard ``-32601`` is what a JSON-RPC
        client owes them; verified against qwen 0.23.3, where the turn then
        continues normally to ``stopReason: end_turn``.
        """
        self._send(
            {
                "jsonrpc": "2.0",
                "id": obj.get("id"),
                "error": {"code": -32601, "message": "method not found"},
            }
        )

    # -- running -----------------------------------------------------------

    def _prompt_text(self, request: AgentRequest, context: AgentContext) -> str:
        """One turn's prompt.

        ACP has no system-prompt channel, so the brief rides at the head of
        the session's *first* prompt and never again -- the harness keeps
        the conversation, so repeating it would only cost tokens.
        """
        if self._sent_system_prompt:
            return build_prompt(request, context)
        self._sent_system_prompt = True
        return build_full_prompt(request, context)

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        request_cwd = getattr(context, "cwd", None)
        if request_cwd and not self._cwd_pinned and self._session_id is None:
            self._cwd = str(request_cwd)
        try:
            self.start()
            prompt_id = self._send_request(
                "session/prompt",
                {
                    "sessionId": self._session_id,
                    "prompt": [{"type": "text", "text": self._prompt_text(request, context)}],
                },
            )
        except AcpError as exc:
            self.close()
            yield AgentEvent(kind=EventKind.ERROR, error=str(exc))
            return

        self._cancelled = False
        self._said.clear()
        try:
            yield from self._events(prompt_id)
        finally:
            self._said.clear()

    def _events(self, prompt_id: int) -> Iterator[AgentEvent]:
        """The turn's event loop, from the sent prompt to DONE/ERROR."""
        while not self._cancelled:
            # A dead child is only the end of the turn once everything it
            # already said has been drained -- a fast final burst (including
            # a real error frame) must not be lost to the exit notice.
            drained = not self._held and self._queue.empty()
            if self._proc is not None and self._proc.poll() is not None and drained:
                yield AgentEvent(
                    kind=EventKind.ERROR, error=f"{self._name} process exited{self._exit_detail()}"
                )
                return
            try:
                obj = self._next_object(_POLL_INTERVAL_SECONDS)
            except queue.Empty:
                continue
            event = self._map(obj, prompt_id)
            if event is None:
                continue
            yield event
            if self._cancelled or event.kind in (EventKind.DONE, EventKind.ERROR):
                return

    def _map(self, obj: Any, prompt_id: int) -> AgentEvent | None:
        """One frame from the child as an event, or ``None`` to skip it."""
        if obj is None:
            return AgentEvent(
                kind=EventKind.ERROR, error=f"{self._name} closed its output{self._exit_detail()}"
            )
        if not isinstance(obj, Mapping):  # pragma: no cover - reader only queues dicts/None
            return None

        method = obj.get("method")
        if method == "session/update":
            params = obj.get("params")
            update = params.get("update") if isinstance(params, Mapping) else None
            return self._update_event(update) if isinstance(update, Mapping) else None
        if method == REQUEST_PERMISSION_METHOD:
            return self._permission_event(obj)
        if method:
            if obj.get("id") is not None:
                self._answer_unknown_request(obj)
            return None  # vendor notifications are not panel material

        if obj.get("id") == prompt_id:
            if "error" in obj:
                return AgentEvent(kind=EventKind.ERROR, error=_error_text(obj))
            return AgentEvent(kind=EventKind.DONE)
        return None

    def _update_event(self, update: Mapping[str, object]) -> AgentEvent | None:
        kind = str(update.get("sessionUpdate") or "")

        if kind == "agent_thought_chunk":
            return AgentEvent(kind=EventKind.THINKING, text=_text_of(update.get("content")))
        if kind == "agent_message_chunk":
            text = _text_of(update.get("content"))
            if not text:
                # The final chunk of a qwen message is an empty string
                # carrying only usage metadata.
                return None
            self._said.append(text)
            return AgentEvent(kind=EventKind.TEXT_DELTA, text=text)
        if kind == "tool_call":
            raw_input = update.get("rawInput")
            return AgentEvent(
                kind=EventKind.TOOL_CALL,
                tool=_tool_name(update),
                args=dict(raw_input) if isinstance(raw_input, Mapping) else {},
            )
        if kind == "tool_call_update":
            status = str(update.get("status") or "")
            if status in _PENDING_STATUSES:
                return None
            # A finished tool resets the rationale window: what the model
            # says next is about the next thing it wants to do.
            self._said.clear()
            return AgentEvent(
                kind=EventKind.TOOL_RESULT,
                tool=_tool_name(update),
                result=_text_of(update.get("content")),
            )
        if kind in _QUIET_UPDATES:
            return None
        return AgentEvent(kind=EventKind.STATUS, text=kind)

    def _permission_event(self, obj: Mapping[str, object]) -> AgentEvent:
        """One ``session/request_permission`` as a PROPOSAL the panel shows.

        The request is remembered, not answered: nvsh's approve loop decides,
        and :meth:`respond_ui` relays that answer back as the harness's own
        ``allow_once`` / ``reject_once`` option.
        """
        params = obj.get("params")
        params = params if isinstance(params, Mapping) else {}
        tool_call = params.get("toolCall")
        tool_call = tool_call if isinstance(tool_call, Mapping) else {}
        options = params.get("options")
        options = list(options) if isinstance(options, list) else []

        raw_id = obj.get("id")
        request_id = str(raw_id)
        self._pending[request_id] = (raw_id, options)

        command, rationale = _proposal_fields(tool_call)
        return AgentEvent(
            kind=EventKind.PROPOSAL,
            proposal=Proposal(
                command=command,
                rationale=rationale or self._rationale_text() or str(tool_call.get("title") or ""),
                kind=ProposalKind.FIX,
            ),
            args={"request_id": request_id, "method": REQUEST_PERMISSION_METHOD},
        )

    def _rationale_text(self) -> str:
        """The assistant text since the last tool result, trimmed for a panel."""
        text = " ".join("".join(self._said).split())
        if len(text) <= _RATIONALE_LIMIT:
            return text
        return text[: _RATIONALE_LIMIT - 1] + "…"

    # -- answering ---------------------------------------------------------

    def respond_ui(self, request_id: str, **fields: object) -> None:
        """Answer a pending permission request with the operator's decision.

        Called by the daemon/panel with the same field vocabulary every
        other adapter gets (``value="once"``/``"deny"``/a scope token, or
        ``cancelled=True``). Only ``allow_once`` and ``reject_once`` are ever
        selected: an "always" answer would move the decision into the
        harness's own store where nvsh's approve loop cannot see or revoke
        it (spec c7), so a widened scope is recorded on nvsh's side alone.
        """
        pending = self._pending.pop(str(request_id), None)
        if pending is None:
            return
        raw_id, options = pending
        allow = _is_allow(fields)
        option_id, outcome = _selected_option(options, allow)
        if outcome == "selected":
            result: dict[str, object] = {"outcome": {"outcome": "selected", "optionId": option_id}}
        else:
            result = {"outcome": {"outcome": "cancelled"}}
        self._send({"jsonrpc": "2.0", "id": raw_id, "result": result})

    # -- cancel / close ----------------------------------------------------

    def cancel(self) -> None:
        """Stop yielding, cancel the turn, and release any open dialog.

        A permission request left unanswered would leave the harness parked
        forever, so anything still pending is cancelled before the
        ``session/cancel`` notification goes out.
        """
        self._cancelled = True
        if self._proc is None or self._proc.poll() is not None:
            self._pending.clear()
            return
        # respond_ui() always pops the id it is given, so draining the dict
        # by repeatedly answering its first key terminates without a
        # snapshot copy.
        while self._pending:
            request_id = next(iter(self._pending))
            self.respond_ui(request_id, cancelled=True)
        if self._session_id:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": self._session_id},
                }
            )

    def close(self) -> None:
        """Idempotent teardown through the shared escalation helper (d5)."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        self._proc = None
        self._pending.clear()
        if proc is None:
            return
        escalate_close(proc, wait=_CLOSE_WAIT_SECONDS)
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=_CLOSE_WAIT_SECONDS)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=self._policy.tool_calling,
            cancellation=True,
            persistent_session=True,
            local_model=self._policy.local_model,
            thinking=self._thinking,
            effort=self._policy.effort_supported,
            # Not a filesystem path: every ACP harness is reached the same
            # way, over the protocol, and that is what a caller needs to know.
            path="acp",
            approval=self._approval,
            # nvsh advertises no fs capability, so the agent reads and writes
            # with its own tools and nvsh never sees those calls.
            unmediated_file_access=True,
        )


def _error_text(obj: Mapping[str, object]) -> str:
    """Readable text for a JSON-RPC error frame."""
    error = obj.get("error")
    if isinstance(error, Mapping):
        message = str(error.get("message", "") or "")
        data = error.get("data")
        if data is not None:
            return f"{message}: {json.dumps(data)}" if message else json.dumps(data)
        return message or "unknown error"
    return str(error or "unknown error")


def _is_allow(fields: Mapping[str, object]) -> bool:
    """Did the operator approve? Anything but an explicit yes is a no."""
    if fields.get("cancelled"):
        return False
    if "confirmed" in fields:
        return bool(fields["confirmed"])
    value = fields.get("value")
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text or text.startswith("deny") or text in {"no", "reject", "cancel"}:
        return False
    return True


def build(
    name: str,
    settings: Mapping[str, object] | None = None,
    *,
    model: str | None = None,
    effort: str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> AcpAgent:
    """Build the :class:`AcpAgent` for registered harness ``name``.

    ``settings`` is that harness's ``[agents.<name>]`` table; ``model`` and
    ``effort`` are the per-request overrides a routed ``Target`` carries and
    win over the configured values.

    Approval decides the session's mode (decision c53): with nvsh mediating
    (the default) qwen sits in ``plan``, where it analyses and proposes but
    does not execute, and reports ``tool_calling=False``. Only when the
    operator writes ``approval = "harness"`` does it move to ``default`` --
    the mode where qwen asks before each tool call -- and report both
    ``approval="harness"`` and ``tool_calling=True``. ``auto``, ``auto-edit``
    and ``yolo`` are never reachable from any setting.
    """
    try:
        entry = ENTRIES[name]
    except KeyError:
        raise ValueError(
            f"unknown ACP harness {name!r}: expected one of {sorted(ENTRIES)}"
        ) from None

    table = dict(settings or {})
    approval = str(table.get("approval") or "nvsh")
    harness_approval = approval == "harness"
    mode = entry.harness_mode if harness_approval else entry.default_mode
    raw_extra = table.get("extra_args")
    extra_args = [str(item) for item in raw_extra] if isinstance(raw_extra, list) else []

    resolved_model = model if model is not None else table.get("model")
    resolved_effort = effort if effort is not None else table.get("effort")

    return AcpAgent(
        list(entry.command),
        entry.name,
        str(resolved_model) if resolved_model is not None else None,
        str(resolved_effort) if resolved_effort is not None else None,
        extra_args,
        approval,
        mode=mode,
        thinking=entry.thinking,
        policy=AcpPolicy(
            local_model=entry.local_model,
            effort_supported=entry.effort,
            tool_calling=entry.approval_channel or harness_approval,
            model_flag=entry.model_flag,
        ),
        env=env,
        cwd=cwd,
    )
