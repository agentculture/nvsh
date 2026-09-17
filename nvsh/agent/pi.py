"""PiAgent: an :class:`NvshAgent` adapter that drives ``pi --mode rpc``.

Spawns ``pi`` (https://github.com/earendil-works/pi-coding-agent, "Pi.dev")
as one long-lived subprocess with plain :mod:`subprocess` pipes -- no
``asyncio`` -- so a shell hook that only ever runs one request at a time
does not pay for an event loop it does not need. A background reader thread
drains stdout and puts parsed JSON objects on a :class:`queue.Queue`;
``run()`` pulls from that queue with a timeout so a dead or hung process is
always noticed rather than hanging the caller.

The wire protocol is summarized in ``docs/pi-rpc.md`` (itself distilled from
the real ``@earendil-works/pi-coding-agent`` package's shipped
``docs/rpc.md``). Nothing here imports the ``pi`` package -- it is an
external CLI reached only through argv.
"""

from __future__ import annotations

import collections
import importlib.resources
import json
import os
import queue
import shutil
import subprocess  # external `pi` CLI (bandit B404, allowed repo-wide)
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..redact import redact
from ._subprocess import escalate_close, kill_tree
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
from .prompt import build_prompt as _build_prompt

#: How long a single queue.Queue.get() waits before re-checking process
#: liveness and the cancellation flag. Keeps run() responsive to a killed
#: process or a cancel() without busy-waiting.
_POLL_INTERVAL_SECONDS = 0.2

#: pi's per-turn lifecycle and progress bookkeeping. These say only that the
#: rpc loop is doing its job, which is nothing an operator at a failing
#: prompt can act on, so they map to no event at all -- deviation d11, where
#: the catch-all STATUS fallback below put "... agent_start",
#: "... turn_start", "... message_start", "... message_end" straight into
#: the panel, and a live run against nemotron/associate added three
#: "... tool_execution_update" lines per tool call on top. The panel already
#: says which tool is running and when it finished. Genuinely unknown event
#: types still surface as STATUS, so nothing new is dropped silently.
_QUIET_EVENT_TYPES = frozenset(
    {
        "agent_start",
        "turn_start",
        "turn_end",
        "message_start",
        "message_end",
        "message_final",
        "agent_settled",
        "tool_execution_update",
        # Routine from d16 on: every steer changes pi's steering queue, so a
        # steered turn printed "... queue_update" twice for one instruction.
        "queue_update",
    }
)

#: How long close() waits for a clean exit after closing stdin, and then
#: after terminate(), before escalating.
_CLOSE_WAIT_SECONDS = 2.0

#: How long one rpc *command* may go unacknowledged before PiAgent gives up
#: and says so out loud. Commands are acknowledged in milliseconds once pi
#: is up (measured: ~0.2 s on the Spark, cold), so this is a liveness bound,
#: not a thinking budget -- a turn's own thinking time is bounded by the
#: daemon's turn cap, not by this.
_ACK_TIMEOUT_SECONDS = 20.0

#: Environment override for that bound, in seconds.
ACK_TIMEOUT_ENV = "NVSH_PI_ACK_TIMEOUT"

#: How much of pi's stderr is kept for the tail of an error message.
_STDERR_TAIL_BYTES = 2048

#: How much of the assistant text that preceded a tool call is kept as that
#: proposal's rationale (deviation d17). Long enough for the two or three
#: sentences a model spends saying what it found, short enough that the
#: panel still shows a proposal and not an essay.
_RATIONALE_LIMIT = 600


class PiRpcError(RuntimeError):
    """pi did not hold up its end of the rpc protocol.

    Raised instead of returning quietly: every caller (the daemon's
    ``_run_locked``, ``client_transport.one_shot``) turns an exception into
    an ``error`` event the operator can read, and silence is exactly the
    failure mode deviation d14 was.
    """


def ack_timeout(env: Mapping[str, str] | None = None) -> float:
    """How long to wait for one command's ``response`` ack, in seconds."""
    resolved = os.environ if env is None else env
    try:
        value = float(resolved.get(ACK_TIMEOUT_ENV, ""))
    except (TypeError, ValueError):
        return _ACK_TIMEOUT_SECONDS
    return value if value > 0 else _ACK_TIMEOUT_SECONDS


def _default_pi_agent_config() -> dict[str, object]:
    """Return the built-in ``[agents.pi]`` defaults without touching disk.

    Imported lazily-by-value (not via ``nvsh.config.load()``) so constructing
    a :class:`PiAgent` never reads ``config.toml`` off the caller's real
    filesystem; a caller that wants the operator's configured provider/model
    reads ``nvsh.config.load()`` itself and passes the result in.
    """
    from nvsh.config import Config

    return dict(Config().agents.get("pi", {}))


def default_session_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve ``$XDG_STATE_HOME/nvsh/pi-sessions`` through an injectable env mapping.

    Falls back to ``$HOME/.local/state/nvsh/pi-sessions`` when
    ``XDG_STATE_HOME`` is unset, per the XDG base directory spec -- the same
    pattern ``nvsh.agent.audit.default_audit_path`` uses.
    """
    resolved_env = os.environ if env is None else env
    xdg_state_home = resolved_env.get("XDG_STATE_HOME")
    if xdg_state_home:
        base = Path(xdg_state_home)
    else:
        home = resolved_env.get("HOME") or os.path.expanduser("~")
        base = Path(home) / ".local" / "state"
    return base / "nvsh" / "pi-sessions"


def resolve_nvsh_bin(env: Mapping[str, str]) -> str | None:
    """Where ``nvsh`` is, for the approval extension running under pi.

    ``nvsh setup`` exports ``NVSH_BIN`` into the operator's rc file, so an
    env that came from a hooked shell already says. Otherwise look on that
    env's own ``PATH``, and finally next to the interpreter running nvsh --
    which is where a ``uv tool`` / venv console script lives even when its
    ``bin`` directory is not on the daemon's ``PATH``. ``None`` means "not
    found": the extension then falls back to a bare ``nvsh`` lookup and, if
    that fails too, reports the failure instead of asking (d21).
    """
    declared = env.get("NVSH_BIN")
    if declared:
        return declared
    found = shutil.which("nvsh", path=env.get("PATH"))
    if found:
        return found
    beside = Path(sys.executable).with_name("nvsh")
    return str(beside) if beside.exists() else None


def child_env(env: Mapping[str, str]) -> dict[str, str]:
    """The environment ``pi`` -- and the approval extension under it -- gets.

    Deviation d21: the extension shells out to ``nvsh approve check`` on
    every tool call, so it has to find the same binary and read the same
    store the panel writes. A daemon-spawned pi inherits whatever env the
    shell that first triggered it had; when that env named neither
    ``NVSH_BIN`` nor an ``nvsh`` on ``PATH``, every spawn failed with ENOENT
    and every command was asked about again, approved or not. So the
    variables the extension depends on are made explicit here.

    Only *missing* values are filled in, and only with the same defaults the
    readers themselves use (:func:`nvsh.approvals._config_dir`,
    :func:`nvsh.agent.audit.default_audit_path`), so parent and child can
    never disagree about where the store is. ``XDG_RUNTIME_DIR`` is the one
    exception: it is set only when ``/run/user/<uid>`` exists, because
    :func:`nvsh.approvals.runtime_dir`'s last resort is a *differently
    named* temp directory and inventing a value here would split the session
    store in two.
    """
    resolved = dict(env)
    nvsh_bin = resolve_nvsh_bin(resolved)
    if nvsh_bin:
        resolved["NVSH_BIN"] = nvsh_bin
    home = resolved.get("HOME") or os.path.expanduser("~")
    resolved.setdefault("XDG_CONFIG_HOME", str(Path(home) / ".config"))
    resolved.setdefault("XDG_STATE_HOME", str(Path(home) / ".local" / "state"))
    if not resolved.get("XDG_RUNTIME_DIR"):
        run_user = Path(f"/run/user/{os.getuid()}")
        if run_user.is_dir():
            resolved["XDG_RUNTIME_DIR"] = str(run_user)
    return resolved


def default_approval_extension_path() -> Path:
    """Path to the approval extension ``pi -e`` loads.

    Computed with :mod:`importlib.resources` so it works from an installed
    wheel, not just a checkout. The file itself (``approval.ts``) is written
    by a later task (t11); this only computes where it will live. Building
    an argv with this path does not require the file to exist yet.
    """
    return Path(str(importlib.resources.files("nvsh.agent") / "pi_ext" / "approval.ts"))


#: The prompt text sent to pi for one request, context folded in. Shared
#: verbatim with every other adapter (see :mod:`nvsh.agent.prompt`) so what
#: ``nvsh context --show`` prints is what each backend actually sends.
build_prompt = _build_prompt


def default_system_prompt() -> str:
    """The system brief for *this* machine, composed once per pi process.

    pi takes ``--append-system-prompt <text>`` (verified against the
    installed CLI's ``--help``, pi 0.85.x; documented in ``docs/pi-rpc.md``),
    so the brief is a launch flag: it costs its tokens once for the session
    instead of riding every turn's prompt. The detected platform block is
    read here rather than taken from a request's :class:`AgentContext`
    because argv is built before any request exists -- and it is the same
    machine either way. Detection failing is not fatal: the brief degrades
    to its generic playbook.
    """
    from .prompt import build_system_prompt

    try:
        from ..platform import detect

        block = detect().render_block()
    except Exception:  # noqa: BLE001 - a generic brief beats no brief
        block = ""
    return build_system_prompt(AgentContext(platform=block))


def _approval_envelope(obj: Mapping[str, object]) -> dict | None:
    """Parse the approval extension's JSON envelope out of a dialog request.

    ``pi``'s ``select`` request carries only ``title``/``options``/``timeout``
    (see ``docs/pi-rpc.md``), so ``nvsh/agent/pi_ext/approval.ts`` puts a
    machine-readable envelope --
    ``{"nvsh": "approval", "v": 1, "tool": ..., "command": ..., "reason": ...}``
    -- in ``title``. Returns the decoded envelope, or ``None`` when this is
    some other dialog (a plain human-facing ``title``/``message``).
    """
    for field in ("title", "message"):
        raw = obj.get(field)
        if not isinstance(raw, str) or not raw.startswith("{"):
            continue
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and decoded.get("nvsh") == "approval":
            return decoded
    return None


def _proposal_fields(obj: Mapping[str, object]) -> tuple[str, str]:
    """Return ``(command, rationale)`` for one ``extension_ui_request``.

    The command is *only* ever a field that holds a command: the approval
    envelope's ``command``, or a structured top-level ``command`` field.
    Human-facing prompt text (``title``/``message``) is never treated as a
    command -- doing so is deviation d8, where pi's rendered panel text came
    back as ``Proposal.command`` and would have been run verbatim on Enter.
    A dialog that carries no command yields ``""``: there is nothing to run.
    """
    envelope = _approval_envelope(obj)
    if envelope is not None:
        return str(envelope.get("command") or ""), str(envelope.get("reason") or "")
    raw_command = obj.get("command")
    rationale = str(obj.get("rationale") or obj.get("message") or obj.get("title") or "")
    if isinstance(raw_command, str):
        return raw_command, rationale
    return "", rationale


class PiAgent(NvshAgent):
    """Drives ``pi --mode rpc`` as one persistent subprocess per instance."""

    def __init__(
        self,
        *,
        pi_path: str = "pi",
        provider: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        extra_args: list[str] | None = None,
        approval: str = "nvsh",
        env: Mapping[str, str] | None = None,
        session_dir: str | Path | None = None,
        extension_path: str | Path | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._pi_path = pi_path
        #: Resolved lazily in build_argv() so constructing a PiAgent never
        #: runs platform detection (tests construct one per case).
        self._system_prompt = system_prompt
        # d21: pi's children (the approval extension) must find nvsh and the
        # same approval store the panel writes, so those variables are made
        # explicit rather than left to whatever the daemon inherited.
        self._env: dict[str, str] = child_env(os.environ if env is None else env)

        defaults = _default_pi_agent_config()
        self._provider = provider if provider is not None else defaults.get("provider")
        self._model = model if model is not None else defaults.get("model")
        #: Opaque effort string (decision c24: never validated by nvsh),
        #: passed verbatim to pi as ``--thinking <effort>`` when set.
        self._effort = effort if effort is not None else defaults.get("effort")
        #: Extra argv appended verbatim after every other flag (task t5's
        #: per-harness ``extra_args`` knob).
        self._extra_args = (
            list(extra_args) if extra_args is not None else list(defaults.get("extra_args") or [])
        )
        #: Who mediates approval of a proposed command. pi's approval
        #: extension always gates tool calls through ``nvsh approve check``
        #: (see ``docs/pi-rpc.md``), so this backend's own answer is
        #: ``"nvsh"`` unless a caller explicitly overrides it.
        self._approval = approval

        self._session_dir = Path(session_dir) if session_dir else default_session_dir(self._env)
        self._extension_path = (
            Path(extension_path) if extension_path else default_approval_extension_path()
        )

        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail: collections.deque[bytes] = collections.deque(maxlen=40)
        self._queue: queue.Queue = queue.Queue()
        #: Objects pulled off the queue while waiting for a command ack but
        #: belonging to the event stream. Read *before* the queue so the
        #: order pi sent them in survives the detour.
        self._held: collections.deque = collections.deque()
        self._cancelled = False
        self._closed = False
        self._write_lock = threading.Lock()
        self._ack_timeout = ack_timeout(self._env)
        #: Assistant text streamed since the last tool result. This is what
        #: the model said on its way to the tool call it is now asking to
        #: run, so it is that proposal's rationale when the approval
        #: envelope carries no ``reason`` of its own -- which, with pi
        #: 0.85.1's bash tool (``{command, timeout}``), is always (d17).
        self._said: list[str] = []
        #: True between a `prompt` ack and the turn's DONE/ERROR. Only then
        #: can a steer reach the running turn.
        self._streaming = False
        #: Set by force_stop(); tells the next run() that the process it is
        #: about to (re)spawn is a fresh session, not a continuation, so
        #: that run() opens with a STATUS 'new session' event (c33) instead
        #: of silently picking up where a killed process left off.
        self._new_session_pending = False

    # -- argv / lifecycle --------------------------------------------------

    def build_argv(self) -> list[str]:
        """Build the ``pi --mode rpc`` argv with launch-hygiene flags.

        Order matches the spec's acceptance criterion verbatim: mode, then
        the hygiene flags, then ``--session-dir``, then ``-e <extension>``,
        then ``--append-system-prompt`` (the system brief, once for the
        session -- deviation d19), then ``--provider``/``--model`` (never an
        API key -- pi reads its own ``models.json``), then ``--thinking
        <effort>`` when an effort is configured, then any ``extra_args``
        verbatim (task t5/t9's per-harness knobs).
        """
        if self._system_prompt is None:
            self._system_prompt = default_system_prompt()
        argv = [
            self._pi_path,
            "--mode",
            "rpc",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-approve",
            "--session-dir",
            str(self._session_dir),
            "-e",
            str(self._extension_path),
            "--append-system-prompt",
            self._system_prompt,
        ]
        if self._provider:
            argv += ["--provider", str(self._provider)]
        if self._model:
            argv += ["--model", str(self._model)]
        if self._effort:
            argv += ["--thinking", str(self._effort)]
        if self._extra_args:
            argv += list(self._extra_args)
        return argv

    def start(self) -> None:
        """Spawn the pi subprocess and handshake with it. Idempotent.

        The handshake is one ``get_state`` command whose ``response`` is
        waited for: it proves the process booted *and* that its rpc loop is
        reading stdin, so a pi that dies on launch (missing node, a broken
        install, a rejected model) is reported as an error naming its exit
        code and the tail of its stderr instead of leaving the caller
        waiting on a stream that will never start (deviation d14).
        """
        if self._proc is not None:
            return
        self._closed = False
        self._cancelled = False
        self._held.clear()
        self._stderr_tail.clear()
        self._session_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._session_dir, 0o700)

        # argv is a fixed list built by build_argv(), never shell=True (bandit B603,
        # allowed repo-wide).
        argv = self.build_argv()
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            # New session/process group (task t2's kill_tree contract): pi's
            # own tool grandchildren die with it under force_stop() instead
            # of being orphaned when only pi itself is signalled.
            start_new_session=True,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()
        try:
            self._command("get_state")
        except PiRpcError:
            self.close()
            raise

    def _reader_loop(self) -> None:
        """Read stdout line by line, splitting on LF only, then decode and parse.

        ``BufferedReader.readline()`` (unlike ``.read(n)``, which blocks
        until it can fill the requested size or hits EOF -- a deadlock
        against a process that has written less than that and is now
        waiting on the next command) returns as soon as it has one complete
        ``b"\\n"``-terminated line, or the stream ends. It splits on the
        literal byte ``0x0A`` only, so a literal U+2028/U+2029 inside a JSON
        string (legal JSON, illegal input to Node's ``readline``) can never
        be mistaken for a record boundary -- it only ever exists after
        decode, entirely inside one already LF-delimited line.
        """
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        try:
            for raw_line in iter(stdout.readline, b""):
                line = raw_line
                if line.endswith(b"\n"):
                    line = line[:-1]
                if line.endswith(b"\r"):
                    line = line[:-1]
                self._queue_line(line)
        finally:
            self._queue.put(None)  # EOF sentinel

    def _queue_line(self, line: bytes) -> None:
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = {"type": "_malformed", "raw": text}
        self._queue.put(obj)

    def _stderr_loop(self) -> None:
        """Keep the tail of pi's stderr so a failure can quote it.

        Drained in its own thread for two reasons: a full stderr pipe would
        block pi mid-turn, and a process that dies on launch says why only
        here. Only the last few lines are kept, and they are redacted before
        anyone reads them (:meth:`_stderr_text`).
        """
        proc = self._proc
        if proc is None or proc.stderr is None:  # pragma: no cover - always piped
            return
        for raw_line in iter(proc.stderr.readline, b""):
            self._stderr_tail.append(raw_line)

    def _stderr_text(self) -> str:
        """The redacted tail of pi's stderr, as a single short string."""
        blob = b"".join(self._stderr_tail)[-_STDERR_TAIL_BYTES:]
        text = redact(blob).decode("utf-8", errors="replace").strip()
        return " | ".join(line.strip() for line in text.splitlines() if line.strip())

    def _exit_detail(self) -> str:
        """How the pi process is doing, for the tail of an error message."""
        proc = self._proc
        if proc is None:
            return " (no pi process)"
        code = proc.poll()
        state = "still running" if code is None else f"exited with code {code}"
        stderr = self._stderr_text()
        if stderr:
            return f" (pi {state}; stderr tail: {stderr})"
        return f" (pi {state}; no stderr)"

    def _send(self, obj: dict) -> bool:
        """Write one command line. ``False`` when pi's stdin is gone."""
        if self._proc is None or self._proc.stdin is None:
            return False
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (ValueError, OSError):  # OSError covers BrokenPipeError
                return False
        return True

    # -- command/ack round trips -------------------------------------------

    def _next_object(self, timeout: float) -> Any:
        """The next thing pi said: held objects first, then the live queue."""
        if self._held:
            return self._held.popleft()
        return self._queue.get(timeout=timeout)

    def _command(self, command: str, timeout: float | None = None, **fields: object) -> dict:
        """Send one rpc command and wait for *its* ``response`` acknowledgement.

        **Never pipeline commands.** pi 0.85.1 drops both commands --
        silently, forever, with no response and no events -- when a second
        command line arrives before the first has been acknowledged. That is
        deviation d14: the daemon wrote ``new_session`` and then, a
        millisecond later, ``prompt``; pi went mute and the operator's panel
        waited out the client's 120 s stream timeout. Waiting for the ack
        costs one round trip (~0.2 s measured against the real pi) and makes
        the sequence deterministic.

        Raises :class:`PiRpcError` when the ack does not come, when pi
        rejects the command, or when pi has gone away.
        """
        if not self._send({"type": command, **fields}):
            raise PiRpcError(f"could not send {command} to pi{self._exit_detail()}")
        return self._await_response(command, self._ack_timeout if timeout is None else timeout)

    def _ack_or_none(self, obj: dict, command: str) -> dict | None:
        """*obj* if it is the ``response`` ack for *command*, else ``None``.

        Raises :class:`PiRpcError` when it is the ack and pi rejected the
        command -- a rejection is never something to keep waiting through.
        """
        if obj.get("type") != "response" or obj.get("command") != command:
            return None
        if not obj.get("success", True):
            raise PiRpcError(f"pi rejected {command}: {obj.get('error', 'no reason given')}")
        return obj

    def _await_response(self, command: str, timeout: float) -> dict:
        carried: list = []
        deadline = time.monotonic() + timeout
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise PiRpcError(
                        f"pi did not acknowledge {command} within {timeout:g}s"
                        f"{self._exit_detail()}"
                    )
                try:
                    obj = self._next_object(min(_POLL_INTERVAL_SECONDS, left))
                except queue.Empty:
                    if self._proc is not None and self._proc.poll() is not None:
                        raise PiRpcError(
                            f"pi went away before acknowledging {command}{self._exit_detail()}"
                        ) from None
                    continue
                if obj is None:  # EOF sentinel: keep it for run() to report too
                    carried.append(None)
                    raise PiRpcError(
                        f"pi closed its output before acknowledging {command}"
                        f"{self._exit_detail()}"
                    )
                ack = self._ack_or_none(obj, command)
                if ack is not None:
                    return ack
                carried.append(obj)
        finally:
            self._held.extendleft(reversed(carried))

    # -- running -------------------------------------------------------

    def _prompt_for(self, request: AgentRequest, context: AgentContext) -> str:
        """One turn's prompt: the facts block only.

        The system brief is *not* repeated here -- pi already holds it for
        the whole session through ``--append-system-prompt`` in
        :meth:`build_argv`.
        """
        return build_prompt(request, context)

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self.start()
        self._cancelled = False
        self._said.clear()
        prompt_text = self._prompt_for(request, context)
        if self._new_session_pending:
            self._new_session_pending = False
            yield AgentEvent(kind=EventKind.STATUS, text="new session")
        # Acknowledged, like every other command: a prompt pi never answers
        # for is an error the operator gets to read, not silence (d14).
        self._command("prompt", message=prompt_text)
        self._streaming = True
        try:
            yield from self._events()
        finally:
            self._streaming = False
            self._said.clear()

    def _events(self) -> Iterator[AgentEvent]:
        """The turn's event loop, from the acked prompt to DONE/ERROR."""
        while not self._cancelled:
            if self._proc is not None and self._proc.poll() is not None:
                yield from self._post_exit_events()
                return

            try:
                obj = self._next_object(_POLL_INTERVAL_SECONDS)
            except queue.Empty:
                continue

            event = self._live_event(obj)
            if event is None:
                continue
            yield event
            if self._cancelled or event.kind in (EventKind.DONE, EventKind.ERROR):
                return

    def _post_exit_events(self) -> Iterator[AgentEvent]:
        """What pi still owes once its process has exited.

        Drains anything already queued first, so a fast final burst of
        events (including a real error event) is not lost, then reports the
        exit itself.
        """
        while not self._cancelled:
            event = self._drain_one_nowait()
            if event is None:
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    error=f"pi process exited{self._exit_detail()}",
                )
                return
            yield event
            if event.kind in (EventKind.DONE, EventKind.ERROR):
                return

    def _live_event(self, obj: Any) -> AgentEvent | None:
        """One queued object as an event: the EOF sentinel is an error."""
        if obj is None:
            return AgentEvent(kind=EventKind.ERROR, error="pi process stdout closed unexpectedly")
        return self._map_event(obj)

    def _drain_one_nowait(self) -> AgentEvent | None:
        if self._held:
            obj = self._held.popleft()
        else:
            try:
                obj = self._queue.get_nowait()
            except queue.Empty:
                return None
        if obj is None:
            return None
        return self._map_event(obj)

    def _response_event(self, obj: dict) -> AgentEvent | None:
        """A command ack is not an event -- unless it rejected the prompt."""
        if obj.get("command") == "prompt" and not obj.get("success", True):
            return AgentEvent(kind=EventKind.ERROR, error=str(obj.get("error", "prompt rejected")))
        return None

    def _message_update_event(self, obj: dict) -> AgentEvent | None:
        """Assistant text deltas, also kept as the next proposal's rationale.

        ``thinking_delta`` mirrors ``text_delta`` (task t9) but maps to
        :attr:`EventKind.THINKING` instead and is never folded into
        :attr:`_said` -- the next proposal's rationale is what the model
        *said*, not what it thought on the way there.
        """
        ame = obj.get("assistantMessageEvent") or {}
        ame_type = ame.get("type")
        if ame_type == "thinking_delta":
            return AgentEvent(kind=EventKind.THINKING, text=str(ame.get("delta", "")))
        if ame_type != "text_delta":
            return None
        delta = str(ame.get("delta", ""))
        self._said.append(delta)
        return AgentEvent(kind=EventKind.TEXT_DELTA, text=delta)

    def _ui_request_event(self, obj: dict) -> AgentEvent:
        """One ``extension_ui_request`` dialog as a PROPOSAL the panel can show."""
        command, rationale = _proposal_fields(obj)
        proposal = Proposal(
            command=command,
            rationale=rationale or self._rationale_text(),
            kind=ProposalKind.FIX,
        )
        return AgentEvent(
            kind=EventKind.PROPOSAL,
            proposal=proposal,
            args={"request_id": obj.get("id"), "method": obj.get("method")},
        )

    def _map_event(self, obj: dict) -> AgentEvent | None:
        msg_type = obj.get("type")

        if msg_type == "response":
            return self._response_event(obj)

        if msg_type == "message_update":
            return self._message_update_event(obj)

        if msg_type == "tool_execution_start":
            return AgentEvent(
                kind=EventKind.TOOL_CALL,
                tool=str(obj.get("toolName", "")),
                args=dict(obj.get("args") or {}),
            )

        if msg_type == "tool_execution_end":
            # A finished tool resets the rationale window: what the model
            # says next is about the *next* thing it wants to do (d17).
            self._said.clear()
            return AgentEvent(
                kind=EventKind.TOOL_RESULT,
                tool=str(obj.get("toolName", "")),
                result=obj.get("result"),
            )

        if msg_type == "extension_ui_request":
            return self._ui_request_event(obj)

        if msg_type == "agent_end":
            return AgentEvent(kind=EventKind.DONE)

        if msg_type in _QUIET_EVENT_TYPES:
            return None  # lifecycle/progress bookkeeping is not panel material (d11)

        if msg_type == "error":
            return AgentEvent(kind=EventKind.ERROR, error=str(obj.get("error", "")))

        if msg_type == "_malformed":
            return AgentEvent(kind=EventKind.ERROR, error=f"malformed line: {obj.get('raw', '')}")

        # Unknown/uninteresting event types (queue_update, turn_start, ...)
        # surface as STATUS so nothing is silently dropped.
        return AgentEvent(kind=EventKind.STATUS, text=str(msg_type or ""))

    def _rationale_text(self) -> str:
        """The assistant text since the last tool result, trimmed for a panel.

        This is deviation d17's answer to ``why: (no rationale given)``:
        pi's bash tool schema has no ``reason`` field, so the approval
        envelope's ``reason`` is always empty and every proposal looked
        unmotivated -- while the model had just said, in the text streaming
        above the box, exactly why.
        """
        text = " ".join("".join(self._said).split())
        if len(text) <= _RATIONALE_LIMIT:
            return text
        return text[: _RATIONALE_LIMIT - 1] + "\u2026"

    def steer(self, text: str) -> bool:
        """Inject ``text`` into the turn that is running now (deviation d16).

        The wire shape is pi's own (``docs/pi-rpc.md``):
        ``{"type": "prompt", "message": <text>, "streamingBehavior": "steer"}``
        -- queued while the agent is running and delivered after the
        current assistant turn finishes its tool calls, before the next LLM
        call. Sending a bare ``prompt`` mid-stream is an error on pi's side,
        so the field is never optional here.

        This is the second write (after ``abort``) that is deliberately not
        ack-waited, and for the same reason: it is only ever sent *mid-turn*,
        when no command is outstanding, so "never pipeline" (d14) still
        holds -- nothing is written ahead of an unacknowledged command. The
        ack cannot be waited for here anyway: during a turn the event
        consumer owns the stdout queue, and two readers would race for the
        `response` line. A rejection is not lost: ``_map_event`` turns a
        ``response`` for ``prompt`` with ``success: false`` into an ERROR
        event the operator reads.
        """
        if not text or not self._streaming:
            return False
        if self._proc is None or self._proc.poll() is not None:
            return False
        return self._send({"type": "prompt", "message": text, "streamingBehavior": "steer"})

    def respond_ui(self, request_id: str, **fields: object) -> None:
        """Answer a pending ``extension_ui_request`` dialog.

        Not part of the :class:`NvshAgent` contract -- the interactive
        loop/panel calls this directly once the operator has answered a
        ``PROPOSAL`` event that came from an ``extension_ui_request``. Pass
        the response fields the method expects, e.g.
        ``respond_ui(request_id, confirmed=True)`` or
        ``respond_ui(request_id, value="Allow")`` or
        ``respond_ui(request_id, cancelled=True)``.
        """
        self._send({"type": "extension_ui_response", "id": request_id, **fields})

    def new_session(self) -> str:
        """Start a fresh pi session and return the file pi will store it in.

        Acknowledged before returning, so the caller's next command (the
        daemon's very next move is the prompt) cannot race it -- pipelining
        these two is what made every daemon-routed turn hang (d14).

        pi names its own session file inside ``--session-dir``; there is no
        rpc command to choose the name. So the path is read back from
        ``get_state`` and returned for the caller to remember and hand to
        :meth:`switch_session` later. Returns ``""`` when pi reports no
        session file (``--no-session``), and the caller keeps its own key.
        """
        self._command("new_session")
        return self.session_file()

    def session_file(self) -> str:
        """The session file pi is currently writing, or ``""``."""
        state = self._command("get_state").get("data") or {}
        path = state.get("sessionFile")
        return str(path) if isinstance(path, str) else ""

    def switch_session(self, path: str) -> None:
        """Resume a stored session. Acknowledged before returning (see above)."""
        self._command("switch_session", sessionPath=str(path))

    # -- cancel / close ---------------------------------------------------

    def cancel(self) -> None:
        """Mark the in-flight run() cancelled and ask pi to abort.

        Returns immediately: sending ``{"type": "abort"}`` is a single
        non-blocking write, well under the 1s budget the spec requires.
        """
        self._cancelled = True
        if self._proc is not None and self._proc.poll() is None:
            self._send({"type": "abort"})

    def force_stop(self) -> None:
        """End the turn for certain, even when pi ignores ``abort``.

        ``abort`` is still sent first, as a courtesy to a pi that is merely
        slow rather than wedged, but nothing here waits on it: the whole
        process group is then walked down with
        :func:`~nvsh.agent._subprocess.kill_tree` (SIGTERM, wait, SIGKILL),
        which reaches any tool grandchild pi's bash tool started too, not
        just pi itself -- the fake harness's ``NVSH_FAKE_IGNORE_CANCEL``
        mode exists to prove exactly this path runs when the protocol-level
        ``abort`` gets no reply at all.

        Internal state is reset so the process is never reused half-dead:
        the next :meth:`run` sees no live process, calls :meth:`start` to
        spawn a fresh one, and -- because :attr:`_new_session_pending` is
        left set here -- that run opens with a STATUS event naming it a
        'new session' (assumption c33), so the operator is told the
        previous turn's context is gone rather than left to assume it
        continued.
        """
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._send({"type": "abort"})
        kill_tree(proc, grace=_CLOSE_WAIT_SECONDS)
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=_CLOSE_WAIT_SECONDS)
        self._proc = None
        self._reader_thread = None
        self._stderr_thread = None
        self._closed = True
        self._queue = queue.Queue()
        self._held.clear()
        self._new_session_pending = True

    def close(self) -> None:
        """Idempotent teardown: close stdin, wait, then escalate to kill.

        The escalation itself is :func:`~nvsh.agent._subprocess.escalate_close`,
        shared with every other adapter (deviation d5).
        """
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        escalate_close(proc, wait=_CLOSE_WAIT_SECONDS)
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=_CLOSE_WAIT_SECONDS)
        self._proc = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=True,
            cancellation=True,
            persistent_session=True,
            local_model=True,
            thinking=True,
            effort=True,
            path="rpc",
            approval=self._approval,
            unmediated_file_access=False,
            steer=True,
        )
