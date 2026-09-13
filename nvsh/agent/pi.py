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

import importlib.resources
import json
import os
import queue
import subprocess  # external `pi` CLI (bandit B404, allowed repo-wide)
import threading
from pathlib import Path
from typing import Iterator, Mapping

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

#: How long close() waits for a clean exit after closing stdin, and then
#: after terminate(), before escalating.
_CLOSE_WAIT_SECONDS = 2.0


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


class PiAgent(NvshAgent):
    """Drives ``pi --mode rpc`` as one persistent subprocess per instance."""

    def __init__(
        self,
        *,
        pi_path: str = "pi",
        provider: str | None = None,
        model: str | None = None,
        env: Mapping[str, str] | None = None,
        session_dir: str | Path | None = None,
        extension_path: str | Path | None = None,
    ) -> None:
        self._pi_path = pi_path
        self._env: dict[str, str] = dict(os.environ if env is None else env)

        defaults = _default_pi_agent_config()
        self._provider = provider if provider is not None else defaults.get("provider")
        self._model = model if model is not None else defaults.get("model")

        self._session_dir = Path(session_dir) if session_dir else default_session_dir(self._env)
        self._extension_path = (
            Path(extension_path) if extension_path else default_approval_extension_path()
        )

        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()
        self._cancelled = False
        self._closed = False
        self._write_lock = threading.Lock()

    # -- argv / lifecycle --------------------------------------------------

    def build_argv(self) -> list[str]:
        """Build the ``pi --mode rpc`` argv with launch-hygiene flags.

        Order matches the spec's acceptance criterion verbatim: mode, then
        the hygiene flags, then ``--session-dir``, then ``-e <extension>``,
        then ``--provider``/``--model`` (never an API key -- pi reads its
        own ``models.json``).
        """
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
        ]
        if self._provider:
            argv += ["--provider", str(self._provider)]
        if self._model:
            argv += ["--model", str(self._model)]
        return argv

    def start(self) -> None:
        """Spawn the pi subprocess. Idempotent: a second call is a no-op."""
        if self._proc is not None:
            return
        self._closed = False
        self._cancelled = False
        self._session_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._session_dir, 0o700)

        # argv is a fixed list built by build_argv(), never shell=True (bandit B603,
        # allowed repo-wide).
        argv = self.build_argv()
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self._env,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

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

    def _send(self, obj: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

    # -- running -------------------------------------------------------

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self.start()
        self._cancelled = False
        prompt_text = build_prompt(request, context)
        self._send({"type": "prompt", "message": prompt_text})

        while True:
            if self._cancelled:
                return
            if self._proc is not None and self._proc.poll() is not None:
                # Process has exited. Drain anything already queued first so
                # a fast final burst of events (including a real error
                # event) is not lost, then report the exit.
                event = self._drain_one_nowait()
                if event is not None:
                    yield event
                    if self._cancelled:
                        return
                    if event.kind in (EventKind.DONE, EventKind.ERROR):
                        return
                    continue
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    error=f"pi process exited (code {self._proc.returncode})",
                )
                return

            try:
                obj = self._queue.get(timeout=_POLL_INTERVAL_SECONDS)
            except queue.Empty:
                continue

            if obj is None:
                yield AgentEvent(
                    kind=EventKind.ERROR, error="pi process stdout closed unexpectedly"
                )
                return

            event = self._map_event(obj)
            if event is None:
                continue
            yield event
            if self._cancelled:
                return
            if event.kind in (EventKind.DONE, EventKind.ERROR):
                return

    def _drain_one_nowait(self) -> AgentEvent | None:
        try:
            obj = self._queue.get_nowait()
        except queue.Empty:
            return None
        if obj is None:
            return None
        return self._map_event(obj)

    def _map_event(self, obj: dict) -> AgentEvent | None:
        msg_type = obj.get("type")

        if msg_type == "response":
            if obj.get("command") == "prompt" and not obj.get("success", True):
                return AgentEvent(
                    kind=EventKind.ERROR, error=str(obj.get("error", "prompt rejected"))
                )
            return None  # command acks are not events

        if msg_type == "message_update":
            ame = obj.get("assistantMessageEvent") or {}
            if ame.get("type") == "text_delta":
                return AgentEvent(kind=EventKind.TEXT_DELTA, text=ame.get("delta", ""))
            return None

        if msg_type == "tool_execution_start":
            return AgentEvent(
                kind=EventKind.TOOL_CALL,
                tool=str(obj.get("toolName", "")),
                args=dict(obj.get("args") or {}),
            )

        if msg_type == "tool_execution_end":
            return AgentEvent(
                kind=EventKind.TOOL_RESULT,
                tool=str(obj.get("toolName", "")),
                result=obj.get("result"),
            )

        if msg_type == "extension_ui_request":
            command = obj.get("command") or obj.get("message") or obj.get("title") or ""
            rationale = str(obj.get("message") or obj.get("title") or "")
            proposal = Proposal(command=str(command), rationale=rationale, kind=ProposalKind.FIX)
            return AgentEvent(
                kind=EventKind.PROPOSAL,
                proposal=proposal,
                args={"request_id": obj.get("id"), "method": obj.get("method")},
            )

        if msg_type == "agent_end":
            return AgentEvent(kind=EventKind.DONE)

        if msg_type == "error":
            return AgentEvent(kind=EventKind.ERROR, error=str(obj.get("error", "")))

        if msg_type == "_malformed":
            return AgentEvent(kind=EventKind.ERROR, error=f"malformed line: {obj.get('raw', '')}")

        # Unknown/uninteresting event types (queue_update, turn_start, ...)
        # surface as STATUS so nothing is silently dropped.
        return AgentEvent(kind=EventKind.STATUS, text=str(msg_type or ""))

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

    def new_session(self) -> None:
        """Pass-through for the session daemon (t12): start a fresh pi session."""
        self._send({"type": "new_session"})

    def switch_session(self, path: str) -> None:
        """Pass-through for the session daemon (t12): resume a stored session."""
        self._send({"type": "switch_session", "sessionPath": str(path)})

    # -- cancel / close ---------------------------------------------------

    def cancel(self) -> None:
        """Mark the in-flight run() cancelled and ask pi to abort.

        Returns immediately: sending ``{"type": "abort"}`` is a single
        non-blocking write, well under the 1s budget the spec requires.
        """
        self._cancelled = True
        if self._proc is not None and self._proc.poll() is None:
            self._send({"type": "abort"})

    def close(self) -> None:
        """Idempotent teardown: close stdin, wait, then escalate to kill."""
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass

        if proc.poll() is None:
            try:
                proc.wait(timeout=_CLOSE_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=_CLOSE_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=_CLOSE_WAIT_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=_CLOSE_WAIT_SECONDS)
        self._proc = None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=True,
            cancellation=True,
            persistent_session=True,
            local_model=True,
        )
