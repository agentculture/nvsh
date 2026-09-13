"""The nvsh session daemon: one per user, owning the warm agent processes.

Started lazily by the hook client on the first qualifying failure -- never
at shell start -- and listening on a per-user unix socket under
``$XDG_RUNTIME_DIR/nvsh/daemon.sock`` (directory ``0700``, socket ``0600``).
Each shell session (keyed by the hook's shell PID) owns its own
:class:`Conversation`; ``sessions.max`` (config) bounds how many agent
*processes* may exist at once, defaulting to one. When a request arrives
for a shell whose conversation is not the one the agent currently holds,
the daemon puts the active conversation to sleep and activates the caller's
(``switch_session`` / ``new_session`` on the adapter), so contexts from
different terminals never mix.

Wire protocol (documented in ``docs/daemon.md``): the client sends **one**
JSON line and then reads JSON lines, one per :class:`AgentEvent`, until a
``done`` or ``error`` line; the connection closes after.

Stdlib only, no third-party imports: a login shell must not depend on a
wheel that may be broken. The daemon never writes to a terminal -- it logs
to ``$XDG_STATE_HOME/nvsh/daemon.log`` (0600).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import socket
import socketserver
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional

from .agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    EventKind,
    NvshAgent,
    RequestKind,
    event_to_dict,
)
from .config import Config

#: Seconds with no request after which the daemon shuts itself down.
DEFAULT_IDLE_TIMEOUT = 900.0

#: How often the idle watchdog wakes up.
_WATCHDOG_INTERVAL = 0.1

#: Control kinds that carry no agent request.
_CONTROL_KINDS = frozenset(
    {"register", "unregister", "cancel", "ui_response", "status", "stop", "ping", "undo"}
)

_LOGGER_NAME = "nvsh.daemon"


class DaemonAlreadyRunning(OSError):
    """Raised when another daemon already holds this user's single-instance lock.

    An :class:`OSError` so every existing caller (``main``, ``nvsh daemon
    run``) already treats it as "could not serve" and exits 2.
    """


# --- paths -----------------------------------------------------------------


def _resolve_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def runtime_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$XDG_RUNTIME_DIR/nvsh``, or a per-uid temp directory when unset.

    Never a hard-coded path, and never a directory shared between users:
    without ``XDG_RUNTIME_DIR`` we fall back to ``<tmp>/nvsh-<uid>``.
    """
    resolved = _resolve_env(env)
    xdg = resolved.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "nvsh"
    return Path(tempfile.gettempdir()) / f"nvsh-{os.getuid()}"


def socket_path(env: Mapping[str, str] | None = None) -> Path:
    """Where the daemon listens. Pure: creates nothing, touches nothing."""
    return runtime_dir(env) / "daemon.sock"


def lock_path(env: Mapping[str, str] | None = None) -> Path:
    """The single-instance lock next to the socket. Pure: creates nothing.

    One ``flock`` on this file, held for the daemon's whole life, is what
    makes "one daemon per user" true: whoever holds it owns the socket, and
    a daemon that cannot take it exits at once instead of unlinking a live
    daemon's socket and binding a rival one (deviation d7).
    """
    return runtime_dir(env) / "daemon.lock"


def lock_held(env: Mapping[str, str] | None = None) -> bool:
    """Is some process holding the single-instance lock right now?

    Answered by trying to take the lock in a private file descriptor and
    giving it straight back: ``flock`` locks belong to an open file
    description, so probing never disturbs the real holder.
    """
    path = lock_path(env)
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def log_path(env: Mapping[str, str] | None = None) -> Path:
    """``$XDG_STATE_HOME/nvsh/daemon.log`` (``~/.local/state`` fallback)."""
    resolved = _resolve_env(env)
    xdg_state = resolved.get("XDG_STATE_HOME")
    if xdg_state:
        base = Path(xdg_state)
    else:
        home = resolved.get("HOME") or os.path.expanduser("~")
        base = Path(home) / ".local" / "state"
    return base / "nvsh" / "daemon.log"


def is_running(env: Mapping[str, str] | None = None, timeout: float = 1.0) -> bool:
    """Is a daemon accepting connections on this user's socket right now?"""
    path = socket_path(env)
    if not path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(path))
        return True
    except OSError:
        return False


def spawn(env: Mapping[str, str] | None = None, *, idle_timeout: float | None = None) -> int:
    """Start a detached daemon process and return its pid.

    Uses ``sys.executable -m nvsh.daemon --foreground`` so it works from a
    checkout and from a wheel alike, with its own session
    (``start_new_session``) so it outlives the shell that spawned it.
    """
    resolved = dict(_resolve_env(env))
    argv = [sys.executable, "-m", "nvsh.daemon", "--foreground"]
    if idle_timeout is not None:
        argv += ["--idle-timeout", str(idle_timeout)]
    proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        argv,
        env=resolved,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


# --- request decoding ------------------------------------------------------


def request_from_dict(data: Mapping[str, object] | None) -> AgentRequest:
    """Decode an :class:`AgentRequest` from the wire, tolerating junk."""
    data = data or {}
    try:
        kind = RequestKind(str(data.get("kind", RequestKind.EXPLICIT.value)))
    except ValueError:
        kind = RequestKind.EXPLICIT
    raw_exit = data.get("exit_code")
    exit_code = int(raw_exit) if isinstance(raw_exit, (int, float)) else None
    return AgentRequest(
        kind=kind,
        prompt=str(data.get("prompt", "")),
        command=str(data.get("command", "")),
        exit_code=exit_code,
        failure_id=str(data.get("failure_id", "")),
    )


def context_from_dict(data: Mapping[str, object] | None) -> AgentContext:
    """Decode an :class:`AgentContext` from the wire, tolerating junk."""
    data = data or {}
    report = data.get("redaction_report") or ()
    return AgentContext(
        platform=str(data.get("platform", "")),
        output=str(data.get("output", "")),
        cwd=str(data.get("cwd", "")),
        shell_pid=int(data.get("shell_pid", 0) or 0),
        redaction_report=tuple(str(item) for item in report),
    )


# --- conversations ---------------------------------------------------------


@dataclass
class Conversation:
    """One shell's agent conversation, awake or asleep.

    ``transcript`` is nvsh's own record of turns (request prompt + the
    accumulated response text), kept purely so :meth:`undo` has something to
    drop; it is never replayed to the backend. ``pending_proposal`` mirrors
    the most recent ``proposal`` event this conversation streamed, cleared
    once a fresh turn starts.
    """

    shell: str
    session_path: str | None = None
    started: bool = False
    sleeping: bool = False
    requests: int = 0
    transcript: list[dict] = field(default_factory=list)
    pending_proposal: dict | None = None

    def undo(self) -> bool:
        """Drop the last request/answer pair and any pending proposal.

        This only changes nvsh's own view of the conversation -- a backend
        like pi keeps its own on-disk session file untouched (there is no
        RPC to selectively erase a turn there), so the operator's next
        request still reaches a warm backend, just without nvsh re-showing
        the undone turn or its proposal. Never runs anything on the machine.
        Returns ``True`` if there was anything to drop.
        """
        had = bool(self.transcript) or self.pending_proposal is not None
        if self.transcript:
            self.transcript.pop()
        self.pending_proposal = None
        return had


@dataclass
class _Slot:
    """One live agent process and the conversation it currently holds."""

    agent: NvshAgent
    shell: str | None = None
    last_used: float = field(default_factory=time.monotonic)


AgentFactory = Callable[[], NvshAgent]
WhichFn = Callable[[str], Optional[str]]


# --- the daemon ------------------------------------------------------------


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, path: str, handler, owner: "Daemon") -> None:
        self.owner = owner
        super().__init__(path, handler)


class _Handler(socketserver.StreamRequestHandler):
    """One connection: read a single JSON line, stream JSON-line events back."""

    def handle(self) -> None:  # pragma: no cover - exercised through sockets
        owner: Daemon = self.server.owner  # type: ignore[attr-defined]
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            message = json.loads(raw.decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("message must be a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            self._write(AgentEvent(kind=EventKind.ERROR, error=f"bad request: {exc}"))
            return
        for event in owner.handle_message(message):
            if not self._write(event):
                owner.cancel_shell(str(message.get("shell", "")))
                return

    def _write(self, event: AgentEvent) -> bool:
        try:
            self.wfile.write((json.dumps(event_to_dict(event)) + "\n").encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ValueError):
            return False


class Daemon:
    """Per-user session daemon: owns the agent processes and the conversations.

    ``agent_factory`` builds one adapter; when it is ``None`` the daemon
    picks a backend itself through :func:`nvsh.agent.registry.choose`, and
    reports the pick (e.g. ``"pi unavailable: pi not on PATH..."``) as a
    ``status`` event ahead of the first run's output.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        env: Mapping[str, str] | None = None,
        agent_factory: AgentFactory | None = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        which: WhichFn | None = None,
    ) -> None:
        self.config = config if config is not None else Config()
        self.env: Mapping[str, str] = dict(_resolve_env(env))
        self.idle_timeout = float(idle_timeout)
        self._agent_factory = agent_factory
        self._which = which

        self._lock = threading.RLock()
        self._run_lock = threading.RLock()
        self._slots: list[_Slot] = []
        self._conversations: dict[str, Conversation] = {}
        self._shells: set[str] = set()
        self._had_shells = False
        self._backend_name = "" if agent_factory is None else "injected"
        self._backend_reason = ""
        self._fallback_notice = ""
        self._notified: set[str] = set()

        self._server: _Server | None = None
        self._lock_fd: int | None = None
        self._owns_socket = False
        self._stopping = False
        self._stopped = threading.Event()
        self._last_activity = time.monotonic()
        self._log = self._make_logger()

    # -- logging -----------------------------------------------------------

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{_LOGGER_NAME}.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        path = log_path(self.env)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(path.parent, 0o700)
            if not path.exists():
                path.touch()
            os.chmod(path, 0o600)
            handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
        except OSError:  # pragma: no cover - unwritable state dir
            handler = logging.NullHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    # -- socket ------------------------------------------------------------

    @property
    def socket_path(self) -> Path:
        return socket_path(self.env)

    def _reap_stale_socket(self, path: Path) -> None:
        """Remove a socket left behind by a crashed daemon (connect fails)."""
        if not path.exists():
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                probe.connect(str(path))
        except OSError:
            self._log.info("reaping stale socket %s", path)
            path.unlink(missing_ok=True)
            return
        raise OSError(f"another nvsh daemon is already listening on {path}")

    def _acquire_lock(self) -> None:
        """Take the single-instance lock, or raise :class:`DaemonAlreadyRunning`.

        Non-blocking on purpose: a daemon that loses the race must exit
        immediately rather than linger as an orphan process holding no
        socket. The lock file is never unlinked -- a fresh inode under a
        waiting process would let two daemons hold "the" lock.
        """
        path = lock_path(self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise DaemonAlreadyRunning(f"another nvsh daemon already holds {path}: {exc}") from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        self._lock_fd = fd

    def _release_lock(self) -> None:
        fd, self._lock_fd = self._lock_fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - the fd is about to go anyway
            pass
        os.close(fd)

    def _bind(self) -> _Server:
        path = self.socket_path
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        self._acquire_lock()
        try:
            self._reap_stale_socket(path)
            server = _Server(str(path), _Handler, self)
            os.chmod(path, 0o600)
        except OSError:
            self._release_lock()
            raise
        self._owns_socket = True
        return server

    # -- serving -----------------------------------------------------------

    def serve_forever(self) -> None:
        """Run the daemon in the foreground until it is stopped."""
        self._server = self._bind()
        self._last_activity = time.monotonic()
        watchdog = threading.Thread(target=self._watchdog, daemon=True)
        watchdog.start()
        self._log.info(
            "daemon listening on %s (idle timeout %.1fs)", self.socket_path, self.idle_timeout
        )
        try:
            self._server.serve_forever(poll_interval=0.05)
        finally:
            self._teardown()

    def start_background(self) -> threading.Thread:
        """Serve in a daemon thread (tests, and ``nvsh daemon run`` in-process)."""
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not is_running(self.env):
            time.sleep(0.01)
        return thread

    def _watchdog(self) -> None:
        while not self._stopping:
            time.sleep(_WATCHDOG_INTERVAL)
            if self._stopping:
                return
            if time.monotonic() - self._last_activity > self.idle_timeout:
                self._log.info("idle for %.1fs; shutting down", self.idle_timeout)
                self.shutdown()
                return

    def shutdown(self) -> None:
        """Close every agent, stop serving and unlink the socket. Idempotent."""
        with self._lock:
            if self._stopping:
                return
            self._stopping = True
        server = self._server
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()
        else:  # never served: still release the agents
            self._teardown()

    def _teardown(self) -> None:
        self._stopping = True
        with self._lock:
            slots, self._slots = self._slots, []
        for slot in slots:
            try:
                slot.agent.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                self._log.warning("closing agent failed: %s", exc)
        if self._server is not None:
            try:
                self._server.server_close()
            except OSError:  # pragma: no cover
                pass
        if self._owns_socket:
            # Only the daemon that bound this socket may remove it; a loser
            # that unlinks it orphans the live daemon (deviation d7).
            self.socket_path.unlink(missing_ok=True)
            self._owns_socket = False
        self._release_lock()
        self._log.info("daemon stopped")
        self._stopped.set()

    # -- agents ------------------------------------------------------------

    def _session_dir(self) -> Path:
        from .agent.pi import default_session_dir

        return default_session_dir(self.env)

    def _session_path(self, shell: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in shell)
        return self._session_dir() / f"shell-{safe}.jsonl"

    def _build_agent(self) -> NvshAgent:
        if self._agent_factory is not None:
            return self._agent_factory()
        from .agent import registry

        which = self._which if self._which is not None else None
        if which is None:
            name, reason = registry.choose(self.config)
        else:
            name, reason = registry.choose(self.config, which=which)
        configured = self.config.agent_provider
        self._backend_name = name
        self._backend_reason = reason
        if name != configured:
            self._fallback_notice = f"{configured} unavailable: {reason}"
            self._log.info("%s", self._fallback_notice)
        return registry.ADAPTERS[name].factory(self.config)

    def _acquire(self, shell: str) -> tuple[_Slot, Conversation]:
        """Return the slot serving *shell*, switching sessions if it must."""
        with self._lock:
            conversation = self._conversations.setdefault(shell, Conversation(shell))
            for slot in self._slots:
                if slot.shell == shell:
                    slot.last_used = time.monotonic()
                    conversation.sleeping = False
                    return slot, conversation

            max_agents = max(1, int(self.config.sessions_max or 1))
            if len(self._slots) < max_agents:
                agent = self._build_agent()
                agent.start()
                slot = _Slot(agent=agent)
                self._slots.append(slot)
            else:
                slot = min(self._slots, key=lambda item: item.last_used)
                if slot.shell is not None and slot.shell != shell:
                    sleeping = self._conversations.get(slot.shell)
                    if sleeping is not None:
                        sleeping.sleeping = True
                        self._log.info("putting shell %s to sleep", slot.shell)

            self._activate(slot, conversation)
            slot.shell = shell
            slot.last_used = time.monotonic()
            conversation.sleeping = False
            return slot, conversation

    def _activate(self, slot: _Slot, conversation: Conversation) -> None:
        new_session = getattr(slot.agent, "new_session", None)
        switch_session = getattr(slot.agent, "switch_session", None)
        if not callable(new_session) or not callable(switch_session):
            return  # a stateless adapter has nothing to swap
        if conversation.started and conversation.session_path:
            switch_session(conversation.session_path)
            self._log.info("resumed shell %s", conversation.shell)
        else:
            conversation.session_path = str(self._session_path(conversation.shell))
            new_session()
            conversation.started = True
            self._log.info("new conversation for shell %s", conversation.shell)

    def cancel_shell(self, shell: str) -> None:
        """Cancel whatever the agent serving *shell* is streaming."""
        with self._lock:
            slots = [slot for slot in self._slots if slot.shell == shell] or list(self._slots)
        for slot in slots:
            try:
                slot.agent.cancel()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("cancel failed: %s", exc)

    # -- message handling --------------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return {
                "running": True,
                "pid": os.getpid(),
                "socket": str(self.socket_path),
                "shells": sorted(self._shells),
                "agents": len(self._slots),
                "conversations": {
                    shell: {
                        "sleeping": conversation.sleeping,
                        "requests": conversation.requests,
                        "session_path": conversation.session_path,
                    }
                    for shell, conversation in self._conversations.items()
                },
                "backend": self._backend_name or self.config.agent_provider,
                "backend_reason": self._backend_reason,
                "fallback_notice": self._fallback_notice,
                "idle_timeout": self.idle_timeout,
            }

    def handle_message(self, message: Mapping[str, object]) -> Iterator[AgentEvent]:
        """Handle one decoded request line, yielding the events to stream back."""
        self._last_activity = time.monotonic()
        shell = str(message.get("shell", "") or "")
        kind = str(message.get("kind", "") or "")

        if kind in _CONTROL_KINDS:
            yield from self._handle_control(kind, shell, message)
            return

        if not shell:
            yield AgentEvent(kind=EventKind.ERROR, error="request is missing 'shell'")
            return

        with self._lock:
            self._shells.add(shell)
            self._had_shells = True

        request = request_from_dict(message.get("request"))  # type: ignore[arg-type]
        context = context_from_dict(message.get("context"))  # type: ignore[arg-type]
        yield from self._run(shell, request, context)

    def _handle_control(
        self, kind: str, shell: str, message: Mapping[str, object]
    ) -> Iterator[AgentEvent]:
        if kind == "register":
            with self._lock:
                if shell:
                    self._shells.add(shell)
                    self._had_shells = True
            yield AgentEvent(kind=EventKind.STATUS, text=f"registered {shell}")
            yield AgentEvent(kind=EventKind.DONE)
            return

        if kind == "unregister":
            with self._lock:
                self._shells.discard(shell)
                empty = self._had_shells and not self._shells
            yield AgentEvent(kind=EventKind.STATUS, text=f"unregistered {shell}")
            yield AgentEvent(kind=EventKind.DONE)
            if empty:
                self._log.info("last shell left; shutting down")
                self.shutdown()
            return

        if kind == "cancel":
            self.cancel_shell(shell)
            yield AgentEvent(kind=EventKind.STATUS, text=f"cancelled {shell}")
            yield AgentEvent(kind=EventKind.DONE)
            return

        if kind == "undo":
            with self._lock:
                conversation = self._conversations.get(shell)
                dropped = conversation.undo() if conversation is not None else False
            text = "undone" if dropped else "nothing to undo"
            yield AgentEvent(kind=EventKind.STATUS, text=text)
            yield AgentEvent(kind=EventKind.DONE)
            return

        if kind == "ui_response":
            yield from self._handle_ui_response(shell, message)
            return

        if kind in ("status", "ping"):
            yield AgentEvent(kind=EventKind.STATUS, text=json.dumps(self.state()))
            yield AgentEvent(kind=EventKind.DONE)
            return

        # kind == "stop"
        yield AgentEvent(kind=EventKind.STATUS, text="stopping")
        yield AgentEvent(kind=EventKind.DONE)
        self.shutdown()

    def _handle_ui_response(
        self, shell: str, message: Mapping[str, object]
    ) -> Iterator[AgentEvent]:
        request_id = str(message.get("request_id", "") or "")
        raw_fields = message.get("fields")
        fields = dict(raw_fields) if isinstance(raw_fields, Mapping) else {}
        with self._lock:
            slots = [slot for slot in self._slots if slot.shell == shell] or list(self._slots)
        delivered = False
        for slot in slots:
            respond = getattr(slot.agent, "respond_ui", None)
            if callable(respond):
                respond(request_id, **fields)
                delivered = True
        if not delivered:
            yield AgentEvent(kind=EventKind.ERROR, error="no agent is waiting for a UI response")
            return
        yield AgentEvent(kind=EventKind.STATUS, text=f"ui response {request_id} delivered")
        yield AgentEvent(kind=EventKind.DONE)

    def _run(
        self, shell: str, request: AgentRequest, context: AgentContext
    ) -> Iterator[AgentEvent]:
        # One request at a time: pi itself steers one-at-a-time, and a single
        # shared agent process must never interleave two conversations.
        with self._run_lock:
            try:
                slot, conversation = self._acquire(shell)
            except Exception as exc:  # noqa: BLE001 - a backend that won't start
                self._log.warning("could not start an agent: %s", exc)
                yield AgentEvent(kind=EventKind.ERROR, error=f"no agent available: {exc}")
                return

            if self._fallback_notice and shell not in self._notified:
                self._notified.add(shell)
                yield AgentEvent(kind=EventKind.STATUS, text=self._fallback_notice)

            conversation.requests += 1
            turn = {"prompt": request.prompt, "text": ""}
            conversation.transcript.append(turn)
            conversation.pending_proposal = None
            saw_terminal = False
            try:
                for event in slot.agent.run(request, context):
                    self._last_activity = time.monotonic()
                    if event.kind is EventKind.TEXT_DELTA and event.text:
                        turn["text"] += event.text
                    elif event.kind is EventKind.PROPOSAL and event.proposal is not None:
                        conversation.pending_proposal = {
                            "command": event.proposal.command,
                            "rationale": event.proposal.rationale,
                            "kind": event.proposal.kind.value,
                        }
                    saw_terminal = event.kind in (EventKind.DONE, EventKind.ERROR)
                    yield event
                    if saw_terminal:
                        break
            except Exception as exc:  # noqa: BLE001 - adapter crash must not kill us
                self._log.warning("agent run failed: %s", exc)
                yield AgentEvent(kind=EventKind.ERROR, error=f"agent error: {exc}")
                return
            if not saw_terminal:
                yield AgentEvent(kind=EventKind.DONE)
            self._last_activity = time.monotonic()


# --- foreground entry point ------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m nvsh.daemon --foreground``."""
    parser = argparse.ArgumentParser(
        prog="nvsh-daemon", description="Run the nvsh session daemon in the foreground."
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in this process (the only supported mode of this module).",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Seconds of inactivity after which the daemon exits.",
    )
    args = parser.parse_args(argv)

    from .config import load as load_config

    try:
        config = load_config()
    except Exception:  # noqa: BLE001 - a broken config must not block diagnosis
        config = Config()
    daemon = Daemon(config, idle_timeout=args.idle_timeout)
    try:
        daemon.serve_forever()
    except OSError as exc:
        # Losing the lock is the normal outcome of two shells failing at
        # once: log it and exit at once, touching nothing this daemon does
        # not own.
        daemon._log.info("not starting: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
