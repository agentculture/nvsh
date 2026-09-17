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
import contextlib
import fcntl
import json
import logging
import os
import select
import socket
import socketserver
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, cast

from nvsh import __version__, runtimedir

from .agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    EventKind,
    NvshAgent,
    Target,
    event_to_dict,
    request_from_dict,
    target_to_dict,
)
from .config import DEFAULT_ALIAS, Config

#: Seconds with no request after which the daemon shuts itself down.
DEFAULT_IDLE_TIMEOUT = 900.0

#: How often the idle watchdog wakes up.
_WATCHDOG_INTERVAL = 0.1

#: Wall-clock seconds one agent turn may take before the daemon aborts it.
#: Generous on purpose -- a cold local model on a Jetson can think for
#: minutes -- but finite, because a turn that never ends blocks every other
#: shell (deviation d12).
DEFAULT_TURN_TIMEOUT = 300.0

#: Environment override for that cap, in seconds (tests use a small value).
TURN_TIMEOUT_ENV = "NVSH_TURN_TIMEOUT"

#: How often the per-turn watcher checks the owning client and the cap.
_TURN_WATCH_INTERVAL = 0.05

#: How often a queued request re-announces that it is still waiting. Well
#: under the client's stream timeout (120 s), so a queued client keeps
#: receiving bytes instead of timing out in silence.
_QUEUE_NOTICE_INTERVAL = 15.0

#: How long a ``kill`` control waits for the killed turn to hand back the run
#: lock before answering. The kill itself is not bounded by this; it only
#: decides whether the reply says "killed" or "still stopping".
_KILL_WAIT = 3.0

#: How long a request that got a ``busy`` event waits for a ``busy_choice``
#: before it gives up on the prompt and queues as it always did. Bounds the
#: wait for a client that ignores ``busy`` (one that predates t10), so it is
#: never stranded; well under the client's 120 s stream timeout.
_BUSY_CHOICE_TIMEOUT = 60.0

#: The answers a ``busy`` prompt accepts.
BUSY_CHOICES = ("steer", "replace", "exit")

#: Control kinds that carry no agent request.
_CONTROL_KINDS = frozenset(
    {
        "register",
        "unregister",
        "cancel",
        "kill",
        "kill_active",
        "busy_choice",
        "ui_response",
        "steer",
        "status",
        "stop",
        "ping",
        "undo",
    }
)

_LOGGER_NAME = "nvsh.daemon"

#: Wire key carrying the sender's ``nvsh.__version__`` (task t16). A client
#: that predates the handshake omits it, and is served normally -- the point
#: of the handshake is to notice a daemon left over from an *upgrade*, not to
#: refuse anyone.
VERSION_KEY = "version"


def peer_version(message: Mapping[str, object]) -> str:
    """The ``nvsh`` version the sender declared, or ``""`` when it declared none."""
    raw = message.get(VERSION_KEY)
    return raw if isinstance(raw, str) and raw else ""


def version_mismatch(message: Mapping[str, object], ours: str = __version__) -> str:
    """The sender's version when it differs from *ours*, else ``""``.

    An absent version is never a mismatch: a pre-handshake client still gets
    an answer rather than an error it has no code to understand.
    """
    declared = peer_version(message)
    return "" if not declared or declared == ours else declared


class DaemonAlreadyRunning(OSError):
    """Raised when another daemon already holds this user's single-instance lock.

    An :class:`OSError` so every existing caller (``main``, ``nvsh daemon
    run``) already treats it as "could not serve" and exits 2.
    """


# --- paths -----------------------------------------------------------------


def _suppressed() -> contextlib.AbstractContextManager:
    """Swallow anything a best-effort teardown raises. Never used for logic."""
    return contextlib.suppress(Exception)


def _resolve_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def runtime_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$XDG_RUNTIME_DIR/nvsh``, or a per-uid temp directory when unset.

    Never a hard-coded path, and never a directory shared between users:
    without ``XDG_RUNTIME_DIR`` we fall back to ``<tmp>/nvsh-<uid>``.
    """
    return runtimedir.runtime_dir(_resolve_env(env))


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


def turn_timeout(env: Mapping[str, str] | None = None) -> float:
    """How long one agent turn may run before it is aborted, in seconds.

    ``$NVSH_TURN_TIMEOUT`` overrides :data:`DEFAULT_TURN_TIMEOUT`. Junk or
    non-positive values fall back to the default rather than disabling the
    cap: "no cap" is exactly the state deviation d12 wedged in.
    """
    resolved = _resolve_env(env)
    try:
        value = float(resolved.get(TURN_TIMEOUT_ENV, ""))
    except (TypeError, ValueError):
        return DEFAULT_TURN_TIMEOUT
    return value if value > 0 else DEFAULT_TURN_TIMEOUT


#: Module-level alias, so :class:`Daemon` can resolve the cap from the
#: environment inside a method whose parameter shadows ``turn_timeout``.
_turn_timeout_from_env = turn_timeout


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


#: Re-exported so ``nvsh.daemon.request_from_dict`` keeps working. The codec
#: itself lives in :mod:`nvsh.agent.base` next to ``request_to_dict``, so the
#: encoder and the decoder cannot drift apart (they did: the daemon's own
#: copy never learned about ``AgentRequest.target``). It stays lenient about
#: missing keys, so a request line written by a pre-``target`` client still
#: parses -- it simply decodes with ``target=None``.


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


def _peer_is_gone(connection: socket.socket | None) -> bool:
    """Has the client on *connection* closed it (EOF, reset, or a half-close)?

    A closed or reset peer makes the socket readable with nothing to read;
    the client only ever sends one line, which the handler has already
    consumed, so readable-with-data means a live (if chatty) client. Probing
    uses ``MSG_PEEK`` so nothing is ever taken off the stream. ``None`` --
    an in-process caller with no socket -- is never "gone".
    """
    if connection is None:
        return False
    try:
        readable, _, _ = select.select([connection], [], [], 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        peeked = connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return False
    except OSError:
        return True
    return not peeked


@dataclass
class _ActiveTurn:
    """The one agent turn currently running, and how to end it from outside.

    The daemon serves one agent request at a time (pi steers one-at-a-time,
    and a shared process must never interleave two conversations), so this
    object *is* the thing every other shell is waiting on. It carries what a
    watcher thread needs to end it without touching the handler thread that
    is blocked inside ``agent.run()``: the slot to cancel, the ids of any
    approval dialogs that have to be answered first, and a flag the handler
    reads to tell an abort apart from a normal finish.
    """

    shell: str
    slot: _Slot
    started: float = field(default_factory=time.time)
    since: float = field(default_factory=time.monotonic)
    pending_ui: list[str] = field(default_factory=list)
    aborted: str = ""
    finished: threading.Event = field(default_factory=threading.Event)

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.since)

    def snapshot(self) -> dict:
        return {"shell": self.shell, "started": self.started, "elapsed": self.elapsed()}


@dataclass
class _BusyPrompt:
    """One request parked on a ``busy`` event, waiting for the operator's choice."""

    turn: _ActiveTurn
    steerable: bool
    choice: str = ""
    chosen: threading.Event = field(default_factory=threading.Event)


def _shell_pid_gone(shell: str) -> bool:
    """Is *shell* a pid that no longer exists?

    A shell id is the hook's ``$$``. Anything that is not a positive integer
    (a test's ``"A"``, an in-process caller) cannot be probed and counts as
    alive, as does a pid we may not signal: only ``ESRCH`` means gone.
    """
    try:
        pid = int(shell)
    except (TypeError, ValueError):  # ValueError also covers int's digit limit
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, OverflowError, ValueError):
        # OverflowError: a numeric id too large for a C pid_t cannot be
        # probed at all, so it counts as alive like any other unprobeable id.
        return False
    return False


def _turn_matches(turn: _ActiveTurn, expected: Mapping[str, object]) -> bool:
    """Is *turn* the one *expected* (``{"shell", "started"}``) names?"""
    started = expected.get("started")
    if isinstance(started, bool) or not isinstance(started, (int, float)):
        return False
    return expected.get("shell") == turn.shell and float(started) == turn.started


def shell_pid_gone(shell: str) -> bool:
    """Public wrapper around :func:`_shell_pid_gone` (task t19).

    ``nvsh.doctor_checks`` and ``nvsh doctor --apply`` need the same
    pid-liveness predicate the daemon itself uses to decide whether a
    live-owner kill needs the operator's confirmation, so this is exposed
    rather than duplicated.
    """
    return _shell_pid_gone(shell)


def _overrides_steer(agent: NvshAgent) -> bool:
    """Does *agent*'s adapter have a mid-turn channel (it overrides ``steer``)?"""
    return getattr(type(agent), "steer", NvshAgent.steer) is not NvshAgent.steer


@dataclass
class _Waiter:
    """One request queued behind :class:`_ActiveTurn`."""

    shell: str
    started: float = field(default_factory=time.time)
    since: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict:
        return {
            "shell": self.shell,
            "started": self.started,
            "waiting": max(0.0, time.monotonic() - self.since),
        }


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
        except ValueError as exc:  # ValueError covers UnicodeDecodeError
            self._write(AgentEvent(kind=EventKind.ERROR, error=f"bad request: {exc}"))
            return
        for event in owner.handle_message(message, connection=self.connection):
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
        turn_timeout: float | None = None,
        which: WhichFn | None = None,
    ) -> None:
        self.config = config if config is not None else Config()
        self.env: Mapping[str, str] = dict(_resolve_env(env))
        self.idle_timeout = float(idle_timeout)
        self.turn_timeout = (
            float(turn_timeout)
            if turn_timeout is not None and turn_timeout > 0
            else _turn_timeout_from_env(self.env)
        )
        self._agent_factory = agent_factory
        self._which = which

        #: The ``nvsh`` version this daemon reports and compares clients
        #: against. An attribute rather than a module constant so a test can
        #: stand up a daemon that looks like a leftover from another build
        #: without patching the module every other daemon reads.
        self.version = __version__

        self._lock = threading.RLock()
        self._run_lock = threading.RLock()
        self._slots: list[_Slot] = []
        self._conversations: dict[str, Conversation] = {}
        self._shells: set[str] = set()
        self._had_shells = False
        self._backend_name = "" if agent_factory is None else "injected"
        self._backend_reason = ""
        #: The target the warm agent was actually built for (t16). ``None``
        #: until the first agent is built; ``state()`` falls back to the
        #: freshly resolved default so a caller can render it before then.
        self._target: Target | None = None
        self._fallback_notice = ""
        self._notified: set[str] = set()
        self._active: _ActiveTurn | None = None
        self._waiting: list[_Waiter] = []
        self._busy: dict[str, _BusyPrompt] = {}

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

    def default_target(self) -> Target:
        """What ``default`` resolves to right now (t16, decision c25).

        One call into :meth:`Config.resolve_target`, so the daemon's warm
        session honours ``[aliases] default`` exactly like every other entry
        point instead of reading the legacy ``[agent] provider`` directly. A
        config too broken to resolve still yields a usable target -- the
        legacy provider -- because a daemon that cannot name a backend is a
        daemon that cannot diagnose anything.
        """
        try:
            backend, model, effort, by_alias = self.config.resolve_target(DEFAULT_ALIAS)
        except Exception:  # noqa: BLE001 - a broken alias must not block a diagnosis
            backend = self.config.agent_provider
            model = str(self.config.agents.get(backend, {}).get("model") or "") or None
            effort, by_alias = None, True
        return Target(
            backend=backend,
            model=model,
            effort=effort,
            alias=DEFAULT_ALIAS if by_alias else None,
        )

    def _config_for(self, target: Target) -> Config:
        """A copy of the config whose ``[agents.<backend>]`` carries *target*.

        Every adapter factory reads its own ``[agents.<name>]`` table for
        ``model``/``effort``/``extra_args``/``approval``, so folding a
        resolved target's model and effort into that table is what passes
        all four to the adapter -- without the daemon having to know which
        keyword each of the seven adapters happens to take.
        """
        settings = dict(self.config.agents.get(target.backend, {}))
        if target.model:
            settings["model"] = target.model
        if target.effort:
            settings["effort"] = target.effort
        agents = dict(self.config.agents)
        agents[target.backend] = settings
        return cast(Config, replace(self.config, agent_provider=target.backend, agents=agents))

    def _make_agent(self, target: Target, *, forced: bool = False) -> tuple[NvshAgent, str, str]:
        """Build one adapter for *target*. Records nothing on the daemon.

        ``forced=True`` is the explicitly-targeted path: a missing binary
        raises rather than silently becoming ``openai-compat``, because the
        operator asked for *that* harness on purpose.
        """
        from .agent import registry

        cfg = self._config_for(target)
        kwargs: dict = {}
        if self._which is not None:
            kwargs["which"] = self._which
        if forced:
            kwargs["forced"] = target
        name, reason = registry.choose(cfg, **kwargs)
        return registry.ADAPTERS[name].factory(cfg), name, reason

    def _build_agent(self) -> NvshAgent:
        """Build the *warm* agent: whatever ``default`` resolves to."""
        if self._agent_factory is not None:
            return self._agent_factory()

        wanted = self.default_target()
        agent, name, reason = self._make_agent(wanted)
        self._backend_name = name
        self._backend_reason = reason
        if name == wanted.backend:
            self._target = wanted
        else:
            # The fallback backend is a different harness, so the resolved
            # model/effort belonged to the one that was not there.
            self._target = Target(backend=name, alias=wanted.alias)
            self._fallback_notice = f"{wanted.backend} unavailable: {reason}"
            self._log.info("%s", self._fallback_notice)
        return agent

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
                try:
                    agent.start()
                except Exception:
                    # A backend that will not start owns no slot: close it
                    # so no half-live process is left behind, and let the
                    # next request build a fresh one.
                    with _suppressed():
                        agent.close()
                    raise
                slot = _Slot(agent=agent)
                self._slots.append(slot)
            else:
                slot = min(self._slots, key=lambda item: item.last_used)
                if slot.shell is not None and slot.shell != shell:
                    sleeping = self._conversations.get(slot.shell)
                    if sleeping is not None:
                        sleeping.sleeping = True
                        self._log.info("putting shell %s to sleep", slot.shell)

            try:
                self._activate(slot, conversation)
            except Exception:
                # The agent is in an unknown session state; retire it rather
                # than run a turn against it (d14).
                self._retire(slot)
                raise
            slot.shell = shell
            slot.last_used = time.monotonic()
            conversation.sleeping = False
            return slot, conversation

    def _retire(self, slot: _Slot) -> None:
        """Drop *slot* and close its agent. Never raises."""
        with self._lock:
            if slot in self._slots:
                self._slots.remove(slot)
        with _suppressed():
            slot.agent.close()

    def _activate(self, slot: _Slot, conversation: Conversation) -> None:
        """Point the agent at *conversation*'s session before the turn runs.

        Both calls are synchronous against the backend -- ``PiAgent`` waits
        for pi's acknowledgement before returning -- so the prompt the
        daemon sends next cannot overtake the session command. It used to,
        and pi then answered neither: the daemon-routed turn produced no
        event at all until the client's 120 s stream timeout gave up
        (deviation d14).
        """
        new_session = getattr(slot.agent, "new_session", None)
        switch_session = getattr(slot.agent, "switch_session", None)
        if not callable(new_session) or not callable(switch_session):
            return  # a stateless adapter has nothing to swap
        if conversation.started and conversation.session_path:
            try:
                switch_session(conversation.session_path)
            except Exception as exc:  # noqa: BLE001 - a stale/missing session file
                # Losing the history of one terminal is a nuisance; refusing
                # to answer at all is the d14 failure. Say so and start over.
                self._log.error(
                    "shell %s: could not resume session %s (%s); starting a fresh one",
                    conversation.shell,
                    conversation.session_path,
                    exc,
                )
                conversation.started = False
                conversation.session_path = None
                self._activate(slot, conversation)
                return
            self._log.info("resumed shell %s", conversation.shell)
        else:
            # The backend names its own session file (pi has no rpc command
            # that chooses one), so remember what it reports and fall back
            # to this daemon's per-shell key only when it reports nothing.
            reported = new_session()
            conversation.session_path = (
                str(reported) if reported else str(self._session_path(conversation.shell))
            )
            conversation.started = True
            self._log.info(
                "new conversation for shell %s (session %s)",
                conversation.shell,
                conversation.session_path,
            )

    def cancel_shell(self, shell: str) -> None:
        """Cancel whatever the agent serving *shell* is streaming.

        Strictly per-shell: a shell that holds no slot has nothing to cancel.
        Falling back to every slot meant a terminal queued behind another
        terminal's turn cancelled *that* terminal's agent the moment its own
        operator interrupted the panel.
        """
        with self._lock:
            slots = [slot for slot in self._slots if slot.shell == shell]
        for slot in slots:
            try:
                slot.agent.cancel()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("cancel failed: %s", exc)

    def kill_shell(self, shell: str, *, wait: float = _KILL_WAIT) -> str:
        """Force-stop *shell*'s own running turn, for a harness that ignores cancel.

        The second press (t9): the slot's adapter gets ``force_stop()``, the
        turn is marked aborted so its handler thread stops streaming, and the
        slot is dropped from the pool so the next request builds a *fresh*
        agent instead of reusing a process in an unknown state. The run lock
        is released by the killed turn's own handler thread as ``run()``
        returns; this waits up to *wait* seconds to see that happen.

        Strictly per-shell, like :meth:`cancel_shell`: a shell whose turn is
        not the active one kills nothing. Returns ``"killed"``,
        ``"stopping"`` (force-stopped but the turn has not ended yet) or
        ``"idle"`` (nothing of this shell's to kill). Never raises.
        """
        with self._lock:
            turn = self._active
            if turn is None or not shell or turn.shell != shell:
                return "idle"
        return self._force_stop_turn(turn, wait=wait)

    def _force_stop_turn(self, turn: _ActiveTurn, *, wait: float = _KILL_WAIT) -> str:
        """Force-stop *turn*, whoever owns it. Callers decide whether they may.

        :meth:`kill_shell` allows only the owner; a ``replace`` answer to a
        busy prompt also allows any shell once the owner's pid is gone.

        Every caller captured *turn* under the lock and then let go of it, so
        by now the turn may have ended and its slot may serve later work.
        The identity is re-checked under the lock: a turn that is no longer
        the active one, or has already finished, is left alone and
        ``"idle"`` is returned without signalling anything.
        """
        with self._lock:
            if self._active is not turn or turn.finished.is_set():
                return "idle"
            if not turn.aborted:
                turn.aborted = "killed"
            slot = turn.slot
            if slot in self._slots:
                self._slots.remove(slot)
            conversation = self._conversations.get(turn.shell)
            if conversation is not None:
                conversation.pending_proposal = None
                conversation.sleeping = True
        self._log.info("force-stopping shell %s's turn", turn.shell)
        try:
            slot.agent.force_stop()
        except Exception as exc:  # noqa: BLE001 - a kill must never wedge the daemon
            self._log.warning("force_stop failed: %s", exc)
        with _suppressed():
            slot.agent.close()
        return "killed" if turn.finished.wait(max(0.0, wait)) else "stopping"

    def kill_active_turn(
        self,
        *,
        confirmed: bool,
        expected: Mapping[str, object] | None = None,
        wait: float = _KILL_WAIT,
    ) -> str:
        """Force-stop the active turn whoever owns it, for ``nvsh doctor --apply`` (t19).

        Unlike :meth:`kill_shell`, the caller here (``nvsh doctor``, running
        as its own process) is never the turn's owner, so ownership alone
        can't gate this. The daemon decides for itself, never trusting the
        caller's own belief about the owner: a dead-owner turn (the shell
        that started it no longer exists) is killed outright regardless of
        *confirmed*; a turn with a live owner is killed only when *confirmed*
        is ``True`` -- the caller's signal that the operator was shown that
        shell's id and agreed -- *and* *expected* names that very turn.

        *expected* is the ``{"shell", "started"}`` identity from the status
        snapshot the operator approved. When given, it is compared with the
        active turn under the lock, and a different turn is never killed
        (``"changed"``): an approval naming one shell's turn must not end
        whatever turn happens to be running by the time it arrives.

        Returns ``"idle"`` (nothing running), ``"changed"`` (the active turn is
        not the expected one), ``"refused"`` (live owner, not confirmed for
        this turn) or :meth:`_force_stop_turn`'s ``"killed"``/``"stopping"``.
        Never raises.
        """
        with self._lock:
            turn = self._active
            if turn is None:
                return "idle"
            if expected is not None and not _turn_matches(turn, expected):
                return "changed"
            owner = turn.shell
        if not (confirmed and expected is not None) and not _shell_pid_gone(owner):
            return "refused"
        return self._force_stop_turn(turn, wait=wait)

    # -- the active turn (deviation d12) -----------------------------------

    def active_turn(self) -> dict | None:
        """The turn currently holding the agent, or ``None`` when idle."""
        with self._lock:
            return self._active.snapshot() if self._active is not None else None

    def _abort_turn(self, turn: _ActiveTurn, why: str, detail: str) -> None:
        """End *turn* from outside the thread that is blocked inside it.

        Order matters: a turn parked on an approval dialog is not waiting on
        the model at all, so the dialog is denied *first* (otherwise the
        adapter has nothing to cancel and stays parked), then the adapter is
        cancelled, then the conversation is left idle. Every step is
        best-effort -- an adapter that raises here must not keep the run lock
        held for everyone else.
        """
        with self._lock:
            if turn.aborted:
                return
            turn.aborted = why
            pending = list(turn.pending_ui)
        self._log.info("aborting shell %s's turn: %s", turn.shell, detail)

        agent = turn.slot.agent
        respond = getattr(agent, "respond_ui", None)
        if callable(respond):
            for request_id in pending:
                try:
                    respond(request_id, cancelled=True)
                except Exception as exc:  # noqa: BLE001 - abort must not raise
                    self._log.warning("denying dialog %s failed: %s", request_id, exc)
        try:
            agent.cancel()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("cancel failed: %s", exc)

        with self._lock:
            conversation = self._conversations.get(turn.shell)
            if conversation is not None:
                conversation.pending_proposal = None
                conversation.sleeping = True

    def _watch_turn(self, turn: _ActiveTurn, connection: socket.socket | None) -> None:
        """Abort *turn* when its client goes away or it outruns the cap.

        Runs in its own thread because the thread that owns the turn is
        parked inside ``agent.run()`` and cannot notice either condition.
        """
        deadline = turn.since + self.turn_timeout
        while not turn.finished.wait(_TURN_WATCH_INTERVAL):
            if _peer_is_gone(connection):
                self._abort_turn(turn, "client-gone", "its client closed the connection")
                return
            if time.monotonic() >= deadline:
                self._abort_turn(
                    turn, "timeout", f"it ran past the {self.turn_timeout:g}s turn cap"
                )
                return

    def _wait_for_the_agent(self, shell: str) -> Iterator[AgentEvent]:
        """Take the run lock, telling the caller out loud while it waits.

        Yields a ``status`` event *before* the first blocking wait and again
        every :data:`_QUEUE_NOTICE_INTERVAL`, so a queued client keeps
        receiving bytes and never hits its stream timeout in silence --
        which is how the wedge showed up as ``daemon connection lost``.
        """
        if self._run_lock.acquire(blocking=False):
            return
        waiter = _Waiter(shell)  # positional: bandit B604 trips on any `shell=` kwarg
        with self._lock:
            self._waiting.append(waiter)
        try:
            yield AgentEvent(kind=EventKind.STATUS, text=self._busy_notice())
            while not self._run_lock.acquire(timeout=_QUEUE_NOTICE_INTERVAL):
                yield AgentEvent(kind=EventKind.STATUS, text=self._busy_notice(still=True))
        finally:
            with self._lock:
                if waiter in self._waiting:
                    self._waiting.remove(waiter)

    def _busy_notice(self, *, still: bool = False) -> str:
        with self._lock:
            busy = self._active.shell if self._active is not None else "?"
        lead = "still waiting" if still else "waiting"
        return f"{lead} for the agent (busy with shell {busy})"

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
                "version": self.version,
                "target": target_to_dict(self._target or self.default_target()),
                "backend": self._backend_name or self.config.agent_provider,
                "backend_reason": self._backend_reason,
                "fallback_notice": self._fallback_notice,
                "idle_timeout": self.idle_timeout,
                "turn_timeout": self.turn_timeout,
                "active_turn": self._active.snapshot() if self._active is not None else None,
                "queued": [waiter.snapshot() for waiter in self._waiting],
            }

    def handle_message(
        self, message: Mapping[str, object], *, connection: socket.socket | None = None
    ) -> Iterator[AgentEvent]:
        """Handle one decoded request line, yielding the events to stream back.

        ``connection`` is the client's socket when there is one; the daemon
        watches it for the duration of an agent turn so a client that walks
        away releases the agent instead of wedging every other shell behind
        it (deviation d12).
        """
        self._last_activity = time.monotonic()
        shell = str(message.get("shell", "") or "")
        kind = str(message.get("kind", "") or "")

        if kind in _CONTROL_KINDS:
            yield from self._handle_control(kind, shell, message)
            return

        if not shell:
            yield AgentEvent(kind=EventKind.ERROR, error="request is missing 'shell'")
            return

        stale = version_mismatch(message, self.version)
        if stale:
            # Only agent requests are refused: ``stop``/``status``/``ping``
            # and the register/unregister bookkeeping still answer, because
            # stopping this daemon is exactly what the client does next.
            yield self._version_error(stale)
            return

        with self._lock:
            self._shells.add(shell)
            self._had_shells = True

        request = request_from_dict(message.get("request"))  # type: ignore[arg-type]
        context = context_from_dict(message.get("context"))  # type: ignore[arg-type]
        yield from self._run(shell, request, context, connection=connection)

    def _handle_register(self, shell: str) -> Iterator[AgentEvent]:
        """Remember a shell that has just come up."""
        with self._lock:
            if shell:
                self._shells.add(shell)
                self._had_shells = True
        yield AgentEvent(kind=EventKind.STATUS, text=f"registered {shell}")
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_unregister(self, shell: str) -> Iterator[AgentEvent]:
        """Forget a shell, and shut down once the last one has gone."""
        with self._lock:
            self._shells.discard(shell)
            empty = self._had_shells and not self._shells
        yield AgentEvent(kind=EventKind.STATUS, text=f"unregistered {shell}")
        yield AgentEvent(kind=EventKind.DONE)
        if empty:
            self._log.info("last shell left; shutting down")
            self.shutdown()

    def _handle_undo(self, shell: str) -> Iterator[AgentEvent]:
        """Drop this shell's last exchange, if it has one."""
        with self._lock:
            conversation = self._conversations.get(shell)
            dropped = conversation.undo() if conversation is not None else False
        yield AgentEvent(kind=EventKind.STATUS, text="undone" if dropped else "nothing to undo")
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_control(
        self, kind: str, shell: str, message: Mapping[str, object]
    ) -> Iterator[AgentEvent]:
        handlers: dict[str, Callable[[str, Mapping[str, object]], Iterator[AgentEvent]]] = {
            "register": lambda sh, _msg: self._handle_register(sh),
            "unregister": lambda sh, _msg: self._handle_unregister(sh),
            "cancel": lambda sh, _msg: self._handle_cancel(sh),
            "kill": lambda sh, _msg: self._handle_kill(sh),
            "kill_active": lambda _sh, msg: self._handle_kill_active(msg),
            "busy_choice": self._handle_busy_choice,
            "undo": lambda sh, _msg: self._handle_undo(sh),
            "ui_response": self._handle_ui_response,
            "steer": self._handle_steer,
            "status": lambda _sh, _msg: self._handle_status(),
            "ping": lambda _sh, _msg: self._handle_status(),
        }
        handler = handlers.get(kind)
        if handler is not None:
            yield from handler(shell, message)
            return

        # kind == "stop"
        yield AgentEvent(kind=EventKind.STATUS, text="stopping")
        yield AgentEvent(kind=EventKind.DONE)
        self.shutdown()

    def _handle_cancel(self, shell: str) -> Iterator[AgentEvent]:
        self.cancel_shell(shell)
        yield AgentEvent(kind=EventKind.STATUS, text=f"cancelled {shell}")
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_kill(self, shell: str) -> Iterator[AgentEvent]:
        outcome = self.kill_shell(shell)
        if outcome == "idle":
            yield AgentEvent(kind=EventKind.ERROR, error=f"no running turn to kill for {shell}")
            return
        yield AgentEvent(kind=EventKind.STATUS, text=f"{outcome} {shell}")
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_kill_active(self, message: Mapping[str, object]) -> Iterator[AgentEvent]:
        # Only JSON ``true`` confirms: bool("false") is truthy, and a value
        # that merely looks confirming must never authorize a live-owner kill.
        confirmed = message.get("confirmed", False) is True
        raw_expected = message.get("expected")
        expected = raw_expected if isinstance(raw_expected, Mapping) else None
        outcome = self.kill_active_turn(confirmed=confirmed, expected=expected)
        errors = {
            "idle": "no active turn to kill",
            "changed": "active turn changed since it was confirmed; re-run nvsh doctor",
            "refused": "active turn has a live owner; confirm required",
        }
        if outcome in errors:
            yield AgentEvent(kind=EventKind.ERROR, error=errors[outcome])
            return
        yield AgentEvent(kind=EventKind.STATUS, text=outcome)
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_status(self) -> Iterator[AgentEvent]:
        yield AgentEvent(kind=EventKind.STATUS, text=json.dumps(self.state()))
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_busy_choice(
        self, shell: str, message: Mapping[str, object]
    ) -> Iterator[AgentEvent]:
        """Deliver the operator's answer to this shell's own open busy prompt."""
        choice = str(message.get("choice", "") or "")
        if choice not in BUSY_CHOICES:
            yield AgentEvent(
                kind=EventKind.ERROR,
                error=f"busy_choice must be one of {', '.join(BUSY_CHOICES)}",
            )
            return
        with self._lock:
            prompt = self._busy.get(shell) if shell else None
            if prompt is not None and choice == "steer" and not prompt.steerable:
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    error="the running turn's harness has no mid-turn channel to steer",
                )
                return
            if prompt is not None and not prompt.chosen.is_set():
                prompt.choice = choice
                prompt.chosen.set()
            else:
                prompt = None
        if prompt is None:
            yield AgentEvent(kind=EventKind.ERROR, error=f"no busy prompt open for {shell}")
            return
        yield AgentEvent(kind=EventKind.STATUS, text=f"busy choice {choice} for {shell}")
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_steer(self, shell: str, message: Mapping[str, object]) -> Iterator[AgentEvent]:
        """Inject the operator's text into this shell's running turn (d16).

        Mirrors :meth:`_handle_ui_response`: the turn is held by another
        thread parked inside ``agent.run()``, and the only thing that can
        reach it is the adapter itself. An adapter with no mid-turn channel
        (or a shell with no turn running) answers ``False``, which comes
        back as an ``error`` -- the client then sends the text as the next
        request rather than believing it landed.

        Only the requesting shell's own slot is consulted: an idle, stale or
        mis-identified shell used to fall back to *every* slot and inject its
        text into another operator's running conversation.
        """
        text = str(message.get("text", "") or "")
        if not text:
            yield AgentEvent(kind=EventKind.ERROR, error="steer needs text")
            return
        with self._lock:
            slots = [slot for slot in self._slots if slot.shell == shell]
        delivered = False
        for slot in slots:
            send_steer = getattr(slot.agent, "steer", None)
            if not callable(send_steer):
                continue
            try:
                delivered = bool(send_steer(text)) or delivered
            except Exception as exc:  # noqa: BLE001 - a steer must never wedge the daemon
                self._log.warning("steer failed: %s", exc)
        if not delivered:
            yield AgentEvent(kind=EventKind.ERROR, error="no running turn to steer")
            return
        self._log.info("steered shell %s's turn", shell)
        yield AgentEvent(kind=EventKind.STATUS, text=f"steer delivered to {shell}")
        yield AgentEvent(kind=EventKind.DONE)

    def _handle_ui_response(
        self, shell: str, message: Mapping[str, object]
    ) -> Iterator[AgentEvent]:
        """Answer this shell's own pending dialog (same isolation as steer)."""
        request_id = str(message.get("request_id", "") or "")
        raw_fields = message.get("fields")
        fields = dict(raw_fields) if isinstance(raw_fields, Mapping) else {}
        with self._lock:
            slots = [slot for slot in self._slots if slot.shell == shell]
        delivered = False
        for slot in slots:
            respond = getattr(slot.agent, "respond_ui", None)
            if callable(respond):
                respond(request_id, **fields)
                delivered = True
        if not delivered:
            yield AgentEvent(kind=EventKind.ERROR, error="no agent is waiting for a UI response")
            return
        with self._lock:
            # Answered: an abort must not deny this dialog a second time.
            if self._active is not None and request_id in self._active.pending_ui:
                self._active.pending_ui.remove(request_id)
        yield AgentEvent(kind=EventKind.STATUS, text=f"ui response {request_id} delivered")
        yield AgentEvent(kind=EventKind.DONE)

    def _version_error(self, client_version: str) -> AgentEvent:
        """The daemon's half of the handshake: its own version, out loud.

        The client uses ``args`` to tell this apart from any other error
        without parsing prose: it stops this daemon, starts one built from
        the same wheel it is running, and retries the request.
        """
        self._log.info(
            "refusing a request from nvsh %s (this daemon runs %s)", client_version, self.version
        )
        return AgentEvent(
            kind=EventKind.ERROR,
            error=(
                f"nvsh version mismatch: this daemon runs {self.version}, "
                f"the client runs {client_version}"
            ),
            args={
                "version_mismatch": True,
                "daemon_version": self.version,
                "client_version": client_version,
            },
        )

    def _targeted(self, target: Target | None) -> bool:
        """Does *target* name something other than this daemon's warm default?

        Decision c25: the warm session belongs to ``default``; anything else
        runs one-shot. The client normally makes that call itself and never
        sends a non-default target here -- this is what happens when one
        arrives anyway (an older client, or a direct API caller).
        """
        if target is None or target.alias == DEFAULT_ALIAS:
            return False
        default = self.default_target()
        return (target.backend, target.model, target.effort) != (
            default.backend,
            default.model,
            default.effort,
        )

    def _run_targeted(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        """Serve one non-default target without touching the warm session."""
        target = request.target
        assert target is not None
        try:
            agent, name, reason = self._make_agent(target, forced=True)
        # Reported to the client as an ERROR event, never raised.
        except Exception as exc:  # noqa: BLE001
            self._log.error("could not build %s: %s", target.backend, exc)
            yield AgentEvent(kind=EventKind.ERROR, error=f"no agent available: {exc}")
            return
        yield AgentEvent(kind=EventKind.STATUS, text=f"one-shot {name}: {reason}")
        saw_terminal = False
        try:
            agent.start()
            for event in agent.run(request, context):
                self._last_activity = time.monotonic()
                saw_terminal = event.kind in (EventKind.DONE, EventKind.ERROR)
                yield event
                if saw_terminal:
                    break
        except Exception as exc:  # noqa: BLE001 - an adapter crash must not kill us
            self._log.error("targeted run failed: %s", exc, exc_info=True)
            yield AgentEvent(kind=EventKind.ERROR, error=f"agent error: {exc}")
            return
        finally:
            # Whatever happened, the one-shot child goes with the turn: this
            # is the path that would otherwise leak a harness per request.
            with _suppressed():
                agent.close()
            self._last_activity = time.monotonic()
        if not saw_terminal:
            yield AgentEvent(kind=EventKind.DONE)

    def _run(
        self,
        shell: str,
        request: AgentRequest,
        context: AgentContext,
        *,
        connection: socket.socket | None = None,
    ) -> Iterator[AgentEvent]:
        # One request at a time: pi itself steers one-at-a-time, and a single
        # shared agent process must never interleave two conversations. The
        # wait is announced out loud, and the turn it waits for is watched,
        # so "one at a time" can never become "one, forever" (d12).
        handled = yield from self._offer_busy(shell, request, connection)
        if handled:
            return
        yield from self._wait_for_the_agent(shell)
        try:
            yield from self._run_locked(shell, request, context, connection)
        finally:
            self._run_lock.release()

    def _offer_busy(
        self, shell: str, request: AgentRequest, connection: socket.socket | None
    ) -> Iterator[AgentEvent]:
        """The busy prompt (t10). Returns ``True`` when the request is finished.

        Offered only to the shell that owns the running turn, or to any shell
        once the owning shell's pid is gone (c8: a live other shell's turn is
        never touched -- that shell just queues with the usual notice). The
        request then parks until a ``busy_choice`` control arrives, the turn
        ends on its own, the client walks away, or
        :data:`_BUSY_CHOICE_TIMEOUT` passes; the last two fall back to the
        queue. A targeted run holds the run lock without an active turn and
        so never opens a prompt (plan risk r6).
        """
        with self._lock:
            turn = self._active
            if turn is None or turn.aborted or not shell or shell in self._busy:
                return False
            owner = turn.shell
            if owner != shell and not _shell_pid_gone(owner):
                return False
            prompt = _BusyPrompt(turn, _overrides_steer(turn.slot.agent))
            self._busy[shell] = prompt
        try:
            choices = [c for c in BUSY_CHOICES if c != "steer" or prompt.steerable]
            yield AgentEvent(
                kind=EventKind.BUSY,
                text=f"the agent is busy with shell {owner}: {', '.join(choices)}?",
                args={
                    "owner": owner,
                    "elapsed": round(turn.elapsed(), 1),
                    "steerable": prompt.steerable,
                    "choices": choices,
                },
            )
            choice = yield from self._await_busy_choice(prompt, connection)
        finally:
            with self._lock:
                if self._busy.get(shell) is prompt:
                    del self._busy[shell]
        return (yield from self._apply_busy_choice(shell, request, prompt, choice))

    def _await_busy_choice(
        self, prompt: _BusyPrompt, connection: socket.socket | None
    ) -> Iterator[AgentEvent]:
        """Wait for *prompt*'s answer; ``""`` means "queue as usual"."""
        began = time.monotonic()
        last_notice = began
        while not prompt.chosen.wait(_TURN_WATCH_INTERVAL):
            now = time.monotonic()
            if prompt.turn.finished.is_set():
                return ""
            if _peer_is_gone(connection):
                return "gone"
            if now - began >= _BUSY_CHOICE_TIMEOUT:
                yield AgentEvent(
                    kind=EventKind.STATUS, text="no answer to the busy prompt; queueing"
                )
                return ""
            if now - last_notice >= _QUEUE_NOTICE_INTERVAL:
                last_notice = now
                yield AgentEvent(kind=EventKind.STATUS, text="waiting for a busy choice")
        return prompt.choice

    def _apply_busy_choice(
        self, shell: str, request: AgentRequest, prompt: _BusyPrompt, choice: str
    ) -> Iterator[AgentEvent]:
        """Act on the answer. Returns ``True`` when the request is finished."""
        owner = prompt.turn.shell
        if choice == "gone":
            return True
        if choice == "exit":
            self._log.info("shell %s left shell %s's turn running", shell, owner)
            yield AgentEvent(kind=EventKind.STATUS, text=f"left shell {owner}'s turn running")
            yield AgentEvent(kind=EventKind.DONE, args={"busy_choice": "exit"})
            return True
        if choice == "steer":
            delivered = False
            if not prompt.turn.finished.is_set():
                try:
                    delivered = bool(prompt.turn.slot.agent.steer(request.prompt))
                except Exception as exc:  # noqa: BLE001 - a steer must never wedge the daemon
                    self._log.warning("steer failed: %s", exc)
            if delivered:
                self._log.info("shell %s steered shell %s's turn", shell, owner)
                yield AgentEvent(kind=EventKind.STATUS, text=f"steer delivered to {owner}")
                yield AgentEvent(kind=EventKind.DONE, args={"busy_choice": "steer"})
                return True
            # d16: no mid-turn channel took it, so it becomes the next request.
            yield AgentEvent(
                kind=EventKind.STATUS, text="steer not taken; sending it as the next request"
            )
            return False
        if choice == "replace":
            self._log.info("shell %s replaces shell %s's turn", shell, owner)
            yield AgentEvent(kind=EventKind.STATUS, text=f"replacing shell {owner}'s turn")
            self._force_stop_turn(prompt.turn)
        return False

    def _run_locked(
        self,
        shell: str,
        request: AgentRequest,
        context: AgentContext,
        connection: socket.socket | None,
    ) -> Iterator[AgentEvent]:
        if self._targeted(request.target):
            yield from self._run_targeted(request, context)
            return
        try:
            slot, conversation = self._acquire(shell)
        except Exception as exc:  # noqa: BLE001 - a backend that won't start
            # Loud on both channels: the operator sees an error panel, and
            # the log keeps the traceback that says which backend broke and
            # how (d14 -- this used to be a silent wait).
            self._log.error("shell %s: could not start an agent: %s", shell, exc, exc_info=True)
            yield AgentEvent(kind=EventKind.ERROR, error=f"no agent available: {exc}")
            return

        if self._fallback_notice and shell not in self._notified:
            self._notified.add(shell)
            yield AgentEvent(kind=EventKind.STATUS, text=self._fallback_notice)

        conversation.requests += 1
        turn = {"prompt": request.prompt, "text": ""}
        conversation.transcript.append(turn)
        conversation.pending_proposal = None

        active = _ActiveTurn(shell, slot)  # positional: see _Waiter above
        with self._lock:
            self._active = active
        watcher = threading.Thread(target=self._watch_turn, args=(active, connection), daemon=True)
        watcher.start()

        saw_terminal = False
        try:
            for event in slot.agent.run(request, context):
                self._last_activity = time.monotonic()
                self._record_event(event, turn, conversation, active)
                saw_terminal = event.kind in (EventKind.DONE, EventKind.ERROR)
                yield event
                if saw_terminal:
                    break
                if active.aborted:
                    break
        except Exception as exc:  # noqa: BLE001 - adapter crash must not kill us
            self._log.error("shell %s: agent run failed: %s", shell, exc, exc_info=True)
            # The backend's state after a mid-turn failure is unknown (a pi
            # that never acked its prompt may still answer it later), so the
            # process is retired rather than reused for the next request.
            self._retire(slot)
            yield AgentEvent(kind=EventKind.ERROR, error=f"agent error: {exc}")
            return
        finally:
            active.finished.set()
            with self._lock:
                if self._active is active:
                    self._active = None

        closing = self._closing_event(active, saw_terminal)
        if closing is not None:
            yield closing
        self._last_activity = time.monotonic()

    def _record_event(
        self,
        event: AgentEvent,
        turn: dict,
        conversation: Conversation,
        active: _ActiveTurn,
    ) -> None:
        """Fold one streamed event into the transcript and the dialog state."""
        if event.kind is EventKind.TEXT_DELTA and event.text:
            turn["text"] += event.text
        elif event.kind is EventKind.PROPOSAL and event.proposal is not None:
            conversation.pending_proposal = {
                "command": event.proposal.command,
                "rationale": event.proposal.rationale,
                "kind": event.proposal.kind.value,
            }
            request_id = str(event.args.get("request_id") or "")
            if request_id:
                with self._lock:
                    active.pending_ui.append(request_id)

    def _closing_event(self, active: _ActiveTurn, saw_terminal: bool) -> AgentEvent | None:
        """What a turn owes the client after the adapter's stream ended.

        A turn the cap aborted says so; one that ended without a terminal
        event of its own gets a DONE; one that reported its own terminal
        event gets nothing more.
        """
        if active.aborted == "killed":
            return AgentEvent(kind=EventKind.ERROR, error="the agent turn was force-stopped")
        if active.aborted == "timeout":
            return AgentEvent(
                kind=EventKind.ERROR,
                error=(
                    f"the agent turn was aborted after {self.turn_timeout:g}s "
                    f"(daemon turn cap; raise ${TURN_TIMEOUT_ENV} if this backend "
                    "legitimately needs longer)"
                ),
            )
        if not saw_terminal:
            return AgentEvent(kind=EventKind.DONE)
        return None


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
