"""The hook client's transport: talk to the session daemon, or fall back.

The failure path must never leave the operator without a diagnosis, so every
entry point here degrades instead of raising:

* the daemon answers -> its events are streamed through verbatim;
* no daemon (never started, or crashed) -> one is started in the background
  (``autostart``, but never a second one while another holds the lock) and
  waited for until it *answers*; only if it still does not does the request
  run **one-shot** in this process through
  :func:`nvsh.agent.registry.choose`, preceded by a ``status`` event naming
  the reason (``daemon did not start within 10s`` and friends);
* nothing works at all -> a single ``error`` event describing why.

Stdlib only. Importing this module starts no process and creates no file --
starting a shell must cost nothing (the hook only calls ``nvsh`` on a
qualifying failure; see ``tests/test_hook_bash.py``).
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Mapping

from . import daemon as _daemon
from .agent.base import AgentContext, AgentEvent, AgentRequest, EventKind, event_from_dict
from .config import Config

#: How long a ``connect()`` on an existing socket may take.
_CONNECT_TIMEOUT = 5.0

#: Socket-level read timeout while streaming one request's events. Separate
#: from the connect wait on purpose: a cold backend (pi loading node, a model
#: warming up) can take many seconds to produce its first token, and the
#: client must keep listening rather than declare the daemon lost.
_DEFAULT_TIMEOUT = 120.0

#: How long an autostart waits for the daemon to accept *and answer*.
_DEFAULT_START_TIMEOUT = 10.0

#: Environment override for that bound (seconds).
START_TIMEOUT_ENV = "NVSH_DAEMON_START_TIMEOUT"

daemon_socket_path = _daemon.socket_path


def _resolve_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def _shell_id(shell_id: str | int | None) -> str:
    return str(shell_id) if shell_id is not None else str(os.getpid())


def _connect(env: Mapping[str, str] | None, timeout: float) -> socket.socket | None:
    path: Path = _daemon.socket_path(env)
    if not path.exists():
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect(str(path))
    except OSError:
        sock.close()
        return None
    return sock


def start_timeout(env: Mapping[str, str] | None = None) -> float:
    """How long an autostart may wait for the daemon, in seconds.

    ``$NVSH_DAEMON_START_TIMEOUT`` overrides the default, so an operator on
    a slow Jetson (or one pointing pi at a cold remote model) can give the
    daemon more room without editing code. Junk or non-positive values fall
    back to the default rather than failing on the failure path.
    """
    resolved = _resolve_env(env)
    raw = resolved.get(START_TIMEOUT_ENV, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_START_TIMEOUT
    return value if value > 0 else _DEFAULT_START_TIMEOUT


def _ping(env: Mapping[str, str] | None, timeout: float) -> bool:
    """Did a daemon accept a connection **and answer on it**?

    A socket file that exists, or even a ``connect()`` that succeeds, only
    proves something bound the path -- a daemon still inside its imports has
    both. Only a reply proves the accept loop is running.
    """
    sock = _connect(env, timeout)
    if sock is None:
        return False
    try:
        for event in _stream(sock, {"shell": "-", "kind": "ping"}):
            if event.kind in (EventKind.STATUS, EventKind.DONE):
                return True
    except OSError:
        return False
    return False


def _wait_for_daemon(env: Mapping[str, str] | None, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        left = deadline - time.monotonic()
        if _ping(env, min(_CONNECT_TIMEOUT, max(0.1, left))):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _autostart(env: Mapping[str, str] | None, timeout: float) -> tuple[socket.socket | None, str]:
    """Start a daemon if none holds the lock, then wait for it to answer.

    Returns ``(socket, "")`` on success, or ``(None, reason)`` -- a reason
    the operator can act on, never a bare fallback.
    """
    already = _daemon.lock_held(env)
    if not already:
        try:
            _daemon.spawn(env)
        except OSError as exc:
            return None, f"could not start a daemon: {exc}"
    if not _wait_for_daemon(env, timeout):
        if already:
            return None, f"daemon did not answer within {timeout:g}s"
        return None, f"daemon did not start within {timeout:g}s"
    sock = _connect(env, _CONNECT_TIMEOUT)
    if sock is None:
        return None, "daemon refused the connection"
    return sock, ""


def _stream(sock: socket.socket, payload: dict) -> Iterator[AgentEvent]:
    with sock:
        stream = sock.makefile("rwb")
        stream.write((json.dumps(payload) + "\n").encode("utf-8"))
        stream.flush()
        for raw in stream:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = event_from_dict(json.loads(line))
            except ValueError:
                event = AgentEvent(kind=EventKind.ERROR, error=f"bad event line: {line[:200]}")
            yield event
            if event.kind in (EventKind.DONE, EventKind.ERROR):
                return


def one_shot(
    request: AgentRequest,
    context: AgentContext | None = None,
    *,
    config: Config | None = None,
) -> Iterator[AgentEvent]:
    """Run *request* in this process, with no daemon, and close the adapter.

    Used when the daemon is missing or crashed. Slower (a cold backend per
    call) but it always produces something.
    """
    from .agent import registry

    cfg = config if config is not None else _load_config()
    try:
        name, reason = registry.choose(cfg)
        agent = registry.ADAPTERS[name].factory(cfg)
    except Exception as exc:  # noqa: BLE001 - report, never raise on the failure path
        yield AgentEvent(kind=EventKind.ERROR, error=f"no agent available: {exc}")
        return

    yield AgentEvent(kind=EventKind.STATUS, text=f"one-shot {name}: {reason}")
    ctx = context if context is not None else AgentContext()
    saw_terminal = False
    try:
        agent.start()
        for event in agent.run(request, ctx):
            saw_terminal = event.kind in (EventKind.DONE, EventKind.ERROR)
            yield event
            if saw_terminal:
                break
    except Exception as exc:  # noqa: BLE001
        yield AgentEvent(kind=EventKind.ERROR, error=f"agent error: {exc}")
        return
    finally:
        with contextlib.suppress(Exception):  # teardown must not mask the answer
            agent.close()
    if not saw_terminal:
        yield AgentEvent(kind=EventKind.DONE)


def _load_config() -> Config:
    from .config import load

    try:
        return load()
    except Exception:  # noqa: BLE001 - a broken config must not block diagnosis
        return Config()


def send(
    request: AgentRequest,
    context: AgentContext | None = None,
    *,
    shell_id: str | int | None = None,
    env: Mapping[str, str] | None = None,
    config: Config | None = None,
    autostart: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Iterator[AgentEvent]:
    """Stream the daemon's events for one request, falling back to one-shot.

    ``shell_id`` defaults to this process's pid -- the hook passes ``$$`` so
    every terminal keeps its own conversation. ``timeout`` is the *stream*
    timeout (how long one read may wait for the next event), not the connect
    or autostart wait: those are :data:`_CONNECT_TIMEOUT` and
    :func:`start_timeout`. A fallback to one-shot is always preceded by a
    ``status`` event naming the reason.
    """
    payload = {
        "shell": _shell_id(shell_id),
        "kind": request.kind.value,
        "request": asdict(request) | {"kind": request.kind.value},
        "context": asdict(context) if context is not None else {},
    }

    sock = _connect(env, _CONNECT_TIMEOUT)
    reason = ""
    if sock is None and autostart:
        sock, reason = _autostart(env, start_timeout(env))
    if sock is None:
        if reason:  # say why the warm session was given up on
            yield AgentEvent(kind=EventKind.STATUS, text=reason)
        yield from one_shot(request, context, config=config)
        return

    # The connect wait is over; from here the stream may be idle for as long
    # as a cold backend needs to say its first word.
    sock.settimeout(timeout)
    try:
        yield from _stream(sock, payload)
    except OSError as exc:
        yield AgentEvent(kind=EventKind.STATUS, text=f"daemon connection lost: {exc}")
        yield from one_shot(request, context, config=config)


def control(
    kind: str,
    *,
    shell_id: str | int | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 5.0,
    **extra: object,
) -> list[AgentEvent]:
    """Send one control message (``register``/``unregister``/``stop``/...).

    Returns the events the daemon sent back, or an empty list when no daemon
    is listening -- control messages never start one.
    """
    sock = _connect(env, timeout)
    if sock is None:
        return []
    payload = {"shell": _shell_id(shell_id), "kind": kind} | dict(extra)
    try:
        return list(_stream(sock, payload))
    except OSError:
        return []


def register(*, shell_id: str | int | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Tell a running daemon this shell is alive. No daemon -> ``False``."""
    return bool(control("register", shell_id=shell_id, env=env))


def unregister(*, shell_id: str | int | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Tell the daemon this shell exited (the bash EXIT trap calls this)."""
    return bool(control("unregister", shell_id=shell_id, env=env))


def cancel(*, shell_id: str | int | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Ask the daemon to abort whatever it is streaming for this shell."""
    return bool(control("cancel", shell_id=shell_id, env=env))


def respond_ui(
    request_id: str,
    fields: Mapping[str, object] | None = None,
    *,
    shell_id: str | int | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Answer a pending approval dialog raised by a ``proposal`` event."""
    events = control(
        "ui_response",
        shell_id=shell_id,
        env=env,
        request_id=request_id,
        fields=dict(fields or {}),
    )
    return any(event.kind is EventKind.STATUS for event in events)


def stop(*, env: Mapping[str, str] | None = None) -> bool:
    """Stop a running daemon. Returns ``False`` when none was listening."""
    return bool(control("stop", env=env))


def status(*, env: Mapping[str, str] | None = None) -> dict:
    """Report the daemon's state, or ``{"running": False, ...}`` when it is down."""
    events = control("status", env=env)
    for event in events:
        if event.kind is EventKind.STATUS:
            try:
                return json.loads(event.text)
            except ValueError:  # pragma: no cover - daemon always sends JSON here
                break
    return {"running": False, "socket": str(_daemon.socket_path(env))}
