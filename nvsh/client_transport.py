"""The hook client's transport: talk to the session daemon, or fall back.

The failure path must never leave the operator without a diagnosis, so every
entry point here degrades instead of raising:

* the daemon answers -> its events are streamed through verbatim;
* no daemon (never started, or crashed) -> one is started in the background
  (``autostart``) and, if that still does not answer, the request runs
  **one-shot** in this process through :func:`nvsh.agent.registry.choose`;
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

#: How long to wait for the socket to appear after an autostart.
_AUTOSTART_WAIT = 5.0

#: Socket-level timeout for one request/stream.
_DEFAULT_TIMEOUT = 120.0

daemon_socket_path = _daemon.socket_path


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


def _wait_for_socket(env: Mapping[str, str] | None, timeout: float) -> socket.socket | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sock = _connect(env, 5.0)
        if sock is not None:
            return sock
        time.sleep(0.02)
    return None


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
    every terminal keeps its own conversation.
    """
    payload = {
        "shell": _shell_id(shell_id),
        "kind": request.kind.value,
        "request": asdict(request) | {"kind": request.kind.value},
        "context": asdict(context) if context is not None else {},
    }

    sock = _connect(env, timeout)
    if sock is None and autostart:
        try:
            _daemon.spawn(env)
        except OSError:
            sock = None
        else:
            sock = _wait_for_socket(env, _AUTOSTART_WAIT)
    if sock is None:
        yield from one_shot(request, context, config=config)
        return

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
