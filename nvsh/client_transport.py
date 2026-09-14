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
from typing import Iterator, Mapping, cast

from . import __version__
from . import daemon as _daemon
from .agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    EventKind,
    Target,
    event_from_dict,
    request_to_dict,
)
from .config import DEFAULT_ALIAS, Config

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

#: How long to wait for a stale daemon's socket to go away after ``stop``.
_STOP_TIMEOUT = 5.0

daemon_socket_path = _daemon.socket_path


class Responder:
    """Where the operator's answer to a proposal dialog goes.

    A proposal that carries a ``request_id`` was raised by the backend
    itself (pi's approval extension), and that backend stays blocked until
    the dialog is answered -- so the answer has to reach *the agent that
    asked*. Which agent that is depends on how the request was served, and
    the client cannot know up front: it asks the daemon, and only mid-stream
    may it learn there is none and that a one-shot agent is running in this
    very process (deviation d11, where the answer went to an absent daemon
    and pi waited forever while the panel's Enter appeared to do nothing).

    So this object is deliberately mutable: it starts out pointed at the
    daemon socket, and :func:`one_shot` rebinds it to the live in-process
    agent the moment it builds one.
    """

    def __init__(
        self,
        *,
        shell_id: str | int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.shell_id = shell_id
        self.env = env
        self.agent: object | None = None

    def bind_agent(self, agent: object) -> None:
        """Point answers at ``agent`` -- the one-shot, in-process backend."""
        self.agent = agent

    def steer(self, text: str) -> bool:
        """Inject ``text`` into the turn that is running now (deviation d16).

        Returns ``True`` only when a backend really took it mid-turn. A
        ``False`` is not a failure: most adapters have no mid-turn channel,
        and the caller then sends the text as the next request in the same
        conversation. Like :meth:`respond`, this follows the agent -- the
        in-process one when :func:`one_shot` bound it, the daemon otherwise.
        """
        if not text:
            return False
        agent = self.agent
        if agent is None:
            return steer(text, shell_id=self.shell_id, env=self.env)
        send_steer = getattr(agent, "steer", None)
        if not callable(send_steer):
            return False
        try:
            return bool(send_steer(text))
        except Exception:  # noqa: BLE001 - a dead backend must not break the panel
            return False

    def respond(self, request_id: str, fields: Mapping[str, object]) -> bool:
        """Answer dialog ``request_id``. Never raises on the failure path."""
        if not request_id:
            return False
        agent = self.agent
        if agent is None:
            return respond_ui(request_id, fields, shell_id=self.shell_id, env=self.env)
        respond = getattr(agent, "respond_ui", None)
        if not callable(respond):
            return False
        try:
            respond(request_id, **dict(fields))
        except Exception:  # noqa: BLE001 - a dead backend must not break the panel
            return False
        return True


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


def targeted_config(cfg: Config, target: Target) -> Config:
    """A copy of *cfg* whose ``[agents.<backend>]`` carries *target*.

    The mirror of ``Daemon._config_for``: every adapter factory reads its
    own table for ``model``/``effort``/``extra_args``/``approval``, so this
    is how a resolved target's model and effort reach the adapter without
    the caller knowing which keyword each adapter takes.
    """
    from dataclasses import replace

    settings = dict(cfg.agents.get(target.backend, {}))
    if target.model:
        settings["model"] = target.model
    if target.effort:
        settings["effort"] = target.effort
    agents = dict(cfg.agents)
    agents[target.backend] = settings
    return cast(Config, replace(cfg, agent_provider=target.backend, agents=agents))


def one_shot(
    request: AgentRequest,
    context: AgentContext | None = None,
    *,
    config: Config | None = None,
    responder: Responder | None = None,
) -> Iterator[AgentEvent]:
    """Run *request* in this process, with no daemon, and close the adapter.

    Used when the daemon is missing or crashed, and -- always -- when the
    request names a target other than ``default`` (decision c25).  Slower (a
    cold backend per call) but it always produces something.

    ``responder``, when given, is bound to the live adapter before the first
    event is yielded, so a caller answering a dialog raised during this run
    talks to *this* agent rather than to a daemon that is not there (d11).
    """
    from .agent import registry

    cfg = config if config is not None else _load_config()
    target = request.target
    try:
        if target is not None:
            # A named target is *forced*: a missing binary is a loud error,
            # never a silent swap for openai-compat. The operator asked for
            # that harness on purpose.
            cfg = targeted_config(cfg, target)
            name, reason = registry.choose(cfg, forced=target)
        else:
            name, reason = registry.choose(cfg)
        agent = registry.ADAPTERS[name].factory(cfg)
    except Exception as exc:  # noqa: BLE001 - reported rather than raised on the failure path
        yield AgentEvent(kind=EventKind.ERROR, error=f"no agent available: {exc}")
        return

    if responder is not None:
        responder.bind_agent(agent)
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


def daemon_version(env: Mapping[str, str] | None = None) -> str | None:
    """The ``nvsh`` version of the daemon that is listening, if one is.

    ``None`` means "no daemon answered"; ``""`` means one answered but named
    no version -- a daemon from before the handshake existed, which is by
    definition not this version.
    """
    state = status(env=env)
    if not state.get("running"):
        return None
    raw = state.get("version")
    return raw if isinstance(raw, str) else ""


def _wait_socket_gone(env: Mapping[str, str] | None, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _daemon.is_running(env) and not _daemon.lock_held(env):
            return True
        time.sleep(0.02)
    return not _daemon.is_running(env)


def _retire_stale_daemon(env: Mapping[str, str] | None) -> str:
    """Stop a listening daemon built from a different nvsh. Returns a notice.

    The half of the handshake the client owns. A daemon is a long-lived
    process started lazily and kept for 15 minutes, so an ``nvsh`` upgrade
    (or a ``uv sync`` in a checkout) routinely leaves one running that
    predates the code now talking to it -- adapters, wire fields and
    protocol alike. Rather than guess which differences are survivable, the
    client stops it; the very next ``_connect``/``_autostart`` starts one
    from the wheel this process is running. ``""`` when nothing was done.
    """
    running = daemon_version(env)
    if running is None or running == __version__:
        return ""
    stop(env=env)
    _wait_socket_gone(env, _STOP_TIMEOUT)
    return (
        f"restarted the daemon: it was running nvsh {running or 'an older build'}, "
        f"this client is {__version__}"
    )


def _payload(
    request: AgentRequest,
    context: AgentContext | None,
    shell_id: str | int | None,
) -> dict:
    """One request line: the shell, the version handshake, request, context."""
    return {
        "shell": _shell_id(shell_id),
        "kind": request.kind.value,
        _daemon.VERSION_KEY: __version__,
        "request": request_to_dict(request),
        "context": asdict(context) if context is not None else {},
    }


def is_default_target(target: Target | None) -> bool:
    """Is *target* the warm daemon session's own target (decision c25)?

    ``None`` (the caller said nothing) and the resolved ``default`` alias
    both are; every other target runs one-shot, because the warm session
    holds a conversation with the *default* backend and serving someone
    else's ``@claude/opus`` out of it would answer from the wrong model.
    """
    return target is None or target.alias == DEFAULT_ALIAS


def send(
    request: AgentRequest,
    context: AgentContext | None = None,
    *,
    shell_id: str | int | None = None,
    env: Mapping[str, str] | None = None,
    config: Config | None = None,
    autostart: bool = True,
    timeout: float = _DEFAULT_TIMEOUT,
    responder: Responder | None = None,
) -> Iterator[AgentEvent]:
    """Stream the daemon's events for one request, falling back to one-shot.

    ``shell_id`` defaults to this process's pid -- the hook passes ``$$`` so
    every terminal keeps its own conversation. ``timeout`` is the *stream*
    timeout (how long one read may wait for the next event), not the connect
    or autostart wait: those are :data:`_CONNECT_TIMEOUT` and
    :func:`start_timeout`. A fallback to one-shot is always preceded by a
    ``status`` event naming the reason.

    A request naming a non-default target never reaches the daemon at all
    (:func:`is_default_target`, decision c25), and a daemon left over from a
    different nvsh is stopped and replaced before the request goes out
    (:func:`_retire_stale_daemon`).
    """
    if not is_default_target(request.target):
        yield from one_shot(request, context, config=config, responder=responder)
        return

    payload = _payload(request, context, shell_id)

    notice = _retire_stale_daemon(env)
    if notice:
        yield AgentEvent(kind=EventKind.STATUS, text=notice)

    sock = _connect(env, _CONNECT_TIMEOUT)
    reason = ""
    if sock is None and autostart:
        sock, reason = _autostart(env, start_timeout(env))
    if sock is None:
        if reason:  # say why the warm session was given up on
            yield AgentEvent(kind=EventKind.STATUS, text=reason)
        yield from one_shot(request, context, config=config, responder=responder)
        return

    # The connect wait is over; from here the stream may be idle for as long
    # as a cold backend needs to say its first word.
    sock.settimeout(timeout)
    try:
        events = _stream(sock, payload)
        first = next(events, None)
        if first is not None and _is_version_mismatch(first):
            # The probe above raced an upgrade, or a daemon came up between
            # it and the connect. Same cure, one retry deep.
            yield from _retry_after_mismatch(
                first, payload, request, context, env, config, responder, timeout, autostart
            )
            return
        if first is not None:
            yield first
            if first.kind not in (EventKind.DONE, EventKind.ERROR):
                yield from events
    except OSError as exc:
        yield AgentEvent(kind=EventKind.STATUS, text=f"daemon connection lost: {exc}")
        yield from one_shot(request, context, config=config, responder=responder)


def _is_version_mismatch(event: AgentEvent) -> bool:
    """Is *event* the daemon's "I am a different nvsh" answer?

    Read off ``args``, not the prose: the message is for the operator, the
    flag is for this code.
    """
    return event.kind is EventKind.ERROR and bool((event.args or {}).get("version_mismatch"))


def _retry_after_mismatch(
    event: AgentEvent,
    payload: dict,
    request: AgentRequest,
    context: AgentContext | None,
    env: Mapping[str, str] | None,
    config: Config | None,
    responder: Responder | None,
    timeout: float,
    autostart: bool,
) -> Iterator[AgentEvent]:
    """Stop the mismatched daemon, start a fresh one, and send again. Once."""
    running = str((event.args or {}).get("daemon_version") or "an older build")
    yield AgentEvent(
        kind=EventKind.STATUS,
        text=(f"restarted the daemon: it was running nvsh {running}, this client is {__version__}"),
    )
    stop(env=env)
    _wait_socket_gone(env, _STOP_TIMEOUT)
    sock = _connect(env, _CONNECT_TIMEOUT)
    reason = ""
    if sock is None and autostart:
        sock, reason = _autostart(env, start_timeout(env))
    if sock is None:
        if reason:
            yield AgentEvent(kind=EventKind.STATUS, text=reason)
        yield from one_shot(request, context, config=config, responder=responder)
        return
    sock.settimeout(timeout)
    try:
        for retried in _stream(sock, payload):
            if _is_version_mismatch(retried):
                # Twice is not a race, it is a broken install: say so and
                # answer in this process rather than loop.
                yield AgentEvent(kind=EventKind.STATUS, text=retried.error)
                yield from one_shot(request, context, config=config, responder=responder)
                return
            yield retried
    except OSError as exc:
        yield AgentEvent(kind=EventKind.STATUS, text=f"daemon connection lost: {exc}")
        yield from one_shot(request, context, config=config, responder=responder)


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
    payload = {
        "shell": _shell_id(shell_id),
        "kind": kind,
        _daemon.VERSION_KEY: __version__,
    } | dict(extra)
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


def steer(
    text: str,
    *,
    shell_id: str | int | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Ask a running daemon to steer this shell's in-flight turn.

    ``False`` when no daemon is listening, when this shell has no turn
    running, or when its backend has no mid-turn channel -- every one of
    which means the same thing to the caller: send the text as the next
    request instead.
    """
    events = control("steer", shell_id=shell_id, env=env, text=text)
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
