"""Process-lifecycle tests for the session daemon: real (fake) pi processes.

These drive the daemon with the ``tests/fakes/pi`` stub on PATH, so a real
subprocess is spawned and can be counted. The stub writes ``<pid>.pid`` into
``$FAKE_PI_PIDDIR`` at start and removes it at exit, which is more reliable
than ``pgrep -f`` under ``pytest -n auto`` (several tests' fakes run at
once). Every test keeps its own budget under ~2 s.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

import pytest

from nvsh import client_transport
from nvsh import daemon as daemon_mod
from nvsh.agent.base import AgentEvent, AgentRequest, EventKind, RequestKind
from nvsh.config import Config

FAKES = Path(__file__).parent / "fakes"

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="the daemon needs unix domain sockets"
)


def _env(tmp_path: Path) -> dict[str, str]:
    run = tmp_path / "run"
    state = tmp_path / "state"
    piddir = tmp_path / "pids"
    for directory in (run, state, piddir):
        directory.mkdir(exist_ok=True)
    return {
        "PATH": f"{FAKES}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "XDG_RUNTIME_DIR": str(run),
        "XDG_STATE_HOME": str(state),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "FAKE_PI_PIDDIR": str(piddir),
        "FAKE_PI_CMDLOG": str(tmp_path / "pi-commands.jsonl"),
        "PYTHONPATH": str(Path(__file__).parent.parent),
    }


def live_pis(env: dict[str, str]) -> list[int]:
    piddir = Path(env["FAKE_PI_PIDDIR"])
    alive: list[int] = []
    for entry in sorted(piddir.glob("*.pid")):
        try:
            pid = int(entry.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        alive.append(pid)
    return alive


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def pi_commands(env: dict[str, str]) -> list[str]:
    path = Path(env["FAKE_PI_CMDLOG"])
    if not path.is_file():
        return []
    return [
        json.loads(line).get("type", "")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _failure(prompt: str) -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, prompt=prompt, command="ls /nope", exit_code=2)


def _drain(events: Iterator[AgentEvent]) -> list[AgentEvent]:
    return list(events)


def _launch_daemon(env: dict[str, str], *, idle_timeout: float = 30.0) -> subprocess.Popen:
    """Start ``python -m nvsh.daemon --foreground`` without waiting for it."""
    return subprocess.Popen(  # nosec B603 - fixed argv
        [
            sys.executable,
            "-m",
            "nvsh.daemon",
            "--foreground",
            "--idle-timeout",
            str(idle_timeout),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _spawn_daemon(env: dict[str, str], *, idle_timeout: float = 30.0) -> subprocess.Popen:
    proc = _launch_daemon(env, idle_timeout=idle_timeout)
    assert wait_for(lambda: daemon_mod.is_running(env), 10.0), "daemon never listened"
    return proc


def _kill(*procs: subprocess.Popen) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def _slow_pi(tmp_path: Path, delay: float) -> Path:
    """Write a ``pi`` stub that takes *delay* seconds to come up.

    Stands in for a cold backend (pi loading node, a model warming up): the
    process exists at once but says nothing for *delay* seconds. Returns the
    directory to put in front of ``PATH``.
    """
    bindir = tmp_path / "slowbin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "pi"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        f"time.sleep({delay!r})\n"
        f"os.execv(sys.executable, [sys.executable, {str(FAKES / 'pi')!r}] + sys.argv[1:])\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bindir


def _stub_one_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the one-shot fallback cheap, loud and offline.

    The real fallback reaches the operator's own configured backend
    (``one_shot`` loads config from the process environment), which is
    neither deterministic nor offline under CI.
    """

    def fake_one_shot(request, context=None, **kwargs):
        yield AgentEvent(kind=EventKind.STATUS, text="one-shot stub")
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "one_shot", fake_one_shot)


def _statuses(events: list[AgentEvent]) -> list[str]:
    return [event.text for event in events if event.kind is EventKind.STATUS]


# --- daemon process lifecycle ---------------------------------------------


def test_daemon_starts_pi_lazily_and_the_last_shell_stops_both(tmp_path: Path) -> None:
    env = _env(tmp_path)
    proc = _spawn_daemon(env)
    try:
        assert live_pis(env) == [], "no pi process before the first request"
        client_transport.register(shell_id="1", env=env)
        assert live_pis(env) == [], "registering a shell must not start pi"

        _drain(client_transport.send(_failure("boom"), shell_id="1", env=env, autostart=False))
        assert wait_for(lambda: len(live_pis(env)) == 1), "the failure must start exactly one pi"

        client_transport.unregister(shell_id="1", env=env)
        assert wait_for(lambda: proc.poll() is not None), "the last shell must stop the daemon"
        assert wait_for(lambda: live_pis(env) == []), "pi must die with the daemon"
        assert not daemon_mod.socket_path(env).exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_idle_timeout_stops_the_daemon_and_pi(tmp_path: Path) -> None:
    env = _env(tmp_path)
    proc = _spawn_daemon(env, idle_timeout=1.0)
    try:
        _drain(client_transport.send(_failure("boom"), shell_id="1", env=env, autostart=False))
        assert wait_for(lambda: len(live_pis(env)) == 1)
        assert wait_for(lambda: proc.poll() is not None, 8.0), "idle timeout must stop the daemon"
        assert wait_for(lambda: live_pis(env) == [])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_first_failure_autostarts_exactly_one_daemon(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert not daemon_mod.socket_path(env).exists()
    try:
        _drain(client_transport.send(_failure("boom"), shell_id="1", env=env))
        status = client_transport.status(env=env)
        assert status["running"] is True
        assert wait_for(lambda: len(live_pis(env)) == 1)
        _drain(client_transport.send(_failure("again"), shell_id="1", env=env))
        assert live_pis(env) and len(live_pis(env)) == 1, "one daemon, one pi"
    finally:
        client_transport.stop(env=env)
        wait_for(lambda: live_pis(env) == [])


# --- conversations over a real pi subprocess ------------------------------


def test_one_pi_serves_two_shells_by_swapping_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    daemon = daemon_mod.Daemon(Config(agent_provider="pi", sessions_max=1), env=env)
    thread = daemon.start_background()
    try:
        _drain(client_transport.send(_failure("A fails"), shell_id="A", env=env, autostart=False))
        _drain(client_transport.send(_failure("B fails"), shell_id="B", env=env, autostart=False))
        assert len(live_pis(env)) == 1, "sessions.max=1 means one pi process throughout"
        _drain(
            client_transport.send(
                AgentRequest(kind=RequestKind.SLASH, prompt="what did you just see"),
                shell_id="A",
                env=env,
                autostart=False,
            )
        )
        assert len(live_pis(env)) == 1
    finally:
        daemon.shutdown()
        thread.join(timeout=5)
    commands = pi_commands(env)
    assert commands.count("new_session") == 2
    assert "switch_session" in commands
    assert wait_for(lambda: live_pis(env) == [])


def test_sessions_max_two_starts_a_second_pi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    daemon = daemon_mod.Daemon(Config(agent_provider="pi", sessions_max=2), env=env)
    thread = daemon.start_background()
    try:
        _drain(client_transport.send(_failure("A fails"), shell_id="A", env=env, autostart=False))
        _drain(client_transport.send(_failure("B fails"), shell_id="B", env=env, autostart=False))
        assert wait_for(lambda: len(live_pis(env)) == 2), "sessions.max=2 allows a second process"
    finally:
        daemon.shutdown()
        thread.join(timeout=5)
    assert "switch_session" not in pi_commands(env)
    assert wait_for(lambda: live_pis(env) == [])


# --- autostart: wait for an answer, start exactly one daemon, say why -----
#
# Regression tests for deviation d7 (docs/verification.md, row v7): after
# `nvsh daemon stop`, the first qualifying failure printed
# `daemon connection lost: timed out` and answered one-shot, and several
# orphan daemons accumulated against one socket.


def test_autostart_waits_for_a_cold_backend_instead_of_falling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend slower than the connect wait must still be served warm."""
    env = _env(tmp_path)
    env["PATH"] = f"{_slow_pi(tmp_path, 6.0)}:{env['PATH']}"
    _stub_one_shot(monkeypatch)
    try:
        events = _drain(client_transport.send(_failure("boom"), shell_id="1", env=env))
        texts = _statuses(events)
        assert not any("one-shot" in text for text in texts), f"fell back: {texts}"
        assert not any("connection lost" in text for text in texts), f"gave up: {texts}"
        state = client_transport.status(env=env)
        assert state["running"] is True
        assert state["conversations"]["1"]["requests"] == 1, "the daemon must have served it"
    finally:
        client_transport.stop(env=env)
        wait_for(lambda: live_pis(env) == [])


def test_a_second_daemon_refuses_while_another_holds_the_lock(tmp_path: Path) -> None:
    """Losing the single-instance lock must exit, not orphan the live daemon."""
    env = _env(tmp_path)
    first = _spawn_daemon(env)
    # What a client sees mid-flight if the socket file goes missing: the old
    # code simply bound a fresh socket and left `first` orphaned.
    daemon_mod.socket_path(env).unlink()
    second = _launch_daemon(env)
    try:
        try:
            returncode = second.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail("a second daemon kept running while another held the lock")
        assert returncode != 0, "the loser must exit non-zero"
        assert first.poll() is None, "the live daemon must survive"
    finally:
        _kill(first, second)


def test_autostart_never_starts_a_rival_daemon_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a daemon already holding the lock, autostart must not spawn one."""
    env = _env(tmp_path)
    env["NVSH_DAEMON_START_TIMEOUT"] = "1.0"
    first = _spawn_daemon(env)
    sock = daemon_mod.socket_path(env)
    try:
        sock.unlink()
        _stub_one_shot(monkeypatch)
        events = _drain(client_transport.send(_failure("boom"), shell_id="1", env=env))
        texts = _statuses(events)
        assert not sock.exists(), "a rival daemon bound a fresh socket"
        assert any("did not answer within 1s" in text for text in texts), texts
        assert any("one-shot stub" in text for text in texts), texts
        assert first.poll() is None
    finally:
        _kill(first)


# --- d14: a daemon-routed pi turn answers, and failures are never silent ---
#
# On the Spark, every request routed through the daemon produced no event at
# all: the daemon logged "new conversation for shell N" and then nothing, the
# client waited out its 120 s stream timeout and answered one-shot. The cause
# was pipelining -- the daemon wrote new_session and then the prompt without
# waiting for pi's acknowledgement of the first, and pi 0.85.1 answers
# neither. ``FAKE_PI_STRICT_ACK=1`` makes tests/fakes/pi behave the same way,
# so this test is red against the pipelining daemon and green against the
# acknowledged one.


def test_daemon_routed_request_answers_quickly_and_records_pis_session_file(
    tmp_path: Path,
) -> None:
    env = _env(tmp_path)
    env["FAKE_PI_STRICT_ACK"] = "1"
    proc = _spawn_daemon(env)
    try:
        start = time.monotonic()
        events = _drain(
            client_transport.send(_failure("boom"), shell_id="7", env=env, autostart=False)
        )
        elapsed = time.monotonic() - start
        kinds = [event.kind for event in events]
        assert EventKind.TEXT_DELTA in kinds, f"the daemon answered nothing: {events}"
        assert elapsed < 2.0, f"first answer took {elapsed:.2f}s"

        state = client_transport.status(env=env)
        session_path = state["conversations"]["7"]["session_path"]
        assert session_path, "the shell's conversation must know its session file"
        assert Path(session_path).is_file(), "pi's session file must exist on disk"
        assert Path(session_path).parent == Path(env["XDG_STATE_HOME"]) / "nvsh" / "pi-sessions"
    finally:
        _kill(proc)
        wait_for(lambda: live_pis(env) == [])


def test_daemon_reports_a_pi_that_exits_at_once_as_an_error(tmp_path: Path) -> None:
    """A backend that dies on launch is an error event, not a silent wait."""
    env = _env(tmp_path)
    bindir = tmp_path / "deadbin"
    bindir.mkdir()
    stub = bindir / "pi"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stderr.write('pi: cannot find module foo\\n')\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    proc = _spawn_daemon(env)
    try:
        start = time.monotonic()
        events = _drain(
            client_transport.send(_failure("boom"), shell_id="1", env=env, autostart=False)
        )
        elapsed = time.monotonic() - start
        errors = [event.error for event in events if event.kind is EventKind.ERROR]
        assert errors, f"a dead backend must produce an error event: {events}"
        assert elapsed < 2.0, f"the error took {elapsed:.2f}s"
        assert "code 3" in errors[-1], errors
        assert "cannot find module foo" in errors[-1], errors
    finally:
        _kill(proc)


def test_fallback_reports_that_the_daemon_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-shot fallback must name the reason, not just happen."""
    env = _env(tmp_path)
    env["NVSH_DAEMON_START_TIMEOUT"] = "0.5"
    monkeypatch.setattr(daemon_mod, "spawn", lambda *a, **k: 0)  # nothing ever starts
    _stub_one_shot(monkeypatch)
    events = _drain(client_transport.send(_failure("boom"), shell_id="1", env=env))
    texts = _statuses(events)
    assert any("daemon did not start within 0.5s" in text for text in texts), texts
    assert any("one-shot stub" in text for text in texts), texts
    assert not daemon_mod.socket_path(env).exists()
