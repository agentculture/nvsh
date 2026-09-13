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
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import time
from pathlib import Path
from typing import Callable, Iterator

import pytest

from nvsh import client_transport
from nvsh import daemon as daemon_mod
from nvsh.agent.base import AgentEvent, AgentRequest, RequestKind
from nvsh.config import Config

FAKES = Path(__file__).parent / "fakes"


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


def _spawn_daemon(env: dict[str, str], *, idle_timeout: float = 30.0) -> subprocess.Popen:
    proc = subprocess.Popen(  # nosec B603 - fixed argv
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
    assert wait_for(lambda: daemon_mod.is_running(env), 10.0), "daemon never listened"
    return proc


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
