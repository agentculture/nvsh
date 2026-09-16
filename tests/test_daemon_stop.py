"""The daemon's ``kill`` control: a second press really ends a hung turn (t9).

A polite ``cancel`` only works when the harness honours it. These tests use
an agent whose ``run()`` ignores ``cancel()`` outright -- only
``force_stop()`` ends it -- and check that the ``kill`` control message:

1. releases the daemon's run lock quickly, so the request queued behind the
   hung turn runs, and runs on a *fresh* agent rather than the killed one;
2. is scoped to the calling shell: shell B cannot kill shell A's turn.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from nvsh import client_transport
from nvsh import daemon as daemon_mod
from nvsh.agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    NvshAgent,
    RequestKind,
)
from nvsh.config import Config

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="the daemon needs unix domain sockets"
)

#: Prompts starting with this make :class:`StubbornAgent` hang.
BLOCK = "block"


class StubbornAgent(NvshAgent):
    """A fake harness that ignores ``cancel()``; only ``force_stop()`` ends a turn."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.turn_started = threading.Event()
        self.killed = threading.Event()
        self.cancels = 0
        self.force_stops = 0
        self.closed = False
        self.prompts: list[str] = []

    def start(self) -> None:
        return None

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self.prompts.append(request.prompt)
        if not request.prompt.startswith(BLOCK):
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text=f"{self.name}: {request.prompt}")
            yield AgentEvent(kind=EventKind.DONE)
            return
        self.turn_started.set()
        # cancel() is deliberately not consulted: this is the harness that
        # ignores a polite stop.
        self.killed.wait(30.0)

    def cancel(self) -> None:
        self.cancels += 1

    def force_stop(self) -> None:
        self.force_stops += 1
        self.killed.set()

    def close(self) -> None:
        self.closed = True

    def capabilities(self) -> Capabilities:
        return Capabilities(persistent_session=True)


def _env(tmp_path: Path) -> dict[str, str]:
    run = tmp_path / "run"
    state = tmp_path / "state"
    run.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    return {
        "XDG_RUNTIME_DIR": str(run),
        "XDG_STATE_HOME": str(state),
        "HOME": str(tmp_path),
    }


def _start(daemon: daemon_mod.Daemon) -> None:
    threading.Thread(target=daemon.serve_forever, daemon=True).start()
    for _ in range(500):
        if daemon_mod.is_running(daemon.env):
            return
        time.sleep(0.01)
    raise AssertionError("daemon never created its socket")


def _request(prompt: str) -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, prompt=prompt, command="ls /nope", exit_code=2)


def _send_in_background(
    env: dict[str, str], shell: str, prompt: str
) -> tuple[threading.Thread, list[AgentEvent], list[float]]:
    events: list[AgentEvent] = []
    finished: list[float] = []

    def go() -> None:
        events.extend(
            client_transport.send(
                _request(prompt), shell_id=shell, env=env, autostart=False, timeout=30.0
            )
        )
        finished.append(time.monotonic())

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    return thread, events, finished


def _factory(agents: list[StubbornAgent]):
    def build() -> StubbornAgent:
        agent = StubbornAgent(f"agent{len(agents)}")
        agents.append(agent)
        return agent

    return build


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_kill_releases_the_run_lock_and_the_queued_request_runs_on_a_fresh_slot(
    tmp_path: Path,
) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    try:
        hung, hung_events, _ = _send_in_background(env, "A", f"{BLOCK} A")
        assert _wait_for(lambda: bool(agents) and agents[0].turn_started.is_set())

        queued, queued_events, queued_done = _send_in_background(env, "B", "B fails")
        assert _wait_for(lambda: bool(daemon.state()["queued"])), "B never queued"

        # A polite cancel is ignored by this harness: the turn keeps the lock.
        assert client_transport.cancel(shell_id="A", env=env)
        time.sleep(0.2)
        assert daemon.active_turn() is not None
        assert agents[0].cancels >= 1

        began = time.monotonic()
        assert client_transport.kill(shell_id="A", env=env) is True
        hung.join(5.0)
        queued.join(5.0)
        assert not queued.is_alive(), "the queued request never ran after the kill"
        released_after = queued_done[0] - began
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert agents[0].force_stops == 1
    assert released_after < 3.0, f"the run lock was held {released_after:.2f}s after kill"
    assert hung_events and hung_events[-1].kind in (EventKind.ERROR, EventKind.DONE)
    assert len(agents) == 2, "the killed slot was not replaced with a fresh agent"
    assert agents[1].prompts == ["B fails"]
    assert "B fails" not in agents[0].prompts
    assert any("agent1: B fails" in event.text for event in queued_events)
    assert queued_events[-1].kind is EventKind.DONE
    assert daemon._run_lock.acquire(blocking=False)
    daemon._run_lock.release()


def test_kill_from_another_shell_does_nothing_to_this_shells_turn(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    try:
        hung, _, _ = _send_in_background(env, "A", f"{BLOCK} A")
        assert _wait_for(lambda: bool(agents) and agents[0].turn_started.is_set())

        assert client_transport.kill(shell_id="B", env=env) is False
        time.sleep(0.3)

        turn = daemon.active_turn()
        assert turn is not None and turn["shell"] == "A"
        assert agents[0].force_stops == 0
        assert agents[0].cancels == 0
        assert not agents[0].closed
        assert len(daemon._slots) == 1 and daemon._slots[0].agent is agents[0]
        assert hung.is_alive()
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()


def test_kill_is_a_control_message_that_never_queues() -> None:
    assert "kill" in daemon_mod._CONTROL_KINDS


def test_kill_without_a_daemon_returns_false(tmp_path: Path) -> None:
    assert client_transport.kill(shell_id="A", env=_env(tmp_path)) is False
