"""The daemon's ``kill`` control: a second press really ends a hung turn (t9).

A polite ``cancel`` only works when the harness honours it. These tests use
an agent whose ``run()`` ignores ``cancel()`` outright -- only
``force_stop()`` ends it -- and check that the ``kill`` control message:

1. releases the daemon's run lock quickly, so the request queued behind the
   hung turn runs, and runs on a *fresh* agent rather than the killed one;
2. is scoped to the calling shell: shell B cannot kill shell A's turn.
"""

from __future__ import annotations

import os
import socket
import subprocess  # nosec B404 - fixed argv
import sys
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


# --- the busy prompt (t10) ---------------------------------------------------
#
# Wire shape. When a request arrives from the shell that owns the running
# turn -- or from any shell once the owning shell's pid is gone -- the daemon
# streams one event before queueing:
#
#   {"kind": "busy", "text": "...",
#    "args": {"owner": "<shell>", "elapsed": <seconds>, "steerable": <bool>,
#             "choices": ["steer", "replace", "exit"]}}   # "steer" only if steerable
#
# and waits for a control message on a *new* connection from the same shell:
#
#   {"kind": "busy_choice", "shell": "<shell>", "choice": "steer|replace|exit"}
#
# steer  -> agent.steer(prompt); taken: STATUS + DONE; not taken: queue as next request
# replace-> force-stop the running turn, then run this request on a fresh agent
# exit   -> leave the running turn alone; STATUS + DONE(args={"busy_choice": "exit"})
#
# A client that ignores ``busy`` (an older one decodes it as a status line)
# is never stranded: after ``_BUSY_CHOICE_TIMEOUT`` the request queues as
# today, and a turn that ends while the prompt is open lets the request run.


class SteerableAgent(StubbornAgent):
    """A stubborn harness with a mid-turn channel (like pi or codex)."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.steered: list[str] = []

    def steer(self, text: str) -> bool:
        self.steered.append(text)
        return True


def _steerable_factory(agents: list[StubbornAgent]):
    def build() -> StubbornAgent:
        agent = SteerableAgent(f"agent{len(agents)}")
        agents.append(agent)
        return agent

    return build


def _collect_in_background(
    env: dict[str, str], shell: str, prompt: str
) -> tuple[threading.Thread, list[AgentEvent]]:
    events: list[AgentEvent] = []

    def go() -> None:
        for event in client_transport.send(
            _request(prompt), shell_id=shell, env=env, autostart=False, timeout=30.0
        ):
            events.append(event)

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    return thread, events


def _busy(events: list[AgentEvent]) -> list[AgentEvent]:
    return [event for event in events if event.kind is EventKind.BUSY]


def _dead_pid() -> str:
    child = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603 - fixed argv
    child.wait()
    return str(child.pid)


def _hang(env, agents, shell: str) -> threading.Thread:
    hung, _ = _collect_in_background(env, shell, f"{BLOCK} {shell}")
    assert _wait_for(lambda: bool(agents) and agents[0].turn_started.is_set())
    return hung


@pytest.mark.parametrize(("factory", "steerable"), [(_factory, False), (_steerable_factory, True)])
def test_owning_shell_gets_a_busy_event_and_exit_leaves_the_turn_running(
    tmp_path: Path, factory, steerable: bool
) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=factory(agents))
    _start(daemon)
    try:
        _hang(env, agents, "A")
        second, events = _collect_in_background(env, "A", "A again")
        assert _wait_for(lambda: bool(_busy(events))), "no busy event for the owning shell"
        busy = _busy(events)[0]
        assert busy.args["owner"] == "A"
        assert busy.args["steerable"] is steerable
        assert isinstance(busy.args["elapsed"], (int, float))
        assert ("steer" in busy.args["choices"]) is steerable
        assert {"replace", "exit"} <= set(busy.args["choices"])

        assert client_transport.busy_choice("exit", shell_id="A", env=env) is True
        second.join(5.0)
        assert not second.is_alive(), "exit did not end the request"
        assert events[-1].kind is EventKind.DONE
        assert events[-1].args.get("busy_choice") == "exit"

        turn = daemon.active_turn()
        assert turn is not None and turn["shell"] == "A"
        assert agents[0].force_stops == 0 and agents[0].cancels == 0
        assert agents[0].prompts == [f"{BLOCK} A"]
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()


def test_another_live_shell_queues_with_the_notice_and_never_sees_busy(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    try:
        _hang(env, agents, "A")
        queued, events = _collect_in_background(env, "B", "B fails")
        assert _wait_for(lambda: bool(daemon.state()["queued"])), "B never queued"
        time.sleep(0.3)
        assert not _busy(events)
        assert client_transport.busy_choice("replace", shell_id="B", env=env) is False
        agents[0].killed.set()
        queued.join(5.0)
        assert not queued.is_alive()
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert not _busy(events)
    assert any("busy with shell A" in event.text for event in events)
    assert any("agent0: B fails" in event.text for event in events)
    assert agents[0].force_stops == 0


def test_a_dead_owner_turn_offers_busy_and_replace_runs_on_a_fresh_agent(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    owner = _dead_pid()
    try:
        hung = _hang(env, agents, owner)
        second, events = _collect_in_background(env, "B", "B fails")
        assert _wait_for(lambda: bool(_busy(events))), "no busy event for a dead-owner turn"
        assert _busy(events)[0].args["owner"] == owner
        assert _busy(events)[0].args["steerable"] is False

        assert client_transport.busy_choice("steer", shell_id="B", env=env) is False
        assert client_transport.busy_choice("replace", shell_id="B", env=env) is True
        second.join(5.0)
        hung.join(5.0)
        assert not second.is_alive(), "replace never ran the new request"
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert agents[0].force_stops == 1
    assert len(agents) == 2
    assert agents[1].prompts == ["B fails"]
    assert any("agent1: B fails" in event.text for event in events)
    assert events[-1].kind is EventKind.DONE


def test_replace_from_the_owning_shell_kills_the_turn_and_runs_the_request(
    tmp_path: Path,
) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    try:
        hung = _hang(env, agents, "A")
        second, events = _collect_in_background(env, "A", "A again")
        assert _wait_for(lambda: bool(_busy(events)))
        assert client_transport.busy_choice("replace", shell_id="A", env=env) is True
        second.join(5.0)
        hung.join(5.0)
        assert not second.is_alive() and not hung.is_alive()
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert agents[0].force_stops == 1
    assert agents[1].prompts == ["A again"]
    assert events[-1].kind is EventKind.DONE


def test_steer_choice_delivers_the_request_into_the_running_turn(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_steerable_factory(agents))
    _start(daemon)
    try:
        _hang(env, agents, "A")
        second, events = _collect_in_background(env, "A", "try the other flag")
        assert _wait_for(lambda: bool(_busy(events)))
        assert client_transport.busy_choice("steer", shell_id="A", env=env) is True
        second.join(5.0)
        assert not second.is_alive()
        assert daemon.active_turn() is not None
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert cast_steerable(agents[0]).steered == ["try the other flag"]
    assert agents[0].prompts == [f"{BLOCK} A"]
    assert events[-1].kind is EventKind.DONE


def cast_steerable(agent: StubbornAgent) -> SteerableAgent:
    assert isinstance(agent, SteerableAgent)
    return agent


def test_a_client_that_ignores_busy_falls_back_to_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_mod, "_BUSY_CHOICE_TIMEOUT", 0.3)
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    try:
        _hang(env, agents, "A")
        second, events = _collect_in_background(env, "A", "A again")
        assert _wait_for(lambda: bool(_busy(events)))
        assert _wait_for(lambda: bool(daemon.state()["queued"])), "never fell back to the queue"
        assert client_transport.busy_choice("exit", shell_id="A", env=env) is False
        agents[0].killed.set()
        second.join(5.0)
        assert not second.is_alive()
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert any("agent0: A again" in event.text for event in events)
    assert events[-1].kind is EventKind.DONE


def test_busy_choice_is_a_control_message_and_needs_a_pending_prompt(tmp_path: Path) -> None:
    assert "busy_choice" in daemon_mod._CONTROL_KINDS
    assert client_transport.busy_choice("exit", shell_id="A", env=_env(tmp_path)) is False
    daemon = daemon_mod.Daemon(Config(), env=_env(tmp_path), agent_factory=_factory([]))
    events = list(daemon.handle_message({"kind": "busy_choice", "shell": "A", "choice": "nope"}))
    assert events[-1].kind is EventKind.ERROR
    events = list(daemon.handle_message({"kind": "busy_choice", "shell": "A", "choice": "exit"}))
    assert events[-1].kind is EventKind.ERROR


def test_shell_pid_gone_only_for_a_numeric_dead_pid() -> None:
    assert daemon_mod._shell_pid_gone(_dead_pid()) is True
    assert daemon_mod._shell_pid_gone(str(os.getpid())) is False
    assert daemon_mod._shell_pid_gone("A") is False
    assert daemon_mod._shell_pid_gone("0") is False


def test_shell_pid_gone_public_wrapper_matches_the_private_predicate() -> None:
    """``nvsh.doctor_checks`` calls the public name; it must agree with the private one."""
    assert daemon_mod.shell_pid_gone(_dead_pid()) is True
    assert daemon_mod.shell_pid_gone(str(os.getpid())) is False


# --- kill_active: owner-authority kill for `nvsh doctor --apply` (t19) ------


def test_kill_active_is_a_control_message_that_never_queues() -> None:
    assert "kill_active" in daemon_mod._CONTROL_KINDS


def test_kill_active_without_a_daemon_returns_no_daemon(tmp_path: Path) -> None:
    assert client_transport.kill_active(env=_env(tmp_path)) == "no_daemon"


def test_kill_active_with_no_running_turn_is_idle(tmp_path: Path) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory([]))
    _start(daemon)
    try:
        assert client_transport.kill_active(env=env) == "idle"
    finally:
        daemon.shutdown()


def test_kill_active_kills_a_dead_owner_turn_without_confirmation(tmp_path: Path) -> None:
    """The core of doctor --apply: a dead-owner turn is killed with owner authority."""
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    owner = _dead_pid()
    try:
        hung = _hang(env, agents, owner)
        assert client_transport.kill_active(confirmed=False, env=env) in ("killed", "stopping")
        hung.join(5.0)
        assert not hung.is_alive()
        assert daemon.active_turn() is None
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert agents[0].force_stops == 1


def test_kill_active_refuses_a_live_owner_without_confirmation(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    try:
        hung = _hang(env, agents, "A")
        assert client_transport.kill_active(confirmed=False, env=env) == "refused"
        time.sleep(0.2)
        turn = daemon.active_turn()
        assert turn is not None and turn["shell"] == "A"
        assert agents[0].force_stops == 0
        assert hung.is_alive()
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()


def test_kill_active_kills_a_live_owner_once_confirmed(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agents: list[StubbornAgent] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=_factory(agents))
    _start(daemon)
    try:
        hung = _hang(env, agents, "A")
        assert client_transport.kill_active(confirmed=True, env=env) in ("killed", "stopping")
        hung.join(5.0)
        assert not hung.is_alive()
        assert daemon.active_turn() is None
    finally:
        for agent in agents:
            agent.killed.set()
        daemon.shutdown()

    assert agents[0].force_stops == 1
