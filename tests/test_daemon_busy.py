"""A busy daemon must never wedge behind one never-ending turn (deviation d12).

Observed live on a DGX Spark: one shell's ``/ask`` turn became the daemon's
active conversation and never ended (its client -- a test pty -- had gone
away). Because :meth:`nvsh.daemon.Daemon._run` serializes every agent
request behind one lock and nothing aborted or capped that turn, every later
failure from four other shells sat silently on the run lock until the
client's *stream* timeout fired and fell back to one-shot. ``status``
answered instantly throughout (control messages never take the run lock),
which is how the wedge was spotted at all.

These tests measure the three behaviors that make that impossible now, with
a fake agent whose turn really does block:

1. the owning client's socket closing aborts the turn (:func:`cancel`, a
   deny for any pending dialog) and the next request runs;
2. a turn longer than ``turn_timeout`` is aborted the same way and its
   client is told so;
3. a request that has to wait says so immediately, and ``daemon status``
   shows the active turn and the queue.

Plus the property the wedge depended on and that must stay true: control
messages never queue behind a turn.
"""

from __future__ import annotations

import json
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
    Proposal,
    ProposalKind,
    RequestKind,
)
from nvsh.config import Config

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="the daemon needs unix domain sockets"
)

#: Prompts starting with this make :class:`BlockingAgent` hold the turn open.
BLOCK = "block"


class BlockingAgent(NvshAgent):
    """A fake whose turn blocks until released -- or until ``cancel()``.

    Only prompts starting with ``block`` hang; anything else answers at
    once, so one agent process (``sessions.max = 1``) can hold a wedged turn
    for shell A and still serve shell B the moment the wedge clears.
    """

    def __init__(self, *, propose: bool = False) -> None:
        self.release = threading.Event()
        self.turn_started = threading.Event()
        self.cancels = 0
        self.ui_responses: list[tuple[str, dict]] = []
        self.closed = False
        self._cancelled = False
        self._propose = propose

    def start(self) -> None:
        return None

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self._cancelled = False
        if not request.prompt.startswith(BLOCK):
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text=f"answer for {request.prompt}")
            yield AgentEvent(kind=EventKind.DONE)
            return
        if self._propose:
            yield AgentEvent(
                kind=EventKind.PROPOSAL,
                proposal=Proposal(command="nvidia-smi", rationale="look", kind=ProposalKind.FIX),
                args={"request_id": "ui-block"},
            )
        self.turn_started.set()
        while not self.release.wait(0.01):
            if self._cancelled:
                return
        yield AgentEvent(kind=EventKind.DONE)

    def cancel(self) -> None:
        self.cancels += 1
        self._cancelled = True

    def close(self) -> None:
        self.closed = True

    def capabilities(self) -> Capabilities:
        return Capabilities(persistent_session=True)

    def new_session(self) -> None:
        return None

    def switch_session(self, path: str) -> None:
        return None

    def respond_ui(self, request_id: str, **fields: object) -> None:
        self.ui_responses.append((request_id, dict(fields)))


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    run = tmp_path / "run"
    state = tmp_path / "state"
    run.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    return {
        "XDG_RUNTIME_DIR": str(run),
        "XDG_STATE_HOME": str(state),
        "HOME": str(tmp_path),
    } | extra


def _start(daemon: daemon_mod.Daemon) -> threading.Thread:
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    for _ in range(500):
        if daemon_mod.is_running(daemon.env):
            return thread
        time.sleep(0.01)
    raise AssertionError("daemon never created its socket")


def _request(prompt: str) -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, prompt=prompt, command="ls /nope", exit_code=2)


def _payload(shell: str, prompt: str) -> bytes:
    message = {
        "shell": shell,
        "kind": "failure",
        "request": {"kind": "failure", "prompt": prompt, "command": "ls /nope", "exit_code": 2},
        "context": {},
    }
    return (json.dumps(message) + "\n").encode("utf-8")


def _raw_client(daemon: daemon_mod.Daemon, shell: str, prompt: str) -> socket.socket:
    """Start a turn over a bare socket we can kill without closing it politely."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(str(daemon.socket_path))
    sock.sendall(_payload(shell, prompt))
    return sock


# --- 1. the owning client goes away ---------------------------------------


def test_turn_is_aborted_when_its_client_disappears(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agent = BlockingAgent()
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        gone = _raw_client(daemon, "A", f"{BLOCK} A")
        assert agent.turn_started.wait(5.0), "the fake agent never began A's turn"
        gone.close()  # the pty went away mid-turn, exactly as on the Spark

        began = time.monotonic()
        events = list(
            client_transport.send(
                _request("B fails"), shell_id="B", env=env, autostart=False, timeout=5.0
            )
        )
        elapsed = time.monotonic() - began
    finally:
        agent.release.set()
        daemon.shutdown()

    assert elapsed < 2.0, f"B waited {elapsed:.2f}s behind a dead client's turn"
    assert agent.cancels >= 1, "the daemon never cancelled the abandoned turn"
    assert events[-1].kind is EventKind.DONE
    assert any("answer for B fails" in event.text for event in events)


def test_aborting_an_abandoned_turn_denies_its_pending_dialog(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agent = BlockingAgent(propose=True)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        gone = _raw_client(daemon, "A", f"{BLOCK} A")
        assert agent.turn_started.wait(5.0)
        gone.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not agent.ui_responses:
            time.sleep(0.02)
    finally:
        agent.release.set()
        daemon.shutdown()

    assert agent.ui_responses, "the pending approval dialog was never answered"
    request_id, fields = agent.ui_responses[0]
    assert request_id == "ui-block"
    assert fields.get("cancelled") is True


def test_an_abandoned_turn_leaves_the_conversation_idle(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agent = BlockingAgent(propose=True)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        gone = _raw_client(daemon, "A", f"{BLOCK} A")
        assert agent.turn_started.wait(5.0)
        gone.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and daemon.active_turn() is not None:
            time.sleep(0.02)
        assert daemon.active_turn() is None
        assert daemon._conversations["A"].pending_proposal is None
    finally:
        agent.release.set()
        daemon.shutdown()


# --- 2. the wall-clock cap -------------------------------------------------


def test_a_turn_longer_than_the_cap_is_aborted_and_reported(tmp_path: Path) -> None:
    env = _env(tmp_path, NVSH_TURN_TIMEOUT="1")
    agent = BlockingAgent()
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    assert daemon.turn_timeout == pytest.approx(1.0)
    _start(daemon)
    try:
        began = time.monotonic()
        events = list(
            client_transport.send(
                _request(f"{BLOCK} A"), shell_id="A", env=env, autostart=False, timeout=10.0
            )
        )
        elapsed = time.monotonic() - began
    finally:
        agent.release.set()
        daemon.shutdown()

    assert events[-1].kind is EventKind.ERROR
    assert "1" in events[-1].error
    assert "turn" in events[-1].error.lower()
    assert agent.cancels >= 1
    assert 0.9 <= elapsed < 6.0, f"the cap fired after {elapsed:.2f}s"


def test_the_turn_cap_defaults_to_something_generous() -> None:
    assert daemon_mod.DEFAULT_TURN_TIMEOUT >= 300.0
    assert daemon_mod.turn_timeout({}) == daemon_mod.DEFAULT_TURN_TIMEOUT
    assert daemon_mod.turn_timeout({"NVSH_TURN_TIMEOUT": "junk"}) == (
        daemon_mod.DEFAULT_TURN_TIMEOUT
    )
    assert daemon_mod.turn_timeout({"NVSH_TURN_TIMEOUT": "-2"}) == daemon_mod.DEFAULT_TURN_TIMEOUT
    assert daemon_mod.turn_timeout({"NVSH_TURN_TIMEOUT": "1.5"}) == pytest.approx(1.5)


# --- 3. queue visibility ---------------------------------------------------


def test_a_queued_request_is_told_it_is_waiting_before_it_waits(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agent = BlockingAgent()
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        blocker = _raw_client(daemon, "A", f"{BLOCK} A")
        assert agent.turn_started.wait(5.0)

        stream = client_transport.send(
            _request("B fails"), shell_id="B", env=env, autostart=False, timeout=5.0
        )
        began = time.monotonic()
        first = next(stream)
        waited = time.monotonic() - began
        assert first.kind is EventKind.STATUS
        assert "waiting for the agent" in first.text
        assert "A" in first.text
        assert waited < 2.0, f"the waiting notice took {waited:.2f}s"

        agent.release.set()
        rest = list(stream)
        assert rest[-1].kind is EventKind.DONE
        blocker.close()
    finally:
        agent.release.set()
        daemon.shutdown()


def test_status_exposes_the_active_turn_and_the_queue(tmp_path: Path) -> None:
    env = _env(tmp_path)
    agent = BlockingAgent()
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    queued_events: list[AgentEvent] = []

    def queued() -> None:
        queued_events.extend(
            client_transport.send(
                _request("B fails"), shell_id="B", env=env, autostart=False, timeout=10.0
            )
        )

    waiter = threading.Thread(target=queued, daemon=True)
    try:
        blocker = _raw_client(daemon, "A", f"{BLOCK} A")
        assert agent.turn_started.wait(5.0)
        waiter.start()

        state: dict = {}
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            state = client_transport.status(env=env)
            if state.get("queued"):
                break
            time.sleep(0.02)

        active = state.get("active_turn")
        assert active is not None, "status hid the active turn"
        assert active["shell"] == "A"
        assert active["elapsed"] >= 0.0
        assert active["started"] > 0.0
        assert [entry["shell"] for entry in state["queued"]] == ["B"]

        agent.release.set()
        waiter.join(timeout=10)
        blocker.close()
        idle = client_transport.status(env=env)
        assert idle["active_turn"] is None
        assert idle["queued"] == []
    finally:
        agent.release.set()
        daemon.shutdown()


# --- 4. control messages never queue --------------------------------------


def test_control_messages_never_queue_behind_a_turn(tmp_path: Path) -> None:
    """The one property the wedge left intact -- pinned so it stays intact."""
    env = _env(tmp_path)
    agent = BlockingAgent(propose=True)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        blocker = _raw_client(daemon, "A", f"{BLOCK} A")
        assert agent.turn_started.wait(5.0)

        began = time.monotonic()
        state = client_transport.status(env=env)
        assert state["running"] is True
        assert client_transport.respond_ui("ui-1", {"confirmed": True}, shell_id="A", env=env)
        assert client_transport.register(shell_id="C", env=env)
        assert client_transport.control("undo", shell_id="C", env=env)
        elapsed = time.monotonic() - began
        assert elapsed < 2.0, f"control messages took {elapsed:.2f}s behind a turn"

        began = time.monotonic()
        assert client_transport.stop(env=env)
        assert time.monotonic() - began < 2.0
        blocker.close()
    finally:
        agent.release.set()
        daemon.shutdown()
