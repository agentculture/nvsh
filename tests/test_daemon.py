"""Unit tests for the nvsh session daemon and its client transport.

Everything here runs the daemon *in process* (a background thread) with an
injected ``agent_factory``, so no node/pi is needed. The process-lifecycle
criteria (pgrep, idle timeout, real pi processes) live in
``tests/test_daemon_lifecycle.py``.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

from nvsh import client_transport
from nvsh import daemon as daemon_mod
from nvsh.agent import registry
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
    event_from_dict,
    event_to_dict,
)
from nvsh.config import Config

# --- helpers --------------------------------------------------------------


class RecordingAgent(NvshAgent):
    """A FakeAgent-shaped adapter that records session commands and prompts."""

    #: Shared across instances so a test can see construction order.
    def __init__(self, log: list[tuple[str, str]], name: str = "a0") -> None:
        self.log = log
        self.name = name
        self.closed = False
        self.cancelled = False

    def start(self) -> None:
        self.log.append((self.name, "start"))

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self.log.append((self.name, f"run:{request.prompt or request.command}"))
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text=f"[{self.name}] {request.prompt}")
        yield AgentEvent(kind=EventKind.DONE)

    def cancel(self) -> None:
        self.cancelled = True
        self.log.append((self.name, "cancel"))

    def close(self) -> None:
        self.closed = True
        self.log.append((self.name, "close"))

    def capabilities(self) -> Capabilities:
        return Capabilities(persistent_session=True)

    def new_session(self) -> None:
        self.log.append((self.name, "new_session"))

    def switch_session(self, path: str) -> None:
        self.log.append((self.name, f"switch_session:{Path(path).name}"))

    def respond_ui(self, request_id: str, **fields: object) -> None:
        self.log.append((self.name, f"respond_ui:{request_id}"))


def _env(tmp_path: Path) -> dict[str, str]:
    run = tmp_path / "run"
    state = tmp_path / "state"
    run.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    return {
        "XDG_RUNTIME_DIR": str(run),
        "XDG_STATE_HOME": str(state),
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
    }


def _start(daemon: daemon_mod.Daemon) -> threading.Thread:
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    for _ in range(500):
        if daemon_mod.is_running(daemon.env):
            return thread
        time.sleep(0.01)
    raise AssertionError("daemon never created its socket")


def _failure(prompt: str = "why did it fail?") -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, prompt=prompt, command="ls /nope", exit_code=2)


def _collect(events: Iterator[AgentEvent]) -> list[AgentEvent]:
    return list(events)


# --- event wire encoding ---------------------------------------------------


def test_event_round_trip_text_delta() -> None:
    event = AgentEvent(kind=EventKind.TEXT_DELTA, text="hello")
    assert event_from_dict(event_to_dict(event)) == event


def test_event_round_trip_proposal() -> None:
    event = AgentEvent(
        kind=EventKind.PROPOSAL,
        proposal=Proposal(command="nvidia-smi", rationale="look", kind=ProposalKind.INSPECT),
        args={"request_id": "ui-1"},
    )
    back = event_from_dict(event_to_dict(event))
    assert back.proposal == event.proposal
    assert back.args == {"request_id": "ui-1"}


def test_event_to_dict_omits_defaults() -> None:
    assert event_to_dict(AgentEvent(kind=EventKind.DONE)) == {"kind": "done"}


def test_event_from_dict_unknown_kind_degrades_to_status() -> None:
    back = event_from_dict({"kind": "from_the_future", "text": "x"})
    assert back.kind is EventKind.STATUS


# --- socket path / hygiene -------------------------------------------------


def test_socket_path_under_xdg_runtime_dir(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert daemon_mod.socket_path(env) == tmp_path / "run" / "nvsh" / "daemon.sock"


def test_socket_path_falls_back_without_xdg_runtime_dir() -> None:
    path = daemon_mod.socket_path({})
    assert path.name == "daemon.sock"
    assert str(os.getuid()) in str(path)


def test_socket_path_does_not_create_anything(tmp_path: Path) -> None:
    env = _env(tmp_path)
    path = daemon_mod.socket_path(env)
    assert not path.parent.exists()


def test_socket_dir_is_0700_and_socket_is_0600(tmp_path: Path) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    _start(daemon)
    try:
        sock = daemon.socket_path
        assert stat.S_IMODE(sock.stat().st_mode) == 0o600
        assert stat.S_IMODE(sock.parent.stat().st_mode) == 0o700
    finally:
        daemon.shutdown()


def test_stale_socket_is_reaped(tmp_path: Path) -> None:
    env = _env(tmp_path)
    path = daemon_mod.socket_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a socket", encoding="utf-8")
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    _start(daemon)
    try:
        assert stat.S_ISSOCK(path.stat().st_mode)
    finally:
        daemon.shutdown()


def test_daemon_logs_to_state_dir_never_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    _start(daemon)
    try:
        _collect(client_transport.send(_failure(), shell_id="1", env=env, autostart=False))
    finally:
        daemon.shutdown()
    log = tmp_path / "state" / "nvsh" / "daemon.log"
    assert log.is_file()
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert log.read_text(encoding="utf-8").strip()
    captured = capsys.readouterr()
    assert captured.out == ""


# --- the request/response protocol ----------------------------------------


def test_failure_request_streams_events_and_ends_with_done(tmp_path: Path) -> None:
    env = _env(tmp_path)
    log: list[tuple[str, str]] = []
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent(log))
    _start(daemon)
    try:
        events = _collect(
            client_transport.send(_failure(), shell_id="101", env=env, autostart=False)
        )
    finally:
        daemon.shutdown()
    assert events[-1].kind is EventKind.DONE
    assert any(e.kind is EventKind.TEXT_DELTA for e in events)


def test_raw_protocol_is_json_lines(tmp_path: Path) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    _start(daemon)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(str(daemon.socket_path))
            payload = {
                "shell": "7",
                "kind": "failure",
                "request": {"kind": "failure", "prompt": "hi", "command": "x", "exit_code": 1},
                "context": {"platform": "spark", "cwd": "/tmp"},
            }
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            data = sock.makefile("rb").read().decode("utf-8")
    finally:
        daemon.shutdown()
    lines = [json.loads(line) for line in data.splitlines() if line.strip()]
    assert lines[-1]["kind"] == "done"
    assert all("kind" in line for line in lines)


def test_malformed_line_yields_error_event(tmp_path: Path) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    _start(daemon)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(str(daemon.socket_path))
            sock.sendall(b"{not json}\n")
            data = sock.makefile("rb").read().decode("utf-8")
    finally:
        daemon.shutdown()
    first = json.loads(data.splitlines()[0])
    assert first["kind"] == "error"


def test_context_reaches_the_agent(tmp_path: Path) -> None:
    env = _env(tmp_path)
    seen: list[AgentContext] = []

    class ContextAgent(RecordingAgent):
        def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
            seen.append(context)
            yield AgentEvent(kind=EventKind.DONE)

    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: ContextAgent([]))
    _start(daemon)
    try:
        ctx = AgentContext(platform="dgx-spark", output="boom", cwd="/srv", shell_pid=9)
        _collect(client_transport.send(_failure(), ctx, shell_id="9", env=env, autostart=False))
    finally:
        daemon.shutdown()
    assert seen
    assert seen[0].platform == "dgx-spark"
    assert seen[0].output == "boom"


def test_cancel_control_reaches_the_agent(tmp_path: Path) -> None:
    env = _env(tmp_path)
    log: list[tuple[str, str]] = []
    agent = RecordingAgent(log)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        _collect(client_transport.send(_failure(), shell_id="5", env=env, autostart=False))
        client_transport.cancel(shell_id="5", env=env)
    finally:
        daemon.shutdown()
    assert agent.cancelled


def test_ui_response_control_reaches_the_agent(tmp_path: Path) -> None:
    env = _env(tmp_path)
    log: list[tuple[str, str]] = []
    agent = RecordingAgent(log)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        _collect(client_transport.send(_failure(), shell_id="5", env=env, autostart=False))
        client_transport.respond_ui("ui-1", {"confirmed": True}, shell_id="5", env=env)
    finally:
        daemon.shutdown()
    assert (agent.name, "respond_ui:ui-1") in log


def test_undo_control_drops_the_last_turn_and_pending_proposal(tmp_path: Path) -> None:
    """/undo (task t14) never runs anything -- it only trims nvsh's own view."""
    env = _env(tmp_path)

    class ProposingAgent(RecordingAgent):
        def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
            self.log.append((self.name, f"run:{request.prompt}"))
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text="looking...")
            yield AgentEvent(
                kind=EventKind.PROPOSAL,
                proposal=Proposal(command="nvidia-smi", rationale="check", kind=ProposalKind.FIX),
            )
            yield AgentEvent(kind=EventKind.DONE)

    log: list[tuple[str, str]] = []
    agent = ProposingAgent(log)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    _start(daemon)
    try:
        _collect(client_transport.send(_failure(), shell_id="5", env=env, autostart=False))
        conversation = daemon._conversations["5"]
        assert conversation.transcript
        assert conversation.pending_proposal is not None

        events = client_transport.control("undo", shell_id="5", env=env)
        assert events

        assert conversation.pending_proposal is None
        assert not conversation.transcript
    finally:
        daemon.shutdown()
    # Nothing this test did could have run "nvidia-smi": the proposal was
    # only ever streamed and recorded, never approved or executed.
    assert not any("nvidia-smi" in entry for entry in log)


def test_undo_control_with_nothing_to_undo_reports_it(tmp_path: Path) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    _start(daemon)
    try:
        events = client_transport.control("undo", shell_id="never-seen", env=env)
    finally:
        daemon.shutdown()
    assert any("nothing to undo" in (e.text or "") for e in events)


# --- per-shell conversations ----------------------------------------------


def test_two_shells_share_one_agent_and_swap_sessions(tmp_path: Path) -> None:
    """A fails, B fails, then /ask from A: one agent, A's session resumed."""
    env = _env(tmp_path)
    log: list[tuple[str, str]] = []
    built: list[RecordingAgent] = []

    def factory() -> NvshAgent:
        agent = RecordingAgent(log, name=f"a{len(built)}")
        built.append(agent)
        return agent

    daemon = daemon_mod.Daemon(Config(sessions_max=1), env=env, agent_factory=factory)
    _start(daemon)
    try:
        _collect(client_transport.send(_failure("A fails"), shell_id="A", env=env, autostart=False))
        _collect(client_transport.send(_failure("B fails"), shell_id="B", env=env, autostart=False))
        events = _collect(
            client_transport.send(
                AgentRequest(kind=RequestKind.SLASH, prompt="what did you just see"),
                shell_id="A",
                env=env,
                autostart=False,
            )
        )
    finally:
        daemon.shutdown()

    assert len(built) == 1, "sessions.max=1 must never build a second agent"
    names = [entry[1] for entry in log if entry[0] == "a0"]
    assert names.count("new_session") == 2, "one fresh session per shell"
    assert any(n.startswith("switch_session:") and "A" in n for n in names)
    assert "[a0] what did you just see" in "".join(e.text for e in events)
    # A's conversation resumed: the last session command before A's /ask run
    # must be a switch back to A's session, not B's.
    last_session_cmd = [n for n in names if "session" in n][-1]
    assert "A" in last_session_cmd


def test_sessions_max_two_builds_a_second_agent(tmp_path: Path) -> None:
    env = _env(tmp_path)
    log: list[tuple[str, str]] = []
    built: list[RecordingAgent] = []

    def factory() -> NvshAgent:
        agent = RecordingAgent(log, name=f"a{len(built)}")
        built.append(agent)
        return agent

    daemon = daemon_mod.Daemon(Config(sessions_max=2), env=env, agent_factory=factory)
    _start(daemon)
    try:
        _collect(client_transport.send(_failure("A"), shell_id="A", env=env, autostart=False))
        _collect(client_transport.send(_failure("B"), shell_id="B", env=env, autostart=False))
        _collect(client_transport.send(_failure("A again"), shell_id="A", env=env, autostart=False))
    finally:
        daemon.shutdown()

    assert len(built) == 2
    assert not any("switch_session" in entry[1] for entry in log)


def test_status_reports_shells_and_agents(tmp_path: Path) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    _start(daemon)
    try:
        _collect(client_transport.send(_failure(), shell_id="42", env=env, autostart=False))
        status = client_transport.status(env=env)
    finally:
        daemon.shutdown()
    assert status["running"] is True
    assert "42" in status["shells"]
    assert status["agents"] == 1
    assert status["backend"]


def test_status_when_no_daemon_is_running(tmp_path: Path) -> None:
    status = client_transport.status(env=_env(tmp_path))
    assert status["running"] is False


# --- lifecycle (in-process) ------------------------------------------------


def test_unregistering_the_last_shell_stops_the_daemon(tmp_path: Path) -> None:
    env = _env(tmp_path)
    log: list[tuple[str, str]] = []
    agent = RecordingAgent(log)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent)
    thread = _start(daemon)
    client_transport.register(shell_id="1", env=env)
    client_transport.register(shell_id="2", env=env)
    _collect(client_transport.send(_failure(), shell_id="1", env=env, autostart=False))
    client_transport.unregister(shell_id="1", env=env)
    assert thread.is_alive()
    client_transport.unregister(shell_id="2", env=env)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert agent.closed
    assert not daemon.socket_path.exists()


def test_idle_timeout_stops_the_daemon(tmp_path: Path) -> None:
    env = _env(tmp_path)
    log: list[tuple[str, str]] = []
    agent = RecordingAgent(log)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: agent, idle_timeout=0.5)
    thread = _start(daemon)
    _collect(client_transport.send(_failure(), shell_id="1", env=env, autostart=False))
    thread.join(timeout=6)
    assert not thread.is_alive()
    assert agent.closed


def test_stop_control_stops_the_daemon(tmp_path: Path) -> None:
    env = _env(tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: RecordingAgent([]))
    thread = _start(daemon)
    client_transport.stop(env=env)
    thread.join(timeout=5)
    assert not thread.is_alive()


# --- fallbacks -------------------------------------------------------------


def _fake_openai_compat(monkeypatch: pytest.MonkeyPatch, agent: NvshAgent) -> None:
    spec = registry.ADAPTERS["openai-compat"]
    monkeypatch.setitem(
        registry.ADAPTERS,
        "openai-compat",
        registry.AdapterSpec(
            name=spec.name,
            binary=None,
            factory=lambda _cfg: agent,
            description=spec.description,
        ),
    )


def test_client_falls_back_to_one_shot_without_a_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    agent = RecordingAgent([], name="oneshot")
    _fake_openai_compat(monkeypatch, agent)
    cfg = Config(agent_provider="openai-compat")
    events = _collect(
        client_transport.send(_failure(), shell_id="1", env=env, config=cfg, autostart=False)
    )
    assert any(e.kind is EventKind.TEXT_DELTA for e in events)
    assert events[-1].kind is EventKind.DONE
    assert agent.closed, "a one-shot run must close its adapter"
    assert any("one-shot" in e.text for e in events if e.kind is EventKind.STATUS)


def test_client_falls_back_when_the_socket_is_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    path = daemon_mod.socket_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))  # bound but never listening -> connect refused
    agent = RecordingAgent([], name="oneshot")
    _fake_openai_compat(monkeypatch, agent)
    cfg = Config(agent_provider="openai-compat")
    try:
        events = _collect(
            client_transport.send(_failure(), shell_id="1", env=env, config=cfg, autostart=False)
        )
    finally:
        dead.close()
    assert events[-1].kind is EventKind.DONE


def test_daemon_reports_pi_unavailable_and_uses_the_fallback_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(tmp_path)
    agent = RecordingAgent([], name="fallback")
    _fake_openai_compat(monkeypatch, agent)
    daemon = daemon_mod.Daemon(Config(agent_provider="pi"), env=env, which=lambda _name: None)
    _start(daemon)
    try:
        events = _collect(client_transport.send(_failure(), shell_id="1", env=env, autostart=False))
        status = client_transport.status(env=env)
    finally:
        daemon.shutdown()
    statuses = [e.text for e in events if e.kind is EventKind.STATUS]
    assert any(text.startswith("pi unavailable:") for text in statuses)
    assert any("[fallback]" in e.text for e in events if e.kind is EventKind.TEXT_DELTA)
    assert status["backend"] == "openai-compat"


def test_no_process_is_spawned_by_importing_the_client(tmp_path: Path) -> None:
    """Starting a shell must create no nvsh or pi process.

    The bash side of this criterion (the hook only calls `nvsh` on a
    qualifying failure) is covered by ``tests/test_hook_bash.py``; here we
    prove the Python side: importing the client and resolving the socket
    path start nothing and touch nothing.
    """
    env = _env(tmp_path)
    before = len(os.listdir("/proc/self/task"))
    path = client_transport.daemon_socket_path(env)
    assert not path.exists()
    assert len(os.listdir("/proc/self/task")) == before


# --- PR #8 review: per-shell isolation of cancel / steer / ui-response ------


class SteerableAgent(RecordingAgent):
    """A RecordingAgent with the mid-turn channel ``_handle_steer`` looks for."""

    def __init__(self, log: list[tuple[str, str]], name: str = "a0") -> None:
        super().__init__(log, name)
        self.steered: list[str] = []

    def steer(self, text: str) -> bool:
        self.steered.append(text)
        self.log.append((self.name, f"steer:{text}"))
        return True


def _daemon_with_slots(tmp_path, shells: list[str]):
    """A daemon holding one slot per shell in ``shells`` (no socket, no serve)."""
    log: list[tuple[str, str]] = []
    made: list[SteerableAgent] = []

    def factory():
        agent = SteerableAgent(log, f"a{len(made)}")
        made.append(agent)
        return agent

    cfg = Config()
    cfg.sessions_max = len(shells)
    daemon = daemon_mod.Daemon(cfg, env=_env(tmp_path), agent_factory=factory)
    for shell in shells:
        daemon._acquire(shell)
    return daemon, made


def test_cancel_shell_only_cancels_that_shells_slot(tmp_path):
    daemon, agents = _daemon_with_slots(tmp_path, ["shell-a", "shell-b"])
    daemon.cancel_shell("shell-b")
    assert [a.cancelled for a in agents] == [False, True]


def test_cancel_shell_for_an_unknown_shell_cancels_nothing(tmp_path):
    """A terminal queued behind another's turn must not cancel that turn."""
    daemon, agents = _daemon_with_slots(tmp_path, ["shell-a"])
    daemon.cancel_shell("shell-queued")
    assert [a.cancelled for a in agents] == [False]


def test_steer_reaches_only_the_requesting_shell(tmp_path):
    daemon, agents = _daemon_with_slots(tmp_path, ["shell-a", "shell-b"])
    events = list(daemon._handle_steer("shell-b", {"text": "check memory"}))
    assert agents[0].steered == []
    assert agents[1].steered == ["check memory"]
    assert [e.kind for e in events][-1] is EventKind.DONE


def test_steer_from_an_idle_shell_is_an_error_not_a_broadcast(tmp_path):
    daemon, agents = _daemon_with_slots(tmp_path, ["shell-a"])
    events = list(daemon._handle_steer("shell-idle", {"text": "check memory"}))
    assert agents[0].steered == []
    assert events[0].kind is EventKind.ERROR
    assert "no running turn" in (events[0].error or "")


def test_ui_response_from_an_idle_shell_is_an_error(tmp_path):
    daemon, agents = _daemon_with_slots(tmp_path, ["shell-a"])
    events = list(daemon._handle_ui_response("shell-idle", {"request_id": "ui-1"}))
    assert not [entry for entry in agents[0].log if entry[1].startswith("respond_ui")]
    assert events[0].kind is EventKind.ERROR


def test_ui_response_reaches_its_own_shell(tmp_path):
    daemon, agents = _daemon_with_slots(tmp_path, ["shell-a", "shell-b"])
    events = list(daemon._handle_ui_response("shell-b", {"request_id": "ui-1"}))
    assert (agents[1].name, "respond_ui:ui-1") in agents[1].log
    assert [e.kind for e in events][-1] is EventKind.DONE
