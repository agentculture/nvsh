"""Task t16: target routing, the version handshake and child containment.

Three acceptance criteria, in order:

1. ``Daemon._build_agent`` resolves ``default`` through
   :meth:`nvsh.config.Config.resolve_target` and builds the warm agent with
   that target's ``model``/``effort`` plus the backend's ``extra_args`` and
   ``approval``; a request carrying a *non-default* target always runs
   one-shot in the client (decision c25); two consecutive failures from one
   shell reuse **one** warm process.
2. The client declares its ``nvsh`` version and the daemon answers with its
   own; on a mismatch the client stops the stale daemon, starts a fresh one
   and the request still succeeds. A request line from a pre-handshake
   client (no ``version``, no ``target``) still parses.
3. ``close()``'s wait -> terminate -> kill escalation reaches every adapter,
   and neither an idle exit nor ``nvsh uninstall`` leaves a harness child
   behind (driven with real, long-running fake binaries).

Everything runs against the fakes under ``tests/fakes/`` -- no harness, no
node and no network.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

from nvsh import __version__, client_transport
from nvsh import daemon as daemon_mod
from nvsh.agent import registry
from nvsh.agent.acp import AcpAgent
from nvsh.agent.base import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    NvshAgent,
    RequestKind,
    Target,
    request_from_dict,
    request_to_dict,
)
from nvsh.config import Config

FAKES = Path(__file__).parent / "fakes"

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="the daemon needs unix domain sockets"
)


# --- helpers ---------------------------------------------------------------


def _env(tmp_path: Path) -> dict[str, str]:
    run = tmp_path / "run"
    state = tmp_path / "state"
    for directory in (run, state):
        directory.mkdir(exist_ok=True)
    return {
        "PATH": f"{FAKES}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "XDG_RUNTIME_DIR": str(run),
        "XDG_STATE_HOME": str(state),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
    }


def _start(daemon: daemon_mod.Daemon) -> threading.Thread:
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    for _ in range(500):
        if daemon_mod.is_running(daemon.env):
            return thread
        time.sleep(0.01)
    raise AssertionError("daemon never created its socket")


def _failure(prompt: str = "why did it fail?", target: Target | None = None) -> AgentRequest:
    return AgentRequest(
        kind=RequestKind.FAILURE,
        prompt=prompt,
        command="ls /nope",
        exit_code=2,
        target=target,
    )


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


class _RecordingAgent(NvshAgent):
    """Counts starts/closes and remembers the config it was built from."""

    def __init__(self, built: list["_RecordingAgent"], config: Config) -> None:
        self.config = config
        self.starts = 0
        self.runs: list[AgentRequest] = []
        self.closed = False
        built.append(self)

    def start(self) -> None:
        self.starts += 1

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        self.runs.append(request)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="ok")
        yield AgentEvent(kind=EventKind.DONE)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _register_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    factory,
    *,
    binary: str | None = None,
    path: str = "acp",
) -> None:
    """Add one adapter to the registry for the length of a test."""
    adapters = dict(registry.ADAPTERS)
    adapters[name] = registry.AdapterSpec(
        name=name,
        binary=binary,
        factory=factory,
        description=f"test-only {name}",
        path=path,
        hosted=False,
    )
    monkeypatch.setattr(registry, "ADAPTERS", adapters)


# --- criterion 1a: the warm agent is built from the resolved default -------


def test_default_target_resolves_through_the_alias_table() -> None:
    config = Config(
        agent_provider="pi",
        aliases={"default": "claude/opus/high"},
        agents={"claude": {"model": "sonnet"}},
    )
    daemon = daemon_mod.Daemon(config, env={"HOME": "/nonexistent"})
    assert daemon.default_target() == Target(
        backend="claude", model="opus", effort="high", alias="default"
    )


def test_default_target_falls_back_to_the_legacy_provider() -> None:
    """No ``[aliases] default`` -> the legacy ``[agent] provider``, with the
    backend's own configured model."""
    config = Config(agent_provider="pi", agents={"pi": {"model": "associate"}})
    daemon = daemon_mod.Daemon(config, env={"HOME": "/nonexistent"})
    assert daemon.default_target() == Target(
        backend="pi", model="associate", effort=None, alias="default"
    )


def test_build_agent_passes_model_effort_extra_args_and_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance criterion 1: the warm agent is built for the resolved
    ``default`` target, and all four per-harness settings reach the factory."""
    built: list[_RecordingAgent] = []
    _register_fake_backend(
        monkeypatch,
        "t16",
        lambda config: _RecordingAgent(built, config),
        binary=None,
    )
    config = Config(
        agent_provider="pi",
        aliases={"default": "t16/big/high"},
        agents={"t16": {"model": "small", "extra_args": ["--x"], "approval": "harness"}},
    )
    daemon = daemon_mod.Daemon(config, env={"HOME": "/nonexistent"})

    agent = daemon._build_agent()

    assert isinstance(agent, _RecordingAgent)
    settings = agent.config.agents["t16"]
    assert settings["model"] == "big"  # the alias' model wins over [agents.t16]
    assert settings["effort"] == "high"
    assert settings["extra_args"] == ["--x"]  # untouched, and still delivered
    assert settings["approval"] == "harness"
    assert agent.config.agent_provider == "t16"
    assert daemon._target == Target(backend="t16", model="big", effort="high", alias="default")


def test_state_reports_the_daemons_version_and_resolved_target() -> None:
    config = Config(agent_provider="pi", agents={"pi": {"model": "associate"}})
    daemon = daemon_mod.Daemon(config, env={"HOME": "/nonexistent"})
    state = daemon.state()
    assert state["version"] == __version__
    assert state["target"] == {"backend": "pi", "model": "associate", "alias": "default"}


# --- criterion 1b: a non-default target runs one-shot in the client --------


def test_non_default_target_never_reaches_the_warm_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision c25: the warm session belongs to ``default``; anything else
    is answered in this process, even with a daemon right there."""
    env = _env(tmp_path)
    warm: list[_RecordingAgent] = []
    daemon = daemon_mod.Daemon(
        Config(), env=env, agent_factory=lambda: _RecordingAgent(warm, Config())
    )
    _start(daemon)

    one_shot_calls: list[AgentRequest] = []

    def fake_one_shot(request, context=None, **kwargs):
        one_shot_calls.append(request)
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "one_shot", fake_one_shot)
    try:
        events = list(
            client_transport.send(
                _failure(target=Target(backend="claude", model="opus")),
                AgentContext(),
                shell_id="s1",
                env=env,
                autostart=False,
            )
        )
    finally:
        daemon.shutdown()

    assert [event.kind for event in events] == [EventKind.DONE]
    assert len(one_shot_calls) == 1
    assert one_shot_calls[0].target == Target(backend="claude", model="opus")
    assert warm == []  # the daemon never built, let alone ran, its agent


def test_the_resolved_default_target_still_goes_to_the_daemon(tmp_path: Path) -> None:
    """The other half of c25: ``default`` is exactly what the warm session is."""
    env = _env(tmp_path)
    warm: list[_RecordingAgent] = []
    daemon = daemon_mod.Daemon(
        Config(), env=env, agent_factory=lambda: _RecordingAgent(warm, Config())
    )
    _start(daemon)
    try:
        target = Target(backend="pi", model="associate", alias="default")
        events = list(
            client_transport.send(
                _failure(target=target),
                AgentContext(),
                shell_id="s1",
                env=env,
                autostart=False,
            )
        )
    finally:
        daemon.shutdown()
    assert EventKind.TEXT_DELTA in [event.kind for event in events]
    assert len(warm) == 1
    assert warm[0].runs[0].target == target


def test_a_targeted_request_that_reaches_the_daemon_anyway_runs_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older client, or a direct API caller, can still send one. The
    daemon serves it outside the warm slot and closes the adapter."""
    built: list[_RecordingAgent] = []
    _register_fake_backend(monkeypatch, "t16", lambda config: _RecordingAgent(built, config))
    daemon = daemon_mod.Daemon(Config(), env={"HOME": "/nonexistent"})

    events = list(
        daemon.handle_message(
            {
                "shell": "s1",
                "kind": "failure",
                "request": request_to_dict(
                    _failure(target=Target(backend="t16", model="big", alias="fast"))
                ),
                "context": {},
            }
        )
    )

    assert [event.kind for event in events][-1] is EventKind.DONE
    assert len(built) == 1
    assert built[0].config.agents["t16"]["model"] == "big"
    assert built[0].closed  # the one-shot child goes with the turn
    assert daemon._slots == []  # and the warm session was never touched


# --- criterion 1c: two consecutive failures reuse one warm process --------


def test_two_consecutive_failures_reuse_one_warm_agent(tmp_path: Path) -> None:
    env = _env(tmp_path)
    built: list[_RecordingAgent] = []
    daemon = daemon_mod.Daemon(
        Config(), env=env, agent_factory=lambda: _RecordingAgent(built, Config())
    )
    _start(daemon)
    try:
        for prompt in ("first failure", "second failure"):
            list(
                client_transport.send(
                    _failure(prompt), AgentContext(), shell_id="s1", env=env, autostart=False
                )
            )
    finally:
        daemon.shutdown()

    assert len(built) == 1, "the second failure built a second agent"
    assert built[0].starts == 1
    assert [request.prompt for request in built[0].runs] == ["first failure", "second failure"]


def _acp_env(tmp_path: Path) -> dict[str, str]:
    """PATH pointed at ``tests/fakes/acp`` with a one-line, approval-free script."""
    env = dict(os.environ)
    env["PATH"] = f"{FAKES}:{env.get('PATH', '')}"
    env["NVSH_TEST_ACP_SCRIPT"] = json.dumps(
        [
            {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "the path does not exist"},
                }
            }
        ]
    )
    return env


def test_an_acp_harness_keeps_one_child_across_two_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 1, with a real child: one ``acp`` process serves both turns,
    and it is gone once the daemon shuts down."""
    env = _env(tmp_path)
    acp_env = _acp_env(tmp_path)
    agents: list[AcpAgent] = []

    def factory(config: Config) -> AcpAgent:
        agent = AcpAgent(["acp"], "t16acp", env=acp_env)
        agents.append(agent)
        return agent

    _register_fake_backend(monkeypatch, "t16acp", factory, binary="acp", path="acp")
    daemon = daemon_mod.Daemon(
        Config(agent_provider="t16acp"),
        env=env,
        which=lambda name: str(FAKES / name) if (FAKES / name).exists() else None,
    )
    _start(daemon)
    pids: list[int] = []
    try:
        for prompt in ("first failure", "second failure"):
            events = list(
                client_transport.send(
                    _failure(prompt), AgentContext(), shell_id="s1", env=env, autostart=False
                )
            )
            assert events[-1].kind is EventKind.DONE, events
            assert len(agents) == 1
            assert agents[0]._proc is not None
            pids.append(agents[0]._proc.pid)
    finally:
        daemon.shutdown()

    assert pids[0] == pids[1], "the second failure spawned a second acp child"
    assert _wait_for(lambda: not _alive(pids[0])), "the acp child outlived the daemon"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# --- criterion 2: the version handshake -----------------------------------


def test_the_client_declares_its_version_on_every_request(tmp_path: Path) -> None:
    seen: list[dict] = []
    env = _env(tmp_path)

    class _Recorder(daemon_mod.Daemon):
        def handle_message(self, message, *, connection=None):
            seen.append(dict(message))
            return super().handle_message(message, connection=connection)

    built: list[_RecordingAgent] = []
    daemon = _Recorder(Config(), env=env, agent_factory=lambda: _RecordingAgent(built, Config()))
    _start(daemon)
    try:
        list(
            client_transport.send(
                _failure(), AgentContext(), shell_id="s1", env=env, autostart=False
            )
        )
        client_transport.register(shell_id="s1", env=env)
    finally:
        daemon.shutdown()

    assert seen, "the daemon saw no message at all"
    assert all(message.get("version") == __version__ for message in seen)


def test_the_daemon_answers_a_mismatched_client_with_its_own_version() -> None:
    daemon = daemon_mod.Daemon(Config(), env={"HOME": "/nonexistent"})
    events = list(
        daemon.handle_message(
            {
                "shell": "s1",
                "kind": "failure",
                "version": "0.0.1-from-another-wheel",
                "request": request_to_dict(_failure()),
                "context": {},
            }
        )
    )
    assert len(events) == 1
    assert events[0].kind is EventKind.ERROR
    assert events[0].args["version_mismatch"] is True
    assert events[0].args["daemon_version"] == __version__
    assert events[0].args["client_version"] == "0.0.1-from-another-wheel"
    assert __version__ in events[0].error


def test_control_messages_still_answer_a_mismatched_client() -> None:
    """``stop`` in particular: it is what the client does about a mismatch."""
    daemon = daemon_mod.Daemon(Config(), env={"HOME": "/nonexistent"})
    events = list(daemon.handle_message({"shell": "s1", "kind": "ping", "version": "0.0.1"}))
    assert events[-1].kind is EventKind.DONE
    assert json.loads(events[0].text)["version"] == __version__


def test_a_stale_daemon_is_stopped_restarted_and_the_request_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion 2, end to end: a daemon left over from another
    nvsh is replaced, and the operator still gets their answer."""
    env = _env(tmp_path)
    built: list[_RecordingAgent] = []
    stale = daemon_mod.Daemon(
        Config(), env=env, agent_factory=lambda: _RecordingAgent(built, Config())
    )
    stale.version = "0.0.1-stale"  # a leftover from another wheel
    _start(stale)

    started: list[daemon_mod.Daemon] = []

    def fake_spawn(spawn_env=None, **kwargs):
        fresh = daemon_mod.Daemon(
            Config(), env=env, agent_factory=lambda: _RecordingAgent(built, Config())
        )
        started.append(fresh)
        _start(fresh)
        return 4242

    monkeypatch.setattr(daemon_mod, "spawn", fake_spawn)
    try:
        events = list(
            client_transport.send(
                _failure(), AgentContext(), shell_id="s1", env=env, autostart=True
            )
        )
    finally:
        for served in started:
            served.shutdown()
        stale.shutdown()

    statuses = [event.text for event in events if event.kind is EventKind.STATUS]
    assert any("restarted the daemon" in text for text in statuses), statuses
    assert any("0.0.1-stale" in text for text in statuses), statuses
    assert stale._stopping, "the stale daemon was left running"
    assert len(started) == 1, "a fresh daemon was not started"
    assert [event.kind for event in events][-1] is EventKind.DONE
    assert any(event.kind is EventKind.TEXT_DELTA for event in events), "no answer was produced"


def test_a_matching_daemon_is_left_alone(tmp_path: Path) -> None:
    env = _env(tmp_path)
    built: list[_RecordingAgent] = []
    daemon = daemon_mod.Daemon(
        Config(), env=env, agent_factory=lambda: _RecordingAgent(built, Config())
    )
    _start(daemon)
    try:
        events = list(
            client_transport.send(
                _failure(), AgentContext(), shell_id="s1", env=env, autostart=False
            )
        )
    finally:
        daemon.shutdown()
    statuses = [event.text for event in events if event.kind is EventKind.STATUS]
    assert not any("restarted the daemon" in text for text in statuses), statuses
    assert len(built) == 1


def test_an_old_shape_request_line_still_parses(tmp_path: Path) -> None:
    """A request written by a pre-``target``, pre-``version`` client: no
    ``version`` key, no ``target`` key, ``kind`` only inside ``request``."""
    env = _env(tmp_path)
    built: list[_RecordingAgent] = []
    daemon = daemon_mod.Daemon(
        Config(), env=env, agent_factory=lambda: _RecordingAgent(built, Config())
    )
    _start(daemon)
    line = json.dumps(
        {
            "shell": "s1",
            "kind": "failure",
            "request": {
                "kind": "failure",
                "prompt": "why did it fail?",
                "command": "ls /nope",
                "exit_code": 2,
                "failure_id": "f1",
            },
            "context": {"platform": "dgx-spark", "cwd": "/srv", "shell_pid": 1},
        }
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(10.0)
            sock.connect(str(daemon_mod.socket_path(env)))
            stream = sock.makefile("rwb")
            stream.write((line + "\n").encode("utf-8"))
            stream.flush()
            kinds = [json.loads(raw)["kind"] for raw in stream]
    finally:
        daemon.shutdown()

    assert "text_delta" in kinds
    assert kinds[-1] == "done"
    assert built[0].runs[0].command == "ls /nope"
    assert built[0].runs[0].target is None


def test_the_daemon_and_base_share_one_request_codec() -> None:
    """The daemon used to carry its own decoder, which never learned about
    ``target``; it is now the one in ``nvsh.agent.base``."""
    assert daemon_mod.request_from_dict is request_from_dict
    request = _failure(target=Target(backend="claude", model="opus", alias="fast"))
    assert request_from_dict(request_to_dict(request)) == request


# --- criterion 3: no child harness process is left behind ------------------


def _long_running_binary(tmp_path: Path, name: str, *, trap_sigterm: bool) -> Path:
    """A fake harness that never exits on its own; returns its bin directory.

    It writes its pid where the test can find it, then sleeps -- and, when
    asked, ignores SIGTERM, so only the last rung of the escalation ends it.
    """
    bindir = tmp_path / f"bin-{name}"
    bindir.mkdir(exist_ok=True)
    script = bindir / name
    trap = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if trap_sigterm else ""
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, signal, sys, time\n"
        f"{trap}"
        f"open({str(bindir / 'pid')!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(600)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bindir


class _LongRunningAgent(NvshAgent):
    """An adapter whose child ignores everything short of the escalation."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        self._proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        yield AgentEvent(kind=EventKind.DONE)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        from nvsh.agent._subprocess import escalate_close

        escalate_close(self._proc, wait=0.2, grace=2.0)

    def capabilities(self) -> Capabilities:
        return Capabilities()


@pytest.mark.parametrize("trap_sigterm", [False, True], ids=["polite", "traps-sigterm"])
def test_an_idle_exit_leaves_no_harness_child(tmp_path: Path, trap_sigterm: bool) -> None:
    """Acceptance criterion 3: the idle watchdog's teardown reaps the child,
    whether it dies on SIGTERM or only on SIGKILL."""
    env = _env(tmp_path)
    bindir = _long_running_binary(tmp_path, "fake-harness", trap_sigterm=trap_sigterm)
    argv = [sys.executable, str(bindir / "fake-harness")]
    daemon = daemon_mod.Daemon(
        Config(),
        env=env,
        agent_factory=lambda: _LongRunningAgent(argv),
        idle_timeout=0.3,
    )
    thread = _start(daemon)
    try:
        list(
            client_transport.send(
                _failure(), AgentContext(), shell_id="s1", env=env, autostart=False
            )
        )
        assert _wait_for(lambda: (bindir / "pid").is_file())
        pid = int((bindir / "pid").read_text(encoding="utf-8"))
        assert _alive(pid)
        thread.join(timeout=15.0)
        assert _wait_for(lambda: not _alive(pid), 10.0), "the harness child outlived the daemon"
    finally:
        daemon.shutdown()


def test_uninstall_stops_the_daemon_and_its_harness_child(tmp_path: Path) -> None:
    """``nvsh uninstall`` calls ``nvsh daemon stop``; the stop must take the
    harness child with it, not just the socket."""
    env = _env(tmp_path)
    bindir = _long_running_binary(tmp_path, "fake-harness", trap_sigterm=True)
    argv = [sys.executable, str(bindir / "fake-harness")]
    daemon = daemon_mod.Daemon(Config(), env=env, agent_factory=lambda: _LongRunningAgent(argv))
    thread = _start(daemon)
    list(client_transport.send(_failure(), AgentContext(), shell_id="s1", env=env, autostart=False))
    assert _wait_for(lambda: (bindir / "pid").is_file())
    pid = int((bindir / "pid").read_text(encoding="utf-8"))
    assert _alive(pid)

    assert client_transport.stop(env=env) is True
    thread.join(timeout=15.0)
    assert _wait_for(lambda: not _alive(pid), 10.0), "uninstall left a harness child running"
    assert not daemon_mod.socket_path(env).exists()


@pytest.mark.parametrize(
    "build, attribute",
    [
        (lambda: __import__("nvsh.agent.pi", fromlist=["PiAgent"]).PiAgent(), "_proc"),
        (lambda: __import__("nvsh.agent.acp", fromlist=["build"]).build("qwen", {}), "_proc"),
        (lambda: __import__("nvsh.agent.agy", fromlist=["AgyAgent"]).AgyAgent(), "_proc"),
        (
            lambda: __import__("nvsh.agent.claude", fromlist=["ClaudeAgent"]).ClaudeAgent({}),
            "_proc",
        ),
        (lambda: __import__("nvsh.agent.qwen", fromlist=["QwenAgent"]).QwenAgent({}), "_proc"),
        (lambda: __import__("nvsh.agent.codex", fromlist=["CodexAgent"]).CodexAgent({}), "_rpc"),
    ],
    ids=["pi", "acp", "agy", "claude", "qwen", "codex"],
)
def test_every_adapters_close_escalates_to_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build, attribute: str
) -> None:
    """Acceptance criterion 3, adapter by adapter (deviation d5): whatever the
    harness, ``close()`` ends a child that ignores both EOF and SIGTERM.

    The child is planted on the adapter rather than launched through it, so
    the test is about the *close* path and needs no harness on PATH.
    """
    from nvsh.agent import _subprocess as shared
    from nvsh.agent import acp, agy, codex, pi

    # Keep every rung short: this test is about *reaching* the last one, not
    # about how patiently each adapter waits at the first.
    monkeypatch.setattr(shared, "CLOSE_WAIT_SECONDS", 0.2)
    for module, name in (
        (pi, "_CLOSE_WAIT_SECONDS"),
        (acp, "_CLOSE_WAIT_SECONDS"),
        (codex, "_CLOSE_WAIT_SECONDS"),
        (agy, "_TERMINATE_TIMEOUT_SECONDS"),
    ):
        monkeypatch.setattr(module, name, 0.2)

    bindir = _long_running_binary(tmp_path, f"stubborn-{attribute}", trap_sigterm=True)
    proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        [sys.executable, str(bindir / f"stubborn-{attribute}")],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    agent = build()
    setattr(agent, attribute, proc)
    try:
        agent.close()
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a regression
            proc.kill()
            proc.wait(timeout=5)
    assert proc.poll() is not None, "close() left the child running"
    assert not _alive(proc.pid)


def test_targeted_config_folds_the_target_into_the_agents_table() -> None:
    """The client's half of the same fold the daemon does, so a one-shot and
    a warm run hand the adapter identical settings."""
    config = Config(agents={"claude": {"model": "sonnet", "extra_args": ["--x"]}})
    folded = client_transport.targeted_config(
        config, Target(backend="claude", model="opus", effort="high")
    )
    assert folded.agent_provider == "claude"
    assert folded.agents["claude"] == {
        "model": "opus",
        "effort": "high",
        "extra_args": ["--x"],
    }
    # The caller's config is untouched -- one request's target must not
    # become every later request's default.
    assert config.agents["claude"]["model"] == "sonnet"


def test_replace_keeps_a_request_otherwise_identical() -> None:
    """``_stream_request`` stamps the target onto the request it was given."""
    request = _failure()
    stamped = replace(request, target=Target(backend="claude"))
    assert stamped.command == request.command
    assert stamped.target == Target(backend="claude")


# --- the resolved target reaches the panel and the audit log (t16/t18) -----


def _drive_stream_request(monkeypatch: pytest.MonkeyPatch, config: Config, **kwargs):
    """Run ``client._stream_request`` against a stub transport.

    Returns ``(panel_lines, captured_audit)`` -- the panel's rendered lines
    and the audit object the proposal handler was actually given.
    """
    import io

    from nvsh import client
    from nvsh.panel import Panel

    captured: dict = {}
    real_handler = client._proposal_handler

    def spy(panel, **handler_kwargs):
        captured["audit"] = handler_kwargs["audit"]
        return real_handler(panel, **handler_kwargs)

    monkeypatch.setattr(client, "_proposal_handler", spy)

    sent: list[AgentRequest] = []

    def fake_send(request, context, **_kwargs):
        sent.append(request)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="hi")
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client, "_send", fake_send)
    out = io.StringIO()
    panel = Panel(out=out, in_=io.StringIO(""), env={"NO_COLOR": "1", "TERM": "dumb"}, isatty=False)
    client._stream_request(
        panel,
        AgentRequest(kind=RequestKind.FAILURE, command="ls /nope"),
        AgentContext(),
        env={},
        shell_id=1,
        config=config,
        approvals=client._load_approvals(),
        inspections=[],
        **kwargs,
    )
    return out.getvalue().splitlines(), captured, sent


def test_the_panel_header_names_the_resolved_default_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(agent_provider="pi", agents={"pi": {"model": "associate"}})
    lines, _captured, sent = _drive_stream_request(monkeypatch, config)
    assert lines[0] == "pi/associate · rpc · warm"
    assert sent[0].target is None  # nothing named -> the warm default


def test_the_panel_header_names_an_explicit_target_as_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(agents={"claude": {}})
    target = Target(backend="claude", model="opus", effort="high", alias="fast")
    lines, _captured, sent = _drive_stream_request(
        monkeypatch, config, one_shot=True, target=target
    )
    assert lines[0] == "claude/opus/high · stream-json · one-shot"
    assert sent[0].target == target  # and it rides on the request


def test_the_panel_header_calls_the_named_default_target_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``@default hi`` names a target but still rides the warm session (c25)."""
    from nvsh import client

    config = Config(agent_provider="pi", agents={"pi": {"model": "associate"}})
    target = Target(backend="pi", model="associate", alias="default")
    lines, _captured, _sent = _drive_stream_request(
        monkeypatch, config, one_shot=client._is_one_shot(target), target=target
    )
    assert lines[0] == "pi/associate · rpc · warm"


def test_every_audit_line_of_a_turn_carries_the_resolved_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nvsh.agent.audit import AuditLog
    from nvsh.agent.base import Proposal, ProposalKind

    audit = AuditLog(path=tmp_path / "audit.jsonl")
    config = Config(agent_provider="pi", agents={"pi": {"model": "associate"}})
    _lines, captured, _sent = _drive_stream_request(monkeypatch, config, audit=audit)

    captured["audit"].record(
        event="decision",
        proposal=Proposal("ls", "look", ProposalKind.INSPECT),
        decision="once",
    )
    entries = audit.read_all()
    assert entries[-1]["target"] == {"backend": "pi", "model": "associate", "alias": "default"}
