"""Client wiring for the two-press stop (task t16 of reliable-agent-stop).

The panel (t14) calls ``cancel`` on the first Ctrl+C and ``force_stop`` on
the second; this is where the client decides what those two callables reach:

* one-shot -- the adapter built in *this* process by
  :func:`nvsh.client_transport.one_shot` (bound onto the ``Responder``):
  its own ``cancel()`` then ``force_stop()``. Since pi/codex/acp/agy-warm
  now spawn in their own session, terminal SIGINT no longer reaches the
  harness, so without this nothing stops a one-shot turn (plan risk r1).
* daemon -- ``client_transport.cancel`` then ``client_transport.kill``.

The one-shot cases run the real ``CodexAgent`` against the
``tests/fakes/codex-app-server`` fake with ``NVSH_FAKE_IGNORE_CANCEL=1`` (it
drops ``turn/interrupt``) and ``NVSH_FAKE_GRANDCHILD=1`` (a ``sleep 600``
grandchild), parked on an approval request so the turn is genuinely open
when the operator presses Ctrl+C.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from nvsh import client, client_transport
from nvsh.agent import registry
from nvsh.agent.base import AgentContext, AgentEvent, AgentRequest, EventKind, RequestKind
from nvsh.agent.codex import CodexAgent
from nvsh.config import Config
from nvsh.panel import Panel

FAKES_DIR = Path(__file__).parent / "fakes"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie still answers kill(0); it is dead for our purposes
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _wait_gone(pids: list[int], within: float = 5.0) -> list[int]:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if _pid_alive(pid)]


def _wait_for_pid_file(path: Path, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


def _panel() -> Panel:
    return Panel(out=io.StringIO(), in_=io.StringIO(), env={"NO_COLOR": "1"}, isatty=False)


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt="why did it fail?")


def _press_when(event: threading.Event, errors: list[str], timeout: float = 10.0) -> None:
    """SIGINT this process once ``event`` is set (the panel's handler takes it)."""
    if not event.wait(timeout):
        errors.append("the awaited moment never came")
        return
    time.sleep(0.05)
    os.kill(os.getpid(), signal.SIGINT)


class _SpyCodex(CodexAgent):
    """The real codex adapter, recording which stop method the client reached."""

    def __init__(self, env: dict[str, str], *, forward_cancel: bool = True) -> None:
        super().__init__({}, binary="codex-app-server", env=env)
        self.calls: list[str] = []
        self.teardown: list[str] = []
        self.proposed = threading.Event()
        self.cancelled = threading.Event()
        self._forward_cancel = forward_cancel

    def run(self, request, context):
        for event in super().run(request, context):
            if event.kind is EventKind.PROPOSAL:
                self.proposed.set()
            yield event

    # force_stop() and close() reuse cancel() internally; only the calls the
    # client itself made are recorded.
    _inside = False

    def cancel(self) -> None:
        if self._inside:
            super().cancel()
            return
        self._record("cancel")
        if self._forward_cancel:
            super().cancel()
        self.cancelled.set()

    def force_stop(self) -> None:
        self._record("force_stop")
        self._within(super().force_stop)

    def _record(self, name: str) -> None:
        # The panel calls a press's stop method from the main thread; the
        # one-shot teardown runs on the panel's event-feeder thread.
        if threading.current_thread() is threading.main_thread():
            self.calls.append(name)
        else:
            self.teardown.append(name)

    def close(self) -> None:
        self._within(super().close)

    def _within(self, action) -> None:
        self._inside = True
        try:
            action()
        finally:
            self._inside = False


@pytest.fixture
def codex_one_shot(monkeypatch, tmp_path):
    """Route one-shot adapter construction to a spy codex on the ignoring fake."""
    pid_file = tmp_path / "pids.json"
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env.update(
        {
            "NVSH_FAKE_IGNORE_CANCEL": "1",
            "NVSH_FAKE_GRANDCHILD": "1",
            "NVSH_FAKE_PID_FILE": str(pid_file),
            "NVSH_FAKE_CODEX_APPROVAL_TIMEOUT": "20",
        }
    )
    built: list[_SpyCodex] = []

    def install(*, forward_cancel: bool = True) -> list[_SpyCodex]:
        def factory(_cfg):
            agent = _SpyCodex(env, forward_cancel=forward_cancel)
            built.append(agent)
            return agent

        spec = dataclasses.replace(registry.ADAPTERS["codex"], factory=factory)
        monkeypatch.setitem(registry.ADAPTERS, "codex", spec)
        monkeypatch.setattr(registry, "choose", lambda cfg, forced=None: ("codex", "test"))
        return built

    # The daemon must never be asked on the one-shot path.
    monkeypatch.setattr(client_transport, "cancel", _forbidden("cancel"))
    monkeypatch.setattr(client_transport, "kill", _forbidden("kill"))
    yield install, pid_file
    for agent in built:
        agent.close()


def _forbidden(name: str):
    def call(**_kwargs):
        raise AssertionError(f"one-shot stop reached the daemon's {name}")

    return call


def _one_shot_stream(tmp_path: Path):
    return client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path), "XDG_STATE_HOME": str(tmp_path)},
        shell_id=os.getpid(),
        config=Config(),
        one_shot=True,
    )


def test_one_shot_first_press_calls_in_process_cancel(codex_one_shot, tmp_path):
    install, pid_file = codex_one_shot
    built = install()
    errors: list[str] = []
    armed = threading.Event()

    def press() -> None:
        deadline = time.monotonic() + 10
        while not built and time.monotonic() < deadline:
            time.sleep(0.01)
        if not built:
            errors.append("no adapter was built")
            return
        built[0].proposed.wait(10)
        armed.set()
        _press_when(armed, errors)

    presser = threading.Thread(target=press, daemon=True)
    presser.start()
    result = _one_shot_stream(tmp_path)
    presser.join(5)

    assert errors == []
    assert result.interrupted is True
    assert built[0].calls == ["cancel"]
    # The polite stop still ends in a tree kill when the client tears down.
    assert built[0].teardown == ["force_stop"]
    pids = _wait_for_pid_file(pid_file)
    # The stream ended and the client closed the adapter: nothing survives it.
    assert _wait_gone([pids["harness"]]) == [], "harness"
    assert _wait_gone([pids["grandchild"]]) == [], "grandchild"


def test_one_shot_second_press_calls_in_process_force_stop(codex_one_shot, tmp_path):
    install, pid_file = codex_one_shot
    # cancel() records but sends nothing, so the harness (which already drops
    # turn/interrupt) keeps the turn open: only force_stop() can end it.
    built = install(forward_cancel=False)
    errors: list[str] = []

    def press() -> None:
        deadline = time.monotonic() + 10
        while not built and time.monotonic() < deadline:
            time.sleep(0.01)
        if not built:
            errors.append("no adapter was built")
            return
        _press_when(built[0].proposed, errors)
        _press_when(built[0].cancelled, errors)

    presser = threading.Thread(target=press, daemon=True)
    presser.start()
    started = time.monotonic()
    result = _one_shot_stream(tmp_path)
    presser.join(5)

    assert errors == []
    assert result.interrupted is True
    assert built[0].calls == ["cancel", "force_stop"]
    assert time.monotonic() - started < 30
    pids = _wait_for_pid_file(pid_file)
    assert _wait_gone([pids["harness"], pids["grandchild"]]) == []


# -- daemon path --------------------------------------------------------------


def test_daemon_second_press_sends_kill_exactly_once(monkeypatch, tmp_path):
    calls: list[tuple[str, object]] = []
    first = threading.Event()
    killed = threading.Event()

    def fake_send(request, context, **kwargs):
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        first.set()
        killed.wait(10)

    def fake_cancel(*, shell_id=None, env=None):
        calls.append(("cancel", shell_id))
        return True

    def fake_kill(*, shell_id=None, env=None):
        calls.append(("kill", shell_id))
        killed.set()
        return True

    monkeypatch.setattr(client_transport, "send", fake_send)
    monkeypatch.setattr(client_transport, "cancel", fake_cancel)
    monkeypatch.setattr(client_transport, "kill", fake_kill)

    errors: list[str] = []
    cancelled = threading.Event()

    def press() -> None:
        _press_when(first, errors)
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled.set()
        _press_when(cancelled, errors)

    presser = threading.Thread(target=press, daemon=True)
    presser.start()
    result = client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path)},
        shell_id=4242,
        config=Config(),
    )
    presser.join(5)

    assert errors == []
    assert result.interrupted is True
    assert calls == [("cancel", 4242), ("kill", 4242)]
    assert [name for name, _ in calls].count("kill") == 1


def test_daemon_fallback_to_one_shot_stops_the_bound_agent(monkeypatch, tmp_path):
    """send() may give up on the daemon mid-call and bind an in-process agent."""

    class Agent:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.stopped = threading.Event()

        def cancel(self) -> None:
            self.calls.append("cancel")

        def force_stop(self) -> None:
            self.calls.append("force_stop")
            self.stopped.set()

    agent = Agent()
    first = threading.Event()

    def fake_send(request, context, *, responder=None, **kwargs):
        responder.bind_agent(agent)
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        first.set()
        agent.stopped.wait(10)

    monkeypatch.setattr(client_transport, "send", fake_send)
    monkeypatch.setattr(client_transport, "cancel", _forbidden("cancel"))
    monkeypatch.setattr(client_transport, "kill", _forbidden("kill"))

    errors: list[str] = []
    cancelled = threading.Event()

    def press() -> None:
        _press_when(first, errors)
        deadline = time.monotonic() + 5
        while not agent.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled.set()
        _press_when(cancelled, errors)

    presser = threading.Thread(target=press, daemon=True)
    presser.start()
    result = client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path)},
        shell_id=4242,
        config=Config(),
    )
    presser.join(5)

    assert errors == []
    assert result.interrupted is True
    assert agent.calls == ["cancel", "force_stop"]


# -- t17: busy prompt, declined exit code and stop audit ----------------------


@pytest.fixture
def stop_env(tmp_path, monkeypatch):
    """XDG dirs under tmp, both in os.environ (config/approvals) and passed as env."""
    for name in ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR"):
        path = tmp_path / name.lower()
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("NVSH_SHELL_PID", "4242")
    monkeypatch.setattr(client, "_platform_block", lambda: "platform: generic")
    return {
        "XDG_STATE_HOME": str(tmp_path / "xdg_state_home"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg_config_home"),
        "XDG_RUNTIME_DIR": str(tmp_path / "xdg_runtime_dir"),
        "NVSH_SHELL_PID": "4242",
    }


def _typed_panel(typed: str) -> Panel:
    return Panel(out=io.StringIO(), in_=io.StringIO(typed), env={"NO_COLOR": "1"}, isatty=False)


def _stops(env: dict[str, str]) -> list[dict]:
    from nvsh.agent.audit import AuditLog

    return [entry for entry in AuditLog(env=env).read_all() if entry["event"] == "stop"]


def _busy_event(steerable: bool) -> AgentEvent:
    choices = ["steer", "replace", "exit"] if steerable else ["replace", "exit"]
    return AgentEvent(
        kind=EventKind.BUSY,
        text="a turn is still running",
        args={"owner": "4242", "elapsed": 42.0, "steerable": steerable, "choices": choices},
    )


def _busy_daemon(monkeypatch, *, steerable: bool, accept: bool = True):
    """Fake daemon: BUSY, then whatever the chosen control would produce."""
    chosen: list[str] = []
    answered = threading.Event()

    def fake_busy_choice(choice, *, shell_id=None, env=None):
        chosen.append(choice)
        answered.set()
        return accept

    def fake_send(request, context, **kwargs):
        yield _busy_event(steerable)
        if not answered.wait(5):
            yield AgentEvent(kind=EventKind.ERROR, error="no busy choice arrived")
            return
        if not accept:
            # A daemon that already fell back to the queue: nothing more comes.
            time.sleep(30)
            return
        yield AgentEvent(kind=EventKind.STATUS, text="ok")
        if chosen[-1] == "replace":
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text="fresh answer")
        yield AgentEvent(kind=EventKind.DONE, args={"busy_choice": chosen[-1]})

    monkeypatch.setattr(client_transport, "send", fake_send)
    monkeypatch.setattr(client_transport, "busy_choice", fake_busy_choice)
    return chosen


def _assert_stop_shape(entry: dict, kind: str) -> None:
    assert entry["kind"] == kind
    assert entry["shell"] == 4242
    assert "target" in entry
    assert isinstance(entry["elapsed"], (int, float))
    assert entry["outcome"] not in (None, "")


def test_busy_exit_makes_ask_exit_declined_and_audits_once(stop_env, monkeypatch):
    from nvsh.cli._errors import EXIT_DECLINED

    chosen = _busy_daemon(monkeypatch, steerable=True)
    code = client.ask("why?", panel=_typed_panel("\x1b\n"), env=stop_env)

    assert code == EXIT_DECLINED == 3
    assert chosen == ["exit"]
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["busy_exit", "declined"]
    for entry in stops:
        _assert_stop_shape(entry, entry["kind"])
    assert stops[0]["elapsed"] == 42.0


def test_busy_exit_with_the_prompt_already_closed_still_ends_declined(stop_env, monkeypatch):
    chosen = _busy_daemon(monkeypatch, steerable=False, accept=False)
    panel = _typed_panel("\x1b\n")
    started = time.monotonic()
    code = client.ask("why?", panel=panel, env=stop_env)

    assert code == 3
    assert time.monotonic() - started < 10
    assert chosen == ["exit"]
    assert [entry["kind"] for entry in _stops(stop_env)] == ["busy_exit", "declined"]


def test_busy_steer_sends_the_steer_control_and_audits_it(stop_env, monkeypatch):
    chosen = _busy_daemon(monkeypatch, steerable=True)
    code = client.ask("why?", panel=_typed_panel("t\n"), env=stop_env)

    assert code == 0
    assert chosen == ["steer"]
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["steer"]
    _assert_stop_shape(stops[0], "steer")


def test_busy_replace_sends_the_replace_control_and_audits_it(stop_env, monkeypatch):
    chosen = _busy_daemon(monkeypatch, steerable=False)
    panel = _typed_panel("r\n")
    code = client.ask("why?", panel=panel, env=stop_env)

    assert code == 0
    assert chosen == ["replace"]
    assert "fresh answer" in panel.out.getvalue()
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["replace"]
    _assert_stop_shape(stops[0], "replace")


def _proposal_daemon(monkeypatch):
    from nvsh.agent.base import Proposal, ProposalKind

    proposal = Proposal(command="sudo reboot", kind=ProposalKind.FIX, rationale="try it")

    def fake_send(request, context, **kwargs):
        yield AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal)
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "send", fake_send)


def test_proposal_ignore_makes_ask_exit_declined_and_audits_once(stop_env, monkeypatch):
    _proposal_daemon(monkeypatch)
    code = client.ask("why?", panel=_typed_panel("\x1b\n"), env=stop_env)

    assert code == 3
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["declined"]
    _assert_stop_shape(stops[0], "declined")


def test_proposal_ignore_makes_handle_failure_exit_declined(stop_env, monkeypatch):
    import types

    _proposal_daemon(monkeypatch)
    args = types.SimpleNamespace(
        exit=2, pipestatus="2", line="ls /nope", cwd=stop_env["XDG_STATE_HOME"], log="", json=False
    )
    code = client.handle_failure(args, panel=_typed_panel("q\n"), env=stop_env)

    assert code == 3
    assert [entry["kind"] for entry in _stops(stop_env)] == ["declined"]


def test_slash_json_reports_declined_exit_code_but_exits_zero(stop_env, monkeypatch, capsys):
    from nvsh.cli import main

    _busy_daemon(monkeypatch, steerable=False)
    monkeypatch.delenv("NVSH_DRAFT", raising=False)
    monkeypatch.setattr(client, "_panel_for", lambda panel, env: _typed_panel("\x1b\n"))
    code = main(["slash", "--json", "--platform", "generic", "/ask why?"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {"command": "ask", "exit_code": 3}


def test_ctrl_c_during_work_exits_130_and_audits_cancel_and_force_kill_once(stop_env, monkeypatch):
    first = threading.Event()
    killed = threading.Event()
    calls: list[str] = []

    def fake_send(request, context, **kwargs):
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        first.set()
        killed.wait(10)

    def fake_cancel(*, shell_id=None, env=None):
        calls.append("cancel")
        return True

    def fake_kill(*, shell_id=None, env=None):
        calls.append("kill")
        killed.set()
        return True

    monkeypatch.setattr(client_transport, "send", fake_send)
    monkeypatch.setattr(client_transport, "cancel", fake_cancel)
    monkeypatch.setattr(client_transport, "kill", fake_kill)

    errors: list[str] = []
    cancelled = threading.Event()

    def press() -> None:
        _press_when(first, errors)
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled.set()
        _press_when(cancelled, errors)

    presser = threading.Thread(target=press, daemon=True)
    presser.start()
    code = client.ask("why?", panel=_typed_panel(""), env=stop_env)
    presser.join(5)

    assert errors == []
    assert code == 130
    assert calls == ["cancel", "kill"]
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["cancel", "force_kill"]
    for entry in stops:
        _assert_stop_shape(entry, entry["kind"])


def test_single_ctrl_c_exits_130_and_audits_only_cancel(stop_env, monkeypatch):
    first = threading.Event()
    stopped = threading.Event()

    def fake_send(request, context, **kwargs):
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        first.set()
        stopped.wait(10)
        yield AgentEvent(kind=EventKind.DONE)

    def fake_cancel(*, shell_id=None, env=None):
        stopped.set()
        return True

    monkeypatch.setattr(client_transport, "send", fake_send)
    monkeypatch.setattr(client_transport, "cancel", fake_cancel)
    monkeypatch.setattr(client_transport, "kill", _forbidden("kill"))

    errors: list[str] = []
    presser = threading.Thread(target=_press_when, args=(first, errors), daemon=True)
    presser.start()
    code = client.ask("why?", panel=_typed_panel(""), env=stop_env)
    presser.join(5)

    assert errors == []
    assert code == 130
    assert [entry["kind"] for entry in _stops(stop_env)] == ["cancel"]


# --- Ctrl+C typed at a raw-mode prompt (PR #16 review, Qodo 4) --------------


@pytest.mark.parametrize("prompt", ["proposal", "busy"])
def test_ctrl_c_at_a_raw_prompt_exits_130_cancels_once_and_never_declines(
    stop_env, monkeypatch, prompt
):
    import pty

    from nvsh.agent.audit import AuditLog
    from nvsh.agent.base import Proposal, ProposalKind

    master, slave = pty.openpty()
    tty_in = os.fdopen(slave, "rb", buffering=0)
    stopped = threading.Event()
    calls: list[str] = []

    def fake_send(request, context, **kwargs):
        if prompt == "proposal":
            proposal = Proposal(command="sudo reboot", kind=ProposalKind.FIX, rationale="x")
            yield AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal)
        else:
            yield _busy_event(steerable=True)
        stopped.wait(5)
        yield AgentEvent(kind=EventKind.DONE)

    def fake_cancel(*, shell_id=None, env=None):
        calls.append("cancel")
        stopped.set()
        return True

    monkeypatch.setattr(client_transport, "send", fake_send)
    monkeypatch.setattr(client_transport, "cancel", fake_cancel)
    monkeypatch.setattr(client_transport, "kill", _forbidden("kill"))
    monkeypatch.setattr(client_transport, "busy_choice", _forbidden("busy_choice"))
    panel = Panel(out=io.StringIO(), in_=tty_in, env={"NO_COLOR": "1"}, isatty=True)
    # After raw mode is entered (tty.setraw flushes earlier input).
    typist = threading.Timer(0.5, lambda: os.write(master, b"\x03"))
    typist.start()
    try:
        code = client.ask("why?", panel=panel, env=stop_env)
    finally:
        typist.cancel()
        tty_in.close()
        os.close(master)

    assert code == 130
    assert calls == ["cancel"]
    assert [entry["kind"] for entry in _stops(stop_env)] == ["cancel"]
    events = [entry["event"] for entry in AuditLog(env=stop_env).read_all()]
    assert "decision" not in events


def test_with_prompt_keeps_every_field_but_the_prompt():
    """The follow-up request copies every ``AgentRequest`` field except the prompt."""
    import dataclasses

    from nvsh.agent.base import AgentRequest, RequestKind, Target
    from nvsh.client import _with_prompt

    original = AgentRequest(
        kind=RequestKind.FAILURE,
        prompt="old",
        command="ls /nope",
        exit_code=2,
        failure_id="f1",
        ask="why",
        target=Target(backend="pi"),
    )
    updated = _with_prompt(original, "new")
    assert updated.prompt == "new"
    assert dataclasses.replace(updated, prompt="old") == original
