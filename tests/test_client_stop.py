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

Since the stop-choice prompt (t5-t8) there is a second thing to decide: not
every stop is an operator stop. ``[t] stop & correct`` cancels the running
turn so the operator's correction can be resent as a *self-contained* next
request, and that is not 130 -- the exit status is the follow-up turn's own.
Only a plain ``[s]``, and the kill press that may follow either kind of
stop, still exits 130. The t8 cases below drive the real client against a
scripted ``Panel.stream``; the panel's own half of the same story is proven
on a pty in ``tests/test_panel_stop.py``.
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
from nvsh.agent.base import AgentContext, AgentEvent, AgentRequest, EventKind, RequestKind, Target
from nvsh.agent.codex import CodexAgent
from nvsh.config import Config
from nvsh.panel import Panel, StreamResult
from tests.test_agent_stop import FAMILIES as _FAMILIES

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
    """Rewritten for t5-t8: this panel is not a tty, so no choice prompt can
    be shown and the first press stops at once, exactly as before (spec c9).
    It is a plain ``[s]``-equivalent stop and never a stop-and-correct, so
    ``interrupted`` is set and ``stopped_to_correct`` is not."""
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
    assert result.stopped_to_correct is False
    assert result.not_running is False
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
    """Rewritten for t8: a stop with no prompt to answer on (this panel is
    not a tty) is still the plain two-press stop -- 130, one ``cancel`` and
    one ``force_kill`` -- and the ``cancel`` line carries neither ``origin``
    nor ``reason``, which is what tells it from a stop-and-correct's cancel.
    No follow-up request is sent."""
    first = threading.Event()
    killed = threading.Event()
    calls: list[str] = []
    requests: list[object] = []

    def fake_send(request, context, **kwargs):
        requests.append(request)
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
    assert len(requests) == 1
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["cancel", "force_kill"]
    for entry in stops:
        _assert_stop_shape(entry, entry["kind"])
    assert "origin" not in stops[0] and "reason" not in stops[0]


def test_single_ctrl_c_exits_130_and_audits_only_cancel(stop_env, monkeypatch):
    """Rewritten for t8: one press, no prompt, one ``cancel`` -- and, since
    nothing was steered, no ``steer`` line, no second request and no
    stop-and-correct markers on the cancel line."""
    first = threading.Event()
    stopped = threading.Event()
    requests: list[object] = []

    def fake_send(request, context, **kwargs):
        requests.append(request)
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
    assert len(requests) == 1
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["cancel"]
    assert "origin" not in stops[0] and "reason" not in stops[0]


# --- Ctrl+C typed at a raw-mode prompt (PR #16 review, Qodo 4) --------------


@pytest.mark.parametrize("prompt", ["proposal", "busy"])
def test_ctrl_c_at_a_raw_prompt_exits_130_cancels_once_and_never_declines(
    stop_env, monkeypatch, prompt
):
    """Rewritten for t5-t8: a Ctrl+C typed while *another* nvsh prompt is
    open still goes straight to stop -- the choice prompt only opens for a
    press made while the panel is streaming (spec decision). So this is a
    plain stop: 130, one cancel with no stop-prompt markers, no steer line
    and no follow-up request."""
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
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["cancel"]
    assert "origin" not in stops[0] and "reason" not in stops[0]
    events = [entry["event"] for entry in AuditLog(env=stop_env).read_all()]
    assert "decision" not in events


# -- t7: the stop-choice prompt's [t] label, redaction and audit -------------


def _drained(events) -> None:
    for _ in events:
        pass


def _stub_stream(monkeypatch, *, drive=None):
    """Replace ``Panel.stream`` with a spy that records the kwargs the client
    passed it and, optionally, drives ``on_choice``/``on_steer`` itself --
    the pattern the task brief asks for instead of a pty."""
    captured: dict = {}

    def fake_stream(self, events, **kwargs):
        captured.update(kwargs)
        _drained(events)
        if drive is not None:
            drive(captured)
        return StreamResult()

    monkeypatch.setattr(Panel, "stream", fake_stream)
    return captured


def _done_send(request, context, **kwargs):
    yield AgentEvent(kind=EventKind.DONE)


def test_steer_label_resolved_from_capability_on_daemon_path(monkeypatch, tmp_path):
    """c31/c57: no daemon message carries the capability -- the client reads
    it itself via ``registry.steer_capable``, against a stub daemon
    (``fake_send``) that never sends anything capability-shaped."""
    captured = _stub_stream(monkeypatch)
    monkeypatch.setattr(client_transport, "send", _done_send)

    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path)},
        shell_id=4242,
        config=Config(),
        target=Target(backend="pi"),
    )
    assert captured["steer_label"] == "steer"
    assert captured["stop_prompt"] is True
    assert callable(captured["on_choice"])
    assert callable(captured["on_steer"])

    captured.clear()
    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path)},
        shell_id=4242,
        config=Config(),
        target=Target(backend="claude"),
    )
    assert captured["steer_label"] == "stop & correct"


def test_steer_label_resolved_from_capability_on_one_shot_path(monkeypatch, tmp_path):
    captured = _stub_stream(monkeypatch)
    monkeypatch.setattr(client_transport, "one_shot", _done_send)

    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path)},
        shell_id=4242,
        config=Config(),
        target=Target(backend="codex"),
        one_shot=True,
    )
    assert captured["steer_label"] == "steer"

    captured.clear()
    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path)},
        shell_id=4242,
        config=Config(),
        target=Target(backend="openai-compat"),
        one_shot=True,
    )
    assert captured["steer_label"] == "stop & correct"


def test_correction_is_redacted_before_delivery_and_never_reaches_the_audit_log(
    stop_env, monkeypatch
):
    """c29/c52: a token typed at the stop prompt reaches the adapter
    redacted and never appears in audit.jsonl, only its length does."""
    delivered: list[str] = []

    def fake_steer(text, *, shell_id=None, env=None):
        delivered.append(text)
        return True

    def note(*, captured):
        assert captured["on_steer"]("HF_TOKEN=abc123secret\nretry with that") is True

    _stub_stream(monkeypatch, drive=lambda c: note(captured=c))
    monkeypatch.setattr(client_transport, "send", _done_send)
    monkeypatch.setattr(client_transport, "steer", fake_steer)

    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env=stop_env,
        shell_id=4242,
        config=Config(),
        target=Target(backend="pi"),
        audit=client._audit(stop_env),
    )

    assert delivered, "Responder.steer was never reached"
    assert "abc123secret" not in delivered[0]
    assert "HF_TOKEN" in delivered[0]  # the key name survives redaction; the value doesn't

    from nvsh.agent.audit import AuditLog

    raw = Path(AuditLog(env=stop_env).path).read_text(encoding="utf-8")
    assert "abc123secret" not in raw

    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["steer"]
    assert stops[0]["origin"] == "stop_prompt"
    assert stops[0]["correction_chars"] == len(delivered[0])
    assert "correction" not in stops[0]


def test_steer_not_delivered_queues_onto_the_turns_steers_list(stop_env, monkeypatch):
    """A harness with no mid-turn channel (Capabilities.steer False) queues
    the redacted text onto the same ``steers`` list ``_injector`` uses,
    rather than losing it. Since t8 it also answers ``STOP_BEGUN``: the
    panel is being asked to stop this turn so the queued text can be
    resent."""
    from nvsh.panel import STOP_BEGUN

    def fake_steer(text, *, shell_id=None, env=None):
        return False

    def drive(captured):
        assert captured["on_steer"]("do the other thing") == STOP_BEGUN

    _stub_stream(monkeypatch, drive=drive)
    monkeypatch.setattr(client_transport, "send", _done_send)
    monkeypatch.setattr(client_transport, "steer", fake_steer)

    steers: list[str] = []
    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env=stop_env,
        shell_id=4242,
        config=Config(),
        target=Target(backend="claude"),
        steers=steers,
        audit=client._audit(stop_env),
    )
    assert steers == ["do the other thing"]
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["steer"]
    assert stops[0]["outcome"] == "queued"


@pytest.mark.parametrize("reason", ["key", "timeout"])
def test_keep_going_writes_exactly_one_audit_line_with_its_reason(stop_env, monkeypatch, reason):
    from nvsh.panel import KEEP_GOING

    def drive(captured):
        captured["on_choice"](KEEP_GOING, reason)

    _stub_stream(monkeypatch, drive=drive)
    monkeypatch.setattr(client_transport, "send", _done_send)

    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env=stop_env,
        shell_id=4242,
        config=Config(),
        target=Target(backend="pi"),
        audit=client._audit(stop_env),
    )
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["keep_going"]
    assert stops[0]["origin"] == "stop_prompt"
    assert stops[0]["reason"] == reason


def test_keep_going_at_eof_omits_the_reason_field(stop_env, monkeypatch):
    """The panel's EOF reason is ``""`` (spec c34); it maps to no ``reason``
    field rather than an empty string, per the task's mapping rule."""
    from nvsh.panel import KEEP_GOING, REASON_NONE

    def drive(captured):
        captured["on_choice"](KEEP_GOING, REASON_NONE)

    _stub_stream(monkeypatch, drive=drive)
    monkeypatch.setattr(client_transport, "send", _done_send)

    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env=stop_env,
        shell_id=4242,
        config=Config(),
        target=Target(backend="pi"),
        audit=client._audit(stop_env),
    )
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["keep_going"]
    assert "reason" not in stops[0]


def test_stop_outcome_writes_no_extra_audit_line_beyond_the_existing_cancel(stop_env, monkeypatch):
    """[s] at the choice prompt takes the same cancel path as the plain first
    press: the choice prompt must not add a second 'stop' audit line."""
    from nvsh.panel import STOP

    def drive(captured):
        captured["on_choice"](STOP, "key")
        # The panel itself is what would call ``cancel`` on a real [s]
        # press; here we call it directly, exactly as the panel does, to
        # prove the client's ``on_choice`` alone writes nothing.
        captured["cancel"]()

    _stub_stream(monkeypatch, drive=drive)
    monkeypatch.setattr(client_transport, "send", _done_send)
    monkeypatch.setattr(client_transport, "cancel", lambda *, shell_id=None, env=None: True)

    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env=stop_env,
        shell_id=4242,
        config=Config(),
        target=Target(backend="pi"),
        audit=client._audit(stop_env),
    )
    stops = _stops(stop_env)
    assert [entry["kind"] for entry in stops] == ["cancel"]


# -- d2: --json disables the stop-choice prompt -------------------------------


def test_stream_request_json_mode_disables_the_stop_prompt(monkeypatch, tmp_path):
    captured = _stub_stream(monkeypatch)
    monkeypatch.setattr(client_transport, "send", _done_send)

    client._stream_request(
        _panel(),
        _request(),
        AgentContext(),
        env={"XDG_RUNTIME_DIR": str(tmp_path)},
        shell_id=4242,
        config=Config(),
        target=Target(backend="pi"),
        json_mode=True,
    )
    assert captured["stop_prompt"] is False


def test_slash_json_disables_the_stop_prompt_on_ask(stop_env, monkeypatch):
    """'nvsh slash --json /ask ...' reaches _stream_request with
    stop_prompt=False -- the whole point of deviation d2."""
    from nvsh.cli import main

    captured = _stub_stream(monkeypatch)
    monkeypatch.setattr(client_transport, "send", _done_send)

    code = main(["slash", "--json", "--platform", "generic", "/ask why?"])

    assert code == 0
    assert captured["stop_prompt"] is False


def test_hook_json_disables_the_stop_prompt(stop_env, monkeypatch, tmp_path):
    """'nvsh hook --json ...' reaches handle_failure's args.json and turns
    off the stop-choice prompt the same way (d2)."""
    import types

    captured = _stub_stream(monkeypatch)
    monkeypatch.setattr(client_transport, "send", _done_send)

    args = types.SimpleNamespace(
        exit=1,
        pipestatus="1",
        line="ls /nope",
        cwd=str(tmp_path),
        log="",
        json=True,
        failure_id="",
    )
    client.handle_failure(args, panel=_typed_panel(""), env=stop_env)
    assert captured["stop_prompt"] is False


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


# -- t8: stop and correct -----------------------------------------------------
#
# ``[t]`` on a harness with no mid-turn channel is "stop & correct": the
# client cancels the running turn once, waits for it to end, and then sends
# the operator's correction as a *self-contained* next request. Only a plain
# ``[s]`` stop (and the kill that may follow it) still exits 130.
#
# These drive the real client against a scripted ``Panel.stream`` that plays
# the part the panel plays for real -- calling ``on_steer`` and then, for a
# :data:`STOP_BEGUN` answer, the very ``cancel`` callable the client handed
# it (``tests/test_panel_stop.py`` is where the panel's own half is proven
# on a pty).

CORRECTION = "look at nvpmodel instead"


def _scripted_stream(monkeypatch, *steps):
    """Replace ``Panel.stream`` with one step per call.

    Each step is ``step(kwargs) -> StreamResult | None`` and stands for what
    the panel would do with the callbacks the client passed it. Calls past
    the end of the script just drain their events and finish normally.
    """
    calls: list[dict] = []

    def fake_stream(self, events, **kwargs):
        calls.append(kwargs)
        _drained(events)
        index = len(calls) - 1
        if index < len(steps):
            return steps[index](kwargs) or StreamResult(done=True)
        return StreamResult(done=True)

    monkeypatch.setattr(Panel, "stream", fake_stream)
    return calls


def _stop_and_correct(text: str = CORRECTION, *, killed: bool = False):
    """The panel's half of a stop-and-correct, as a scripted step."""

    def step(kwargs) -> StreamResult:
        from nvsh.panel import STOP_BEGUN

        answer = kwargs["on_steer"](text)
        assert answer == STOP_BEGUN, answer
        kwargs["cancel"]()  # exactly what Panel._stop_to_correct does
        # A harness that ignored the cancel is killed by a further press;
        # that press -- not the stop-and-correct -- is what interrupts.
        return StreamResult(done=not killed, interrupted=killed, stopped_to_correct=True)

    return step


def _requests(monkeypatch) -> list[AgentRequest]:
    """Record every request that leaves the client, on both transports.

    ``@name`` targets ride the one-shot path and everything else the
    daemon's, so both are stubbed: no harness process is ever started, which
    is also what makes the "no harness session state" claim testable.
    """
    sent: list[AgentRequest] = []

    def send(request, context, **kwargs):
        sent.append(request)
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "send", send)
    monkeypatch.setattr(client_transport, "one_shot", send)
    monkeypatch.setattr(client_transport, "steer", lambda text, **kwargs: False)
    return sent


def _hook_args(stop_env, json: bool = False):
    import types

    return types.SimpleNamespace(
        exit=2,
        pipestatus="2",
        line="ls /nope",
        cwd=stop_env["XDG_STATE_HOME"],
        log="",
        json=json,
        failure_id="",
    )


def _entry_point(name: str, stop_env):
    """Run one of the three call sites and return its exit code."""
    panel = _typed_panel("")
    if name == "ask":
        return client.ask("why did it fail?", panel=panel, env=stop_env)
    if name == "handle_failure":
        return client.handle_failure(_hook_args(stop_env), panel=panel, env=stop_env)
    # the slash path: /fix, /explain -> client._on_last_failure
    client.save_last_failure(_hook_args(stop_env), env=stop_env)
    return client.fix(panel=panel, env=stop_env)


ENTRY_POINTS = ["handle_failure", "ask", "slash"]


@pytest.fixture
def claude_default(monkeypatch, stop_env):
    """Make ``default`` resolve to a harness that cannot steer mid-turn."""
    config = Path(stop_env["XDG_CONFIG_HOME"]) / "nvsh"
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.toml").write_text('[aliases]\ndefault = "claude"\n', encoding="utf-8")


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_stop_and_correct_cancels_once_then_streams_the_correction(
    stop_env, monkeypatch, claude_default, entry
):
    """Criterion 1: t plus a typed line on a steer=False target cancels the
    turn once and then sends a second request -- no entry point returns 130
    before that follow-up."""
    cancelled: list[int] = []
    monkeypatch.setattr(
        client_transport, "cancel", lambda *, shell_id=None, env=None: cancelled.append(1) or True
    )
    monkeypatch.setattr(client_transport, "kill", _forbidden("kill"))
    sent = _requests(monkeypatch)
    calls = _scripted_stream(monkeypatch, _stop_and_correct())

    code = _entry_point(entry, stop_env)

    assert code == 0
    assert cancelled == [1], "the cancelled turn must be cancelled exactly once"
    assert len(sent) == 2, "the correction must come back as the next request"
    assert len(calls) == 2
    assert CORRECTION in sent[1].prompt
    assert client.STOPPED_NOTICE in sent[1].prompt
    assert sent[0].prompt in sent[1].prompt
    # One line per outcome: the queued steer, and the cancel it caused.
    stops = _stops(stop_env)
    assert [entry_["kind"] for entry_ in stops] == ["steer", "cancel"]
    assert stops[0]["outcome"] == "queued"
    assert stops[0]["origin"] == "stop_prompt"
    # The cancel names the stop prompt as its origin; a plain [s] stop does
    # not, which is what tells a stop-and-correct from an operator stop.
    assert stops[1]["origin"] == "stop_prompt"


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_killed_stop_and_correct_exits_130_and_sends_no_follow_up(
    stop_env, monkeypatch, claude_default, entry
):
    """The harness ignored the cancel and the operator pressed again: the
    kill press is an interruption, so the follow-up is never sent and the
    exit status is 130 -- the operator did ask for this one to stop."""
    monkeypatch.setattr(client_transport, "cancel", lambda *, shell_id=None, env=None: True)
    monkeypatch.setattr(client_transport, "kill", lambda *, shell_id=None, env=None: True)
    sent = _requests(monkeypatch)
    _scripted_stream(monkeypatch, _stop_and_correct(killed=True))

    assert _entry_point(entry, stop_env) == 130
    assert len(sent) == 1


@pytest.mark.parametrize("family", [f.adapter for f in _FAMILIES], ids=[f.name for f in _FAMILIES])
def test_the_follow_up_prompt_is_self_contained_for_every_adapter_family(
    stop_env, monkeypatch, family
):
    """Criterion 2 / spec c23: the follow-up carries the original request,
    the fact that the previous attempt was stopped, and the redacted
    correction -- composed from the client's own AgentRequest, so no harness
    session state is read and no harness process is started."""
    monkeypatch.setattr(client_transport, "cancel", lambda *, shell_id=None, env=None: True)
    sent = _requests(monkeypatch)
    _scripted_stream(monkeypatch, _stop_and_correct("HF_TOKEN=abc123secret retry with that"))
    monkeypatch.setattr(
        registry, "installed", lambda name, config=None, env=None: True, raising=False
    )
    # pi and codex are offered as "steer"; their steer() refusing at runtime
    # (c30) puts them on this same path once the operator agrees.
    monkeypatch.setattr(Panel, "confirm", lambda self, question: True)

    code = client.ask("why did it fail?", panel=_typed_panel(""), env=stop_env, agent=family)

    assert code == 0, family
    assert len(sent) == 2, family
    follow_up = sent[1].prompt
    assert "why did it fail?" in follow_up
    assert client.STOPPED_NOTICE in follow_up
    assert "retry with that" in follow_up
    # Redacted before it ever left the process (c29), on this path too.
    assert "abc123secret" not in follow_up


def test_the_follow_up_prompt_is_identical_across_families(stop_env, monkeypatch):
    """The same proof, stated once: what nvsh composes does not depend on
    which harness is behind it."""
    composed = {
        family.name: client._follow_up_prompt(
            [], ["do the other thing"], stopped=True, original="why did it fail?"
        )
        for family in _FAMILIES
    }
    assert len(set(composed.values())) == 1, composed


def test_not_running_correction_is_a_plain_next_request(stop_env, monkeypatch, claude_default):
    """c36: the turn had already finished when the prompt closed. Nothing is
    cancelled, nothing exits 130, and the correction is simply the next
    request -- with no claim that anything was stopped."""
    monkeypatch.setattr(client_transport, "cancel", _forbidden("cancel"))
    monkeypatch.setattr(client_transport, "kill", _forbidden("kill"))
    sent = _requests(monkeypatch)

    def finished(kwargs) -> StreamResult:
        kwargs["on_steer"](CORRECTION)
        # The panel saw the terminal event already queued: it cancels nothing.
        return StreamResult(done=True, not_running=True)

    _scripted_stream(monkeypatch, finished)

    code = client.ask("why did it fail?", panel=_typed_panel(""), env=stop_env)

    assert code == 0
    assert len(sent) == 2
    assert sent[1].prompt == CORRECTION
    assert client.STOPPED_NOTICE not in sent[1].prompt
    assert [e["kind"] for e in _stops(stop_env)] == ["steer"]


@pytest.fixture
def pi_installed(monkeypatch):
    """``agent="pi"`` is refused unless the harness is installed, and CI has
    no ``pi`` on PATH (a developer box often does -- which is how three of
    these tests passed locally and failed in CI). The tests below are about
    what happens *after* the target resolves, so say it is installed."""
    from nvsh.agent import registry

    monkeypatch.setattr(registry, "installed", lambda name, *args, **kwargs: True)


# -- t8 criterion 3: the runtime fallback (spec c30) --------------------------


def _runtime_refusal(monkeypatch, *, agrees: bool):
    """A steer-capable target whose steer() refuses at runtime."""
    monkeypatch.setattr(client_transport, "steer", lambda text, **kwargs: False)
    asked: list[str] = []

    def confirm(self, question: str) -> bool:
        asked.append(question)
        return agrees

    monkeypatch.setattr(Panel, "confirm", confirm)
    return asked


def test_a_refused_steer_asks_once_and_stop_and_corrects_on_yes(
    pi_installed, stop_env, monkeypatch
):
    asked = _runtime_refusal(monkeypatch, agrees=True)
    cancelled: list[int] = []
    monkeypatch.setattr(
        client_transport, "cancel", lambda *, shell_id=None, env=None: cancelled.append(1) or True
    )
    sent = _requests(monkeypatch)
    _scripted_stream(monkeypatch, _stop_and_correct())
    panel = _typed_panel("")

    # ``pi`` declares Capabilities.steer, so [t] is offered as "steer".
    code = client.ask("why did it fail?", panel=panel, env=stop_env, agent="pi")

    assert code == 0
    assert len(asked) == 1, "the operator is asked exactly once"
    assert client.COULD_NOT_STEER_NOTE in panel.out.getvalue()
    assert cancelled == [1]
    assert len(sent) == 2
    assert CORRECTION in sent[1].prompt
    assert client.STOPPED_NOTICE in sent[1].prompt
    assert [e["kind"] for e in _stops(stop_env)] == ["steer", "cancel"]


def test_a_refused_steer_discards_the_text_on_anything_but_yes(pi_installed, stop_env, monkeypatch):
    """Not a third outcome and never silent: the text is dropped only because
    the operator said so, it is said on the panel, and it is audited."""
    asked = _runtime_refusal(monkeypatch, agrees=False)
    monkeypatch.setattr(client_transport, "cancel", _forbidden("cancel"))
    monkeypatch.setattr(client_transport, "kill", _forbidden("kill"))
    sent = _requests(monkeypatch)

    def refused(kwargs) -> StreamResult:
        assert kwargs["on_steer"](CORRECTION) is False
        return StreamResult(done=True)

    _scripted_stream(monkeypatch, refused)
    panel = _typed_panel("")

    code = client.ask("why did it fail?", panel=panel, env=stop_env, agent="pi")

    assert code == 0, "declining the stop leaves the turn's own status alone"
    assert len(asked) == 1
    assert len(sent) == 1, "nothing is resent"
    text = panel.out.getvalue()
    assert client.COULD_NOT_STEER_NOTE in text
    assert client.CORRECTION_DISCARDED_NOTE in text
    stops = _stops(stop_env)
    assert [e["kind"] for e in stops] == ["steer"]
    assert stops[0]["outcome"] == "discarded"
    assert stops[0]["correction_chars"] == len(CORRECTION)


# -- t8 criterion 4: exit status ---------------------------------------------


def test_a_delivered_steer_exits_zero_with_no_follow_up(pi_installed, stop_env, monkeypatch):
    monkeypatch.setattr(client_transport, "steer", lambda text, **kwargs: True)
    monkeypatch.setattr(client_transport, "cancel", _forbidden("cancel"))
    sent = _requests(monkeypatch)
    monkeypatch.setattr(client_transport, "steer", lambda text, **kwargs: True)

    def delivered(kwargs) -> StreamResult:
        assert kwargs["on_steer"](CORRECTION) is True
        return StreamResult(done=True)

    _scripted_stream(monkeypatch, delivered)

    assert client.ask("why did it fail?", panel=_typed_panel(""), env=stop_env, agent="pi") == 0
    assert len(sent) == 1
    stops = _stops(stop_env)
    assert [e["kind"] for e in stops] == ["steer"]
    assert stops[0]["outcome"] == "delivered"


def test_keep_going_exits_zero(stop_env, monkeypatch):
    from nvsh.panel import KEEP_GOING

    sent = _requests(monkeypatch)
    monkeypatch.setattr(client_transport, "cancel", _forbidden("cancel"))

    def kept(kwargs) -> StreamResult:
        kwargs["on_choice"](KEEP_GOING, "key")
        return StreamResult(done=True)

    _scripted_stream(monkeypatch, kept)

    assert client.ask("why did it fail?", panel=_typed_panel(""), env=stop_env) == 0
    assert len(sent) == 1
    assert [e["kind"] for e in _stops(stop_env)] == ["keep_going"]


def test_the_follow_up_turns_own_status_is_what_stop_and_correct_reports(
    stop_env, monkeypatch, claude_default
):
    """The second turn is a turn like any other: interrupt it and the exit
    status is 130, from that press rather than from the correction."""
    monkeypatch.setattr(client_transport, "cancel", lambda *, shell_id=None, env=None: True)
    sent = _requests(monkeypatch)

    def interrupted(_kwargs) -> StreamResult:
        return StreamResult(interrupted=True)

    _scripted_stream(monkeypatch, _stop_and_correct(), interrupted)

    assert client.ask("why did it fail?", panel=_typed_panel(""), env=stop_env) == 130
    assert len(sent) == 2


def test_json_exit_code_and_the_audit_log_agree_on_a_stop_and_correct(
    stop_env, monkeypatch, capsys, claude_default
):
    """--json turns the prompt off, so this is the report side of the same
    story: the code on stdout is the one the entry point returned."""
    from nvsh.cli import main

    monkeypatch.setattr(client_transport, "cancel", lambda *, shell_id=None, env=None: True)
    sent = _requests(monkeypatch)
    calls = _scripted_stream(monkeypatch, _stop_and_correct())
    monkeypatch.setattr(client, "_panel_for", lambda panel, env: _typed_panel(""))

    assert main(["slash", "--json", "--platform", "generic", "/ask why did it fail?"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload == {"command": "ask", "exit_code": 0}
    assert calls[0]["stop_prompt"] is False
    assert len(sent) == 2
    assert [e["kind"] for e in _stops(stop_env)] == ["steer", "cancel"]


def test_openai_compat_does_not_send_the_correction_twice():
    """openai-compat's steer() queues the text in ``_pending_steer`` before
    the stop-and-correct cancel (t7 calls ``Responder.steer`` first on every
    harness). That queue is only a flag -- it makes the adapter prepend the
    *previous exchange* to the next request -- so the correction itself
    travels exactly once, inside the follow-up prompt nvsh composed."""
    from nvsh.agent.openai_compat import OpenAICompatAgent

    agent = OpenAICompatAgent({"base_url": "http://127.0.0.1:1/v1"})
    agent._last_prompt = "why did it fail?"
    agent._last_reply = "half an answer"
    assert agent.steer(CORRECTION) is False

    follow_up = f"{client.STOPPED_NOTICE}\n\nwhy did it fail?\n\n{CORRECTION}"
    messages = agent._messages(follow_up)

    assert [m["content"] for m in messages] == [
        "why did it fail?",
        "half an answer",
        follow_up,
    ]
    assert sum(m["content"].count(CORRECTION) for m in messages) == 1
    assert agent._pending_steer == []
