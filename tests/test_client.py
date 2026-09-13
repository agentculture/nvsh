"""Tests for nvsh.client (task t13): context assembly, rate state, proposals,
retry/verify and the slash entry points.
"""

from __future__ import annotations

import io
import json
import stat
import types

import pytest

from nvsh import client as client_mod
from nvsh import client_transport
from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind, Proposal, ProposalKind, RequestKind

OSC_C = b"\x1b]133;C\x07"
OSC_D = b"\x1b]133;D;2\x07"


@pytest.fixture()
def xdg(tmp_path, monkeypatch):
    state = tmp_path / "state"
    config = tmp_path / "config"
    state.mkdir()
    config.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("NVSH_NO_DAEMON", "1")
    monkeypatch.setattr(
        client_mod, "_platform_block", lambda: "platform: dgx-spark\n  gb10: yes  [file: /etc/x]"
    )
    return types.SimpleNamespace(state=state, config=config, tmp=tmp_path)


def _args(tmp_path, log: str = "", line: str = "ls /nope", exit_code: int = 2):
    return types.SimpleNamespace(
        exit=exit_code, pipestatus=str(exit_code), line=line, cwd=str(tmp_path), log=log, json=False
    )


def _log_with(tmp_path, body: bytes):
    log = tmp_path / "session.log"
    log.write_bytes(b"prompt$ ls /nope\n" + OSC_C + body + OSC_D)
    return log


def _panel(stdin_text: str = ""):
    return panel_mod.Panel(out=io.StringIO(), in_=io.StringIO(stdin_text), env={}, isatty=False)


# --- context -------------------------------------------------------------


def test_build_context_carries_platform_output_cwd_and_redaction(xdg):
    log = _log_with(xdg.tmp, b"ls: /nope: No such file\nHF_TOKEN=hf_secretsecret\n")
    ctx = client_mod.build_context(_args(xdg.tmp, log=str(log)))
    assert ctx.platform.startswith("platform: ")
    assert "No such file" in ctx.output
    assert "hf_secretsecret" not in ctx.output
    assert ctx.redaction_report
    assert ctx.cwd == str(xdg.tmp)
    assert ctx.shell_pid > 0


def test_build_context_without_a_log_reports_no_capture(xdg):
    ctx = client_mod.build_context(_args(xdg.tmp, log=""))
    assert ctx.output == "no capture"
    assert ctx.redaction_report == ()


# --- last-failure state --------------------------------------------------


def test_save_last_failure_writes_0600_json_and_loads_back(xdg):
    state = client_mod.save_last_failure(_args(xdg.tmp))
    path = client_mod.last_failure_path()
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = client_mod.load_last_failure()
    assert loaded["line"] == "ls /nope"
    assert loaded["exit"] == 2
    assert loaded["failure_id"] == state["failure_id"]
    assert json.loads(path.read_text(encoding="utf-8"))["cwd"] == str(xdg.tmp)


def test_load_last_failure_is_none_when_absent(xdg):
    assert client_mod.load_last_failure() is None


# --- rate limiting persists across processes -----------------------------


def _stub_send(calls, events=None):
    def send(request, context=None, **kwargs):
        calls.append((request, context, kwargs))
        yield from (
            events
            or [AgentEvent(kind=EventKind.TEXT_DELTA, text="hi"), AgentEvent(kind=EventKind.DONE)]
        )

    return send


def test_rate_state_persists_so_a_second_failure_is_rate_limited(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel()) == 0
    assert len(calls) == 1
    assert client_mod.rate_state_path().exists()
    # A second failure inside the window must not reach the agent at all.
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel()) == 0
    assert len(calls) == 1


def test_rate_state_file_records_the_last_auto_call(xdg, monkeypatch):
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    data = json.loads(client_mod.rate_state_path().read_text(encoding="utf-8"))
    assert data["last_auto_call"] > 0


# --- streaming -----------------------------------------------------------


def test_handle_failure_streams_the_answer_into_the_panel(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    p = _panel()
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    out = p.out.getvalue()
    assert "ls /nope failed (exit 2)" in out
    assert "hi" in out
    request, context, kwargs = calls[0]
    assert request.kind is RequestKind.FAILURE
    assert request.command == "ls /nope"
    assert request.exit_code == 2
    assert context.cwd == str(xdg.tmp)
    assert kwargs["autostart"] is False  # NVSH_NO_DAEMON=1


# --- proposals -----------------------------------------------------------


def _proposal_events(proposal):
    return [
        AgentEvent(kind=EventKind.TEXT_DELTA, text="try this"),
        AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal),
        AgentEvent(kind=EventKind.DONE),
    ]


def test_allowlisted_inspect_proposal_runs_and_is_fed_back(xdg, monkeypatch):
    (xdg.config / "nvsh").mkdir(parents=True)
    (xdg.config / "nvsh" / "approved.toml").write_text(
        'user_patterns = [\n  "echo *",\n]\n', encoding="utf-8"
    )
    proposal = Proposal(command="echo inspector-ran", rationale="look", kind=ProposalKind.INSPECT)
    calls = []
    streams = [
        _proposal_events(proposal),
        [AgentEvent(kind=EventKind.TEXT_DELTA, text="done"), AgentEvent(kind=EventKind.DONE)],
    ]

    def send(request, context=None, **kwargs):
        calls.append((request, context, kwargs))
        yield from streams[min(len(calls) - 1, len(streams) - 1)]

    monkeypatch.setattr(client_transport, "send", send)
    p = _panel()
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    assert len(calls) == 2, "the inspector result must be fed back as a follow-up"
    follow_up = calls[1][0]
    assert "inspector-ran" in follow_up.prompt
    assert "echo inspector-ran" in follow_up.prompt


def test_non_allowlisted_proposal_goes_to_the_panel_and_is_not_run(xdg, monkeypatch):
    marker = xdg.tmp / "ran"
    proposal = Proposal(command=f"touch {marker}", rationale="fix it", kind=ProposalKind.FIX)
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls, _proposal_events(proposal)))
    p = _panel("q\n")  # 'q' -> ignore
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    assert not marker.exists()
    assert f"touch {marker}" in p.out.getvalue()
    assert len(calls) == 1


def test_approved_fix_proposal_runs_only_after_enter(xdg, monkeypatch):
    marker = xdg.tmp / "ran"
    proposal = Proposal(command=f"touch {marker}", rationale="fix it", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    p = _panel("\n")  # Enter -> approve
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    assert marker.exists()


def test_sudo_proposal_is_displayed_verbatim_and_never_run_without_enter(xdg, monkeypatch):
    proposal = Proposal(command="sudo nvpmodel -m 0", rationale="power mode", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd))
    p = _panel("\x1b")  # Esc -> ignore
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    assert "sudo nvpmodel -m 0" in p.out.getvalue()
    assert executed == []


def test_sudo_is_never_auto_run_even_if_a_pattern_allows_it(xdg, monkeypatch):
    (xdg.config / "nvsh").mkdir(parents=True)
    (xdg.config / "nvsh" / "approved.toml").write_text(
        'user_patterns = [\n  "* *",\n]\n', encoding="utf-8"
    )
    proposal = Proposal(command="sudo nvpmodel -m 0", rationale="power", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel("q\n"))
    assert executed == []


def test_pi_style_proposal_is_answered_through_respond_ui(xdg, monkeypatch):
    proposal = Proposal(command="apt install foo", rationale="install", kind=ProposalKind.FIX)
    events = [
        AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal, args={"request_id": "req-1"}),
        AgentEvent(kind=EventKind.DONE),
    ]
    monkeypatch.setattr(client_transport, "send", _stub_send([], events))
    answered = []
    monkeypatch.setattr(
        client_transport, "respond_ui", lambda rid, fields, **kw: answered.append((rid, fields))
    )
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel("\n"))
    assert answered == [("req-1", {"value": "once"})]
    assert executed == [], "pi runs the command itself; nvsh must not double-run it"


def test_ignored_pi_proposal_answers_deny(xdg, monkeypatch):
    proposal = Proposal(command="apt install foo", rationale="install", kind=ProposalKind.FIX)
    events = [
        AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal, args={"request_id": "req-9"}),
        AgentEvent(kind=EventKind.DONE),
    ]
    monkeypatch.setattr(client_transport, "send", _stub_send([], events))
    answered = []
    monkeypatch.setattr(
        client_transport, "respond_ui", lambda rid, fields, **kw: answered.append((rid, fields))
    )
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel("q\n"))
    assert answered == [("req-9", {"value": "deny"})]


def test_client_never_writes_readline_line(xdg):
    source = client_mod.__file__
    text = open(source, encoding="utf-8").read()
    assert "READLINE_LINE" not in text


# --- interrupt -----------------------------------------------------------


def test_interrupted_stream_returns_130_and_keeps_the_last_failure(xdg, monkeypatch):
    import os
    import signal

    def send(request, context=None, **kwargs):
        yield AgentEvent(kind=EventKind.TEXT_DELTA, text="working")
        os.kill(os.getpid(), signal.SIGINT)
        for _ in range(200):
            import time

            time.sleep(0.01)
            yield AgentEvent(kind=EventKind.TEXT_DELTA, text=".")

    monkeypatch.setattr(client_transport, "send", send)
    monkeypatch.setattr(client_transport, "cancel", lambda **kw: True)
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel()) == 130
    assert client_mod.load_last_failure()["line"] == "ls /nope"


# --- retry / verify ------------------------------------------------------


def test_retry_runs_only_after_enter_and_verify_reports_the_new_status(xdg):
    marker = xdg.tmp / "retried"
    client_mod.save_last_failure(_args(xdg.tmp, line=f"touch {marker}"))
    p = _panel("\n")
    rc = client_mod.retry(panel=p)
    assert marker.exists()
    assert rc == 0
    assert "verify" in p.out.getvalue()
    assert "exit 0" in p.out.getvalue()


def test_retry_does_nothing_when_the_operator_declines(xdg):
    marker = xdg.tmp / "retried"
    client_mod.save_last_failure(_args(xdg.tmp, line=f"touch {marker}"))
    p = _panel("q\n")
    assert client_mod.retry(panel=p) == 0
    assert not marker.exists()


def test_retry_reports_a_still_failing_command(xdg):
    client_mod.save_last_failure(_args(xdg.tmp, line="exit 3"))
    p = _panel("\n")
    assert client_mod.retry(panel=p) == 3
    assert "exit 3" in p.out.getvalue()


def test_retry_without_a_last_failure_is_a_clear_message(xdg):
    p = _panel("\n")
    assert client_mod.retry(panel=p) == 1
    assert "no recorded failure" in p.out.getvalue()


# --- slash ---------------------------------------------------------------


def test_handle_slash_routes_each_verb(xdg, monkeypatch):
    seen = []
    monkeypatch.setattr(client_mod, "ask", lambda prompt, **kw: seen.append(("ask", prompt)) or 0)
    monkeypatch.setattr(client_mod, "fix", lambda **kw: seen.append(("fix",)) or 0)
    monkeypatch.setattr(client_mod, "explain", lambda **kw: seen.append(("explain",)) or 0)
    monkeypatch.setattr(client_mod, "retry", lambda **kw: seen.append(("retry",)) or 0)
    monkeypatch.setattr(client_mod, "context_show", lambda **kw: seen.append(("context",)) or 0)
    for line in ("/ask why?", "/fix", "/explain", "/retry", "/context"):
        assert client_mod.handle_slash(line) == 0
    assert seen == [("ask", "why?"), ("fix",), ("explain",), ("retry",), ("context",)]


def test_handle_slash_unknown_verb_is_a_user_error(xdg):
    p = _panel()
    assert client_mod.handle_slash("/nope", panel=p) == 1
    assert "unknown" in p.out.getvalue().lower()


def test_ask_sends_an_explicit_request_with_the_draft(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    assert client_mod.ask("why is memory high?", draft="nvidia-smi", panel=_panel()) == 0
    request = calls[0][0]
    assert request.kind is RequestKind.EXPLICIT
    assert "why is memory high?" in request.prompt
    assert "nvidia-smi" in request.prompt


def test_fix_and_explain_use_the_recorded_failure(xdg, monkeypatch):
    client_mod.save_last_failure(_args(xdg.tmp))
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    assert client_mod.fix(panel=_panel()) == 0
    assert client_mod.explain(panel=_panel()) == 0
    assert [c[0].command for c in calls] == ["ls /nope", "ls /nope"]
    assert [c[0].kind for c in calls] == [RequestKind.FAILURE, RequestKind.FAILURE]
    assert "fix" in calls[0][0].prompt.lower()
    assert "explain" in calls[1][0].prompt.lower()


def test_fix_without_a_recorded_failure_says_so(xdg, monkeypatch):
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    p = _panel()
    assert client_mod.fix(panel=p) == 1
    assert "no recorded failure" in p.out.getvalue()
    assert calls == []
