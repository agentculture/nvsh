"""Tests for nvsh.client (task t13): context assembly, rate state, proposals,
retry/verify and the slash entry points.
"""

from __future__ import annotations

import io
import json
import os
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


# --- d15: approve for this session / for this user ------------------------


def _approvals():
    from nvsh.approvals import Approvals

    return Approvals.load()


def _decisions(entries):
    return [e["decision"] for e in entries if e["event"] == "decision"]


def _audit_entries():
    from nvsh.agent.audit import AuditLog

    return AuditLog().read_all()


def test_session_key_on_a_pi_dialog_answers_session_and_never_runs_locally(xdg, monkeypatch):
    proposal = Proposal(command="apt install foo", rationale="install", kind=ProposalKind.FIX)
    events = [
        AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal, args={"request_id": "req-s"}),
        AgentEvent(kind=EventKind.DONE),
    ]
    monkeypatch.setattr(client_transport, "send", _stub_send([], events))
    answered = []
    monkeypatch.setattr(
        client_transport, "respond_ui", lambda rid, fields, **kw: answered.append((rid, fields))
    )
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel("s\n"))
    assert answered == [("req-s", {"value": "session"})]
    assert executed == [], "the extension widens and runs it; nvsh must not double-run"
    assert "session" in _decisions(_audit_entries())


def test_user_key_on_a_pi_dialog_answers_user(xdg, monkeypatch):
    proposal = Proposal(command="apt install foo", rationale="install", kind=ProposalKind.FIX)
    events = [
        AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal, args={"request_id": "req-u"}),
        AgentEvent(kind=EventKind.DONE),
    ]
    monkeypatch.setattr(client_transport, "send", _stub_send([], events))
    answered = []
    monkeypatch.setattr(
        client_transport, "respond_ui", lambda rid, fields, **kw: answered.append((rid, fields))
    )
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel("u\n"))
    assert answered == [("req-u", {"value": "user"})]
    assert "user" in _decisions(_audit_entries())


def test_session_key_without_a_dialog_adds_the_exact_line_and_runs_it(xdg, monkeypatch):
    marker = xdg.tmp / "ran-session"
    command = f"touch {marker}"
    proposal = Proposal(command=command, rationale="fix it", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel("s\n")) == 0
    assert marker.exists(), "the run-once behaviour must still happen"
    approvals = _approvals()
    assert command in approvals.session_patterns
    assert command not in approvals.user_patterns
    assert approvals.decide(command) == "session"
    assert "session" in _decisions(_audit_entries())


def test_user_key_without_a_dialog_widens_to_first_word_and_persists(xdg, monkeypatch):
    marker = xdg.tmp / "ran-user"
    command = f"touch {marker}"
    proposal = Proposal(command=command, rationale="fix it", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel("u\n")) == 0
    assert marker.exists()
    saved = (xdg.config / "nvsh" / "approved.toml").read_text(encoding="utf-8")
    assert "touch *" in saved, saved
    assert _approvals().decide("touch /somewhere/else") == "user"
    assert "user" in _decisions(_audit_entries())


def test_a_session_approval_is_honoured_by_the_next_client_process(xdg, monkeypatch):
    """Two hook invocations are two processes; the second must not re-ask."""
    command = "echo from-a-previous-process"
    proposal = Proposal(command=command, rationale="look", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel("s\n"))
    client_mod.rate_state_path(os.environ).unlink()  # a later window, not a burst

    calls = []
    streams = [
        _proposal_events(proposal),
        [AgentEvent(kind=EventKind.TEXT_DELTA, text="done"), AgentEvent(kind=EventKind.DONE)],
    ]

    def send(request, context=None, **kwargs):
        calls.append((request, context, kwargs))
        yield from streams[min(len(calls) - 1, len(streams) - 1)]

    monkeypatch.setattr(client_transport, "send", send)
    p = _panel("")  # EOF: nothing typed, so nothing may be asked
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    assert len(calls) == 2, "the approved inspection must auto-run and feed back"
    assert "from-a-previous-process" in calls[1][0].prompt


@pytest.mark.parametrize("typed", ["s\n", "u\n"])
def test_privileged_command_refuses_the_scope_keys_and_re_asks(xdg, monkeypatch, typed):
    command = "sudo nvpmodel -m 0"
    proposal = Proposal(command=command, rationale="power mode", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd) or _ok())
    p = _panel(typed + "\n")  # refused, then Enter -> run once
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    text = p.out.getvalue()
    assert "cannot approve" in text, text
    assert executed == [command], "run-once must stay available after the refusal"
    approvals = _approvals()
    assert approvals.session_patterns == []
    assert approvals.decide(command) == "ask"


def test_destructive_command_refuses_the_user_key(xdg, monkeypatch):
    command = "rm -rf /var/tmp/x"
    proposal = Proposal(command=command, rationale="clean", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd) or _ok())
    p = _panel("u\nq\n")  # refused, then ignore
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    text = p.out.getvalue()
    assert "cannot approve" in text, text
    assert "rm" in text
    assert executed == []
    assert _approvals().decide(command) == "ask"


def _ok():
    return client_mod.RunResult(exit_code=0)


# --- d11: a one-shot proposal must reach the in-process agent -------------
#
# On the DGX Spark the daemon was unreachable, so the client fell back to
# one-shot pi. pi proposed a command through its approval extension, the
# panel offered "[Enter] run", and Enter did nothing: the answer went out as
# a `ui_response` control message to a daemon socket nobody was listening
# on, while the in-process PiAgent sat waiting for its `extension_ui_response`.
# These tests drive the *real* one-shot path (PiAgent over the scripted fake
# pi) and assert the dialog is answered by the agent that raised it.


def _one_shot_pi(monkeypatch, tmp_path, script, responses, commands=None):
    """Make the one-shot path build a PiAgent talking to ``tests/fakes/pi_scripted``."""
    import dataclasses
    from pathlib import Path

    from nvsh.agent import registry
    from nvsh.agent.pi import PiAgent

    fakes = Path(__file__).resolve().parent / "fakes"
    pi_env = dict(os.environ)
    pi_env["PATH"] = str(fakes) + os.pathsep + pi_env.get("PATH", "")
    pi_env["HOME"] = str(tmp_path / "pi-home")
    pi_env["XDG_STATE_HOME"] = str(tmp_path / "pi-state")
    pi_env["NVSH_TEST_PI_SCRIPT"] = json.dumps(script)
    pi_env["NVSH_TEST_PI_AWAIT_UI"] = "1"
    pi_env["NVSH_TEST_PI_RESPONSES"] = str(responses)
    if commands is not None:
        pi_env["NVSH_TEST_PI_COMMANDS"] = str(commands)

    spec = dataclasses.replace(
        registry.ADAPTERS["pi"],
        binary=None,
        factory=lambda cfg: PiAgent(pi_path="pi_scripted", env=pi_env),
    )
    monkeypatch.setitem(registry.ADAPTERS, "pi", spec)


def _approval_request(command: str, reason: str = "", request_id: str = "ui-11") -> dict:
    title = json.dumps(
        {"nvsh": "approval", "v": 1, "tool": "bash", "command": command, "reason": reason}
    )
    return {
        "type": "extension_ui_request",
        "id": request_id,
        "method": "select",
        "title": title,
        "options": ["once", "session", "user", "deny"],
    }


def _dialog_script(command: str):
    return [
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "let me check the GPU. "},
        },
        _approval_request(command, reason="see whether nvidia-smi is installed"),
        {"type": "tool_execution_start", "toolName": "bash", "args": {"command": command}},
        {"type": "tool_execution_end", "toolName": "bash", "result": {"output": "GPU 0: ok"}},
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "the driver is fine."},
        },
        {"type": "agent_end"},
    ]


def _responses(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_one_shot_proposal_is_answered_by_the_in_process_agent(xdg, monkeypatch):
    responses = xdg.tmp / "ui-responses.jsonl"
    _one_shot_pi(monkeypatch, xdg.tmp, _dialog_script("which nvidia-smi"), responses)

    def no_daemon(*a, **kw):
        raise AssertionError("the one-shot path must not answer through the daemon socket")

    monkeypatch.setattr(client_transport, "respond_ui", no_daemon)

    p = _panel("\n")  # Enter -> run
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0

    assert _responses(responses) == [
        {"type": "extension_ui_response", "id": "ui-11", "value": "once"}
    ]
    text = p.out.getvalue()
    assert "which nvidia-smi" in text
    assert "the driver is fine." in text, "the turn must continue after the dialog is answered"


def test_one_shot_ignored_proposal_denies_through_the_in_process_agent(xdg, monkeypatch):
    responses = xdg.tmp / "ui-responses.jsonl"
    _one_shot_pi(monkeypatch, xdg.tmp, _dialog_script("which nvidia-smi"), responses)

    p = _panel("q\n")  # anything but Enter -> ignore
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0

    assert _responses(responses) == [
        {"type": "extension_ui_response", "id": "ui-11", "value": "deny"}
    ]


def test_client_never_writes_readline_line(xdg):
    source = client_mod.__file__
    text = open(source, encoding="utf-8").read()
    assert "READLINE_LINE" not in text


# --- interrupt -----------------------------------------------------------


def test_interrupted_stream_returns_130_and_keeps_the_last_failure(xdg, monkeypatch):
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


# --- transport timeouts (deviation d7) -----------------------------------


def _tiny_daemon(path, stop):
    """A listener that accepts and answers control lines, and nothing else.

    Stands in for a daemon that is up and accepting but whose cold backend
    has not produced a first token yet.
    """
    import socket as _socket

    server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(8)
    server.settimeout(0.05)
    try:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            with conn:
                try:
                    stream = conn.makefile("rwb")
                    stream.readline()
                    stream.write(b'{"kind": "status", "text": "{}"}\n{"kind": "done"}\n')
                    stream.flush()
                except OSError:  # a client that hung up early
                    continue
    finally:
        server.close()


def test_autostart_streams_with_the_request_timeout_not_the_connect_wait(tmp_path, monkeypatch):
    """The socket handed to the stream must carry the request timeout.

    Deviation d7's root cause: after an autostart the client streamed on a
    socket whose timeout was the 5 s connect wait, so any backend slower
    than that raised ``timed out`` on the very first read.
    """
    import threading

    from nvsh import daemon as daemon_mod
    from nvsh.agent.base import AgentRequest

    env = {"XDG_RUNTIME_DIR": str(tmp_path / "run"), "HOME": str(tmp_path)}
    sock_path = daemon_mod.socket_path(env)
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    thread = threading.Thread(target=_tiny_daemon, args=(sock_path, stop), daemon=True)

    def fake_spawn(*args, **kwargs):
        thread.start()
        return 4242

    seen: list[float | None] = []

    def fake_stream(sock, payload):
        if payload.get("kind") != "ping":  # the hello has its own short wait
            seen.append(sock.gettimeout())
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(daemon_mod, "spawn", fake_spawn)
    monkeypatch.setattr(client_transport, "_stream", fake_stream)
    try:
        events = list(
            client_transport.send(
                AgentRequest(kind=RequestKind.FAILURE, command="ls /nope", exit_code=2),
                shell_id="1",
                env=env,
                timeout=37.0,
            )
        )
    finally:
        stop.set()
        if thread.is_alive():
            thread.join(timeout=5)
    assert events[-1].kind is EventKind.DONE
    assert seen == [37.0], f"streamed with the wrong socket timeout: {seen}"


# --- the rate limiter says when it holds back (deviation d10) -------------


def test_rate_limited_failure_prints_one_held_back_line(xdg, monkeypatch, capsys):
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel()) == 0
    capsys.readouterr()

    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel()) == 0
    err = capsys.readouterr().err
    assert "held back" in err
    assert "/fix" in err
    assert err.count("held back") == 1


def test_held_back_notice_is_printed_at_most_once_per_window(xdg, monkeypatch, capsys):
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    capsys.readouterr()

    # A burst of further failures inside the same window stays silent.
    for _ in range(3):
        client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    assert "held back" not in capsys.readouterr().err


def test_held_back_notice_returns_in_the_next_window(xdg, monkeypatch, capsys):
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    capsys.readouterr()

    # Age both the last auto call and the notice out of the window.
    path = client_mod.rate_state_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    stale = data["last_auto_call"] - 10_000
    data["last_auto_call"] = stale
    data["last_held_back_notice"] = stale
    path.write_text(json.dumps(data), encoding="utf-8")

    # The next failure is allowed through (no notice), the one after is held.
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    capsys.readouterr()
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    assert "held back" in capsys.readouterr().err


def test_a_successful_call_prints_no_held_back_line(xdg, monkeypatch, capsys):
    monkeypatch.setattr(client_transport, "send", _stub_send([]))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel())
    assert "held back" not in capsys.readouterr().err


# --- d16: [t] tell -- steer the running turn ------------------------------


def _steer_script(command: str):
    """A dialog, then (after the steer) a second, different proposal."""
    return [
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "let me reinstall it. "},
        },
        _approval_request(command, request_id="ui-t"),
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "ok, memory it is."},
        },
        {"type": "agent_end"},
    ]


def test_tell_key_denies_with_the_reason_and_steers_the_same_turn(xdg, monkeypatch):
    responses = xdg.tmp / "ui-responses.jsonl"
    commands = xdg.tmp / "pi-commands.jsonl"
    _one_shot_pi(
        monkeypatch,
        xdg.tmp,
        _steer_script("apt install -y linux-tools"),
        responses,
        commands=commands,
    )
    p = _panel("t\njust run free -h\n")
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0

    assert _responses(responses) == [
        {
            "type": "extension_ui_response",
            "id": "ui-t",
            "value": "deny",
            "reason": "just run free -h",
        }
    ]
    steers = [
        c
        for c in _responses(commands)
        if c.get("type") == "prompt" and c.get("streamingBehavior") == "steer"
    ]
    assert len(steers) == 1, _responses(commands)
    assert steers[0]["message"] == "just run free -h"
    text = p.out.getvalue()
    assert "ok, memory it is." in text, "the panel must keep streaming the steered turn"
    assert "nvsh: not run" not in text


def test_tell_with_an_empty_line_returns_to_the_proposal(xdg, monkeypatch):
    proposal = Proposal(command="apt install foo", rationale="install", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd) or _ok())
    p = _panel("t\n\n\n")  # tell, empty line -> cancel, then Enter -> run once
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    assert executed == ["apt install foo"]


def test_steer_on_an_adapter_without_a_mid_turn_channel_becomes_the_next_request(xdg, monkeypatch):
    proposal = Proposal(command="apt install foo", rationale="install", kind=ProposalKind.FIX)
    calls = []
    streams = [
        _proposal_events(proposal),
        [AgentEvent(kind=EventKind.TEXT_DELTA, text="ok"), AgentEvent(kind=EventKind.DONE)],
    ]

    def send(request, context=None, **kwargs):
        calls.append((request, context, kwargs))
        yield from streams[min(len(calls) - 1, len(streams) - 1)]

    monkeypatch.setattr(client_transport, "send", send)
    p = _panel("t\njust run free -h\n")
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    assert len(calls) == 2, "the steer must come back as the next request"
    follow_up = calls[1][0]
    # Verbatim: the conversation (the daemon's session, or the adapter's own
    # memory -- see tests/test_agent_openai_compat.py) carries the prior turn,
    # so the client must not paraphrase the operator's words into it.
    assert follow_up.prompt == "just run free -h"
    assert follow_up.command == "ls /nope"
    assert "no mid-turn channel" in p.out.getvalue()


# --- d17: rationale, and explaining by asking -----------------------------


def test_explain_with_a_rationale_prints_it_without_asking_the_agent(xdg, monkeypatch):
    proposal = Proposal(command="free -h", rationale="check memory", kind=ProposalKind.FIX)
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls, _proposal_events(proposal)))
    p = _panel("e\nq\n")
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    text = p.out.getvalue()
    assert "why: check memory" in text
    assert "(no rationale given)" not in text
    assert len(calls) == 1, "a rationale we already have must not cost a round trip"


def test_explain_without_a_rationale_asks_the_agent_instead_of_apologising(xdg, monkeypatch):
    responses = xdg.tmp / "ui-responses.jsonl"
    commands = xdg.tmp / "pi-commands.jsonl"
    script = [
        _approval_request("apt install -y linux-tools", reason="", request_id="ui-e"),
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "because perf is missing."},
        },
        {"type": "agent_end"},
    ]
    _one_shot_pi(monkeypatch, xdg.tmp, script, responses, commands=commands)
    p = _panel("e\n")
    assert client_mod.handle_failure(_args(xdg.tmp), panel=p) == 0
    text = p.out.getvalue()
    assert "(no rationale given)" not in text
    assert "because perf is missing." in text
    steers = [
        c
        for c in _responses(commands)
        if c.get("type") == "prompt" and c.get("streamingBehavior") == "steer"
    ]
    assert len(steers) == 1
    assert steers[0]["message"] == (
        "Explain in two sentences why you propose: apt install -y linux-tools"
    )


# --- d18: details ---------------------------------------------------------


def test_details_report_approval_patterns_backend_conversation_and_context(xdg, monkeypatch):
    log = _log_with(xdg.tmp, b"ls: /nope: No such file\n")
    proposal = Proposal(command="whatis ls", rationale="look it up", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    p = _panel("d\nq\n")
    assert client_mod.handle_failure(_args(xdg.tmp, log=str(log)), panel=p) == 0
    text = p.out.getvalue()
    assert "kind: inspect" in text
    assert "command: whatis ls" in text
    assert "approved: no" in text
    assert "session pattern: whatis ls" in text
    assert "user pattern: whatis *" in text
    assert "backend:" in text
    assert "conversation: one-shot" in text
    assert f"shell {os.getpid()}" in text or "shell " in text
    assert "bytes" in text


def test_details_name_the_matching_approval_when_there_is_one(xdg, monkeypatch):
    (xdg.config / "nvsh").mkdir(parents=True)
    (xdg.config / "nvsh" / "approved.toml").write_text(
        'user_patterns = [\n  "whatis *",\n]\n', encoding="utf-8"
    )
    proposal = Proposal(command="whatis ls", rationale="look", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    p = _panel("d\nq\n")
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    assert "approved: user pattern 'whatis *'" in p.out.getvalue()


def test_details_report_the_refusal_reason_for_a_privileged_command(xdg, monkeypatch):
    proposal = Proposal(command="sudo nvpmodel -m 0", rationale="power", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    p = _panel("d\nq\n")
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    text = p.out.getvalue()
    assert "refused:" in text
    assert "escalates privilege" in text


# --- d19 (panel half): the auto-inspect line names the command ------------


def test_an_auto_inspected_command_is_announced_by_name_and_exit_code(xdg, monkeypatch):
    (xdg.config / "nvsh").mkdir(parents=True)
    (xdg.config / "nvsh" / "approved.toml").write_text(
        'user_patterns = [\n  "echo *",\n]\n', encoding="utf-8"
    )
    proposal = Proposal(command="echo hi", rationale="look", kind=ProposalKind.INSPECT)
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
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    text = p.out.getvalue()
    assert "... running: echo hi" in text
    assert "... finished (exit 0)" in text


# --- d16: /steer ----------------------------------------------------------


def test_slash_steer_sends_the_text_as_the_next_request_on_the_last_failure(xdg, monkeypatch):
    client_mod.save_last_failure(_args(xdg.tmp))
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    monkeypatch.setattr(client_transport, "control", lambda *a, **kw: [])
    assert client_mod.handle_slash("/steer just run free -h", panel=_panel()) == 0
    assert len(calls) == 1
    assert calls[0][0].prompt == "just run free -h"
    assert calls[0][0].command == "ls /nope"


def test_slash_steer_prefers_a_running_turn_when_there_is_one(xdg, monkeypatch):
    client_mod.save_last_failure(_args(xdg.tmp))
    calls = []
    monkeypatch.setattr(client_transport, "send", _stub_send(calls))
    seen = []

    def control(kind, **kwargs):
        seen.append((kind, kwargs.get("text")))
        return [AgentEvent(kind=EventKind.STATUS, text="steer delivered")]

    monkeypatch.setattr(client_transport, "control", control)
    p = _panel()
    assert client_mod.handle_slash("/steer stop and check memory", panel=p) == 0
    assert seen == [("steer", "stop and check memory")]
    assert calls == [], "a running turn is steered, not restarted"
    assert "steer" in p.out.getvalue().lower()


def test_slash_steer_without_text_is_a_user_error(xdg):
    p = _panel()
    assert client_mod.handle_slash("/steer", panel=p) == 1
    assert "/steer" in p.out.getvalue()


# --- PR #8 review: retry runs where the command failed ---------------------


def test_retry_runs_in_the_recorded_directory(xdg, monkeypatch):
    """Relative paths must resolve against the directory that failed.

    ``/retry`` after a ``cd`` would otherwise touch different files than the
    invocation the operator approved.
    """
    work = xdg.tmp / "work"
    work.mkdir()
    client_mod.save_last_failure(_args(work, line="touch relative-marker"))
    elsewhere = xdg.tmp / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    p = _panel("\n")
    assert client_mod.retry(panel=p) == 0
    assert (work / "relative-marker").exists()
    assert not (elsewhere / "relative-marker").exists()
    assert os.getcwd() == str(elsewhere), "the process cwd must be restored"


def test_retry_refuses_when_the_recorded_directory_is_gone(xdg, monkeypatch):
    gone = xdg.tmp / "gone"
    gone.mkdir()
    client_mod.save_last_failure(_args(gone, line="touch marker"))
    monkeypatch.chdir(xdg.tmp)
    gone.rmdir()

    p = _panel("\n")
    rc = client_mod.retry(panel=p)
    assert rc == 1
    assert not (xdg.tmp / "marker").exists()
    assert "director" in p.out.getvalue()


def test_retry_without_a_recorded_cwd_still_runs_here(xdg, monkeypatch):
    """An old state file with no cwd keeps working (runs in the current dir)."""
    here = xdg.tmp / "here"
    here.mkdir()
    monkeypatch.chdir(here)
    p = _panel("\n")
    assert client_mod.retry({"line": "touch no-cwd-marker", "exit": 2}, panel=p) == 0
    assert (here / "no-cwd-marker").exists()


# --- PR #8 review: the rate decision is atomic across shells ---------------


def _lock_is_held(path) -> bool:
    """Can another *process* take an exclusive flock on ``path`` right now?"""
    import subprocess
    import sys

    probe = (
        "import fcntl, os, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
        "try:\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except OSError:\n"
        "    sys.exit(3)\n"
        "sys.exit(0)\n"
    )
    return (
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", probe, str(path)], capture_output=True
        ).returncode
        == 3
    )


def test_rate_decision_runs_under_an_exclusive_lock(xdg, monkeypatch):
    """Two shells failing at once must not both read the same stale state."""
    from nvsh import config as nvsh_config
    from nvsh import triggers as triggers_mod

    seen = {}
    real_decide = triggers_mod.decide

    def spy(event):
        seen["held"] = _lock_is_held(client_mod.rate_lock_path())
        return real_decide(event)

    monkeypatch.setattr(triggers_mod, "decide", spy)
    client_mod._rate_limited(_args(xdg.tmp), nvsh_config.load(), None)
    assert seen["held"] is True


def test_rate_lock_is_released_afterwards(xdg):
    from nvsh import config as nvsh_config

    client_mod._rate_limited(_args(xdg.tmp), nvsh_config.load(), None)
    assert _lock_is_held(client_mod.rate_lock_path()) is False


def test_rate_lock_failure_does_not_break_the_decision(xdg, monkeypatch):
    """A filesystem that refuses locking still gets its diagnosis."""
    from nvsh import config as nvsh_config

    def no_locking(*_args, **_kwargs):
        raise OSError("no locking on this filesystem")

    monkeypatch.setattr(client_mod.fcntl, "flock", no_locking)
    assert client_mod._rate_limited(_args(xdg.tmp), nvsh_config.load(), None) is False
    assert client_mod.rate_state_path().exists()


# --- d24: the specific form (S/U) and one pattern per pipeline stage ------


def test_user_specific_key_without_a_dialog_stores_the_first_argument(xdg, monkeypatch):
    proposal = Proposal(command="ssh orin uptime", rationale="look", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: _ok())
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel("U\n")) == 0
    saved = (xdg.config / "nvsh" / "approved.toml").read_text(encoding="utf-8")
    assert "ssh orin *" in saved, saved
    assert '"ssh *"' not in saved, saved
    approvals = _approvals()
    assert approvals.decide("ssh orin free -h") == "user"
    assert approvals.decide("ssh thor free -h") == "ask"
    assert "user-specific" in _decisions(_audit_entries())


def test_session_specific_key_without_a_dialog_stores_the_first_argument(xdg, monkeypatch):
    proposal = Proposal(command="ssh orin uptime", rationale="look", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: _ok())
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel("S\n")) == 0
    approvals = _approvals()
    assert "ssh orin *" in approvals.session_patterns
    assert "ssh orin *" not in approvals.user_patterns


@pytest.mark.parametrize(
    "typed,value",
    [("S\n", "session-specific"), ("U\n", "user-specific")],
)
def test_specific_keys_on_a_pi_dialog_forward_their_own_token(xdg, monkeypatch, typed, value):
    proposal = Proposal(command="ssh orin uptime", rationale="look", kind=ProposalKind.FIX)
    events = [
        AgentEvent(kind=EventKind.PROPOSAL, proposal=proposal, args={"request_id": "req-x"}),
        AgentEvent(kind=EventKind.DONE),
    ]
    monkeypatch.setattr(client_transport, "send", _stub_send([], events))
    answered = []
    monkeypatch.setattr(
        client_transport, "respond_ui", lambda rid, fields, **kw: answered.append((rid, fields))
    )
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd))
    client_mod.handle_failure(_args(xdg.tmp), panel=_panel(typed))
    assert answered == [("req-x", {"value": value})]
    assert executed == [], "the extension owns the store write and the run"


def test_a_pipeline_stores_one_pattern_per_stage(xdg, monkeypatch):
    command = "ps -eo pid,rss | head -n 20"
    proposal = Proposal(command=command, rationale="look", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: _ok())
    assert client_mod.handle_failure(_args(xdg.tmp), panel=_panel("u\n")) == 0
    approvals = _approvals()
    assert "ps *" in approvals.user_patterns
    assert "head *" in approvals.user_patterns
    assert approvals.decide(command) == "user"


def test_a_pipeline_with_a_privileged_stage_refuses_every_scope_key(xdg, monkeypatch):
    command = "ls /nope | sudo tee /etc/x"
    proposal = Proposal(command=command, rationale="fix", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    executed = []
    monkeypatch.setattr(client_mod, "_run_command", lambda cmd, **kw: executed.append(cmd) or _ok())
    p = _panel("U\nq\n")  # refused, then ignore
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    assert "cannot approve" in p.out.getvalue()
    approvals = _approvals()
    assert "ls *" not in approvals.user_patterns
    assert approvals.session_patterns == []
    assert executed == []


def test_the_scope_line_lists_every_stage_pattern(xdg, monkeypatch):
    command = "ps -eo pid,rss | head -n 20"
    proposal = Proposal(command=command, rationale="look", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    p = _panel("q\n")
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    text = p.out.getvalue()
    assert "'ps *' 'head *'" in text, text
    assert "each stage exactly" in text, text


def test_the_scope_line_says_when_the_specific_form_adds_nothing(xdg, monkeypatch):
    proposal = Proposal(command="htop", rationale="look", kind=ProposalKind.INSPECT)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    p = _panel("q\n")
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    text = p.out.getvalue()
    assert "same as [u] (no second word)" in text, text
    assert "same as [s] (no second word)" in text, text


def test_details_quote_the_specific_patterns_too(xdg, monkeypatch):
    proposal = Proposal(command="ssh orin uptime", rationale="look", kind=ProposalKind.FIX)
    monkeypatch.setattr(client_transport, "send", _stub_send([], _proposal_events(proposal)))
    p = _panel("d\nq\n")
    client_mod.handle_failure(_args(xdg.tmp), panel=p)
    text = p.out.getvalue()
    assert "user pattern: ssh *" in text, text
    assert "user-specific pattern: ssh orin *" in text, text
    assert "session-specific pattern: ssh orin *" in text, text
