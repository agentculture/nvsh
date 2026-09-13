"""The failure client treats a typed sentence as a question to answer (d20)."""

from __future__ import annotations

import io
import types

import pytest

from nvsh import client as client_mod
from nvsh import client_transport
from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind, RequestKind


@pytest.fixture()
def xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("NVSH_NO_DAEMON", "1")
    monkeypatch.setattr(client_mod, "_platform_block", lambda: "platform: dgx-spark")
    return tmp_path


def _args(tmp_path, line, exit_code):
    return types.SimpleNamespace(
        exit=exit_code,
        pipestatus=str(exit_code),
        line=line,
        cwd=str(tmp_path),
        log="",
        json=False,
    )


def _run(xdg, monkeypatch, line, exit_code):
    captured = []

    def send(request, context=None, **kwargs):
        captured.append(request)
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "send", send)
    out = io.StringIO()
    panel = panel_mod.Panel(out=out, in_=io.StringIO(), env={}, isatty=False)
    client_mod.handle_failure(_args(xdg, line, exit_code), panel=panel)
    return captured[0], out.getvalue()


def test_a_typed_question_becomes_an_explicit_request(xdg, monkeypatch):
    request, out = _run(xdg, monkeypatch, "what are the memory levels?", 127)
    assert request.kind is RequestKind.EXPLICIT
    assert request.prompt == "what are the memory levels?"
    assert request.command == ""
    assert "nvsh: asking the agent: what are the memory levels?" in out
    assert "failed (exit 127)" not in out


def test_a_real_failure_keeps_the_failure_header(xdg, monkeypatch):
    request, out = _run(xdg, monkeypatch, "ls /nope", 2)
    assert request.kind is RequestKind.FAILURE
    assert request.command == "ls /nope"
    assert "ls /nope failed (exit 2)" in out
    assert "asking the agent" not in out


def test_the_request_carries_the_sentence_for_the_panel(xdg, monkeypatch):
    request, _out = _run(xdg, monkeypatch, "why is the gpu slow", 127)
    assert request.ask == "why is the gpu slow"


def test_failure_requests_carry_no_ask(xdg, monkeypatch):
    request, _out = _run(xdg, monkeypatch, "ls /nope", 2)
    assert request.ask == ""


def test_the_ask_form_is_handed_to_the_panel_header_once_it_takes_one(xdg, monkeypatch):
    """d16 owns ``Panel.header``; this call site uses its ``ask`` form as
    soon as the signature has one, and the plain line until then."""
    seen = {}

    class _AskPanel(panel_mod.Panel):
        def header(self, command, exit_code, ask=None):
            seen["args"] = (command, exit_code, ask)

    def send(request, context=None, **kwargs):
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "send", send)
    panel = _AskPanel(out=io.StringIO(), in_=io.StringIO(), env={}, isatty=False)
    client_mod.handle_failure(_args(xdg, "what are the memory levels?", 127), panel=panel)
    assert seen["args"] == ("what are the memory levels?", 127, "what are the memory levels?")
