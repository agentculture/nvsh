"""The failure client treats a typed sentence as a question to answer (d20)."""

from __future__ import annotations

import io
import os
import types
from pathlib import Path

import pytest

from nvsh import client as client_mod
from nvsh import client_transport
from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind, RequestKind


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("NVSH_NO_DAEMON", "1")
    monkeypatch.setattr(client_mod, "_platform_block", lambda: "platform: dgx-spark")
    return tmp_path


FAKES_DIR = str(Path(__file__).parent / "fakes")


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
    # The header names the chosen harness; pin it to the fake pi on PATH so the
    # label does not depend on whether the runner has a real pi installed.
    monkeypatch.setenv("PATH", FAKES_DIR + os.pathsep + os.environ.get("PATH", ""))
    out = io.StringIO()
    panel = panel_mod.Panel(out=out, in_=io.StringIO(), env={}, isatty=False)
    client_mod.handle_failure(_args(xdg, line, exit_code), panel=panel)
    return captured[0], out.getvalue()


def test_a_typed_question_becomes_an_explicit_request(xdg, monkeypatch):
    request, out = _run(xdg, monkeypatch, "what are the memory levels?", 127)
    assert request.kind is RequestKind.EXPLICIT
    assert request.prompt == "what are the memory levels?"
    assert request.command == ""
    assert "nvsh: asking pi/associate: what are the memory levels?" in out
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
        def header(self, command, exit_code, ask=None, backend_label=""):
            seen["args"] = (command, exit_code, ask)

    def send(request, context=None, **kwargs):
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "send", send)
    panel = _AskPanel(out=io.StringIO(), in_=io.StringIO(), env={}, isatty=False)
    client_mod.handle_failure(_args(xdg, "what are the memory levels?", 127), panel=panel)
    assert seen["args"] == ("what are the memory levels?", 127, "what are the memory levels?")


# --- d23: the ? and @name marks -------------------------------------------


def _run_marked(xdg, monkeypatch, line, *, adapters=("pi", "qwen"), installed=("pi",)):
    """Drive handle_failure over a marked line with the registry mocked."""
    from nvsh.agent import registry

    captured = []
    configs = []

    def one_shot(request, context=None, **kwargs):
        captured.append(request)
        configs.append(kwargs.get("config"))
        yield AgentEvent(kind=EventKind.DONE)

    def send(request, context=None, **kwargs):
        captured.append(request)
        configs.append(kwargs.get("config"))
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "one_shot", one_shot)
    monkeypatch.setattr(client_transport, "send", send)
    monkeypatch.setattr(registry, "ADAPTERS", {name: registry.ADAPTERS[name] for name in adapters})
    monkeypatch.setattr(registry, "installed", lambda name, which=None: name in installed)
    out = io.StringIO()
    panel = panel_mod.Panel(out=out, in_=io.StringIO(), env={}, isatty=False)
    rc = client_mod.handle_failure(_args(xdg, line, 127), panel=panel)
    return rc, captured, configs, out.getvalue()


def test_question_mark_asks_the_default_agent(xdg, monkeypatch):
    rc, captured, _configs, out = _run_marked(xdg, monkeypatch, "? what are the ram levels?")
    assert rc == 0
    assert captured[0].kind is RequestKind.EXPLICIT
    assert captured[0].prompt == "what are the ram levels?"
    assert captured[0].command == ""
    assert "nvsh: asking pi/associate: what are the ram levels?" in out


def test_a_mark_is_never_held_back_by_the_rate_limiter(xdg, monkeypatch):
    """Explicit calls bypass the auto-call window, like Ctrl+G (d23)."""
    _rc, first, _c, _o = _run_marked(xdg, monkeypatch, "? what are the ram levels?")
    assert len(first) == 1
    rc, second, _c2, out = _run_marked(xdg, monkeypatch, "? and the disk?")
    assert rc == 0
    assert len(second) == 1, "the second mark was held back"
    assert "held back" not in out


def test_a_mark_does_not_consume_the_auto_call_window(xdg, monkeypatch):
    """A mark must not count *against* a later automatic call either."""
    _run_marked(xdg, monkeypatch, "? what are the ram levels?")
    request, out = _run(xdg, monkeypatch, "ls /nope", 2)
    assert request.kind is RequestKind.FAILURE
    assert "held back" not in out


def test_at_name_routes_to_that_harness_for_one_request(xdg, monkeypatch):
    rc, captured, configs, out = _run_marked(
        xdg, monkeypatch, "@pi how much ram is free?", installed=("pi", "qwen")
    )
    assert rc == 0
    assert captured[0].prompt == "how much ram is free?"
    assert configs[0].agent_provider == "pi"
    assert "asking pi/associate: how much ram is free?" in out


def test_an_unavailable_harness_is_one_line_and_nothing_else(xdg, monkeypatch):
    rc, captured, _configs, out = _run_marked(
        xdg, monkeypatch, "@qwen how much ram is free?", installed=("pi",)
    )
    assert rc == 0
    assert captured == []
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    assert lines[0].startswith("nvsh: @qwen is not available: ")


def test_an_unmarked_sentence_still_pays_the_rate_limit(xdg, monkeypatch, capsys):
    """d20 prose is a guess, so it keeps the auto-call window (unchanged)."""
    _run(xdg, monkeypatch, "what are the memory levels?", 127)
    capsys.readouterr()
    out = io.StringIO()
    panel = panel_mod.Panel(out=out, in_=io.StringIO(), env={}, isatty=False)
    monkeypatch.setattr(
        client_transport, "send", lambda *a, **k: iter([AgentEvent(kind=EventKind.DONE)])
    )
    client_mod.handle_failure(_args(xdg, "why is the gpu slow", 127), panel=panel)
    assert "held back" in capsys.readouterr().err


def test_an_unconfigured_tier_adapter_names_its_config_key_not_a_binary(xdg, monkeypatch):
    _rc, _captured, _configs, out = _run_marked(
        xdg, monkeypatch, "@lfm why did that fail?", adapters=("pi", "lfm"), installed=("pi",)
    )
    assert "[tiers.lfm] model" in out


def test_an_unconfigured_tier_adapter_never_says_not_on_path(xdg, monkeypatch):
    _rc, _captured, _configs, out = _run_marked(
        xdg, monkeypatch, "@lfm why did that fail?", adapters=("pi", "lfm"), installed=("pi",)
    )
    assert "PATH" not in out
