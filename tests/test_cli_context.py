"""Tests for ``nvsh context --show`` (task t13).

The contract: the verb prints *exactly* the bytes that would be sent to the
agent for the last recorded failure — the prompt text
:func:`nvsh.agent.pi.build_prompt` builds from the request and context. The
byte-equality test captures what ``client_transport.send`` actually receives
and rebuilds the prompt from it.
"""

from __future__ import annotations

import io
import json
import types

import pytest

from nvsh import client as client_mod
from nvsh import client_transport
from nvsh import panel as panel_mod
from nvsh.agent.base import AgentEvent, EventKind
from nvsh.agent.pi import build_prompt
from nvsh.agent.prompt import build_full_prompt, build_system_prompt
from nvsh.cli import main


@pytest.fixture()
def xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("NVSH_NO_DAEMON", "1")
    monkeypatch.setattr(
        client_mod, "_platform_block", lambda: "platform: dgx-spark\n  gb10: yes  [file: /etc/x]"
    )
    return tmp_path


def _args(tmp_path, log=""):
    return types.SimpleNamespace(
        exit=2, pipestatus="2", line="ls /nope", cwd=str(tmp_path), log=log, json=False
    )


def test_context_show_prints_exactly_what_would_be_sent(xdg, monkeypatch, capsys):
    log = xdg / "session.log"
    log.write_bytes(b"\x1b]133;C\x07ls: /nope: No such file or directory\n\x1b]133;D;2\x07")
    captured = []

    def send(request, context=None, **kwargs):
        captured.append((request, context))
        yield AgentEvent(kind=EventKind.DONE)

    monkeypatch.setattr(client_transport, "send", send)
    p = panel_mod.Panel(out=io.StringIO(), in_=io.StringIO(), env={}, isatty=False)
    client_mod.handle_failure(_args(xdg, log=str(log)), panel=p)
    capsys.readouterr()

    rc = main(["context", "--show"])
    out = capsys.readouterr().out
    assert rc == 0
    request, context = captured[0]
    assert out == build_full_prompt(request, context) + "\n"
    assert build_system_prompt(context) in out
    assert build_prompt(request, context) in out
    assert "No such file or directory" in out


def test_context_show_includes_the_system_brief(xdg, capsys):
    """d19: the operator is shown the brief the model is given, too."""
    client_mod.save_last_failure(_args(xdg))
    rc = main(["context", "--show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("You are nvsh")
    assert "GB10" in out  # the dgx-spark playbook, chosen from the detected kind
    assert out.index("Command: ls /nope") > out.index("GB10")


def test_context_show_json_carries_the_system_brief(xdg, capsys):
    client_mod.save_last_failure(_args(xdg))
    rc = main(["context", "--show", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["system_prompt"].startswith("You are nvsh")
    assert "GB10" in payload["system_prompt"]


def test_context_show_json_shape(xdg, capsys):
    client_mod.save_last_failure(_args(xdg))
    rc = main(["context", "--show", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["command"] == "ls /nope"
    assert payload["exit_code"] == 2
    assert payload["platform"].startswith("platform: ")
    assert payload["cwd"] == str(xdg)
    assert "prompt" in payload
    assert isinstance(payload["redaction_rules"], list)


def test_context_show_without_a_recorded_failure_still_prints_the_block(xdg, capsys):
    rc = main(["context", "--show"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "platform: " in out


def test_context_is_in_the_explain_catalog():
    from nvsh.explain.catalog import ENTRIES

    assert ("context",) in ENTRIES
    assert ("context", "show") in ENTRIES


def test_context_redacts_before_printing(xdg, capsys):
    log = xdg / "session.log"
    log.write_bytes(
        b"\x1b]133;C\x07error: HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz\n\x1b]133;D;2\x07"
    )
    client_mod.save_last_failure(_args(xdg, log=str(log)))
    main(["context", "--show"])
    out = capsys.readouterr().out
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in out
