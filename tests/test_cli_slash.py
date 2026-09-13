"""Tests for ``nvsh slash`` / ``nvsh complete`` (task t14): the CLI verbs over
:mod:`nvsh.slash`, their --json shapes, and catalog presence.
"""

from __future__ import annotations

import json

import pytest

from nvsh.cli import main
from nvsh.explain import catalog


@pytest.fixture(autouse=True)
def xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("NVSH_NO_DAEMON", "1")
    return tmp_path


# --- nvsh slash ------------------------------------------------------------


def test_slash_help_text(capsys):
    rc = main(["slash", "/help", "--platform", "dgx-spark"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "/ask" in out


def test_slash_json_shape(capsys):
    rc = main(["slash", "/help", "--platform", "dgx-spark", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    payload = json.loads(lines[-1])
    assert payload == {"command": "help", "exit_code": 0}


def test_slash_unknown_command_exit_code(capsys):
    rc = main(["slash", "/nope", "--platform", "dgx-spark"])
    assert rc == 1
    err = capsys.readouterr()
    assert "unknown" in err.out.lower()


def test_slash_power_hidden_on_dgx_spark(capsys):
    rc = main(["slash", "/power", "--platform", "dgx-spark"])
    assert rc == 1


def test_slash_power_stub_on_jetson(capsys):
    rc = main(["slash", "/power", "--platform", "jetson"])
    assert rc == 0
    assert "not implemented" in capsys.readouterr().out


# --- nvsh complete -----------------------------------------------------


def test_complete_json_shape_full_palette(capsys):
    rc = main(["complete", "--json", "--platform", "dgx-spark"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "items" in payload
    values = {item["value"] for item in payload["items"]}
    assert "/doctor" in values
    for item in payload["items"]:
        assert set(item) == {"value", "description"}


def test_complete_json_shape_matches_bash_contract():
    """The shape 'nvsh complete --json' emits must match docs/shell-integration.md
    and tests/fakes/nvsh's fixture used by the (hermetic) readline tests."""
    rc_and_out = main
    import io
    import sys

    captured = io.StringIO()
    old = sys.stdout
    sys.stdout = captured
    try:
        rc = rc_and_out(["complete", "--json", "--platform", "dgx-spark"])
    finally:
        sys.stdout = old
    assert rc == 0
    payload = json.loads(captured.getvalue())
    assert isinstance(payload["items"], list)
    for item in payload["items"]:
        assert isinstance(item["value"], str)
        assert isinstance(item["description"], str)


def test_complete_arguments_for_doctor(capsys):
    rc = main(["complete", "--json", "--platform", "dgx-spark", "--", "/doctor", "--st"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    values = {item["value"] for item in payload["items"]}
    assert "--strict" in values


def test_complete_hides_jetson_only_commands_on_dgx_spark(capsys):
    assert main(["complete", "--json", "--platform", "dgx-spark"]) == 0
    payload = json.loads(capsys.readouterr().out)
    values = {item["value"] for item in payload["items"]}
    assert "/power" not in values


def test_complete_shows_jetson_only_commands_on_jetson(capsys):
    assert main(["complete", "--json", "--platform", "jetson"]) == 0
    payload = json.loads(capsys.readouterr().out)
    values = {item["value"] for item in payload["items"]}
    assert "/power" in values


def test_complete_text_mode(capsys):
    rc = main(["complete", "--platform", "dgx-spark"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "/doctor" in out


# --- catalog presence ------------------------------------------------------


def test_catalog_has_slash_and_complete_entries():
    assert ("slash",) in catalog.ENTRIES
    assert ("complete",) in catalog.ENTRIES


def test_catalog_has_an_entry_per_slash_command():
    from nvsh import slash as slash_mod

    for cmd in slash_mod.visible_commands("jetson"):
        assert ("slash", cmd.name) in catalog.ENTRIES, cmd.name


def test_explain_slash_verb(capsys):
    rc = main(["explain", "slash"])
    assert rc == 0
    assert "nvsh slash" in capsys.readouterr().out


def test_explain_complete_verb(capsys):
    rc = main(["explain", "complete"])
    assert rc == 0
    assert "nvsh complete" in capsys.readouterr().out
