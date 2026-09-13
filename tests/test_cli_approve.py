"""Tests for ``nvsh approve`` — check/add/list/remove over the Approvals store."""

from __future__ import annotations

import json

import pytest

from nvsh.cli import main


@pytest.fixture(autouse=True)
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    runtime = tmp_path / "run"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    return tmp_path


def test_approve_check_matches_default_json(capsys):
    rc = main(["approve", "check", "nvidia-smi -q", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "user"
    assert payload["pattern"] == "nvidia-smi *"


def test_approve_check_no_match_json(capsys):
    rc = main(["approve", "check", "rm -rf /", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "ask"
    assert payload["pattern"] is None


def test_approve_check_text(capsys):
    rc = main(["approve", "check", "nvidia-smi -q"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "user" in out


def test_approve_add_and_list(capsys):
    rc = main(["approve", "add", "kubectl get *"])
    assert rc == 0
    capsys.readouterr()
    rc = main(["approve", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "kubectl get *" in payload["user"]


def test_approve_add_session_is_visible_to_a_later_process_with_its_scope(capsys):
    """d15: ``--session`` is worthless if it dies with this CLI process.

    The pattern belongs to the login session (the runtime dir), so a later
    ``nvsh approve list`` — a whole new process — must show it, under
    ``session`` and never under ``user``.
    """
    rc = main(["approve", "add", "kubectl get *", "--session"])
    assert rc == 0
    capsys.readouterr()
    rc = main(["approve", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "kubectl get *" in payload["session"]
    assert "kubectl get *" not in payload["user"]


def test_approve_list_text_labels_the_session_scope(capsys):
    main(["approve", "add", "kubectl get *", "--session"])
    capsys.readouterr()
    rc = main(["approve", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    session_block = out.split("session:", 1)[1]
    assert "kubectl get *" in session_block


def test_approve_check_sees_a_session_pattern_added_by_another_process(capsys):
    main(["approve", "add", "apt install foo", "--session"])
    capsys.readouterr()
    rc = main(["approve", "check", "apt install foo", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "session"


def test_approve_remove_drops_a_session_pattern_for_later_processes(capsys):
    main(["approve", "add", "apt install foo", "--session"])
    capsys.readouterr()
    rc = main(["approve", "remove", "apt install foo"])
    assert rc == 0
    capsys.readouterr()
    rc = main(["approve", "check", "apt install foo", "--json"])
    assert json.loads(capsys.readouterr().out)["decision"] == "ask"


def test_approve_add_refused_pattern_errors(capsys):
    rc = main(["approve", "add", "sudo *"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


def test_approve_remove(capsys):
    main(["approve", "add", "kubectl get *"])
    capsys.readouterr()
    rc = main(["approve", "remove", "kubectl get *"])
    assert rc == 0
    capsys.readouterr()
    rc = main(["approve", "list", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "kubectl get *" not in payload["user"]


def test_approve_list_text(capsys):
    rc = main(["approve", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nvsh *" in out
