"""Tests for ``nvsh capture`` — prints the last captured-output slice."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nvsh.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "capture"


@pytest.fixture(autouse=True)
def xdg_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


def _install_fixture_log(xdg_runtime: Path, pid: int) -> Path:
    log_dir = xdg_runtime / "nvsh"
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"{pid}.log"
    target.write_bytes((FIXTURES / "ts.log").read_bytes())
    return target


def test_capture_show_json(capsys, xdg_runtime, monkeypatch):
    pid = os.getppid()
    _install_fixture_log(xdg_runtime, pid)
    rc = main(["capture", "--show", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert "hello-out" in payload["text"]
    assert payload["source"] in ("script", "none", "tmux")
    assert isinstance(payload["redaction_rules"], list)
    assert isinstance(payload["bytes_total"], int)


def test_capture_show_text(capsys, xdg_runtime):
    pid = os.getppid()
    _install_fixture_log(xdg_runtime, pid)
    rc = main(["capture", "--show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hello-out" in out


def test_capture_show_explicit_pid(capsys, xdg_runtime):
    _install_fixture_log(xdg_runtime, 424242)
    rc = main(["capture", "--show", "--pid", "424242", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"


def test_capture_show_no_log_json(capsys, xdg_runtime):
    rc = main(["capture", "--show", "--pid", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no capture"


def test_capture_show_never_leaks_log_path(capsys, xdg_runtime):
    pid = os.getppid()
    log = _install_fixture_log(xdg_runtime, pid)
    rc = main(["capture", "--show", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(log) not in out


def test_capture_bare_defaults_to_show(capsys, xdg_runtime):
    pid = os.getppid()
    _install_fixture_log(xdg_runtime, pid)
    rc = main(["capture"])
    assert rc == 0
    assert capsys.readouterr().out.strip()


def test_capture_unknown_flag_structured_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["capture", "--bogus"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err
