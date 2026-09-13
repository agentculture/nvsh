"""Tests for the ``nvsh daemon`` verb group."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from nvsh import daemon as daemon_mod
from nvsh.agent.fake import FakeAgent
from nvsh.cli import main
from nvsh.config import Config
from nvsh.explain import catalog


def _xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name, sub in (
        ("XDG_RUNTIME_DIR", "run"),
        ("XDG_STATE_HOME", "state"),
        ("XDG_CONFIG_HOME", "config"),
    ):
        (tmp_path / sub).mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(tmp_path / sub))
    monkeypatch.setenv("HOME", str(tmp_path))


def test_daemon_status_json_when_not_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["daemon", "status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] is False
    assert payload["socket"].endswith("daemon.sock")


def test_daemon_status_text_when_not_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["daemon", "status"]) == 0
    assert "not running" in capsys.readouterr().out


def test_daemon_stop_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["daemon", "stop", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["stopped"] is False


def test_daemon_unregister_without_a_daemon_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["daemon", "unregister", "--shell", "4242", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["running"] is False


def test_daemon_run_foreground_exits_on_idle_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["daemon", "run", "--foreground", "--idle-timeout", "0.3"]) == 0


def test_daemon_bare_shows_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["daemon"]) == 0
    assert capsys.readouterr().out.strip()


def test_daemon_unknown_flag_is_a_structured_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["daemon", "status", "--bogus"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


def test_daemon_status_and_stop_against_a_live_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    daemon = daemon_mod.Daemon(Config(), env=dict(os.environ), agent_factory=lambda: FakeAgent([]))
    thread = daemon.start_background()
    try:
        assert main(["daemon", "status"]) == 0
        out = capsys.readouterr().out
        assert "running" in out
        assert "agents: 0" in out
        assert main(["daemon", "unregister", "--shell", "77"]) == 0
        assert main(["daemon", "stop"]) == 0
        assert "daemon stopped" in capsys.readouterr().out
    finally:
        daemon.shutdown()
        thread.join(timeout=5)


def test_daemon_run_detached_starts_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _xdg(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))
    assert main(["daemon", "run", "--idle-timeout", "5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["started"] is True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not daemon_mod.is_running():
        time.sleep(0.02)
    try:
        assert daemon_mod.is_running()
    finally:
        assert main(["daemon", "stop"]) == 0


@pytest.mark.parametrize(
    "path",
    [
        ("daemon",),
        ("daemon", "run"),
        ("daemon", "status"),
        ("daemon", "stop"),
        ("daemon", "unregister"),
    ],
)
def test_catalog_has_daemon_entries(path: tuple[str, ...]) -> None:
    assert path in catalog.ENTRIES
    assert catalog.ENTRIES[path].strip().startswith("# nvsh daemon")
