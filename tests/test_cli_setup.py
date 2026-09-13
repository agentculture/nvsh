"""Tests for the ``nvsh setup`` / ``uninstall`` / ``on`` / ``off`` / ``hook`` verbs."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from nvsh.cli import main as cli_main

UBUNTU_RC = """\
# ~/.bashrc

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

HISTCONTROL=ignoredups
"""


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("NVSH_HOOK_VERSION", raising=False)
    yield


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


def _rc(tmp_path):
    return tmp_path / "home" / ".bashrc"


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def test_setup_inserts_block_after_guard(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["block_inserted"] is True
    text = rc.read_text()
    assert "# >>> nvsh setup >>>" in text
    before_block = text.split("# >>> nvsh setup >>>", 1)[0]
    assert before_block.rstrip().endswith("esac")


def test_setup_writes_a_backup(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    backups = list(rc.parent.glob(".bashrc.nvsh-backup-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == UBUNTU_RC


def test_setup_is_idempotent_byte_identical_second_run(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    first_text = rc.read_text()
    code, out, err = _run(["setup", "--rc", str(rc), "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["block_inserted"] is False
    assert rc.read_text() == first_text


def test_setup_renders_shell_files(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    shell_dir = tmp_path / "xdg-data" / "nvsh" / "shell"
    assert (shell_dir / "hook.bash").is_file()
    assert (shell_dir / "readline.bash").is_file()


def test_setup_block_has_no_tilde_literal(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    block = rc.read_text().split("# >>> nvsh setup >>>", 1)[1]
    assert "~/" not in block


def test_setup_reports_agent_choice(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--json"])
    payload = json.loads(out)
    assert "agent" in payload
    assert "name" in payload["agent"] and "reason" in payload["agent"]


def test_setup_on_missing_rc_creates_it(tmp_path):
    rc = _rc(tmp_path)
    assert not rc.exists()
    code, out, err = _run(["setup", "--rc", str(rc), "--json"])
    assert code == 0, err
    assert rc.exists()
    assert "# >>> nvsh setup >>>" in rc.read_text()


# --------------------------------------------------------------------------
# uninstall
# --------------------------------------------------------------------------


def test_uninstall_restores_rc_byte_identical_to_backup(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    code, out, err = _run(["uninstall", "--rc", str(rc), "--json"])
    assert code == 0, err
    assert rc.read_text() == UBUNTU_RC


def test_uninstall_restores_backup_when_block_was_edited(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    tampered = rc.read_text().replace("NVSH_HOOK_VERSION", "NVSH_HOOK_VERSION_HACKED")
    rc.write_text(tampered)
    code, out, err = _run(["uninstall", "--rc", str(rc), "--json"])
    assert code == 0, err
    assert rc.read_text() == UBUNTU_RC


def test_uninstall_removes_rendered_shell_files(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    shell_dir = tmp_path / "xdg-data" / "nvsh" / "shell"
    assert (shell_dir / "hook.bash").exists()
    _run(["uninstall", "--rc", str(rc), "--json"])
    assert not (shell_dir / "hook.bash").exists()


def test_uninstall_removes_runtime_logs_and_socket(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    runtime = tmp_path / "run" / "nvsh"
    runtime.mkdir(parents=True)
    (runtime / "1234.log").write_text("log")
    (runtime / "daemon.sock").write_text("")
    _run(["setup", "--rc", str(rc), "--json"])
    _run(["uninstall", "--rc", str(rc), "--json"])
    assert not (runtime / "1234.log").exists()
    assert not (runtime / "daemon.sock").exists()


def test_uninstall_on_never_installed_rc_is_a_noop(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["uninstall", "--rc", str(rc), "--json"])
    assert code == 0, err
    assert rc.read_text() == UBUNTU_RC


# --------------------------------------------------------------------------
# on / off
# --------------------------------------------------------------------------


def test_off_shell_mode_prints_bash_snippet(tmp_path):
    code, out, err = _run(["off", "--shell"])
    assert code == 0, err
    assert "__nvsh_hook_unload" in out
    assert "__nvsh_readline_unbind" in out
    assert "NVSH_DISABLE=1" in out


def test_on_shell_mode_prints_source_lines(tmp_path):
    code, out, err = _run(["on", "--shell"])
    assert code == 0, err
    assert "unset NVSH_DISABLE" in out
    assert "source " in out
    assert "hook.bash" in out
    assert "readline.bash" in out


def test_off_without_shell_flag_prints_json_with_eval_field(tmp_path):
    code, out, err = _run(["off", "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert "eval" in payload
    assert "__nvsh_hook_unload" in payload["eval"]


# --------------------------------------------------------------------------
# hook
# --------------------------------------------------------------------------


def test_hook_skip_exits_zero_silently_for_success():
    code, out, err = _run(
        ["hook", "--exit", "0", "--pipestatus", "0", "--line", "true", "--cwd", "/tmp", "--log", ""]
    )
    assert code == 0
    assert out == ""


def test_hook_skip_for_sigint():
    code, out, err = _run(
        [
            "hook",
            "--exit",
            "130",
            "--pipestatus",
            "130",
            "--line",
            "some-cmd",
            "--cwd",
            "/tmp",
            "--log",
            "",
        ]
    )
    assert code == 0


def test_hook_ask_without_client_prints_placeholder():
    code, out, err = _run(
        [
            "hook",
            "--exit",
            "2",
            "--pipestatus",
            "2",
            "--line",
            "ls /nope",
            "--cwd",
            "/tmp",
            "--log",
            "",
        ]
    )
    assert code == 0
    assert "ls /nope" in err
    assert "exit 2" in err
    assert "agent client not installed" in err


def test_hook_prints_refresh_notice_once_per_session(tmp_path, monkeypatch):
    from nvsh import __version__

    monkeypatch.setenv("NVSH_HOOK_VERSION", "0.0.1")
    argv = [
        "hook",
        "--exit",
        "2",
        "--pipestatus",
        "2",
        "--line",
        "ls /nope",
        "--cwd",
        "/tmp",
        "--log",
        "",
    ]
    code1, out1, err1 = _run(argv)
    assert "run 'nvsh setup' to refresh" in err1
    code2, out2, err2 = _run(argv)
    assert "run 'nvsh setup' to refresh" not in err2
    assert __version__ != "0.0.1"
