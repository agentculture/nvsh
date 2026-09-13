"""Tests for the ``nvsh setup`` / ``uninstall`` / ``on`` / ``off`` / ``hook`` verbs."""

from __future__ import annotations

import io
import json
import sys
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


# --------------------------------------------------------------------------
# setup: missing-tool detection and install offers (deviation d1)
# --------------------------------------------------------------------------


def _which_factory(present: set[str]):
    def _which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return _which


def test_setup_json_lists_install_offers_and_runs_nothing(tmp_path, monkeypatch):
    """--json is non-interactive: offers are listed, nothing is run without --yes."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    which = _which_factory({"apt-get", "snap"})  # orin-like: no node/pi/uv/tmux
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", which)
    ran = []
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.installers.run_install",
        lambda *a, **k: ran.append((a, k)) or None,
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert "installs" in payload
    names = {item["tool"] for item in payload["installs"]}
    assert names == {"pi", "node", "uv", "tmux"}
    for item in payload["installs"]:
        assert item["ran"] is False
    assert ran == []


def test_setup_no_install_flag_lists_offers_and_runs_nothing(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    which = _which_factory({"apt-get", "snap"})
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", which)
    ran = []
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.installers.run_install",
        lambda *a, **k: ran.append((a, k)) or None,
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert len(payload["installs"]) == 4
    assert ran == []


def test_setup_yes_runs_planned_steps_in_order_and_records_audit(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    which = _which_factory({"apt-get", "snap"})
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", which)

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        import subprocess as _sp

        return _sp.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    import nvsh.installers as installers_mod

    monkeypatch.setattr(installers_mod.subprocess, "run", fake_run)

    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--yes"])
    assert code == 0, err
    payload = json.loads(out)
    ran_tools = [item["tool"] for item in payload["installs"] if item["ran"]]
    # pi cannot run (no npm on this fake machine); node/uv/tmux can.
    assert set(ran_tools) == {"node", "uv", "tmux"}
    assert [c[0] for c in calls if c[0] == "sudo"] or calls  # something ran

    from nvsh.agent.audit import AuditLog

    audit = AuditLog(path=tmp_path / "xdg-state" / "nvsh" / "audit.jsonl")
    entries = audit.read_all()
    install_entries = [e for e in entries if e["event"] == "install"]
    assert len(install_entries) == 4  # one per missing tool, including pi's non-executable one


def test_setup_curl_only_uv_is_never_executed_even_with_yes(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    # No snap, but curl present: uv can only offer the curl-pipe-sh installer.
    which = _which_factory({"apt-get", "curl", "node", "npm", "tmux"})
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", which)

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        import subprocess as _sp

        return _sp.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    import nvsh.installers as installers_mod

    monkeypatch.setattr(installers_mod.subprocess, "run", fake_run)

    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--yes"])
    assert code == 0, err
    payload = json.loads(out)
    uv_item = next(item for item in payload["installs"] if item["tool"] == "uv")
    assert uv_item["ran"] is False
    assert "curl" in uv_item["command"]
    assert all(argv[0] != "curl" for argv in calls)


def test_setup_reruns_agent_choice_after_installing_pi(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    # npm present so pi is plannable; pretend the install makes pi appear.
    state = {"pi_installed": False}

    def which(name: str) -> str | None:
        if name == "pi" and state["pi_installed"]:
            return "/usr/bin/pi"
        if name in {"apt-get", "npm", "node", "snap", "tmux", "uv"}:
            return f"/usr/bin/{name}"
        return None

    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", which)

    def fake_run(argv, **kwargs):
        if argv[:2] == ["npm", "install"]:
            state["pi_installed"] = True
        import subprocess as _sp

        return _sp.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    import nvsh.installers as installers_mod

    monkeypatch.setattr(installers_mod.subprocess, "run", fake_run)

    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--yes"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["name"] == "pi"


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


_HOOK_ARGV = [
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


def test_hook_ask_hands_off_to_the_failure_client(monkeypatch):
    """A qualifying failure reaches nvsh.client.handle_failure (task t13)."""
    import nvsh.client

    seen = []
    monkeypatch.setattr(nvsh.client, "handle_failure", lambda args: seen.append(args) or 0)
    code, _out, _err = _run(list(_HOOK_ARGV))
    assert code == 0
    assert len(seen) == 1
    assert seen[0].line == "ls /nope"
    assert seen[0].exit == 2


def test_hook_degrades_when_the_client_is_missing(monkeypatch):
    """A broken/absent client must never swallow the failure silently."""
    monkeypatch.setitem(sys.modules, "nvsh.client", None)
    code, _out, err = _run(list(_HOOK_ARGV))
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


# --------------------------------------------------------------------------
# the agent choice says where the gateway key goes (deviation d10)
# --------------------------------------------------------------------------


def test_setup_agent_choice_documents_the_key_file_for_openai_compat(tmp_path, monkeypatch):
    from nvsh.agent import registry

    monkeypatch.setattr(registry, "choose", lambda cfg: ("openai-compat", "fallback"))
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert "api_key" in out


def test_setup_agent_choice_json_carries_the_key_hint(tmp_path, monkeypatch):
    from nvsh.agent import registry

    monkeypatch.setattr(registry, "choose", lambda cfg: ("openai-compat", "fallback"))
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install", "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert "api_key" in (payload["agent"]["key_hint"] or "")


def test_setup_agent_choice_has_no_key_hint_for_other_backends(tmp_path, monkeypatch):
    from nvsh.agent import registry

    monkeypatch.setattr(registry, "choose", lambda cfg: ("pi", "on PATH"))
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install", "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["key_hint"] is None
