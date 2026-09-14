"""Tests for the ``nvsh setup`` / ``uninstall`` / ``on`` / ``off`` / ``hook`` verbs."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

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

    from nvsh.cli._commands import setup as setup_mod

    # `setup` now stops the daemon after writing a new default, and
    # `uninstall` always does: point the resolved entrypoint at /bin/true so
    # no test spawns a real `nvsh daemon stop` (or, once `shutil.which` is
    # faked, whatever sys.argv[0] happens to be under pytest).
    monkeypatch.setattr(setup_mod.render, "resolve_nvsh_bin", lambda: "/bin/true")
    # No test reads stdin: a tty is opted into explicitly, per test.
    monkeypatch.setattr(setup_mod, "_is_interactive", lambda: False)
    monkeypatch.setattr(
        setup_mod,
        "_prompt_input",
        lambda prompt: pytest.fail("setup must not read stdin"),
    )
    # Reachability is a real subprocess/network probe; default it to a
    # passing stub so tests that don't care about it stay fast and hermetic.
    monkeypatch.setattr(
        setup_mod,
        "_check_agent_reachable",
        lambda cfg: {"passed": True, "message": "stub: reachable"},
    )


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
    assert "name" in payload["agent"]
    assert "reason" in payload["agent"]


def test_setup_writes_aliases_default(tmp_path):
    """'nvsh setup' persists the chosen backend as [aliases].default."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["default_alias_written"] is True

    from nvsh.config import DEFAULT_ALIAS, load

    cfg = load()
    assert cfg.aliases[DEFAULT_ALIAS] == payload["agent"]["name"]


def test_setup_is_idempotent_for_the_default_alias(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    code, out, err = _run(["setup", "--rc", str(rc), "--json"])
    assert code == 0, err
    payload = json.loads(out)
    # the second run finds [aliases].default already matching, so it writes
    # nothing further.
    assert payload["agent"]["default_alias_written"] is False


def test_setup_never_overwrites_agent_provider_when_falling_back(tmp_path, monkeypatch):
    """A fallback pick (e.g. openai-compat, nothing else on PATH) only touches
    [aliases].default -- [agent] provider (the operator's own intent) is left
    alone."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    cfg_dir = tmp_path / "xdg-config" / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text('[agent]\nprovider = "pi"\n', encoding="utf-8")

    from nvsh.agent import registry

    which = _which_factory(set())  # nothing on PATH -- pi falls back to openai-compat
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", which)
    monkeypatch.setattr(registry, "choose", lambda cfg, *a, **k: ("openai-compat", "fallback"))

    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["name"] == "openai-compat"

    from nvsh.config import DEFAULT_ALIAS, load

    cfg = load()
    assert cfg.agent_provider == "pi"
    assert cfg.aliases[DEFAULT_ALIAS] == "openai-compat"


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
    # Nothing is on PATH, so the probe is empty and the pick is the
    # openai-compat fallback: the offers stay unscoped, which is the only
    # way an operator can bootstrap a harness on a bare machine.
    assert payload["agent"]["probe"] == []
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
            # The harness chooser looks pi up on the real PATH (its `which` is a
            # def-time default); a CI runner has no pi, so the fake one appears.
            monkeypatch.setenv(
                "PATH", str(Path(__file__).parent / "fakes") + os.pathsep + os.environ["PATH"]
            )
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

    monkeypatch.setattr(registry, "probe", lambda *a, **k: [])
    monkeypatch.setattr(registry, "choose", lambda cfg, *a, **k: ("openai-compat", "fallback"))
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert "api_key" in out


def test_setup_agent_choice_json_carries_the_key_hint(tmp_path, monkeypatch):
    from nvsh.agent import registry

    monkeypatch.setattr(registry, "probe", lambda *a, **k: [])
    monkeypatch.setattr(registry, "choose", lambda cfg, *a, **k: ("openai-compat", "fallback"))
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install", "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert "api_key" in (payload["agent"]["key_hint"] or "")


def test_setup_agent_choice_has_no_key_hint_for_other_backends(tmp_path, monkeypatch):
    from nvsh.agent import registry

    monkeypatch.setattr(registry, "probe", lambda *a, **k: [])
    monkeypatch.setattr(registry, "choose", lambda cfg, *a, **k: ("pi", "on PATH"))
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install", "--json"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["key_hint"] is None


# --------------------------------------------------------------------------
# PR #8 review
# --------------------------------------------------------------------------


def test_rc_block_guard_treats_disable_zero_as_enabled(tmp_path):
    """``NVSH_DISABLE=0`` is the documented "off switch off" value.

    The hook's own kill switch reads ``-n $NVSH_DISABLE && != 0``; the rc
    block skipped sourcing on *any* non-empty value, so an operator who set
    the documented false value got no hook at all.
    """
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])
    block = rc.read_text().split("# >>> nvsh setup >>>", 1)[1]
    guard = next(ln for ln in block.splitlines() if "hook.bash" in ln)

    # Keep the condition, swap the sourcing for a marker: this asserts on the
    # guard bash actually runs, not on a restatement of it.
    condition, _, _body = guard.partition("|| ")
    probe = f"{condition}|| __NVSH_PROBE=1"
    for value, expect in (("", "1"), ("0", "1"), ("1", "0"), ("yes", "0")):
        proc = subprocess.run(  # noqa: S603
            ["bash", "-c", f'{probe}\necho "SOURCED=${{__NVSH_PROBE:-0}}"'],
            capture_output=True,
            text=True,
            env={"PATH": os.environ.get("PATH", ""), "NVSH_DISABLE": value},
            check=False,
        )
        assert f"SOURCED={expect}" in proc.stdout, (value, proc.stdout, proc.stderr)


def test_uninstall_stops_the_daemon_before_unlinking_its_socket(tmp_path, monkeypatch):
    """Unlinking first makes the running daemon unreachable through its socket."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    runtime = tmp_path / "run" / "nvsh"
    runtime.mkdir(parents=True)
    sock = runtime / "daemon.sock"
    sock.write_text("")
    _run(["setup", "--rc", str(rc), "--json"])

    from nvsh.cli._commands import setup as setup_mod

    seen: list[bool] = []

    def fake_run(argv, **kwargs):
        seen.append(sock.exists())
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(setup_mod.subprocess, "run", fake_run)
    code, out, err = _run(["uninstall", "--rc", str(rc), "--json"])
    assert code == 0, err
    assert seen == [True], "daemon stop must run while the socket is still bound"
    assert json.loads(out)["daemon_stopped"] is True
    assert not sock.exists()


def test_uninstall_reports_a_daemon_that_did_not_stop(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--rc", str(rc), "--json"])

    from nvsh.cli._commands import setup as setup_mod

    monkeypatch.setattr(
        setup_mod.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1),
    )
    code, out, err = _run(["uninstall", "--rc", str(rc), "--json"])
    assert code == 0, err
    assert json.loads(out)["daemon_stopped"] is False


# --------------------------------------------------------------------------
# setup: --agent, the harness probe and the pick (task t4)
# --------------------------------------------------------------------------


def _setup_mod():
    from nvsh.cli._commands import setup as setup_mod

    return setup_mod


def _default_alias():
    from nvsh.config import DEFAULT_ALIAS, load

    return load().aliases.get(DEFAULT_ALIAS)


def test_setup_agent_flag_writes_the_literal_target(tmp_path, monkeypatch):
    """--agent keeps model and effort: the whole target is the default alias."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"codex", "uv", "tmux"})
    )
    code, out, err = _run(
        ["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "codex/gpt-5/high"]
    )
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["name"] == "codex"
    assert payload["agent"]["target"] == "codex/gpt-5/high"
    assert payload["agent"]["probe"] == []  # --agent skips the probe entirely
    assert _default_alias() == "codex/gpt-5/high"


def test_setup_agent_flag_with_the_binary_missing_is_an_env_error(tmp_path, monkeypatch):
    """A forced target that isn't installed fails loudly and writes no config."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", _which_factory(set()))
    code, out, err = _run(
        ["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "codex/gpt-5/high"]
    )
    assert code == 2, (code, out, err)
    assert "codex" in (out + err)
    assert _default_alias() is None
    assert not (tmp_path / "xdg-config" / "nvsh" / "config.toml").exists()


def test_setup_one_probed_harness_becomes_the_default_silently(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    # A terminal is available -- one hit must still not prompt.
    monkeypatch.setattr(_setup_mod(), "_is_interactive", lambda: True)
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["name"] == "claude"
    assert [row["name"] for row in payload["agent"]["probe"]] == ["claude"]
    assert _default_alias() == "claude"


def test_setup_several_harnesses_prompt_once_on_a_tty(tmp_path, monkeypatch):
    """The operator picks from the ordered list; plan-mode rows say so."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which",
        _which_factory({"claude", "qwen", "node", "npm", "uv", "tmux"}),
    )
    setup_mod = _setup_mod()
    monkeypatch.setattr(setup_mod, "_is_interactive", lambda: True)
    prompts = []

    def fake_prompt(text):
        prompts.append(text)
        return "2"

    monkeypatch.setattr(setup_mod, "_prompt_input", fake_prompt)

    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert len(prompts) == 1, prompts
    names = [row for row in prompts[0].splitlines() if ")" in row]
    assert len(names) >= 2
    assert "claude" in prompts[0]
    assert "qwen" in prompts[0]
    # qwen has no approval channel, so it is marked before the pick is made.
    qwen_line = next(line for line in prompts[0].splitlines() if "qwen" in line)
    assert setup_mod.PLAN_MODE_LABEL in qwen_line
    assert _default_alias() == "qwen"


def test_setup_yes_does_not_answer_the_harness_pick(tmp_path, monkeypatch):
    """--yes approves installs only; the operator still chooses the harness."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which",
        _which_factory({"claude", "qwen", "node", "npm", "uv", "tmux"}),
    )
    setup_mod = _setup_mod()
    monkeypatch.setattr(setup_mod, "_is_interactive", lambda: True)
    prompts = []
    monkeypatch.setattr(setup_mod, "_prompt_input", lambda text: prompts.append(text) or "")
    code, out, err = _run(["setup", "--rc", str(rc), "--yes"])
    assert code == 0, err
    assert len(prompts) == 1, "the pick is asked even with --yes"
    # An empty answer takes the first (tool-calling) row.
    assert _default_alias() == "claude"


def test_setup_without_a_tty_takes_the_first_probe_row(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which",
        _which_factory({"claude", "qwen", "node", "npm", "uv", "tmux"}),
    )
    # _prompt_input is the autouse fixture's pytest.fail, so any prompt fails.
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    rows = [row["name"] for row in payload["agent"]["probe"]]
    assert rows[0] == "claude"
    assert "qwen" in rows
    assert payload["agent"]["name"] == "claude"
    assert _default_alias() == "claude"


def test_setup_re_probes_a_sticky_openai_compat_default(tmp_path, monkeypatch):
    """The bare-machine fallback must not outlive the machine being bare."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", _which_factory(set()))
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    assert _default_alias() == "openai-compat"

    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["name"] == "claude"
    assert payload["agent"]["default_alias_written"] is True
    assert _default_alias() == "claude"


def test_setup_keeps_an_openai_compat_default_that_has_a_base_url(tmp_path, monkeypatch):
    """An operator who pointed openai-compat somewhere meant it: keep it."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    cfg_dir = tmp_path / "xdg-config" / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[aliases]\ndefault = "openai-compat"\n\n'
        '[agents.openai-compat]\nbase_url = "http://localhost:8000/v1"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["default_alias_written"] is False
    assert _default_alias() == "openai-compat"


def test_setup_install_offers_are_scoped_to_the_pick(tmp_path, monkeypatch):
    """With only claude on PATH there is no pi row and no key hint."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "apt-get", "snap"})
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    names = {item["tool"] for item in payload["installs"]}
    assert "pi" not in names
    assert names == {"uv", "tmux"}
    assert payload["agent"]["key_hint"] is None


def test_setup_stops_the_daemon_after_writing_the_default(tmp_path, monkeypatch):
    """A warm daemon holds the previous default's session, so it has to go."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    stopped = []
    monkeypatch.setattr(
        _setup_mod(), "_stop_daemon", lambda nvsh_bin: stopped.append(nvsh_bin) or True
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    assert json.loads(out)["daemon_stopped"] is True
    assert len(stopped) == 1

    # A second, no-op setup changes no default, so it leaves the daemon alone.
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    assert json.loads(out)["daemon_stopped"] is False
    assert len(stopped) == 1


# --------------------------------------------------------------------------
# setup: hosted line, reachability report, macOS/zsh warning (task t7)
# --------------------------------------------------------------------------


def test_setup_hosted_line_present_for_hosted_pick(tmp_path, monkeypatch):
    """A hosted pick (claude) gets the data-egress disclosure line in text mode."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert (
        "claude is hosted: on a failure the redacted command, output and "
        "device context leave this machine"
    ) in out


def test_setup_no_hosted_line_for_non_hosted_pick(tmp_path, monkeypatch):
    """pi is not hosted: no data-egress line for it."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", _which_factory({"pi"}))
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert "is hosted" not in out


def test_setup_json_carries_hosted_flag(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["hosted"] is True

    monkeypatch.setattr("nvsh.cli._commands.setup.shutil.which", _which_factory({"pi"}))
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["hosted"] is False


def test_setup_reports_agent_reachable_without_changing_exit_code(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    setup_mod = _setup_mod()
    monkeypatch.setattr(
        setup_mod,
        "_check_agent_reachable",
        lambda cfg: {"passed": False, "message": "'claude' is not on PATH (claude-missing)"},
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["reachable"] == {
        "passed": False,
        "message": "'claude' is not on PATH (claude-missing)",
    }


def test_setup_reachable_call_wraps_exceptions(tmp_path, monkeypatch):
    """A reachability probe that raises never fails or hangs setup."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    setup_mod = _setup_mod()

    def _boom(cfg):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(setup_mod, "_check_agent_reachable", _boom)
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["reachable"] == {"passed": False, "message": "probe exploded"}


def test_setup_darwin_warning(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setattr("nvsh.cli._commands.setup.platform.system", lambda: "Darwin")
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert "nvsh is not tested on macOS/zsh yet (see issue #11)" in payload["warnings"]
    assert payload["rc"] == str(rc)  # still the bash rc

    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert "warning: nvsh is not tested on macOS/zsh yet (see issue #11)" in err
    assert "warning:" not in out


def test_setup_zsh_warning(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setattr("nvsh.cli._commands.setup.platform.system", lambda: "Linux")
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert "nvsh is not tested on macOS/zsh yet (see issue #11)" in payload["warnings"]
    assert payload["rc"] == str(rc)  # still the bash rc

    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert "warning: nvsh is not tested on macOS/zsh yet (see issue #11)" in err
    assert "warning:" not in out


def test_setup_no_warning_on_linux_bash(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setattr("nvsh.cli._commands.setup.platform.system", lambda: "Linux")
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["warnings"] == []

    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert "warning:" not in out


def test_setup_json_stays_parseable_when_daemon_stop_prints(tmp_path, monkeypatch):
    """``nvsh daemon stop`` says "daemon: not running" on its own stdout; that
    line must never land ahead of setup's ``--json`` payload."""
    import stat

    chatty = tmp_path / "nvsh-chatty"
    chatty.write_text("#!/bin/sh\necho 'daemon: not running'\nexit 0\n", encoding="utf-8")
    chatty.chmod(chatty.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(_setup_mod().render, "resolve_nvsh_bin", lambda: str(chatty))
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["daemon_stopped"] is True


def test_setup_agent_accepts_a_bare_adapter_name(tmp_path, monkeypatch):
    """``nvsh setup --agent claude`` is the README's on-ramp; no alias needed."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which",
        _which_factory({"claude", "codex", "uv", "tmux"}),
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "codex"])
    assert code == 0, err
    payload = json.loads(out)
    assert payload["agent"]["name"] == "codex"
    assert _default_alias() == "codex"


# --------------------------------------------------------------------------
# PR #12 review: canonical --agent target, no discarded or repeated pick
# --------------------------------------------------------------------------


def _write_config(tmp_path, text):
    cfg_dir = tmp_path / "xdg-config" / "nvsh"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(text, encoding="utf-8")


def _assert_default_round_trips():
    """The persisted default must resolve to a registered adapter."""
    from nvsh.agent import registry
    from nvsh.config import DEFAULT_ALIAS, load

    backend, _model, _effort, _alias = load().resolve_target(DEFAULT_ALIAS)
    assert backend in registry.ADAPTERS, backend


def test_setup_agent_named_alias_writes_its_target_not_the_name(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _write_config(tmp_path, '[aliases]\nreviewer = "claude/opus/high"\n')
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(
        ["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "reviewer"]
    )
    assert code == 0, err
    assert json.loads(out)["agent"]["target"] == "claude/opus/high"
    assert _default_alias() == "claude/opus/high"
    _assert_default_round_trips()


def test_setup_agent_at_bare_name_writes_the_bare_name(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(
        ["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "@claude"]
    )
    assert code == 0, err
    assert _default_alias() == "claude"
    _assert_default_round_trips()


def test_setup_agent_bare_name_stays_bare_despite_a_configured_model(tmp_path, monkeypatch):
    """A model only from [agents.<backend>].model is not baked into the default."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _write_config(tmp_path, '[agents.claude]\nmodel = "sonnet"\n')
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "claude"])
    assert code == 0, err
    assert _default_alias() == "claude"
    _assert_default_round_trips()


def test_setup_agent_at_literal_target_strips_the_at(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"codex", "uv", "tmux"})
    )
    code, out, err = _run(
        ["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "@codex/gpt-5/high"]
    )
    assert code == 0, err
    assert _default_alias() == "codex/gpt-5/high"
    _assert_default_round_trips()


def test_setup_agent_default_never_writes_a_self_reference(tmp_path, monkeypatch):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _write_config(tmp_path, '[aliases]\ndefault = "claude/sonnet"\n')
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which", _which_factory({"claude", "uv", "tmux"})
    )
    code, out, err = _run(
        ["setup", "--rc", str(rc), "--json", "--no-install", "--agent", "default"]
    )
    assert code == 0, err
    assert _default_alias() == "claude/sonnet"
    _assert_default_round_trips()


def test_setup_kept_default_is_never_prompted_for(tmp_path, monkeypatch):
    """An installed existing default is kept before any pick menu is shown."""
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _write_config(tmp_path, '[aliases]\ndefault = "claude"\n')
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which",
        _which_factory({"claude", "qwen", "node", "npm", "uv", "tmux"}),
    )
    setup_mod = _setup_mod()
    monkeypatch.setattr(setup_mod, "_is_interactive", lambda: True)
    monkeypatch.setattr(
        setup_mod, "_prompt_input", lambda text: pytest.fail("a kept default must not prompt")
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--no-install"])
    assert code == 0, err
    assert "[aliases].default = 'claude' kept" in out
    assert _default_alias() == "claude"


def test_setup_non_harness_install_does_not_re_ask_the_pick(tmp_path, monkeypatch):
    """Installing uv cannot add a harness, so the pick menu is shown once."""
    from nvsh import installers

    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    monkeypatch.setattr(
        "nvsh.cli._commands.setup.shutil.which",
        _which_factory({"claude", "qwen", "node", "npm", "tmux"}),
    )
    setup_mod = _setup_mod()
    monkeypatch.setattr(setup_mod, "_is_interactive", lambda: True)
    calls = []
    monkeypatch.setattr(setup_mod, "_prompt_input", lambda text: calls.append(text) or "1")
    monkeypatch.setattr(
        setup_mod.installers,
        "run_install",
        lambda step, **kwargs: installers.InstallResult(tool=step.tool, ran=True, returncode=0),
    )
    code, out, err = _run(["setup", "--rc", str(rc), "--yes"])
    assert code == 0, err
    assert "uv (" in out
    assert len(calls) == 1, calls
    assert _default_alias() == "claude"
