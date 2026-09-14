"""Task t2: refuse ``demo`` as a *persisted* default agent.

``demo`` (task t1) is a scripted fixture replay (nvsh/agent/demo.py) meant
for the README recording, not a real backend. It must stay usable for one
request -- ``--agent demo``, ``@demo``, an alias pointing at ``demo`` -- but
never become what a bare ``--agent`` / the daemon's warm session resolves to
by default, because that would make every ordinary failure on this machine
replay the same canned fixture instead of calling a real backend.

Four acceptance criteria, in order:

1. ``nvsh agent use demo`` exits 1 with a hint naming demo a scripted
   fixture, in both text and ``--json`` mode.
2. ``nvsh setup --agent demo`` exits 1 with the same hint, in both text and
   ``--json`` mode, and touches nothing (no rc edit, no config write).
3. ``nvsh doctor --json`` includes a check ``default_target_not_demo`` that
   fails when ``config.toml`` has ``[aliases].default = "demo"``, and passes
   otherwise (including when nvsh runs with no config at all).
4. ``--agent demo``, ``@demo`` and an alias pointing at ``demo`` still work
   *per request* -- only persisting it as the default is refused.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from nvsh import doctor_checks
from nvsh.agent import registry
from nvsh.cli import main as cli_main
from nvsh.cli._errors import EXIT_USER_ERROR
from nvsh.config import Config


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli_main(argv)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(autouse=True)
def _xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("NVSH_HOOK_VERSION", raising=False)
    return tmp_path


# --------------------------------------------------------------------------
# 1. nvsh agent use demo
# --------------------------------------------------------------------------


def test_agent_use_demo_exits_1_with_scripted_fixture_hint():
    code, _out, err = _run(["agent", "use", "demo"])
    assert code == EXIT_USER_ERROR
    assert "demo" in err
    assert "scripted fixture" in err
    assert "hint:" in err


def test_agent_use_demo_json_exits_1_with_scripted_fixture_hint():
    code, _out, err = _run(["agent", "use", "demo", "--json"])
    assert code == EXIT_USER_ERROR
    payload = json.loads(err)
    assert "scripted fixture" in payload["message"]
    assert "demo" in payload["message"]
    assert payload["remediation"]


def test_agent_use_demo_never_persists_the_default(tmp_path):
    """A refused 'agent use demo' must not have written config.toml."""
    _run(["agent", "use", "demo"])
    config_toml = tmp_path / "xdg-config" / "nvsh" / "config.toml"
    assert not config_toml.exists()


# --------------------------------------------------------------------------
# 2. nvsh setup --agent demo
# --------------------------------------------------------------------------


UBUNTU_RC = """\
# ~/.bashrc

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac
"""


def _rc(tmp_path):
    return tmp_path / "home" / ".bashrc"


def test_setup_agent_demo_exits_1_with_scripted_fixture_hint(tmp_path):
    _rc(tmp_path).write_text(UBUNTU_RC)
    code, _out, err = _run(["setup", "--agent", "demo"])
    assert code == EXIT_USER_ERROR
    assert "demo" in err
    assert "scripted fixture" in err
    assert "hint:" in err


def test_setup_agent_demo_json_exits_1_with_scripted_fixture_hint(tmp_path):
    _rc(tmp_path).write_text(UBUNTU_RC)
    code, _out, err = _run(["setup", "--agent", "demo", "--json"])
    assert code == EXIT_USER_ERROR
    payload = json.loads(err)
    assert "scripted fixture" in payload["message"]
    assert "demo" in payload["message"]
    assert payload["remediation"]


def test_setup_agent_at_demo_is_also_refused(tmp_path):
    _rc(tmp_path).write_text(UBUNTU_RC)
    code, _out, err = _run(["setup", "--agent", "@demo"])
    assert code == EXIT_USER_ERROR
    assert "scripted fixture" in err


def test_setup_agent_demo_never_touches_the_rc_file(tmp_path):
    rc = _rc(tmp_path)
    rc.write_text(UBUNTU_RC)
    _run(["setup", "--agent", "demo"])
    assert rc.read_text() == UBUNTU_RC


def test_setup_agent_demo_never_writes_config(tmp_path):
    _rc(tmp_path).write_text(UBUNTU_RC)
    _run(["setup", "--agent", "demo"])
    config_toml = tmp_path / "xdg-config" / "nvsh" / "config.toml"
    assert not config_toml.exists()


# --------------------------------------------------------------------------
# 3. doctor's default_target_not_demo check
# --------------------------------------------------------------------------


def test_check_default_target_not_demo_fails_when_default_is_demo():
    cfg = Config(aliases={"default": "demo"})
    check = doctor_checks.check_default_target_not_demo(cfg)
    assert check["id"] == "default_target_not_demo"
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "demo" in check["message"]
    assert check["remediation"]


def test_check_default_target_not_demo_passes_for_a_real_backend():
    cfg = Config(agent_provider="claude")
    check = doctor_checks.check_default_target_not_demo(cfg)
    assert check["passed"] is True


def test_check_default_target_not_demo_passes_with_no_config():
    """A wheel install with no config.toml has nothing that could resolve to
    demo -- this must never be a failure."""
    check = doctor_checks.check_default_target_not_demo(None)
    assert check["passed"] is True


def test_check_default_target_not_demo_passes_with_no_aliases_table():
    """``[aliases].default`` unset falls back to ``[agent] provider``
    (``Config.resolve_target``'s own DEFAULT_ALIAS branch never raises), so
    a config with no alias table at all must still pass cleanly here."""
    cfg = Config()
    assert "default" not in cfg.aliases
    check = doctor_checks.check_default_target_not_demo(cfg)
    assert check["passed"] is True


def test_collect_checks_includes_default_target_not_demo():
    from nvsh.platform import Platform

    checks = doctor_checks.collect_checks(
        env={},
        current_version="1.2.3",
        config=Config(aliases={"default": "demo"}),
        config_error=None,
        platform=Platform(kind="generic", values=()),
        which=lambda name: None,
        run=lambda argv, timeout: (1, "", ""),
    )
    by_id = {c["id"]: c for c in checks}
    assert "default_target_not_demo" in by_id
    assert by_id["default_target_not_demo"]["passed"] is False


def test_doctor_cli_json_reports_failing_default_target_not_demo(tmp_path):
    config_dir = tmp_path / "xdg-config" / "nvsh"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[aliases]\ndefault = "demo"\n', encoding="utf-8")
    code, out, _err = _run(["doctor", "--json"])
    payload = json.loads(out)
    by_id = {c["id"]: c for c in payload["checks"]}
    assert "default_target_not_demo" in by_id
    assert by_id["default_target_not_demo"]["passed"] is False
    assert code != 0


def test_doctor_cli_json_default_target_not_demo_passes_with_no_config(tmp_path):
    code, out, _err = _run(["doctor", "--json"])
    payload = json.loads(out)
    by_id = {c["id"]: c for c in payload["checks"]}
    assert by_id["default_target_not_demo"]["passed"] is True


# --------------------------------------------------------------------------
# 4. --agent demo, @demo and an alias pointing at demo still work per request
# --------------------------------------------------------------------------


def test_forced_agent_demo_still_resolves_via_registry_choose():
    """The per-request path (client_transport.py / daemon.py's
    registry.choose(cfg, forced=target)) is unaffected: this task only
    refuses persisting demo as the default, never a forced pick."""
    cfg = Config()
    backend, reason = registry.choose(cfg, which=lambda name: None, forced="demo")
    assert backend == "demo"
    assert "demo" in reason


def test_forced_agent_at_demo_still_resolves():
    cfg = Config()
    backend, _reason = registry.choose(cfg, which=lambda name: None, forced="@demo")
    assert backend == "demo"


def test_alias_pointing_at_demo_still_resolves_per_request():
    cfg = Config(aliases={"scripted": "demo"})
    backend, reason = registry.choose(cfg, which=lambda name: None, forced="scripted")
    assert backend == "demo"
    assert "scripted" in reason


def test_agent_list_json_still_reports_demo_as_selectable():
    """'nvsh agent list' must keep showing demo -- refusing it as a
    persisted default is not the same as removing it from the registry."""
    code, out, _err = _run(["agent", "list", "--json"])
    assert code == 0
    payload = json.loads(out)
    names = {row["name"] for row in payload["adapters"]}
    assert "demo" in names


# --- review fix (PR #14, Qodo 2): aliases and stored defaults ------------------


def test_setup_refuses_an_alias_whose_target_is_demo(tmp_path, monkeypatch, capsys):
    import json

    from nvsh.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    (tmp_path / "cfg" / "nvsh").mkdir(parents=True)
    (tmp_path / "cfg" / "nvsh" / "config.toml").write_text(
        '[aliases]\ndemo-run = "demo"\n', encoding="utf-8"
    )
    rc = main(["setup", "--agent", "demo-run", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().err)
    assert "scripted fixture" in payload["message"]
    assert not (tmp_path / ".bashrc").exists()


def test_keep_existing_default_never_keeps_demo(tmp_path, monkeypatch):
    from nvsh import config as nvsh_config
    from nvsh.cli._commands.setup import _keep_existing_default

    cfg = nvsh_config.Config(aliases={nvsh_config.DEFAULT_ALIAS: "demo"})
    assert _keep_existing_default(cfg, probe_rows=[]) is False
