"""Tests for ``nvsh agent`` — list/use/install (task t10, extended by t5).

t5 covers every one of the nine :data:`nvsh.agent.registry.ADAPTERS` keys
for both ``use`` and ``install``, and routes ``install`` through
:func:`nvsh.installers.run_install` (so the audit log gets a row) instead of
an inline ``subprocess.run`` call.
"""

from __future__ import annotations

import json
import re
import sys

import pytest

from nvsh.agent import registry
from nvsh.cli import main

ADAPTER_NAMES = sorted(registry.ADAPTERS)

#: The subset of ADAPTERS that get an executable npm step (harness_install_step)
#: when npm is on PATH -- pi (its own dedicated command) plus the four
#: NPM_PACKAGES entries. agy/kiro/openai-compat/demo have no known installer.
EXECUTABLE_ADAPTER_NAMES = ["pi", "qwen", "qwen-p", "claude", "codex"]
NO_INSTALLER_ADAPTER_NAMES = ["agy", "kiro", "openai-compat", "demo"]


@pytest.fixture(autouse=True)
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _fake_npm_run(monkeypatch):
    """Never let a stray ``npm install -g`` actually reach the network.

    ``installers.run_install``'s default ``run`` is ``subprocess.run``; every
    test that wants to observe a real invocation replaces this fake with its
    own tracking callable via ``monkeypatch``.
    """
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("nvsh.installers.subprocess.run", _fake_run)
    return calls


def test_agent_list_json_reports_all_adapters(capsys):
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in payload["adapters"]}
    assert names == {
        "pi",
        "qwen",
        "qwen-p",
        "claude",
        "codex",
        "agy",
        "kiro",
        "openai-compat",
        "demo",
    }
    for row in payload["adapters"]:
        assert "installed" in row
        assert "binary" in row
        assert "description" in row
        assert "configured" in row
        assert "path" in row
        assert "hosted" in row
        assert "capabilities" in row
        assert "default" in row


def test_agent_list_marks_configured_provider(capsys):
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in payload["adapters"]}
    # default provider is 'pi'
    assert by_name["pi"]["configured"] is True
    assert by_name["claude"]["configured"] is False


def test_agent_list_text(capsys):
    rc = main(["agent", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pi" in out
    assert "openai-compat" in out


def test_agent_list_reports_path_and_hosted(capsys):
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in payload["adapters"]}
    assert by_name["pi"]["path"] == "rpc"
    assert by_name["pi"]["hosted"] is False
    assert by_name["claude"]["path"] == "stream-json"
    assert by_name["claude"]["hosted"] is True
    assert by_name["codex"]["hosted"] is True
    assert by_name["agy"]["hosted"] is True
    assert by_name["kiro"]["hosted"] is True
    assert by_name["qwen"]["hosted"] is False
    assert by_name["openai-compat"]["hosted"] is False


def test_agent_list_reports_capabilities(capsys):
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in payload["adapters"]}
    # every adapter's __init__ is cheap (no subprocess started), so every
    # one of the 8 built-in adapters should report real capabilities, never
    # null, when constructed with default config.
    for name, row in by_name.items():
        assert row["capabilities"] is not None, name
        assert "streaming" in row["capabilities"]


def test_agent_list_json_shows_steer_capability_for_every_adapter(capsys):
    """stop-choice-prompt AC4: 'nvsh agent list --json' shows Capabilities.steer
    for every adapter, true for exactly pi and codex -- exercised through the
    real CLI path (main -> cmd_agent_list -> build_adapter_rows), not by
    reading the source."""
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in payload["adapters"]}
    assert set(by_name) == set(ADAPTER_NAMES)
    steerable = set()
    for name, row in by_name.items():
        assert isinstance(row["capabilities"]["steer"], bool), name
        if row["capabilities"]["steer"]:
            steerable.add(name)
    assert steerable == {"pi", "codex"}


def test_agent_list_default_backend_sorts_first(capsys):
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    adapters = payload["adapters"]
    # default provider is 'pi' with no config on disk
    assert adapters[0]["name"] == "pi"
    assert adapters[0]["default"] is True
    assert sum(1 for row in adapters if row["default"]) == 1


def test_agent_list_default_follows_aliases_default_override(capsys, xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agent]\nprovider = "pi"\n\n[aliases]\ndefault = "claude"\n', encoding="utf-8"
    )
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    adapters = payload["adapters"]
    assert adapters[0]["name"] == "claude"
    assert adapters[0]["default"] is True
    by_name = {row["name"]: row for row in adapters}
    assert by_name["pi"]["default"] is False


def test_agent_list_resolves_default_with_only_agent_provider_configured(capsys, xdg_home):
    """A config with only [agent] provider (no [aliases] table) still resolves 'default'."""
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text('[agent]\nprovider = "codex"\n', encoding="utf-8")
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in payload["adapters"]}
    assert by_name["codex"]["default"] is True
    assert payload["adapters"][0]["name"] == "codex"


def test_agent_list_text_tags_hosted_and_default(capsys):
    rc = main(["agent", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = {line: line for line in out.splitlines()}
    pi_line = next(line for line in lines if line.startswith("pi "))
    claude_line = next(line for line in lines if line.startswith("claude "))
    assert "default" in pi_line
    assert "hosted" in claude_line


def test_agent_use_writes_config(capsys, xdg_home):
    rc = main(["agent", "use", "claude", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "claude"

    from nvsh.config import DEFAULT_ALIAS, load

    cfg = load()
    assert cfg.agent_provider == "claude"
    assert cfg.aliases[DEFAULT_ALIAS] == "claude"


@pytest.mark.parametrize("name", [n for n in ADAPTER_NAMES if n != "demo"])
def test_agent_use_accepts_every_registered_adapter(capsys, xdg_home, name):
    """Every adapter but ``demo`` can be the persisted default; ``demo`` is a
    scripted fixture and is refused (covered by test_demo_default_refused.py)."""
    rc = main(["agent", "use", name, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == name

    from nvsh.config import DEFAULT_ALIAS, load

    cfg = load()
    assert cfg.agent_provider == name
    assert cfg.aliases[DEFAULT_ALIAS] == name


def _normalize_wrapped_help(text: str) -> str:
    """Undo argparse's textwrap line-wrapping (including mid-hyphen breaks).

    argparse's textwrap can break a hyphenated adapter name like
    ``openai-compat`` across a line (``openai-\\n              compat``), so
    a naive substring check on the raw help text can miss a name that is
    plainly listed. Join hyphen line-wraps back together first, then
    collapse the remaining whitespace to single spaces.
    """
    joined = re.sub(r"-\n\s*", "-", text)
    return " ".join(joined.split())


def test_agent_use_help_lists_all_nine_adapters(capsys):
    with pytest.raises(SystemExit):
        main(["agent", "use", "--help"])
    out = _normalize_wrapped_help(capsys.readouterr().out)
    for name in registry.ADAPTERS:
        assert name in out, name


def test_agent_use_unknown_name_is_user_error(capsys):
    rc = main(["agent", "use", "not-a-backend"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


def test_agent_use_preserves_other_config(capsys, xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agent]\nprovider = "pi"\n\n[sessions]\nmax = 3\n', encoding="utf-8"
    )
    rc = main(["agent", "use", "codex", "--json"])
    assert rc == 0

    from nvsh.config import load

    cfg = load()
    assert cfg.agent_provider == "codex"
    assert cfg.sessions_max == 3


@pytest.mark.parametrize("name", ADAPTER_NAMES)
def test_agent_install_prints_step_from_harness_install_step(capsys, monkeypatch, name):
    """Every adapter's 'install' step matches installers.harness_install_step."""
    from nvsh import installers

    monkeypatch.setattr("nvsh.installers.shutil.which", lambda tool: "/usr/bin/" + tool)
    expected = installers.harness_install_step(name)

    rc = main(["agent", "install", name, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == expected.shell_line
    assert payload["executable"] == expected.executable
    assert payload["needs_sudo"] == expected.needs_sudo
    assert payload["ran"] is False  # no --yes, no tty in the test runner


def test_agent_install_help_lists_all_nine_adapters(capsys):
    with pytest.raises(SystemExit):
        main(["agent", "install", "--help"])
    out = _normalize_wrapped_help(capsys.readouterr().out)
    for name in registry.ADAPTERS:
        assert name in out, name


@pytest.mark.parametrize("name", EXECUTABLE_ADAPTER_NAMES)
def test_agent_install_runs_only_with_yes_when_executable(capsys, monkeypatch, name, _fake_npm_run):
    monkeypatch.setattr("nvsh.installers.shutil.which", lambda tool: "/usr/bin/" + tool)

    rc = main(["agent", "install", name, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ran"] is False
    assert _fake_npm_run == []

    rc = main(["agent", "install", name, "--yes", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ran"] is True
    assert len(_fake_npm_run) == 1


@pytest.mark.parametrize("name", EXECUTABLE_ADAPTER_NAMES)
def test_agent_install_never_runs_without_yes_on_a_non_tty(
    capsys, monkeypatch, name, _fake_npm_run
):
    monkeypatch.setattr("nvsh.installers.shutil.which", lambda tool: "/usr/bin/" + tool)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    rc = main(["agent", "install", name, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ran"] is False
    assert _fake_npm_run == []


@pytest.mark.parametrize("name", ["agy", "kiro"])
def test_agent_install_agy_and_kiro_print_no_known_installer(capsys, name):
    rc = main(["agent", "install", name])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no known installer" in out


def test_agent_install_records_an_audit_row(monkeypatch, tmp_path, xdg_home, _fake_npm_run):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("nvsh.installers.shutil.which", lambda tool: "/usr/bin/" + tool)

    rc = main(["agent", "install", "pi", "--yes", "--json"])
    assert rc == 0

    from nvsh.agent.audit import default_audit_path

    audit_path = default_audit_path()
    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "install"
    assert entry["decision"] is True


def test_agent_install_reports_env_error_when_the_tool_fails(capsys, monkeypatch):
    """Qodo #8: a failed install (nonzero returncode) must not exit 0."""
    monkeypatch.setattr("nvsh.installers.shutil.which", lambda tool: "/usr/bin/" + tool)

    def _fake_run(argv, **kwargs):
        import subprocess as _subprocess

        return _subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr="EACCES")

    monkeypatch.setattr("nvsh.installers.subprocess.run", _fake_run)

    rc = main(["agent", "install", "claude", "--yes", "--json"])
    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ran"] is True
    assert payload["returncode"] == 1
    assert "claude" in captured.err
    assert "1" in captured.err
    assert "hint:" in captured.err


def test_agent_install_declined_is_still_exit_0(capsys, monkeypatch, _fake_npm_run):
    """A declined install (no --yes, no tty) is not a failure."""
    monkeypatch.setattr("nvsh.installers.shutil.which", lambda tool: "/usr/bin/" + tool)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    rc = main(["agent", "install", "claude", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ran"] is False
    assert _fake_npm_run == []


@pytest.mark.parametrize("name", ["agy", "kiro"])
def test_agent_install_non_executable_step_is_still_exit_0(capsys, monkeypatch, name):
    """A deliberately non-executable step (ran=False, returncode=None) is exit 0."""
    rc = main(["agent", "install", name, "--yes", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ran"] is False
    assert payload["returncode"] is None


def test_agent_install_closed_stdin_declines_instead_of_crashing(
    capsys, monkeypatch, _fake_npm_run
):
    """Qodo #9: a closed/unavailable stdin must decline, not raise."""
    monkeypatch.setattr("nvsh.installers.shutil.which", lambda tool: "/usr/bin/" + tool)

    class _RaisingIsatty:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stdin", _RaisingIsatty())

    rc = main(["agent", "install", "claude", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ran"] is False
    assert _fake_npm_run == []


def test_agent_install_unknown_target_is_user_error(capsys):
    rc = main(["agent", "install", "not-a-backend"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


# --- where to put the gateway key (deviation d10) --------------------------


def test_agent_use_openai_compat_says_where_to_put_the_key(capsys, xdg_home):
    rc = main(["agent", "use", "openai-compat"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "api_key" in out


def test_agent_use_openai_compat_json_reports_the_bearer_source(capsys, xdg_home):
    rc = main(["agent", "use", "openai-compat", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "openai-compat"
    assert payload["bearer_source"] is None
    assert "api_key" in payload["note"]


def test_agent_use_openai_compat_with_a_key_file_reports_its_source(capsys, xdg_home):
    key_file = xdg_home / "nvsh" / "api_key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text("a-bearer-value\n", encoding="utf-8")
    key_file.chmod(0o600)
    rc = main(["agent", "use", "openai-compat", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bearer_source"] == "the default key file"
    assert payload["note"] is None
    assert "a-bearer-value" not in json.dumps(payload)


def test_agent_use_other_backend_has_no_bearer_fields(capsys, xdg_home):
    rc = main(["agent", "use", "claude", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "bearer_source" not in payload
