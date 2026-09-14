"""Tests for ``nvsh agent`` — list/use/install (task t10)."""

from __future__ import annotations

import json

import pytest

from nvsh.cli import main


@pytest.fixture(autouse=True)
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_agent_list_json_reports_all_adapters(capsys):
    rc = main(["agent", "list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in payload["adapters"]}
    assert names == {"pi", "qwen", "qwen-p", "claude", "codex", "agy", "kiro", "openai-compat"}
    for row in payload["adapters"]:
        assert "installed" in row
        assert "binary" in row
        assert "description" in row
        assert "configured" in row


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


def test_agent_use_writes_config(capsys, xdg_home):
    rc = main(["agent", "use", "claude", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "claude"

    from nvsh.config import load

    cfg = load()
    assert cfg.agent_provider == "claude"


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


def test_agent_install_pi_prints_command_without_running(capsys, monkeypatch):
    ran = []
    monkeypatch.setattr("nvsh.cli._commands.agent.subprocess.run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr("nvsh.cli._commands.agent.shutil.which", lambda name: "/usr/bin/npm")
    rc = main(["agent", "install", "pi", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "npm install -g @earendil-works/pi-coding-agent"
    assert payload["ran"] is False
    assert ran == []


def test_agent_install_pi_runs_with_yes(capsys, monkeypatch):
    ran = []
    monkeypatch.setattr("nvsh.cli._commands.agent.subprocess.run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr("nvsh.cli._commands.agent.shutil.which", lambda name: "/usr/bin/npm")
    rc = main(["agent", "install", "pi", "--yes", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ran"] is True
    assert len(ran) == 1


def test_agent_install_unknown_target_is_user_error(capsys):
    rc = main(["agent", "install", "qwen"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")


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
