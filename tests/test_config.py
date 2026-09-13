"""Tests for nvsh.config — the stdlib tomllib config loader.

Acceptance criterion covered (task t5):
config loads $XDG_CONFIG_HOME/nvsh/config.toml with tomllib, defaults to
[agent] provider='pi' and [agents.pi] provider='nemotron' model='associate',
sessions.max=1, and never reads or stores API keys; missing file yields
defaults.
"""

from __future__ import annotations

import pytest

from nvsh.config import Config, ConfigError, default_toml, load


@pytest.fixture()
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_missing_file_yields_defaults(xdg_home):
    cfg = load()
    assert isinstance(cfg, Config)
    assert cfg.agent_provider == "pi"
    assert cfg.agents["pi"]["provider"] == "nemotron"
    assert cfg.agents["pi"]["model"] == "associate"
    assert cfg.sessions_max == 1


def test_missing_file_no_config_dir_at_all(tmp_path, monkeypatch):
    # XDG_CONFIG_HOME unset entirely -> falls back to ~/.config via Path.home().
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load()
    assert cfg.agent_provider == "pi"


def test_explicit_path_overrides_xdg(tmp_path, xdg_home):
    other = tmp_path / "elsewhere.toml"
    other.write_text('[agent]\nprovider = "openai-compat"\n', encoding="utf-8")
    cfg = load(path=other)
    assert cfg.agent_provider == "openai-compat"


def test_loads_overrides_from_toml(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [agent]
        provider = "openai-compat"

        [agents.openai-compat]
        base_url = "http://localhost:8000/v1"
        api_key_env = "NVSH_API_KEY"

        [sessions]
        max = 2
        """,
        encoding="utf-8",
    )
    cfg = load()
    assert cfg.agent_provider == "openai-compat"
    assert cfg.agents["openai-compat"]["base_url"] == "http://localhost:8000/v1"
    assert cfg.agents["openai-compat"]["api_key_env"] == "NVSH_API_KEY"
    assert cfg.sessions_max == 2


def test_defaults_survive_partial_override(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text("[sessions]\nmax = 3\n", encoding="utf-8")
    cfg = load()
    assert cfg.agent_provider == "pi"
    assert cfg.sessions_max == 3


def test_never_reads_api_key_literal(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [agents.openai-compat]
        api_key = "example-placeholder-value-here"
        """,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "api_key" in str(exc.value)


def test_unknown_top_level_key_rejected(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text('[bogus]\nfoo = "bar"\n', encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load()
    msg = str(exc.value)
    assert "bogus" in msg
    # error message lists the valid top-level keys
    assert "agent" in msg


def test_unknown_agent_subkey_rejected(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agents.pi]\nprovider = "nemotron"\nbogus_key = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "bogus_key" in str(exc.value)


def test_malformed_toml_raises_config_error(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text("this is not [valid toml", encoding="utf-8")
    with pytest.raises(ConfigError):
        load()


def test_default_toml_is_parseable_and_localhost_only():
    import tomllib

    text = default_toml()
    data = tomllib.loads(text)
    assert data["agent"]["provider"] == "pi"
    # No literal secrets, no non-localhost endpoints in the shipped template.
    assert "api_key_env" in text
    assert "api_key =" not in text
    assert '"sk-' not in text
    assert "https://" not in text
    assert "http://" not in text or "localhost" in text
