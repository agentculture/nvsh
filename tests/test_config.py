"""Tests for nvsh.config — the stdlib tomllib config loader.

Acceptance criterion covered (task t5):
config loads $XDG_CONFIG_HOME/nvsh/config.toml with tomllib, defaults to
[agent] provider='pi' and [agents.pi] provider='nemotron' model='associate',
sessions.max=1, and never reads or stores API keys; missing file yields
defaults.
"""

from __future__ import annotations

import pytest

from nvsh.config import Config, ConfigError, default_toml, load, save, set_provider


@pytest.fixture
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


# --- save / set_provider (task t10) ---------------------------------------


def test_save_then_load_round_trips(xdg_home):
    cfg = Config()
    cfg.agent_provider = "openai-compat"
    save(cfg)
    reloaded = load()
    assert reloaded.agent_provider == "openai-compat"
    assert reloaded.agents["pi"]["provider"] == "nemotron"


def test_set_provider_writes_and_returns_updated_config(xdg_home):
    cfg = set_provider("claude")
    assert cfg.agent_provider == "claude"
    reloaded = load()
    assert reloaded.agent_provider == "claude"


def test_set_provider_preserves_existing_tables(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agent]\nprovider = "pi"\n\n[sessions]\nmax = 5\n\n'
        '[agents.openai-compat]\nbase_url = "http://localhost:8000/v1"\n'
        'api_key_env = "NVSH_API_KEY"\n',
        encoding="utf-8",
    )
    set_provider("openai-compat")
    reloaded = load()
    assert reloaded.agent_provider == "openai-compat"
    assert reloaded.sessions_max == 5
    assert reloaded.agents["openai-compat"]["base_url"] == "http://localhost:8000/v1"


def test_save_never_writes_a_literal_api_key(xdg_home):
    cfg = Config()
    cfg.agents["openai-compat"] = {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "NVSH_API_KEY",
    }
    save(cfg)
    from nvsh.config import _default_path

    text = _default_path().read_text(encoding="utf-8")
    assert "api_key_env" in text
    assert '"api_key"' not in text


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


# --- api_key_file: a bearer source that survives a real interactive shell (d10)


def test_api_key_file_is_a_valid_backend_key(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agents.openai-compat]\napi_key_file = "$XDG_CONFIG_HOME/nvsh/api_key"\n',
        encoding="utf-8",
    )
    cfg = load()
    assert cfg.agents["openai-compat"]["api_key_file"] == "$XDG_CONFIG_HOME/nvsh/api_key"


def test_literal_api_key_still_rejected_alongside_api_key_file(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agents.openai-compat]\napi_key_file = "/tmp/k"\n'
        'api_key = "example-placeholder-value-here"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "api_key" in str(exc.value)


def _write_key(path, value: str = "file-bearer-value", mode: int = 0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def test_resolve_bearer_prefers_a_set_env_var_over_the_file(tmp_path):
    from nvsh.config import resolve_bearer

    key_file = _write_key(tmp_path / "api_key")
    env = {"NVSH_API_KEY": "env-bearer-value", "HOME": str(tmp_path)}
    got = resolve_bearer({"api_key_env": "NVSH_API_KEY", "api_key_file": str(key_file)}, env=env)
    assert got.bearer == "env-bearer-value"
    assert got.source == "$NVSH_API_KEY"


def test_resolve_bearer_falls_through_an_empty_env_var_to_the_file(tmp_path):
    from nvsh.config import KEY_SOURCE_FILE, resolve_bearer

    key_file = _write_key(tmp_path / "api_key")
    env = {"NVSH_API_KEY": "", "HOME": str(tmp_path)}
    got = resolve_bearer({"api_key_env": "NVSH_API_KEY", "api_key_file": str(key_file)}, env=env)
    assert got.bearer == "file-bearer-value"
    assert got.source == KEY_SOURCE_FILE
    assert got.diagnostic is None


def test_resolve_bearer_expands_tilde_and_xdg_config_home(tmp_path):
    from nvsh.config import resolve_bearer

    _write_key(tmp_path / "conf" / "nvsh" / "api_key", "xdg-bearer")
    env = {"XDG_CONFIG_HOME": str(tmp_path / "conf"), "HOME": str(tmp_path)}
    got = resolve_bearer({"api_key_file": "$XDG_CONFIG_HOME/nvsh/api_key"}, env=env)
    assert got.bearer == "xdg-bearer"

    _write_key(tmp_path / "keys" / "api_key", "home-bearer")
    got = resolve_bearer({"api_key_file": "~/keys/api_key"}, env=env)
    assert got.bearer == "home-bearer"


def test_resolve_bearer_uses_the_default_key_file_when_nothing_is_configured(tmp_path):
    from nvsh.config import KEY_SOURCE_DEFAULT_FILE, default_key_file, resolve_bearer

    env = {"XDG_CONFIG_HOME": str(tmp_path / "conf"), "HOME": str(tmp_path)}
    _write_key(default_key_file(env), "default-file-bearer")
    got = resolve_bearer({}, env=env)
    assert got.bearer == "default-file-bearer"
    assert got.source == KEY_SOURCE_DEFAULT_FILE


def test_resolve_bearer_default_key_file_falls_back_to_home_config(tmp_path):
    from nvsh.config import default_key_file

    env = {"HOME": str(tmp_path)}
    assert default_key_file(env) == tmp_path / ".config" / "nvsh" / "api_key"


def test_resolve_bearer_is_empty_when_no_source_exists(tmp_path):
    from nvsh.config import resolve_bearer

    env = {"XDG_CONFIG_HOME": str(tmp_path / "conf"), "HOME": str(tmp_path)}
    got = resolve_bearer({"api_key_env": "NVSH_UNSET_KEY"}, env=env)
    assert got.bearer is None
    assert got.source is None
    assert got.diagnostic is None


def test_resolve_bearer_refuses_a_group_or_world_readable_key_file(tmp_path):
    from nvsh.config import resolve_bearer

    key_file = _write_key(tmp_path / "api_key", "too-open-bearer", mode=0o640)
    env = {"HOME": str(tmp_path)}
    got = resolve_bearer({"api_key_file": str(key_file)}, env=env)
    assert got.bearer is None
    assert got.diagnostic is not None
    # The mode is reported; the key never is, and neither is its directory.
    assert "0640" in got.diagnostic
    assert "too-open-bearer" not in got.diagnostic
    assert str(tmp_path) not in got.diagnostic
    assert "api_key" in got.diagnostic


def test_resolve_bearer_reports_a_missing_configured_key_file(tmp_path):
    from nvsh.config import resolve_bearer

    env = {"HOME": str(tmp_path)}
    got = resolve_bearer({"api_key_file": str(tmp_path / "nope" / "api_key")}, env=env)
    assert got.bearer is None
    assert got.diagnostic is not None
    assert "api_key" in got.diagnostic


def test_resolve_bearer_strips_surrounding_whitespace(tmp_path):
    from nvsh.config import resolve_bearer

    key_file = tmp_path / "api_key"
    key_file.write_text("  padded-bearer \n\n", encoding="utf-8")
    key_file.chmod(0o600)
    got = resolve_bearer({"api_key_file": str(key_file)}, env={"HOME": str(tmp_path)})
    assert got.bearer == "padded-bearer"


def test_default_toml_documents_the_key_file():
    text = default_toml()
    assert "api_key_file" in text


def test_config_example_documents_the_key_file():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "docs" / "config.example.toml"
    if not example.is_file():  # pragma: no cover - wheel install, no docs tree
        pytest.skip("docs/config.example.toml not present")
    text = example.read_text(encoding="utf-8")
    assert "api_key_file" in text
    assert "0600" in text
