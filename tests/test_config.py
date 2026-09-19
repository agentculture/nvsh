"""Tests for nvsh.config — the stdlib tomllib config loader.

Acceptance criterion covered (task t5):
config loads $XDG_CONFIG_HOME/nvsh/config.toml with tomllib, defaults to
[agent] provider='pi' and [agents.pi] provider='nemotron' model='associate',
sessions.max=1, and never reads or stores API keys; missing file yields
defaults.
"""

from __future__ import annotations

import pytest

import nvsh.config
from nvsh.config import Config, ConfigError, _dump_toml, default_toml, load, save, set_provider


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


def test_no_yaml_import_anywhere():
    import pathlib

    text = pathlib.Path(nvsh.config.__file__).read_text(encoding="utf-8")
    assert "yaml" not in text


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


# --- per-agent effort / extra_args / approval (task t5, c10) ---------------


def test_agent_effort_extra_args_approval_accepted(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [agents.claude]
        provider = "claude"
        model = "sonnet"
        effort = "high"
        extra_args = ["--foo", "--bar=baz"]
        approval = "harness"
        """,
        encoding="utf-8",
    )
    cfg = load()
    assert cfg.agents["claude"]["effort"] == "high"
    assert cfg.agents["claude"]["extra_args"] == ["--foo", "--bar=baz"]
    assert cfg.agents["claude"]["approval"] == "harness"


def test_agent_effort_is_opaque_string_never_validated(xdg_home):
    # decision c24: effort (and model) are opaque strings, never checked
    # against an enum — any non-empty string is accepted.
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agents.claude]\neffort = "whatever-the-backend-calls-it"\n',
        encoding="utf-8",
    )
    cfg = load()
    assert cfg.agents["claude"]["effort"] == "whatever-the-backend-calls-it"


def test_agent_approval_rejects_invalid_value(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agents.claude]\napproval = "sometimes"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "approval" in str(exc.value)


def test_agent_extra_args_rejects_non_list(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agents.claude]\nextra_args = "--foo"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "extra_args" in str(exc.value)


def test_agent_extra_args_rejects_non_string_items(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        "[agents.claude]\nextra_args = [1, 2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "extra_args" in str(exc.value)


def test_unknown_agent_subkey_still_rejected_with_new_keys_present(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agents.claude]\neffort = "high"\nbogus_key = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "bogus_key" in str(exc.value)


# --- [aliases] flat table (task t5, c10/h8) ---------------------------------


def test_aliases_table_loads_flat_strings(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [aliases]
        default = "claude/sonnet/medium"
        reviewer = "claude/opus"
        local = "pi"
        """,
        encoding="utf-8",
    )
    cfg = load()
    assert cfg.aliases["default"] == "claude/sonnet/medium"
    assert cfg.aliases["reviewer"] == "claude/opus"
    assert cfg.aliases["local"] == "pi"


def test_aliases_value_must_be_string(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        "[aliases]\ndefault = 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load()


def test_aliases_value_rejects_more_than_three_segments(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[aliases]\ndefault = "claude/sonnet/medium/extra"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load()


def test_aliases_value_rejects_empty_segment(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[aliases]\ndefault = "claude//medium"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load()


def test_aliases_value_rejects_empty_string(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[aliases]\ndefault = ""\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load()


def test_aliases_added_to_valid_top_level_keys(xdg_home):
    # [aliases] itself must not be rejected as an unknown top-level table.
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text('[aliases]\ndefault = "pi"\n', encoding="utf-8")
    cfg = load()
    assert cfg.aliases["default"] == "pi"


def test_aliases_round_trip_through_save_and_load(xdg_home):
    cfg = Config()
    cfg.aliases = {"default": "claude/sonnet/medium", "reviewer": "claude/opus"}
    save(cfg)
    reloaded = load()
    assert reloaded.aliases == {"default": "claude/sonnet/medium", "reviewer": "claude/opus"}


# --- Config.resolve_target (task t5, c10/h8) --------------------------------


def test_resolve_target_uses_defined_alias(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [aliases]
        reviewer = "claude/opus/high"
        """,
        encoding="utf-8",
    )
    cfg = load()
    backend, model, effort, alias = cfg.resolve_target("reviewer")
    assert (backend, model, effort, alias) == ("claude", "opus", "high", True)


def test_resolve_target_alias_without_model_falls_back_to_agents_model(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [agents.pi]
        provider = "nemotron"
        model = "associate"

        [aliases]
        local = "pi"
        """,
        encoding="utf-8",
    )
    cfg = load()
    backend, model, effort, alias = cfg.resolve_target("local")
    assert backend == "pi"
    assert model == "associate"
    assert effort is None
    assert alias is True


def test_resolve_target_default_from_aliases(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [aliases]
        default = "claude/sonnet/medium"
        """,
        encoding="utf-8",
    )
    cfg = load()
    backend, model, effort, alias = cfg.resolve_target("default")
    assert (backend, model, effort, alias) == ("claude", "sonnet", "medium", True)


def test_resolve_target_default_falls_back_to_legacy_agent_provider(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [agent]
        provider = "pi"

        [agents.pi]
        provider = "nemotron"
        model = "associate"
        """,
        encoding="utf-8",
    )
    cfg = load()
    backend, model, effort, alias = cfg.resolve_target("default")
    assert backend == "pi"
    assert model == "associate"
    assert effort is None
    assert alias is True


def test_resolve_target_default_falls_back_with_no_config_at_all(xdg_home):
    cfg = load()
    backend, model, effort, alias = cfg.resolve_target("default")
    assert backend == "pi"
    assert model == "associate"
    assert effort is None
    assert alias is True


def test_resolve_target_literal_spec_not_registered_as_alias(xdg_home):
    cfg = load()
    backend, model, effort, alias = cfg.resolve_target("claude/sonnet/medium")
    assert (backend, model, effort, alias) == ("claude", "sonnet", "medium", False)


def test_resolve_target_literal_spec_with_at_prefix(xdg_home):
    cfg = load()
    backend, model, effort, alias = cfg.resolve_target("@claude/sonnet/medium")
    assert (backend, model, effort, alias) == ("claude", "sonnet", "medium", False)


def test_resolve_target_unknown_bare_name_raises(xdg_home):
    cfg = load()
    with pytest.raises(ConfigError):
        cfg.resolve_target("nonexistent-alias")


# --- set_provider now also writes [aliases].default -------------------------


def test_set_provider_writes_aliases_default(xdg_home):
    cfg = set_provider("claude")
    assert cfg.aliases["default"] == "claude"
    reloaded = load()
    assert reloaded.aliases["default"] == "claude"
    backend, _model, _effort, alias = reloaded.resolve_target("default")
    assert backend == "claude"
    assert alias is True


# --- docs/config.example.toml documents aliases (task t5) -------------------


def test_config_example_documents_aliases():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "docs" / "config.example.toml"
    if not example.is_file():  # pragma: no cover - wheel install, no docs tree
        pytest.skip("docs/config.example.toml not present")
    text = example.read_text(encoding="utf-8")
    assert "[aliases]" in text
    assert "default" in text
    assert "reviewer" in text
    assert "local" in text


def test_default_toml_documents_aliases():
    text = default_toml()
    data_ = None
    import tomllib

    data_ = tomllib.loads(text)
    assert "aliases" in data_
    assert "default" in data_["aliases"]
    assert "reviewer" in data_["aliases"]
    assert "local" in data_["aliases"]


# --- [tiers] table (task t6, c14, c10, c33) ---------------------------------


def test_tiers_defaults_when_absent(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[agent]\nprovider = "pi"\n',
        encoding="utf-8",
    )
    cfg = load()
    assert cfg.tiers["enabled"] is False


def test_tiers_parses_all_keys(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [tiers]
        enabled = true
        needle_min_confidence = 0.85
        memory_floor_mb = 2048
        idle_unload_seconds = 600
        records_cap_mb = 16
        store_request_text = true

        [tiers.lfm]
        engine = "vllm"
        mode = "attach"
        base_url = "http://127.0.0.1:8000/v1"
        model = "llama-3.1-8b"
        """,
        encoding="utf-8",
    )
    cfg = load()
    assert cfg.tiers["enabled"] is True
    assert cfg.tiers["needle_min_confidence"] == 0.85
    assert cfg.tiers["memory_floor_mb"] == 2048
    assert cfg.tiers["idle_unload_seconds"] == 600
    assert cfg.tiers["records_cap_mb"] == 16
    assert cfg.tiers["store_request_text"] is True
    assert cfg.tiers["lfm"]["engine"] == "vllm"
    assert cfg.tiers["lfm"]["mode"] == "attach"
    assert cfg.tiers["lfm"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert cfg.tiers["lfm"]["model"] == "llama-3.1-8b"


def test_tiers_unknown_key_rejected(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[tiers]\nbogus_key = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "bogus_key" in str(exc.value)
    assert "tiers" in str(exc.value).lower()


def test_tiers_lfm_unknown_key_rejected(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        """
        [tiers.lfm]
        engine = "llama-server"
        bogus_field = 42
        """,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "bogus_field" in str(exc.value)


def test_tiers_type_errors(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    cases = [
        ('[tiers]\nenabled = "yes"\n', "[tiers] enabled must be true or false"),
        (
            "[tiers]\nmemory_floor_mb = -1\n",
            "[tiers] memory_floor_mb must be a non-negative integer",
        ),
        (
            "[tiers]\nmemory_floor_mb = true\n",
            "[tiers] memory_floor_mb must be a non-negative integer",
        ),
        (
            "[tiers]\nneedle_min_confidence = 2\n",
            "[tiers] needle_min_confidence must be between 0 and 1",
        ),
    ]
    for toml_snippet, error_fragment in cases:
        (cfg_dir / "config.toml").write_text(toml_snippet, encoding="utf-8")
        with pytest.raises(ConfigError) as exc:
            load()
        msg = str(exc.value)
        assert error_fragment in msg, f"{error_fragment!r} not in {msg!r} for {toml_snippet!r}"


def test_tiers_lfm_engine_and_mode_validated(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[tiers.lfm]\nengine = "unknown"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "engine" in str(exc.value)

    (cfg_dir / "config.toml").write_text(
        '[tiers.lfm]\nengine = "llama-server"\nmode = "unknown"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "mode" in str(exc.value)


def test_tiers_lfm_base_url_must_be_localhost(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        '[tiers.lfm]\nengine = "llama-server"\nbase_url = "http://example.com/v1"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as exc:
        load()
    assert "base_url" in str(exc.value)


def test_tiers_roundtrip_through_save(xdg_home):
    cfg_dir = xdg_home / "nvsh"
    cfg_dir.mkdir(parents=True)
    cfg = Config()
    cfg.tiers["enabled"] = True
    cfg.tiers["lfm"]["base_url"] = "http://127.0.0.1:8080/v1"
    save(cfg)
    reloaded = load()
    assert reloaded.tiers["enabled"] is True
    assert reloaded.tiers["lfm"]["base_url"] == "http://127.0.0.1:8080/v1"


def test_config_without_tiers_dumps_unchanged():
    """A config with no tiers table: _dump_toml output contains no '[tiers]'."""
    cfg = Config()
    cfg.agent_provider = "pi"
    text = _dump_toml(cfg)
    assert "[tiers]" not in text
