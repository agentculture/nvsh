"""Tests for nvsh.agent.registry — the harness chooser (task t10).

Acceptance criteria covered:
- 'nvsh agent list' reports pi, qwen, claude, codex, openai-compat with
  installed status derived from PATH.
- choose() picks the configured provider when installed, else falls back to
  openai-compat (always "available", no binary needed) with a reason string
  naming why (missing binary, missing node).
- install_offer() only returns the npm command when npm exists and the user
  answered y; it never runs anything itself.
- no_harness_message() offers 'install pi' and 'choose another harness'.
"""

from __future__ import annotations

from nvsh.agent import registry
from nvsh.config import Config


def _which_all_missing(_name: str) -> str | None:
    return None


def _which_factory(present: set[str]):
    def _which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return _which


def test_adapters_registry_has_all_five_names():
    assert set(registry.ADAPTERS) == {"pi", "qwen", "claude", "codex", "openai-compat"}


def test_adapter_spec_shape():
    spec = registry.ADAPTERS["pi"]
    assert spec.name == "pi"
    assert spec.binary == "pi"
    assert callable(spec.factory)
    assert isinstance(spec.description, str) and spec.description
    assert spec.needs_node is True


def test_openai_compat_has_no_binary_and_no_node_requirement():
    spec = registry.ADAPTERS["openai-compat"]
    assert spec.binary is None
    assert spec.needs_node is False


def test_installed_true_when_binary_on_path():
    which = _which_factory({"claude"})
    assert registry.installed("claude", which=which) is True


def test_installed_false_when_binary_missing():
    assert registry.installed("claude", which=_which_all_missing) is False


def test_installed_openai_compat_always_true():
    assert registry.installed("openai-compat", which=_which_all_missing) is True


def test_available_adapters_reports_all_five_with_installed_status():
    which = _which_factory({"pi", "claude"})
    rows = registry.available_adapters(which=which)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {"pi", "qwen", "claude", "codex", "openai-compat"}
    assert by_name["pi"]["installed"] is True
    assert by_name["claude"]["installed"] is True
    assert by_name["qwen"]["installed"] is False
    assert by_name["codex"]["installed"] is False
    assert by_name["openai-compat"]["installed"] is True
    for row in rows:
        assert "binary" in row
        assert "description" in row


def test_choose_uses_configured_provider_when_installed():
    cfg = Config()
    cfg.agent_provider = "claude"
    name, reason = registry.choose(cfg, which=_which_factory({"claude"}))
    assert name == "claude"
    assert "claude" in reason


def test_choose_falls_back_to_openai_compat_when_not_installed():
    cfg = Config()
    cfg.agent_provider = "pi"
    cfg.agents["openai-compat"] = {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "NVSH_API_KEY",
    }
    name, reason = registry.choose(cfg, which=_which_all_missing)
    assert name == "openai-compat"
    assert "node" in reason
    assert "http://localhost:8000/v1" in reason


def test_choose_reason_mentions_missing_binary_when_node_present():
    cfg = Config()
    cfg.agent_provider = "claude"
    name, reason = registry.choose(cfg, which=_which_factory({"node"}))
    assert name == "openai-compat"
    assert "claude" in reason


def test_choose_unknown_configured_provider_falls_back():
    cfg = Config()
    cfg.agent_provider = "not-a-real-backend"
    name, reason = registry.choose(cfg, which=_which_all_missing)
    assert name == "openai-compat"
    assert "not-a-real-backend" in reason


def test_install_offer_returns_none_without_npm():
    result = registry.install_offer("pi", which=_which_all_missing, prompt=lambda _msg: "y")
    assert result is None


def test_install_offer_returns_none_when_user_declines():
    result = registry.install_offer("pi", which=_which_factory({"npm"}), prompt=lambda _msg: "n")
    assert result is None


def test_install_offer_returns_command_on_yes_with_npm_present():
    result = registry.install_offer("pi", which=_which_factory({"npm"}), prompt=lambda _msg: "y")
    assert result == "npm install -g @earendil-works/pi-coding-agent"


def test_install_offer_never_executes_anything(monkeypatch):
    # install_offer has no subprocess dependency at all -- it only ever
    # returns a command string, never runs one. Assert that directly.
    import inspect

    source = inspect.getsource(registry.install_offer)
    assert "subprocess" not in source
    assert "os.system" not in source
    result = registry.install_offer("pi", which=_which_factory({"npm"}), prompt=lambda _msg: "y")
    assert isinstance(result, str)


def test_install_offer_only_for_pi():
    assert (
        registry.install_offer("qwen", which=_which_factory({"npm"}), prompt=lambda _m: "y") is None
    )


def test_choose_orin_without_node_selects_openai_compat_with_reason():
    # Orin has no pi, no node, no npm on PATH -- 'nvsh setup' (t21) calls this
    # exact function to decide, and must print why.
    cfg = Config()
    cfg.agent_provider = "pi"
    cfg.agents["openai-compat"] = {"base_url": "http://192.168.1.138:8000/v1"}
    name, reason = registry.choose(cfg, which=_which_all_missing)
    assert name == "openai-compat"
    assert "node" in reason


def test_no_harness_message_offers_both_options():
    message = registry.no_harness_message(which=_which_all_missing)
    assert "install pi" in message
    assert "choose another harness" in message
    assert "nvsh agent use" in message
