"""Tests for nvsh.agent.registry — the harness chooser (tasks t10, t8).

Acceptance criteria covered:
- 'nvsh agent list' reports pi, qwen, qwen-p, claude, codex, agy, kiro,
  openai-compat, demo with installed status derived from PATH.
- Every AdapterSpec carries a 'path' (transport protocol) and 'hosted' flag.
- choose() picks the configured provider when installed, else falls back to
  openai-compat (always "available", no binary needed) with a reason string
  naming why (missing binary, missing node).
- choose(config, forced=...) resolves an alias/literal target/Target through
  Config.resolve_target, returns the forced adapter when installed, and
  raises a CliError naming the missing binary otherwise -- never falling
  back to openai-compat.
- The agy/kiro/qwen(acp) factories import their concrete adapter modules
  lazily, so this module imports cleanly before nvsh.agent.agy/acp exist.
- install_offer() only returns the npm command when npm exists and the user
  answered y; it never runs anything itself.
- no_harness_message() offers 'install pi' and 'choose another harness'.
"""

from __future__ import annotations

import dataclasses

import pytest

from nvsh.agent import registry
from nvsh.agent.base import Target
from nvsh.cli._errors import CliError
from nvsh.config import Config

_ALL_ADAPTER_NAMES = {
    "pi",
    "qwen",
    "qwen-p",
    "claude",
    "codex",
    "agy",
    "kiro",
    "openai-compat",
    "demo",
    "needle",
}


def _which_all_missing(_name: str) -> str | None:
    return None


def _which_factory(present: set[str]):
    def _which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return _which


def test_adapters_registry_has_all_ten_names():
    assert set(registry.ADAPTERS) == _ALL_ADAPTER_NAMES


def test_agy_and_kiro_adapter_specs():
    agy = registry.ADAPTERS["agy"]
    assert agy.binary == "agy"
    assert agy.path == "stream-json"
    assert agy.hosted is True

    kiro = registry.ADAPTERS["kiro"]
    assert kiro.binary == "kiro-cli"
    assert kiro.path == "acp"
    assert kiro.hosted is True


def test_qwen_switched_to_acp_path():
    qwen = registry.ADAPTERS["qwen"]
    assert qwen.path == "acp"
    assert qwen.hosted is False
    # the print-mode fallback stays reachable under its own name
    assert "qwen-p" in registry.ADAPTERS


def test_every_adapter_spec_carries_path_and_hosted():
    for name, spec in registry.ADAPTERS.items():
        assert isinstance(spec.path, str), name
        assert spec.path, name
        assert spec.path in registry.PATH_VALUES, (name, spec.path)
        assert isinstance(spec.hosted, bool), name


def test_hosted_flags_match_acceptance_criteria():
    hosted_true = {"agy", "claude", "codex", "kiro"}
    hosted_false = {"pi", "qwen", "qwen-p", "openai-compat"}
    for name in hosted_true:
        assert registry.ADAPTERS[name].hosted is True, name
    for name in hosted_false:
        assert registry.ADAPTERS[name].hosted is False, name


def test_path_values_match_acceptance_criteria():
    expected = {
        "pi": "rpc",
        "claude": "stream-json",
        "codex": "app-server",
        "qwen": "acp",
        "kiro": "acp",
        "agy": "stream-json",
        "openai-compat": "http",
    }
    for name, path in expected.items():
        assert registry.ADAPTERS[name].path == path, name


def test_agy_and_acp_modules_are_imported_lazily():
    # Importing the registry must not import the agy/acp adapter modules:
    # they are only pulled in by the factory that actually needs them.
    import subprocess
    import sys

    code = (
        "import sys, nvsh.agent.registry; "
        "print(sorted(m for m in sys.modules if m in "
        "('nvsh.agent.agy', 'nvsh.agent.acp')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_agy_and_acp_factories_build_the_right_adapters():
    from nvsh.agent.acp import AcpAgent
    from nvsh.agent.agy import AgyAgent

    assert isinstance(registry.ADAPTERS["agy"].factory(Config()), AgyAgent)
    kiro = registry.ADAPTERS["kiro"].factory(Config())
    assert isinstance(kiro, AcpAgent)
    assert kiro.capabilities().path == "acp"
    qwen = registry.ADAPTERS["qwen"].factory(Config())
    assert isinstance(qwen, AcpAgent)
    # decision c53: qwen is read-only (plan mode) unless approval = "harness"
    assert qwen.capabilities().tool_calling is False
    assert qwen.capabilities().approval == "nvsh"
    config = Config()
    config.agents["qwen"] = {"approval": "harness"}
    opted_in = registry.ADAPTERS["qwen"].factory(config)
    assert opted_in.capabilities().tool_calling is True
    assert opted_in.capabilities().approval == "harness"


def test_adapter_spec_shape():
    spec = registry.ADAPTERS["pi"]
    assert spec.name == "pi"
    assert spec.binary == "pi"
    assert callable(spec.factory)
    assert isinstance(spec.description, str)
    assert spec.description
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


def test_steer_capable_reads_capability_without_starting_the_adapter(monkeypatch):
    """steer_capable() must construct-and-read, never start(), the adapter --
    built the same way registry._tool_calling already reads tool_calling
    (AC3)."""
    from nvsh.agent.base import Capabilities
    from nvsh.agent.fake import FakeAgent

    class _StartRaises(FakeAgent):
        def start(self) -> None:  # pragma: no cover - must never be called
            raise AssertionError("steer_capable must not start the adapter")

    def _factory(_config: Config) -> _StartRaises:
        return _StartRaises([], capabilities=Capabilities(steer=True))

    fake_spec = registry.AdapterSpec(
        name="fake-steer",
        binary=None,
        factory=_factory,
        description="test-only fake",
        path="fixture",
        hosted=False,
    )
    monkeypatch.setitem(registry.ADAPTERS, "fake-steer", fake_spec)
    assert registry.steer_capable("fake-steer", Config()) is True


def test_steer_capable_false_by_default():
    assert registry.steer_capable("openai-compat", Config()) is False


def test_steer_capable_true_for_pi_and_codex():
    assert registry.steer_capable("pi", Config()) is True
    assert registry.steer_capable("codex", Config()) is True


def test_available_adapters_reports_all_ten_with_installed_status():
    which = _which_factory({"pi", "claude"})
    rows = registry.available_adapters(which=which)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == _ALL_ADAPTER_NAMES
    assert by_name["pi"]["installed"] is True
    assert by_name["claude"]["installed"] is True
    assert by_name["qwen"]["installed"] is False
    assert by_name["codex"]["installed"] is False
    assert by_name["agy"]["installed"] is False
    assert by_name["kiro"]["installed"] is False
    assert by_name["openai-compat"]["installed"] is True
    # needle's 'installed' never consults `which` at all -- see the
    # dedicated installed_check tests below.
    for row in rows:
        assert "binary" in row
        assert "description" in row


# -- needle: installed_check, not `which` -------------------------------------


def test_needle_spec_shape():
    spec = registry.ADAPTERS["needle"]
    assert spec.binary is None
    assert spec.path == "inproc"
    assert spec.hosted is False
    assert spec.installed_check is not None


def test_needle_installed_uses_installed_check_not_which():
    """`which` is passed but must never be consulted for needle -- a `which`
    that raises on any input proves that (never called with 'None', either --
    the bug installed() already guards against for binary=None adapters)."""

    def _which_raises(_name: str) -> str | None:
        raise AssertionError("needle.installed() must not call which()")

    assert registry.installed("needle", which=_which_raises) in (True, False)


def test_needle_installed_reflects_flavor_check(monkeypatch):
    monkeypatch.setitem(
        registry.ADAPTERS,
        "needle",
        dataclasses.replace(registry.ADAPTERS["needle"], installed_check=lambda: True),
    )
    assert registry.installed("needle", which=_which_all_missing) is True

    monkeypatch.setitem(
        registry.ADAPTERS,
        "needle",
        dataclasses.replace(registry.ADAPTERS["needle"], installed_check=lambda: False),
    )
    assert registry.installed("needle", which=_which_all_missing) is False


def test_needle_is_probe_excluded():
    assert "needle" in registry.PROBE_EXCLUDED
    rows = registry.probe(_which_all_missing)
    assert "needle" not in {row["name"] for row in rows}


def test_needle_factory_builds_needle_agent():
    from nvsh.agent.needle import NeedleAgent

    agent = registry.ADAPTERS["needle"].factory(Config())
    assert isinstance(agent, NeedleAgent)
    caps = agent.capabilities()
    assert caps.local_model is True
    assert caps.approval == "nvsh"
    assert caps.unmediated_file_access is False
    assert caps.path == "inproc"


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


def test_choose_with_no_customization_returns_pi():
    # A default Config() (agent_provider defaults to 'pi') with pi on PATH
    # still resolves to pi -- the unforced default behaviour is unchanged.
    name, _reason = registry.choose(Config(), which=_which_factory({"pi"}))
    assert name == "pi"


def test_choose_forced_alias_returns_adapter_when_installed():
    cfg = Config()
    cfg.aliases["fast"] = "claude/haiku"
    name, reason = registry.choose(cfg, which=_which_factory({"claude"}), forced="fast")
    assert name == "claude"
    assert "fast" in reason


def test_choose_forced_literal_target_returns_adapter_when_installed():
    cfg = Config()
    name, reason = registry.choose(cfg, which=_which_factory({"codex"}), forced="codex/o1")
    assert name == "codex"


def test_choose_forced_target_object_returns_adapter_when_installed():
    cfg = Config()
    target = Target(backend="codex", model="o1")
    name, _reason = registry.choose(cfg, which=_which_factory({"codex"}), forced=target)
    assert name == "codex"


def test_choose_forced_raises_cli_error_naming_binary_when_not_installed():
    cfg = Config()
    with pytest.raises(CliError) as excinfo:
        registry.choose(cfg, which=_which_all_missing, forced="claude/opus")
    assert "claude" in str(excinfo.value.message)
    # never silently falls back to openai-compat
    assert "openai-compat" not in str(excinfo.value.message)


def test_choose_forced_unknown_backend_raises_cli_error():
    cfg = Config()
    with pytest.raises(CliError):
        registry.choose(cfg, which=_which_all_missing, forced="not-a-real-backend/model")


def test_choose_forced_unresolvable_alias_raises_cli_error():
    cfg = Config()
    with pytest.raises(CliError):
        registry.choose(cfg, which=_which_all_missing, forced="no-such-alias")


def test_no_harness_message_offers_both_options():
    message = registry.no_harness_message()
    assert "install pi" in message
    assert "choose another harness" in message
    assert "nvsh agent use" in message


# -- probe() -----------------------------------------------------------------


def test_probe_excludes_openai_compat_even_when_everything_installed():
    which = _which_factory(set(registry.ADAPTERS) | {"kiro-cli", "node"})
    rows = registry.probe(which)
    assert "openai-compat" not in {row["name"] for row in rows}


def test_probe_row_shape():
    rows = registry.probe(_which_factory({"claude"}))
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"name", "hosted", "tool_calling"}
    assert row["name"] == "claude"
    assert row["hosted"] is True
    assert row["tool_calling"] is True


@pytest.mark.parametrize(
    "present,expected_names,expected_first_tool_calling",
    [
        (set(), [], None),
        ({"claude"}, ["claude"], True),
        ({"claude", "codex"}, ["claude", "codex"], True),
        # qwen and qwen-p share the 'qwen' binary: only the first adapter
        # registered for that binary (qwen, ahead of qwen-p in ADAPTERS
        # order) shows up -- qwen-p stays selectable, just not probed.
        ({"qwen", "claude"}, ["claude", "qwen"], True),
        ({"pi", "claude"}, ["pi", "claude"], True),
    ],
)
def test_probe_table_over_which_sets(present, expected_names, expected_first_tool_calling):
    rows = registry.probe(_which_factory(present))
    assert [row["name"] for row in rows] == expected_names
    if expected_names:
        assert rows[0]["tool_calling"] is expected_first_tool_calling


def test_probe_orders_tool_calling_first_then_adapters_order():
    # qwen (acp, tool_calling False by default) and claude/codex/pi
    # (tool_calling True) present: tool-calling adapters come first, in
    # ADAPTERS registration order (pi, claude, codex), then qwen. qwen-p
    # shares qwen's binary and is de-duplicated out of the probe table.
    rows = registry.probe(_which_factory({"pi", "qwen", "claude", "codex"}))
    assert [row["name"] for row in rows] == ["pi", "claude", "codex", "qwen"]


def test_probe_reflects_config_overrides_like_build_adapter_rows():
    # qwen's tool_calling flips to True when [agents.qwen] approval="harness"
    # -- the same underlying source build_adapter_rows uses.
    config = Config()
    config.agents["qwen"] = {"approval": "harness"}
    rows = registry.probe(_which_factory({"qwen"}), config=config)
    assert rows == [
        {"name": "qwen", "hosted": False, "tool_calling": True},
    ]


def test_probe_lists_one_row_per_binary():
    # A qwen-only machine sees exactly one row -- 'qwen' -- not both 'qwen'
    # and 'qwen-p' (Qodo #2, PR #12 review): they share one binary, and
    # qwen-p is a fallback meant for explicit selection, not a second row
    # setup would prompt between.
    rows = registry.probe(_which_factory({"qwen"}))
    assert [row["name"] for row in rows] == ["qwen"]


# -- choose() consulting probe() ----------------------------------------------


def test_choose_default_config_picks_installed_tool_calling_adapter():
    # default Config() has agent_provider == 'pi'; pi is not installed here,
    # but claude and codex are -- probe() picks the first tool-calling row.
    name, reason = registry.choose(Config(), which=_which_factory({"claude", "codex"}))
    assert name == "claude"
    assert "claude" in reason
    assert "codex" in reason
    assert "picked claude" in reason


def test_choose_which_empty_still_returns_openai_compat():
    name, _reason = registry.choose(Config(), which=_which_all_missing)
    assert name == "openai-compat"


@pytest.mark.parametrize(
    "present,expected",
    [
        (set(), "openai-compat"),
        ({"claude"}, "claude"),
        ({"claude", "codex"}, "claude"),
        ({"qwen", "claude"}, "claude"),
        ({"pi", "claude"}, "pi"),
    ],
)
def test_choose_table_over_which_sets(present, expected):
    cfg = Config()
    name, _reason = registry.choose(cfg, which=_which_factory(present))
    assert name == expected


def test_choose_configured_and_installed_provider_still_wins_over_probe():
    cfg = Config()
    cfg.agent_provider = "codex"
    name, reason = registry.choose(cfg, which=_which_factory({"codex", "claude"}))
    assert name == "codex"
    assert "configured" in reason


# ---------------------------------------------------------------------------
# forced target: a bare adapter name (post-plan fix, deviation d2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forced", ["claude", "@claude", "codex"])
def test_choose_forced_accepts_a_bare_adapter_name(forced):
    """``nvsh setup --agent claude`` is the documented on-ramp; a bare name
    resolves without an alias, exactly like ``@claude`` at the prompt."""
    cfg = Config()
    which = lambda name: "/usr/bin/x" if name in {"claude", "codex"} else None  # noqa: E731
    backend, reason = registry.choose(cfg, which, forced=forced)
    assert backend == forced.lstrip("@")
    assert "forced" in reason


def test_choose_forced_bare_name_prefers_an_alias_of_the_same_name():
    cfg = Config(aliases={"claude": "codex/gpt-5/high"})
    which = lambda name: "/usr/bin/x" if name in {"claude", "codex"} else None  # noqa: E731
    backend, _ = registry.choose(cfg, which, forced="claude")
    assert backend == "codex"


def test_choose_forced_bare_name_still_fails_when_not_installed():
    cfg = Config()
    with pytest.raises(CliError) as excinfo:
        registry.choose(cfg, lambda _n: None, forced="claude")
    assert "claude" in str(excinfo.value.message)


def test_choose_forced_qwen_p_still_selectable_despite_probe_dedup():
    # qwen-p is dropped from probe()'s table (it shares qwen's binary), but
    # remains fully reachable through an explicit forced target.
    cfg = Config()
    name, reason = registry.choose(cfg, which=_which_factory({"qwen"}), forced="qwen-p")
    assert name == "qwen-p"
    assert "forced" in reason
