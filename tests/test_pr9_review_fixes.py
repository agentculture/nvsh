"""Regression tests for the PR #9 review findings (Qodo threads 2, 7, 8-11, 13, 14, 19).

Each test names the finding it pins so a future refactor that reopens the
gap fails loudly instead of quietly regressing an adapter's approval story.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvsh import client as client_mod
from nvsh.agent import registry
from nvsh.agent.agy import AgyAgent
from nvsh.agent.base import Target
from nvsh.agent.claude import ClaudeAgent
from nvsh.agent.codex import CodexAgent
from nvsh.agent.pi import PiAgent
from nvsh.config import DEFAULT_ALIAS, Config, ConfigError, load

# ---------------------------------------------------------------------------
# 9/10/11 -- extra_args can never smuggle an approval bypass into a harness
# ---------------------------------------------------------------------------

_BYPASS = [
    ["--dangerously-skip-permissions"],
    ["--full-auto"],
    ["--yolo"],
    ["--trust-all-tools"],
    ["--permission-mode", "bypassPermissions"],
]


@pytest.mark.parametrize("extra", _BYPASS, ids=lambda e: " ".join(e))
def test_claude_refuses_bypass_extra_args(extra: list[str]) -> None:
    with pytest.raises(ValueError, match="bypass"):
        ClaudeAgent({}, extra_args=extra)


@pytest.mark.parametrize("extra", _BYPASS + [["-c", "approval_policy=never"]], ids=str)
def test_codex_refuses_bypass_extra_args(extra: list[str]) -> None:
    with pytest.raises(ValueError, match="bypass"):
        CodexAgent({}, extra_args=extra)


@pytest.mark.parametrize("extra", _BYPASS, ids=lambda e: " ".join(e))
def test_agy_refuses_bypass_extra_args(extra: list[str]) -> None:
    with pytest.raises(ValueError, match="bypass"):
        AgyAgent(extra_args=extra)


def test_harmless_extra_args_are_still_accepted() -> None:
    assert ClaudeAgent({}, extra_args=["--verbose"])._extra_args == ["--verbose"]
    assert CodexAgent({}, extra_args=["--search"])._extra_args == ["--search"]
    assert AgyAgent(extra_args=["--quiet"])._extra_args == ["--quiet"]


# ---------------------------------------------------------------------------
# 7 -- codex answers the shared responder's respond_ui
# ---------------------------------------------------------------------------


class _Sent:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def __call__(self, obj: dict) -> bool:
        self.messages.append(obj)
        return True


def test_codex_respond_ui_allows_and_denies_a_pending_approval() -> None:
    agent = CodexAgent({})
    sent = _Sent()
    agent._send = sent  # type: ignore[method-assign]
    for request_id, fields, expected in (
        (7, {"confirmed": True}, "accept"),
        (8, {"value": "allow"}, "accept"),
        (9, {"cancelled": True}, "decline"),
        (10, {"value": "deny"}, "decline"),
    ):
        agent._pending_approvals[request_id] = "item/commandExecution/requestApproval"
        agent.respond_ui(request_id, **fields)
        assert sent.messages[-1] == {"id": request_id, "result": {"decision": expected}}
    assert not agent._pending_approvals


def test_codex_answers_an_unknown_server_request_instead_of_leaving_it_pending() -> None:
    agent = CodexAgent({})
    sent = _Sent()
    agent._send = sent  # type: ignore[method-assign]
    event = agent._map({"id": 42, "method": "thread/somethingNew", "params": {}})
    assert event is not None
    assert event.text == "thread/somethingNew"
    assert sent.messages[-1]["id"] == 42
    assert sent.messages[-1]["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# 8 -- the registry forwards every [agents.pi] knob
# ---------------------------------------------------------------------------


def test_pi_factory_forwards_effort_extra_args_and_approval() -> None:
    cfg = Config()
    cfg.agents["pi"] = {
        "provider": "nemotron",
        "model": "associate",
        "effort": "high",
        "extra_args": ["--no-color"],
        "approval": "nvsh",
    }
    agent = registry.ADAPTERS["pi"].factory(cfg)
    assert isinstance(agent, PiAgent)
    argv = agent.build_argv()
    assert argv[argv.index("--thinking") + 1] == "high"
    assert "--no-color" in argv


# ---------------------------------------------------------------------------
# 13 -- alias names must round-trip through save()
# ---------------------------------------------------------------------------


def test_alias_names_that_are_not_bare_toml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[aliases]\n"my alias" = "pi"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="bare key"):
        load(path)


# ---------------------------------------------------------------------------
# 19 -- @default is the warm target, not a one-shot
# ---------------------------------------------------------------------------


def test_default_alias_target_is_not_one_shot() -> None:
    assert client_mod._is_one_shot(None) is False
    assert client_mod._is_one_shot(Target("pi", "associate", None, DEFAULT_ALIAS)) is False
    assert client_mod._is_one_shot(Target("claude", "sonnet", "medium", "reviewer")) is True
    assert client_mod._is_one_shot(Target("claude", "sonnet", "medium", None)) is True


# ---------------------------------------------------------------------------
# 2 -- setup keeps an explicit, usable default alias
# ---------------------------------------------------------------------------


def test_setup_keeps_a_rich_default_alias_whose_backend_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nvsh.cli import main

    xdg = tmp_path / "xdg"
    (xdg / "nvsh").mkdir(parents=True)
    (xdg / "nvsh" / "config.toml").write_text(
        '[agent]\nprovider = "pi"\n\n[aliases]\ndefault = "claude/opus/high"\n', encoding="utf-8"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".bashrc").write_text("# rc\n", encoding="utf-8")
    monkeypatch.setattr(registry, "choose", lambda cfg, which=None, forced=None: ("pi", "test"))
    monkeypatch.setattr(registry, "installed", lambda name, which=None, config=None: True)
    rc = main(["setup", "--json", "--no-install"])
    assert rc == 0
    text = (xdg / "nvsh" / "config.toml").read_text(encoding="utf-8")
    assert 'default = "claude/opus/high"' in text


# ---------------------------------------------------------------------------
# 14 -- doctor probes the default alias's backend
# ---------------------------------------------------------------------------


def test_doctor_probes_the_default_alias_backend_not_the_legacy_provider() -> None:
    from nvsh import doctor_checks

    cfg = Config(agent_provider="pi")
    cfg.aliases[DEFAULT_ALIAS] = "codex"
    calls: list[list[str]] = []

    def run(argv, timeout):
        calls.append(list(argv))
        return 0, "codex-cli 0.147.0\n", ""

    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/codex", run=run)
    assert calls
    assert calls[0][0] == "codex"
    assert check["passed"] is True, json.dumps(check)
