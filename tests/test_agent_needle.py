"""The ``needle`` adapter (task t14): the explicit, Tier-1-only ``@needle``.

Acceptance criteria covered (t14, spec targets c14/h13):

1. ``NeedleAgent`` is registered as ``registry.ADAPTERS['needle']``, its
   ``__init__`` is cheap (spawns nothing), and it can propose a command --
   or explain why it has none -- driven end to end through
   ``tests/fakes/needle_worker``, never through ``cactus-needle``.
2. ``needle`` is in ``registry.PROBE_EXCLUDED`` (``nvsh setup`` never offers
   it as a default) and ``nvsh agent use needle`` is refused the same way
   ``nvsh agent use demo`` is.
3. ``nvsh agent list --json`` reports it with ``local_model=True``,
   ``approval='nvsh'``, ``unmediated_file_access=False``.

Nothing here needs ``cactus-needle``: every turn is driven against the fake
worker ``tests/fakes/needle_worker`` (the same fixture
``tests/test_tier_needle.py`` uses), with a fixed ``Platform`` and an
injected records path so nothing touches real XDG state.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - monkeypatched in a test, never run
import sys
import types
from pathlib import Path

import pytest

from nvsh.agent import needle as needle_mod
from nvsh.agent import registry
from nvsh.agent.base import AgentContext, AgentRequest, EventKind, RequestKind
from nvsh.agent.needle import NEEDLE_CAPABILITIES, NeedleAgent
from nvsh.cli import main as cli_main
from nvsh.cli._errors import EXIT_USER_ERROR, CliError
from nvsh.config import Config
from nvsh.platform._model import Platform

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "needle_worker"

#: A call the operation table accepts and that renders deterministically
#: (no groundable args, so grounding never calls a runner) -- see
#: ``nvsh/ops/table.py``/``nvsh/ops/render.py``'s ``disk_stats``.
GOOD_CALL = {"name": "disk_stats", "arguments": {}}

_PLATFORM = Platform(kind="jetson")


def _agent(tmp_path: Path, script: list, **kwargs) -> NeedleAgent:
    """A NeedleAgent wired to the fake worker, with the given per-launch script."""
    pid_file = tmp_path / "pids"
    env = dict(os.environ)
    env["NVSH_TEST_NEEDLE_PIDS"] = str(pid_file)
    env["NVSH_TEST_NEEDLE_SCRIPT"] = json.dumps(script)
    build_kwargs: dict = {
        "worker_argv": [sys.executable, str(FAKE_WORKER)],
        "tier_env": env,
        "records_path": tmp_path / "tiers.jsonl",
        "availability": lambda: None,
        "platform": _PLATFORM,
    }
    build_kwargs.update(kwargs)
    agent = NeedleAgent({}, **build_kwargs)
    agent.pid_file = pid_file  # type: ignore[attr-defined]
    return agent


def _pids(agent: NeedleAgent) -> list[int]:
    path: Path = agent.pid_file  # type: ignore[attr-defined]
    if not path.exists():
        return []
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _request(prompt: str = "show disk usage", kind: RequestKind = RequestKind.EXPLICIT):
    return AgentRequest(kind=kind, prompt=prompt)


def _context() -> AgentContext:
    return AgentContext(platform="jetson")


# ---------------------------------------------------------------------------
# registration (AC1/AC2/AC3)
# ---------------------------------------------------------------------------


def test_needle_is_registered():
    assert "needle" in registry.ADAPTERS
    spec = registry.ADAPTERS["needle"]
    assert spec.path == "inproc"
    assert spec.path in registry.PATH_VALUES
    assert spec.hosted is False
    assert spec.binary is None


def test_needle_is_probe_excluded():
    assert "needle" in registry.PROBE_EXCLUDED
    rows = registry.probe(lambda _name: "/usr/bin/x")
    assert "needle" not in {row["name"] for row in rows}


def test_agent_use_needle_is_refused_like_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    code = cli_main(["agent", "use", "needle", "--json"])
    assert code == EXIT_USER_ERROR


def test_agent_use_needle_refusal_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cli_main(["agent", "use", "needle"])
    err = capsys.readouterr().err
    assert "needle" in err
    assert "hint:" in err


def test_agent_list_json_reports_needle_capabilities(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    code = cli_main(["agent", "list", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in payload["adapters"]}
    row = by_name["needle"]
    assert row["capabilities"]["local_model"] is True
    assert row["capabilities"]["approval"] == "nvsh"
    assert row["capabilities"]["unmediated_file_access"] is False
    assert row["path"] == "inproc"


# ---------------------------------------------------------------------------
# NeedleAgent: cheap __init__, capabilities
# ---------------------------------------------------------------------------


def test_init_spawns_nothing(tmp_path):
    agent = _agent(tmp_path, [[{}]])
    assert _pids(agent) == []  # constructing never starts the worker


def test_factory_from_config_is_cheap():
    """registry.ADAPTERS['needle'].factory(Config()) must not raise or spawn."""
    agent = registry.ADAPTERS["needle"].factory(Config())
    assert isinstance(agent, NeedleAgent)


def test_factory_from_config_builds_no_router(monkeypatch):
    """The factory must not reach :meth:`NeedleAgent.start`.

    ``isinstance`` alone (the test above) would pass for a constructor that
    built the router or started the worker (Qodo #3, PR review), so this
    makes both fatal: ``start`` is what imports ``nvsh.tiers`` and builds
    the tier, and ``Popen`` is what would spawn its child.
    """

    def _boom(*args, **kwargs):
        raise AssertionError(f"the needle factory started something: {args!r}")

    monkeypatch.setattr(NeedleAgent, "start", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    agent = registry.ADAPTERS["needle"].factory(Config())
    assert isinstance(agent, NeedleAgent)


# ---------------------------------------------------------------------------
# needle is never persisted as the default (AC2, Qodo #1 of the PR review)
# ---------------------------------------------------------------------------

#: An Ubuntu ``.bashrc`` down to its interactive guard -- the file ``nvsh
#: setup`` would edit if a refusal did not come first.
UBUNTU_RC = """\
# ~/.bashrc

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac
"""


@pytest.fixture
def setup_home(tmp_path, monkeypatch):
    """A throwaway HOME/XDG with an rc file, for ``nvsh setup`` refusals."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text(UBUNTU_RC, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("NVSH_HOOK_VERSION", raising=False)
    return types.SimpleNamespace(home=home, rc=home / ".bashrc", config=tmp_path / "xdg-config")


def _alias(setup_home, body: str) -> None:
    nvsh_dir = setup_home.config / "nvsh"
    nvsh_dir.mkdir(parents=True, exist_ok=True)
    (nvsh_dir / "config.toml").write_text(body, encoding="utf-8")


def test_setup_agent_needle_is_refused(setup_home):
    assert cli_main(["setup", "--agent", "needle", "--json"]) == EXIT_USER_ERROR


def test_setup_agent_needle_refusal_names_tier_one(setup_home, capsys):
    cli_main(["setup", "--agent", "needle", "--json"])
    payload = json.loads(capsys.readouterr().err)
    assert "Tier 1" in payload["message"]


def test_setup_agent_at_needle_is_refused(setup_home):
    assert cli_main(["setup", "--agent", "@needle", "--json"]) == EXIT_USER_ERROR


def test_setup_agent_needle_never_touches_the_rc_file(setup_home):
    cli_main(["setup", "--agent", "needle", "--json"])
    assert setup_home.rc.read_text(encoding="utf-8") == UBUNTU_RC


def test_setup_refuses_an_alias_whose_target_is_needle(setup_home):
    _alias(setup_home, '[aliases]\ntier1 = "needle"\n')
    assert cli_main(["setup", "--agent", "tier1", "--json"]) == EXIT_USER_ERROR


def test_setup_refuses_a_model_qualified_needle_target(setup_home):
    assert cli_main(["setup", "--agent", "needle/needle3", "--json"]) == EXIT_USER_ERROR


def test_keep_existing_default_never_keeps_needle():
    """A ``[aliases].default = "needle"`` already on disk is re-probed, not kept."""
    from nvsh import config as nvsh_config
    from nvsh.cli._commands.setup import _keep_existing_default

    cfg = nvsh_config.Config(aliases={nvsh_config.DEFAULT_ALIAS: "needle"})
    assert _keep_existing_default(cfg, probe_rows=[]) is False


# ---------------------------------------------------------------------------
# a forced --agent needle with no flavor installed (Qodo #13 of the review)
# ---------------------------------------------------------------------------


def _uninstalled(monkeypatch) -> None:
    monkeypatch.setattr(registry, "installed", lambda name, which=None: False)


def _no_binary(_name: str) -> None:
    """A ``which`` that finds nothing."""
    return None


def test_forced_needle_without_the_flavor_names_the_flavor(monkeypatch):
    _uninstalled(monkeypatch)
    config = Config()
    with pytest.raises(CliError) as caught:
        registry.choose(config, _no_binary, forced="needle")
    assert "needle flavor" in caught.value.message


def test_forced_needle_without_the_flavor_never_says_none(monkeypatch):
    """``needle`` has no ``binary``; the old message formatted it anyway."""
    _uninstalled(monkeypatch)
    config = Config()
    with pytest.raises(CliError) as caught:
        registry.choose(config, _no_binary, forced="needle")
    assert "None" not in caught.value.message


def test_forced_needle_without_the_flavor_remediates_with_an_install(monkeypatch):
    _uninstalled(monkeypatch)
    config = Config()
    with pytest.raises(CliError) as caught:
        registry.choose(config, _no_binary, forced="needle")
    assert "pip install" in caught.value.remediation


def test_forced_claude_without_its_binary_still_names_the_binary(monkeypatch):
    """The binary-bearing branch's message is unchanged."""
    _uninstalled(monkeypatch)
    config = Config()
    with pytest.raises(CliError) as caught:
        registry.choose(config, _no_binary, forced="claude")
    assert caught.value.message.startswith("claude is not installed")


def test_capabilities_match_the_declared_needle_shape():
    agent = NeedleAgent()
    caps = agent.capabilities()
    assert caps is NEEDLE_CAPABILITIES
    assert caps.local_model is True
    assert caps.approval == "nvsh"
    assert caps.unmediated_file_access is False
    assert caps.tool_calling is True
    assert caps.steer is False
    assert caps.path == "inproc"


def test_steer_always_returns_false(tmp_path):
    agent = _agent(tmp_path, [[{}]])
    assert agent.steer("do this instead") is False


# ---------------------------------------------------------------------------
# one turn: propose, or explain -- never silently escalate
# ---------------------------------------------------------------------------


def test_a_good_decision_proposes_and_ends_with_done(tmp_path):
    agent = _agent(tmp_path, [[{"calls": [GOOD_CALL]}]])
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert EventKind.PROPOSAL in kinds
    assert kinds[-1] is EventKind.DONE
    proposal = next(e.proposal for e in events if e.kind is EventKind.PROPOSAL)
    assert proposal.command == "df -h"
    assert "needle" in proposal.rationale


def test_a_decline_explains_in_one_line_and_ends_with_done_never_escalating(tmp_path):
    """No calls at all is a NO_CALL decline: the adapter must explain why it
    had no answer and stop there -- it must never silently call another
    agent (there is no full-agent fallback wired into this adapter)."""
    agent = _agent(tmp_path, [[{"calls": []}]])
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert kinds[-1] is EventKind.DONE
    assert EventKind.PROPOSAL not in kinds
    status_texts = " ".join(e.text for e in events if e.kind is EventKind.STATUS)
    assert "needle" in status_texts
    assert "full agent" not in status_texts


def test_an_unavailable_tier_reports_its_own_status_text(tmp_path):
    agent = _agent(tmp_path, [[{}]], availability=lambda: "cactus-needle is not installed")
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert events[-1].kind is EventKind.DONE
    explanation = next(e.text for e in events if e.kind is EventKind.STATUS and e.text)
    assert "unavailable" in explanation
    assert "cactus-needle is not installed" in explanation


def test_a_failure_kind_request_never_reaches_tier_1(tmp_path):
    """A FAILURE request is not the instruction-shaped ask Needle answers
    (nvsh/tiers/router.py's Route._order): it must still end cleanly with an
    explanation and DONE, never hang waiting on a tier that is never asked."""
    agent = _agent(tmp_path, [[{"calls": [GOOD_CALL]}]])
    request = AgentRequest(kind=RequestKind.FAILURE, command="./run.sh", exit_code=1)
    try:
        events = list(agent.run(request, _context()))
    finally:
        agent.close()
    assert events[-1].kind is EventKind.DONE
    assert EventKind.PROPOSAL not in [e.kind for e in events]
    assert _pids(agent) == []  # the child never even started


# ---------------------------------------------------------------------------
# cancel / force_stop / close: kill the child
# ---------------------------------------------------------------------------


def test_close_kills_a_running_child(tmp_path):
    agent = _agent(tmp_path, [[{"calls": [GOOD_CALL]}]])
    list(agent.run(_request(), _context()))
    pids = _pids(agent)
    assert pids
    agent.close()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_force_stop_kills_a_running_child(tmp_path):
    agent = _agent(tmp_path, [[{"calls": [GOOD_CALL]}]])
    list(agent.run(_request(), _context()))
    pids = _pids(agent)
    assert pids
    agent.force_stop()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_a_fresh_child_answers_the_next_request_after_close(tmp_path):
    agent = _agent(tmp_path, [[{"calls": [GOOD_CALL]}], [{"calls": [GOOD_CALL]}]])
    try:
        list(agent.run(_request(), _context()))
        agent.close()
        second = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert second[-1].kind is EventKind.DONE
    proposals = [e for e in second if e.kind is EventKind.PROPOSAL]
    assert proposals, "the second turn, against a freshly-spawned child, must still propose"


def test_an_explicit_needle_request_never_claims_to_ask_the_full_agent():
    from nvsh.tiers import router

    assert needle_mod._FULL_AGENT == router.AGENT
