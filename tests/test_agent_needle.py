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
import sys
from pathlib import Path

import pytest

from nvsh.agent import needle as needle_mod
from nvsh.agent import registry
from nvsh.agent.base import AgentContext, AgentRequest, EventKind, RequestKind
from nvsh.agent.needle import NEEDLE_CAPABILITIES, NeedleAgent
from nvsh.cli import main as cli_main
from nvsh.cli._errors import EXIT_USER_ERROR
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
