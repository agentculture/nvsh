"""Tests for the system brief and per-platform playbooks (deviations d19/d20).

The failure this locks down: with only the failure text and a ``Platform:``
block, the real associate model on the Spark answered ``whats memory levels
are now?`` by proposing ``which pi && pi --help`` — it investigated its own
harness because nothing told it who it was, what machine it was on, or what
to look at. The brief and the playbook are that missing background, and
they are *code*, not configuration, so every install has them.
"""

from __future__ import annotations

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.agent.playbooks import GENERIC, playbook_for
from nvsh.agent.prompt import build_full_prompt, build_prompt, build_system_prompt, platform_kind


def _context(kind: str = "dgx-spark") -> AgentContext:
    return AgentContext(
        platform=f"platform: {kind}\n  mem_available: 2 kB  [file: /proc/meminfo]",
        output="bash: what: command not found",
        cwd="/home/op",
    )


def _request() -> AgentRequest:
    return AgentRequest(
        kind=RequestKind.FAILURE, prompt="Diagnose this.", command="ls /nope", exit_code=2
    )


def test_platform_kind_parses_the_render_block():
    assert platform_kind(_context("jetson")) == "jetson"
    assert platform_kind(AgentContext()) == "generic"
    assert platform_kind(AgentContext(platform="platform: dgx_spark")) == "dgx-spark"


def test_system_brief_states_identity_and_rules():
    brief = build_system_prompt(_context())
    lowered = " ".join(brief.lower().split())  # the brief is hard-wrapped
    assert "nvsh" in lowered
    assert "one command at a time" in lowered
    assert "bash tool" in lowered
    assert "sudo" in lowered
    # never investigate the harness itself -- the d19 failure mode
    assert "pi" in lowered
    assert "harness" in lowered
    # d20: a plain-language question is a request to do the work
    assert "never answer a question with a list of commands" in lowered
    assert "one-line diagnosis" in lowered


def test_playbook_is_chosen_by_detected_kind():
    spark = build_system_prompt(_context("dgx-spark"))
    jetson = build_system_prompt(_context("jetson"))
    assert "GB10" in spark
    assert "tegrastats" not in spark
    assert "tegrastats" in jetson
    assert "nvpmodel" in jetson
    assert "/proc/meminfo" in spark
    assert "/proc/meminfo" in jetson


def test_generic_fallback_for_an_unknown_kind():
    unknown = build_system_prompt(_context("something-else"))
    assert playbook_for("something-else") == GENERIC
    assert "free -h" in unknown
    assert "journalctl" in unknown


def test_every_playbook_is_short():
    for kind in ("dgx-spark", "jetson", "rtx", "generic"):
        assert len(playbook_for(kind).splitlines()) <= 40, kind


def test_full_prompt_puts_the_facts_block_after_the_brief():
    request, context = _request(), _context()
    full = build_full_prompt(request, context)
    brief = build_system_prompt(context)
    facts = build_prompt(request, context)
    assert full.startswith(brief)
    assert full.endswith(facts)
    assert full.index(brief) < full.index("Command: ls /nope")


def test_system_brief_stays_under_the_size_bound():
    for kind in ("dgx-spark", "jetson", "rtx", "generic"):
        size = len(build_system_prompt(_context(kind)).encode("utf-8"))
        assert size <= 6144, (kind, size)
