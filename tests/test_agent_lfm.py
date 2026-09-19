"""The ``lfm`` adapter (task t19): the explicit, Tier-2-only ``@lfm``.

Acceptance criteria covered (t19, spec targets c14/h13/c7):

1. ``LfmAgent`` is registered as ``registry.ADAPTERS['lfm']`` (see
   ``tests/test_agent_registry.py``), joins the conformance suite through
   the generic per-adapter loops (``tests/test_agent_conformance.py``'s
   ``test_exactly_pi_and_codex_declare_steer_true``,
   ``tests/test_cli_agent.py``'s help/list checks -- both iterate
   ``registry.ADAPTERS`` directly, the same way ``needle`` joined in t14),
   is ``PROBE_EXCLUDED`` and refused as a persisted default
   (``tests/test_demo_default_refused.py``).
2. The router's Tier 2 slot uses the real ``LfmTier`` when
   ``[tiers.lfm] model`` is configured, and a FAILURE request reaches it
   directly (``tests/test_daemon_tiers.py``'s manager-level test).

This module covers what those don't: ``LfmAgent`` itself, driven against a
fake :class:`~nvsh.tiers.runtime.Runtime` and fake chat client (mirroring
``tests/test_tier_lfm.py``'s ``_Runtime``/``_FakeChat``) so nothing here
starts Docker or talks to a real model server -- a cheap constructor,
propose/explain/decline, the no-model-configured explanation, a FAILURE
request (unlike ``needle``, Tier 2 answers these too), and cancel/close.
"""

from __future__ import annotations

from nvsh.agent import registry
from nvsh.agent.base import AgentContext, AgentRequest, EventKind, RequestKind
from nvsh.agent.lfm import LFM_CAPABILITIES, LfmAgent
from nvsh.config import Config
from nvsh.platform._model import Platform
from nvsh.tiers.lfm import ESCALATE_TOOL, EXPLAIN_TOOL, PROPOSE_TOOL
from nvsh.tiers.runtime import RuntimeUnavailable
from nvsh.tiers.toolchat import ChatReply, ToolCall

_PLATFORM = Platform(kind="jetson")
_MODEL_CONFIG = {"lfm": {"model": "test-model"}}


class _Runtime:
    """A fake :class:`~nvsh.tiers.runtime.Runtime`, mirroring
    ``tests/test_tier_lfm.py``'s helper of the same name."""

    def __init__(self, error: str = "") -> None:
        self.error = error
        self.ensured = 0
        self.stopped = 0

    def ensure(self) -> str:
        self.ensured += 1
        if self.error:
            raise RuntimeUnavailable(self.error)
        return "http://127.0.0.1:65000"

    def stop(self) -> None:
        self.stopped += 1

    def status(self) -> str:
        return "fake runtime"


class _FakeChat:
    """Replays scripted :class:`ChatReply` objects, mirroring
    ``tests/test_tier_lfm.py``'s helper of the same name."""

    def __init__(self, replies: list[ChatReply]) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict]] = []
        self.stopped = 0

    def factory(self, base_url: str) -> "_FakeChat":
        del base_url
        return self

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        del tools
        self.seen.append(messages)
        if not self._replies:
            return ChatReply(text="", tool_calls=())
        return self._replies.pop(0)

    def stop(self) -> None:
        self.stopped += 1


def _tool_reply(name: str, **arguments: object) -> ChatReply:
    return ChatReply(text="", tool_calls=(ToolCall(name=name, arguments=dict(arguments)),))


def _low_memory_reader() -> str:
    return "MemAvailable:     100 kB\n"


def _agent(chat: _FakeChat, *, runtime: _Runtime | None = None, config=None, **kwargs) -> LfmAgent:
    return LfmAgent(
        config if config is not None else _MODEL_CONFIG,
        runtime=runtime if runtime is not None else _Runtime(),
        chat_factory=chat.factory,
        platform=_PLATFORM,
        **kwargs,
    )


def _request(prompt: str = "why is this machine slow", kind: RequestKind = RequestKind.EXPLICIT):
    return AgentRequest(kind=kind, prompt=prompt)


def _context() -> AgentContext:
    return AgentContext(platform="jetson")


# ---------------------------------------------------------------------------
# cheap __init__, capabilities
# ---------------------------------------------------------------------------


def test_init_never_raises_or_needs_a_runtime():
    assert isinstance(LfmAgent(), LfmAgent)


def test_factory_from_config_is_cheap():
    agent = registry.ADAPTERS["lfm"].factory(Config())
    assert isinstance(agent, LfmAgent)


def test_factory_from_config_builds_no_router(monkeypatch):
    """The factory must not reach :meth:`LfmAgent.start` -- ``isinstance``
    alone would pass for a constructor that built the router too."""

    def _boom(*args, **kwargs):
        raise AssertionError(f"the lfm factory started something: {args!r}")

    monkeypatch.setattr(LfmAgent, "start", _boom)
    agent = registry.ADAPTERS["lfm"].factory(Config())
    assert isinstance(agent, LfmAgent)


def test_capabilities_match_the_declared_lfm_shape():
    agent = LfmAgent()
    caps = agent.capabilities()
    assert caps is LFM_CAPABILITIES
    assert caps.local_model is True
    assert caps.approval == "nvsh"
    assert caps.unmediated_file_access is False
    assert caps.tool_calling is True
    assert caps.steer is False
    assert caps.path == "inproc"


def test_steer_always_returns_false():
    agent = _agent(_FakeChat([]))
    assert agent.steer("do this instead") is False


# ---------------------------------------------------------------------------
# no model configured: one status line, no router, never builds a broken tier
# ---------------------------------------------------------------------------


def test_no_model_configured_explains_and_ends_with_done():
    agent = LfmAgent({})
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert kinds == [EventKind.STATUS, EventKind.DONE]
    assert "model" in events[0].text
    assert "[tiers.lfm]" in events[0].text


def test_no_model_configured_never_builds_a_runtime():
    """A model-less config must never reach ``build_runtime`` -- if it did,
    it would try to launch Docker."""

    class _ExplodingRuntimeModule:
        @staticmethod
        def build_runtime(*_args, **_kwargs):
            raise AssertionError("build_runtime called with no model configured")

    import nvsh.tiers.runtime_docker as runtime_docker_mod

    agent = LfmAgent({})
    original = runtime_docker_mod.build_runtime
    runtime_docker_mod.build_runtime = _ExplodingRuntimeModule.build_runtime
    try:
        list(agent.run(_request(), _context()))
    finally:
        runtime_docker_mod.build_runtime = original
        agent.close()


# ---------------------------------------------------------------------------
# one turn: propose, explain, or decline -- never silently escalate
# ---------------------------------------------------------------------------


def test_a_propose_reply_proposes_and_ends_with_done():
    chat = _FakeChat([_tool_reply(PROPOSE_TOOL, operation="disk_stats", arguments={})])
    agent = _agent(chat)
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert EventKind.PROPOSAL in kinds
    assert kinds[-1] is EventKind.DONE
    proposal = next(e.proposal for e in events if e.kind is EventKind.PROPOSAL)
    assert proposal.command == "df -h"
    assert "lfm" in proposal.rationale


def test_an_explain_reply_explains_and_ends_with_done():
    chat = _FakeChat([_tool_reply(EXPLAIN_TOOL, text="disk looks fine")])
    agent = _agent(chat)
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert kinds[-1] is EventKind.DONE
    assert EventKind.PROPOSAL not in kinds


def test_an_escalate_reply_declines_in_one_line_and_never_claims_to_ask_the_full_agent():
    chat = _FakeChat([_tool_reply(ESCALATE_TOOL, reason="needs deeper investigation")])
    agent = _agent(chat)
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert kinds[-1] is EventKind.DONE
    assert EventKind.PROPOSAL not in kinds
    status_texts = " ".join(e.text for e in events if e.kind is EventKind.STATUS)
    assert "lfm" in status_texts
    assert "full agent" not in status_texts


def test_a_memory_floor_decline_names_the_reason():
    chat = _FakeChat([])
    config = {"lfm": {"model": "test-model"}, "memory_floor_mb": 999999999}
    agent = LfmAgent(
        config,
        runtime=_Runtime(),
        chat_factory=chat.factory,
        floor_reader=_low_memory_reader,
        platform=_PLATFORM,
    )
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert events[-1].kind is EventKind.DONE
    status_texts = " ".join(e.text for e in events if e.kind is EventKind.STATUS)
    assert "memory" in status_texts
    assert not chat.seen  # the model was never even asked


def test_a_runtime_unavailable_tier_reports_its_own_status_text():
    chat = _FakeChat([])
    agent = _agent(chat, runtime=_Runtime(error="model server down"))
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert events[-1].kind is EventKind.DONE
    explanation = next(e.text for e in events if e.kind is EventKind.STATUS and e.text)
    assert "lfm" in explanation


def test_a_failure_kind_request_reaches_lfm_unlike_needle():
    """Unlike Tier 1, Tier 2 is in every router order (Route._order): a
    FAILURE-kind request must still reach the model, not skip straight to
    an explanation."""
    chat = _FakeChat([_tool_reply(EXPLAIN_TOOL, text="the container is unhealthy")])
    agent = _agent(chat)
    request = AgentRequest(kind=RequestKind.FAILURE, command="./run.sh", exit_code=1)
    try:
        events = list(agent.run(request, _context()))
    finally:
        agent.close()
    assert events[-1].kind is EventKind.DONE
    assert chat.seen


# ---------------------------------------------------------------------------
# cancel / force_stop / close: stop the runtime and chat
# ---------------------------------------------------------------------------


def test_close_before_any_run_does_not_raise():
    agent = _agent(_FakeChat([]))
    agent.close()


def test_close_stops_the_runtime_and_chat():
    chat = _FakeChat([_tool_reply(EXPLAIN_TOOL, text="fine")])
    runtime = _Runtime()
    agent = _agent(chat, runtime=runtime)
    list(agent.run(_request(), _context()))
    agent.close()
    assert runtime.stopped == 1
    assert chat.stopped == 1


def test_cancel_stops_the_runtime_and_chat_too():
    chat = _FakeChat([_tool_reply(EXPLAIN_TOOL, text="fine")])
    runtime = _Runtime()
    agent = _agent(chat, runtime=runtime)
    list(agent.run(_request(), _context()))
    agent.cancel()
    assert runtime.stopped == 1
    assert chat.stopped == 1


def test_force_stop_stops_the_runtime_and_chat():
    chat = _FakeChat([_tool_reply(EXPLAIN_TOOL, text="fine")])
    runtime = _Runtime()
    agent = _agent(chat, runtime=runtime)
    list(agent.run(_request(), _context()))
    agent.force_stop()
    assert runtime.stopped == 1
    assert chat.stopped == 1
