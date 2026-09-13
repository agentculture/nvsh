"""Shared conformance suite, run against every registered NvshAgent adapter.

How to add an adapter (for t10 and later backends): append a zero-arg
factory to ``ADAPTERS`` below. Each factory takes a ``script`` (a list of
``AgentEvent`` -- or, for adapters that support it, raised exceptions -- to
replay) and returns a fresh, unstarted ``NvshAgent`` instance. ``ADAPTERS``
holds a ``FakeAgent`` factory and a ``PiAgent`` factory (``_pi_agent_factory``
below); a real adapter (a stdlib-HTTP adapter, ...) should be wrapped the
same way, e.g. with a fixture backend/record-replay mode so the suite stays
offline. Every adapter in ``ADAPTERS`` must pass every test in this module.

``_pi_agent_factory`` drives ``PiAgent`` against ``tests/fakes/pi_scripted``,
a fake ``pi --mode rpc`` that replays an arbitrary caller-supplied event
script (unlike ``tests/fakes/pi``, whose canned responses are fixed and are
instead exercised directly by ``tests/test_pi_agent.py``). This suite only
ever scripts ``STATUS``, ``TEXT_DELTA``, ``ERROR`` and ``DONE`` events, so
``_agent_event_to_wire`` only needs to cover those four kinds -- the wire
shapes it produces are exactly what ``PiAgent._map_event`` (see
``nvsh/agent/pi.py``) maps back to the original ``AgentEvent``: an
unrecognized wire ``"type"`` maps to ``STATUS`` with that type as its text,
which is what lets a plain ``{"type": "one"}`` line stand in for
``AgentEvent(kind=STATUS, text="one")`` without a bespoke wire vocabulary.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nvsh.agent import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    FakeAgent,
    PiAgent,
    RequestKind,
)
from tests._fake_adapters import ClaudeAgentViaFake, CodexAgentViaFake, QwenAgentViaFake

FAKES_DIR = os.path.join(os.path.dirname(__file__), "fakes")


def _agent_event_to_wire(event: AgentEvent) -> dict:
    """Inverse of ``PiAgent._map_event`` for the event kinds this suite scripts."""
    if event.kind == EventKind.STATUS:
        return {"type": event.text}
    if event.kind == EventKind.TEXT_DELTA:
        return {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": event.text},
        }
    if event.kind == EventKind.DONE:
        return {"type": "agent_end"}
    if event.kind == EventKind.ERROR:
        return {"type": "error", "error": event.error}
    raise NotImplementedError(f"no conformance wire mapping for {event.kind}")


def _pi_agent_factory(script):
    """Build a ``PiAgent`` that replays ``script`` via ``tests/fakes/pi_scripted``."""
    tmp_home = tempfile.mkdtemp(prefix="nvsh-pi-conformance-")
    env = dict(os.environ)
    env["PATH"] = FAKES_DIR + os.pathsep + env.get("PATH", "")
    env["HOME"] = tmp_home
    env["XDG_STATE_HOME"] = os.path.join(tmp_home, "state")
    wire = [_agent_event_to_wire(e) for e in script]
    env["NVSH_TEST_PI_SCRIPT"] = json.dumps(wire)
    return PiAgent(pi_path="pi_scripted", env=env)


_pi_agent_factory.__name__ = "PiAgent"

# Registry of adapter factories. Append here to bring a new backend under the
# same conformance suite.
ADAPTERS = [
    FakeAgent,
    ClaudeAgentViaFake,
    CodexAgentViaFake,
    QwenAgentViaFake,
    _pi_agent_factory,
]


@pytest.fixture(params=ADAPTERS, ids=lambda factory: factory.__name__)
def adapter_factory(request):
    return request.param


def _request() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="ls /nope", exit_code=2)


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="ls: /nope: No such file", cwd="/tmp")


def test_streaming_order(adapter_factory):
    Event = AgentEvent
    script = [
        Event(kind=EventKind.STATUS, text="inspecting"),
        Event(kind=EventKind.TEXT_DELTA, text="looking at "),
        Event(kind=EventKind.TEXT_DELTA, text="the failure"),
        Event(kind=EventKind.DONE),
    ]
    agent = adapter_factory(script)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert [e.kind for e in events] == [
        EventKind.STATUS,
        EventKind.TEXT_DELTA,
        EventKind.TEXT_DELTA,
        EventKind.DONE,
    ]
    assert events[1].text == "looking at "
    assert events[2].text == "the failure"


def test_cancel_mid_stream(adapter_factory):
    Event = AgentEvent
    script = [
        Event(kind=EventKind.STATUS, text="one"),
        Event(kind=EventKind.STATUS, text="two"),
        Event(kind=EventKind.STATUS, text="three"),
        Event(kind=EventKind.DONE),
    ]
    agent = adapter_factory(script)
    agent.start()
    seen = []
    try:
        for event in agent.run(_request(), _context()):
            seen.append(event)
            if len(seen) == 1:
                agent.cancel()
    finally:
        agent.close()
    # Cancellation is checked between yields: exactly the events already in
    # flight when cancel() was called are seen, and nothing scripted after.
    assert seen == [Event(kind=EventKind.STATUS, text="one")]


def test_error_propagation(adapter_factory):
    Event = AgentEvent
    script = [
        Event(kind=EventKind.STATUS, text="inspecting"),
        Event(kind=EventKind.ERROR, error="backend unreachable"),
        Event(kind=EventKind.DONE),
    ]
    agent = adapter_factory(script)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    # The error event is delivered, and it ends the stream: DONE never comes.
    assert [e.kind for e in events] == [EventKind.STATUS, EventKind.ERROR]
    assert events[-1].error == "backend unreachable"


def test_capability_report(adapter_factory):
    agent = adapter_factory([])
    caps = agent.capabilities()
    assert isinstance(caps, Capabilities)
    assert isinstance(caps.streaming, bool)
    assert isinstance(caps.tool_calling, bool)
    assert isinstance(caps.cancellation, bool)
    assert isinstance(caps.persistent_session, bool)
    assert isinstance(caps.local_model, bool)


def test_teardown(adapter_factory):
    agent = adapter_factory([AgentEvent(kind=EventKind.DONE)])
    agent.start()
    list(agent.run(_request(), _context()))
    # close() must not raise, and must be safe to call more than once.
    agent.close()
    agent.close()
