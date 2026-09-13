"""Shared conformance suite, run against every registered NvshAgent adapter.

How to add an adapter (for t9/t10 and later backends): append a zero-arg
factory to ``ADAPTERS`` below. Each factory takes a ``script`` (a list of
``AgentEvent`` -- or, for adapters that support it, raised exceptions -- to
replay) and returns a fresh, unstarted ``NvshAgent`` instance. Today
``ADAPTERS`` holds only a ``FakeAgent`` factory; a real adapter (PiAgent,
a stdlib-HTTP adapter, ...) should be wrapped the same way, e.g. with a
fixture backend/record-replay mode so the suite stays offline. Every
adapter in ``ADAPTERS`` must pass every test in this module.
"""

from __future__ import annotations

import pytest

from nvsh.agent import (
    AgentContext,
    AgentEvent,
    AgentRequest,
    Capabilities,
    EventKind,
    FakeAgent,
    RequestKind,
)

# Registry of adapter factories. Append here to bring a new backend under the
# same conformance suite.
ADAPTERS = [
    FakeAgent,
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
