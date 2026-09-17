"""FakeAgent: a scripted adapter used by tests (and available to real callers offline)."""

from __future__ import annotations

from typing import Iterator

from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind, NvshAgent


class FakeAgent(NvshAgent):
    """Replays a scripted list of events (or raised exceptions).

    ``script`` items are either :class:`AgentEvent` (yielded as-is; an
    ``ERROR``-kind event ends the stream after being yielded) or
    ``BaseException`` instances/classes (raised in place, simulating an
    adapter crashing mid-stream). Cancellation is checked between yields,
    same contract every real adapter must honour.
    """

    def __init__(
        self,
        script: list[AgentEvent | BaseException],
        capabilities: Capabilities | None = None,
    ) -> None:
        self._script = list(script)
        self._capabilities = capabilities or Capabilities()
        self._started = False
        self._closed = False
        self._cancelled = False

    def start(self) -> None:
        self._started = True
        self._cancelled = False

    def run(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        # A cancel ends one turn, like the real adapters; cleared eagerly so
        # one that lands before the first next() still stops this turn.
        self._cancelled = False
        return self._turn(request, context)

    def _turn(self, request: AgentRequest, context: AgentContext) -> Iterator[AgentEvent]:
        for item in self._script:
            if self._cancelled:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
            if item.kind == EventKind.ERROR:
                return

    def cancel(self) -> None:
        self._cancelled = True

    def close(self) -> None:
        self._closed = True

    def capabilities(self) -> Capabilities:
        return self._capabilities

    # Test-only introspection (not part of the NvshAgent contract).
    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def cancelled(self) -> bool:
        return self._cancelled
