"""FakeTier: a scripted tier used by tests (style-matched to nvsh/agent/fake.py)."""

from __future__ import annotations

from ..agent.base import AgentContext, AgentRequest
from .base import Decline, Explanation, Tier, TierDecision, decide


class FakeTier(Tier):
    """Replays a scripted list of outputs, one per call to :meth:`select`.

    ``script`` items are each one of:

    - a raw output (whatever a real tier would emit) -- passed through
      :func:`decide` exactly as a real caller would;
    - a ready-made :class:`TierDecision` or :class:`Decline` -- returned
      as-is, unvalidated, so a test can construct an exact fixture;
    - a ``BaseException`` instance -- raised in place, simulating a tier
      crashing (timeout, process died, ...).

    Records every request/context pair seen via ``requests_seen``, and
    whether :meth:`close` was called via ``closed``.
    """

    def __init__(self, script: list[object], name: str = "fake") -> None:
        self.name = name
        self._script = list(script)
        self._index = 0
        self.requests_seen: list[AgentRequest] = []
        self.closed = False

    def select(
        self, request: AgentRequest, context: AgentContext
    ) -> TierDecision | Decline | Explanation:
        self.requests_seen.append(request)
        if self._index >= len(self._script):
            raise IndexError(f"{self.name}: script exhausted after {self._index} call(s)")
        item = self._script[self._index]
        self._index += 1

        if isinstance(item, BaseException):
            raise item
        if isinstance(item, (TierDecision, Decline, Explanation)):
            return item
        return decide(item)

    def close(self) -> None:
        self.closed = True
