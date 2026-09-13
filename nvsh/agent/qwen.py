"""QwenAgent: thin subprocess mapper over ``qwen -p``.

``qwen -p`` has no structured output protocol: it prints plain assistant
text. Each non-empty stdout line becomes a ``TEXT_DELTA``. A ``[status] ``
line prefix (a heuristic for verbose/tool-status output some ``-p`` modes
print) maps to ``STATUS`` instead. ``DONE`` is synthesized on a clean exit;
a non-zero exit yields ``ERROR`` with stderr as the message.
"""

from __future__ import annotations

from ._subprocess import SubprocessAgent, build_prompt
from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind

_STATUS_PREFIX = "[status] "


class QwenAgent(SubprocessAgent):
    binary = "qwen"

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        prompt = build_prompt(request, context)
        return [self.binary, "-p", prompt]

    def _parse_line(self, line: str) -> AgentEvent | None:
        if not line:
            return None
        if line.startswith(_STATUS_PREFIX):
            return AgentEvent(kind=EventKind.STATUS, text=line[len(_STATUS_PREFIX) :])
        return AgentEvent(kind=EventKind.TEXT_DELTA, text=line)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=False,
            cancellation=True,
            persistent_session=False,
            local_model=False,
        )
