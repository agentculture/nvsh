"""CodexAgent: thin subprocess mapper over ``codex exec --json``.

Maps ``{"msg": {"type": ...}}`` envelope lines to :class:`AgentEvent`:

* ``task_started``       -> ``STATUS``
* ``agent_message_delta``-> ``TEXT_DELTA``
* ``error``              -> ``ERROR``
* ``task_complete``      -> ``DONE``

Non-JSON lines and unrecognized ``msg`` types are skipped. Tool calls are not
mapped in v1 (``tool_calling`` capability is ``False``).
"""

from __future__ import annotations

import json

from ._subprocess import SubprocessAgent, build_full_prompt
from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind


class CodexAgent(SubprocessAgent):
    binary = "codex"

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        # ``codex exec`` has no --system-prompt/--append-system-prompt flag
        # (checked against the installed CLI's --help), so the system brief
        # leads the prompt text itself.
        prompt = build_full_prompt(request, context)
        return [self.binary, "exec", "--json", prompt]

    def _parse_line(self, line: str) -> AgentEvent | None:
        if not line.strip():
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        msg = obj.get("msg", {})
        kind = msg.get("type")
        if kind == "task_started":
            return AgentEvent(kind=EventKind.STATUS, text=msg.get("text", ""))
        if kind == "agent_message_delta":
            return AgentEvent(kind=EventKind.TEXT_DELTA, text=msg.get("delta", ""))
        if kind == "error":
            return AgentEvent(kind=EventKind.ERROR, error=msg.get("message", ""))
        if kind == "task_complete":
            return AgentEvent(kind=EventKind.DONE)
        return None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_calling=False,
            cancellation=True,
            persistent_session=False,
            local_model=False,
        )
