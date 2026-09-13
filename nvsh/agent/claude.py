"""ClaudeAgent: thin subprocess mapper over ``claude -p --output-format stream-json``.

Maps stream-json envelope lines to :class:`AgentEvent`:

* ``{"type": "system", ...}``                       -> ``STATUS``
* ``{"type": "assistant", "message": {...}}``        -> ``TEXT_DELTA`` (joined text parts)
* ``{"type": "result", "subtype": "error", ...}``     -> ``ERROR``
* ``{"type": "result", ...}`` (any other subtype)     -> ``DONE``

Non-JSON lines are skipped. Tool calls are not mapped in v1 (``tool_calling``
capability is ``False``); a future version can add ``tool_use``/``tool_result``
content-block mapping without changing this contract.
"""

from __future__ import annotations

import json

from ._subprocess import SubprocessAgent, build_prompt, build_system_prompt
from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind


class ClaudeAgent(SubprocessAgent):
    binary = "claude"

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        prompt = build_prompt(request, context)
        return [
            self.binary,
            "-p",
            prompt,
            "--append-system-prompt",
            build_system_prompt(context),
            "--output-format",
            "stream-json",
        ]

    def _parse_line(self, line: str) -> AgentEvent | None:
        if not line.strip():
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        kind = obj.get("type")
        if kind == "system":
            return AgentEvent(kind=EventKind.STATUS, text=obj.get("text", ""))
        if kind == "assistant":
            content = obj.get("message", {}).get("content", [])
            text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
            return AgentEvent(kind=EventKind.TEXT_DELTA, text=text)
        if kind == "result":
            if obj.get("subtype") == "error":
                return AgentEvent(kind=EventKind.ERROR, error=obj.get("error", ""))
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
