"""QwenAgent: thin subprocess mapper over ``qwen --output-format stream-json``.

This is the fallback path the registry picks when the ACP path is disabled
(or not yet built): ``qwen``'s print mode with structured output, mirroring
``nvsh.agent.claude.ClaudeAgent``'s stream-json envelope parsing. Maps
envelope lines to :class:`AgentEvent`:

* ``{"type": "system", ...}``                          -> ``STATUS``
* ``{"type": "assistant", "message": {"content": [...]}}``
    - a ``thinking`` content part                        -> ``THINKING``
    - a ``tool_use`` content part                        -> ``TOOL_CALL``
    - a ``text`` content part (when neither of the above is present)
                                                          -> ``TEXT_DELTA``
* ``{"type": "user", "message": {"content": [...]}}``
    - a ``tool_result`` content part                     -> ``TOOL_RESULT``
* ``{"type": "result", "subtype": "error", ...}``        -> ``ERROR``
* ``{"type": "result", ...}`` (any other subtype)        -> ``DONE``

Non-JSON lines, and envelope types this adapter does not know about (e.g.
qwen's ``stream_event`` progress lines), are skipped. The prior ``[status]
``-prefix heuristic over plain-text ``qwen -p`` output is gone now that
``--output-format stream-json`` gives structured events instead.

Shapes verified against a real ``qwen --output-format stream-json
--approval-mode plan -p <prompt>`` run, qwen 0.23.3 (recorded, redacted, in
``tests/fakes/qwen``).
"""

from __future__ import annotations

import json

from ._subprocess import SubprocessAgent, build_prompt, build_system_prompt
from .base import AgentContext, AgentEvent, AgentRequest, Capabilities, EventKind


class QwenAgent(SubprocessAgent):
    binary = "qwen"

    def __init__(
        self,
        config: dict | None = None,
        *,
        env: dict[str, str] | None = None,
        model: str | None = None,
        effort: str | None = None,
        extra_args: list[str] | None = None,
        approval: str | None = None,
    ) -> None:
        super().__init__(config, env=env)
        #: Explicit keyword arguments win over the config dict, matching
        #: every other backend's "direct constructor overrides config"
        #: convention this wave settled on.
        self._model = model if model is not None else self._config.get("model")
        #: qwen's print mode has no reasoning-effort knob (checked against
        #: the installed CLI's --help): the value is accepted and stored
        #: for parity with other backends, but never placed on argv, and
        #: ``capabilities().effort`` reports ``False`` regardless of it.
        self._effort = effort if effort is not None else self._config.get("effort")
        raw_extra_args = extra_args if extra_args is not None else self._config.get("extra_args")
        self._extra_args = list(raw_extra_args or [])
        self._approval = (
            approval if approval is not None else (self._config.get("approval") or "nvsh")
        )
        #: tool_use id -> tool name, so a later ``tool_result`` line (which
        #: only carries the id) can still report which tool it answers.
        #: Reset per run so a stale id from a prior turn never survives.
        self._tool_names: dict[str, str] = {}

    def start(self) -> None:
        super().start()
        self._tool_names = {}

    def _argv(self, request: AgentRequest, context: AgentContext) -> list[str]:
        prompt = build_prompt(request, context)
        argv = [self.binary, "--output-format", "stream-json", "--approval-mode", "plan"]
        if self._model:
            argv += ["--model", self._model]
        argv += ["--append-system-prompt", build_system_prompt(context)]
        argv += self._extra_args
        argv += ["-p", prompt]
        return argv

    def _parse_line(self, line: str) -> AgentEvent | None:
        if not line.strip():
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        kind = obj.get("type")
        if kind == "system":
            return AgentEvent(kind=EventKind.STATUS, text=obj.get("subtype", ""))
        if kind == "assistant":
            return self._parse_assistant(obj)
        if kind == "user":
            return self._parse_user(obj)
        if kind == "result":
            if obj.get("subtype") == "error" or obj.get("is_error"):
                error = obj.get("result") or obj.get("error") or ""
                return AgentEvent(kind=EventKind.ERROR, error=str(error))
            return AgentEvent(kind=EventKind.DONE)
        return None

    def _parse_assistant(self, obj: dict) -> AgentEvent | None:
        content = obj.get("message", {}).get("content", [])
        thinking_parts = [p.get("thinking", "") for p in content if p.get("type") == "thinking"]
        if thinking_parts:
            return AgentEvent(kind=EventKind.THINKING, text="".join(thinking_parts))
        tool_use_parts = [p for p in content if p.get("type") == "tool_use"]
        if tool_use_parts:
            call = tool_use_parts[0]
            tool_use_id = str(call.get("id", ""))
            tool_name = str(call.get("name", ""))
            if tool_use_id:
                self._tool_names[tool_use_id] = tool_name
            return AgentEvent(
                kind=EventKind.TOOL_CALL, tool=tool_name, args=dict(call.get("input", {}))
            )
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        if text:
            return AgentEvent(kind=EventKind.TEXT_DELTA, text=text)
        return None

    def _parse_user(self, obj: dict) -> AgentEvent | None:
        content = obj.get("message", {}).get("content", [])
        result_parts = [p for p in content if p.get("type") == "tool_result"]
        if not result_parts:
            return None
        result = result_parts[0]
        tool_use_id = str(result.get("tool_use_id", ""))
        tool_name = self._tool_names.get(tool_use_id, "")
        return AgentEvent(kind=EventKind.TOOL_RESULT, tool=tool_name, result=result.get("content"))

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            # Print mode runs with ``--approval-mode plan`` and has no
            # approval callback: it *reports* tool activity as TOOL_CALL /
            # TOOL_RESULT events but can never pause a tool call for
            # ``nvsh approve``. Per the spec, bare print-mode fallbacks run
            # read-only and declare ``tool_calling=False``.
            tool_calling=False,
            cancellation=True,
            persistent_session=False,
            local_model=False,
            thinking=True,
            effort=False,
            path="stream-json",
            approval=self._approval,
            # Print-mode qwen reads files with its own tools (plan mode still
            # allows read tools); nvsh never sees those reads.
            unmediated_file_access=True,
            # Print mode has no mid-turn channel; a correction becomes the
            # next request ("stop and correct").
            steer=False,
        )
