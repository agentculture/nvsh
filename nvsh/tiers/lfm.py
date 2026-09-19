"""Tier 2: a small local agent running a bounded inspect-interpret-propose loop.

Tier 2 sits between Needle3's single reflex pick (Tier 1) and the configured
full agent (Tier 3). It talks to one OpenAI-compatible endpoint on localhost
-- whatever :class:`~nvsh.tiers.runtime.Runtime` hands it -- and is allowed a
small, fixed amount of work per request:

* at most ``max_rounds`` model turns (4 by default);
* **read-only operations only** inside the loop. A bare call to a mutating
  operation is refused with one line and fed back to the model; the runner is
  never reached for it;
* a **fixed context budget**: a static system brief plus the platform kind,
  the request itself (for a FAILURE: command, exit status and a bounded,
  redacted output tail), and only the last ``keep_results`` tool results
  verbatim -- older ones become a one-line stub. The whole message list stays
  under :data:`MAX_CONTEXT_CHARS`, however long the machine's output was.

and exactly three outcomes:

``TierDecision``
    the model called ``propose``; the *router* grounds, renders and shows it
    for approval. The tier never builds a command for the operator and never
    runs a proposed operation, mutating or not.
``Explanation``
    the model called ``explain`` (or answered in plain words).
``Decline``
    the model called ``escalate``, produced nothing usable, or ran out of
    rounds. The decline carries every inspection it already did, so the full
    agent does not repeat the same read-only work.

Nothing here is keyed on a particular operation's name: the tools offered to
the model are generated from :mod:`nvsh.ops.table`, and ``read_only`` is read
off the table entry. Adding an operation needs no change in this module.

The model's output is untrusted: every call it makes travels
:func:`~nvsh.tiers.base.decide` -> :func:`~nvsh.ops.ground.ground` ->
:func:`~nvsh.ops.render.render` before an argv exists at all, and everything
that comes back is redacted (:func:`nvsh.redact.redact`) before it re-enters
the context.

This module is **not** importable on the hot success path: import it lazily
from any CLI/doctor/daemon code.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from ..agent.base import AgentContext, AgentRequest, RequestKind
from ..ops import ground as ops_ground
from ..ops import table as ops_table
from ..ops._model import ArgSpec, Operation
from ..ops.render import render as render_argv
from ..platform._model import Platform
from ..redact import redact
from ._bounded import bounded_cut
from .base import Decline, DeclineReason, Explanation, Tier, TierDecision, decide
from .memfloor import FloorResult
from .runtime import Runtime, RuntimeUnavailable
from .toolchat import ChatReply, ToolCall, ToolChat, ToolChatError

#: The three control tools, offered alongside every operation in the table.
PROPOSE_TOOL = "propose"
EXPLAIN_TOOL = "explain"
ESCALATE_TOOL = "escalate"

#: What the model is told when it calls a mutating operation as an inspection
#: step. The spec's wording, verbatim: a fix is *proposed*, never run here.
MUTATING_REFUSED = "mutating operations cannot run here; use propose"

#: Loop and context defaults. Every one of them is a constructor argument.
MAX_ROUNDS = 4
OP_TIMEOUT_SECONDS = 10.0
RESULT_CHARS = 2048
KEEP_RESULTS = 3

#: Bounds on the pieces of the context that come from outside this module.
EXPLANATION_CHARS = 2000
OUTPUT_TAIL_CHARS = 1500
REQUEST_CHARS = 1500
LABEL_CHARS = 200
#: How much of one *argument value* survives when the serialised arguments do
#: not fit :data:`LABEL_CHARS`. Small enough that several clamped values still
#: fit the label bound together.
ARG_VALUE_CHARS = 60
COMMAND_CHARS = 400
REASON_CHARS = 400
KIND_CHARS = 64

#: How much text past a bound is still handed to the redactor.
#: :func:`nvsh.redact.redact` is quadratic on long unbroken input (40k
#: characters measured at 0.8 s, 160k at 12.7 s), so nothing may be redacted
#: at its full length: a model calling ``explain`` with a megabyte of text
#: would otherwise hold this tier's lock for minutes. Every text is cut to
#: ``limit + REDACT_SLACK`` *first*, redacted, and only then clamped to
#: ``limit``. The slack is what makes that safe: a secret straddling the
#: clamp cut lies wholly inside the slack, so the redactor still sees it
#: whole and replaces it, and the clamp can then only cut into the marker
#: that replaced it -- never back into the secret itself.
REDACT_SLACK = 512

#: Hard ceiling on the whole message list, measured as the JSON that goes on
#: the wire. With the defaults the worst case is about 10.8k characters (a
#: ~0.5k brief, a 1.8k request, four echoed calls clamped to
#: :data:`LABEL_CHARS` each, and three :data:`RESULT_CHARS` results), so this
#: leaves headroom without ever letting the context grow with the machine's
#: output. ``tests/test_tier_lfm.py`` asserts it after four rounds of
#: maximum-size results (8865 characters as measured there).
MAX_CONTEXT_CHARS = 12000

_SYSTEM_BRIEF = (
    "You are nvsh's local assistant on a {kind} machine.\n"
    "Call the read-only operations offered as tools to inspect this machine.\n"
    "Mutating operations cannot be run here: to fix something, call"
    f" {PROPOSE_TOOL} with the operation and its arguments, and the operator"
    " approves it.\n"
    f"End your turn in exactly one of three ways: {PROPOSE_TOOL}(operation,"
    f" arguments), {EXPLAIN_TOOL}(text), or {ESCALATE_TOOL}(reason) when this"
    " needs a bigger agent.\n"
    "You have at most {rounds} inspection rounds. Never write a shell command."
)

_NO_USABLE_OUTPUT = "the model produced no usable output"

#: What an assistant turn echoes when even the clamped arguments do not fit:
#: valid JSON, so the conversation replays, and self-describing, so the model
#: can see why its own call came back shortened.
_DROPPED_ARGUMENTS = json.dumps({"_arguments_dropped": True})

ChatFactory = Callable[[str], "ChatLike"]
FloorFn = Callable[[], FloorResult]


class ChatLike(Protocol):
    """The two methods Tier 2 needs from a chat client."""

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        """One completion. Raises nothing but ``ToolChatError``."""

    def stop(self) -> None:
        """Unblock a request in flight. Safe from another thread."""


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def _clamp(text: str, limit: int) -> str:
    """*text* cut to *limit* characters, with an ellipsis when it was cut."""
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _clamp_tail(text: str, limit: int) -> str:
    """The LAST *limit* characters of *text*, with a leading ellipsis when cut."""
    if len(text) <= limit:
        return text
    return "..." + text[-max(limit - 3, 0) :]


def _one_line(text: object) -> str:
    """Collapse *text* to a single line so a fed-back error stays one line."""
    return " ".join(str(text).split())


def _redacted(text: object, limit: int, *, tail: bool = False) -> str:
    """*text* bounded to *limit*, with every secret shape replaced by a marker.

    The bound is applied **before** the redactor runs (see
    :data:`REDACT_SLACK` for why that order is not optional) and again after
    it. ``tail=True`` keeps the end of the text rather than its start, which
    is what a failed command's output needs: the error is at the bottom.

    Cutting first would break the one redaction rule that needs a whole
    multi-line structure, so the cut goes through
    :func:`~nvsh.tiers._bounded.bounded_cut`, which replaces a private-key
    block the cut split with a fixed placeholder.
    """
    raw = text if isinstance(text, str) else str(text)
    window = limit + REDACT_SLACK
    cut = bounded_cut(raw, window, tail=tail)
    cleaned = redact(cut.encode("utf-8", "replace")).decode("utf-8", "replace")
    return _clamp_tail(cleaned, limit) if tail else _clamp(cleaned, limit)


def _describe(operation: str, args: Mapping[str, str]) -> str:
    """``service_logs service=nginx.service`` -- one inspection, named."""
    rendered = " ".join(f"{name}={args[name]}" for name in sorted(args))
    return _clamp(f"{operation} {rendered}".strip(), LABEL_CHARS)


# ---------------------------------------------------------------------------
# tool schemas, generated from the operation table
# ---------------------------------------------------------------------------


def _arg_schema(spec: ArgSpec) -> dict[str, Any]:
    if spec.kind == "choice":
        return {"type": "string", "enum": list(spec.choices)}
    return {"type": "string"}


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _operation_tool(operation: Operation) -> dict:
    """The OpenAI tool schema for one operation. Nothing is named by hand."""
    return _function(
        operation.name,
        operation.description,
        {spec.name: _arg_schema(spec) for spec in operation.args},
        [spec.name for spec in operation.args],
    )


def tools_for(operations: tuple[Operation, ...] | None = None) -> list[dict]:
    """Every operation in the table, plus the three control tools.

    The operation schemas come from the table itself, so an operation added
    to :mod:`nvsh.ops.table` is offered to the model with no change here.
    """
    table = ops_table.OPERATIONS if operations is None else operations
    names = [operation.name for operation in table]
    return [_operation_tool(operation) for operation in table] + [
        _function(
            PROPOSE_TOOL,
            "Propose one operation for the operator to approve. The only way to"
            " end with a command, and the only way to reach a mutating operation.",
            {
                "operation": {"type": "string", "enum": names},
                "arguments": {"type": "object"},
            },
            ["operation"],
        ),
        _function(
            EXPLAIN_TOOL,
            "Answer the operator in plain words, with no command.",
            {"text": {"type": "string"}},
            ["text"],
        ),
        _function(
            ESCALATE_TOOL,
            "Hand this request to the full agent, saying why.",
            {"reason": {"type": "string"}},
            ["reason"],
        ),
    ]


# ---------------------------------------------------------------------------
# the context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Round:
    """One model turn that produced a tool result (an inspection or an error).

    ``inspection`` is the ``(operation + args, excerpt)`` pair the decline or
    explanation carries outward; it is ``None`` for a fed-back error, which
    the next tier has no use for.
    """

    call_id: str
    name: str
    arguments: str
    result: str
    inspection: tuple[str, str] | None


def system_brief(platform: Platform, max_rounds: int = MAX_ROUNDS) -> str:
    """Static text plus the platform kind. The operations arrive as tools."""
    return _SYSTEM_BRIEF.format(kind=_clamp(str(platform.kind), KIND_CHARS), rounds=max_rounds)


def request_message(request: AgentRequest, context: AgentContext) -> str:
    """The request, bounded: a failure's command/status/output tail, or the ask."""
    if request.kind is RequestKind.FAILURE:
        tail = _redacted(context.output, OUTPUT_TAIL_CHARS, tail=True)
        command = _redacted(request.command, COMMAND_CHARS)
        return f"command: {command}\nexit status: {request.exit_code}\noutput tail:\n{tail}"
    return _redacted(request.prompt or request.ask or request.command, REQUEST_CHARS)


def _messages(system: str, ask: str, rounds: list[_Round], keep_results: int) -> list[dict]:
    """The whole context for one completion, with old results stubbed out.

    Only the last *keep_results* tool results survive verbatim; every earlier
    one is replaced by a one-line stub. That, plus the clamps on every piece
    that comes from outside, is what keeps the context fixed in size.
    """
    out: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": ask},
    ]
    keep_from = len(rounds) - keep_results
    for index, entry in enumerate(rounds):
        out.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": entry.call_id,
                        "type": "function",
                        "function": {"name": entry.name, "arguments": entry.arguments},
                    }
                ],
            }
        )
        kept = entry.result if index >= keep_from else f"({entry.name}: result dropped)"
        out.append({"role": "tool", "tool_call_id": entry.call_id, "content": kept})
    return out


# ---------------------------------------------------------------------------
# the tier
# ---------------------------------------------------------------------------


class LfmTier(Tier):
    """Tier 2: inspects read-only, then proposes, explains or escalates.

    The constructor is cheap: it opens no socket and starts no process. The
    runtime is asked for its URL on the first :meth:`select`, which is also
    where the chat client is built -- so a Tier 2 that is configured but never
    reached costs nothing.

    :meth:`select` never raises and never executes a proposed operation.
    """

    name = "lfm"

    def __init__(
        self,
        runtime: Runtime,
        platform: Platform,
        *,
        model: str,
        runner: ops_ground.Runner = ops_ground.default_runner,
        chat_factory: ChatFactory | None = None,
        max_rounds: int = MAX_ROUNDS,
        op_timeout: float = OP_TIMEOUT_SECONDS,
        result_chars: int = RESULT_CHARS,
        keep_results: int = KEEP_RESULTS,
        floor_check: FloorFn | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runtime = runtime
        self._platform = platform
        self._model = model
        self._runner = runner
        self._chat_factory = chat_factory if chat_factory is not None else self._default_chat
        self._max_rounds = int(max_rounds)
        self._op_timeout = float(op_timeout)
        self._result_chars = int(result_chars)
        self._keep_results = int(keep_results)
        self._floor_check = floor_check
        self._clock = clock
        self._chat: ChatLike | None = None
        # One model turn at a time: the daemon shares a tier across handler
        # threads, and two interleaved loops would read each other's rounds.
        self._lock = threading.Lock()

    # -- public surface --

    def select(
        self, request: AgentRequest, context: AgentContext
    ) -> TierDecision | Decline | Explanation:
        """Run the bounded loop for *request*. Never raises, never executes."""
        try:
            with self._lock:
                return self._select(request, context)
        except Exception as exc:  # noqa: BLE001 -- the model and its server are untrusted
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"lfm tier failed: {type(exc).__name__}: {_one_line(exc)}",
            )

    def status(self) -> str:
        """One line for ``nvsh overview`` / ``doctor``. Starts nothing."""
        return f"lfm tier ({self._model}): {_one_line(self._runtime.status())}"

    def close(self) -> None:
        """Stop the chat and release the runtime. Never raises."""
        chat, self._chat = self._chat, None
        for release in (getattr(chat, "stop", None), self._runtime.stop):
            _quietly(release)

    # -- the loop --

    def _select(
        self, request: AgentRequest, context: AgentContext
    ) -> TierDecision | Decline | Explanation:
        floor = self._floor_check() if self._floor_check is not None else None
        if floor is not None and not floor.ok:
            return Decline(reason=DeclineReason.MEMORY_FLOOR, detail=_one_line(floor.status))

        chat = self._ensure_chat()
        if isinstance(chat, Decline):
            return chat

        system = system_brief(self._platform, self._max_rounds)
        ask = request_message(request, context)
        tools = tools_for()
        rounds: list[_Round] = []
        started = self._clock()
        for _ in range(self._max_rounds):
            outcome = self._turn(chat, system, ask, tools, rounds)
            if outcome is not None:
                return outcome
        elapsed = self._clock() - started
        return Decline(
            reason=DeclineReason.ESCALATED,
            detail=f"stopped after {self._max_rounds} inspection rounds ({elapsed:.1f}s)",
            inspections=_inspections(rounds),
        )

    def _turn(
        self,
        chat: ChatLike,
        system: str,
        ask: str,
        tools: list[dict],
        rounds: list[_Round],
    ) -> TierDecision | Decline | Explanation | None:
        """One model turn. ``None`` means "keep going"; anything else ends the loop."""
        messages = _messages(system, ask, rounds, self._keep_results)
        try:
            # One attempt per round: a retry would double the operator's wait,
            # and the round budget is what bounds the loop.
            reply = chat.complete(messages, tools)
        except ToolChatError as exc:
            return Decline(
                reason=DeclineReason.TIER_ERROR,
                detail=f"lfm chat failed: {_one_line(exc)}",
                inspections=_inspections(rounds),
            )
        if not reply.tool_calls:
            return self._spoke(reply, rounds)
        # More than one call in a reply: the first is handled and the rest are
        # ignored. The loop is a chain of single steps, and the model sees the
        # result of the call it asked for first on the next round.
        return self._call(reply.tool_calls[0], rounds)

    def _spoke(self, reply: ChatReply, rounds: list[_Round]) -> Decline | Explanation:
        """A reply with no tool call: plain words, or nothing usable at all."""
        text = _redacted(reply.text, EXPLANATION_CHARS).strip()
        if not text:
            return Decline(
                reason=DeclineReason.ESCALATED,
                detail=_NO_USABLE_OUTPUT,
                inspections=_inspections(rounds),
            )
        return Explanation(text=text, inspections=_inspections(rounds))

    def _call(
        self, call: ToolCall, rounds: list[_Round]
    ) -> TierDecision | Decline | Explanation | None:
        """Dispatch one tool call: a control tool, or an inspection step."""
        if call.name == ESCALATE_TOOL:
            reason = call.arguments.get("reason")
            return Decline(
                reason=DeclineReason.ESCALATED,
                detail=(
                    _one_line(_redacted(reason, REASON_CHARS)) if isinstance(reason, str) else ""
                ),
                inspections=_inspections(rounds),
            )
        if call.name == EXPLAIN_TOOL:
            return self._explained(call, rounds)
        if call.name == PROPOSE_TOOL:
            return self._proposed(call, rounds)
        self._inspect(call, rounds)
        return None

    def _explained(self, call: ToolCall, rounds: list[_Round]) -> Explanation | None:
        text = call.arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            self._feedback(call, rounds, f"{EXPLAIN_TOOL} needs a non-empty 'text' argument")
            return None
        return Explanation(
            text=_redacted(text, EXPLANATION_CHARS).strip(),
            inspections=_inspections(rounds),
        )

    def _proposed(self, call: ToolCall, rounds: list[_Round]) -> TierDecision | None:
        """The only way to end with a command -- and the tier still runs nothing."""
        arguments = call.arguments.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _loads(arguments)
        decision = decide([{"name": call.arguments.get("operation"), "arguments": arguments}])
        if isinstance(decision, Decline):
            self._feedback(call, rounds, decision.detail or decision.reason.value)
            return None
        return decision

    # -- one inspection step --

    def _inspect(self, call: ToolCall, rounds: list[_Round]) -> None:
        """Run one read-only operation and feed its result back. Never proposes.

        A mutating operation is refused here without ever reaching the runner:
        ``read_only`` is read off the table entry, so no operation name is
        special-cased.
        """
        operation = ops_table.get(call.name)
        if operation is not None and not operation.read_only:
            self._feedback(call, rounds, MUTATING_REFUSED)
            return
        prepared = self._prepare(call)
        if isinstance(prepared, str):
            self._feedback(call, rounds, prepared)
            return
        argv, label = prepared
        exit_code, output = self._call_runner(argv)
        result = _clamp(
            f"exit {exit_code}\n{_redacted(output, self._result_chars)}", self._result_chars
        )
        rounds.append(_record(call, len(rounds), result, (label, result)))

    def _prepare(self, call: ToolCall) -> tuple[list[str], str] | str:
        """``(argv, label)`` for a validated, grounded, rendered call, or one error line."""
        decision = decide([{"name": call.name, "arguments": call.arguments}])
        if isinstance(decision, Decline):
            return _one_line(decision.detail or decision.reason.value)
        operation = ops_table.get(decision.operation)
        if operation is None:
            return f"unknown operation {decision.operation!r}"
        grounded = ops_ground.ground(operation, dict(decision.args), self._runner)
        if isinstance(grounded, ops_ground.GroundDecline):
            return _one_line(grounded.message)
        argv = render_argv(operation.name, dict(grounded.args), self._platform)
        if argv is None:
            return f"no single command runs {operation.name} on this machine"
        return (argv, _describe(operation.name, grounded.args))

    def _call_runner(self, argv: list[str]) -> tuple[int, str]:
        """The injected runner, with anything it raises turned into a failed run."""
        try:
            exit_code, output = self._runner(list(argv), self._op_timeout)
        except Exception as exc:  # noqa: BLE001 -- the runner is injectable
            return (127, f"the inspection could not be run: {_one_line(exc)}")
        return (int(exit_code), output if isinstance(output, str) else str(output))

    def _feedback(self, call: ToolCall, rounds: list[_Round], message: str) -> None:
        """Tell the model, in one line, why its call did not run. Costs a round."""
        rounds.append(_record(call, len(rounds), f"error: {_one_line(message)}", None))

    # -- the chat client --

    def _ensure_chat(self) -> ChatLike | Decline:
        if self._chat is not None:
            return self._chat
        try:
            base_url = self._runtime.ensure()
        except RuntimeUnavailable as exc:
            return Decline(reason=DeclineReason.TIER_UNAVAILABLE, detail=_one_line(exc))
        try:
            self._chat = self._chat_factory(base_url)
        except Exception as exc:  # noqa: BLE001 -- the factory is injectable
            return Decline(
                reason=DeclineReason.TIER_UNAVAILABLE,
                detail=f"could not open a chat on {base_url}: {_one_line(exc)}",
            )
        return self._chat

    def _default_chat(self, base_url: str) -> ChatLike:
        """A non-streaming :class:`~nvsh.tiers.toolchat.ToolChat`, localhost only."""
        return ToolChat(base_url, self._model, stream=False)


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _clamped_values(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """*arguments* with every long string value cut, keys and shape intact."""
    return {
        str(name): _clamp(value, ARG_VALUE_CHARS) if isinstance(value, str) else value
        for name, value in arguments.items()
    }


def _bounded_arguments(arguments: object) -> str:
    """The call's arguments as JSON that is **always parseable** and bounded.

    Clamping the serialised form as plain text would cut mid-token and hand
    the server invalid JSON in the assistant message replayed on the next
    round, which a strict server may reject outright. So an oversized
    argument set is re-serialised with its long string *values* clamped, and
    if that is still too long it degrades to a valid one-key object saying
    the arguments were dropped. ``json.loads`` succeeds on every branch.
    """
    raw = json.dumps(arguments, default=str)
    if len(raw) <= LABEL_CHARS:
        return raw
    if isinstance(arguments, Mapping):
        raw = json.dumps(_clamped_values(arguments), default=str)
        if len(raw) <= LABEL_CHARS:
            return raw
    return _DROPPED_ARGUMENTS


def _record(call: ToolCall, index: int, result: str, inspection: tuple[str, str] | None) -> _Round:
    """One round's echo and result, with every part already bounded."""
    return _Round(
        call_id=f"call_{index}",
        name=_clamp(str(call.name), LABEL_CHARS),
        arguments=_bounded_arguments(call.arguments),
        result=result,
        inspection=inspection,
    )


def _inspections(rounds: list[_Round]) -> tuple[tuple[str, str], ...]:
    """Every read-only result this loop produced, in order."""
    return tuple(entry.inspection for entry in rounds if entry.inspection is not None)


def _loads(raw: str) -> object:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _quietly(release: Callable[[], None] | None) -> None:
    if release is None:
        return
    try:
        release()
    except Exception:  # noqa: BLE001 -- close() must never raise
        # Nothing left to release: the caller is shutting down anyway, and a
        # tier that fails to close must not take the daemon's shutdown with it.
        return
