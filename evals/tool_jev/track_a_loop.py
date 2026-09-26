"""Track A for reference models through the candidates' own LfmTier loop (issue #64, d1).

Deviation d1 (approved): a reference model answers Track A through the same
multi-round loop the local candidates were measured with -- up to
``nvsh.tiers.lfm.MAX_ROUNDS`` model turns, read-only inspections executed
against the recorded ground snapshot, ending in propose / explain /
escalate -- instead of one single-turn tool call. That is what makes spec
claim c33 / honesty condition h23 hold: the reference sees the measure.py
prompt, the offered candidate list and the ground snapshot exactly as the
candidates did.

Nothing about the loop is reimplemented here. For every case this module
builds a real :class:`nvsh.tiers.lfm.LfmTier` the way
``scripts/lfm-finetune/measure.py`` does (loaded by path, never copied):

- grounding and inspections run through ``measure.snapshot_runner`` over
  the recorded snapshot;
- the case becomes an ``AgentRequest``/``AgentContext`` pair the way
  ``nvsh.tiers.bench.request_for``/``context_for`` build one (via
  ``request._agent_request_and_context``, t11's reading of the same rule);
- the tool list is narrowed per case with ``measure.restrict_tools``,
  exactly as ``measure.RecordingChat.complete`` narrows it;
- the final result becomes a decision through ``measure._decided`` (the
  same mapping measure.py applies to a candidate), then a
  :class:`~evals.tool_jev.trace.RawRecord` via ``from_provider_answer``.

The reply a reference gives is read the way the candidates' own chat client
(``nvsh.tiers.toolchat``) reads a local model's: every call is sent with
``tool_choice: "auto"`` (ToolChat sends none, so the server default ``auto``
applied to the candidates); a tool call is a tool call; a reply with no
tool call hands its visible text (never reasoning) to ``LfmTier``, with any
tool call printed in that text parsed the same way ToolChat parses one --
so plain words are an explanation for a reference exactly as for a
candidate (plan risk r9). Two things never become an explanation: a
structural refusal, and a reply the provider cut at the output budget
(``invalid``, reason ``truncated``).

What differs is only the chat client: :class:`DeferredChat`. LfmTier calls
``complete(messages, tools)`` once per round; ``DeferredChat`` turns that
round into a :class:`~evals.tool_jev.providers.base.CallRequest` (system and
user message plus the prior rounds as ``history``), keys it in the ledger
(``CallSpec.target = "tool_call:r<round>"``, ``prompt_hash`` over the exact
system text, messages and tools), and either replays the cached answer --
re-read by the provider adapter's own ``result_from_raw``, then turned into
the ``ChatReply`` LfmTier reads -- or stops the loop at the first uncached
round, handing that round's call back as a :class:`PendingCall`.

Driving it (:func:`run_round`) is replay-from-scratch: every call re-runs a
case's loop from round 1, replaying every cached round (inspections are
deterministic against the snapshot) until the case finishes or reaches its
first uncached round. The module never sends anything: the runner submits
the round's pending calls -- synchronously or as ONE batch -- and records
the answers (:func:`record_results`), and the next :func:`run_round` gets
one round further. A crash between rounds loses nothing: the ledger holds
every answered round, and a reopened ledger resumes at the first missing one.

No module under ``nvsh/`` imports this one; no network, no subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from nvsh.platform._model import Platform
from nvsh.tiers import lfm
from nvsh.tiers.toolchat import ChatReply, ToolCall, parse_raw_tool_calls

from . import request as contract
from .cases import Case
from .ledger import DONE, INVALID, CachedResponse, CallSpec, Ledger, canonical_json, prompt_hash
from .providers.base import CallRequest, CallResult, Provider, ReplyText
from .providers.errors import Outcome
from .trace import RawRecord

_MEASURE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "lfm-finetune" / "measure.py"


def _load_measure() -> Any:
    spec = importlib.util.spec_from_file_location("_tool_jev_track_a_measure", _MEASURE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


measure = _load_measure()

#: ``CallSpec.target`` of round *n* is ``f"{TARGET_PREFIX}{n}"``.
TARGET_PREFIX = "tool_call:r"

#: The subject role reference calls are keyed under in the ledger.
DEFAULT_SUBJECT_ROLE = "reference"

#: ``invalid_reason`` of a final reply the provider cut at the output budget.
TRUNCATED = "truncated"

#: The loop's ``tool_choice``, for every provider (see the module docstring).
LOOP_TOOL_CHOICE = "auto"

#: A base URL no one ever connects to: the loop's chat client is
#: :class:`DeferredChat`, so LfmTier's runtime is only asked for a string.
_NO_SERVER = "http://127.0.0.1:9/v1"


class NeedsReply(Exception):
    """Raised by :meth:`DeferredChat.complete` at the first uncached round.

    Carries the round's :class:`PendingCall`. LfmTier turns any exception
    from its chat client into a ``tier_error`` decline, so the loop driver
    never reads that decline: it reads :attr:`DeferredChat.pending` instead.
    """

    def __init__(self, pending: "PendingCall") -> None:
        super().__init__(f"{pending.request.case_id}: round {pending.round} has no answer yet")
        self.pending = pending


@dataclass(frozen=True)
class PendingCall:
    """One round's call the ledger has no answer for yet: send ``request``, record under ``key``."""

    key: str
    spec: CallSpec
    request: CallRequest
    round: int


@dataclass
class RoundResult:
    """What one :func:`run_round` pass found.

    ``finished`` maps a case id to its final record (the loop ended);
    ``pending`` holds the calls to submit before the next pass, at most one
    per case, in case order.
    """

    finished: dict[str, RawRecord] = field(default_factory=dict)
    pending: list[PendingCall] = field(default_factory=list)


# ---------------------------------------------------------------------------
# messages <-> CallRequest
# ---------------------------------------------------------------------------


def _neutral_turn(message: Mapping[str, Any]) -> dict:
    """One LfmTier history message in :attr:`CallRequest.history`'s neutral shape."""
    if message["role"] == "tool":
        return {
            "role": "tool",
            "tool_call_id": message["tool_call_id"],
            "content": message["content"],
        }
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": [
            {
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
            for call in message.get("tool_calls", [])
        ],
    }


def _round_of(messages: Sequence[Mapping[str, Any]]) -> int:
    """1 + the number of model turns already in *messages*."""
    return 1 + sum(1 for message in messages if message.get("role") == "assistant")


def content_hash(request: CallRequest) -> str:
    """sha256 over exactly what every adapter sends: system, messages, tools.

    Provider-native replay blocks are left out: they are derived from the
    cached answers themselves, never part of what identifies the call.
    """
    system, messages, tools, _labels = contract.canonical_content(request)
    for message in messages:
        message.pop("native", None)
    return prompt_hash(canonical_json({"system": system, "messages": messages, "tools": tools}))


def chat_reply(result: CallResult, spoken: ReplyText | None = None) -> ChatReply:
    """The ``ChatReply`` LfmTier reads for one adapter answer.

    Every adapter's Track A answer is the raw first tool call,
    ``{"name", "arguments"}`` (the request contract). A reply with no tool
    call reads *spoken* -- the reply's visible text from the adapter's
    ``reply_text`` -- the way ToolChat reads a local model's content: the
    text, plus any tool call printed in it. No *spoken* text (a structural
    refusal, a truncated reply, nothing said) is a reply with no tool call
    and no text, which LfmTier reads as "no usable output".
    """
    call = contract._as_tool_call(result.answer) if result.answer is not None else None
    if call is not None:
        name, arguments = call
        return ChatReply(text="", tool_calls=(ToolCall(name=name, arguments=dict(arguments)),))
    if spoken is None or spoken.truncated or not spoken.text:
        return ChatReply(text="", tool_calls=())
    return ChatReply(text=spoken.text, tool_calls=parse_raw_tool_calls(spoken.text))


def _tokens(usage: Mapping[str, int]) -> int | None:
    for key in ("output_tokens", "completion_tokens"):
        if isinstance(usage.get(key), int):
            return int(usage[key])
    return None


# ---------------------------------------------------------------------------
# the deferred chat client
# ---------------------------------------------------------------------------


class DeferredChat:
    """LfmTier's chat client for a reference model: ledger cache or a pending call.

    ToolChat-compatible (``complete(messages, tools) -> ChatReply`` and
    ``stop()``). One instance serves one ``select`` of one case.
    """

    def __init__(
        self,
        *,
        case: Case,
        provider: Provider,
        model: str,
        ledger: Ledger,
        params: Mapping[str, Any],
        subject_role: str = DEFAULT_SUBJECT_ROLE,
    ) -> None:
        self._case = case
        self._provider = provider
        self._model = model
        self._ledger = ledger
        # Keyed into every call's ledger spec too: "auto" is part of the call.
        self._params = {**dict(params), "tool_choice": LOOP_TOOL_CHOICE}
        self._subject_role = subject_role
        self._offered = None if case.candidates is None else tuple(case.candidates)
        #: round -> provider-native blocks of that round's cached answer.
        self._native: dict[int, dict] = {}
        #: The round the loop stopped at, when it did.
        self.pending: PendingCall | None = None
        #: Any failure other than a missing answer (never swallowed by LfmTier).
        self.error: BaseException | None = None
        #: Every replayed answer, in round order.
        self.results: list[CallResult] = []
        #: What measure.py records per reply (tokens; no logprobs from a provider).
        self.replies: list = []
        #: Every CallRequest built, in round order (the last one may be pending).
        self.requests: list[CallRequest] = []
        #: Whether the last replayed reply was cut at the output budget.
        self.last_truncated = False

    def build(self, messages: list[dict], tools: list[dict]) -> tuple[CallSpec, CallRequest, int]:
        """The ``(spec, request, round)`` LfmTier's *messages*/*tools* stand for."""
        round_no = _round_of(messages)
        history = []
        assistant_rounds = 0
        for message in messages[2:]:
            turn = _neutral_turn(message)
            if turn["role"] == "assistant":
                assistant_rounds += 1
                native = self._native.get(assistant_rounds)
                if native:
                    turn["native"] = native
            history.append(turn)
        params = {"tools": measure.restrict_tools(tools, self._offered), **self._params}
        request = CallRequest(
            case_id=self._case.id,
            split=self._case.split,
            case_text=messages[1]["content"],
            prompt=messages[0]["content"],
            # The loop, not the adapter, judges what a round's tool call
            # means (an inspection is not a final answer), so adapters only
            # apply structural checks here.
            offered_candidates=(),
            params=params,
            interface="tool_call",
            history=tuple(history),
        )
        spec = CallSpec(
            provider=self._provider.name,
            model=self._model,
            subject_role=self._subject_role,
            case_id=self._case.id,
            target=f"{TARGET_PREFIX}{round_no}",
            prompt_hash=content_hash(request),
            params=dict(self._params),
        )
        return spec, request, round_no

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        try:
            return self._complete(messages, tools)
        except NeedsReply:
            raise
        except BaseException as exc:  # noqa: BLE001 -- re-raised by the driver, never lost
            self.error = exc
            raise

    def _complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        spec, request, round_no = self.build(messages, tools)
        self.requests.append(request)
        key = self._ledger.register(spec)
        entry = self._ledger.entry(key)
        if entry.state not in (DONE, INVALID):
            self.pending = PendingCall(key=key, spec=spec, request=request, round=round_no)
            raise NeedsReply(self.pending)
        cached = self._ledger.cached(key)
        spoken = None
        if cached is None:
            # An invalid answer recorded without its bytes: nothing to replay.
            result = CallResult(
                case_id=self._case.id,
                outcome=Outcome.INVALID,
                reason=entry.reason or "invalid",
                provider=self._provider.name,
                model_id=self._model,
            )
        else:
            result = self._provider.result_from_raw(request, cached.raw)
            if result.reason != "refusal":
                spoken = self._provider.reply_text(cached.raw)
            native = getattr(self._provider, "native_turn", None)
            turn = native(cached.raw) if native is not None else None
            if turn:
                self._native[round_no] = turn
        reply = chat_reply(result, spoken)
        self.last_truncated = bool(spoken and spoken.truncated and not reply.tool_calls)
        self.results.append(result)
        self.replies.append(
            measure.ReplyRecord(
                reply=reply, tokens=_tokens(result.usage), think=False, logprobs=None, at=0.0
            )
        )
        return reply

    def stop(self) -> None:
        """Nothing in flight: a deferred chat never blocks."""


class _NoRuntime:
    """LfmTier's runtime when the model is a hosted reference: nothing to start."""

    def ensure(self) -> str:
        return _NO_SERVER

    def stop(self) -> None:
        """Nothing was started."""

    def status(self) -> str:
        return "reference model (no local runtime)"


# ---------------------------------------------------------------------------
# one case, one round
# ---------------------------------------------------------------------------


def _final_record(
    chat: DeferredChat, agent_request: Any, result: Any, *, provider: Provider, model: str
) -> RawRecord:
    select = measure.SelectRecord(
        request=agent_request, started=0.0, ended=0.0, result=result, replies=list(chat.replies)
    )
    outcome, operation, arguments, reason = measure._decided(None, select)
    last = chat.results[-1] if chat.results else None
    said_nothing = bool(chat.replies) and not (
        chat.replies[-1].reply.tool_calls or chat.replies[-1].reply.text.strip()
    )
    if outcome == "invalid" and said_nothing:
        if chat.last_truncated:
            reason = TRUNCATED
        elif last is not None and last.outcome is Outcome.INVALID:
            reason = last.reason  # the adapter's own reason (malformed, refusal, ...)
    return RawRecord.from_provider_answer(
        provider=provider.name,
        model=model,
        returned_model=last.returned_model if last is not None else None,
        interface="tool_call",
        outcome=outcome,
        operation=operation,
        arguments=arguments,
        candidates=None,
        invalid_reason=reason,
    )


def run_case(
    case: Case,
    *,
    provider: Provider,
    model: str,
    ledger: Ledger,
    snapshot: Mapping[str, object],
    platform: Platform,
    params: Mapping[str, Any] | None = None,
    subject_role: str = DEFAULT_SUBJECT_ROLE,
) -> RawRecord | PendingCall:
    """Replay *case* through a real LfmTier: its final record, or the round it waits on.

    *snapshot* is the recorded ground snapshot (``measure.load_snapshot``'s
    first value) and *platform* the tier platform the candidates ran with.
    *params* are the request knobs keyed into every call (default: the
    request contract's reasoning effort and output budget).
    """
    knobs = dict(
        params
        if params is not None
        else {
            "reasoning": contract.DEFAULT_REASONING,
            "max_output_tokens": contract.DEFAULT_MAX_OUTPUT_TOKENS,
        }
    )
    chat = DeferredChat(
        case=case,
        provider=provider,
        model=model,
        ledger=ledger,
        params=knobs,
        subject_role=subject_role,
    )
    tier = lfm.LfmTier(
        _NoRuntime(),
        platform,
        model=model,
        runner=measure.snapshot_runner(snapshot),
        chat_factory=lambda _base_url: chat,
    )
    agent_request, context = contract._agent_request_and_context(case)
    result = tier.select(agent_request, context)
    if chat.error is not None:
        raise chat.error
    if chat.pending is not None:
        return chat.pending
    return _final_record(chat, agent_request, result, provider=provider, model=model)


def run_round(
    cases: Sequence[Case],
    *,
    provider: Provider,
    model: str,
    ledger: Ledger,
    snapshot: Mapping[str, object],
    platform: Platform,
    params: Mapping[str, Any] | None = None,
    subject_role: str = DEFAULT_SUBJECT_ROLE,
) -> RoundResult:
    """One pass over *cases*: finished records, and this round's calls to send.

    Registers every pending call in the ledger (as ``pending``) but sends
    nothing: the caller submits ``pending`` -- sync, or as ONE batch -- and
    records the answers with :func:`record_results` before the next pass.
    """
    out = RoundResult()
    for case in cases:
        outcome = run_case(
            case,
            provider=provider,
            model=model,
            ledger=ledger,
            snapshot=snapshot,
            platform=platform,
            params=params,
            subject_role=subject_role,
        )
        if isinstance(outcome, PendingCall):
            out.pending.append(outcome)
        else:
            out.finished[case.id] = outcome
    return out


def cached_response(result: CallResult) -> CachedResponse:
    """The ledger's cache record for one provider answer."""
    return CachedResponse(
        raw=result.raw,
        model_id=result.returned_model or result.model_id or "",
        response_id=result.response_id,
        usage=dict(result.usage),
    )


def record_results(
    ledger: Ledger, pending: Sequence[PendingCall], results: Sequence[CallResult]
) -> list[CallResult]:
    """Record a round's answers under their keys; return the ones left unrecorded.

    A round holds at most one call per case, so a result finds its call by
    ``case_id``. ``ok`` answers are recorded done and ``invalid`` ones
    invalid (with their bytes, so they are never paid for twice); an
    infrastructure ``pending`` result is returned untouched for the runner's
    own stop handling.
    """
    by_case = {call.request.case_id: call for call in pending}
    if len(by_case) != len(pending):
        raise ValueError("a round holds at most one pending call per case")
    left: list[CallResult] = []
    for result in results:
        call = by_case.get(result.case_id)
        if call is None:
            raise ValueError(f"no pending call for case {result.case_id!r} in this round")
        if result.outcome is Outcome.OK:
            ledger.record_done(call.key, cached_response(result))
        elif result.outcome is Outcome.INVALID:
            ledger.mark_invalid(call.key, result.reason or "invalid", cached_response(result))
        else:
            left.append(result)
    return left


def replay_messages(request: CallRequest) -> list[dict]:
    """The LfmTier message list a loop ``CallRequest`` was built from (for audits/tests)."""
    system, messages, _tools, _labels = contract.canonical_content(request)
    out: list[dict] = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] == "assistant":
            out.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in message.get("tool_calls", [])
                    ],
                }
            )
        else:
            message.pop("native", None)
            out.append(message)
    return out
