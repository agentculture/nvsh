"""Provider protocol and base class for tool_jev reference-model calls.

Every concrete provider adapter (OpenAI, Anthropic, OpenRouter,
build.nvidia.com, a local OpenAI-compatible server, ...) implements the
:class:`Provider` protocol. Adapters subclass :class:`BaseProvider` rather
than implementing the protocol from scratch so two rules cannot be skipped
by an individual adapter:

1. **Redaction is not optional.** Every outgoing case text and prompt
   passes through ``nvsh.redact.redact`` before an adapter's own transport
   code ever sees it (issue #64, c35/h25: "cases are redacted with
   nvsh/redact.py before any leave the machine").
2. **The held-out guard runs before any network call.** A request whose
   split tag is ``heldout`` or ``heldout-mc`` is refused in this base
   class, so no adapter subclass can accidentally send a sealed case (c36
   / h26: "the provider layer refuses to send any case whose source split
   is held-out, checked by split tag before any network call").

``BaseProvider`` enforces both by making ``submit_sync`` / ``submit_batch``
non-overridable entry points that call a private ``_send_*`` method the
subclass implements; the public methods do the guard + redaction, then
hand a *redacted* request to the subclass.

Batch crash recovery: the ledger issues a submit ref
(``Ledger.begin_submit``) *before* the provider call. ``submit_batch``
takes that ref and the adapter stores it with the batch on the provider's
side (batch metadata, or a custom-id prefix), so after a crash between
provider acceptance and ``Ledger.mark_submitted`` the runner calls
``find_batch(ref)``: a found handle is re-attached, ``None`` means the
batch never reached the provider and the keys can be abandoned and resent.

Reading provider API keys is deliberately *not* this module's job for
literal values: :func:`read_api_key` only ever accepts an **environment
variable name** (never a literal key) and reads it from ``os.environ`` at
call time, so a concrete adapter cannot be constructed with a hard-coded
key even if it tried — "keys read only from environment variable names
given by the manifest" (c13/h12/h15).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from nvsh.redact import redact

from .errors import Outcome

#: The two answer interfaces a request can use (tool call vs. constrained choice).
INTERFACES = frozenset({"tool_call", "choice"})

#: Split tags that must never reach a network call. Both spellings from the
#: issue-64 spec: the plain held-out set and its missing-candidate slice.
HELDOUT_SPLIT_TAGS = frozenset({"heldout", "heldout-mc"})


class HeldoutSplitRefused(Exception):
    """Raised when a request's split tag is heldout/heldout-mc.

    Raised before any network call — the caller never reaches the
    subclass's ``_send_sync``/``_send_batch``.
    """


class BatchLookupUnresolved(Exception):
    """Raised by ``find_batch`` when the provider cannot say either way.

    ``None`` from ``find_batch`` means "no batch was ever accepted under this
    ref" and authorizes a resubmit. This exception means the opposite is
    *possible* but cannot be confirmed right now (a batch that could be ours
    is still processing and hides its custom ids, a results download failed,
    or the listing ran past its page bound). A runner must never resubmit on
    it: it stops and asks the operator (plan risk r6).
    """


class MissingProviderKey(Exception):
    """Raised by :func:`read_api_key` when the named env var is unset/empty."""


def read_api_key(env_var_name: str) -> str:
    """Read a provider API key from the environment variable named ``env_var_name``.

    Concrete adapters call this instead of accepting a literal key
    argument, so a key can only ever come from the process environment
    (populated by e.g. ``grant run --inject``), never from a committed
    literal or a manifest value itself.
    """
    value = os.environ.get(env_var_name)
    if not value:
        raise MissingProviderKey(
            f"environment variable {env_var_name!r} is not set; provider keys "
            "are read only from environment variable names, never literals"
        )
    return value


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can do, so callers don't have to special-case names.

    - ``logprobs``: the provider can return per-candidate token
      probabilities (needed for calibration metrics; Anthropic returns
      none — issue #64 c-"record capability per provider").
    - ``batch``: the provider has an async batch API (``submit_batch`` /
      ``poll_batch`` / ``fetch_batch``) distinct from a synchronous call.
    - ``reasoning``: the provider accepts a reasoning-effort/thinking
      parameter.
    """

    logprobs: bool
    batch: bool
    reasoning: bool


#: The two roles a :attr:`CallRequest.history` turn may have.
HISTORY_ROLES = frozenset({"assistant", "tool"})


def _check_history(history: tuple) -> None:
    """Refuse a history turn that is not in the neutral shape (see CallRequest)."""
    for index, turn in enumerate(history):
        where = f"history[{index}]"
        if not isinstance(turn, dict) or turn.get("role") not in HISTORY_ROLES:
            raise ValueError(f"{where} must be a dict with role in {sorted(HISTORY_ROLES)}")
        if turn["role"] == "tool":
            if not isinstance(turn.get("tool_call_id"), str) or not turn["tool_call_id"]:
                raise ValueError(f"{where}: a tool turn needs a non-empty tool_call_id")
            if not isinstance(turn.get("content"), str):
                raise ValueError(f"{where}: a tool turn's content must be a string")
            continue
        if not isinstance(turn.get("content", ""), str):
            raise ValueError(f"{where}: an assistant turn's content must be a string")
        calls = turn.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError(f"{where}: tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict) or not all(
                isinstance(call.get(key), str) for key in ("id", "name", "arguments")
            ):
                raise ValueError(f"{where}: each tool call needs string id, name and arguments")


@dataclass(frozen=True)
class CallRequest:
    """One planned call: a case routed at a specific provider/model.

    ``split`` is the case's split tag (e.g. ``"test"``, ``"test-mc"``,
    ``"heldout"``, ``"heldout-mc"``) and is checked by the held-out guard
    before anything else happens. ``case_text``, ``prompt`` and every text
    field of ``history`` carry case content (or machine output) and are
    redacted before a subclass ever sees them.

    ``history`` (deviation d1) is the ordered conversation *after* the
    system and user messages: the prior rounds of a multi-round tool-use
    loop, in a provider-neutral shape every adapter renders natively::

        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "name": "service_status", "arguments": "<JSON text>"}]}
        {"role": "tool", "tool_call_id": "call_0", "content": "<tool result text>"}

    ``arguments`` is the exact JSON text the loop echoed (a string, as
    ``nvsh.tiers.lfm`` sends it). An assistant turn may also carry
    ``"native": {"<provider kind>": [...]}``: that provider's own content
    blocks for the same turn, replayed verbatim by the matching adapter only
    (e.g. Anthropic thinking blocks, which must come back unchanged); the
    *content* every adapter sends is still the neutral fields. Empty for a
    single-turn request.
    """

    case_id: str
    split: str
    case_text: str
    prompt: str = ""
    offered_candidates: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    interface: str = "tool_call"
    history: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        if self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {sorted(INTERFACES)}: {self.interface!r}")
        if not isinstance(self.history, tuple):
            object.__setattr__(self, "history", tuple(self.history))
        _check_history(self.history)


@dataclass(frozen=True)
class CallResult:
    """The outcome of one call, whether synchronous or fetched from a batch.

    Beyond the classified answer it carries everything downstream needs
    without adapter-specific access:

    - ``candidates``: the full candidate -> probability distribution, in the
      order the provider/offered set gave it, **only** when the provider
      actually returned logprobs; ``None`` otherwise. Never estimated.
    - ``raw``: the exact provider response bytes (what the ledger cache
      stores as ``CachedResponse.raw``).
    - ``response_id``: the provider's own id for this response.
    - ``returned_model``: the model id the provider *reported* (may differ
      from the requested ``model_id``, e.g. a dated snapshot).
    - ``usage``: token counts where reported (``input_tokens``,
      ``output_tokens``, ``reasoning_tokens``, ...), ints only.
    - ``interface``: ``"tool_call"`` or ``"choice"``.
    """

    case_id: str
    outcome: Outcome
    answer: str | None = None
    reason: str = ""
    provider: str = ""
    model_id: str | None = None
    candidates: dict[str, float] | None = None
    raw: bytes = b""
    response_id: str = ""
    returned_model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    interface: str = "tool_call"

    def __post_init__(self) -> None:
        if not isinstance(self.raw, bytes):
            raise TypeError("CallResult.raw must be bytes (the exact provider response)")
        for name, value in self.usage.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"usage[{name!r}] must be an int, got {value!r}")
        if self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {sorted(INTERFACES)}: {self.interface!r}")
        if self.candidates is not None:
            for name, value in self.candidates.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"candidates[{name!r}] must be a probability, got {value!r}")


@dataclass(frozen=True)
class BatchHandle:
    """Opaque handle returned by ``submit_batch``, re-attachable across a resume.

    ``submit_ref`` is the ledger's submit ref the batch was submitted under
    (empty only for a handle built by hand).
    """

    batch_id: str
    provider: str
    submit_ref: str = ""


@dataclass(frozen=True)
class BatchStatus:
    """The state of a submitted batch as reported by ``poll_batch``."""

    batch_id: str
    complete: bool
    expired: bool = False


@dataclass(frozen=True)
class ReplyText:
    """The visible text of one cached answer, as :meth:`Provider.reply_text` reads it.

    ``text`` is what the model said to the user: never thinking blocks,
    reasoning items or ``reasoning_content``, and with any inline
    ``<think>...</think>`` reasoning removed (:func:`visible_text`).
    ``truncated`` is the provider's own signal that the answer was cut at the
    output budget (OpenAI ``status: incomplete``, Anthropic ``stop_reason:
    max_tokens``, chat-completions ``finish_reason: length``); a truncated
    reply is never read as a finished explanation.
    """

    text: str
    truncated: bool = False


_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def visible_text(text: object) -> str:
    """*text* without inline ``<think>...</think>`` reasoning, stripped.

    A few open models served over chat completions print their reasoning
    inline instead of in a separate field; that is reasoning, not the reply.
    An unclosed ``<think>`` runs to the end of the text.
    """
    if not isinstance(text, str):
        return ""
    return _THINK_BLOCK.sub("", text).strip()


#: ``CallRequest.params["tool_choice"]`` values an adapter honours over its
#: own forced/auto default.
TOOL_CHOICES = frozenset({"auto", "required"})


def tool_choice_forced(request: "CallRequest", default: bool) -> bool:
    """Whether *request* forces a tool call: its ``tool_choice`` param, else *default*.

    The Track A loop sends ``"auto"`` for every provider, as the candidates'
    own chat client did (it sends no ``tool_choice``, so the server default
    ``auto`` applied): a reference may then answer in plain words, exactly
    as a candidate could.
    """
    choice = (request.params or {}).get("tool_choice")
    if choice is None:
        return default
    if choice not in TOOL_CHOICES:
        raise ValueError(f"tool_choice must be one of {sorted(TOOL_CHOICES)}: {choice!r}")
    return choice == "required"


@runtime_checkable
class Provider(Protocol):
    """The provider-agnostic surface every adapter (and the fake) presents.

    The runner (a later task) and the ledger (t9) only ever call these
    five methods plus read ``name``/``capabilities`` — never anything
    provider-specific.
    """

    name: str
    capabilities: ProviderCapabilities

    def submit_sync(self, request: CallRequest) -> CallResult: ...

    def submit_batch(self, requests: list[CallRequest], submit_ref: str) -> BatchHandle: ...

    def find_batch(self, submit_ref: str) -> BatchHandle | None:
        """The batch accepted under *submit_ref*, ``None`` if none ever was.

        Raises :class:`BatchLookupUnresolved` when the provider cannot tell
        (never resubmit on it).
        """

    def poll_batch(self, handle: BatchHandle) -> BatchStatus: ...

    def fetch_batch(self, handle: BatchHandle) -> list[CallResult]: ...

    def result_from_raw(self, request: CallRequest, raw: bytes) -> CallResult:
        """Re-read a cached ``CallResult.raw`` for *request* exactly as a fresh answer."""

    def reply_text(self, raw: bytes) -> ReplyText:
        """The visible text of a cached answer (never reasoning) and whether it was cut."""


def _redact_text(text: str) -> str:
    """Round-trip ``text`` through ``nvsh.redact.redact`` (bytes in/out)."""
    return redact(text.encode("utf-8", errors="surrogateescape")).decode(
        "utf-8", errors="surrogateescape"
    )


def _redact_value(value):
    """Every string inside a decoded JSON value redacted; shape and keys kept."""
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    return value


def _redact_arguments(arguments: str) -> str:
    """Redact a tool call's JSON arguments text without breaking its JSON.

    Redacting the raw text could let a rule swallow a closing quote, so the
    JSON is decoded, every string value redacted, and re-encoded -- but only
    when that changed something, so clean arguments stay byte-identical to
    what the loop echoed. Text that is not JSON is redacted as text.
    """
    try:
        decoded = json.loads(arguments)
    except ValueError:
        return _redact_text(arguments)
    cleaned = _redact_value(decoded)
    return arguments if cleaned == decoded else json.dumps(cleaned)


#: Anthropic content-block keys that must come back byte-for-byte (the
#: thinking signature binds them); they are never rewritten by redaction.
_NATIVE_VERBATIM_BLOCKS = frozenset({"thinking", "redacted_thinking"})


def _redact_native_block(block):
    if not isinstance(block, dict) or block.get("type") in _NATIVE_VERBATIM_BLOCKS:
        return block
    return _redact_value(block)


def _redacted_turn(turn: dict) -> dict:
    out = dict(turn)
    if isinstance(out.get("content"), str):
        out["content"] = _redact_text(out["content"])
    if out.get("role") == "assistant":
        out["tool_calls"] = [
            {
                **call,
                "name": _redact_text(call["name"]),
                "arguments": _redact_arguments(call["arguments"]),
            }
            for call in out.get("tool_calls", [])
        ]
        native = out.get("native")
        if isinstance(native, dict):
            out["native"] = {
                kind: [_redact_native_block(block) for block in blocks]
                for kind, blocks in native.items()
            }
    return out


def _redacted_request(request: CallRequest) -> CallRequest:
    return dataclasses.replace(
        request,
        case_text=_redact_text(request.case_text),
        prompt=_redact_text(request.prompt),
        history=tuple(_redacted_turn(turn) for turn in request.history),
    )


def _require_submit_ref(submit_ref: str) -> None:
    if not isinstance(submit_ref, str) or not submit_ref.strip():
        raise ValueError("submit_ref must be the non-empty ref from Ledger.begin_submit")


class BaseProvider(ABC):
    """Common enforcement every concrete provider adapter inherits.

    Subclasses implement ``_send_sync``, ``_send_batch``, ``_find_batch``,
    ``_check_batch`` and ``_collect_batch`` — the real transport code. They
    never override ``submit_sync`` / ``submit_batch`` / ``find_batch`` /
    ``poll_batch`` / ``fetch_batch`` themselves, which keeps the held-out
    guard and the redaction choke point in one place, unskippable by any one
    adapter.
    """

    name: str
    capabilities: ProviderCapabilities

    def _refuse_heldout(self, request: CallRequest) -> None:
        if request.split in HELDOUT_SPLIT_TAGS:
            raise HeldoutSplitRefused(
                f"{self.name}: refusing case {request.case_id!r} "
                f"(split={request.split!r}) before any network call"
            )

    def submit_sync(self, request: CallRequest) -> CallResult:
        self._refuse_heldout(request)
        return self._send_sync(_redacted_request(request))

    def submit_batch(self, requests: list[CallRequest], submit_ref: str) -> BatchHandle:
        _require_submit_ref(submit_ref)
        for one in requests:
            self._refuse_heldout(one)
        return self._send_batch([_redacted_request(one) for one in requests], submit_ref)

    def find_batch(self, submit_ref: str) -> BatchHandle | None:
        _require_submit_ref(submit_ref)
        return self._find_batch(submit_ref)

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        return self._check_batch(handle)

    def fetch_batch(self, handle: BatchHandle) -> list[CallResult]:
        return self._collect_batch(handle)

    def result_from_raw(self, request: CallRequest, raw: bytes) -> CallResult:
        """Re-read a cached answer (``CallResult.raw`` / ``CachedResponse.raw``).

        What a warm rerun or the multi-round loop (``track_a_loop``) uses to
        turn the ledger's cached bytes back into the same ``CallResult`` the
        adapter returned when the answer first arrived -- parsed by the
        adapter's own code, never re-derived by the caller. Sends nothing.
        """
        raise NotImplementedError(f"{self.name}: this provider cannot re-read a cached answer")

    def reply_text(self, raw: bytes) -> ReplyText:
        """The visible text of a cached answer and whether the provider cut it short.

        What the Track A loop hands ``LfmTier`` when a reply carries no tool
        call (plain words are an explanation, as for a candidate), and what a
        runner reads for a judge's free-text answer. Sends nothing.
        """
        raise NotImplementedError(f"{self.name}: this provider cannot read a reply's text")

    def native_turn(self, raw: bytes) -> dict | None:
        """Provider-native content of a cached answer, for :attr:`CallRequest.history`.

        ``{"<provider kind>": [blocks]}`` when the provider needs its own
        blocks back on a later turn (Anthropic thinking blocks), else ``None``.
        """
        del raw
        return None

    @abstractmethod
    def _send_sync(self, request: CallRequest) -> CallResult:
        """Send one already-guarded, already-redacted request."""

    @abstractmethod
    def _send_batch(self, requests: list[CallRequest], submit_ref: str) -> BatchHandle:
        """Submit already-guarded, already-redacted requests as a batch.

        Must store ``submit_ref`` with the batch on the provider's side
        (batch metadata or a custom-id prefix) so ``_find_batch`` can find
        it after a crash, and return a handle whose ``submit_ref`` is set.
        """

    @abstractmethod
    def _find_batch(self, submit_ref: str) -> BatchHandle | None:
        """Look up a batch the provider accepted under ``submit_ref``; None if none."""

    @abstractmethod
    def _check_batch(self, handle: BatchHandle) -> BatchStatus:
        """Report whether a submitted batch is complete/expired."""

    @abstractmethod
    def _collect_batch(self, handle: BatchHandle) -> list[CallResult]:
        """Fetch results for a completed batch."""
