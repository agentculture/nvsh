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
import os
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


@dataclass(frozen=True)
class CallRequest:
    """One planned call: a case routed at a specific provider/model.

    ``split`` is the case's split tag (e.g. ``"test"``, ``"test-mc"``,
    ``"heldout"``, ``"heldout-mc"``) and is checked by the held-out guard
    before anything else happens. ``case_text`` and ``prompt`` are the two
    fields that carry case content and are redacted before a subclass ever
    sees them.
    """

    case_id: str
    split: str
    case_text: str
    prompt: str = ""
    offered_candidates: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    interface: str = "tool_call"

    def __post_init__(self) -> None:
        if self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {sorted(INTERFACES)}: {self.interface!r}")


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

    def find_batch(self, submit_ref: str) -> BatchHandle | None: ...

    def poll_batch(self, handle: BatchHandle) -> BatchStatus: ...

    def fetch_batch(self, handle: BatchHandle) -> list[CallResult]: ...


def _redact_text(text: str) -> str:
    """Round-trip ``text`` through ``nvsh.redact.redact`` (bytes in/out)."""
    return redact(text.encode("utf-8", errors="surrogateescape")).decode(
        "utf-8", errors="surrogateescape"
    )


def _redacted_request(request: CallRequest) -> CallRequest:
    return dataclasses.replace(
        request,
        case_text=_redact_text(request.case_text),
        prompt=_redact_text(request.prompt),
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
