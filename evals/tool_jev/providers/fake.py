"""Scriptable fake provider for tool_jev tests (issue #64, t10).

``FakeProvider`` implements :class:`~evals.tool_jev.providers.base.Provider`
entirely in-process: no network, no subprocess, nothing executed. Every
call it resolves consumes the next entry from a script the test supplies,
so a test can drive a provider through the exact sequence of outcomes an
acceptance criterion needs (an answer, a refusal, a malformed reply, a 402,
a 429, a timeout, an expired batch) without touching a real API.

Every request the base class hands ``FakeProvider`` has already passed the
held-out guard and ``nvsh.redact.redact`` (see ``base.BaseProvider``), and
``FakeProvider`` records each one verbatim in ``received`` so a test can
assert on exactly what a provider adapter would have sent — including that
a hostile string embedded in case text or a scripted answer is stored as
plain data and never executed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .base import (
    BaseProvider,
    BatchHandle,
    BatchStatus,
    CallRequest,
    CallResult,
    ProviderCapabilities,
)
from .errors import Classification, classify_answer, classify_transport

#: Scripted outcome kinds FakeProvider understands.
ANSWER_KINDS = frozenset({"answer"})
ANSWER_FAILURE_KINDS = frozenset({"refusal", "malformed"})
INFRA_KINDS = frozenset({"402", "429", "timeout", "network", "reset", "batch_expired"})
ALL_KINDS = ANSWER_KINDS | ANSWER_FAILURE_KINDS | INFRA_KINDS

_STATUS_BY_KIND = {"402": 402, "429": 429, "timeout": 408}
_ERROR_TYPE_BY_KIND = {
    "network": "network_error",
    "reset": "connection_reset",
    "batch_expired": "batch_expired",
}


@dataclass(frozen=True)
class ScriptedOutcome:
    """One scripted outcome. ``kind`` is one of :data:`ALL_KINDS`."""

    kind: str
    answer: str | None = None


class FakeProviderError(Exception):
    """Raised when a scripted infrastructure outcome (402/429/timeout/...) fires.

    Carries the :class:`~evals.tool_jev.providers.errors.Classification`
    (always ``outcome=PENDING, stop=True``) and the provider name, so a
    caller's run loop can build the clean-stop message with
    ``errors.stop_message`` without re-deriving anything.
    """

    def __init__(self, classification: Classification, provider: str, case_id: str) -> None:
        self.classification = classification
        self.provider = provider
        self.case_id = case_id
        super().__init__(f"{provider}: {classification.reason} on case {case_id!r}")


class ScriptExhausted(AssertionError):
    """Raised when a FakeProvider call runs out of scripted outcomes."""


def _coerce(entry) -> ScriptedOutcome:
    if isinstance(entry, ScriptedOutcome):
        return entry
    if isinstance(entry, tuple):
        kind, answer = entry
        return ScriptedOutcome(kind=kind, answer=answer)
    return ScriptedOutcome(kind=entry)


class FakeProvider(BaseProvider):
    """In-process, fully scriptable :class:`Provider` double.

    Parameters
    ----------
    name:
        Provider name reported on every :class:`CallResult` and in stop
        messages.
    capabilities:
        Defaults to ``logprobs=True, batch=True, reasoning=False`` — a
        fully-featured provider — override for a test that needs a
        provider missing one capability (e.g. no logprobs, like
        Anthropic).
    script:
        Outcomes consumed one per call, in order, by both ``submit_sync``
        and ``fetch_batch``. Each entry is a :class:`ScriptedOutcome`, a
        ``(kind, answer)`` tuple, or a bare kind string (for
        answer-less kinds such as ``"402"``). Use :meth:`queue` to append
        more after construction (e.g. to model "topping up" a provider
        mid-test).
    provider_kind:
        Which :mod:`errors` classification table to classify infra
        outcomes against (``"openai"``, ``"anthropic"`` or
        ``"openai_compat"``, all identical today). Defaults to
        ``"openai_compat"``.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        capabilities: ProviderCapabilities | None = None,
        script: list | None = None,
        provider_kind: str = "openai_compat",
    ) -> None:
        self.name = name
        self.capabilities = capabilities or ProviderCapabilities(
            logprobs=True, batch=True, reasoning=False
        )
        self.provider_kind = provider_kind
        self._script: deque = deque(_coerce(entry) for entry in (script or []))
        #: Every request handed to this provider, post-guard, post-redaction.
        self.received: list[CallRequest] = []
        self._batches: dict[str, list[CallRequest]] = {}
        self._batch_counter = 0

    def queue(self, kind: str, answer: str | None = None) -> None:
        """Append one more scripted outcome (e.g. to simulate a top-up)."""
        self._script.append(ScriptedOutcome(kind=kind, answer=answer))

    def remaining_script(self) -> int:
        return len(self._script)

    def _next_outcome(self, case_id: str) -> ScriptedOutcome:
        if not self._script:
            raise ScriptExhausted(
                f"{self.name}: fake provider script exhausted before case {case_id!r}"
            )
        return self._script.popleft()

    def _resolve(self, request: CallRequest, outcome: ScriptedOutcome) -> CallResult:
        if outcome.kind in ANSWER_KINDS:
            classification = classify_answer(outcome.answer, request.offered_candidates or None)
            return CallResult(
                case_id=request.case_id,
                outcome=classification.outcome,
                answer=outcome.answer,
                reason=classification.reason,
                provider=self.name,
            )
        if outcome.kind == "refusal":
            classification = classify_answer(outcome.answer or "I cannot help with that request.")
            return CallResult(
                case_id=request.case_id,
                outcome=classification.outcome,
                answer=outcome.answer,
                reason=classification.reason,
                provider=self.name,
            )
        if outcome.kind == "malformed":
            classification = classify_answer(outcome.answer, malformed=True)
            return CallResult(
                case_id=request.case_id,
                outcome=classification.outcome,
                answer=outcome.answer,
                reason=classification.reason,
                provider=self.name,
            )
        if outcome.kind in _STATUS_BY_KIND:
            classification = classify_transport(
                self.provider_kind, status_code=_STATUS_BY_KIND[outcome.kind]
            )
        elif outcome.kind in _ERROR_TYPE_BY_KIND:
            classification = classify_transport(
                self.provider_kind, error_type=_ERROR_TYPE_BY_KIND[outcome.kind]
            )
        else:
            raise ValueError(f"unknown scripted outcome kind: {outcome.kind!r}")
        raise FakeProviderError(classification, self.name, request.case_id)

    # -- BaseProvider hooks --------------------------------------------------

    def _send_sync(self, request: CallRequest) -> CallResult:
        self.received.append(request)
        outcome = self._next_outcome(request.case_id)
        return self._resolve(request, outcome)

    def _send_batch(self, requests: list[CallRequest]) -> BatchHandle:
        self._batch_counter += 1
        batch_id = f"{self.name}-batch-{self._batch_counter}"
        self._batches[batch_id] = list(requests)
        self.received.extend(requests)
        return BatchHandle(batch_id=batch_id, provider=self.name)

    def _check_batch(self, handle: BatchHandle) -> BatchStatus:
        return BatchStatus(batch_id=handle.batch_id, complete=True)

    def _collect_batch(self, handle: BatchHandle) -> list[CallResult]:
        requests = self._batches.get(handle.batch_id, [])
        results = []
        for request in requests:
            outcome = self._next_outcome(request.case_id)
            results.append(self._resolve(request, outcome))
        return results
