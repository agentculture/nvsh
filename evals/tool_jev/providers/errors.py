"""Error classification for tool_jev reference-provider calls.

Operator decision (issue #64, c45/h31): only a **model-answer failure** —
refusal, malformed or empty output, or an answer outside the offered
candidate set — is scored ``invalid`` and counted in the denominator. Any
**infrastructure stop** (HTTP 402 / insufficient credit or quota, a
provider or local budget cap reached, HTTP 429 rate limiting, a timeout,
network loss, a machine reset, an expired batch) leaves the call
``pending`` and the run stops cleanly, naming the provider and how many
calls remain — it is never counted invalid and never silently dropped.

Two kinds of stop are kept apart (``Classification.retryable``):

- **transient** (402, 408, 429, 5xx, timeouts, network loss, reset, an
  expired batch): continuing the run unchanged may succeed once the
  condition clears (top-up, back-off, reconnect).
- **request rejected** (HTTP 400/401/403/404/422, or a named
  ``invalid_request`` / ``auth_failed`` / ``model_not_found`` /
  ``unsupported_parameter``): the provider refused *this request as
  built*, so continuing unchanged hits the same wall. Reason family
  ``request_rejected:<detail>``, ``retryable=False``; the operator must fix
  the manifest or params first. Still ``pending``, never ``invalid`` — it
  is not a model failure and never enters the denominator.

A refusal is only ever recognised **structurally**: the adapter sees the
provider's own refusal signal (OpenAI ``message.refusal``, Anthropic
``stop_reason == "refusal"``) and passes ``refused=True`` to
:func:`classify_answer`. Answer text is never searched for refusal-sounding
phrases — "I cannot determine X from this output; run Y" is a valid
explanation, not a refusal.

This module holds the classification tables (one per provider *kind*:
``openai``, ``anthropic``, ``openai_compat``) plus the answer-level
classifier. It has no network code and no provider-specific transport
logic — concrete adapters (built in later tasks) translate their own SDK
exceptions/HTTP responses into the ``status_code`` / ``error_type``
vocabulary used here and call :func:`classify_transport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Outcome(Enum):
    """The three things a planned call can end up as."""

    OK = "ok"
    INVALID = "invalid"
    PENDING = "pending"


@dataclass(frozen=True)
class Classification:
    """The result of classifying one error or one answer.

    ``stop`` is set on infrastructure classifications: the run should stop
    cleanly rather than keep spending calls into the same wall.

    ``retryable`` says whether continuing the run *unchanged* can succeed:
    ``True`` for a transient stop, ``False`` for a rejected request (reason
    ``request_rejected:<detail>``) and for answer classifications, which are
    final.
    """

    outcome: Outcome
    reason: str
    stop: bool = False
    retryable: bool = False

    @property
    def rejected(self) -> bool:
        """True when the provider rejected the request as built (not transient)."""
        return self.reason.startswith(REQUEST_REJECTED_PREFIX)


#: Reason prefix for permanent request errors (bad params, auth, unknown model).
REQUEST_REJECTED_PREFIX = "request_rejected:"


def _pending(reason: str) -> Classification:
    """A transient infrastructure stop: continuing unchanged may succeed."""
    return Classification(Outcome.PENDING, reason, stop=True, retryable=True)


def rejected(reason: str) -> Classification:
    """A request-level rejection with *reason* (not retryable until something changes)."""
    return _rejected(reason)


def _rejected(detail: str) -> Classification:
    """A permanent request error: stop, stay pending, fix the request first."""
    return Classification(
        Outcome.PENDING, f"{REQUEST_REJECTED_PREFIX}{detail}", stop=True, retryable=False
    )


# ---------------------------------------------------------------------------
# Provider-kind classification tables.
#
# Every provider kind shares the same infrastructure vocabulary today (HTTP
# status codes and named error types mean the same thing everywhere calls
# go over HTTPS), so the three tables start identical. They are kept as
# separate dict instances (not one shared reference) per provider kind so a
# later task can special-case one provider (e.g. Anthropic's own overload
# error shape) without touching the others.
# ---------------------------------------------------------------------------

_STATUS_TABLE: dict[int, Classification] = {
    400: _rejected("bad_request"),
    401: _rejected("auth_failed"),
    402: _pending("insufficient_credit"),
    403: _rejected("forbidden"),
    404: _rejected("not_found"),
    408: _pending("timeout"),
    422: _rejected("unprocessable"),
    429: _pending("rate_limited"),
    500: _pending("network_or_server_error"),
    502: _pending("network_or_server_error"),
    503: _pending("network_or_server_error"),
    504: _pending("timeout"),
}

_ERROR_TYPE_TABLE: dict[str, Classification] = {
    "insufficient_quota": _pending("insufficient_credit"),
    "budget_cap_reached": _pending("budget_cap_reached"),
    "rate_limited": _pending("rate_limited"),
    "timeout": _pending("timeout"),
    "network_error": _pending("network_loss"),
    "connection_reset": _pending("machine_reset"),
    "batch_expired": _pending("expired_batch"),
    "invalid_request": _rejected("invalid_request"),
    "auth_failed": _rejected("auth_failed"),
    "model_not_found": _rejected("model_not_found"),
    "unsupported_parameter": _rejected("unsupported_parameter"),
}

#: One classification table per provider *kind*. Keyed on the kind name a
#: concrete adapter declares (not the adapter's own module name), so
#: several adapters that share a transport shape (e.g. two OpenAI-compatible
#: local servers) can share a kind.
PROVIDER_ERROR_TABLES: dict[str, dict[str, dict]] = {
    "openai": {"status": dict(_STATUS_TABLE), "error_type": dict(_ERROR_TYPE_TABLE)},
    "anthropic": {"status": dict(_STATUS_TABLE), "error_type": dict(_ERROR_TYPE_TABLE)},
    "openai_compat": {"status": dict(_STATUS_TABLE), "error_type": dict(_ERROR_TYPE_TABLE)},
}


def classify_transport(
    provider_kind: str,
    *,
    status_code: int | None = None,
    error_type: str | None = None,
) -> Classification:
    """Classify an infrastructure condition for ``provider_kind``.

    Looks the condition up in that provider kind's table: a recognised
    named error type first (it is the more specific signal, e.g. a 400 that
    says ``unsupported_parameter``), then the status code. An unrecognized
    condition is still classified ``pending`` and retryable (never guessed
    into ``invalid``) so an operator sees an unfamiliar stop rather than
    losing the call to the denominator.
    """
    table = PROVIDER_ERROR_TABLES.get(provider_kind)
    if table is None:
        raise KeyError(f"unknown provider kind: {provider_kind!r}")
    if error_type is not None and error_type in table["error_type"]:
        return table["error_type"][error_type]
    if status_code is not None and status_code in table["status"]:
        return table["status"][status_code]
    unknown_reason = error_type or (f"http_{status_code}" if status_code is not None else "unknown")
    return _pending(f"unrecognized_infra_condition:{unknown_reason}")


def classify_answer(
    answer: str | None,
    offered_candidates: tuple[str, ...] | None = None,
    *,
    malformed: bool = False,
    refused: bool = False,
) -> Classification:
    """Classify a model answer.

    Only these four shapes are ``invalid`` (and counted in the
    denominator): malformed output, a refusal, empty output, or an answer
    outside the offered candidate set. Anything else is ``ok``.

    ``refused`` must come from the provider's *structural* refusal signal
    (OpenAI ``message.refusal`` present, Anthropic ``stop_reason ==
    "refusal"``); the answer text itself is never searched for
    refusal-sounding phrases.
    """
    if malformed:
        return Classification(Outcome.INVALID, "malformed")
    if refused:
        return Classification(Outcome.INVALID, "refusal")
    if answer is None or not str(answer).strip():
        return Classification(Outcome.INVALID, "empty_answer")
    if offered_candidates and answer not in offered_candidates:
        return Classification(Outcome.INVALID, "outside_offered_set")
    return Classification(Outcome.OK, "answer")


def stop_message(provider: str, classification: Classification, remaining: int) -> str:
    """Render the clean-stop message naming the provider and remaining calls.

    Used by the (per-task) run loop when a :class:`Classification` with
    ``stop=True`` comes back from a call, so every provider's stop message
    has the same shape. The message says which kind of stop it is: a
    rejected request (fix the manifest/params before continuing) or a
    transient condition (continue once it clears).
    """
    if classification.rejected or not classification.retryable:
        return (
            f"{provider}: request rejected ({classification.reason}); "
            f"{remaining} call(s) left pending; not retryable as sent: "
            "fix the manifest/params, then continue"
        )
    return (
        f"{provider}: stopping cleanly ({classification.reason}); "
        f"{remaining} call(s) left pending; transient: continue the run "
        "once the condition clears"
    )
