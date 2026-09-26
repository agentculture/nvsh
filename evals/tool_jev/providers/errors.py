"""Error classification for tool_jev reference-provider calls.

Operator decision (issue #64, c45/h31): only a **model-answer failure** —
refusal, malformed or empty output, or an answer outside the offered
candidate set — is scored ``invalid`` and counted in the denominator. Any
**infrastructure stop** (HTTP 402 / insufficient credit or quota, a
provider or local budget cap reached, HTTP 429 rate limiting, a timeout,
network loss, a machine reset, an expired batch) leaves the call
``pending`` and the run stops cleanly, naming the provider and how many
calls remain — it is never counted invalid and never silently dropped.

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
    """

    outcome: Outcome
    reason: str
    stop: bool = False


def _pending(reason: str) -> Classification:
    return Classification(Outcome.PENDING, reason, stop=True)


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
    402: _pending("insufficient_credit"),
    408: _pending("timeout"),
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

    Looks the condition up in that provider kind's table (status code
    first, then named error type). An unrecognized condition is still
    classified ``pending`` (never guessed into ``invalid``) so an operator
    sees an unfamiliar stop rather than losing the call to the denominator.
    """
    table = PROVIDER_ERROR_TABLES.get(provider_kind)
    if table is None:
        raise KeyError(f"unknown provider kind: {provider_kind!r}")
    if status_code is not None and status_code in table["status"]:
        return table["status"][status_code]
    if error_type is not None and error_type in table["error_type"]:
        return table["error_type"][error_type]
    unknown_reason = error_type or (f"http_{status_code}" if status_code is not None else "unknown")
    return _pending(f"unrecognized_infra_condition:{unknown_reason}")


_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i won't",
    "i will not",
    "as an ai",
    "unable to comply",
)


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.strip().lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def classify_answer(
    answer: str | None,
    offered_candidates: tuple[str, ...] | None = None,
    *,
    malformed: bool = False,
) -> Classification:
    """Classify a model answer.

    Only these four shapes are ``invalid`` (and counted in the
    denominator): malformed output, empty output, a refusal, or an answer
    outside the offered candidate set. Anything else is ``ok``.
    """
    if malformed:
        return Classification(Outcome.INVALID, "malformed")
    if answer is None or not str(answer).strip():
        return Classification(Outcome.INVALID, "empty_answer")
    if _looks_like_refusal(answer):
        return Classification(Outcome.INVALID, "refusal")
    if offered_candidates and answer not in offered_candidates:
        return Classification(Outcome.INVALID, "outside_offered_set")
    return Classification(Outcome.OK, "answer")


def stop_message(provider: str, classification: Classification, remaining: int) -> str:
    """Render the clean-stop message naming the provider and remaining calls.

    Used by the (per-task) run loop when a :class:`Classification` with
    ``stop=True`` comes back from a call, so every provider's stop message
    has the same shape.
    """
    return (
        f"{provider}: stopping cleanly ({classification.reason}); "
        f"{remaining} call(s) left pending"
    )
