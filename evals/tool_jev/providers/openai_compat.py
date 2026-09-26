"""OpenAI-compatible chat-completions adapter (OpenRouter, NVIDIA, local).

One :class:`OpenAICompatProvider`, parameterized by ``kind`` (one of
``"openrouter"``, ``"nvidia"``, ``"local"``), covers three hosts that all
speak the same ``POST /chat/completions`` shape (issue #64, t14):

======== ================================== ==========================
kind     default base URL                   default key env var
======== ================================== ==========================
openrouter https://openrouter.ai/api/v1      OPEN_ROUTER_API_KEY
nvidia     https://integrate.api.nvidia.com/v1 NGC_API_KEY
local      http://localhost:8001/v1 (lobes)  LOBES_GATEWAY_API_KEY
======== ================================== ==========================

These defaults are **code**, never committed JSON/TOML (``manifest.py``'s
own docstring: "provider default base URLs belong in the provider
adapters ... not in the manifest"). A caller may still override
``base_url``/``api_key_env`` at construction time (e.g. from the operator's
private manifest or environment) subject to one rule enforced by
:func:`_validate_base_url`: ``kind="local"`` must resolve to a localhost
host (any port/path), and the two hosted kinds must resolve to a
non-localhost host -- so a private, operator-supplied local endpoint can
never silently be swapped for a public one, and a "hosted" adapter can
never be pointed at localhost by accident.

No batch API (acceptance criterion, t14 instruction: "No batch API for
these"): ``capabilities.batch`` is always ``False`` and the four
``_send_batch``/``_find_batch``/``_check_batch``/``_collect_batch`` hooks
raise :class:`NotImplementedError` with a message naming
:meth:`submit_sync` as the only entry point a runner may use for this
provider.

Request/response shape is documented at
https://openrouter.ai/docs/api-reference/chat-completion and (NVIDIA hosts
the same shape) https://docs.api.nvidia.com/nim/reference/chat-completions
-- both are OpenAI's ``/v1/chat/completions``:
https://platform.openai.com/docs/api-reference/chat/create . Confirmed
2026-09-26 that OpenRouter's response echoes ``choices[0].logprobs.content``
as a list of ``{"token", "logprob", "top_logprobs": [...]}`` entries, one
per generated token, when the request set ``logprobs: true`` -- the same
shape OpenAI's own API uses.

Track A (``interface="tool_call"``) sends ``tools``/``tool_choice`` from
``request.params`` and reads the answer back out of
``message.tool_calls[0].function``. Track B (``interface="choice"``) asks
for one label and, when ``capabilities.logprobs`` is true, requests
``logprobs=true, top_logprobs=20`` and builds a candidate distribution from
the **first content token's** ``top_logprobs`` -- never a reasoning
token's, since a reasoning model (NVIDIA Nemotron etc.) may stream its
chain-of-thought as a separate ``reasoning``/``reasoning_content`` field
ahead of the content token that actually answers.

Sibling task t11 is adding a shared ``request.distribution_from_logprobs()``
helper (per the wave-2 brief). This module ships a local equivalent,
:func:`_distribution_from_logprobs`, because t11 does not exist yet in this
worktree. **This should switch to the shared helper once t11 merges** --
flagged in this task's report, not silently left as permanent duplication.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

from .base import (
    BaseProvider,
    BatchHandle,
    BatchStatus,
    CallRequest,
    CallResult,
    ProviderCapabilities,
    read_api_key,
)
from .errors import Classification, classify_answer, classify_transport

#: The three OpenAI-compatible hosts this adapter knows how to talk to.
KINDS = ("openrouter", "nvidia", "local")

#: Code defaults -- never read from a committed manifest (see module
#: docstring). Overridable per instance for the operator's private setup.
DEFAULT_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "local": "http://localhost:8001/v1",
}

#: Env var *names* (never literal keys) read via ``base.read_api_key``.
DEFAULT_API_KEY_ENV: dict[str, str | None] = {
    "openrouter": "OPEN_ROUTER_API_KEY",
    "nvidia": "NGC_API_KEY",
    "local": "LOBES_GATEWAY_API_KEY",
}

_LOCALHOST_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})

#: The classification table this adapter's transport errors are read
#: against (see ``providers/errors.py``; identical to "openai"/"anthropic"
#: today, kept as its own kind so NVIDIA/OpenRouter quirks can diverge
#: later without touching the OpenAI/Anthropic adapters).
_PROVIDER_KIND = "openai_compat"

#: Error-body ``error.code``/``error.type`` strings this adapter recognises
#: and forwards to ``errors.classify_transport`` as the *named* error type
#: (more specific than the bare HTTP status). Anything else falls back to
#: the status-code table.
_RECOGNISED_ERROR_TYPES = frozenset(
    {
        "insufficient_quota",
        "budget_cap_reached",
        "rate_limited",
        "timeout",
        "network_error",
        "connection_reset",
        "batch_expired",
        "invalid_request",
        "auth_failed",
        "model_not_found",
        "unsupported_parameter",
    }
)


def _validate_base_url(kind: str, base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"base_url {base_url!r} must use http:// or https:// (got scheme {parsed.scheme!r})"
        )
    hostname = (parsed.hostname or "").lower()
    is_localhost = hostname in _LOCALHOST_NAMES
    if kind == "local" and not is_localhost:
        raise ValueError(
            f"kind='local' base_url {base_url!r} must resolve to a localhost host "
            f"(one of {sorted(_LOCALHOST_NAMES)}); non-localhost is only for "
            "kind='openrouter'/'nvidia'"
        )
    if kind != "local" and is_localhost:
        raise ValueError(
            f"kind={kind!r} base_url {base_url!r} resolves to localhost; the two hosted "
            "kinds (openrouter, nvidia) must use a non-localhost URL -- only kind='local' "
            "may point at localhost"
        )


# ---------------------------------------------------------------------------
# Transport: a small, injectable seam. Tests never make a real HTTP call --
# they pass a ``transport`` callable that returns canned (status, bytes)
# pairs built from recorded-shape fixtures.
# ---------------------------------------------------------------------------


class TransportError(Exception):
    """Raised by a transport callable for a connection-level failure.

    ``error_type`` is one of the ``error_type`` keys ``errors.py``'s tables
    understand (``"timeout"``, ``"network_error"``, ``"connection_reset"``).
    Never raised for an HTTP response that came back with a status code --
    that path returns ``(status, body)`` instead so the caller can read the
    error body for a more specific classification.
    """

    def __init__(self, error_type: str) -> None:
        self.error_type = error_type
        super().__init__(error_type)


Transport = Callable[[str, dict, bytes], tuple[int, bytes]]


def _default_transport(
    url: str, headers: dict, body: bytes, *, timeout: float = 60.0
) -> tuple[int, bytes]:
    """Real HTTP transport used outside tests (stdlib only, no third-party dep)."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        # scheme is restricted to http/https by _validate_base_url at
        # construction time, so this is never file:// or another scheme.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except TimeoutError as exc:
        raise TransportError("timeout") from exc
    except urllib.error.URLError as exc:
        raise TransportError("network_error") from exc


# ---------------------------------------------------------------------------
# Client-side rate limiter (NVIDIA free tier is ~40 RPM). Injectable clock
# and sleep so tests never sleep for real.
# ---------------------------------------------------------------------------


class RateLimiter:
    """Spaces calls to at most ``requests_per_minute``, blocking as needed.

    ``clock``/``sleep`` default to ``time.monotonic``/``time.sleep`` but are
    injectable so a test can pass a fake clock plus a ``sleep`` that just
    advances it, proving the spacing logic without ever really sleeping.
    """

    def __init__(
        self,
        requests_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._next_allowed: float | None = None

    def acquire(self) -> None:
        """Block (via the injected ``sleep``) until the next call is allowed."""
        now = self._clock()
        if self._next_allowed is not None and now < self._next_allowed:
            self._sleep(self._next_allowed - now)
            now = self._next_allowed
        self._next_allowed = now + self._interval


# ---------------------------------------------------------------------------
# Infra error (mirrors fake.FakeProviderError's shape so a run loop can
# treat every provider's clean stop the same way).
# ---------------------------------------------------------------------------


class OpenAICompatInfraError(Exception):
    """Raised for a transport-level / non-2xx infrastructure condition.

    Carries the :class:`~evals.tool_jev.providers.errors.Classification` so
    a caller builds the clean-stop message with ``errors.stop_message``
    exactly as it would for ``fake.FakeProviderError``.
    """

    def __init__(self, classification: Classification, provider: str, case_id: str) -> None:
        self.classification = classification
        self.provider = provider
        self.case_id = case_id
        super().__init__(f"{provider}: {classification.reason} on case {case_id!r}")


# ---------------------------------------------------------------------------
# Local equivalent of the (not-yet-merged) request.distribution_from_logprobs.
# ---------------------------------------------------------------------------


def _distribution_from_logprobs(
    top_logprobs: list[dict],
    labels: dict[str, str],
) -> dict[str, float] | None:
    """Build a renormalized candidate distribution from one token's top_logprobs.

    ``top_logprobs`` is the first content token's ``top_logprobs`` list, each
    entry ``{"token": str, "logprob": float, ...}``. ``labels`` maps the
    offered label text (e.g. ``"A"``) to the candidate name it stands for,
    in offered order (dict insertion order).

    Returns ``None`` -- never an estimate -- unless *every* offered label is
    found among ``top_logprobs`` (exact token match, falling back to a
    whitespace-stripped match for a leading-space tokenizer artifact).
    """
    by_token: dict[str, float] = {}
    for entry in top_logprobs:
        token = entry.get("token")
        logprob = entry.get("logprob")
        if isinstance(token, str) and isinstance(logprob, (int, float)):
            by_token[token] = float(logprob)
            by_token.setdefault(token.strip(), float(logprob))

    label_probs: dict[str, float] = {}
    for label, candidate_name in labels.items():
        logprob = by_token.get(label)
        if logprob is None:
            logprob = by_token.get(label.strip())
        if logprob is None:
            return None
        label_probs[candidate_name] = math.exp(logprob)

    total = sum(label_probs.values())
    if total <= 0:
        return None
    return {name: value / total for name, value in label_probs.items()}


# ---------------------------------------------------------------------------
# The adapter itself.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParsedError:
    error_type: str | None


def _error_type_from_body(raw: bytes) -> str | None:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if isinstance(code, str) and code in _RECOGNISED_ERROR_TYPES:
        return code
    error_type = error.get("type")
    if isinstance(error_type, str) and error_type in _RECOGNISED_ERROR_TYPES:
        return error_type
    return None


class OpenAICompatProvider(BaseProvider):
    """One OpenAI-compatible chat-completions adapter for three hosts.

    Parameters
    ----------
    kind:
        One of :data:`KINDS` (``"openrouter"``, ``"nvidia"``, ``"local"``).
    model:
        The model id sent as ``"model"`` in the request body.
    base_url / api_key_env:
        Override the code default for ``kind`` (see :data:`DEFAULT_BASE_URLS`
        / :data:`DEFAULT_API_KEY_ENV`). ``api_key_env`` is an environment
        variable *name*, read at call time via ``base.read_api_key`` --
        never a literal key. Pass ``api_key_env=None`` for a local server
        that needs no credential. ``base_url`` is validated by
        :func:`_validate_base_url`.
    capabilities:
        Data passed in by the caller (from the manifest), never hard-coded
        per model here. ``capabilities.batch`` must be ``False`` -- this
        adapter has no batch API.
    transport:
        Injectable HTTP callable ``(url, headers, body) -> (status, bytes)``,
        raising :class:`TransportError` for a connection-level failure.
        Defaults to a real ``urllib.request``-based transport. Tests always
        inject a canned replacement.
    rate_limiter:
        Optional :class:`RateLimiter`, ``acquire()``-d before every call.
    reasoning_param_name:
        The top-level body field used to pass ``request.params["reasoning"]``
        through for ``kind in ("nvidia", "local")`` when
        ``capabilities.reasoning`` is true (OpenRouter always uses its own
        ``{"reasoning": {"effort": ...}}`` shape instead, regardless of this
        setting). Defaults to ``"reasoning_effort"``, mirroring the
        OpenAI-style reasoning-effort parameter these OpenAI-compatible
        hosts model themselves on -- **unverified against a live NVIDIA/
        local response** (no NVIDIA/local reasoning-capable endpoint was
        probed for this task); report flags this as an assumption standing
        in for a measurement.
    """

    def __init__(
        self,
        kind: str,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = ...,  # sentinel: distinguish "not given" from "explicitly None"
        capabilities: ProviderCapabilities | None = None,
        transport: Transport | None = None,
        rate_limiter: RateLimiter | None = None,
        reasoning_param_name: str = "reasoning_effort",
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.model = model
        self.name = f"{kind}:{model}"
        resolved_base_url = base_url or DEFAULT_BASE_URLS[kind]
        _validate_base_url(kind, resolved_base_url)
        self.base_url = resolved_base_url.rstrip("/")
        self._api_key_env = DEFAULT_API_KEY_ENV.get(kind) if api_key_env is ... else api_key_env
        self.capabilities = capabilities or ProviderCapabilities(
            logprobs=False, batch=False, reasoning=False
        )
        if self.capabilities.batch:
            raise ValueError(
                f"{self.name}: openai_compat has no batch API; capabilities.batch must be False"
            )
        self._transport = transport or _default_transport
        self._rate_limiter = rate_limiter
        self._reasoning_param_name = reasoning_param_name
        self.provider_kind = _PROVIDER_KIND

    # -- payload building ----------------------------------------------

    def _build_payload(self, request: CallRequest) -> dict:
        params = request.params or {}
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.prompt},
                {"role": "user", "content": request.case_text},
            ],
        }
        max_output_tokens = params.get("max_output_tokens")
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens

        if request.interface == "tool_call":
            tools = params.get("tools")
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "required"
        else:  # "choice"
            if self.capabilities.logprobs:
                payload["logprobs"] = True
                payload["top_logprobs"] = 20

        reasoning = params.get("reasoning")
        if reasoning:
            if self.kind == "openrouter":
                payload["reasoning"] = {"effort": reasoning}
            elif self.capabilities.reasoning:
                payload[self._reasoning_param_name] = reasoning
            # nvidia/local without capabilities.reasoning: omit entirely
            # rather than send a param the model doesn't accept (a manifest
            # that lies about this surfaces as request_rejected on a 400
            # unsupported_parameter, per classify_transport).

        if self.kind == "openrouter":
            payload["provider"] = {"data_collection": "deny"}

        return payload

    # -- response parsing -------------------------------------------------

    def _classify_and_answer(
        self, request: CallRequest, response: dict
    ) -> tuple[str | None, str, dict[str, float] | None, bool, bool]:
        """Return (answer, reason-if-malformed-source, candidates, malformed, refused)."""
        choices = response.get("choices") or []
        if not choices:
            return None, "", None, True, False
        message = choices[0].get("message") or {}
        refused = bool(message.get("refusal"))

        if request.interface == "tool_call":
            tool_calls = message.get("tool_calls") or []
            if not tool_calls or refused:
                return None, "", None, not tool_calls and not refused, refused
            function = tool_calls[0].get("function") or {}
            raw_arguments = function.get("arguments")
            try:
                parsed = json.loads(raw_arguments) if raw_arguments is not None else None
            except json.JSONDecodeError:
                return None, "", None, True, refused
            # The tool's own arguments blob is the {"operation", "arguments"}
            # pair the REQUEST CONTRACT specifies -- the tool *name*
            # (e.g. "propose_fix") is the fixed function the model must
            # call, not the operation being proposed.
            if not isinstance(parsed, dict) or not parsed.get("operation"):
                return None, "", None, True, refused
            answer = json.dumps(
                {"operation": parsed.get("operation"), "arguments": parsed.get("arguments")},
                sort_keys=True,
            )
            return answer, "", None, False, refused

        # interface == "choice"
        content = message.get("content")
        answer = content.strip() if isinstance(content, str) else None
        labels = (request.params or {}).get("labels", {})
        candidate_name = labels.get(answer, answer) if answer is not None else None

        candidates = None
        if self.capabilities.logprobs:
            logprobs_block = choices[0].get("logprobs") or {}
            content_tokens = logprobs_block.get("content") or []
            if content_tokens:
                first_top_logprobs = content_tokens[0].get("top_logprobs") or []
                candidates = _distribution_from_logprobs(first_top_logprobs, labels)

        return candidate_name, "", candidates, False, refused

    # -- BaseProvider hooks -----------------------------------------------

    def _send_sync(self, request: CallRequest) -> CallResult:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        payload = self._build_payload(request)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key_env:
            headers["Authorization"] = f"Bearer {read_api_key(self._api_key_env)}"

        try:
            status, raw = self._transport(f"{self.base_url}/chat/completions", headers, body)
        except TransportError as exc:
            classification = classify_transport(self.provider_kind, error_type=exc.error_type)
            raise OpenAICompatInfraError(classification, self.name, request.case_id) from exc

        if status >= 400:
            error_type = _error_type_from_body(raw)
            classification = classify_transport(
                self.provider_kind, status_code=status, error_type=error_type
            )
            raise OpenAICompatInfraError(classification, self.name, request.case_id)

        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            classification = classify_answer(None, malformed=True)
            return CallResult(
                case_id=request.case_id,
                outcome=classification.outcome,
                answer=None,
                reason=classification.reason,
                provider=self.name,
                raw=raw,
                interface=request.interface,
            )

        answer, _, candidates, malformed, refused = self._classify_and_answer(request, response)
        offered = request.offered_candidates or None
        classification = classify_answer(answer, offered, malformed=malformed, refused=refused)

        usage_raw = response.get("usage") or {}
        usage: dict[str, int] = {}
        for key, source_key in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
        ):
            value = usage_raw.get(source_key)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] = value
        reasoning_tokens = (usage_raw.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            usage["reasoning_tokens"] = reasoning_tokens

        return CallResult(
            case_id=request.case_id,
            outcome=classification.outcome,
            answer=answer,
            reason=classification.reason,
            provider=self.name,
            model_id=self.model,
            candidates=candidates,
            raw=raw,
            response_id=str(response.get("id", "")),
            returned_model=response.get("model"),
            usage=usage,
            interface=request.interface,
        )

    def _send_batch(self, requests: list[CallRequest], submit_ref: str) -> BatchHandle:
        raise NotImplementedError(
            f"{self.name}: no batch API for openai_compat providers "
            "(capabilities.batch=False); a runner must call submit_sync only"
        )

    def _find_batch(self, submit_ref: str) -> BatchHandle | None:
        raise NotImplementedError(
            f"{self.name}: no batch API for openai_compat providers "
            "(capabilities.batch=False); a runner must call submit_sync only"
        )

    def _check_batch(self, handle: BatchHandle) -> BatchStatus:
        raise NotImplementedError(
            f"{self.name}: no batch API for openai_compat providers "
            "(capabilities.batch=False); a runner must call submit_sync only"
        )

    def _collect_batch(self, handle: BatchHandle) -> list[CallResult]:
        raise NotImplementedError(
            f"{self.name}: no batch API for openai_compat providers "
            "(capabilities.batch=False); a runner must call submit_sync only"
        )
