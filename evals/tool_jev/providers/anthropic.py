"""Anthropic (Claude) provider adapter for tool_jev (issue #64, task t13).

Subclasses :class:`~evals.tool_jev.providers.base.BaseProvider`. Transport is
stdlib ``urllib`` only, injected as a callable so tests never touch the
network (see :data:`Transport`). Only ``https://api.anthropic.com`` is ever
called (:func:`_host_guard`) and the API key is read only from the
``ANTHROPIC_API_KEY`` environment variable via
:func:`~evals.tool_jev.providers.base.read_api_key` (never a literal).

Docs consulted while building the request/response shapes (cited inline
where a specific choice depends on one):

- Messages API request/response, headers, ``tool_choice``:
  https://platform.claude.com/docs/en/api/messages
- Effort / adaptive thinking, and why ``thinking`` itself is left unset:
  https://platform.claude.com/docs/en/api/errors ("Thinking cannot be
  disabled", "Extended thinking not supported" -- both say the *newer*
  models Claude Opus 5.5 / Claude Sonnet 5 run adaptive thinking whenever
  the ``thinking`` param is omitted, and reject an explicit ``thinking``
  param with a 400; ``output_config.effort`` is the supported knob).
- Creating a Message Batch, ``custom_id`` constraints:
  https://platform.claude.com/docs/en/api/creating-message-batches
- Retrieving one batch, ``processing_status`` enum, ``request_counts``:
  https://platform.claude.com/docs/en/api/messages/batches/retrieve
- Listing batches, pagination (``after_id``/``before_id``/``limit``,
  ``has_more``): https://platform.claude.com/docs/en/api/messages/batches/list
- Errors: HTTP status -> ``error.type`` table, error shape:
  https://platform.claude.com/docs/en/api/errors

Known limitations (report these, do not hide them):

1. **In-progress batches cannot be matched to a submit ref.** The List
   Batches endpoint returns no per-batch custom_id or metadata field, and
   ``results_url`` (the only place a request's ``custom_id`` appears) is
   ``null`` until a batch's ``processing_status`` is ``"ended"``
   (see the two "Listing batches"/"Retrieving one batch" docs above).
   :meth:`AnthropicProvider._find_batch` can therefore only confirm a
   match for an *ended* batch (it scans that batch's results for a
   ``custom_id`` whose ref-prefix matches). If the process crashes between
   a batch being accepted and the ledger recording it as submitted, and
   the batch is still ``in_progress`` at resume time, ``find_batch``
   returns ``None`` even though the batch may in fact exist and be running
   under this ref -- there is no documented Anthropic API that can tell the
   two cases apart. The caller (the ledger/runner) will then treat the
   batch as never-submitted and may resubmit, double-charging for those
   requests. This is a genuine residual risk, not a bug in this module.
2. **Per-request interface context is best-effort across a process
   restart.** ``CallResult`` requires ``case_id`` (and, to classify a
   ``choice`` answer, the case's offered candidates / label map). This
   module keeps that context in an in-memory dict populated by
   ``_send_batch`` (``self._batch_requests``), exactly like
   ``FakeProvider._batches``. Within the *same* process that submitted the
   batch, results are given back their full request context. Recovered
   after a crash (a *different* process that only called ``find_batch``),
   that cache is empty: ``case_id`` is still recovered exactly (it is
   base64-encoded directly into ``custom_id``, see
   :func:`build_custom_id`/:func:`parse_custom_id`, not merely guessed),
   but the request's ``interface``/``offered_candidates``/``labels`` are
   not recoverable from the Anthropic API and are inferred from the
   response shape instead (a ``tool_use`` content block present ->
   ``"tool_call"``, else ``"choice"`` with no offered-candidate check).
3. **Forced tool use is rejected by ``claude-opus-5-5``.** Per
   https://platform.claude.com/docs/en/api/errors ("Forced tool use not
   supported"), Claude Opus 5.5 (also Fable 5.1 / Mythos 5.1) 400s on
   ``tool_choice: {"type": "any"}`` -- the exact value the task instructed
   this adapter to send for the ``tool_call`` interface. This adapter still
   sends it (the contract is fixed for all wave-2 providers), so any
   ``tool_call`` request routed at ``claude-opus-5-5`` will be rejected as
   ``request_rejected:unsupported_parameter`` today. Report this to the
   operator before routing Track A (tool-call) cases at that model.
4. Request-contract integration note (for the main agent): this module
   reads ``request.prompt`` / ``request.case_text`` / ``request.params``
   directly, as instructed, because task t11's
   ``CallRequest.canonical_content()`` did not exist yet in this worktree.
   When it lands, this adapter's payload building should switch to it so
   redaction/canonicalization stays in one place.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Callable
from urllib.parse import urlparse

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

__all__ = [
    "API_HOST",
    "API_BASE",
    "ANTHROPIC_VERSION",
    "Transport",
    "urllib_transport",
    "AnthropicProviderError",
    "AnthropicProvider",
    "build_custom_id",
    "parse_custom_id",
]

#: Only this host is ever called (acceptance criterion 2).
API_HOST = "api.anthropic.com"
API_BASE = f"https://{API_HOST}"

#: Current documented stable version string.
#: https://platform.claude.com/docs/en/api/messages ("Headers")
ANTHROPIC_VERSION = "2023-06-01"

#: A transport is a plain callable so tests can inject a fake one instead of
#: touching the network: (method, url, headers, body_bytes) -> (status, body).
#: Every response -- including HTTP error responses -- comes back as
#: ``(status_code, body_bytes)``; a transport never raises for a documented
#: HTTP error, only for something below HTTP (see ``urllib_transport``'s use
#: of status ``0`` for a network-level failure).
Transport = Callable[[str, str, dict, "bytes | None"], "tuple[int, bytes]"]


def urllib_transport(
    method: str, url: str, headers: dict, body: "bytes | None"
) -> "tuple[int, bytes]":
    """Default transport: stdlib ``urllib.request``, no third-party HTTP client."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(
            request, timeout=60
        ) as response:  # nosec B310 - fixed https host
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError:
        # DNS failure, connection refused, timeout below the HTTP layer:
        # status 0 signals "no HTTP response at all" to _classify_error.
        return 0, b""


class AnthropicProviderError(Exception):
    """Raised for a transport-classified failure (sync send, or batch admin call).

    Carries the :class:`~evals.tool_jev.providers.errors.Classification` the
    same way ``fake.FakeProviderError`` does, so a caller's run loop handles
    both the same way. ``case_id`` is empty for a batch-admin-level failure
    (submit/list/status/results calls are not about one case).
    """

    def __init__(self, classification: Classification, provider: str, case_id: str = "") -> None:
        self.classification = classification
        self.provider = provider
        self.case_id = case_id
        detail = f" (case {case_id!r})" if case_id else ""
        super().__init__(f"{provider}: {classification.reason}{detail}")


#: Anthropic's own top-level `error.type` values that map cleanly onto one
#: of errors.py's named error_type keys. https://platform.claude.com/docs/en/api/errors
_ANTHROPIC_ERROR_TYPE_MAP = {
    "authentication_error": "auth_failed",
    "rate_limit_error": "rate_limited",
    "timeout_error": "timeout",
    "billing_error": "insufficient_quota",
}

# ---------------------------------------------------------------------------
# custom_id encoding: submit_ref prefix (fixed width) + index (fixed width,
# zero-padded) + case_id losslessly base64-encoded (padding stripped, since
# '=' is outside Anthropic's custom_id charset ^[a-zA-Z0-9_-]{1,64}$). Fixed
# widths mean the fields can be sliced apart without a separator, so a
# submit_ref or case_id that itself contains '-' can never make the split
# ambiguous (unlike splitting on a delimiter char shared with the payload
# alphabet). See module docstring, limitation 2, for what this does and
# does not guarantee across a process restart.
# ---------------------------------------------------------------------------

_REF_LEN = 10
_IDX_LEN = 6
_CUSTOM_ID_MAX = 64
_CUSTOM_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _sanitize_ref(ref: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", ref)
    return safe[:_REF_LEN].ljust(_REF_LEN, "0")


def _b64_case_id(case_id: str) -> str:
    return base64.urlsafe_b64encode(case_id.encode("utf-8")).decode("ascii").rstrip("=")


def _unb64_case_id(token: str) -> str:
    padded = token + ("=" * (-len(token) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")


def build_custom_id(submit_ref: str, index: int, case_id: str) -> str:
    """Build one batch request's ``custom_id``: ref prefix + index + case_id.

    Encodes ``submit_ref`` (sanitized/truncated to :data:`_REF_LEN` chars)
    so :meth:`AnthropicProvider._find_batch` can recognise this batch's
    requests, and losslessly base64-encodes ``case_id`` so
    :meth:`AnthropicProvider._collect_batch` can recover the exact case id
    for every result line even when it has no in-memory request cache for
    this batch (a process restart -- see limitation 2 in the module
    docstring). ``index`` is the position within this ``_send_batch`` call.
    """
    if index < 0 or index >= 10**_IDX_LEN:
        raise ValueError(f"index {index} does not fit a {_IDX_LEN}-digit custom_id field")
    prefix = _sanitize_ref(submit_ref)
    idx = f"{index:0{_IDX_LEN}d}"
    case_token = _b64_case_id(case_id)
    custom_id = f"{prefix}{idx}{case_token}"
    if len(custom_id) > _CUSTOM_ID_MAX or not _CUSTOM_ID_RE.fullmatch(custom_id):
        budget = _CUSTOM_ID_MAX - _REF_LEN - _IDX_LEN
        raise ValueError(
            f"case_id {case_id!r} does not fit Anthropic's 64-character custom_id "
            f"budget once base64-encoded (budget is {budget} base64 chars, "
            f"~{budget * 3 // 4} bytes of case_id)"
        )
    return custom_id


def parse_custom_id(custom_id: str) -> "tuple[str, int, str]":
    """Reverse :func:`build_custom_id`: -> ``(ref_prefix, index, case_id)``."""
    prefix = custom_id[:_REF_LEN]
    idx = int(custom_id[_REF_LEN : _REF_LEN + _IDX_LEN])
    case_token = custom_id[_REF_LEN + _IDX_LEN :]
    return prefix, idx, _unb64_case_id(case_token)


class AnthropicProvider(BaseProvider):
    """Messages API (sync) + Message Batches API (batch) adapter.

    Parameters
    ----------
    model_id:
        e.g. ``"claude-opus-5-5"`` or ``"claude-sonnet-5"``.
    api_key_env:
        Environment variable name read via
        :func:`~evals.tool_jev.providers.base.read_api_key`. Defaults to
        ``"ANTHROPIC_API_KEY"``.
    transport:
        Injected for tests; defaults to :func:`urllib_transport`.
    """

    def __init__(
        self,
        model_id: str,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        name: str = "anthropic",
        provider_kind: str = "anthropic",
        transport: "Transport | None" = None,
        list_page_limit: int = 100,
        max_list_pages: int = 20,
    ) -> None:
        self.name = name
        self.model_id = model_id
        # No logprobs on this provider: candidates are always None, never
        # estimated (task instruction; issue #64 contract).
        self.capabilities = ProviderCapabilities(logprobs=False, batch=True, reasoning=True)
        self._api_key_env = api_key_env
        self.provider_kind = provider_kind
        self._transport = transport or urllib_transport
        self._list_page_limit = list_page_limit
        self._max_list_pages = max_list_pages
        #: batch_id -> {case_id: CallRequest}, populated by _send_batch.
        #: Empty after a process restart -- see module docstring limitation 2.
        self._batch_requests: "dict[str, dict[str, CallRequest]]" = {}

    # -- transport plumbing ---------------------------------------------

    def _headers(self) -> dict:
        key = read_api_key(self._api_key_env)
        return {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _call(
        self, method: str, path_or_url: str, body: "dict | None" = None
    ) -> "tuple[int, bytes]":
        url = path_or_url if path_or_url.startswith("https://") else f"{API_BASE}{path_or_url}"
        host = urlparse(url).netloc
        if host != API_HOST:
            raise ValueError(f"{self.name}: refusing to call non-Anthropic host {host!r}")
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        return self._transport(method, url, self._headers(), payload)

    # -- request building -------------------------------------------------

    @staticmethod
    def _tools_payload(tools: list) -> list:
        """OpenAI-style function tool dicts -> Anthropic tool schema."""
        converted = []
        for tool in tools:
            fn = tool.get("function", tool)
            converted.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters", fn.get("input_schema", {"type": "object", "properties": {}})
                    ),
                }
            )
        return converted

    def _build_payload(self, request: CallRequest) -> dict:
        payload: dict = {
            "model": self.model_id,
            "max_tokens": int(request.params.get("max_output_tokens", 1024)),
            "system": request.prompt,
            "messages": [{"role": "user", "content": request.case_text}],
        }
        effort = request.params.get("reasoning")
        if effort:
            # `thinking` is deliberately left unset: on these models it
            # defaults to adaptive, and an explicit `thinking` param 400s
            # on the ones that no longer support extended thinking. See
            # module docstring citation to docs/en/api/errors.
            payload["output_config"] = {"effort": effort}
        if request.interface == "tool_call":
            tools = request.params.get("tools") or []
            if tools:
                payload["tools"] = self._tools_payload(tools)
                payload["tool_choice"] = {"type": "any"}
        return payload

    # -- response interpretation ------------------------------------------

    @staticmethod
    def _usage_from(usage_obj: dict) -> "dict[str, int]":
        out = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "thinking_tokens",
        ):
            value = usage_obj.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                out[key] = value
        return out

    @staticmethod
    def _answer_from_message(
        message: dict,
        interface: str,
        offered_candidates: "tuple[str, ...] | None",
        labels: "dict[str, str] | None",
    ) -> "tuple[str | None, Classification]":
        content = message.get("content") or []
        refused = message.get("stop_reason") == "refusal"
        malformed = False
        answer = None
        # `classification_key` is what gets checked against
        # `offered_candidates` (a bare candidate/operation name); `answer`
        # is what actually goes on the CallResult (the tool_call interface's
        # contract says that is the {"operation", "arguments"} JSON blob,
        # not the bare name, so the two must stay distinct).
        classification_key = None
        if interface == "tool_call":
            tool_use = next((block for block in content if block.get("type") == "tool_use"), None)
            if tool_use is not None:
                classification_key = tool_use.get("name")
                answer = json.dumps(
                    {"operation": tool_use.get("name"), "arguments": tool_use.get("input", {})},
                    sort_keys=True,
                )
            else:
                # No tool call: the model answered in plain text, valid for
                # the explain/escalate operations (task instruction).
                text = "".join(
                    block.get("text", "") for block in content if block.get("type") == "text"
                ).strip()
                answer = text or None
                classification_key = answer
                if answer is None and not refused:
                    malformed = True
        else:  # "choice"
            text = "".join(
                block.get("text", "") for block in content if block.get("type") == "text"
            ).strip()
            if not text:
                answer = None
                classification_key = None
                if not refused:
                    malformed = True
            else:
                answer = (labels or {}).get(text, text)
                classification_key = answer
        classification = classify_answer(
            classification_key, offered_candidates or None, malformed=malformed, refused=refused
        )
        return answer, classification

    def _classify_error_type(self, error_type: "str | None", message_text: str) -> "str | None":
        mapped = _ANTHROPIC_ERROR_TYPE_MAP.get(error_type)
        if mapped is None and error_type == "invalid_request_error":
            lowered = (message_text or "").lower()
            if (
                "does not support tool types" in lowered
                or "not supported for this model" in lowered
                or "tool_choice" in lowered
            ):
                mapped = "unsupported_parameter"
            elif "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
                mapped = "model_not_found"
            else:
                mapped = "invalid_request"
        return mapped

    def _classify_error(self, status: int, data: bytes) -> Classification:
        error_type = None
        message_text = ""
        try:
            body = json.loads(data)
            err = body.get("error") or {}
            error_type = err.get("type")
            message_text = err.get("message", "")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        if status == 0:
            # Below-HTTP failure from urllib_transport (URLError).
            return classify_transport(self.provider_kind, error_type="network_error")
        mapped = self._classify_error_type(error_type, message_text)
        if mapped is not None:
            return classify_transport(self.provider_kind, error_type=mapped)
        return classify_transport(self.provider_kind, status_code=status)

    def _classify_batch_result_error(self, error: dict) -> Classification:
        error_type = error.get("type")
        message_text = error.get("message", "")
        mapped = self._classify_error_type(error_type, message_text)
        if mapped is not None:
            return classify_transport(self.provider_kind, error_type=mapped)
        # No HTTP status accompanies a per-request batch error; name the
        # unrecognized Anthropic error type rather than guessing a status.
        return classify_transport(
            self.provider_kind, error_type=f"anthropic_batch_error:{error_type or 'unknown'}"
        )

    # -- BaseProvider hooks: sync ------------------------------------------

    def _send_sync(self, request: CallRequest) -> CallResult:
        payload = self._build_payload(request)
        status, data = self._call("POST", "/v1/messages", payload)
        if status != 200:
            raise AnthropicProviderError(
                self._classify_error(status, data), self.name, request.case_id
            )
        message = json.loads(data)
        labels = request.params.get("labels") if request.interface == "choice" else None
        answer, classification = self._answer_from_message(
            message, request.interface, request.offered_candidates, labels
        )
        return CallResult(
            case_id=request.case_id,
            outcome=classification.outcome,
            answer=answer,
            reason=classification.reason,
            provider=self.name,
            model_id=self.model_id,
            candidates=None,
            raw=data,
            response_id=message.get("id", ""),
            returned_model=message.get("model"),
            usage=self._usage_from(message.get("usage") or {}),
            interface=request.interface,
        )

    # -- BaseProvider hooks: batch ------------------------------------------

    def _send_batch(self, requests: "list[CallRequest]", submit_ref: str) -> BatchHandle:
        batch_requests = []
        by_case: "dict[str, CallRequest]" = {}
        for index, request in enumerate(requests):
            custom_id = build_custom_id(submit_ref, index, request.case_id)
            batch_requests.append({"custom_id": custom_id, "params": self._build_payload(request)})
            by_case[request.case_id] = request
        status, data = self._call("POST", "/v1/messages/batches", {"requests": batch_requests})
        if status not in (200, 201):
            raise AnthropicProviderError(self._classify_error(status, data), self.name)
        body = json.loads(data)
        batch_id = body["id"]
        # Kept only for this process's lifetime; see module limitation 2.
        self._batch_requests[batch_id] = by_case
        return BatchHandle(batch_id=batch_id, provider=self.name, submit_ref=submit_ref)

    def _find_batch(self, submit_ref: str) -> "BatchHandle | None":
        prefix = _sanitize_ref(submit_ref)
        after_id = None
        for _ in range(self._max_list_pages):
            path = f"/v1/messages/batches?limit={self._list_page_limit}"
            if after_id:
                path += f"&after_id={after_id}"
            status, data = self._call("GET", path)
            if status != 200:
                raise AnthropicProviderError(self._classify_error(status, data), self.name)
            body = json.loads(data)
            for batch in body.get("data", []):
                # Only an ENDED batch's results are inspectable for
                # custom_id -- see module docstring limitation 1: an
                # in-progress batch cannot be matched to a ref by any
                # documented API, so it is skipped here, not guessed at.
                if batch.get("processing_status") != "ended":
                    continue
                if self._ended_batch_matches_ref(batch, prefix):
                    return BatchHandle(
                        batch_id=batch["id"], provider=self.name, submit_ref=submit_ref
                    )
            if not body.get("has_more"):
                break
            after_id = body.get("last_id")
            if not after_id:
                break
        return None

    def _ended_batch_matches_ref(self, batch: dict, prefix: str) -> bool:
        results_url = batch.get("results_url")
        if not results_url:
            return False
        status, data = self._call("GET", results_url)
        if status != 200:
            return False
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = obj.get("custom_id", "")
            if custom_id[:_REF_LEN] == prefix:
                return True
        return False

    def _check_batch(self, handle: BatchHandle) -> BatchStatus:
        status, data = self._call("GET", f"/v1/messages/batches/{handle.batch_id}")
        if status != 200:
            raise AnthropicProviderError(self._classify_error(status, data), self.name)
        body = json.loads(data)
        complete = body.get("processing_status") == "ended"
        # Anthropic has no batch-level "expired" processing_status (only
        # in_progress/canceling/ended); per-request expiry is reported in
        # _collect_batch's results, not here.
        return BatchStatus(batch_id=handle.batch_id, complete=complete, expired=False)

    def _collect_batch(self, handle: BatchHandle) -> "list[CallResult]":
        status, data = self._call("GET", f"/v1/messages/batches/{handle.batch_id}")
        if status != 200:
            raise AnthropicProviderError(self._classify_error(status, data), self.name)
        batch = json.loads(data)
        results_url = batch.get("results_url")
        if not results_url:
            return []
        status, raw_results = self._call("GET", results_url)
        if status != 200:
            raise AnthropicProviderError(self._classify_error(status, raw_results), self.name)
        by_case = self._batch_requests.get(handle.batch_id, {})
        results = []
        for line in raw_results.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            results.append(self._result_from_line(line, obj, by_case))
        return results

    def _result_from_line(
        self, raw_line: bytes, obj: dict, by_case: "dict[str, CallRequest]"
    ) -> CallResult:
        custom_id = obj.get("custom_id", "")
        _, _, decoded_case_id = parse_custom_id(custom_id)
        request = by_case.get(decoded_case_id)
        case_id = request.case_id if request is not None else decoded_case_id
        result = obj.get("result", {})
        kind = result.get("type")

        if kind == "succeeded":
            message = result.get("message", {})
            if request is not None:
                interface = request.interface
                offered = request.offered_candidates
                labels = request.params.get("labels") if interface == "choice" else None
            else:
                # Best-effort inference across a process restart -- see
                # module docstring limitation 2.
                has_tool_use = any(
                    block.get("type") == "tool_use" for block in message.get("content", [])
                )
                interface = "tool_call" if has_tool_use else "choice"
                offered = None
                labels = None
            answer, classification = self._answer_from_message(message, interface, offered, labels)
            return CallResult(
                case_id=case_id,
                outcome=classification.outcome,
                answer=answer,
                reason=classification.reason,
                provider=self.name,
                model_id=self.model_id,
                candidates=None,
                raw=raw_line,
                response_id=message.get("id", ""),
                returned_model=message.get("model"),
                usage=self._usage_from(message.get("usage") or {}),
                interface=interface,
            )

        interface = request.interface if request is not None else "tool_call"
        if kind == "errored":
            classification = self._classify_batch_result_error(result.get("error", {}))
        elif kind == "expired":
            # Task instruction: expired -> pending via classify_transport
            # error_type "expired_batch".
            classification = classify_transport(self.provider_kind, error_type="batch_expired")
        elif kind == "canceled":
            classification = Classification(
                classify_transport(self.provider_kind, error_type="batch_expired").outcome,
                "canceled_batch_request",
                stop=True,
                retryable=True,
            )
        else:
            classification = classify_transport(
                self.provider_kind, error_type=f"unknown_batch_result_type:{kind}"
            )
        return CallResult(
            case_id=case_id,
            outcome=classification.outcome,
            answer=None,
            reason=classification.reason,
            provider=self.name,
            model_id=self.model_id,
            candidates=None,
            raw=raw_line,
            response_id="",
            returned_model=None,
            usage={},
            interface=interface,
        )
