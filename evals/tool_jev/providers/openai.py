"""OpenAI adapter (sync + Batch API) for tool_jev reference-model calls.

Covers issue #64 tasks c2/h2: an OpenAI-backed :class:`BaseProvider` for the
two reasoning models this eval targets, ``gpt-6-luna`` and ``gpt-6-sol``.
Both are reasoning models (``reasoning.effort`` accepted) that do not return
token logprobs, so :data:`CAPABILITIES` below records ``logprobs=False`` and
``CallResult.candidates`` is always ``None`` for this provider -- never
estimated.

Endpoint choice: the Responses API (``POST /v1/responses``), not Chat
Completions. Per OpenAI's own guidance, reasoning models paired with
function tools should use ``/v1/responses``, and it is the endpoint the
Batch API documents for ``reasoning.effort`` + tool calls together:

- Responses API reference:
  https://developers.openai.com/api/reference/resources/responses/methods/create
- Reasoning models guide (effort, default "medium"):
  https://developers.openai.com/api/docs/guides/reasoning
- Batch API guide (JSONL shape, custom_id, endpoint, completion_window,
  metadata, output/error files, status values):
  https://developers.openai.com/api/docs/guides/batch
- Create batch reference:
  https://developers.openai.com/api/reference/resources/batches/methods/create
- Files API (purpose="batch"): https://developers.openai.com/api/docs/guides/batch
  (the same guide documents ``POST /v1/files`` with ``purpose=batch``)

Request shape (both sync and each batch JSONL line's ``body``), built ONLY
from the :class:`~evals.tool_jev.providers.base.CallRequest` fields (the
request contract fixed by the main agent for all wave-2 tasks -- see
COMMON2.md):

    {
      "model": "gpt-6-luna",
      "instructions": request.prompt,              # system message content
      "input": [{"role": "user", "content": request.case_text}],
      "tools": request.params["tools"],             # tool_call only
      "tool_choice": "required",                    # tool_call only
      "reasoning": {"effort": request.params.get("reasoning", "medium")},
      "max_output_tokens": request.params.get("max_output_tokens"),
      "metadata": {"nvsh_case_id": request.case_id, "nvsh_interface": request.interface},
    }

``metadata`` is not part of the request contract's ``params`` -- it is this
adapter's own bookkeeping, added purely so batch results (which come back
identified only by ``custom_id``, itself hashed when a case id is long --
see :func:`_custom_id`) can always be traced back to the exact case id and
interface after a crash/resume, per Responses objects supporting a
``metadata`` dict (echoed back verbatim in the response body OpenAI writes
to the batch output file).

For ``interface == "choice"`` no tools are sent (the system prompt already
asks for a single label); the returned text is looked up in
``request.params["labels"]`` (label -> candidate name) to produce
``CallResult.answer`` as a candidate name, matching
``request.offered_candidates``.

REPORT (not yet true, sibling task t11): once
``evals.tool_jev.request.canonical_content`` merges, the runner-level
integration should switch to building ``(system, user, tools, labels)`` via
that helper rather than reading ``request.prompt`` / ``request.case_text`` /
``request.params`` directly, so every provider builds a payload from one
shared, tested place instead of duplicating the same field reads. This
adapter reads the fields directly today because t11 was not merged when
this task started.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from .base import (
    BaseProvider,
    BatchHandle,
    BatchStatus,
    CallRequest,
    CallResult,
    ProviderCapabilities,
    read_api_key,
)
from .errors import classify_answer, classify_transport

#: Requests never go anywhere but this host (acceptance criterion 2).
API_BASE = "https://api.openai.com"

#: The env var name read via read_api_key -- never a literal key (c13/h12/h15).
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

#: The only endpoint this adapter uses for both sync calls and batch lines.
RESPONSES_ENDPOINT = "/v1/responses"

#: Batch statuses that mean "not finished yet" per the Batch API guide.
_INCOMPLETE_STATUSES = frozenset({"validating", "in_progress", "finalizing"})
_COMPLETE_STATUSES = frozenset({"completed"})
_EXPIRED_STATUSES = frozenset({"expired", "failed", "cancelled", "cancelling"})

#: Both reasoning models this task covers. Neither returns logprobs.
MODELS = frozenset({"gpt-6-luna", "gpt-6-sol"})

#: A bounded number of list pages _find_batch will walk before giving up,
#: so a lost submit ref cannot spin forever against a very long batch list.
MAX_LIST_PAGES = 20


@dataclass(frozen=True)
class TransportResponse:
    """What a transport call returns, success or HTTP error alike.

    Keeping HTTP error responses in this same success-shaped return (rather
    than an exception) lets one code path classify both: an OpenAI error
    body always parses the same way regardless of status.
    """

    status: int
    body: bytes


#: A transport is any callable with this signature; tests inject a fake one
#: (never real HTTP) that returns canned bytes for canned requests.
Transport = Callable[[str, str, Optional[bytes], dict], TransportResponse]


class OpenAITransportError(Exception):
    """Raised by the default transport on a connection-level failure (no HTTP response)."""


def _default_transport(
    method: str, url: str, data: bytes | None, headers: dict
) -> TransportResponse:
    """stdlib-urllib transport: real network I/O, used only outside tests."""
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 (fixed https host)
            return TransportResponse(status=response.status, body=response.read())
    except urllib.error.HTTPError as exc:
        return TransportResponse(status=exc.code, body=exc.read())
    except urllib.error.URLError as exc:
        raise OpenAITransportError(str(exc)) from exc


def _decode_custom_id(custom_id: str) -> tuple[str, str]:
    """Best-effort split of an un-hashed ``custom_id`` back into (case_id, interface).

    Used only as the fallback when a batch result's ``custom_id`` is not in
    this process's own :attr:`OpenAIProvider._local_custom_ids` map (a
    fresh process after crash/resume, or an error line that has no
    response body to carry ``metadata``). A hashed custom_id (produced by
    :func:`_custom_id` when ``case_id::interface`` exceeded 64 characters)
    has no ``"::"`` and is returned as-is with interface guessed
    ``"tool_call"`` -- a genuine limitation: a long case id's error line
    cannot be traced back to its case id without the metadata this adapter
    also cannot get on a hard failure (see module docstring's REPORT note
    on ``_result_from_batch_line``).
    """
    if "::" in custom_id:
        case_id, _, interface = custom_id.rpartition("::")
        if interface in ("tool_call", "choice"):
            return case_id, interface
    return custom_id, "tool_call"


def _custom_id(case_id: str, interface: str) -> str:
    """A <=64-char id for a batch JSONL line, ledger-friendly.

    ``case_id::interface`` when it fits; otherwise a stable hash of the
    same string, since OpenAI's ``custom_id`` is capped at 64 characters.
    The hashed form is not itself decodable back to ``case_id`` -- that is
    why every request body also carries ``metadata.nvsh_case_id`` (see
    module docstring), which IS how ``_collect_batch`` recovers the case id
    on a batch fetched after a crash/resume.
    """
    raw = f"{case_id}::{interface}"
    if len(raw) <= 64:
        return raw
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]


def _openai_error_fields(body: bytes) -> tuple[str | None, str | None]:
    """Pull (error_type_or_code, message) out of an OpenAI error body, best effort."""
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(error, dict):
        return None, None
    return (error.get("code") or error.get("type")), error.get("message")


def _classify_http(status: int, body: bytes):
    error_type, _message = _openai_error_fields(body)
    return classify_transport("openai", status_code=status, error_type=error_type)


def _build_request_body(request: CallRequest, model: str) -> dict:
    """Build the Responses API request body from the CallRequest contract fields only."""
    body: dict = {
        "model": model,
        "instructions": request.prompt,
        "input": [{"role": "user", "content": request.case_text}],
        "reasoning": {"effort": request.params.get("reasoning", "medium")},
        "metadata": {"nvsh_case_id": request.case_id, "nvsh_interface": request.interface},
    }
    max_output_tokens = request.params.get("max_output_tokens")
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if request.interface == "tool_call":
        body["tools"] = request.params.get("tools", [])
        body["tool_choice"] = "required"
    else:
        # "choice": no tools; stash the label->candidate map in metadata too,
        # so a batch collected after a crash (this process's own
        # _local_custom_ids empty) can still translate the returned label
        # into a candidate name -- see module docstring's metadata note.
        labels = request.params.get("labels")
        if labels:
            body["metadata"]["nvsh_labels"] = json.dumps(labels, sort_keys=True)
    return body


def _extract_answer(
    response_body: dict, request_interface: str, labels: dict[str, str] | None
) -> tuple[str | None, bool, bool]:
    """Return (answer, malformed, refused) from a Responses API response body.

    ``tool_call``: looks for a ``function_call`` output item. Its arguments
    become ``answer`` as ``json.dumps({"operation": name, "arguments": ...})``
    (sorted keys, deterministic), except for the "explain"/"escalate"
    operations, which answer with plain text -- REPORT: this text-vs-JSON
    split for explain/escalate is this adapter's own reading of the shared
    "answer = JSON of {operation, arguments} ... or the text answer for
    explain/escalate" contract line in COMMON2.md; it is an assumption, not
    a measurement against a merged runner, since no consuming module exists
    yet to confirm the exact shape it expects.

    ``choice``: looks for a plain message/output_text; the label text is
    mapped through ``labels`` (label -> candidate name) if given.

    A structural refusal (an output item / content part of type
    ``"refusal"``) is detected here, never by scanning answer text.
    """
    output = response_body.get("output") or []
    if not isinstance(output, list):
        return None, True, False

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "refusal":
            return item.get("refusal"), False, True
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "refusal":
                    return part.get("refusal"), False, True

    if request_interface == "tool_call":
        for item in output:
            if isinstance(item, dict) and item.get("type") == "function_call":
                name = item.get("name")
                raw_args = item.get("arguments")
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    return None, True, False
                if name in ("explain", "escalate"):
                    text = None
                    if isinstance(arguments, dict):
                        text = arguments.get("text") or arguments.get("explanation")
                    return (text if text else json.dumps(arguments, sort_keys=True)), False, False
                return (
                    json.dumps({"operation": name, "arguments": arguments}, sort_keys=True),
                    (name is None),
                    False,
                )
        return None, True, False

    # interface == "choice": a plain text answer naming one label.
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    label = (part.get("text") or "").strip()
                    if not label:
                        return None, True, False
                    if labels:
                        return labels.get(label, label), False, False
                    return label, False, False
    return None, True, False


def _extract_usage(response_body: dict) -> dict[str, int]:
    usage = response_body.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
    details = usage.get("output_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            out["reasoning_tokens"] = reasoning_tokens
    return out


class OpenAIProvider(BaseProvider):
    """OpenAI Responses API adapter: sync calls + the Batch API.

    Parameters
    ----------
    model:
        One of :data:`MODELS` (``gpt-6-luna`` or ``gpt-6-sol``).
    api_key_env:
        Environment variable name read via
        :func:`~evals.tool_jev.providers.base.read_api_key` (default
        ``OPENAI_API_KEY``, the name the manifest is expected to give).
    transport:
        Injected for tests; defaults to a real stdlib-urllib transport.
        Signature: ``(method, url, data, headers) -> TransportResponse``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        transport: Transport | None = None,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unsupported model {model!r}, expected one of {sorted(MODELS)}")
        self.name = f"openai:{model}"
        self.model = model
        self.api_key_env = api_key_env
        self._transport: Transport = transport or _default_transport
        # gpt-6-luna / gpt-6-sol are reasoning models; neither returns token
        # logprobs, so candidates are always None for this provider.
        self.capabilities = ProviderCapabilities(logprobs=False, batch=True, reasoning=True)
        #: custom_id -> (case_id, interface) for batches submitted THIS
        #: process lifetime (a convenience only; _collect_batch never
        #: depends on it -- see the metadata-based recovery in the module
        #: docstring, needed for the crash/resume path where this dict is
        #: empty because the process restarted).
        self._local_custom_ids: dict[str, tuple[str, str]] = {}

    # -- shared HTTP helpers -------------------------------------------------

    def _headers(self, content_type: str = "application/json") -> dict:
        key = read_api_key(self.api_key_env)
        return {"Authorization": f"Bearer {key}", "Content-Type": content_type}

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        response = self._transport(method, API_BASE + path, data, self._headers())
        if response.status >= 400:
            raise _HttpError(response.status, response.body)
        return json.loads(response.body.decode("utf-8"))

    # -- sync ------------------------------------------------------------

    def _send_sync(self, request: CallRequest) -> CallResult:
        body = _build_request_body(request, self.model)
        try:
            response_body = self._request_json("POST", RESPONSES_ENDPOINT, body)
        except _HttpError as exc:
            raise OpenAIProviderError(
                _classify_http(exc.status, exc.body), self.name, request.case_id
            ) from exc
        except OpenAITransportError as exc:
            raise OpenAIProviderError(
                classify_transport("openai", error_type="network_error"),
                self.name,
                request.case_id,
            ) from exc
        return self._result_from_response(request, response_body)

    def _result_from_response(self, request: CallRequest, response_body: dict) -> CallResult:
        labels = request.params.get("labels") if request.interface == "choice" else None
        answer, malformed, refused = _extract_answer(response_body, request.interface, labels)
        classification = classify_answer(
            answer,
            request.offered_candidates or None,
            malformed=malformed,
            refused=refused,
        )
        raw = json.dumps(response_body, sort_keys=True).encode("utf-8")
        return CallResult(
            case_id=request.case_id,
            outcome=classification.outcome,
            answer=answer,
            reason=classification.reason,
            provider=self.name,
            model_id=self.model,
            candidates=None,  # gpt-6-luna/gpt-6-sol never return logprobs.
            raw=raw,
            response_id=response_body.get("id", ""),
            returned_model=response_body.get("model"),
            usage=_extract_usage(response_body),
            interface=request.interface,
        )

    # -- batch ------------------------------------------------------------

    def _send_batch(self, requests: list[CallRequest], submit_ref: str) -> BatchHandle:
        lines = []
        for request in requests:
            custom_id = _custom_id(request.case_id, request.interface)
            self._local_custom_ids[custom_id] = (request.case_id, request.interface)
            lines.append(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": RESPONSES_ENDPOINT,
                        "body": _build_request_body(request, self.model),
                    }
                )
            )
        jsonl = ("\n".join(lines) + "\n").encode("utf-8")
        file_id = self._upload_batch_file(jsonl)
        batch = self._request_json(
            "POST",
            "/v1/batches",
            {
                "input_file_id": file_id,
                "endpoint": RESPONSES_ENDPOINT,
                "completion_window": "24h",
                "metadata": {"nvsh_submit_ref": submit_ref},
            },
        )
        return BatchHandle(batch_id=batch["id"], provider=self.name, submit_ref=submit_ref)

    def _upload_batch_file(self, jsonl: bytes) -> str:
        """POST /v1/files with purpose=batch, multipart/form-data (stdlib only)."""
        boundary = "nvsh-tool-jev-batch-boundary"
        parts = [
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="purpose"\r\n\r\n'
            "batch\r\n".encode("utf-8"),
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="batch.jsonl"\r\n'
                "Content-Type: application/jsonl\r\n\r\n"
            ).encode("utf-8")
            + jsonl
            + b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        # First part above is already bytes; normalize the list.
        body_bytes = b"".join(p if isinstance(p, bytes) else p.encode("utf-8") for p in parts)
        headers = self._headers(content_type=f"multipart/form-data; boundary={boundary}")
        response = self._transport("POST", API_BASE + "/v1/files", body_bytes, headers)
        if response.status >= 400:
            raise _HttpError(response.status, response.body)
        return json.loads(response.body.decode("utf-8"))["id"]

    def _find_batch(self, submit_ref: str) -> BatchHandle | None:
        after = None
        for _page in range(MAX_LIST_PAGES):
            path = "/v1/batches?limit=100"
            if after:
                path += f"&after={after}"
            page = self._request_json("GET", path, None)
            for batch in page.get("data", []):
                metadata = batch.get("metadata") or {}
                if metadata.get("nvsh_submit_ref") == submit_ref:
                    return BatchHandle(
                        batch_id=batch["id"], provider=self.name, submit_ref=submit_ref
                    )
            if not page.get("has_more"):
                break
            data = page.get("data", [])
            if not data:
                break
            after = data[-1]["id"]
        return None

    def _check_batch(self, handle: BatchHandle) -> BatchStatus:
        batch = self._request_json("GET", f"/v1/batches/{handle.batch_id}", None)
        status = batch.get("status")
        if status in _COMPLETE_STATUSES:
            return BatchStatus(batch_id=handle.batch_id, complete=True, expired=False)
        if status in _EXPIRED_STATUSES:
            return BatchStatus(batch_id=handle.batch_id, complete=True, expired=True)
        # validating/in_progress/finalizing (or an unrecognized status): not
        # complete yet -- never guessed into expired.
        return BatchStatus(batch_id=handle.batch_id, complete=False, expired=False)

    def _collect_batch(self, handle: BatchHandle) -> list[CallResult]:
        batch = self._request_json("GET", f"/v1/batches/{handle.batch_id}", None)
        results: list[CallResult] = []
        output_file_id = batch.get("output_file_id")
        if output_file_id:
            results.extend(self._collect_file(output_file_id, failed=False))
        error_file_id = batch.get("error_file_id")
        if error_file_id:
            results.extend(self._collect_file(error_file_id, failed=True))
        return results

    def _collect_file(self, file_id: str, *, failed: bool) -> list[CallResult]:
        response = self._transport(
            "GET", API_BASE + f"/v1/files/{file_id}/content", None, self._headers()
        )
        if response.status >= 400:
            raise _HttpError(response.status, response.body)
        results = []
        for line in response.body.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            results.append(self._result_from_batch_line(json.loads(line), failed=failed))
        return results

    def _result_from_batch_line(self, line: dict, *, failed: bool) -> CallResult:
        custom_id = line.get("custom_id", "")
        case_id, interface = self._local_custom_ids.get(custom_id) or _decode_custom_id(custom_id)
        if failed:
            error = line.get("error") or {}
            classification = classify_transport(
                "openai", error_type=error.get("code") or error.get("type")
            )
            return CallResult(
                case_id=case_id,
                outcome=classification.outcome,
                answer=None,
                reason=classification.reason,
                provider=self.name,
                model_id=self.model,
                raw=json.dumps(line, sort_keys=True).encode("utf-8"),
                interface=interface,
            )
        response = line.get("response") or {}
        response_body = response.get("body") or {}
        # Recover case_id/interface from the response body's own metadata
        # when this process did not submit the batch itself (crash/resume:
        # _local_custom_ids is empty in a fresh process) -- see module
        # docstring "metadata is not part of the request contract...".
        metadata = response_body.get("metadata") or {}
        case_id = metadata.get("nvsh_case_id", case_id)
        interface = metadata.get("nvsh_interface", interface)
        params: dict = {}
        raw_labels = metadata.get("nvsh_labels")
        if raw_labels:
            try:
                params["labels"] = json.loads(raw_labels)
            except json.JSONDecodeError:
                pass
        request = CallRequest(
            case_id=case_id, split="test", case_text="", interface=interface, params=params
        )
        return self._result_from_response(request, response_body)


class _HttpError(Exception):
    """Internal: an HTTP response with status >= 400, carrying status + body."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}")


class OpenAIProviderError(Exception):
    """Raised on an infrastructure stop (mirrors FakeProviderError's shape).

    Carries the :class:`~evals.tool_jev.providers.errors.Classification` so
    a caller's run loop can build the clean-stop message with
    ``errors.stop_message`` without re-deriving anything.
    """

    def __init__(self, classification, provider: str, case_id: str) -> None:
        self.classification = classification
        self.provider = provider
        self.case_id = case_id
        super().__init__(f"{provider}: {classification.reason} on case {case_id!r}")
