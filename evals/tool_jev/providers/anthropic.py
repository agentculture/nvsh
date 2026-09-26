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
   therefore raises :class:`~evals.tool_jev.providers.base.BatchLookupUnresolved`
   -- never ``None``, which would authorize a resubmit and a double charge.
   The same holds when an ended batch's results cannot be downloaded or the
   listing runs past its page bound. The runner stops and asks the operator
   on it (plan risk r6). The ref carries no submission time, so every
   unconfirmable batch in the account counts: an unrelated batch still in
   flight also keeps the answer unresolved until it ends.
2. **Per-request interface context is best-effort across a process
   restart.** ``CallResult`` requires ``case_id`` (and, to classify a
   ``choice`` answer, the case's offered candidates / label map). This
   module keeps that context in an in-memory dict populated by
   ``_send_batch`` (``self._batch_requests``), exactly like
   ``FakeProvider._batches``. Within the *same* process that submitted the
   batch, results are given back their full request context (keyed by
   ``custom_id``, so one case sent through both interfaces in one batch
   keeps both contexts apart). Recovered after a crash (a *different*
   process that only called ``find_batch``), that cache is empty: ``case_id``
   and ``interface`` are still recovered exactly (both are encoded in
   ``custom_id``, see :func:`build_custom_id`/:func:`parse_custom_id`), but
   the request's ``offered_candidates``/``labels`` are not recoverable from
   the Anthropic API, so only structural checks apply to the answer.
3. **Forced tool use is rejected by some models.** Per
   https://platform.claude.com/docs/en/api/errors ("Forced tool use not
   supported"), Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1
   return HTTP 400 on ``tool_choice`` ``{"type": "any"}`` / ``{"type":
   "tool"}``; only ``auto``/``none`` are accepted. The constructor's
   ``forced_tool_choice`` flag picks ``{"type": "any"}`` (True) or
   ``{"type": "auto"}`` (False). It defaults to False for the model ids in
   :data:`NO_FORCED_TOOL_CHOICE_MODELS` (today only ``claude-opus-5-5``;
   pass ``forced_tool_choice=False`` for any other model that rejects it)
   and True otherwise. Under ``auto`` a model may answer in plain text,
   which the answer contract below classifies ``malformed``.

Request/answer contract: the payload is built only from
:func:`evals.tool_jev.request.canonical_content` (system, user, tools,
labels). For ``tool_call`` the answer is the RAW first ``tool_use`` block,
``json.dumps({"name": <tool name>, "arguments": <input dict>})``, parsed
downstream by :func:`evals.tool_jev.request.parse_tool_call`; a text-only
reply is ``answer=None`` classified ``malformed``. For ``choice`` the answer
is the model's text, stripped, read by
:func:`evals.tool_jev.request.parse_choice`. The outcome is taken from those
parsers when the request's offered set is known (sync, or a batch this
process submitted); after a restart only structural checks apply.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Callable
from urllib.parse import urlparse

from .. import request as contract
from .base import (
    BaseProvider,
    BatchHandle,
    BatchLookupUnresolved,
    BatchStatus,
    CallRequest,
    CallResult,
    ProviderCapabilities,
    ReplyText,
    read_api_key,
    tool_choice_forced,
    visible_text,
)
from .errors import Classification, classify_transport
from .openai import classify_result, tool_call_answer

__all__ = [
    "API_HOST",
    "API_BASE",
    "ANTHROPIC_VERSION",
    "NO_FORCED_TOOL_CHOICE_MODELS",
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

#: Models that reject forced tool use (tool_choice "any"/"tool") with HTTP 400,
#: per https://platform.claude.com/docs/en/api/errors ("Forced tool use not
#: supported"). ``forced_tool_choice`` defaults to False for these.
NO_FORCED_TOOL_CHOICE_MODELS = frozenset({"claude-opus-5-5"})

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
# zero-padded) + one interface letter (``t`` tool_call, ``c`` choice) +
# case_id losslessly base64-encoded (padding stripped, since
# '=' is outside Anthropic's custom_id charset ^[a-zA-Z0-9_-]{1,64}$). Fixed
# widths mean the fields can be sliced apart without a separator, so a
# submit_ref or case_id that itself contains '-' can never make the split
# ambiguous (unlike splitting on a delimiter char shared with the payload
# alphabet). See module docstring, limitation 2, for what this does and
# does not guarantee across a process restart.
# ---------------------------------------------------------------------------

_REF_LEN = 10
_IDX_LEN = 6
#: One letter per interface, so a result's interface is recoverable after a
#: restart without guessing it from the response shape.
_INTERFACE_CODES = {"tool_call": "t", "choice": "c", "text": "x"}
_INTERFACE_BY_CODE = {code: name for name, code in _INTERFACE_CODES.items()}
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


def build_custom_id(submit_ref: str, index: int, case_id: str, interface: str = "tool_call") -> str:
    """Build one batch request's ``custom_id``: ref prefix + index + interface + case_id.

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
    if interface not in _INTERFACE_CODES:
        raise ValueError(f"unknown interface {interface!r}")
    prefix = _sanitize_ref(submit_ref)
    idx = f"{index:0{_IDX_LEN}d}"
    case_token = _b64_case_id(case_id)
    custom_id = f"{prefix}{idx}{_INTERFACE_CODES[interface]}{case_token}"
    if len(custom_id) > _CUSTOM_ID_MAX or not _CUSTOM_ID_RE.fullmatch(custom_id):
        budget = _CUSTOM_ID_MAX - _REF_LEN - _IDX_LEN - 1
        raise ValueError(
            f"case_id {case_id!r} does not fit Anthropic's 64-character custom_id "
            f"budget once base64-encoded (budget is {budget} base64 chars, "
            f"~{budget * 3 // 4} bytes of case_id)"
        )
    return custom_id


def parse_custom_id(custom_id: str) -> "tuple[str, int, str, str]":
    """Reverse :func:`build_custom_id`: -> ``(ref_prefix, index, interface, case_id)``."""
    prefix = custom_id[:_REF_LEN]
    idx = int(custom_id[_REF_LEN : _REF_LEN + _IDX_LEN])
    code = custom_id[_REF_LEN + _IDX_LEN : _REF_LEN + _IDX_LEN + 1]
    if code not in _INTERFACE_BY_CODE:
        raise ValueError(f"custom_id {custom_id!r} carries no interface letter")
    case_token = custom_id[_REF_LEN + _IDX_LEN + 1 :]
    return prefix, idx, _INTERFACE_BY_CODE[code], _unb64_case_id(case_token)


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
    forced_tool_choice:
        ``True`` sends ``tool_choice={"type": "any"}`` for a ``tool_call``
        request, ``False`` sends ``{"type": "auto"}``. ``None`` (default)
        means False for a model in :data:`NO_FORCED_TOOL_CHOICE_MODELS`
        (which 400 on forced tool use) and True otherwise.
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
        forced_tool_choice: "bool | None" = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        if forced_tool_choice is None:
            forced_tool_choice = model_id not in NO_FORCED_TOOL_CHOICE_MODELS
        self.forced_tool_choice = forced_tool_choice
        # No logprobs on this provider: candidates are always None, never
        # estimated (task instruction; issue #64 contract).
        self.capabilities = ProviderCapabilities(logprobs=False, batch=True, reasoning=True)
        self._api_key_env = api_key_env
        self.provider_kind = provider_kind
        self._transport = transport or urllib_transport
        self._list_page_limit = list_page_limit
        self._max_list_pages = max_list_pages
        #: batch_id -> {custom_id: CallRequest}, populated by _send_batch.
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

    @classmethod
    def _native_prefix(cls, native: list, calls: list, wire_ids: dict) -> list:
        """The model's own blocks, in order, up to the tool call the loop acted on.

        Thinking, redacted-thinking and text blocks keep their place (a
        thinking signature is bound to what came before it). The first
        well-formed ``tool_use`` is the call the loop acted on (every adapter
        answers with the first well-formed call): it is replayed under the
        model's own id with the loop's arguments, and everything after it --
        later thinking, text or calls the loop never ran -- is dropped, which
        also leaves no ``tool_use`` without a ``tool_result``. A malformed
        ``tool_use`` before it is dropped for the same reason.
        """
        blocks: list = []
        for item in native:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in ("thinking", "redacted_thinking", "text"):
                blocks.append(dict(item))
                continue
            if kind != "tool_use" or not calls:
                continue
            if tool_call_answer(item.get("name"), item.get("input", {})) is None:
                continue
            call = calls[0]
            wire_id = item.get("id") if isinstance(item.get("id"), str) else call["id"]
            wire_ids[call["id"]] = wire_id
            blocks.append(
                {
                    "type": "tool_use",
                    "id": wire_id,
                    "name": call["name"],
                    "input": cls._tool_input(call["arguments"]),
                }
            )
            break
        return blocks

    @staticmethod
    def _tool_input(arguments: str) -> dict:
        """A neutral tool call's JSON arguments text as a ``tool_use`` input object."""
        try:
            decoded = json.loads(arguments)
        except ValueError:
            return {"_arguments": arguments}
        return decoded if isinstance(decoded, dict) else {"_arguments": decoded}

    @classmethod
    def _messages_payload(cls, messages: list) -> list:
        """``canonical_content``'s messages as Messages API turns.

        A history assistant turn becomes an ``assistant`` message. When it
        carries the model's own blocks (``native["anthropic"]``), they are
        replayed in their original order -- thinking blocks unchanged, since
        their signature binds their position -- up to and including the one
        well-formed ``tool_use`` the loop acted on (``input`` from the loop's
        own arguments); every block after it is dropped, as the loop never
        acted on it (see :meth:`_native_prefix`). Without native blocks the
        turn is its text, then one ``tool_use`` block per neutral tool call
        (``input`` decoded from the neutral ``arguments`` text). A tool turn
        becomes a ``tool_result`` block in a ``user`` message; consecutive
        results share one message. When the native blocks carry the model's
        own ``tool_use`` ids, those ids are kept and the matching results
        point at them.
        """
        out: list = []
        wire_ids: dict = {}
        for message in messages:
            role = message["role"]
            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": wire_ids.get(message["tool_call_id"], message["tool_call_id"]),
                    "content": message["content"],
                }
                last = out[-1] if out else None
                if (
                    last is not None
                    and last["role"] == "user"
                    and isinstance(last["content"], list)
                    and all(item.get("type") == "tool_result" for item in last["content"])
                ):
                    last["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
                continue
            if role == "assistant":
                native = (message.get("native") or {}).get("anthropic") or []
                calls = list(message.get("tool_calls", []))
                blocks = cls._native_prefix(native, calls[:1], wire_ids) if native else []
                if not native and message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                replayed = int(bool(calls and blocks and blocks[-1].get("type") == "tool_use"))
                for call in calls[replayed:]:
                    wire_ids.setdefault(call["id"], call["id"])
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": wire_ids[call["id"]],
                            "name": call["name"],
                            "input": cls._tool_input(call["arguments"]),
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": role, "content": message["content"]})
        return out

    def _build_payload(self, request: CallRequest) -> dict:
        system_text, messages, tools, _labels = contract.canonical_content(request)
        payload: dict = {
            "model": self.model_id,
            "max_tokens": int(request.params.get("max_output_tokens", 1024)),
            "messages": self._messages_payload(messages),
        }
        if system_text:
            # A judge call ("text") has no system text; an empty one is omitted.
            payload["system"] = system_text
        effort = request.params.get("reasoning")
        if effort:
            # `thinking` is deliberately left unset: on these models it
            # defaults to adaptive, and an explicit `thinking` param 400s
            # on the ones that no longer support extended thinking. See
            # module docstring citation to docs/en/api/errors.
            payload["output_config"] = {"effort": effort}
        if request.interface == "tool_call" and tools:
            payload["tools"] = self._tools_payload(tools)
            # Forced ("any") unless this model rejects forced tool use (see
            # module docstring limitation 3) or the request asks for "auto"
            # (the Track A loop always does).
            forced = tool_choice_forced(request, self.forced_tool_choice)
            payload["tool_choice"] = {"type": "any" if forced else "auto"}
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
        answer = None
        if refused:
            malformed = False
        elif interface == "tool_call":
            # The RAW first tool_use block (module docstring's answer
            # contract); a text-only reply is not a tool call -> malformed.
            # The first WELL-FORMED tool_use, as nvsh's ToolChat drops malformed calls.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    answer = tool_call_answer(block.get("name"), block.get("input", {}))
                    if answer is not None:
                        break
            malformed = answer is None
        else:  # "choice": the model's text, stripped
            text = "".join(
                block.get("text", "") for block in content if block.get("type") == "text"
            ).strip()
            answer = text or None
            malformed = answer is None
        classification = classify_result(
            interface,
            answer,
            offered_candidates or None,
            labels,
            malformed=malformed,
            refused=refused,
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
        return self._result_from_message(request, json.loads(data), data)

    def _result_from_message(self, request: CallRequest, message: dict, raw: bytes) -> CallResult:
        labels = contract.canonical_content(request)[3]
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
            raw=raw,
            response_id=message.get("id", ""),
            returned_model=message.get("model"),
            usage=self._usage_from(message.get("usage") or {}),
            interface=request.interface,
        )

    @staticmethod
    def _message_of(raw: bytes) -> dict:
        """The Messages API message inside a cached raw answer (sync body or batch line)."""
        obj = json.loads(raw)
        result = obj.get("result") if isinstance(obj, dict) else None
        if isinstance(result, dict) and "custom_id" in obj:
            if result.get("type") != "succeeded":
                raise ValueError("a cached batch line that did not succeed holds no answer")
            return result.get("message") or {}
        return obj

    def result_from_raw(self, request: CallRequest, raw: bytes) -> CallResult:
        """Re-read a cached answer: a sync message body or a succeeded batch line."""
        return self._result_from_message(request, self._message_of(raw), raw)

    def reply_text(self, raw: bytes) -> ReplyText:
        """The ``text`` blocks (thinking blocks never), and ``stop_reason == "max_tokens"``."""
        message = self._message_of(raw)
        content = message.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return ReplyText(
            text=visible_text(text), truncated=message.get("stop_reason") == "max_tokens"
        )

    def native_turn(self, raw: bytes) -> "dict | None":
        """This answer's content blocks, for replaying the turn natively (thinking included)."""
        content = self._message_of(raw).get("content")
        return {"anthropic": list(content)} if isinstance(content, list) else None

    # -- BaseProvider hooks: batch ------------------------------------------

    def _send_batch(self, requests: "list[CallRequest]", submit_ref: str) -> BatchHandle:
        batch_requests = []
        by_case: "dict[str, CallRequest]" = {}
        for index, request in enumerate(requests):
            custom_id = build_custom_id(submit_ref, index, request.case_id, request.interface)
            batch_requests.append({"custom_id": custom_id, "params": self._build_payload(request)})
            # Keyed by custom_id, never case_id: one case may ride the same
            # batch through both interfaces.
            by_case[custom_id] = request
        status, data = self._call("POST", "/v1/messages/batches", {"requests": batch_requests})
        if status not in (200, 201):
            raise AnthropicProviderError(self._classify_error(status, data), self.name)
        body = json.loads(data)
        batch_id = body["id"]
        # Kept only for this process's lifetime; see module limitation 2.
        self._batch_requests[batch_id] = by_case
        return BatchHandle(batch_id=batch_id, provider=self.name, submit_ref=submit_ref)

    def _find_batch(self, submit_ref: str) -> "BatchHandle | None":
        """The ended batch holding *submit_ref*'s custom ids; ``None`` only when certain.

        Anthropic shows a batch's custom ids only once it has ended (see
        limitation 1). So a batch still in progress, an ended batch whose
        results cannot be read, or a listing cut off by the page bound could
        each be the one accepted under *submit_ref*: any of them makes the
        answer :class:`BatchLookupUnresolved` rather than ``None`` -- ``None``
        would authorize a resubmit, i.e. a possible double charge. Without a
        submission time on the ref, every unconfirmable batch in the listing
        counts, so an account with other batches in flight stays unresolved
        until they end.
        """
        prefix = _sanitize_ref(submit_ref)
        unconfirmed: list = []
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
                if batch.get("processing_status") != "ended":
                    unconfirmed.append(batch.get("id"))
                    continue
                matched = self._ended_batch_matches_ref(batch, prefix)
                if matched is True:
                    return BatchHandle(
                        batch_id=batch["id"], provider=self.name, submit_ref=submit_ref
                    )
                if matched is None:
                    unconfirmed.append(batch.get("id"))
            if not body.get("has_more"):
                break
            after_id = body.get("last_id")
            if not after_id:
                break
        else:
            unconfirmed.append(f"(more than {self._max_list_pages} pages)")
        if unconfirmed:
            raise BatchLookupUnresolved(
                f"{self.name}: cannot confirm whether a batch was accepted under "
                f"{submit_ref!r}; unconfirmable: {', '.join(str(i) for i in unconfirmed)}"
            )
        return None

    def _ended_batch_matches_ref(self, batch: dict, prefix: str) -> "bool | None":
        """True/False once the batch's results were read; ``None`` when they could not be."""
        results_url = batch.get("results_url")
        if not results_url:
            return None
        status, data = self._call("GET", results_url)
        if status != 200:
            return None
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
        _, _, decoded_interface, decoded_case_id = parse_custom_id(custom_id)
        request = by_case.get(custom_id)
        case_id = request.case_id if request is not None else decoded_case_id
        result = obj.get("result", {})
        kind = result.get("type")

        if kind == "succeeded":
            message = result.get("message", {})
            if request is not None:
                interface = request.interface
                offered = request.offered_candidates
                labels = contract.canonical_content(request)[3]
            else:
                # Across a process restart the interface still comes back
                # exactly (it is encoded in custom_id); the offered set and
                # labels do not -- see module docstring limitation 2.
                interface = decoded_interface
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

        interface = request.interface if request is not None else decoded_interface
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
