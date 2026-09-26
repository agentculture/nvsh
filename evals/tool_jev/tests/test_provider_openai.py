"""Tests for evals/tool_jev/providers/openai.py (issue #64, t12).

All HTTP is faked via an injected transport (:class:`ScriptedTransport`
below) that plays back hand-written fixture bytes under
``evals/tool_jev/tests/fixtures/openai/``, built from the documented
Responses/Batch API response shapes (see the doc URLs cited in
``providers/openai.py``). No real network call is ever made; a call to any
host other than ``api.openai.com`` would show up as an unscripted
transport call and fail the test.

Covers the t12 acceptance criteria:

1. recorded-HTTP fixture tests cover submit (sync), poll, fetch and an
   expired batch;
2. requests go only to ``api.openai.com`` and the key comes only from
   ``OPENAI_API_KEY`` (default) / a manifest-given env var name, never a
   literal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tool_jev.providers import base, errors
from evals.tool_jev.providers import openai as openai_provider
from nvsh.tiers import lfm

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "openai"


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _fixture_json(name: str) -> dict:
    return json.loads(_fixture_bytes(name))


class ScriptedTransport:
    """Records every call and answers each with the next matching script entry.

    ``script`` is a list of ``(predicate, response)`` pairs consumed
    first-match-wins (not strictly FIFO) so a test can queue e.g. two GET
    /v1/batches/{id} calls that return different bodies across a poll loop
    by giving each its own narrower predicate and relying on list order.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, method, url, data, headers):
        self.calls.append({"method": method, "url": url, "data": data, "headers": headers})
        for index, (predicate, response) in enumerate(self.script):
            if predicate(method, url, data):
                del self.script[index]
                return response
        raise AssertionError(f"unscripted transport call: {method} {url}")


def _path_is(path: str):
    target = openai_provider.API_BASE + path
    return lambda method, url, data: url == target


def _path_startswith(path: str):
    target = openai_provider.API_BASE + path
    return lambda method, url, data: url.startswith(target)


def _method_and_path_startswith(method_name: str, path: str):
    target = openai_provider.API_BASE + path
    return lambda method, url, data: method == method_name and url.startswith(target)


def _ok(body: bytes) -> openai_provider.TransportResponse:
    return openai_provider.TransportResponse(status=200, body=body)


def _http_error(status: int, body: bytes) -> openai_provider.TransportResponse:
    return openai_provider.TransportResponse(status=status, body=body)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    # Built at runtime, never a literal assigned to a variable named
    # token/secret/password/api_key (scripts/scan-secrets.py rule).
    monkeypatch.setenv("OPENAI_API_KEY", "k" * 8 + "-test")


#: nvsh's real Track A tools (canonical Chat-Completions shape), as
#: request.build_tool_call_request would put them in params["tools"].
REAL_TOOLS = lfm.tools_for()
OFFERED = ("service_status", "service_restart", "explain", "escalate")


def _tool_call_request(case_id: str = "case-1", split: str = "test") -> base.CallRequest:
    return base.CallRequest(
        case_id=case_id,
        split=split,
        case_text="synthetic case body",
        prompt="synthetic system instructions",
        interface="tool_call",
        offered_candidates=OFFERED,
        params={"tools": REAL_TOOLS, "reasoning": "medium", "max_output_tokens": 512},
    )


def _choice_request(case_id: str = "case-2") -> base.CallRequest:
    # params["labels"] is candidate -> letter, as request.build_choice_request builds it.
    return base.CallRequest(
        case_id=case_id,
        split="test",
        case_text="synthetic case body",
        prompt="synthetic system instructions asking for one label",
        interface="choice",
        offered_candidates=("service_restart", "escalate"),
        params={
            "labels": {"service_restart": "A", "escalate": "B"},
            "reasoning": "medium",
        },
    )


# ---------------------------------------------------------------------------
# Construction / key handling (acceptance criterion 2)
# ---------------------------------------------------------------------------


def test_rejects_unsupported_model():
    with pytest.raises(ValueError):
        openai_provider.OpenAIProvider("gpt-5-not-a-real-model", transport=lambda *a: None)


def test_capabilities_no_logprobs_batch_and_reasoning():
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=lambda *a: None)
    assert provider.capabilities.logprobs is False
    assert provider.capabilities.batch is True
    assert provider.capabilities.reasoning is True


def test_key_read_only_from_named_env_var_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = openai_provider.OpenAIProvider(
        "gpt-6-luna", transport=lambda *a: (_ for _ in ()).throw(AssertionError("no network"))
    )
    with pytest.raises(base.MissingProviderKey):
        provider.submit_sync(_tool_call_request())


def test_key_read_from_custom_env_var_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_OPENAI_KEY", "k" * 10)
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_propose.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider(
        "gpt-6-luna", api_key_env="MY_OPENAI_KEY", transport=transport
    )
    result = provider.submit_sync(_tool_call_request())
    assert result.outcome is errors.Outcome.OK
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer " + "k" * 10


def test_all_transport_calls_go_only_to_api_openai_com():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_propose.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    provider.submit_sync(_tool_call_request())
    assert transport.calls, "expected at least one transport call"
    for call in transport.calls:
        assert call["url"].startswith("https://api.openai.com/")


# ---------------------------------------------------------------------------
# Sync submit (acceptance criterion 1: "submit")
# ---------------------------------------------------------------------------


def test_sync_tool_call_success_passes_raw_tool_call_through():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_propose.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    result = provider.submit_sync(_tool_call_request())

    assert result.outcome is errors.Outcome.OK
    assert json.loads(result.answer) == {
        "name": "propose",
        "arguments": {
            "operation": "service_restart",
            "arguments": {"service": "synthetic.service"},
        },
    }
    assert result.candidates is None  # gpt-6-luna never returns logprobs
    assert result.response_id == "resp_propose"
    assert result.returned_model == "gpt-6-luna-2026-01-15"
    assert result.usage["reasoning_tokens"] == 20
    assert result.interface == "tool_call"

    # The exact request body sent used only the CallRequest contract
    # fields: prompt -> instructions, case_text -> the user message.
    sent = json.loads(transport.calls[0]["data"])
    assert sent["instructions"] == "synthetic system instructions"
    assert sent["input"] == [{"role": "user", "content": "synthetic case body"}]
    assert sent["tool_choice"] == "required"
    assert sent["reasoning"] == {"effort": "medium"}
    # Canonical tools are flattened into the Responses API's function shape.
    assert [tool["name"] for tool in sent["tools"]] == [
        tool["function"]["name"] for tool in REAL_TOOLS
    ]
    assert all(tool["type"] == "function" and "function" not in tool for tool in sent["tools"])
    propose = next(tool for tool in sent["tools"] if tool["name"] == "propose")
    assert propose["parameters"] == REAL_TOOLS[-3]["function"]["parameters"]


def test_forced_tool_choice_false_sends_auto():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_propose.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider(
        "gpt-6-luna", transport=transport, forced_tool_choice=False
    )
    provider.submit_sync(_tool_call_request())
    assert json.loads(transport.calls[0]["data"])["tool_choice"] == "auto"


def test_sync_text_only_tool_call_reply_is_malformed():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_text_only.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    result = provider.submit_sync(_tool_call_request())
    assert result.answer is None
    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "malformed"


def test_sync_choice_success_answers_with_the_label_text():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_choice.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-sol", transport=transport)
    result = provider.submit_sync(_choice_request())

    assert result.outcome is errors.Outcome.OK
    assert result.answer == "B"  # the text; request.parse_choice reads it
    assert result.reason == "answer"  # parse_choice: "B" -> escalate, offered
    assert result.candidates is None

    sent = json.loads(transport.calls[0]["data"])
    assert "tools" not in sent
    assert "tool_choice" not in sent


def test_sync_structural_refusal_is_invalid():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_refusal.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    result = provider.submit_sync(_tool_call_request())

    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "refusal"


def test_sync_missing_tool_call_is_malformed():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _ok(_fixture_bytes("responses_malformed.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    result = provider.submit_sync(_tool_call_request())

    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "malformed"


def test_sync_rate_limit_is_pending_and_retryable():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _http_error(429, _fixture_bytes("error_429.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    with pytest.raises(openai_provider.OpenAIProviderError) as excinfo:
        provider.submit_sync(_tool_call_request())
    classification = excinfo.value.classification
    assert classification.outcome is errors.Outcome.PENDING
    assert classification.stop is True
    assert classification.retryable is True


def test_sync_bad_request_is_rejected_not_retryable():
    transport = ScriptedTransport(
        [
            (
                _method_and_path_startswith("POST", "/v1/responses"),
                _http_error(400, _fixture_bytes("error_400.json")),
            )
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    with pytest.raises(openai_provider.OpenAIProviderError) as excinfo:
        provider.submit_sync(_tool_call_request())
    classification = excinfo.value.classification
    assert classification.rejected is True
    assert classification.retryable is False


def test_heldout_split_never_reaches_transport():
    transport = ScriptedTransport([])  # any call at all is a failure
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    request = _tool_call_request(split="heldout")
    with pytest.raises(base.HeldoutSplitRefused):
        provider.submit_sync(request)
    assert transport.calls == []


# ---------------------------------------------------------------------------
# Batch: submit, find, poll, fetch, expired (acceptance criterion 1)
# ---------------------------------------------------------------------------


def test_batch_submit_uploads_jsonl_and_creates_batch_with_submit_ref_metadata():
    upload_response = _ok(json.dumps({"id": "file-in-1"}).encode("utf-8"))
    create_response = _ok(json.dumps({"id": "batch-1", "status": "validating"}).encode("utf-8"))
    transport = ScriptedTransport(
        [
            (_method_and_path_startswith("POST", "/v1/files"), upload_response),
            (_method_and_path_startswith("POST", "/v1/batches"), create_response),
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    requests = [_tool_call_request("case-1"), _tool_call_request("case-2")]
    handle = provider.submit_batch(requests, "submit-ref-abc")

    assert handle.batch_id == "batch-1"
    assert handle.submit_ref == "submit-ref-abc"

    # First call: the multipart file upload for /v1/files.
    file_call = transport.calls[0]
    assert file_call["url"] == "https://api.openai.com/v1/files"
    assert b'name="purpose"' in file_call["data"]
    assert b"batch" in file_call["data"]
    lines = [
        line for line in file_call["data"].split(b"\n") if line.strip().startswith(b'{"custom_id"')
    ]
    assert len(lines) == 2
    first_line = json.loads(lines[0])
    assert first_line["custom_id"] == "case-1::tool_call"
    assert first_line["method"] == "POST"
    assert first_line["url"] == "/v1/responses"
    assert first_line["body"]["metadata"]["nvsh_case_id"] == "case-1"

    # Second call: create the batch, with our submit_ref in metadata.
    batch_call = transport.calls[1]
    assert batch_call["url"] == "https://api.openai.com/v1/batches"
    sent = json.loads(batch_call["data"])
    assert sent["input_file_id"] == "file-in-1"
    assert sent["endpoint"] == "/v1/responses"
    assert sent["completion_window"] == "24h"
    assert sent["metadata"] == {"nvsh_submit_ref": "submit-ref-abc"}


def test_batch_submit_hashes_long_case_ids_into_a_64_char_custom_id():
    long_case_id = "case-" + "x" * 100
    upload_response = _ok(json.dumps({"id": "file-in-2"}).encode("utf-8"))
    create_response = _ok(json.dumps({"id": "batch-2"}).encode("utf-8"))
    transport = ScriptedTransport(
        [
            (_method_and_path_startswith("POST", "/v1/files"), upload_response),
            (_method_and_path_startswith("POST", "/v1/batches"), create_response),
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    provider.submit_batch([_tool_call_request(long_case_id)], "ref-2")

    file_call = transport.calls[0]
    lines = [
        line for line in file_call["data"].split(b"\n") if line.strip().startswith(b'{"custom_id"')
    ]
    custom_id = json.loads(lines[0])["custom_id"]
    assert len(custom_id) <= 64
    assert custom_id != f"{long_case_id}::tool_call"


def test_find_batch_matches_submit_ref_in_metadata():
    list_response = _ok(
        json.dumps(
            {
                "data": [
                    {"id": "batch-other", "metadata": {"nvsh_submit_ref": "not-this-one"}},
                    {"id": "batch-3", "metadata": {"nvsh_submit_ref": "ref-3"}},
                ],
                "has_more": False,
            }
        ).encode("utf-8")
    )
    transport = ScriptedTransport(
        [(_method_and_path_startswith("GET", "/v1/batches"), list_response)]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    handle = provider.find_batch("ref-3")
    assert handle is not None
    assert handle.batch_id == "batch-3"
    assert handle.submit_ref == "ref-3"


def test_find_batch_returns_none_when_no_batch_matches():
    list_response = _ok(json.dumps({"data": [], "has_more": False}).encode("utf-8"))
    transport = ScriptedTransport(
        [(_method_and_path_startswith("GET", "/v1/batches"), list_response)]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    assert provider.find_batch("ref-missing") is None


@pytest.mark.parametrize(
    "status,expected_complete,expected_expired",
    [
        ("validating", False, False),
        ("in_progress", False, False),
        ("finalizing", False, False),
        ("completed", True, False),
        ("expired", True, True),
        ("failed", True, True),
        ("cancelled", True, True),
    ],
)
def test_poll_batch_maps_status(status, expected_complete, expected_expired):
    response = _ok(json.dumps({"id": "batch-4", "status": status}).encode("utf-8"))
    transport = ScriptedTransport(
        [(_method_and_path_startswith("GET", "/v1/batches/batch-4"), response)]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    handle = base.BatchHandle(batch_id="batch-4", provider=provider.name, submit_ref="ref-4")
    result = provider.poll_batch(handle)
    assert result.complete is expected_complete
    assert result.expired is expected_expired


def test_fetch_batch_collects_output_and_error_files():
    batch_get = _ok(
        json.dumps(
            {
                "id": "batch-5",
                "status": "completed",
                "output_file_id": "file-out-5",
                "error_file_id": "file-err-5",
            }
        ).encode("utf-8")
    )
    output_content = _ok(_fixture_bytes("batch_output.jsonl"))
    error_content = _ok(_fixture_bytes("batch_error.jsonl"))
    transport = ScriptedTransport(
        [
            (_method_and_path_startswith("GET", "/v1/batches/batch-5"), batch_get),
            (
                _method_and_path_startswith("GET", "/v1/files/file-out-5/content"),
                output_content,
            ),
            (
                _method_and_path_startswith("GET", "/v1/files/file-err-5/content"),
                error_content,
            ),
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    handle = base.BatchHandle(batch_id="batch-5", provider=provider.name, submit_ref="ref-5")
    results = provider.fetch_batch(handle)

    by_case = {r.case_id: r for r in results}
    assert by_case["case-1"].outcome is errors.Outcome.OK
    assert json.loads(by_case["case-1"].answer) == {
        "name": "propose",
        "arguments": {
            "operation": "service_restart",
            "arguments": {"service": "synthetic.service"},
        },
    }
    assert by_case["case-1"].usage["reasoning_tokens"] == 10

    assert by_case["case-2"].outcome is errors.Outcome.PENDING
    assert by_case["case-2"].reason.startswith(errors.REQUEST_REJECTED_PREFIX)


def test_fetch_batch_recovers_case_id_from_response_metadata_after_process_restart():
    """Simulates find_batch after a crash: a *fresh* provider instance with
    no in-memory custom_id map must still recover case_id/interface from
    each response body's own ``metadata`` (this adapter's crash-recovery
    design; see the module docstring)."""
    batch_get = _ok(
        json.dumps(
            {
                "id": "batch-6",
                "status": "completed",
                "output_file_id": "file-out-6",
                "error_file_id": None,
            }
        ).encode("utf-8")
    )
    output_content = _ok(_fixture_bytes("batch_output.jsonl"))
    transport = ScriptedTransport(
        [
            (_method_and_path_startswith("GET", "/v1/batches/batch-6"), batch_get),
            (_method_and_path_startswith("GET", "/v1/files/file-out-6/content"), output_content),
        ]
    )
    # A brand-new provider: no _send_batch was ever called on it, so its
    # _local_custom_ids dict is empty -- exactly the post-crash situation.
    fresh_provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    handle = base.BatchHandle(batch_id="batch-6", provider=fresh_provider.name, submit_ref="ref-6")
    results = fresh_provider.fetch_batch(handle)

    assert len(results) == 1
    assert results[0].case_id == "case-1"
    assert results[0].outcome is errors.Outcome.OK


def test_expired_batch_poll_and_fetch_yields_only_error_results():
    batch_get_status = _ok(json.dumps({"id": "batch-7", "status": "expired"}).encode("utf-8"))
    provider_for_poll = openai_provider.OpenAIProvider(
        "gpt-6-luna",
        transport=ScriptedTransport(
            [(_method_and_path_startswith("GET", "/v1/batches/batch-7"), batch_get_status)]
        ),
    )
    handle = base.BatchHandle(
        batch_id="batch-7", provider=provider_for_poll.name, submit_ref="ref-7"
    )
    status = provider_for_poll.poll_batch(handle)
    assert status.complete is True
    assert status.expired is True

    # An expired batch has no output file, only (possibly) an error file
    # OpenAI writes describing the expiry; fetch_batch must not crash when
    # output_file_id is absent.
    batch_get_full = _ok(
        json.dumps(
            {
                "id": "batch-7",
                "status": "expired",
                "output_file_id": None,
                "error_file_id": "file-err-7",
            }
        ).encode("utf-8")
    )
    error_content = _ok(_fixture_bytes("batch_error.jsonl"))
    provider_for_fetch = openai_provider.OpenAIProvider(
        "gpt-6-luna",
        transport=ScriptedTransport(
            [
                (_method_and_path_startswith("GET", "/v1/batches/batch-7"), batch_get_full),
                (
                    _method_and_path_startswith("GET", "/v1/files/file-err-7/content"),
                    error_content,
                ),
            ]
        ),
    )
    results = provider_for_fetch.fetch_batch(handle)
    assert len(results) == 1
    assert results[0].outcome is errors.Outcome.PENDING


# ---------------------------------------------------------------------------
# Review fix P2: batch HTTP rejections keep their classification.
# ---------------------------------------------------------------------------


def _fetch_lines(lines: list[dict], *, error_file: bool) -> list:
    body = ("\n".join(json.dumps(line) for line in lines) + "\n").encode("utf-8")
    file_key = "error_file_id" if error_file else "output_file_id"
    batch_get = _ok(json.dumps({"id": "batch-9", "status": "completed", file_key: "f9"}).encode())
    transport = ScriptedTransport(
        [
            (_method_and_path_startswith("GET", "/v1/batches/batch-9"), batch_get),
            (_method_and_path_startswith("GET", "/v1/files/f9/content"), _ok(body)),
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    handle = base.BatchHandle(batch_id="batch-9", provider=provider.name, submit_ref="ref-9")
    return provider.fetch_batch(handle)


@pytest.mark.parametrize("error_file", [True, False])
def test_batch_line_with_http_400_and_null_error_is_request_rejected(error_file):
    line = {
        "id": "batch_req_9",
        "custom_id": "case-9::tool_call",
        "response": {
            "status_code": 400,
            "request_id": "req_9",
            "body": {
                "error": {
                    "message": "Unsupported parameter.",
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
        },
        "error": None,
    }
    (result,) = _fetch_lines([line], error_file=error_file)
    assert result.case_id == "case-9"
    assert result.outcome is errors.Outcome.PENDING
    assert result.reason == errors.REQUEST_REJECTED_PREFIX + "bad_request"
    assert result.answer is None


def test_batch_line_with_http_429_stays_retryable_pending():
    line = {
        "custom_id": "case-9::tool_call",
        "response": {"status_code": 429, "body": {"error": {"type": "rate_limit_error"}}},
        "error": None,
    }
    (result,) = _fetch_lines([line], error_file=True)
    assert result.outcome is errors.Outcome.PENDING
    assert result.reason == "rate_limited"


# ---------------------------------------------------------------------------
# Review fix P2: a failed Responses object is infrastructure, not a bad answer.
# ---------------------------------------------------------------------------

FAILED_RESPONSE = {
    "id": "resp_failed_1",
    "model": "gpt-6-luna",
    "status": "failed",
    "error": {"code": "server_error", "message": "The server had an error."},
    "output": [],
    "metadata": {"nvsh_case_id": "case-9", "nvsh_interface": "tool_call"},
}


def test_sync_failed_response_is_a_pending_stop_not_an_invalid_answer():
    transport = ScriptedTransport(
        [(_path_is("/v1/responses"), _ok(json.dumps(FAILED_RESPONSE).encode()))]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    with pytest.raises(openai_provider.OpenAIProviderError) as caught:
        provider.submit_sync(_tool_call_request())
    assert caught.value.classification.outcome is errors.Outcome.PENDING
    assert caught.value.classification.retryable is True


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_batch_failed_response_is_pending_not_invalid(status):
    body = dict(FAILED_RESPONSE, status=status)
    line = {
        "custom_id": "case-9::tool_call",
        "response": {"status_code": 200, "body": body},
        "error": None,
    }
    (result,) = _fetch_lines([line], error_file=False)
    assert result.outcome is errors.Outcome.PENDING
    assert result.outcome is not errors.Outcome.INVALID


def test_find_batch_is_unresolved_when_the_page_bound_runs_out(monkeypatch):
    monkeypatch.setattr(openai_provider, "MAX_LIST_PAGES", 2)
    page = _ok(
        json.dumps({"data": [{"id": "batch-x", "metadata": {}}], "has_more": True}).encode("utf-8")
    )
    transport = ScriptedTransport(
        [
            (_method_and_path_startswith("GET", "/v1/batches"), page),
            (_method_and_path_startswith("GET", "/v1/batches"), page),
        ]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    with pytest.raises(base.BatchLookupUnresolved):
        provider.find_batch("ref-missing")


# ---------------------------------------------------------------------------
# Deviation d1: history rendered natively; cached answers re-read.
# ---------------------------------------------------------------------------

HISTORY = (
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_0", "name": "service_status", "arguments": '{"service": "x.service"}'}
        ],
    },
    {"role": "tool", "tool_call_id": "call_0", "content": "exit 0\nactive"},
)


def test_history_renders_as_function_call_items():
    transport = ScriptedTransport(
        [(_path_is("/v1/responses"), _ok(_fixture_bytes("responses_propose.json")))]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    request = base.CallRequest(
        case_id="case-h",
        split="test",
        case_text="ask",
        prompt="sys",
        offered_candidates=OFFERED,
        params={"tools": REAL_TOOLS},
        history=HISTORY,
    )
    provider.submit_sync(request)
    sent = json.loads(transport.calls[0]["data"])
    assert sent["input"] == [
        {"role": "user", "content": "ask"},
        {
            "type": "function_call",
            "call_id": "call_0",
            "name": "service_status",
            "arguments": '{"service": "x.service"}',
        },
        {"type": "function_call_output", "call_id": "call_0", "output": "exit 0\nactive"},
    ]


def test_result_from_raw_rereads_the_cached_answer():
    transport = ScriptedTransport(
        [(_path_is("/v1/responses"), _ok(_fixture_bytes("responses_propose.json")))]
    )
    provider = openai_provider.OpenAIProvider("gpt-6-luna", transport=transport)
    fresh = provider.submit_sync(_tool_call_request())
    again = provider.result_from_raw(_tool_call_request(), fresh.raw)
    assert again == fresh


def test_a_restricted_key_missing_a_scope_is_reported_as_missing_scope():
    """Live smoke 2026-09-26: a restricted key without api.files.write 401s the batch upload."""
    body = json.dumps(
        {
            "error": {
                "message": "You have insufficient permissions for this operation. Missing "
                "scopes: api.files.write. Check that you have the correct role.",
                "type": "invalid_request_error",
            }
        }
    ).encode()
    classification = openai_provider._classify_http(401, body)
    assert classification.rejected and not classification.retryable
    assert classification.reason == "request_rejected:missing_scope:api.files.write"
    plain = openai_provider._classify_http(
        401, json.dumps({"error": {"code": "invalid_api_key"}}).encode()
    )
    assert "missing_scope" not in plain.reason
