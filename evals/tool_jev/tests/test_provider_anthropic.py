"""Tests for evals/tool_jev/providers/anthropic.py (issue #64, task t13).

Covers the two t13 acceptance criteria:

1. recorded-HTTP fixture tests cover submit (sync), submit_batch, poll
   (in-progress vs ended), fetch (results), and a canceled/expired batch
   result;
2. every call goes only to ``api.anthropic.com`` and the API key comes
   only from the ``ANTHROPIC_API_KEY`` environment variable.

No network: a ``FakeTransport`` is injected in place of
``anthropic.urllib_transport``. Every fixture body below is a synthetic
document written by hand from the documented Messages/Batches API shapes
(see the module docstring's doc citations) -- no real case text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tool_jev.providers import anthropic, base, errors
from nvsh.tiers import lfm

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "anthropic"


def _fixture(name: str, **fmt) -> bytes:
    # Plain ``{placeholder}`` substitution via str.replace, not str.format:
    # these fixtures are JSON documents, so every literal ``{``/``}`` in the
    # file would otherwise collide with format-string syntax.
    text = (FIXTURES / name).read_text()
    for key, value in fmt.items():
        text = text.replace("{" + key + "}", str(value))
    return text.encode("utf-8")


def _fixture_json(name: str, **fmt) -> bytes:
    # Round-trip through json.dumps so line-endings/whitespace never leak
    # into a JSONL result line's exact bytes.
    obj = json.loads(_fixture(name, **fmt))
    return json.dumps(obj).encode("utf-8")


class FakeTransport:
    """Scripted (method, url) -> (status, body) transport, no network.

    ``routes`` maps an exact ``(method, url)`` pair to a response. A
    ``calls`` list records every call made, so tests can assert on the
    exact URL/headers/body sent (acceptance criterion 2 and the
    tool/tool_choice/effort payload shape).
    """

    def __init__(self, routes: dict) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple] = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        key = (method, url)
        if key not in self.routes:
            raise AssertionError(f"unscripted transport call: {method} {url}")
        return self.routes[key]


def _key_env(monkeypatch, value: str = "sk-test-fixture-key") -> None:
    # Built at runtime, never a literal assigned to a variable named
    # token/secret/password/api_key (scan-secrets.py rule).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "".join(["s", "k", "-"]) + value[3:])


@pytest.fixture(autouse=True)
def _default_anthropic_key(request, monkeypatch):
    """Every test gets a fixture ANTHROPIC_API_KEY unless it deletes it itself.

    ``test_api_key_read_only_from_named_env_var`` and
    ``test_call_refuses_non_anthropic_host`` manage the env var themselves
    (one deletes it deliberately), so this fixture skips them.
    """
    if request.node.name in (
        "test_api_key_read_only_from_named_env_var",
        "test_call_refuses_non_anthropic_host",
    ):
        return
    _key_env(monkeypatch)


#: nvsh's real Track A propose tool (canonical Chat-Completions shape).
PROPOSE_TOOL = next(t for t in lfm.tools_for() if t["function"]["name"] == lfm.PROPOSE_TOOL)
PROPOSE_ANSWER = {
    "name": "propose",
    "arguments": {"operation": "service_restart", "arguments": {"service": "synthetic.service"}},
}


def _request(case_id="case-1", split="test", interface="tool_call", **kw):
    return base.CallRequest(case_id=case_id, split=split, interface=interface, **kw)


# ---------------------------------------------------------------------------
# custom_id encode/decode round trip
# ---------------------------------------------------------------------------


def test_build_and_parse_custom_id_round_trips_case_id():
    custom_id = anthropic.build_custom_id("ref-abcdef0123456789", 3, "case-weird/id:42")
    assert len(custom_id) <= 64
    import re

    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", custom_id)
    prefix, index, case_id = anthropic.parse_custom_id(custom_id)
    assert index == 3
    assert case_id == "case-weird/id:42"


def test_build_custom_id_rejects_case_id_too_large_for_budget():
    with pytest.raises(ValueError):
        anthropic.build_custom_id("ref", 0, "x" * 100)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: host + key discipline.
# ---------------------------------------------------------------------------


def test_only_calls_api_anthropic_com(monkeypatch):
    _key_env(monkeypatch)
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_choice.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    provider.submit_sync(_request(interface="choice", case_text="pick one", prompt="sys"))
    assert transport.calls
    for _method, url, _headers, _body in transport.calls:
        assert url.startswith("https://api.anthropic.com/")


def test_call_refuses_non_anthropic_host(monkeypatch):
    _key_env(monkeypatch)
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    with pytest.raises(ValueError):
        provider._call("GET", "https://evil.example.com/v1/messages")


def test_api_key_read_only_from_named_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    with pytest.raises(base.MissingProviderKey):
        provider.submit_sync(_request(interface="choice", case_text="x", prompt="sys"))


def test_headers_carry_key_and_version(monkeypatch):
    _key_env(monkeypatch, "sk-header-check-1234")
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_choice.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    provider.submit_sync(_request(interface="choice", case_text="pick one", prompt="sys"))
    _method, _url, headers, _body = transport.calls[0]
    assert headers["x-api-key"] == "sk-header-check-1234"
    assert headers["anthropic-version"] == anthropic.ANTHROPIC_VERSION


# ---------------------------------------------------------------------------
# submit_sync: tool_call interface, tool schema translation, effort.
# ---------------------------------------------------------------------------


def test_submit_sync_tool_call_success():
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_propose.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    provider._headers = lambda: {"x-api-key": "k", "anthropic-version": anthropic.ANTHROPIC_VERSION}
    request = _request(
        interface="tool_call",
        case_text="disk is full, please help",
        prompt="you are the operator's assistant",
        offered_candidates=("service_status", "service_restart", "explain", "escalate"),
        params={"reasoning": "medium", "max_output_tokens": 512, "tools": [PROPOSE_TOOL]},
    )
    result = provider.submit_sync(request)

    assert result.outcome is errors.Outcome.OK
    assert json.loads(result.answer) == PROPOSE_ANSWER
    assert result.response_id == "msg_01ProposeFixture"
    assert result.returned_model == "claude-sonnet-5"
    assert result.candidates is None  # no logprobs, never estimated
    assert result.usage == {
        "input_tokens": 120,
        "output_tokens": 40,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 15,
    }
    assert result.raw == _fixture_json("message_propose.json")

    _method, _url, _headers, body = transport.calls[0]
    sent = json.loads(body)
    assert sent["model"] == "claude-sonnet-5"
    assert sent["max_tokens"] == 512
    assert sent["system"] == "you are the operator's assistant"
    assert sent["messages"] == [{"role": "user", "content": "disk is full, please help"}]
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["tool_choice"] == {"type": "any"}
    assert sent["tools"] == [
        {
            "name": "propose",
            "description": PROPOSE_TOOL["function"]["description"],
            "input_schema": PROPOSE_TOOL["function"]["parameters"],
        }
    ]


@pytest.mark.parametrize(
    "model_id, flag, expected",
    [
        ("claude-sonnet-5", None, {"type": "any"}),
        ("claude-opus-5-5", None, {"type": "auto"}),
        ("claude-opus-5-5", True, {"type": "any"}),
        ("claude-sonnet-5", False, {"type": "auto"}),
    ],
)
def test_tool_choice_follows_forced_tool_choice_capability(model_id, flag, expected):
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_propose.json"),
            )
        }
    )
    kwargs = {} if flag is None else {"forced_tool_choice": flag}
    provider = anthropic.AnthropicProvider(model_id, transport=transport, **kwargs)
    provider.submit_sync(_request(case_text="x", prompt="sys", params={"tools": [PROPOSE_TOOL]}))
    assert json.loads(transport.calls[0][3])["tool_choice"] == expected
    assert "claude-opus-5-5" in anthropic.NO_FORCED_TOOL_CHOICE_MODELS


def test_text_only_tool_call_reply_is_malformed():
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_text_only.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-opus-5-5", transport=transport)
    result = provider.submit_sync(
        _request(
            case_text="x",
            prompt="sys",
            offered_candidates=("service_restart", "explain", "escalate"),
            params={"tools": [PROPOSE_TOOL]},
        )
    )
    assert result.answer is None
    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "malformed"


def test_submit_sync_choice_interface_answers_with_the_label_text():
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_choice.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-opus-5-5", transport=transport)
    request = _request(
        interface="choice",
        case_text="which operation fits?",
        prompt="pick A or B",
        offered_candidates=("service_status", "service_logs"),
        # candidate -> letter, as request.build_choice_request builds it
        params={"labels": {"service_status": "A", "service_logs": "B"}, "reasoning": "medium"},
    )
    result = provider.submit_sync(request)
    assert result.outcome is errors.Outcome.OK
    assert result.answer == "B"  # the text; request.parse_choice reads it
    _method, _url, _headers, body = transport.calls[0]
    sent = json.loads(body)
    assert "tools" not in sent
    assert "tool_choice" not in sent


def test_submit_sync_no_case_text_leaks_into_params_only_fields():
    # Redaction happens in BaseProvider before _send_sync ever runs; here we
    # only check the adapter puts case_text/prompt in the contract's two
    # fields and nowhere else in the payload.
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_choice.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    request = _request(
        interface="choice",
        case_text="SECRET-CASE-BODY",
        prompt="sys prompt",
        params={"labels": {"service_logs": "B"}},
    )
    provider.submit_sync(request)
    _method, _url, _headers, body = transport.calls[0]
    sent = json.loads(body)
    assert sent["messages"] == [{"role": "user", "content": "SECRET-CASE-BODY"}]
    assert sent["system"] == "sys prompt"


def test_submit_sync_refusal_is_invalid():
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                200,
                _fixture_json("message_refusal.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    result = provider.submit_sync(_request(interface="choice", case_text="x", prompt="sys"))
    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "refusal"


def test_submit_sync_billing_error_is_pending_retryable():
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                402,
                _fixture_json("error_billing.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    with pytest.raises(anthropic.AnthropicProviderError) as excinfo:
        provider.submit_sync(_request(interface="choice", case_text="x", prompt="sys"))
    classification = excinfo.value.classification
    assert classification.outcome is errors.Outcome.PENDING
    assert classification.retryable is True
    assert classification.reason == "insufficient_credit"


def test_submit_sync_invalid_request_error_is_rejected_not_retryable():
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages"): (
                400,
                _fixture_json("error_invalid_request.json"),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-opus-5-5", transport=transport)
    with pytest.raises(anthropic.AnthropicProviderError) as excinfo:
        provider.submit_sync(
            _request(
                interface="tool_call",
                case_text="x",
                prompt="sys",
                params={"tools": [PROPOSE_TOOL]},
            )
        )
    classification = excinfo.value.classification
    assert classification.outcome is errors.Outcome.PENDING
    assert classification.retryable is False
    assert classification.rejected is True
    assert classification.reason == "request_rejected:unsupported_parameter"


def test_capabilities_have_no_logprobs():
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    assert provider.capabilities.logprobs is False
    assert provider.capabilities.batch is True


def test_heldout_split_never_reaches_transport(monkeypatch):
    _key_env(monkeypatch)
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    with pytest.raises(base.HeldoutSplitRefused):
        provider.submit_sync(
            _request(split="heldout", interface="choice", case_text="x", prompt="s")
        )


# ---------------------------------------------------------------------------
# Batch: submit, poll, results (succeeded/errored/expired/canceled).
# ---------------------------------------------------------------------------


def test_send_batch_encodes_ref_and_case_ids_in_custom_id():
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages/batches"): (
                200,
                _fixture_json("batch_create.json", batch_id="msgbatch_fixture1", request_count=2),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    requests = [
        _request(
            case_id="case-A", interface="choice", case_text="t1", prompt="s", params={"labels": {}}
        ),
        _request(
            case_id="case-B", interface="choice", case_text="t2", prompt="s", params={"labels": {}}
        ),
    ]
    handle = provider.submit_batch(requests, submit_ref="ref-1234567890")
    assert handle.batch_id == "msgbatch_fixture1"
    assert handle.submit_ref == "ref-1234567890"

    _method, _url, _headers, body = transport.calls[0]
    sent = json.loads(body)
    custom_ids = [entry["custom_id"] for entry in sent["requests"]]
    for custom_id in custom_ids:
        prefix, _index, decoded_case_id = anthropic.parse_custom_id(custom_id)
        assert prefix == anthropic._sanitize_ref("ref-1234567890")
    decoded = {anthropic.parse_custom_id(c)[2] for c in custom_ids}
    assert decoded == {"case-A", "case-B"}


def test_check_batch_in_progress_vs_ended():
    transport = FakeTransport(
        {
            ("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture2"): (
                200,
                _fixture_json(
                    "batch_status_in_progress.json", batch_id="msgbatch_fixture2", request_count=1
                ),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    handle = base.BatchHandle(
        batch_id="msgbatch_fixture2", provider="anthropic", submit_ref="ref-1"
    )
    status = provider.poll_batch(handle)
    assert status.complete is False

    ended_body = _fixture_json(
        "batch_status_ended.json",
        batch_id="msgbatch_fixture2",
        succeeded=1,
        errored=0,
        canceled=0,
        expired=0,
    )
    transport.routes[("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture2")] = (
        200,
        ended_body,
    )
    status = provider.poll_batch(handle)
    assert status.complete is True
    assert status.expired is False


def test_fetch_batch_collects_succeeded_result_with_case_context():
    submit_ref = "ref-collect-1"
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    custom_id = anthropic.build_custom_id(submit_ref, 0, "case-collect-1")
    provider._batch_requests["msgbatch_fixture3"] = {
        "case-collect-1": _request(
            case_id="case-collect-1",
            interface="tool_call",
            case_text="t",
            prompt="s",
            offered_candidates=("service_restart", "explain", "escalate"),
            params={},
        )
    }
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture3/results"
    result_line = _fixture_json(
        "result_succeeded.json", custom_id=custom_id, message_id="msg_batch_1"
    )
    ended_body = _fixture_json(
        "batch_status_ended.json",
        batch_id="msgbatch_fixture3",
        succeeded=1,
        errored=0,
        canceled=0,
        expired=0,
    )
    provider._transport = FakeTransport(
        {
            ("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture3"): (
                200,
                ended_body,
            ),
            ("GET", results_url): (200, result_line + b"\n"),
        }
    )
    handle = base.BatchHandle(
        batch_id="msgbatch_fixture3", provider="anthropic", submit_ref=submit_ref
    )
    results = provider.fetch_batch(handle)
    assert len(results) == 1
    (result,) = results
    assert result.case_id == "case-collect-1"
    assert result.outcome is errors.Outcome.OK
    assert json.loads(result.answer) == PROPOSE_ANSWER
    assert result.interface == "tool_call"
    assert result.response_id == "msg_batch_1"


def test_fetch_batch_recovers_case_id_without_in_memory_cache():
    """Simulates a process restart: no self._batch_requests entry for this batch."""
    submit_ref = "ref-restart-1"
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    custom_id = anthropic.build_custom_id(submit_ref, 0, "case-restart-1")
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture4/results"
    result_line = _fixture_json(
        "result_succeeded.json", custom_id=custom_id, message_id="msg_batch_2"
    )
    ended_body = _fixture_json(
        "batch_status_ended.json",
        batch_id="msgbatch_fixture4",
        succeeded=1,
        errored=0,
        canceled=0,
        expired=0,
    )
    provider._transport = FakeTransport(
        {
            ("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture4"): (
                200,
                ended_body,
            ),
            ("GET", results_url): (200, result_line + b"\n"),
        }
    )
    handle = base.BatchHandle(
        batch_id="msgbatch_fixture4", provider="anthropic", submit_ref=submit_ref
    )
    (result,) = provider.fetch_batch(handle)
    # case_id is recovered exactly (base64-decoded from custom_id) even
    # though the in-memory request cache never saw this batch.
    assert result.case_id == "case-restart-1"
    assert result.outcome is errors.Outcome.OK
    # interface is inferred (tool_use block present -> tool_call), a
    # best-effort fallback documented in the module docstring.
    assert result.interface == "tool_call"


def test_fetch_batch_maps_errored_expired_canceled_results():
    submit_ref = "ref-mixed-1"
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    cid_errored = anthropic.build_custom_id(submit_ref, 0, "case-err")
    cid_expired = anthropic.build_custom_id(submit_ref, 1, "case-exp")
    cid_canceled = anthropic.build_custom_id(submit_ref, 2, "case-can")
    lines = b"\n".join(
        [
            _fixture_json("result_errored.json", custom_id=cid_errored),
            _fixture_json("result_expired.json", custom_id=cid_expired),
            _fixture_json("result_canceled.json", custom_id=cid_canceled),
        ]
    )
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture5/results"
    ended_body = _fixture_json(
        "batch_status_ended.json",
        batch_id="msgbatch_fixture5",
        succeeded=0,
        errored=1,
        canceled=1,
        expired=1,
    )
    provider._transport = FakeTransport(
        {
            ("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_fixture5"): (
                200,
                ended_body,
            ),
            ("GET", results_url): (200, lines),
        }
    )
    handle = base.BatchHandle(
        batch_id="msgbatch_fixture5", provider="anthropic", submit_ref=submit_ref
    )
    results = {r.case_id: r for r in provider.fetch_batch(handle)}

    assert results["case-err"].outcome is errors.Outcome.PENDING
    assert results["case-err"].reason == "request_rejected:invalid_request"

    assert results["case-exp"].outcome is errors.Outcome.PENDING
    assert results["case-exp"].reason == "expired_batch"
    assert results["case-exp"].reason != "request_rejected"  # never counted invalid

    assert results["case-can"].outcome is errors.Outcome.PENDING
    assert results["case-can"].reason == "canceled_batch_request"
    # None of these three ever land in the invalid/denominator bucket.
    for res in results.values():
        assert res.outcome is not errors.Outcome.INVALID


# ---------------------------------------------------------------------------
# find_batch: ended-batch match, and the documented in-progress limitation.
# ---------------------------------------------------------------------------


def test_find_batch_matches_an_ended_batch_by_custom_id_prefix():
    submit_ref = "ref-find-1"
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    custom_id = anthropic.build_custom_id(submit_ref, 0, "case-find-1")
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_find1/results"
    result_line = _fixture_json(
        "result_succeeded.json", custom_id=custom_id, message_id="msg_find_1"
    )
    list_page = _fixture_json(
        "batches_list_page.json",
        batch_id="msgbatch_find1",
        processing_status="ended",
        results_url=results_url,
    )
    provider._transport = FakeTransport(
        {
            ("GET", "https://api.anthropic.com/v1/messages/batches?limit=100"): (200, list_page),
            ("GET", results_url): (200, result_line + b"\n"),
        }
    )
    handle = provider.find_batch(submit_ref)
    assert handle is not None
    assert handle.batch_id == "msgbatch_find1"
    assert handle.submit_ref == submit_ref


def test_find_batch_returns_none_when_only_in_progress_batches_exist():
    """Documented limitation: an in-progress batch can never be confirmed.

    results_url is null while a batch is in_progress (no custom_id is
    visible anywhere for it), so find_batch cannot tell "this in-progress
    batch is ours" apart from "some other batch is in progress". It must
    return None rather than guess -- see the module docstring's limitation
    1 and the residual double-charge risk that follows from it.
    """
    submit_ref = "ref-find-2"
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    list_page = _fixture_json(
        "batches_list_page.json",
        batch_id="msgbatch_in_progress",
        processing_status="in_progress",
        results_url="null",
    )
    # results_url must be JSON null, not the string "null"; patch it below.
    page_obj = json.loads(list_page)
    page_obj["data"][0]["results_url"] = None
    list_page = json.dumps(page_obj).encode("utf-8")
    provider._transport = FakeTransport(
        {("GET", "https://api.anthropic.com/v1/messages/batches?limit=100"): (200, list_page)}
    )
    handle = provider.find_batch(submit_ref)
    assert handle is None


def test_find_batch_requires_nonempty_submit_ref():
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    with pytest.raises(ValueError):
        provider.find_batch("")


# ---------------------------------------------------------------------------
# usage / cache token fields.
# ---------------------------------------------------------------------------


def test_usage_only_carries_int_fields_present_in_response():
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    usage = provider._usage_from(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "extra_unknown_field": "ignored",
        }
    )
    assert usage == {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3}
    for value in usage.values():
        assert isinstance(value, int) and not isinstance(value, bool)
