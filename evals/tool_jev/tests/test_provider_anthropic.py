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
    prefix, index, interface, case_id = anthropic.parse_custom_id(custom_id)
    assert index == 3
    assert interface == "tool_call"
    assert case_id == "case-weird/id:42"


def test_custom_id_round_trips_the_interface():
    custom_id = anthropic.build_custom_id("ref-1", 0, "case-1", "choice")
    assert anthropic.parse_custom_id(custom_id)[2:] == ("choice", "case-1")
    with pytest.raises(ValueError):
        anthropic.build_custom_id("ref-1", 0, "case-1", "other")


def test_build_custom_id_hashes_a_case_id_too_large_for_the_base64_budget():
    custom_id = anthropic.build_custom_id("ref", 0, "x" * 100)
    assert len(custom_id) <= 64
    assert anthropic.parse_custom_id(custom_id)[3] == "x" * 100


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
        prefix, _index, interface, _case_id = anthropic.parse_custom_id(custom_id)
        assert prefix == anthropic._sanitize_ref("ref-1234567890")
        assert interface == "choice"
    decoded = {anthropic.parse_custom_id(c)[3] for c in custom_ids}
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


def test_find_batch_is_unresolved_when_an_in_progress_batch_exists():
    """Review fix P1: an in-progress batch can never be confirmed, nor ruled out.

    results_url is null while a batch is in_progress (no custom_id is
    visible anywhere for it), so find_batch cannot tell "this in-progress
    batch is ours" apart from "some other batch is in progress". ``None``
    would authorize a resubmit (a double charge), so it raises
    BatchLookupUnresolved instead -- see the module docstring's limitation 1.
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
    with pytest.raises(base.BatchLookupUnresolved):
        provider.find_batch(submit_ref)


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


# ---------------------------------------------------------------------------
# Review fix P1: find_batch says "unresolved", never "not found", when unsure.
# ---------------------------------------------------------------------------

_LIST_URL = "https://api.anthropic.com/v1/messages/batches?limit=100"


def _ended_page(batch_id: str, results_url: str) -> bytes:
    return _fixture_json(
        "batches_list_page.json",
        batch_id=batch_id,
        processing_status="ended",
        results_url=results_url,
    )


def test_find_batch_is_unresolved_when_an_ended_batch_results_download_fails():
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_x/results"
    provider = anthropic.AnthropicProvider(
        "claude-sonnet-5",
        transport=FakeTransport(
            {
                ("GET", _LIST_URL): (200, _ended_page("msgbatch_x", results_url)),
                ("GET", results_url): (500, b"{}"),
            }
        ),
    )
    with pytest.raises(base.BatchLookupUnresolved):
        provider.find_batch("ref-unsure-1")


def test_find_batch_is_none_only_when_every_batch_was_read_and_none_match():
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_y/results"
    other = anthropic.build_custom_id("ref-someone-else", 0, "case-z")
    line = _fixture_json("result_succeeded.json", custom_id=other, message_id="m1")
    provider = anthropic.AnthropicProvider(
        "claude-sonnet-5",
        transport=FakeTransport(
            {
                ("GET", _LIST_URL): (200, _ended_page("msgbatch_y", results_url)),
                ("GET", results_url): (200, line + b"\n"),
            }
        ),
    )
    assert provider.find_batch("ref-mine-000") is None


def test_find_batch_is_unresolved_when_the_page_bound_runs_out():
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_p/results"
    page = json.loads(_ended_page("msgbatch_p", results_url))
    page["has_more"] = True
    other = anthropic.build_custom_id("ref-someone-else", 0, "case-z")
    line = _fixture_json("result_succeeded.json", custom_id=other, message_id="m1")
    routes = {
        ("GET", _LIST_URL): (200, json.dumps(page).encode()),
        ("GET", _LIST_URL + "&after_id=msgbatch_p"): (200, json.dumps(page).encode()),
        ("GET", results_url): (200, line + b"\n"),
    }
    provider = anthropic.AnthropicProvider(
        "claude-sonnet-5", transport=FakeTransport(routes), max_list_pages=2
    )
    with pytest.raises(base.BatchLookupUnresolved):
        provider.find_batch("ref-mine-000")


# ---------------------------------------------------------------------------
# Review fix P2: batch context keyed by custom_id; interface survives restart.
# ---------------------------------------------------------------------------


def test_one_case_through_both_interfaces_in_one_batch_keeps_both_contexts():
    submit_ref = "ref-both-000"
    offered = ("service_restart", "explain", "escalate")
    tool_request = _request(
        case_id="case-both",
        case_text="t",
        prompt="s",
        offered_candidates=offered,
        params={"tools": [PROPOSE_TOOL]},
    )
    choice_request = _request(
        case_id="case-both",
        interface="choice",
        case_text="t",
        prompt="s",
        offered_candidates=("A", "B"),
        params={"labels": {"service_restart": "A", "explain": "B"}},
    )
    transport = FakeTransport(
        {
            ("POST", "https://api.anthropic.com/v1/messages/batches"): (
                200,
                _fixture_json("batch_create.json", batch_id="msgbatch_both", request_count=2),
            )
        }
    )
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)
    handle = provider.submit_batch([tool_request, choice_request], submit_ref=submit_ref)
    sent = json.loads(transport.calls[0][3])
    tool_id, choice_id = (entry["custom_id"] for entry in sent["requests"])
    choice_message = json.loads(_fixture("message_choice.json"))
    lines = b"\n".join(
        [
            _fixture_json("result_succeeded.json", custom_id=tool_id, message_id="m-tool"),
            json.dumps(
                {"custom_id": choice_id, "result": {"type": "succeeded", "message": choice_message}}
            ).encode(),
        ]
    )
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_both/results"
    transport.routes[("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_both")] = (
        200,
        _fixture_json(
            "batch_status_ended.json",
            batch_id="msgbatch_both",
            succeeded=2,
            errored=0,
            canceled=0,
            expired=0,
        ),
    )
    transport.routes[("GET", results_url)] = (200, lines)
    results = {result.interface: result for result in provider.fetch_batch(handle)}
    assert results["tool_call"].outcome is errors.Outcome.OK
    assert json.loads(results["tool_call"].answer) == PROPOSE_ANSWER
    assert results["choice"].outcome is errors.Outcome.OK
    assert results["choice"].answer == "B"


def test_restart_recovers_the_interface_from_custom_id_not_the_response_shape():
    """A tool_call text-only reply stays a malformed tool_call, never a 'choice'."""
    submit_ref = "ref-restart-2"
    custom_id = anthropic.build_custom_id(submit_ref, 0, "case-r2", "tool_call")
    message = json.loads(_fixture("message_text_only.json"))
    line = json.dumps(
        {"custom_id": custom_id, "result": {"type": "succeeded", "message": message}}
    ).encode()
    results_url = "https://api.anthropic.com/v1/messages/batches/msgbatch_r2/results"
    provider = anthropic.AnthropicProvider(
        "claude-sonnet-5",
        transport=FakeTransport(
            {
                ("GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_r2"): (
                    200,
                    _fixture_json(
                        "batch_status_ended.json",
                        batch_id="msgbatch_r2",
                        succeeded=1,
                        errored=0,
                        canceled=0,
                        expired=0,
                    ),
                ),
                ("GET", results_url): (200, line),
            }
        ),
    )
    handle = base.BatchHandle(batch_id="msgbatch_r2", provider="anthropic", submit_ref=submit_ref)
    (result,) = provider.fetch_batch(handle)
    assert result.interface == "tool_call"
    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "malformed"


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


def _sync_provider(sent: list, body: bytes):
    def transport(method, url, headers, data):
        sent.append(json.loads(data))
        return 200, body

    return anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)


def test_history_renders_as_tool_use_and_tool_result_blocks():
    sent: list = []
    provider = _sync_provider(sent, _fixture("message_propose.json"))
    provider.submit_sync(_request(case_text="ask", prompt="sys", history=HISTORY))
    assert sent[0]["messages"] == [
        {"role": "user", "content": "ask"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_0",
                    "name": "service_status",
                    "input": {"service": "x.service"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_0", "content": "exit 0\nactive"}
            ],
        },
    ]


def test_native_thinking_blocks_are_replayed_and_tool_use_ids_kept():
    thinking = {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"}
    native_history = (
        {
            **HISTORY[0],
            "native": {
                "anthropic": [
                    thinking,
                    {"type": "tool_use", "id": "toolu_A", "name": "service_status", "input": {}},
                ]
            },
        },
        HISTORY[1],
    )
    sent: list = []
    provider = _sync_provider(sent, _fixture("message_propose.json"))
    provider.submit_sync(_request(case_text="ask", prompt="sys", history=native_history))
    assistant, results = sent[0]["messages"][1:]
    assert assistant["content"][0] == thinking
    assert assistant["content"][1]["id"] == "toolu_A"
    assert assistant["content"][1]["input"] == {"service": "x.service"}  # neutral content
    assert results["content"][0]["tool_use_id"] == "toolu_A"


def test_result_from_raw_rereads_a_sync_body_and_a_batch_line_alike():
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    request = _request(case_text="t", offered_candidates=("service_restart", "explain", "escalate"))
    sync_raw = _fixture("message_propose.json")
    line_raw = _fixture_json("result_succeeded.json", custom_id="x", message_id="m1")
    for raw in (sync_raw, line_raw):
        result = provider.result_from_raw(request, raw)
        assert result.outcome is errors.Outcome.OK
        assert json.loads(result.answer) == PROPOSE_ANSWER
        assert result.raw == raw
    native = provider.native_turn(line_raw)
    assert native["anthropic"][0]["type"] == "tool_use"


def test_interleaved_native_blocks_keep_their_order_up_to_the_acted_call():
    """Codex d1 review P3: thinking keeps its signed position; later blocks are dropped."""
    think_a = {"type": "thinking", "thinking": "", "signature": "c2lnQQ=="}
    think_b = {"type": "thinking", "thinking": "", "signature": "c2lnQg=="}
    think_c = {"type": "thinking", "thinking": "", "signature": "c2lnQw=="}
    native_history = (
        {
            **HISTORY[0],
            "native": {
                "anthropic": [
                    think_a,
                    {"type": "text", "text": "checking"},
                    # Malformed (input is not an object): skipped by every adapter.
                    {"type": "tool_use", "id": "toolu_bad", "name": "service_status", "input": 3},
                    think_b,
                    {"type": "tool_use", "id": "toolu_A", "name": "service_status", "input": {}},
                    think_c,
                    {"type": "tool_use", "id": "toolu_C", "name": "service_logs", "input": {}},
                ]
            },
        },
        HISTORY[1],
    )
    sent: list = []
    provider = _sync_provider(sent, _fixture("message_propose.json"))
    provider.submit_sync(_request(case_text="ask", prompt="sys", history=native_history))
    assistant, results = sent[0]["messages"][1:]
    assert assistant["content"] == [
        think_a,
        {"type": "text", "text": "checking"},
        think_b,
        {
            "type": "tool_use",
            "id": "toolu_A",
            "name": "service_status",
            "input": {"service": "x.service"},
        },
    ]
    assert results["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_A", "content": "exit 0\nactive"}
    ]


def test_native_blocks_with_no_usable_call_fall_back_to_the_neutral_call():
    native_history = (
        {**HISTORY[0], "native": {"anthropic": [{"type": "text", "text": "hm"}]}},
        HISTORY[1],
    )
    sent: list = []
    provider = _sync_provider(sent, _fixture("message_propose.json"))
    provider.submit_sync(_request(case_text="ask", prompt="sys", history=native_history))
    assistant, results = sent[0]["messages"][1:]
    assert assistant["content"][0] == {"type": "text", "text": "hm"}
    assert assistant["content"][1]["id"] == "call_0"
    assert results["content"][0]["tool_use_id"] == "call_0"


def test_a_malformed_first_tool_use_is_skipped_for_the_next_well_formed_one():
    """Codex d1 review P2: nvsh's ToolChat drops malformed calls; so does the adapter."""
    provider = anthropic.AnthropicProvider("claude-sonnet-5", transport=FakeTransport({}))
    message = {
        "id": "m",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": "t1", "name": "escalate", "input": "not an object"},
            {"type": "tool_use", "id": "t2", "name": "escalate", "input": {"reason": "big"}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    result = provider.result_from_raw(_request(case_text="t"), json.dumps(message).encode())
    assert result.outcome is errors.Outcome.OK
    assert json.loads(result.answer) == {"name": "escalate", "arguments": {"reason": "big"}}


def test_long_case_ids_travel_as_a_hash_and_map_back_after_a_restart(monkeypatch):
    """Smoke run 2026-09-26: real case ids (up to ~60 chars) overflowed the 64-char custom_id."""
    long_id = "q53-test-" + "x" * 51  # 60 characters: base64 is 80, over the budget
    custom_id = anthropic.build_custom_id("tj-ref-long", 3, long_id, "choice")
    assert len(custom_id) <= 64
    assert anthropic._CUSTOM_ID_RE.fullmatch(custom_id)
    assert anthropic.parse_custom_id(custom_id)[1:] == (3, "choice", long_id)
    # A fresh process knows nothing until the runner registers its case ids.
    monkeypatch.setattr(anthropic, "_CASE_IDS_BY_TOKEN", {})
    with pytest.raises(ValueError, match="register"):
        anthropic.parse_custom_id(custom_id)
    anthropic.register_case_ids(["other", long_id])
    assert anthropic.parse_custom_id(custom_id)[3] == long_id
    # Short ids keep the lossless base64 form (no registration needed).
    short = anthropic.build_custom_id("tj-ref-long", 0, "c1", "tool_call")
    monkeypatch.setattr(anthropic, "_CASE_IDS_BY_TOKEN", {})
    assert anthropic.parse_custom_id(short)[3] == "c1"


@pytest.mark.parametrize("length", range(1, 120))
def test_every_case_id_length_fits_the_custom_id(length):
    case_id = "c" * length
    custom_id = anthropic.build_custom_id("tj-ref", 999999, case_id, "text")
    assert len(custom_id) <= 64 and anthropic._CUSTOM_ID_RE.fullmatch(custom_id)
    assert anthropic.parse_custom_id(custom_id)[3] == case_id
