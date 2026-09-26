"""Tests for evals/tool_jev/providers/openai_compat.py (issue #64, t14).

No network: every test injects a ``transport`` callable that returns canned
``(status, bytes)`` pairs read from ``fixtures/openai_compat/*.json`` --
recorded shapes hand-built from the provider docs cited in the module
docstring, never a real HTTP call. Fixtures carry only synthetic text.

Covers the two t14 acceptance criteria:

1. fixture tests: logprobs present -> candidate distribution; absent ->
   not_measurable (``candidates is None``); 429 -> pending.
2. base URLs come from code defaults (asserted directly) or an explicit
   override, with the localhost/hosted split enforced both ways.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tool_jev.providers import base, errors, openai_compat
from nvsh.tiers import lfm

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "openai_compat"

#: nvsh's real Track A tools, as request.build_tool_call_request offers them.
REAL_TOOLS = lfm.tools_for()
TOOL_OFFERED = ("service_status", "service_restart", "explain", "escalate")


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _make_transport(status: int, raw: bytes, *, calls: list | None = None):
    def transport(url, headers, body):
        if calls is not None:
            calls.append((url, headers, body))
        return status, raw

    return transport


def _tool_call_request(**overrides) -> base.CallRequest:
    defaults = dict(
        case_id="case-tool-1",
        split="test",
        case_text="synthetic case body",
        prompt="synthetic system prompt",
        interface="tool_call",
        offered_candidates=TOOL_OFFERED,
        params={"tools": REAL_TOOLS},
    )
    defaults.update(overrides)
    return base.CallRequest(**defaults)


def _choice_request(**overrides) -> base.CallRequest:
    defaults = dict(
        case_id="case-choice-1",
        split="test",
        case_text="synthetic case body",
        prompt="synthetic system prompt",
        interface="choice",
        offered_candidates=("service_status", "service_logs", "service_restart"),
        # candidate -> letter, as request.build_choice_request builds it.
        params={"labels": {"service_status": "A", "service_logs": "B", "service_restart": "C"}},
    )
    defaults.update(overrides)
    return base.CallRequest(**defaults)


@pytest.fixture(autouse=True)
def _no_real_api_keys(monkeypatch):
    for env_name in ("OPEN_ROUTER_API_KEY", "NGC_API_KEY", "LOBES_GATEWAY_API_KEY"):
        monkeypatch.setenv(env_name, "k" * 8)


# ---------------------------------------------------------------------------
# Construction / base URL + key-env defaults.
# ---------------------------------------------------------------------------


def test_default_base_urls_and_key_envs_per_kind():
    provider = openai_compat.OpenAICompatProvider("openrouter", "qwen/qwen3.8-max-0902")
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider._api_key_env == "OPEN_ROUTER_API_KEY"

    provider = openai_compat.OpenAICompatProvider("nvidia", "moonshotai/kimi-k3")
    assert provider.base_url == "https://integrate.api.nvidia.com/v1"
    assert provider._api_key_env == "NGC_API_KEY"

    provider = openai_compat.OpenAICompatProvider("local", "unsloth/Qwen3.8-27B-NVFP4")
    assert provider.base_url == "http://localhost:8001/v1"
    assert provider._api_key_env == "LOBES_GATEWAY_API_KEY"


def test_local_kind_accepts_any_localhost_override():
    provider = openai_compat.OpenAICompatProvider("local", "m", base_url="http://127.0.0.1:9009/v1")
    assert provider.base_url == "http://127.0.0.1:9009/v1"


def test_local_kind_rejects_non_localhost_override():
    with pytest.raises(ValueError, match="localhost"):
        openai_compat.OpenAICompatProvider("local", "m", base_url="https://example.com/v1")


def test_hosted_kind_rejects_localhost_override():
    with pytest.raises(ValueError, match="localhost"):
        openai_compat.OpenAICompatProvider("nvidia", "m", base_url="http://localhost:8001/v1")


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        openai_compat.OpenAICompatProvider("bogus", "m")


def test_capabilities_batch_true_rejected():
    with pytest.raises(ValueError, match="batch"):
        openai_compat.OpenAICompatProvider(
            "openrouter",
            "m",
            capabilities=base.ProviderCapabilities(logprobs=True, batch=True, reasoning=False),
        )


def test_api_key_env_none_omits_authorization_header():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "local",
        "m",
        api_key_env=None,
        transport=_make_transport(200, _load("choice_no_logprobs.json"), calls=calls),
    )
    provider.submit_sync(_choice_request())
    assert "Authorization" not in calls[0][1]


# ---------------------------------------------------------------------------
# Criterion 1a: logprobs present -> candidate distribution.
# ---------------------------------------------------------------------------


def test_choice_with_logprobs_yields_candidate_distribution():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "qwen/qwen3.8-max-0902",
        capabilities=base.ProviderCapabilities(logprobs=True, batch=False, reasoning=False),
        transport=_make_transport(200, _load("choice_logprobs_success.json"), calls=calls),
    )
    result = provider.submit_sync(_choice_request())

    assert result.outcome == errors.Outcome.OK
    assert result.answer == "A"  # the text; request.parse_choice reads the label
    assert result.reason == "answer"
    assert result.candidates is not None
    assert pytest.approx(sum(result.candidates.values()), rel=1e-6) == 1.0
    # Offered order preserved (dict insertion order of params["labels"]).
    assert list(result.candidates) == ["service_status", "service_logs", "service_restart"]
    assert result.candidates["service_status"] == pytest.approx(0.9 / 1.01, rel=1e-6)
    assert result.returned_model == "qwen/qwen3.8-max-0902"
    assert result.response_id == "gen-synthetic-choice-1"
    assert result.usage == {"prompt_tokens": 200, "completion_tokens": 1, "reasoning_tokens": 40}

    # The request body actually asked for logprobs.
    sent_body = json.loads(calls[0][2])
    assert sent_body["logprobs"] is True
    assert sent_body["top_logprobs"] == 20
    assert sent_body["messages"] == [
        {"role": "system", "content": "synthetic system prompt"},
        {"role": "user", "content": "synthetic case body"},
    ]


def test_logprobs_taken_from_first_content_token_not_reasoning():
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "nvidia/nemotron-3-super-120b-a12b",
        capabilities=base.ProviderCapabilities(logprobs=True, batch=False, reasoning=False),
        transport=_make_transport(200, _load("choice_reasoning_then_content.json")),
    )
    result = provider.submit_sync(_choice_request())
    assert result.answer == "B"
    assert result.candidates is not None
    assert set(result.candidates) == {"service_status", "service_logs", "service_restart"}
    candidates = result.candidates
    assert candidates["service_logs"] > candidates["service_status"] > candidates["service_restart"]
    assert pytest.approx(sum(result.candidates.values()), rel=1e-6) == 1.0
    assert result.usage["reasoning_tokens"] == 512


# ---------------------------------------------------------------------------
# Criterion 1b: logprobs absent (or capability off, or a label missing from
# top_logprobs) -> candidates is None (never estimated) => not_measurable
# downstream.
# ---------------------------------------------------------------------------


def test_choice_without_logprobs_capability_yields_none_candidates():
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "nvidia/nemotron-3-ultra-550b-a55b",
        capabilities=base.ProviderCapabilities(logprobs=False, batch=False, reasoning=False),
        transport=_make_transport(200, _load("choice_no_logprobs.json")),
    )
    result = provider.submit_sync(_choice_request())
    assert result.outcome == errors.Outcome.OK
    assert result.answer == "B"
    assert result.candidates is None


def test_choice_missing_one_offered_label_in_top_logprobs_yields_none_candidates():
    """Only two of the three offered labels appear in top_logprobs (A, X) --
    the distribution must not be estimated for the missing labels (B/C),
    so the whole distribution is None."""
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "qwen/qwen3.8-max-0902",
        capabilities=base.ProviderCapabilities(logprobs=True, batch=False, reasoning=False),
        transport=_make_transport(200, _load("choice_missing_label_logprob.json")),
    )
    result = provider.submit_sync(_choice_request())
    assert result.outcome == errors.Outcome.OK
    assert result.answer == "A"
    assert result.candidates is None


# ---------------------------------------------------------------------------
# Criterion 1c: 429 -> pending, transient, naming the rate-limit reason.
# ---------------------------------------------------------------------------


def test_429_is_pending_and_retryable():
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "moonshotai/kimi-k3",
        transport=_make_transport(429, _load("error_429.json")),
    )
    with pytest.raises(openai_compat.OpenAICompatInfraError) as excinfo:
        provider.submit_sync(_tool_call_request())
    classification = excinfo.value.classification
    assert classification.outcome == errors.Outcome.PENDING
    assert classification.retryable is True
    assert classification.reason == "rate_limited"
    message = errors.stop_message(provider.name, classification, remaining=3)
    assert provider.name in message
    assert "3" in message


def test_400_unsupported_parameter_is_request_rejected_not_retryable():
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "nvidia/nemotron-3-ultra-550b-a55b",
        transport=_make_transport(400, _load("error_400_unsupported_parameter.json")),
    )
    with pytest.raises(openai_compat.OpenAICompatInfraError) as excinfo:
        provider.submit_sync(_choice_request())
    classification = excinfo.value.classification
    assert classification.rejected is True
    assert classification.retryable is False
    assert classification.reason == "request_rejected:unsupported_parameter"


def test_transport_error_network_loss_is_pending():
    def flaky_transport(url, headers, body):
        raise openai_compat.TransportError("network_error")

    provider = openai_compat.OpenAICompatProvider("openrouter", "m", transport=flaky_transport)
    with pytest.raises(openai_compat.OpenAICompatInfraError) as excinfo:
        provider.submit_sync(_tool_call_request())
    assert excinfo.value.classification.reason == "network_loss"
    assert excinfo.value.classification.retryable is True


# ---------------------------------------------------------------------------
# Tool-call track.
# ---------------------------------------------------------------------------


def test_tool_call_success_passes_raw_tool_call_through():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "qwen/qwen3.8-max-0902",
        transport=_make_transport(200, _load("tool_call_propose.json"), calls=calls),
    )
    result = provider.submit_sync(_tool_call_request())

    assert result.outcome == errors.Outcome.OK
    assert json.loads(result.answer) == {
        "name": "propose",
        "arguments": {
            "operation": "service_restart",
            "arguments": {"service": "synthetic.service"},
        },
    }
    assert result.interface == "tool_call"
    assert result.candidates is None
    assert result.usage == {"prompt_tokens": 120, "completion_tokens": 18, "reasoning_tokens": 0}

    sent_body = json.loads(calls[0][2])
    assert sent_body["tool_choice"] == "required"
    assert sent_body["tools"] == REAL_TOOLS


def test_forced_tool_choice_false_sends_auto():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "m",
        transport=_make_transport(200, _load("tool_call_propose.json"), calls=calls),
        forced_tool_choice=False,
    )
    provider.submit_sync(_tool_call_request())
    assert json.loads(calls[0][2])["tool_choice"] == "auto"


def test_text_only_tool_call_reply_is_malformed_with_no_answer():
    provider = openai_compat.OpenAICompatProvider(
        "openrouter", "m", transport=_make_transport(200, _load("tool_call_text_only.json"))
    )
    result = provider.submit_sync(_tool_call_request())
    assert result.answer is None
    assert result.outcome == errors.Outcome.INVALID
    assert result.reason == "malformed"


def test_private_logprob_helper_is_gone():
    # The shared request.distribution_from_logprobs is the only readout.
    assert not hasattr(openai_compat, "_distribution_from_logprobs")


def test_tool_call_refusal_is_invalid():
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "m",
        transport=_make_transport(200, _load("refusal.json")),
    )
    result = provider.submit_sync(_tool_call_request())
    assert result.outcome == errors.Outcome.INVALID
    assert result.reason == "refusal"


def test_choice_answer_outside_offered_set_is_invalid():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "m",
        transport=_make_transport(200, _load("choice_no_logprobs.json"), calls=calls),
    )
    request = _choice_request(
        offered_candidates=("service_status", "service_restart"),
        params={"labels": {"service_status": "A", "service_restart": "C"}},
    )
    result = provider.submit_sync(request)
    # The fixture answers label "B", which isn't one of this request's
    # labels (only A/C are offered): request.parse_choice never guesses it
    # into a pick -- an unrecognised letter is malformed.
    assert result.answer == "B"
    assert result.outcome == errors.Outcome.INVALID
    assert result.reason == "malformed"


# ---------------------------------------------------------------------------
# Reasoning param mapping.
# ---------------------------------------------------------------------------


def test_openrouter_reasoning_mapped_to_effort_object():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "m",
        transport=_make_transport(200, _load("tool_call_propose.json"), calls=calls),
    )
    request = _tool_call_request(
        params={
            "tools": REAL_TOOLS,
            "reasoning": "medium",
        }
    )
    provider.submit_sync(request)
    sent_body = json.loads(calls[0][2])
    assert sent_body["reasoning"] == {"effort": "medium"}
    assert sent_body["provider"] == {"data_collection": "deny"}


def test_nvidia_reasoning_passed_through_only_when_capability_says_so():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "m",
        capabilities=base.ProviderCapabilities(logprobs=False, batch=False, reasoning=True),
        transport=_make_transport(200, _load("tool_call_propose.json"), calls=calls),
    )
    request = _tool_call_request(
        params={
            "tools": REAL_TOOLS,
            "reasoning": "medium",
        }
    )
    provider.submit_sync(request)
    sent_body = json.loads(calls[0][2])
    assert sent_body["reasoning_effort"] == "medium"
    assert "reasoning" not in sent_body


def test_nvidia_reasoning_omitted_when_capability_false():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "m",
        capabilities=base.ProviderCapabilities(logprobs=False, batch=False, reasoning=False),
        transport=_make_transport(200, _load("tool_call_propose.json"), calls=calls),
    )
    request = _tool_call_request(
        params={
            "tools": REAL_TOOLS,
            "reasoning": "medium",
        }
    )
    provider.submit_sync(request)
    sent_body = json.loads(calls[0][2])
    assert "reasoning_effort" not in sent_body
    assert "reasoning" not in sent_body


# ---------------------------------------------------------------------------
# Redaction and the held-out guard are inherited from BaseProvider -- prove
# they still apply through this adapter (not re-testing base.py itself).
# ---------------------------------------------------------------------------


FAKE_KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz" + "123456"


def test_case_text_is_redacted_before_the_transport_sees_it():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "m",
        transport=_make_transport(200, _load("choice_no_logprobs.json"), calls=calls),
    )
    hostile = f"export OPENAI_API_KEY={FAKE_KEY}\nAuthorization: Bearer {FAKE_KEY}"
    provider.submit_sync(_choice_request(case_text=hostile))
    sent_body = json.loads(calls[0][2])
    user_message = sent_body["messages"][1]["content"]
    assert FAKE_KEY not in user_message
    assert "<REDACTED:" in user_message


def test_heldout_split_refused_before_any_transport_call():
    calls: list = []
    provider = openai_compat.OpenAICompatProvider(
        "openrouter",
        "m",
        transport=_make_transport(200, _load("choice_no_logprobs.json"), calls=calls),
    )
    with pytest.raises(base.HeldoutSplitRefused):
        provider.submit_sync(_choice_request(split="heldout"))
    assert calls == []


# ---------------------------------------------------------------------------
# No batch API.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name, args",
    [
        ("submit_batch", ([], "ref-1")),
        ("find_batch", ("ref-1",)),
    ],
)
def test_batch_methods_raise_not_implemented(method_name, args):
    provider = openai_compat.OpenAICompatProvider("openrouter", "m")
    method = getattr(provider, method_name)
    with pytest.raises(NotImplementedError):
        method(*args)


def test_poll_and_fetch_batch_raise_not_implemented():
    provider = openai_compat.OpenAICompatProvider("openrouter", "m")
    handle = base.BatchHandle(batch_id="b1", provider="openrouter:m", submit_ref="ref-1")
    with pytest.raises(NotImplementedError):
        provider.poll_batch(handle)
    with pytest.raises(NotImplementedError):
        provider.fetch_batch(handle)


# ---------------------------------------------------------------------------
# Rate limiter: no real sleeping, an injectable fake clock advances on sleep.
# ---------------------------------------------------------------------------


def test_rate_limiter_spaces_calls_without_real_sleep():
    fake_time = [0.0]

    def clock():
        return fake_time[0]

    def fake_sleep(seconds):
        fake_time[0] += seconds

    limiter = openai_compat.RateLimiter(60, clock=clock, sleep=fake_sleep)  # 1/sec
    limiter.acquire()
    assert fake_time[0] == 0.0
    limiter.acquire()
    # Second acquire had to "wait" one second -- via the fake sleep, not a
    # real one, so this test runs instantly.
    assert fake_time[0] == pytest.approx(1.0)
    limiter.acquire()
    assert fake_time[0] == pytest.approx(2.0)


def test_rate_limiter_used_by_provider_before_each_call():
    calls: list = []
    fake_time = [0.0]
    limiter = openai_compat.RateLimiter(
        120, clock=lambda: fake_time[0], sleep=lambda s: fake_time.__setitem__(0, fake_time[0] + s)
    )
    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "m",
        rate_limiter=limiter,
        transport=_make_transport(200, _load("choice_no_logprobs.json"), calls=calls),
    )
    provider.submit_sync(_tool_call_request())
    provider.submit_sync(_tool_call_request(case_id="case-tool-2"))
    assert len(calls) == 2
    assert fake_time[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Malformed JSON body (not an infra error -- a 200 with unparsable content).
# ---------------------------------------------------------------------------


def test_malformed_json_body_on_200_is_invalid_not_infra_error():
    provider = openai_compat.OpenAICompatProvider(
        "openrouter", "m", transport=_make_transport(200, b"not json at all")
    )
    result = provider.submit_sync(_tool_call_request())
    assert result.outcome == errors.Outcome.INVALID
    assert result.reason == "malformed"


def test_tool_call_missing_tool_calls_is_malformed():
    body = json.dumps(
        {
            "id": "x",
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "no tool call here"}}],
        }
    ).encode()
    provider = openai_compat.OpenAICompatProvider(
        "openrouter", "m", transport=_make_transport(200, body)
    )
    result = provider.submit_sync(_tool_call_request())
    assert result.outcome == errors.Outcome.INVALID
    assert result.reason == "malformed"
