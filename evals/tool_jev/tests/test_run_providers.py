"""The judge call path (t17): a free-text ``"text"`` interface on every adapter.

A judge call is neither a tool call nor a choice: the G-Eval prompt goes out
as the user message with no system text, no tools, no labels and no
logprobs, and the reply is read with ``Provider.reply_text``. Synthetic
bytes only; transports are injected; no network.
"""

from __future__ import annotations

import json

import pytest

from evals.tool_jev.providers import anthropic, fake, openai, openai_compat
from evals.tool_jev.providers.base import INTERFACES, CallRequest, ProviderCapabilities
from evals.tool_jev.providers.errors import Outcome

JUDGE_PROMPT = "Evaluate the answer. Return JSON with score and reason."
REPLY = '{"score": 7, "reason": "clear"}'


def _judge_request(system: str = "") -> CallRequest:
    return CallRequest(
        case_id="j-0123456789abcdef",
        split="test",
        case_text=JUDGE_PROMPT,
        prompt=system,
        interface="text",
        params={"reasoning": "medium", "max_output_tokens": 900},
    )


def test_text_is_an_interface():
    assert "text" in INTERFACES
    assert _judge_request().interface == "text"


# -- OpenAI -----------------------------------------------------------------


def _openai_transport(sent):
    def transport(method, url, data, headers):
        sent.append(json.loads(data))
        body = {
            "id": "resp_1",
            "model": "gpt-6-sol",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": REPLY}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return openai.TransportResponse(status=200, body=json.dumps(body).encode())

    return transport


def test_openai_text_request_has_no_tools_and_no_empty_instructions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k" * 8)
    sent: list = []
    provider = openai.OpenAIProvider("gpt-6-sol", transport=_openai_transport(sent))
    result = provider.submit_sync(_judge_request())
    (body,) = sent
    assert "tools" not in body and "tool_choice" not in body
    assert "instructions" not in body
    assert body["input"] == [{"role": "user", "content": JUDGE_PROMPT}]
    assert body["max_output_tokens"] == 900
    assert result.outcome is Outcome.OK and result.answer == REPLY
    assert result.interface == "text"
    assert provider.reply_text(result.raw).text == REPLY


def test_openai_keeps_a_non_empty_system_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k" * 8)
    sent: list = []
    provider = openai.OpenAIProvider("gpt-6-sol", transport=_openai_transport(sent))
    provider.submit_sync(_judge_request(system="Be brief."))
    assert sent[0]["instructions"] == "Be brief."


def test_openai_batch_custom_id_decodes_the_text_interface():
    assert openai._decode_custom_id("case-1::text") == ("case-1", "text")


# -- Anthropic --------------------------------------------------------------


def test_anthropic_text_request_omits_empty_system_and_tools(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k" * 8)
    sent: list = []

    def transport(method, url, headers, body):
        sent.append(json.loads(body))
        message = {
            "id": "msg_1",
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": REPLY}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        return 200, json.dumps(message).encode()

    provider = anthropic.AnthropicProvider("claude-opus-5-5", transport=transport)
    result = provider.submit_sync(_judge_request())
    (payload,) = sent
    assert "system" not in payload and "tools" not in payload
    assert payload["messages"] == [{"role": "user", "content": JUDGE_PROMPT}]
    assert result.outcome is Outcome.OK and result.answer == REPLY
    assert provider.reply_text(result.raw).text == REPLY


def test_anthropic_custom_id_round_trips_the_text_interface():
    custom_id = anthropic.build_custom_id("tj-abc", 3, "j-0123456789abcdef", "text")
    assert anthropic.parse_custom_id(custom_id)[2:] == ("text", "j-0123456789abcdef")


# -- OpenAI-compatible ------------------------------------------------------


def test_openai_compat_text_request_has_no_logprobs_tools_or_empty_system(monkeypatch):
    monkeypatch.setenv("NGC_API_KEY", "k" * 8)
    sent: list = []

    def transport(url, headers, body):
        sent.append(json.loads(body))
        response = {
            "id": "c1",
            "model": "vendor/judge",
            "choices": [{"message": {"content": REPLY}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return 200, json.dumps(response).encode()

    provider = openai_compat.OpenAICompatProvider(
        "nvidia",
        "vendor/judge",
        capabilities=ProviderCapabilities(logprobs=True, batch=False, reasoning=False),
        transport=transport,
    )
    result = provider.submit_sync(_judge_request())
    (payload,) = sent
    assert "logprobs" not in payload and "tools" not in payload
    assert payload["messages"] == [{"role": "user", "content": JUDGE_PROMPT}]
    assert result.outcome is Outcome.OK and result.answer == REPLY
    assert result.candidates is None
    assert provider.reply_text(result.raw).text == REPLY


# -- Fake -------------------------------------------------------------------


def test_fake_answers_a_text_request():
    provider = fake.FakeProvider(script=[fake.ScriptedOutcome("answer", answer=REPLY, text=REPLY)])
    result = provider.submit_sync(_judge_request())
    assert result.outcome is Outcome.OK and result.interface == "text"
    assert provider.reply_text(result.raw).text == REPLY


@pytest.mark.parametrize("interface", ["tool_call", "choice"])
def test_existing_interfaces_still_send_their_system_text(monkeypatch, interface):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k" * 8)
    provider = anthropic.AnthropicProvider("claude-opus-5-5")
    request = CallRequest(
        case_id="c", split="test", case_text="u", prompt="sys", interface=interface
    )
    assert provider._build_payload(request)["system"] == "sys"


# -- review fixes (codex t17) ----------------------------------------------


@pytest.mark.parametrize(
    "status, complete, expired",
    [
        ("cancelling", False, False),
        ("cancelled", True, True),
        ("failed", True, True),
        ("expired", True, True),
        ("completed", True, False),
    ],
)
def test_openai_cancelling_batch_is_not_terminal(monkeypatch, status, complete, expired):
    """P1-1: a cancelling batch may still be running; keep polling it."""
    monkeypatch.setenv("OPENAI_API_KEY", "k" * 8)

    def transport(method, url, data, headers):
        return openai.TransportResponse(200, json.dumps({"id": "b1", "status": status}).encode())

    provider = openai.OpenAIProvider("gpt-6-sol", transport=transport)
    got = provider.poll_batch(openai.BatchHandle(batch_id="b1", provider=provider.name))
    assert (got.complete, got.expired) == (complete, expired)


def test_rate_limiter_is_thread_safe():
    """P2-10: concurrent acquires each get their own slot, none share one."""
    import threading
    import time

    waits: list[float] = []
    lock = threading.Lock()

    def sleep(seconds):
        with lock:
            waits.append(round(seconds, 6))
        time.sleep(0.02)  # a slow sleeper widens any read-then-write race

    limiter = openai_compat.RateLimiter(60.0, clock=lambda: 0.0, sleep=sleep)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        limiter.acquire()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # One call goes at once; the other seven wait 1, 2, ... 7 seconds.
    assert sorted(waits) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
