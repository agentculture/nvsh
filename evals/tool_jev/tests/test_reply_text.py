"""Reply text and per-request tool_choice across adapters (issue #64, plan risk r9).

``reply_text`` is what the Track A loop hands LfmTier when a reply carries no
tool call, and what a runner reads for a judge's answer: visible words only
(never thinking blocks, reasoning items, ``reasoning_content`` or inline
``<think>``), plus the provider's own cut-at-budget signal. Synthetic bytes
only; no network.
"""

from __future__ import annotations

import json

import pytest

from evals.tool_jev.providers import anthropic, openai, openai_compat
from evals.tool_jev.providers.base import CallRequest, tool_choice_forced, visible_text


def _request(**params):
    return CallRequest(case_id="c1", split="test", case_text="x", prompt="p", params=params)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("  hello  ", "hello"),
        ("<think>hmm</think>The answer.", "The answer."),
        ("<THINK>a\nb</THINK> ok", "ok"),
        ("<think>never closed", ""),
        (None, ""),
    ],
)
def test_visible_text_drops_inline_reasoning(text, expected):
    assert visible_text(text) == expected


def test_tool_choice_param_overrides_the_adapter_default():
    assert tool_choice_forced(_request(), True) is True
    assert tool_choice_forced(_request(), False) is False
    assert tool_choice_forced(_request(tool_choice="auto"), True) is False
    assert tool_choice_forced(_request(tool_choice="required"), False) is True
    with pytest.raises(ValueError):
        tool_choice_forced(_request(tool_choice="none"), True)


def test_openai_reply_text_reads_output_text_not_reasoning():
    body = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "secret plan"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "It is fine."}]},
        ],
    }
    provider = openai.OpenAIProvider("gpt-6-luna")
    spoken = provider.reply_text(json.dumps(body).encode())
    assert (spoken.text, spoken.truncated) == ("It is fine.", False)
    batch_line = {"custom_id": "x", "response": {"status_code": 200, "body": {**body}}}
    batch_line["response"]["body"]["status"] = "incomplete"
    spoken = provider.reply_text(json.dumps(batch_line).encode())
    assert (spoken.text, spoken.truncated) == ("It is fine.", True)


def test_anthropic_reply_text_reads_text_blocks_not_thinking():
    message = {
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "secret plan", "signature": "s"},
            {"type": "text", "text": "It is fine."},
        ],
    }
    provider = anthropic.AnthropicProvider("claude-sonnet-5")
    spoken = provider.reply_text(json.dumps(message).encode())
    assert (spoken.text, spoken.truncated) == ("It is fine.", False)
    message["stop_reason"] = "max_tokens"
    assert provider.reply_text(json.dumps(message).encode()).truncated is True


def test_openai_compat_reply_text_reads_content_not_reasoning_content():
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "It is fine.", "reasoning_content": "secret plan"},
            }
        ]
    }
    provider = openai_compat.OpenAICompatProvider(kind="local", model="m")
    spoken = provider.reply_text(json.dumps(response).encode())
    assert (spoken.text, spoken.truncated) == ("It is fine.", False)
    response["choices"][0]["finish_reason"] = "length"
    assert provider.reply_text(json.dumps(response).encode()).truncated is True
    assert provider.reply_text(b"not json").text == ""
