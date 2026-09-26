"""Integration: the three reference adapters against the one request contract (issue #64).

Each adapter (OpenAI Responses, Anthropic Messages, OpenAI-compatible chat
completions) is driven with a *real* Track A ``CallRequest`` built by
``request.build_tool_call_request`` -- nvsh's own ``lfm`` system brief,
request message and tool list -- through a fake transport that returns a
hand-written fixture naming one of nvsh's real Track A tools (``propose``,
``explain``, ``escalate``). The adapter's ``CallResult.answer`` is then
fed to ``request.parse_tool_call``, the single parser, which proves every
adapter returns the same raw tool-call shape
``{"name": <tool>, "arguments": {...}}``.

It also proves t11/h23's criterion 1 end to end: the (system, user) text
each adapter actually put on the wire is byte-identical across the three
adapters for the same request.

No network: every transport is an in-process fake. Case text is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tool_jev import request as req
from evals.tool_jev.cases import Case
from evals.tool_jev.providers import anthropic, errors, openai, openai_compat
from nvsh.platform._model import Platform

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PLATFORM = Platform(kind="jetson")
OFFERED_OPS = ("service_status", "service_logs", "service_restart")


def _case() -> Case:
    return Case(
        id="contract-c1",
        split="test",
        text="please restart the synthetic service",
        candidates=OFFERED_OPS,
        expect={"operation": "service_restart", "args": {"service": "synthetic.service"}},
        read_only=False,
    )


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    # Built at runtime; never a literal key in the repo.
    for env_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPEN_ROUTER_API_KEY"):
        monkeypatch.setenv(env_name, "k" * 8)


# ---------------------------------------------------------------------------
# One driver per adapter: (fixture key) -> (CallResult, (system, user) sent).
# ---------------------------------------------------------------------------


def _drive_openai(request, key):
    body = (FIXTURES / "openai" / f"responses_{key}.json").read_bytes()
    sent: list[dict] = []

    def transport(method, url, data, headers):
        sent.append(json.loads(data))
        return openai.TransportResponse(status=200, body=body)

    result = openai.OpenAIProvider("gpt-6-luna", transport=transport).submit_sync(request)
    (payload,) = sent
    return result, (payload["instructions"], payload["input"][0]["content"]), payload


def _drive_anthropic(request, key, model="claude-sonnet-5"):
    body = (FIXTURES / "anthropic" / f"message_{key}.json").read_bytes()
    sent: list[dict] = []

    def transport(method, url, headers, data):
        sent.append(json.loads(data))
        return 200, body

    result = anthropic.AnthropicProvider(model, transport=transport).submit_sync(request)
    (payload,) = sent
    return result, (payload["system"], payload["messages"][0]["content"]), payload


def _drive_openai_compat(request, key):
    body = (FIXTURES / "openai_compat" / f"tool_call_{key}.json").read_bytes()
    sent: list[dict] = []

    def transport(url, headers, data):
        sent.append(json.loads(data))
        return 200, body

    provider = openai_compat.OpenAICompatProvider(
        "openrouter", "qwen/qwen3.8-max-0902", transport=transport
    )
    result = provider.submit_sync(request)
    (payload,) = sent
    messages = payload["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    return result, (messages[0]["content"], messages[1]["content"]), payload


DRIVERS = {
    "openai": _drive_openai,
    "anthropic": _drive_anthropic,
    "openai_compat": _drive_openai_compat,
}


def _parsed(adapter, key):
    request = req.build_tool_call_request(_case(), None, PLATFORM)
    result, _sent, _payload = DRIVERS[adapter](request, key)
    classification, operation, arguments = req.parse_tool_call(
        result.answer, request.offered_candidates
    )
    return result, classification, operation, arguments


# ---------------------------------------------------------------------------
# The unified answer contract, per adapter.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", sorted(DRIVERS))
def test_propose_parses_to_offered_operation_with_arguments(adapter):
    result, classification, operation, arguments = _parsed(adapter, "propose")
    assert json.loads(result.answer) == {
        "name": "propose",
        "arguments": {
            "operation": "service_restart",
            "arguments": {"service": "synthetic.service"},
        },
    }
    assert classification.outcome is errors.Outcome.OK
    assert operation == "service_restart"
    assert arguments == {"service": "synthetic.service"}
    assert result.outcome is errors.Outcome.OK
    assert result.interface == "tool_call"


@pytest.mark.parametrize("adapter", sorted(DRIVERS))
def test_explain_parses_ok(adapter):
    result, classification, operation, arguments = _parsed(adapter, "explain")
    assert classification.outcome is errors.Outcome.OK
    assert operation == "explain"
    assert arguments == {"text": "synthetic explanation, no command needed"}
    assert result.outcome is errors.Outcome.OK


@pytest.mark.parametrize("adapter", sorted(DRIVERS))
def test_escalate_parses_ok(adapter):
    result, classification, operation, _arguments = _parsed(adapter, "escalate")
    assert classification.outcome is errors.Outcome.OK
    assert operation == "escalate"
    assert result.outcome is errors.Outcome.OK


@pytest.mark.parametrize("adapter", sorted(DRIVERS))
def test_propose_of_unoffered_operation_is_invalid_outside_offered_set(adapter):
    result, classification, operation, _arguments = _parsed(adapter, "propose_unoffered")
    # The adapter passes the raw call through; only the parser judges it.
    assert json.loads(result.answer)["arguments"]["operation"] == "power_set"
    assert classification.outcome is errors.Outcome.INVALID
    assert classification.reason == "outside_offered_set"
    assert operation is None
    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "outside_offered_set"


@pytest.mark.parametrize("adapter", sorted(DRIVERS))
def test_text_only_reply_is_invalid_malformed(adapter):
    result, classification, operation, _arguments = _parsed(adapter, "text_only")
    assert result.answer is None
    assert result.outcome is errors.Outcome.INVALID
    assert result.reason == "malformed"
    assert classification.outcome is errors.Outcome.INVALID
    assert classification.reason == "malformed"
    assert operation is None


# ---------------------------------------------------------------------------
# Criterion 1 (t11/h23): byte-identical system/user content on the wire.
# ---------------------------------------------------------------------------


def test_system_and_user_sent_byte_identical_across_adapters():
    request = req.build_tool_call_request(_case(), None, PLATFORM)
    sent = {name: driver(request, "propose")[1] for name, driver in DRIVERS.items()}
    system_texts = {name: pair[0].encode("utf-8") for name, pair in sent.items()}
    user_texts = {name: pair[1].encode("utf-8") for name, pair in sent.items()}
    assert len(set(system_texts.values())) == 1, system_texts
    assert len(set(user_texts.values())) == 1, user_texts
    # And it is the contract's own content (redaction is a no-op on this text).
    system_text, user_text, _tools, _labels = req.canonical_content(request)
    assert system_texts["openai"] == system_text.encode("utf-8")
    assert user_texts["openai"] == user_text.encode("utf-8")


def test_every_adapter_offers_the_same_tool_names():
    request = req.build_tool_call_request(_case(), None, PLATFORM)
    expected = [tool["function"]["name"] for tool in request.params["tools"]]
    _r, _s, openai_payload = _drive_openai(request, "propose")
    _r, _s, anthropic_payload = _drive_anthropic(request, "propose")
    _r, _s, compat_payload = _drive_openai_compat(request, "propose")
    assert [tool["name"] for tool in openai_payload["tools"]] == expected
    assert [tool["name"] for tool in anthropic_payload["tools"]] == expected
    assert [tool["function"]["name"] for tool in compat_payload["tools"]] == expected


# ---------------------------------------------------------------------------
# Forced tool choice, per model.
# ---------------------------------------------------------------------------


def test_forced_tool_choice_defaults_per_adapter():
    request = req.build_tool_call_request(_case(), None, PLATFORM)
    assert _drive_openai(request, "propose")[2]["tool_choice"] == "required"
    assert _drive_openai_compat(request, "propose")[2]["tool_choice"] == "required"
    assert _drive_anthropic(request, "propose")[2]["tool_choice"] == {"type": "any"}
    opus = _drive_anthropic(request, "propose", model="claude-opus-5-5")[2]
    assert opus["tool_choice"] == {"type": "auto"}
