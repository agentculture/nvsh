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
    system_text, messages, _tools, _labels = req.canonical_content(request)
    assert messages == [{"role": "user", "content": messages[0]["content"]}]
    user_text = messages[0]["content"]
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


# ---------------------------------------------------------------------------
# Deviation d1: a 2-round loop through every adapter (track_a_loop).
# ---------------------------------------------------------------------------

SNAPSHOT = {
    "services": ["synthetic.service"],
    "containers": [],
    "source": "synthetic test snapshot",
    "created": "2026-09-26",
}


def _loop_openai(sent):
    bodies = [FIXTURES / "openai" / f"responses_{k}.json" for k in ("status", "propose")]

    def transport(method, url, data, headers):
        sent.append(json.loads(data))
        return openai.TransportResponse(status=200, body=bodies[len(sent) - 1].read_bytes())

    return openai.OpenAIProvider("gpt-6-luna", transport=transport)


def _loop_anthropic(sent):
    bodies = [FIXTURES / "anthropic" / f"message_{k}.json" for k in ("status", "propose")]

    def transport(method, url, headers, data):
        sent.append(json.loads(data))
        return 200, bodies[len(sent) - 1].read_bytes()

    return anthropic.AnthropicProvider("claude-sonnet-5", transport=transport)


def _loop_openai_compat(sent):
    bodies = [FIXTURES / "openai_compat" / f"tool_call_{k}.json" for k in ("status", "propose")]

    def transport(url, headers, data):
        sent.append(json.loads(data))
        return 200, bodies[len(sent) - 1].read_bytes()

    return openai_compat.OpenAICompatProvider(
        "openrouter", "qwen/qwen3.8-max-0902", transport=transport
    )


def _history_content(adapter, payload):
    """(tool name, decoded arguments, tool result text) as the adapter put it on the wire."""
    if adapter == "openai":
        call = next(i for i in payload["input"] if i.get("type") == "function_call")
        result = next(i for i in payload["input"] if i.get("type") == "function_call_output")
        assert result["call_id"] == call["call_id"]
        return call["name"], json.loads(call["arguments"]), result["output"]
    if adapter == "anthropic":
        assistant, results = payload["messages"][1:]
        use = next(b for b in assistant["content"] if b["type"] == "tool_use")
        (result,) = results["content"]
        assert result["tool_use_id"] == use["id"]
        return use["name"], use["input"], result["content"]
    assistant, tool = payload["messages"][2:]
    (call,) = assistant["tool_calls"]
    assert tool["tool_call_id"] == call["id"]
    return call["function"]["name"], json.loads(call["function"]["arguments"]), tool["content"]


LOOP_DRIVERS = {
    "openai": _loop_openai,
    "anthropic": _loop_anthropic,
    "openai_compat": _loop_openai_compat,
}


def _run_loop(adapter, tmp_path):
    from evals.tool_jev import track_a_loop
    from evals.tool_jev.ledger import Ledger

    sent: list[dict] = []
    provider = LOOP_DRIVERS[adapter](sent)
    with Ledger(tmp_path / adapter) as ledger:
        while True:
            current = track_a_loop.run_round(
                [_case()],
                provider=provider,
                model="reference-model",
                ledger=ledger,
                snapshot=SNAPSHOT,
                platform=PLATFORM,
            )
            if not current.pending:
                return current.finished["contract-c1"], sent
            results = [provider.submit_sync(call.request) for call in current.pending]
            assert track_a_loop.record_results(ledger, current.pending, results) == []


def test_two_round_loop_decides_the_same_and_sends_the_same_history_everywhere(tmp_path):
    histories = {}
    for adapter in sorted(LOOP_DRIVERS):
        record, sent = _run_loop(adapter, tmp_path)
        assert (record.outcome, record.operation, record.arguments) == (
            "propose",
            "service_restart",
            {"service": "synthetic.service"},
        ), adapter
        assert len(sent) == 2, adapter
        histories[adapter] = _history_content(adapter, sent[1])
    assert len(set(json.dumps(h, sort_keys=True) for h in histories.values())) == 1, histories
    name, arguments, result = histories["openai"]
    assert (name, arguments, result) == (
        "service_status",
        {"service": "synthetic.service"},
        "exit 127\n",
    )


def test_anthropic_round_two_replays_round_one_thinking_block_unchanged(tmp_path):
    _record, sent = _run_loop("anthropic", tmp_path)
    assistant = sent[1]["messages"][1]
    thinking = json.loads((FIXTURES / "anthropic" / "message_status.json").read_text())["content"][
        0
    ]
    assert assistant["content"][0] == thinking
    assert assistant["content"][1]["id"] == "toolu_01Status"
