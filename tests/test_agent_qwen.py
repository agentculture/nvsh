"""Tests for ``nvsh.agent.qwen.QwenAgent`` (task t13).

Covers the argv this adapter builds, its stream-json line parsing
(status/thinking/tool_use/tool_result/text/result envelopes -- shapes
verified against a real ``qwen --output-format stream-json --approval-mode
plan -p ...`` run, qwen 0.23.3), its capabilities, and an end-to-end run
against ``tests/fakes/qwen``. The shared conformance suite
(``tests/test_agent_conformance.py``) covers the STATUS/TEXT_DELTA/ERROR/
DONE path already exercised through every adapter; this file is where the
qwen-specific behavior (thinking/tool events, the dropped ``[status] ``
heuristic, the constructor's model/effort/extra_args/approval knobs) lives.
"""

from __future__ import annotations

import threading

import pytest

from nvsh.agent.base import AgentContext, AgentEvent, AgentRequest, EventKind, RequestKind
from nvsh.agent.qwen import QwenAgent
from tests._fake_adapters import QwenAgentViaFake

# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------


def test_argv_uses_stream_json_and_plan_approval_mode():
    agent = QwenAgent({})
    argv = agent._argv(
        AgentRequest(kind=RequestKind.FAILURE, prompt="why did this fail?"),
        AgentContext(),
    )
    assert argv[0] == "qwen"
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--approval-mode" in argv
    assert argv[argv.index("--approval-mode") + 1] == "plan"
    assert "-p" in argv
    assert argv[argv.index("-p") + 1] == "why did this fail?"


def test_argv_carries_the_model_when_configured():
    agent = QwenAgent({"model": "qwen3-coder-480b"})
    argv = agent._argv(AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"), AgentContext())
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "qwen3-coder-480b"


def test_argv_omits_model_flag_when_not_configured():
    agent = QwenAgent({})
    argv = agent._argv(AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"), AgentContext())
    assert "--model" not in argv


def test_argv_carries_extra_args():
    agent = QwenAgent({"extra_args": ["--yolo-not-really", "1"]})
    argv = agent._argv(AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"), AgentContext())
    assert "--yolo-not-really" in argv
    assert argv[argv.index("--yolo-not-really") + 1] == "1"


def test_argv_still_carries_append_system_prompt():
    agent = QwenAgent({})
    argv = agent._argv(AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"), AgentContext())
    assert "--append-system-prompt" in argv


def test_direct_kwargs_win_over_config_dict():
    agent = QwenAgent({"model": "from-config"}, model="from-kwarg")
    argv = agent._argv(AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"), AgentContext())
    assert argv[argv.index("--model") + 1] == "from-kwarg"


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_report_thinking_but_read_only_and_no_effort():
    # Print mode (plan approval mode) can't pause a tool call for nvsh approve,
    # so the fallback runs read-only: tool_calling is False (spec rule).
    caps = QwenAgent({}).capabilities()
    assert caps.thinking is True
    assert caps.tool_calling is False
    assert caps.effort is False
    assert caps.path == "stream-json"


def test_capabilities_report_the_configured_approval_mediator():
    assert QwenAgent({}).capabilities().approval == "nvsh"
    assert QwenAgent({"approval": "harness"}).capabilities().approval == "harness"
    assert QwenAgent({}, approval="harness").capabilities().approval == "harness"


def test_effort_kwarg_is_accepted_but_never_reaches_argv():
    agent = QwenAgent({}, effort="high")
    argv = agent._argv(AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"), AgentContext())
    assert "high" not in argv
    assert "--effort" not in argv


# ---------------------------------------------------------------------------
# _parse_line: stream-json envelope shapes (qwen 0.23.3, verified locally)
# ---------------------------------------------------------------------------


def test_parse_system_line_is_status():
    agent = QwenAgent({})
    event = agent._parse_line('{"type": "system", "subtype": "init"}')
    assert event == AgentEvent(kind=EventKind.STATUS, text="init")


def test_parse_assistant_thinking_content_is_thinking_event():
    agent = QwenAgent({})
    line = (
        '{"type": "assistant", "message": {"content": '
        '[{"type": "thinking", "thinking": "let me check the logs"}]}}'
    )
    event = agent._parse_line(line)
    assert event.kind is EventKind.THINKING
    assert event.text == "let me check the logs"


def test_parse_assistant_text_content_is_text_delta():
    agent = QwenAgent({})
    line = '{"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}}'
    event = agent._parse_line(line)
    assert event == AgentEvent(kind=EventKind.TEXT_DELTA, text="hello")


def test_parse_assistant_tool_use_content_is_tool_call():
    agent = QwenAgent({})
    line = (
        '{"type": "assistant", "message": {"content": '
        '[{"type": "tool_use", "id": "call-1", "name": "run_shell_command", '
        '"input": {"command": "ls"}}]}}'
    )
    event = agent._parse_line(line)
    assert event.kind is EventKind.TOOL_CALL
    assert event.tool == "run_shell_command"
    assert event.args == {"command": "ls"}


def test_parse_user_tool_result_content_is_tool_result_and_resolves_tool_name():
    agent = QwenAgent({})
    call_line = (
        '{"type": "assistant", "message": {"content": '
        '[{"type": "tool_use", "id": "call-1", "name": "run_shell_command", "input": {}}]}}'
    )
    agent._parse_line(call_line)  # records call-1 -> run_shell_command
    result_line = (
        '{"type": "user", "message": {"content": '
        '[{"type": "tool_result", "tool_use_id": "call-1", "is_error": false, '
        '"content": "total 0"}]}}'
    )
    event = agent._parse_line(result_line)
    assert event.kind is EventKind.TOOL_RESULT
    assert event.tool == "run_shell_command"
    assert event.result == "total 0"


def test_parse_result_success_is_done():
    agent = QwenAgent({})
    event = agent._parse_line('{"type": "result", "subtype": "success"}')
    assert event == AgentEvent(kind=EventKind.DONE)


def test_parse_result_error_is_error_with_message():
    agent = QwenAgent({})
    event = agent._parse_line('{"type": "result", "subtype": "error", "result": "boom"}')
    assert event.kind is EventKind.ERROR
    assert event.error == "boom"


def test_parse_unknown_envelope_type_is_skipped():
    agent = QwenAgent({})
    assert agent._parse_line('{"type": "stream_event", "event": {"type": "goal_state"}}') is None


def test_parse_blank_and_non_json_lines_are_skipped():
    agent = QwenAgent({})
    assert agent._parse_line("") is None
    assert agent._parse_line("   ") is None
    assert agent._parse_line("not json") is None


def test_status_heuristic_prefix_no_longer_special_cased():
    """Acceptance criterion 1: the old ``[status] `` text heuristic is gone."""
    agent = QwenAgent({})
    # Plain text (not a JSON envelope) is simply unparseable now -- it is
    # not sniffed for a "[status] " prefix and turned into a STATUS event.
    assert agent._parse_line("[status] doing a thing") is None


def test_tool_names_reset_between_runs():
    agent = QwenAgent({})
    call_line = (
        '{"type": "assistant", "message": {"content": '
        '[{"type": "tool_use", "id": "call-1", "name": "run_shell_command", "input": {}}]}}'
    )
    agent._parse_line(call_line)
    assert agent._tool_names == {"call-1": "run_shell_command"}
    agent.start()
    assert agent._tool_names == {}


# ---------------------------------------------------------------------------
# end-to-end via tests/fakes/qwen
# ---------------------------------------------------------------------------


def _collect(agent, request, timeout: float = 15.0) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    error: list[BaseException] = []

    def work() -> None:
        try:
            events.extend(agent.run(request, AgentContext()))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            error.append(exc)

    thread = threading.Thread(target=work, daemon=True)
    agent.start()
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        agent.cancel()
        thread.join(5)
        pytest.fail(f"adapter deadlocked: no result within {timeout}s")
    if error:
        raise error[0]
    return events


def test_fake_qwen_status_text_and_done():
    script = [
        AgentEvent(kind=EventKind.STATUS, text="init"),
        AgentEvent(kind=EventKind.TEXT_DELTA, text="hello from qwen"),
        AgentEvent(kind=EventKind.DONE),
    ]
    agent = QwenAgentViaFake(script)
    events = _collect(agent, AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"))
    kinds = [event.kind for event in events]
    assert kinds == [EventKind.STATUS, EventKind.TEXT_DELTA, EventKind.DONE]
    assert events[1].text == "hello from qwen"


def test_fake_qwen_error_path():
    script = [AgentEvent(kind=EventKind.ERROR, error="qwen blew up")]
    agent = QwenAgentViaFake(script)
    events = _collect(agent, AgentRequest(kind=RequestKind.EXPLICIT, prompt="hi"))
    assert events[-1].kind is EventKind.ERROR
    assert events[-1].error == "qwen blew up"
