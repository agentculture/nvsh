"""Tests for :class:`nvsh.agent.agy.AgyAgent`.

Exercises the adapter against ``tests/fakes/agy`` -- a scripted stand-in for
the real ``agy`` CLI that replays recorded NDJSON transcripts -- so these
tests prove the adapter's actual argv-building and line-parsing code, the
same "real subprocess, fake binary" pattern
``tests/test_agent_conformance.py`` uses for ``claude``/``codex``/``qwen``
via ``tests/_fake_adapters.py``.

The three transcripts below (``TEXT_TURN_STDOUT``, ``TOOL_TURN_STDOUT``,
``DENY_TURN_STDOUT``, plus the two-turn ``WARM_TURN1_STDOUT`` /
``WARM_TURN2_STDOUT`` pair) were recorded live against the ``agy`` binary
installed on this box (2026-09-14) with::

    agy -p '<prompt>' --output-format stream-json
    agy -p= --output-format stream-json --input-format stream-json   # warm

# recorded-from: agy 1.2.2

Redacted before being committed: the probe's scratch-directory ``cwd`` was
replaced with ``/home/user/project``, one denied-tool path under
``/home/spark/...`` was replaced with the equivalent ``/home/user/...``
path, and every ``conversation_id`` was replaced with a fixed
``fake-conv-*`` id. Byte-for-byte otherwise -- key order, numeric fields
(``duration_seconds``, token usage) and all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nvsh.agent.agy import AgyAgent
from nvsh.agent.base import AgentContext, AgentRequest, Capabilities, EventKind, RequestKind

FAKES_DIR = Path(__file__).parent / "fakes"

# -- recorded transcripts (redacted; see module docstring) -----------------

TEXT_TURN_STDOUT = [
    '{"event":"init","conversation_id":"fake-conv-text-0001","init":{"cwd":"/home/user/project","tools":["ask_custom_permission","ask_permission","ask_question","browser_click_element","browser_drag_pixel_to_pixel","browser_get_dom","browser_get_network_request","browser_input","browser_list_network_requests","browser_mouse_down","browser_mouse_up","browser_move_mouse","browser_press_key","browser_refresh_page","browser_resize_window","browser_scroll","browser_scroll_dom","browser_select_option","browser_subagent","call_mcp_tool","capture_browser_console_logs","capture_browser_screenshot","click_browser_pixel","command_status","define_subagent","delete_knowledge","execute_browser_javascript","find_by_name","finish","generate_image","grep_search","invoke_subagent","list_browser_pages","list_dir","list_permissions","list_resources","manage_inbox","manage_subagents","manage_task","multi_replace_file_content","notebook_edit","notebook_execution","open_browser_url","read_browser_page","read_resource","read_url_content","replace_file_content","run_command","schedule","search_web","sed_file","send_command_input","send_message","view_file","wait","wait_5_seconds","write_to_file"],"permission_mode":"request-review"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-text-0001","step_index":0,"state":"DONE","step_type":"user_input"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-text-0001","step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"OK"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-text-0001","step_index":1,"state":"DONE","step_type":"agent_response","text_delta":"\\n","duration_seconds":0.129325257,"usage":{"input_tokens":13242,"output_tokens":33,"thinking_tokens":32,"cache_read_tokens":0,"total_tokens":13275}}}',  # noqa: E501
    '{"event":"result","result":{"conversation_id":"fake-conv-text-0001","status":"SUCCESS","response":"OK\\n","duration_seconds":1.634348143,"num_turns":1,"usage":{"input_tokens":13242,"output_tokens":33,"thinking_tokens":32,"cache_read_tokens":0,"total_tokens":13275}}}',  # noqa: E501
]

TOOL_TURN_STDOUT = [
    '{"event":"init","conversation_id":"fake-conv-tool-0001","init":{"cwd":"/home/user/project","tools":["ask_custom_permission","ask_permission","ask_question","browser_click_element","browser_drag_pixel_to_pixel","browser_get_dom","browser_get_network_request","browser_input","browser_list_network_requests","browser_mouse_down","browser_mouse_up","browser_move_mouse","browser_press_key","browser_refresh_page","browser_resize_window","browser_scroll","browser_scroll_dom","browser_select_option","browser_subagent","call_mcp_tool","capture_browser_console_logs","capture_browser_screenshot","click_browser_pixel","command_status","define_subagent","delete_knowledge","execute_browser_javascript","find_by_name","finish","generate_image","grep_search","invoke_subagent","list_browser_pages","list_dir","list_permissions","list_resources","manage_inbox","manage_subagents","manage_task","multi_replace_file_content","notebook_edit","notebook_execution","open_browser_url","read_browser_page","read_resource","read_url_content","replace_file_content","run_command","schedule","search_web","sed_file","send_command_input","send_message","view_file","wait","wait_5_seconds","write_to_file"],"permission_mode":"request-review"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":0,"state":"DONE","step_type":"user_input"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":1,"state":"DONE","step_type":"agent_response","duration_seconds":0.065442921,"usage":{"input_tokens":13264,"output_tokens":754,"thinking_tokens":689,"cache_read_tokens":0,"total_tokens":14018}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"run_command","tool_info":{"name":"run_command","parameters":{"CommandLine":"ls"}}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":2,"state":"DONE","step_type":"tool","tool_name":"run_command","duration_seconds":0.01622623,"tool_info":{"name":"run_command","parameters":{"CommandLine":"ls"}}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":3,"state":"DONE","step_type":"agent_response","duration_seconds":0.014357498,"usage":{"input_tokens":14106,"output_tokens":126,"thinking_tokens":58,"cache_read_tokens":0,"total_tokens":14232}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":4,"state":"ACTIVE","step_type":"tool","tool_name":"run_command","tool_info":{"name":"run_command","parameters":{"CommandLine":"ls -a"}}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":4,"state":"DONE","step_type":"tool","tool_name":"run_command","duration_seconds":0.018799222,"tool_info":{"name":"run_command","parameters":{"CommandLine":"ls -a"},"output":".  ..\\r\\n"}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":5,"state":"ACTIVE","step_type":"agent_response","text_delta":"The current directory is empty, so `ls` returned no files."}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-tool-0001","step_index":5,"state":"DONE","step_type":"agent_response","text_delta":"\\n","duration_seconds":0.023486067,"usage":{"input_tokens":14321,"output_tokens":137,"thinking_tokens":123,"cache_read_tokens":0,"total_tokens":14458}}}',  # noqa: E501
    '{"event":"result","result":{"conversation_id":"fake-conv-tool-0001","status":"SUCCESS","response":"The current directory is empty, so `ls` returned no files.\\n","duration_seconds":6.562379937,"num_turns":1,"usage":{"input_tokens":41691,"output_tokens":1017,"thinking_tokens":870,"cache_read_tokens":0,"total_tokens":42708}}}',  # noqa: E501
]

DENY_TURN_STDOUT = [
    '{"event":"init","conversation_id":"fake-conv-deny-0001","init":{"cwd":"/home/user/project","tools":["ask_custom_permission","ask_permission","ask_question","browser_click_element","browser_drag_pixel_to_pixel","browser_get_dom","browser_get_network_request","browser_input","browser_list_network_requests","browser_mouse_down","browser_mouse_up","browser_move_mouse","browser_press_key","browser_refresh_page","browser_resize_window","browser_scroll","browser_scroll_dom","browser_select_option","browser_subagent","call_mcp_tool","capture_browser_console_logs","capture_browser_screenshot","click_browser_pixel","command_status","define_subagent","delete_knowledge","execute_browser_javascript","find_by_name","finish","generate_image","grep_search","invoke_subagent","list_browser_pages","list_dir","list_permissions","list_resources","manage_inbox","manage_subagents","manage_task","multi_replace_file_content","notebook_edit","notebook_execution","open_browser_url","read_browser_page","read_resource","read_url_content","replace_file_content","run_command","schedule","search_web","sed_file","send_command_input","send_message","view_file","wait","wait_5_seconds","write_to_file"],"permission_mode":"request-review"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-deny-0001","step_index":0,"state":"DONE","step_type":"user_input"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-deny-0001","step_index":1,"state":"DONE","step_type":"agent_response","duration_seconds":0.027110521,"usage":{"input_tokens":5126,"output_tokens":704,"thinking_tokens":655,"cache_read_tokens":8128,"total_tokens":5830}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-deny-0001","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"list_dir","tool_info":{"name":"list_dir","parameters":{"DirectoryPath":"/home/user/.gemini/antigravity-cli"}}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-deny-0001","step_index":2,"state":"ERROR","step_type":"tool","tool_name":"list_dir","duration_seconds":0.01164148,"tool_info":{"name":"list_dir","parameters":{"DirectoryPath":"/home/user/.gemini/antigravity-cli"},"error":{"type":"TOOL_ERROR","message":"permission check failed for read_file \\"/home/user/.gemini/antigravity-cli\\": Permission denied for read_file(/home/user/.gemini/antigravity-cli). Matches hardcoded system protection boundary rule."}}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-deny-0001","step_index":3,"state":"DONE","step_type":"agent_response","duration_seconds":0.023294113,"usage":{"input_tokens":14088,"output_tokens":683,"thinking_tokens":576,"cache_read_tokens":0,"total_tokens":14771}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-deny-0001","step_index":4,"state":"ACTIVE","step_type":"tool","tool_name":"run_command","tool_info":{"name":"run_command","parameters":{"CommandLine":"pwd"}}}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-deny-0001","step_index":4,"state":"DONE","step_type":"tool","tool_name":"run_command","duration_seconds":0.011694153,"tool_info":{"name":"run_command","parameters":{"CommandLine":"pwd"}}}}',  # noqa: E501
    '{"event":"result","result":{"conversation_id":"fake-conv-deny-0001","status":"SUCCESS","response":"","duration_seconds":9.741200387,"num_turns":1,"usage":{"input_tokens":19214,"output_tokens":1387,"thinking_tokens":1231,"cache_read_tokens":8128,"total_tokens":20601},"denied_actions":[{"action":"command","display_name":"RunCommand"}]}}',  # noqa: E501
]

#: agy's own headless-auto-deny notice, printed to stderr (verbatim, minus
#: the "jetski:" program name which is an internal detail).
DENY_TURN_STDERR = [
    'jetski: no output produced \u2014 a tool required the "command" permission '
    "that headless mode cannot prompt for, so it was auto-denied. Add an "
    "allow-rule under permissions.allow in settings.json (e.g. "
    "command(<target>)). Alternatively, re-run with "
    "--dangerously-skip-permissions to auto-approve all tools."
]

WARM_TURN1_STDOUT = [
    '{"event":"init","conversation_id":"fake-conv-warm-0001","init":{"cwd":"/home/user/project","tools":["ask_custom_permission","ask_permission","ask_question","browser_click_element","browser_drag_pixel_to_pixel","browser_get_dom","browser_get_network_request","browser_input","browser_list_network_requests","browser_mouse_down","browser_mouse_up","browser_move_mouse","browser_press_key","browser_refresh_page","browser_resize_window","browser_scroll","browser_scroll_dom","browser_select_option","browser_subagent","call_mcp_tool","capture_browser_console_logs","capture_browser_screenshot","click_browser_pixel","command_status","define_subagent","delete_knowledge","execute_browser_javascript","find_by_name","finish","generate_image","grep_search","invoke_subagent","list_browser_pages","list_dir","list_permissions","list_resources","manage_inbox","manage_subagents","manage_task","multi_replace_file_content","notebook_edit","notebook_execution","open_browser_url","read_browser_page","read_resource","read_url_content","replace_file_content","run_command","schedule","search_web","sed_file","send_command_input","send_message","view_file","wait","wait_5_seconds","write_to_file"],"permission_mode":"request-review"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-warm-0001","step_index":0,"state":"DONE","step_type":"user_input"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-warm-0001","step_index":1,"state":"DONE","step_type":"agent_response","text_delta":"OK\\n","duration_seconds":0.012281372,"usage":{"input_tokens":13242,"output_tokens":79,"thinking_tokens":78,"cache_read_tokens":0,"total_tokens":13321}}}',  # noqa: E501
    '{"event":"result","result":{"conversation_id":"fake-conv-warm-0001","status":"SUCCESS","response":"OK\\n","duration_seconds":1.7223431470000001,"num_turns":1,"usage":{"input_tokens":13242,"output_tokens":79,"thinking_tokens":78,"cache_read_tokens":0,"total_tokens":13321}}}',  # noqa: E501
]

WARM_TURN2_STDOUT = [
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-warm-0001","step_index":2,"state":"DONE","step_type":"user_input"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-warm-0001","step_index":3,"state":"ACTIVE","step_type":"agent_response","text_delta":"DONE"}}',  # noqa: E501
    '{"event":"step_update","step_update":{"conversation_id":"fake-conv-warm-0001","step_index":3,"state":"DONE","step_type":"agent_response","text_delta":"\\n","duration_seconds":0.015063245,"usage":{"input_tokens":13394,"output_tokens":54,"thinking_tokens":53,"cache_read_tokens":0,"total_tokens":13448}}}',  # noqa: E501
    '{"event":"result","result":{"conversation_id":"fake-conv-warm-0001","status":"SUCCESS","response":"DONE\\n","duration_seconds":2.727187934,"num_turns":2,"usage":{"input_tokens":26636,"output_tokens":133,"thinking_tokens":131,"cache_read_tokens":0,"total_tokens":26769}}}',  # noqa: E501
]


# -- fixtures / helpers ------------------------------------------------


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "events.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _fake_env(tmp_path: Path, spec: dict, *, argv_log: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env["NVSH_FAKE_EVENTS"] = str(_write_spec(tmp_path, spec))
    if argv_log:
        env["NVSH_FAKE_ARGV_LOG"] = str(tmp_path / "argv.log")
    return env


def _request(prompt: str = "say OK") -> AgentRequest:
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt=prompt)


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", cwd="/home/user/project")


def _read_argv_log(tmp_path: Path) -> list[list[str]]:
    log_path = tmp_path / "argv.log"
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]


# -- criterion 1: cold argv shape and stream-json mapping ---------------


def test_cold_argv_shape(tmp_path):
    spec = {"stdout": TEXT_TURN_STDOUT, "exit_code": 0}
    env = _fake_env(tmp_path, spec, argv_log=True)
    agent = AgyAgent(binary="agy", model="Gemini 3.8 Flash (High)", effort="high", env=env)
    agent.start()
    try:
        list(agent.run(_request("say OK"), _context()))
    finally:
        agent.close()

    argv_calls = _read_argv_log(tmp_path)
    assert len(argv_calls) == 1
    argv = argv_calls[0]
    # criterion 1, verbatim shape: -p <prompt> --output-format stream-json
    # --model <m> --effort <e>. The prompt carries the system brief plus the
    # request text (build_full_prompt), so only its presence as the token
    # right after -p is asserted, not its exact bytes.
    assert argv[0] == "-p"
    assert "say OK" in argv[1]
    assert argv[2:] == [
        "--output-format",
        "stream-json",
        "--model",
        "Gemini 3.8 Flash (High)",
        "--effort",
        "high",
    ]
    assert "--dangerously-skip-permissions" not in argv


def test_text_turn_maps_text_delta_and_done(tmp_path):
    spec = {"stdout": TEXT_TURN_STDOUT, "exit_code": 0}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(env=env)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()

    deltas = [e.text for e in events if e.kind == EventKind.TEXT_DELTA]
    assert deltas == ["OK", "\n"]
    assert events[-1].kind == EventKind.DONE
    assert events[-1].text == "OK\n"
    assert not any(e.kind == EventKind.ERROR for e in events)


def test_tool_turn_maps_tool_call_and_result(tmp_path):
    spec = {"stdout": TOOL_TURN_STDOUT, "exit_code": 0}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(env=env)
    agent.start()
    try:
        events = list(agent.run(_request("list files"), _context()))
    finally:
        agent.close()

    tool_calls = [e for e in events if e.kind == EventKind.TOOL_CALL]
    tool_results = [e for e in events if e.kind == EventKind.TOOL_RESULT]
    assert [e.tool for e in tool_calls] == ["run_command", "run_command"]
    assert [e.args.get("CommandLine") for e in tool_calls] == ["ls", "ls -a"]
    assert [e.tool for e in tool_results] == ["run_command", "run_command"]
    # The second run_command carried an "output" field; the first did not.
    assert tool_results[1].result == ".  ..\r\n"
    assert events[-1].kind == EventKind.DONE
    assert "empty" in events[-1].text


# -- criterion 1 (warm) / criterion 3 (auto-deny -> STATUS) --------------


def test_auto_deny_is_status_not_error(tmp_path):
    spec = {"stdout": DENY_TURN_STDOUT, "stderr": DENY_TURN_STDERR, "exit_code": 0}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(env=env)
    agent.start()
    try:
        events = list(agent.run(_request("run pwd"), _context()))
    finally:
        agent.close()

    assert not any(e.kind == EventKind.ERROR for e in events)
    status_events = [e for e in events if e.kind == EventKind.STATUS]
    assert status_events, "expected an auto-deny STATUS event"
    assert "auto-denied" in status_events[0].text
    assert events[-1].kind == EventKind.DONE
    # The tool's own ERROR state still surfaces as a TOOL_RESULT (the failed
    # list_dir call), not as a stream-ending adapter ERROR.
    tool_results = [e for e in events if e.kind == EventKind.TOOL_RESULT]
    assert any("Permission denied" in str(e.result) for e in tool_results)


def test_auto_deny_with_no_result_line_is_still_status(tmp_path):
    """The literal "no output produced" case: stdout is empty, only stderr
    carries the auto-deny notice. Must still end in STATUS + DONE, never a
    bare ERROR synthesized from the stderr tail."""
    spec = {"stdout": [], "stderr": DENY_TURN_STDERR, "exit_code": 0}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(env=env)
    agent.start()
    try:
        events = list(agent.run(_request("run pwd"), _context()))
    finally:
        agent.close()

    assert [e.kind for e in events] == [EventKind.STATUS, EventKind.DONE]
    assert "auto-denied" in events[0].text


# -- warm sessions (criterion 1) -----------------------------------------


def test_warm_session_single_process_two_turns(tmp_path):
    spec = {
        "turns": [
            {"stdout": WARM_TURN1_STDOUT, "exit_code": 0},
            {"stdout": WARM_TURN2_STDOUT, "exit_code": 0},
        ]
    }
    env = _fake_env(tmp_path, spec, argv_log=True)
    agent = AgyAgent(warm=True, env=env)
    agent.start()
    try:
        first = list(agent.run(_request("say OK"), _context()))
        second = list(agent.run(_request("say DONE"), _context()))
    finally:
        agent.close()

    assert first[-1].kind == EventKind.DONE
    assert first[-1].text == "OK\n"
    assert second[-1].kind == EventKind.DONE
    assert second[-1].text == "DONE\n"

    # Exactly one agy process for both turns -- that is the entire point of
    # "warm".
    argv_calls = _read_argv_log(tmp_path)
    assert len(argv_calls) == 1
    argv = argv_calls[0]
    assert argv[0] == "-p="
    assert "--input-format" in argv
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"

    # A second turn picked up the same conversation id agy assigned on init.
    assert agent._conversation_id == "fake-conv-warm-0001"


def test_cold_resume_passes_conversation_flag(tmp_path):
    spec = {"stdout": TEXT_TURN_STDOUT, "exit_code": 0}
    env = _fake_env(tmp_path, spec, argv_log=True)
    agent = AgyAgent(conversation_id="fake-conv-text-0001", env=env)
    agent.start()
    try:
        list(agent.run(_request(), _context()))
    finally:
        agent.close()

    argv = _read_argv_log(tmp_path)[0]
    assert "--conversation" in argv
    assert argv[argv.index("--conversation") + 1] == "fake-conv-text-0001"


# -- criterion 2: capabilities, and no settings.json / skip-permissions -


def test_capabilities_cold():
    agent = AgyAgent()
    caps = agent.capabilities()
    assert isinstance(caps, Capabilities)
    assert caps.tool_calling is False
    assert caps.unmediated_file_access is True
    assert caps.local_model is False
    assert caps.approval == "none"
    assert caps.persistent_session is False
    assert caps.effort is True


def test_capabilities_warm_reports_persistent_session():
    agent = AgyAgent(warm=True)
    assert agent.capabilities().persistent_session is True


def test_module_never_touches_settings_json_or_skip_permissions():
    """The module docstring is allowed to *mention* the flag (explaining why
    it is never sent); the code below it must never construct or reference
    it, and must never open agy's ``settings.json``."""
    import ast

    module_path = Path(__import__("nvsh.agent.agy", fromlist=["__file__"]).__file__)
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree, clean=False) or ""
    code_only = source.replace(docstring, "", 1)

    assert "settings.json" not in code_only
    assert "--dangerously-skip-permissions" not in code_only


# -- adapter contract: cancellation, teardown, error propagation --------


def test_error_result_status_maps_to_error_event(tmp_path):
    stdout = [
        '{"event":"init","conversation_id":"fake-conv-err-0001","init":{"cwd":"/home/user/project","tools":[],"permission_mode":"request-review"}}',  # noqa: E501
        '{"event":"result","result":{"conversation_id":"fake-conv-err-0001","status":"ERROR","error":"backend unreachable"}}',  # noqa: E501
    ]
    spec = {"stdout": stdout, "exit_code": 1}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(env=env)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert events[-1].kind == EventKind.ERROR
    assert events[-1].error == "backend unreachable"


def test_teardown_idempotent(tmp_path):
    spec = {"stdout": TEXT_TURN_STDOUT, "exit_code": 0}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(env=env)
    agent.start()
    list(agent.run(_request(), _context()))
    agent.close()
    agent.close()  # must not raise


def test_warm_teardown_idempotent(tmp_path):
    spec = {"turns": [{"stdout": WARM_TURN1_STDOUT, "exit_code": 0}]}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(warm=True, env=env)
    agent.start()
    list(agent.run(_request(), _context()))
    agent.close()
    agent.close()  # must not raise


# -- cancellation (parity with the shared conformance suite's guarantee) -


def test_cancel_mid_stream_stops_after_current_event(tmp_path):
    spec = {"stdout": TOOL_TURN_STDOUT, "exit_code": 0}
    env = _fake_env(tmp_path, spec)
    agent = AgyAgent(env=env)
    agent.start()
    seen = []
    try:
        for event in agent.run(_request(), _context()):
            seen.append(event)
            if len(seen) == 1:
                agent.cancel()
    finally:
        agent.close()
    # Cancellation is checked between yields: nothing past the first event
    # already in flight is seen, and no DONE/ERROR ever arrives.
    assert len(seen) == 1
    assert not any(e.kind in (EventKind.DONE, EventKind.ERROR) for e in seen)


# -- criterion 3: live smoke, opt-in only --------------------------------


@pytest.mark.skipif(
    os.environ.get("NVSH_LIVE_AGY") != "1",
    reason="live agy smoke only runs with NVSH_LIVE_AGY=1 (needs a real, logged-in agy binary)",
)
def test_live_agy_says_ok():
    agent = AgyAgent()
    agent.start()
    try:
        events = list(
            agent.run(
                AgentRequest(kind=RequestKind.EXPLICIT, prompt="Say OK and nothing else."),
                AgentContext(),
            )
        )
    finally:
        agent.close()
    assert not any(e.kind == EventKind.ERROR for e in events)
    assert events[-1].kind == EventKind.DONE
    assert "OK" in events[-1].text
