"""Tests for nvsh.agent.pi.PiAgent (task t9).

Acceptance criteria covered:

* argv hygiene -- ``test_build_argv_*``
* LF-only framing, tolerating literal U+2028 inside a JSON string --
  ``test_reader_*``
* event mapping (message_update deltas, tool_execution_*,
  extension_ui_request, agent_end) -- ``test_run_*``
* cancel() sends abort and returns within 1s; a killed process yields ERROR
  and never hangs -- ``test_cancel_*``, ``test_killed_process_*``
* no file appears under ~/.pi/agent/sessions after a run --
  ``test_no_file_under_dot_pi_agent_sessions``
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from nvsh.agent.base import AgentContext, AgentRequest, Capabilities, EventKind, RequestKind
from nvsh.agent.pi import PiAgent, build_prompt

FAKES_DIR = Path(__file__).resolve().parent / "fakes"


def _env(tmp_path: Path, **extra: str) -> dict:
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env["HOME"] = str(tmp_path / "home")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    env.update(extra)
    return env


def _request(prompt: str = "", command: str = "ls /nope", exit_code: int = 2) -> AgentRequest:
    return AgentRequest(
        kind=RequestKind.FAILURE, prompt=prompt, command=command, exit_code=exit_code
    )


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="ls: /nope: No such file", cwd="/tmp")


# --- argv -------------------------------------------------------------


def test_build_argv_contains_hygiene_flags_in_order(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path), provider="nemotron", model="associate")
    argv = agent.build_argv()

    assert argv[0] == "pi"
    assert "--mode" in argv and argv[argv.index("--mode") + 1] == "rpc"

    hygiene = [
        "--no-context-files",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-approve",
    ]
    hygiene_indices = [argv.index(flag) for flag in hygiene]
    assert hygiene_indices == sorted(hygiene_indices)

    assert "--session-dir" in argv
    session_dir = argv[argv.index("--session-dir") + 1]
    assert session_dir.endswith(os.path.join("nvsh", "pi-sessions"))

    assert "-e" in argv
    ext_path = argv[argv.index("-e") + 1]
    assert ext_path.endswith(os.path.join("pi_ext", "approval.ts"))

    # provider/model come after the hygiene flags
    assert argv.index("--provider") > argv.index("--no-approve")
    assert argv[argv.index("--provider") + 1] == "nemotron"
    assert argv[argv.index("--model") + 1] == "associate"


def test_build_argv_session_dir_respects_xdg_state_home(tmp_path):
    env = _env(tmp_path)
    agent = PiAgent(pi_path="pi", env=env)
    argv = agent.build_argv()
    session_dir = argv[argv.index("--session-dir") + 1]
    assert session_dir == str(Path(env["XDG_STATE_HOME"]) / "nvsh" / "pi-sessions")


def test_build_argv_default_provider_model_from_config(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    argv = agent.build_argv()
    assert argv[argv.index("--provider") + 1] == "nemotron"
    assert argv[argv.index("--model") + 1] == "associate"


def test_no_api_key_ever_in_argv(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    argv = agent.build_argv()
    joined = " ".join(argv)
    assert "api-key" not in joined
    assert "--api-key" not in argv


# --- capabilities -------------------------------------------------------


def test_capabilities(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    caps = agent.capabilities()
    assert caps == Capabilities(
        streaming=True,
        tool_calling=True,
        cancellation=True,
        persistent_session=True,
        local_model=True,
    )


# --- run() / event mapping ----------------------------------------------


def test_run_text_delta_and_done(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert EventKind.TEXT_DELTA in kinds
    assert kinds[-1] == EventKind.DONE
    delta_events = [e for e in events if e.kind == EventKind.TEXT_DELTA]
    assert delta_events[0].text == "looking at the failure"


def test_run_tool_call_and_result(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    try:
        events = list(agent.run(_request(prompt="please TOOL this"), _context()))
    finally:
        agent.close()
    kinds = [e.kind for e in events]
    assert EventKind.TOOL_CALL in kinds
    assert EventKind.TOOL_RESULT in kinds
    call = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert call.tool == "bash"
    assert call.args == {"command": "nvidia-smi"}
    result = next(e for e in events if e.kind == EventKind.TOOL_RESULT)
    assert result.tool == "bash"


def test_run_extension_ui_request_maps_to_proposal(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    try:
        events = list(agent.run(_request(prompt="please PROPOSE this"), _context()))
    finally:
        agent.close()
    proposal_events = [e for e in events if e.kind == EventKind.PROPOSAL]
    assert len(proposal_events) == 1
    event = proposal_events[0]
    assert event.proposal is not None
    assert event.proposal.command == "sudo nvidia-smi -pm 1"
    assert event.args.get("request_id") == "ui-1"


def test_respond_ui_writes_extension_ui_response(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    try:
        list(agent.run(_request(prompt="please PROPOSE this"), _context()))
        # Must not raise, even though the fake pi doesn't read it further.
        agent.respond_ui("ui-1", confirmed=True)
    finally:
        agent.close()


def test_unknown_wire_event_maps_to_status(tmp_path):
    # queue_update is a real pi event type PiAgent does not special-case.
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(
        [{"type": "queue_update", "steering": []}, {"type": "agent_end"}]
    )
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert events[0].kind == EventKind.STATUS
    assert events[0].text == "queue_update"
    assert events[-1].kind == EventKind.DONE


# -- d11: rpc lifecycle bookkeeping is not panel material -------------------
#
# On the Spark the panel printed "... agent_start", "... turn_start",
# "... message_start", "... message_end": pi's per-turn lifecycle events,
# which say nothing an operator can act on, reached the panel through the
# catch-all STATUS fallback. They must map to no event at all.


def test_lifecycle_events_are_dropped_entirely(tmp_path):
    script = [
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start"},
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "hello"},
        },
        {"type": "tool_execution_start", "toolName": "bash", "args": {"command": "ls /"}},
        {"type": "tool_execution_update", "toolName": "bash"},
        {"type": "tool_execution_end", "toolName": "bash", "result": {"output": "bin"}},
        {"type": "message_end"},
        {"type": "turn_end"},
        {"type": "agent_end"},
    ]
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(script)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert [e.kind for e in events] == [
        EventKind.TEXT_DELTA,
        EventKind.TOOL_CALL,
        EventKind.TOOL_RESULT,
        EventKind.DONE,
    ]


# --- cancel / kill --------------------------------------------------------


def test_cancel_returns_within_one_second_and_stops_stream(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    seen = []
    try:
        for event in agent.run(_request(prompt="TWODELTA please"), _context()):
            seen.append(event)
            if len(seen) == 1:
                start = time.monotonic()
                agent.cancel()
                elapsed = time.monotonic() - start
                assert elapsed < 1.0
    finally:
        agent.close()
    # Cancellation is checked between yields: nothing scripted after the
    # in-flight event is observed.
    assert len(seen) == 1


def test_killed_process_yields_error_and_does_not_hang(tmp_path):
    env = _env(tmp_path, FAKE_PI_CRASH="1")
    agent = PiAgent(pi_path="pi", env=env)
    agent.start()
    done = threading.Event()
    events = []

    def _drive():
        try:
            for event in agent.run(_request(), _context()):
                events.append(event)
        finally:
            done.set()

    thread = threading.Thread(target=_drive)
    thread.start()
    finished = done.wait(timeout=5.0)
    agent.close()
    assert finished, "run() hung after the pi process crashed"
    assert events, "expected at least one event before the crash was detected"
    assert events[-1].kind == EventKind.ERROR


def test_no_file_under_dot_pi_agent_sessions(tmp_path):
    env = _env(tmp_path)
    agent = PiAgent(pi_path="pi", env=env)
    agent.start()
    try:
        list(agent.run(_request(), _context()))
    finally:
        agent.close()
    dot_pi_sessions = Path(env["HOME"]) / ".pi" / "agent" / "sessions"
    files = list(dot_pi_sessions.rglob("*")) if dot_pi_sessions.exists() else []
    assert files == []


# --- persistent session / passthroughs ------------------------------------


def test_run_reuses_a_single_process(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    try:
        list(agent.run(_request(), _context()))
        proc1 = agent._proc
        list(agent.run(_request(), _context()))
        proc2 = agent._proc
    finally:
        agent.close()
    assert proc1 is proc2


def test_new_session_and_switch_session_are_passthroughs(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    try:
        agent.new_session()
        agent.switch_session("/tmp/some-session.jsonl")
    finally:
        agent.close()


# --- teardown --------------------------------------------------------------


def test_close_is_idempotent(tmp_path):
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent.start()
    list(agent.run(_request(), _context()))
    agent.close()
    agent.close()


# --- prompt builder ---------------------------------------------------------


def test_build_prompt_includes_command_and_output():
    prompt = build_prompt(_request(), _context())
    assert "ls /nope" in prompt
    assert "No such file" in prompt


# --- reader framing (U+2028 tolerance) -------------------------------------


def test_reader_tolerates_u2028_inside_json_string(tmp_path):
    """The reader must split stdout on LF only, never on U+2028/U+2029."""
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    text_with_separator = "before after"
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(
        [
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": text_with_separator},
            },
            {"type": "agent_end"},
        ]
    )
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    deltas = [e for e in events if e.kind == EventKind.TEXT_DELTA]
    assert len(deltas) == 1
    assert deltas[0].text == text_with_separator


# -- d8: Proposal.command is the bare tool-call command --------------------
#
# Verification on spark with the real pi 0.84.2 + associate model showed
# every approval proposal carrying the *rendered panel text* as its command
# ("nvsh: run this command?\ncommand: type ls"): the approval extension
# folded the human prompt into ctx.ui.select()'s title, the only string
# field pi's `select` request carries, and _map_event's old fallback chain
# (command -> message -> title) then read that whole title as the command.
# The scripts below replay the recorded shapes (docs/pi-rpc.md plus the
# audit log from that run) through the scripted fake pi.

#: What the pre-fix extension put in `title` -- never a command.
PANEL_TEXT = "nvsh: run this command?\ncommand: type ls"


def _proposals_from_script(tmp_path: Path, script: list[dict]):
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(script)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    return [e for e in events if e.kind == EventKind.PROPOSAL]


def _approval_request(command: str, reason: str = "", request_id: str = "ui-7") -> dict:
    """One `extension_ui_request` exactly as the fixed approval.ts produces."""
    title = json.dumps(
        {"nvsh": "approval", "v": 1, "tool": "bash", "command": command, "reason": reason}
    )
    return {
        "type": "extension_ui_request",
        "id": request_id,
        "method": "select",
        "title": title,
        "options": ["once", "session", "user", "deny"],
    }


def test_approval_request_yields_the_bare_command(tmp_path):
    script = [
        {"type": "turn_start"},
        {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "text_delta",
                "delta": "`ls` is not missing -- let me check how the shell resolves it.",
            },
        },
        _approval_request("type ls", reason="check how the shell resolves ls"),
        {"type": "agent_end"},
    ]
    proposals = _proposals_from_script(tmp_path, script)
    assert len(proposals) == 1
    proposal = proposals[0].proposal
    assert proposal is not None
    assert proposal.command == "type ls"
    assert "nvsh: run this command?" not in proposal.command
    assert proposal.rationale == "check how the shell resolves ls"
    assert proposals[0].args.get("request_id") == "ui-7"


def test_approval_request_preserves_a_multiline_command_verbatim(tmp_path):
    command = 'python3 -c "import torch\nprint(torch.cuda.is_available())"'
    proposals = _proposals_from_script(
        tmp_path, [_approval_request(command), {"type": "agent_end"}]
    )
    assert proposals[0].proposal.command == command


def test_panel_text_in_title_never_becomes_the_command(tmp_path):
    """The pre-fix wire shape (d8) must not produce a runnable command."""
    script = [
        {
            "type": "extension_ui_request",
            "id": "ui-1",
            "method": "select",
            "title": PANEL_TEXT,
            "options": ["once", "session", "user", "deny"],
        },
        {"type": "agent_end"},
    ]
    proposals = _proposals_from_script(tmp_path, script)
    proposal = proposals[0].proposal
    assert proposal.command == ""
    assert "nvsh: run this command?" not in proposal.command
    assert PANEL_TEXT in proposal.rationale


def test_explicit_command_field_still_wins(tmp_path):
    """A dialog that does carry a structured `command` field is unchanged."""
    script = [
        {
            "type": "extension_ui_request",
            "id": "ui-2",
            "method": "confirm",
            "title": "Apply fix?",
            "command": "sudo nvidia-smi -pm 1",
            "message": "Run: sudo nvidia-smi -pm 1",
        },
        {"type": "agent_end"},
    ]
    proposals = _proposals_from_script(tmp_path, script)
    assert proposals[0].proposal.command == "sudo nvidia-smi -pm 1"


@pytest.mark.skipif(
    not (os.environ.get("NVSH_LIVE_PI") == "1"),
    reason="set NVSH_LIVE_PI=1 and have the real pi binary on PATH to run this",
)
def test_live_pi_get_state(tmp_path):
    import shutil
    import subprocess

    pi_path = shutil.which("pi")
    if pi_path is None:
        pytest.skip("no real pi binary on PATH")

    env = _env(tmp_path)
    # Remove the fakes dir from PATH so the real pi binary is used.
    env["PATH"] = os.environ.get("PATH", "")
    proc = subprocess.Popen(
        [pi_path, "--mode", "rpc", "--no-session"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )
    try:
        proc.stdin.write((json.dumps({"type": "get_state", "id": "probe"}) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        assert line, "no output from real pi rpc process"
        obj = json.loads(line)
        assert obj.get("type") == "response"
        assert obj.get("command") == "get_state"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
