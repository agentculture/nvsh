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
from nvsh.agent.pi import PiAgent, PiRpcError, build_prompt

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
    assert "--mode" in argv
    assert argv[argv.index("--mode") + 1] == "rpc"

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
    # compaction_start is a real pi event type PiAgent does not special-case.
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(
        [{"type": "compaction_start"}, {"type": "agent_end"}]
    )
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert events[0].kind == EventKind.STATUS
    assert events[0].text == "compaction_start"
    assert events[-1].kind == EventKind.DONE


def test_a_steering_queue_update_is_not_panel_material(tmp_path):
    """d16: every steer changes pi's queue; the panel already said it steered."""
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(
        [{"type": "queue_update", "steering": ["free -h"]}, {"type": "agent_end"}]
    )
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    assert [e.kind for e in events] == [EventKind.DONE]


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


class _ExitedProc:
    """Stands in for a pi process that has already exited."""

    def __init__(self, code: int = 3) -> None:
        self._code = code
        self.stdin = None

    def poll(self):
        return self._code


def test_events_already_queued_when_pi_exits_are_drained_before_the_exit(tmp_path):
    """A fast final burst must not be lost to the process-exited report."""
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent._proc = _ExitedProc()
    agent._queue.put(
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "half a thought"},
        }
    )
    events = list(agent._events())
    assert [event.kind for event in events] == [EventKind.TEXT_DELTA, EventKind.ERROR]
    assert events[0].text == "half a thought"
    assert "exited with code 3" in events[-1].error


def test_a_queued_error_event_ends_the_stream_instead_of_the_generic_exit(tmp_path):
    """pi's own last words beat 'pi process exited' when it left both."""
    agent = PiAgent(pi_path="pi", env=_env(tmp_path))
    agent._proc = _ExitedProc()
    agent._queue.put({"type": "error", "error": "model refused the request"})
    agent._queue.put({"type": "agent_end"})
    events = list(agent._events())
    assert [event.kind for event in events] == [EventKind.ERROR]
    assert events[0].error == "model refused the request"


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


def test_new_session_returns_the_session_file_pi_chose(tmp_path):
    """pi names its own session file; the caller has to be told which one.

    There is no rpc command that chooses the name (``docs/pi-rpc.md``), so
    ``new_session()`` reads it back from ``get_state`` -- otherwise the
    daemon would later ``switch_session`` to a path pi never wrote.
    """
    env = _env(tmp_path)
    agent = PiAgent(pi_path="pi", env=env)
    agent.start()
    try:
        path = agent.new_session()
    finally:
        agent.close()
    assert path, "new_session() must report the session file pi created"
    assert Path(path).is_file()
    assert Path(path).parent == Path(env["XDG_STATE_HOME"]) / "nvsh" / "pi-sessions"


# --- d14: commands are acknowledged, never pipelined -----------------------
#
# pi 0.85.1 drops both commands -- silently and permanently -- when a second
# command line arrives before it has acknowledged the first. On the Spark the
# daemon wrote new_session and then, a millisecond later, the prompt; pi went
# mute and the operator's panel waited out the client's 120 s stream timeout
# with nothing on screen. ``FAKE_PI_STRICT_ACK=1`` makes tests/fakes/pi
# behave the same way, so a client that pipelines hangs here too.


def test_session_command_then_prompt_still_answers_under_a_strict_pi(tmp_path):
    env = _env(tmp_path, FAKE_PI_STRICT_ACK="1")
    agent = PiAgent(pi_path="pi", env=env)
    agent.start()
    try:
        agent.new_session()
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    kinds = [event.kind for event in events]
    assert EventKind.TEXT_DELTA in kinds, "a strict pi answered nothing: commands were pipelined"
    assert kinds[-1] == EventKind.DONE


def test_new_session_returns_only_after_its_ack(tmp_path):
    """The ack is what makes the following prompt safe -- so it is waited for."""
    env = _env(tmp_path, FAKE_PI_STRICT_ACK="1", FAKE_PI_ACK_DELAY="0.4")
    agent = PiAgent(pi_path="pi", env=env)
    agent.start()
    try:
        start = time.monotonic()
        agent.new_session()
        elapsed = time.monotonic() - start
    finally:
        agent.close()
    assert elapsed >= 0.4, "new_session() returned before pi acknowledged it"


# --- d14: a backend that will not start says so out loud -------------------


def _exiting_pi(tmp_path: Path, code: int, stderr: str) -> Path:
    """A ``pi`` stub that prints to stderr and exits at once."""
    bindir = tmp_path / "deadbin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "pi"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stderr.write({stderr!r} + '\\n')\n"
        f"sys.exit({code})\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bindir


def test_start_reports_a_pi_that_exits_at_once(tmp_path):
    env = _env(tmp_path)
    env["PATH"] = str(_exiting_pi(tmp_path, 3, "pi: cannot find module foo")) + (
        os.pathsep + os.environ.get("PATH", "")
    )
    agent = PiAgent(pi_path="pi", env=env)
    start = time.monotonic()
    with pytest.raises(PiRpcError) as excinfo:
        agent.start()
    elapsed = time.monotonic() - start
    agent.close()
    message = str(excinfo.value)
    assert elapsed < 5.0, "a dead pi must be reported at once, not waited out"
    assert "code 3" in message
    assert "cannot find module foo" in message


def test_start_redacts_secrets_out_of_the_stderr_tail(tmp_path):
    env = _env(tmp_path)
    secret = "hf_" + "z" * 24
    env["PATH"] = str(_exiting_pi(tmp_path, 1, f"auth failed\nHF_TOKEN={secret}")) + (
        os.pathsep + os.environ.get("PATH", "")
    )
    agent = PiAgent(pi_path="pi", env=env)
    with pytest.raises(PiRpcError) as excinfo:
        agent.start()
    agent.close()
    message = str(excinfo.value)
    assert "auth failed" in message
    assert secret not in message


def test_ack_timeout_is_bounded_and_names_the_backend(tmp_path):
    """A pi that never answers a command is an error, not an unbounded wait."""
    bindir = tmp_path / "mutebin"
    bindir.mkdir()
    stub = bindir / "pi"
    stub.write_text(
        "#!/usr/bin/env python3\nimport time\nwhile True:\n    time.sleep(1)\n", encoding="utf-8"
    )
    stub.chmod(0o755)
    env = _env(tmp_path, NVSH_PI_ACK_TIMEOUT="0.5")
    env["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
    agent = PiAgent(pi_path="pi", env=env)
    start = time.monotonic()
    with pytest.raises(PiRpcError) as excinfo:
        agent.start()
    elapsed = time.monotonic() - start
    agent.close()
    assert 0.4 <= elapsed < 5.0
    assert "get_state" in str(excinfo.value)
    assert "still running" in str(excinfo.value)


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


# --- d21: the env pi (and the approval extension under it) inherits ------


def test_child_env_names_nvsh_bin_and_the_xdg_dirs(tmp_path):
    """d21: the approval extension shells out to ``nvsh``; pi's env must say where.

    The extension runs inside a pi that the daemon spawned, and it reads
    the approval store through ``nvsh approve check``. When ``NVSH_BIN`` is
    missing from that env and the daemon's ``PATH`` has no ``nvsh`` on it,
    the spawn fails with ENOENT and every command is asked about forever.
    """
    env = _env(tmp_path)
    env.pop("NVSH_BIN", None)
    env.pop("XDG_CONFIG_HOME", None)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "nvsh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "nvsh").chmod(0o755)
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]

    agent = PiAgent(pi_path="pi", env=env)

    assert agent._env["NVSH_BIN"] == str(fake_bin / "nvsh")
    assert agent._env["XDG_CONFIG_HOME"] == str(Path(env["HOME"]) / ".config")
    assert agent._env["XDG_STATE_HOME"] == env["XDG_STATE_HOME"]


def test_child_env_keeps_an_explicit_nvsh_bin_and_xdg_config_home(tmp_path):
    env = _env(tmp_path, NVSH_BIN="/opt/nvsh/bin/nvsh", XDG_CONFIG_HOME=str(tmp_path / "cfg"))
    agent = PiAgent(pi_path="pi", env=env)
    assert agent._env["NVSH_BIN"] == "/opt/nvsh/bin/nvsh"
    assert agent._env["XDG_CONFIG_HOME"] == str(tmp_path / "cfg")


def test_child_env_leaves_xdg_runtime_dir_alone_when_there_is_no_run_user(tmp_path, monkeypatch):
    """Never invent a runtime dir: parent and child must agree on the fallback.

    ``nvsh.approvals.runtime_dir()`` falls back to ``/run/user/<uid>`` and
    then to a per-uid temp dir. Setting ``XDG_RUNTIME_DIR`` to something
    else here would split the session store in two.
    """
    env = _env(tmp_path)
    env.pop("XDG_RUNTIME_DIR", None)
    monkeypatch.setattr("nvsh.agent.pi.Path.is_dir", lambda self: False)
    agent = PiAgent(pi_path="pi", env=env)
    assert "XDG_RUNTIME_DIR" not in agent._env


def test_child_env_passes_an_existing_xdg_runtime_dir_through(tmp_path):
    runtime = tmp_path / "run"
    runtime.mkdir()
    env = _env(tmp_path, XDG_RUNTIME_DIR=str(runtime))
    agent = PiAgent(pi_path="pi", env=env)
    assert agent._env["XDG_RUNTIME_DIR"] == str(runtime)


# --- d17: the assistant text before a tool call is the proposal's rationale


def _proposal_from(tmp_path, script):
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(script)
    agent.start()
    try:
        events = list(agent.run(_request(), _context()))
    finally:
        agent.close()
    return [e for e in events if e.kind == EventKind.PROPOSAL]


def _delta(text: str) -> dict:
    return {
        "type": "message_update",
        "assistantMessageEvent": {"type": "text_delta", "delta": text},
    }


def test_text_streamed_before_a_tool_call_becomes_the_rationale(tmp_path):
    script = [
        _delta("`perf` is not installed. "),
        _delta("The package that ships it is linux-tools."),
        _approval_request("apt install -y linux-tools", reason=""),
        {"type": "agent_end"},
    ]
    proposals = _proposal_from(tmp_path, script)
    assert proposals[0].proposal.rationale == (
        "`perf` is not installed. The package that ships it is linux-tools."
    )


def test_only_the_text_since_the_last_tool_result_counts(tmp_path):
    script = [
        _delta("first let me look around."),
        {"type": "tool_execution_start", "toolName": "bash", "args": {"command": "ls"}},
        {"type": "tool_execution_end", "toolName": "bash", "result": {"output": "x"}},
        _delta("now I know what is missing."),
        _approval_request("apt install -y linux-tools", reason=""),
        {"type": "agent_end"},
    ]
    proposals = _proposal_from(tmp_path, script)
    assert proposals[0].proposal.rationale == "now I know what is missing."


def test_the_envelope_reason_still_wins_over_the_streamed_text(tmp_path):
    script = [
        _delta("some thinking out loud."),
        _approval_request("free -h", reason="check memory pressure"),
        {"type": "agent_end"},
    ]
    proposals = _proposal_from(tmp_path, script)
    assert proposals[0].proposal.rationale == "check memory pressure"


def test_a_long_rationale_is_trimmed_to_six_hundred_characters(tmp_path):
    script = [
        _delta("x" * 1200),
        _approval_request("free -h", reason=""),
        {"type": "agent_end"},
    ]
    rationale = _proposal_from(tmp_path, script)[0].proposal.rationale
    assert len(rationale) <= 600
    assert rationale.endswith("…")


def test_the_rationale_buffer_is_cleared_between_runs(tmp_path):
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(
        [_delta("only for the first turn."), {"type": "agent_end"}]
    )
    agent.start()
    try:
        list(agent.run(_request(), _context()))
        assert agent._rationale_text() == ""
    finally:
        agent.close()


# --- d16: steering a running turn ----------------------------------------


def test_steer_writes_a_prompt_with_streaming_behavior_steer(tmp_path):
    commands = tmp_path / "commands.jsonl"
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_COMMANDS"] = str(commands)
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps(
        [_approval_request("apt install -y linux-tools"), {"type": "agent_end"}]
    )
    agent._env["NVSH_TEST_PI_AWAIT_UI"] = "1"
    agent.start()
    try:
        stream = agent.run(_request(), _context())
        event = next(e for e in stream if e.kind == EventKind.PROPOSAL)
        assert event.proposal is not None
        assert agent.steer("just run free -h") is True
        agent.respond_ui("ui-7", value="deny", reason="just run free -h")
        list(stream)
    finally:
        agent.close()
    sent = [json.loads(line) for line in commands.read_text(encoding="utf-8").splitlines() if line]
    steers = [c for c in sent if c.get("type") == "prompt" and "streamingBehavior" in c]
    assert steers == [
        {"type": "prompt", "message": "just run free -h", "streamingBehavior": "steer"}
    ]


def test_steer_is_refused_when_no_turn_is_running(tmp_path):
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    agent._env["NVSH_TEST_PI_SCRIPT"] = json.dumps([{"type": "agent_end"}])
    agent.start()
    try:
        assert agent.steer("too late") is False
    finally:
        agent.close()


def test_steer_on_a_closed_agent_is_false_and_never_raises(tmp_path):
    agent = PiAgent(pi_path="pi_scripted", env=_env(tmp_path))
    assert agent.steer("nobody is listening") is False
