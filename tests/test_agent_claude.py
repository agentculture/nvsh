"""Tests for ``nvsh/agent/claude.py`` -- the official ``claude`` CLI adapter.

Three things are proved here, all against ``tests/fakes/claude`` (which
replays ``tests/fakes/claude-2.1.270-thinking-tooluse.jsonl``, recorded from
claude 2.1.270) rather than against a mocked stream:

1. **argv.** ``--model``, ``--effort``, ``--include-partial-messages``,
   ``--permission-prompts host``, the ``--permission-prompt-tool stdio`` that
   actually delivers the prompt, and ``--session-id`` / ``--resume`` on a warm
   adapter. ``--dangerously-skip-permissions`` never appears.
2. **Content-part mapping.** ``thinking``, ``tool_use`` and ``tool_result``
   become ``THINKING`` / ``TOOL_CALL`` / ``TOOL_RESULT``, and the partial
   deltas of a message are not replayed again from its aggregate envelope.
3. **The permission round-trip.** A ``can_use_tool`` control_request becomes a
   ``PROPOSAL``; answering it through :meth:`ClaudeAgent.respond_ui` writes a
   ``control_response`` the fake reads, and allow and deny produce the two
   different wire payloads.

**The mechanism finding this file encodes.** ``--permission-prompts host`` is
the CLI's *default* and on its own routes nothing: without a stdio permission
tool the CLI has no host to ask, and a command that would prompt comes back as
``system/permission_denied`` with ``"This command requires approval"``. The
flag that makes the CLI emit ``control_request``/``can_use_tool`` on stdout
and block for a ``control_response`` on stdin is ``--permission-prompt-tool
stdio`` -- which is exactly what the Agent SDK bundled inside the 2.1.270
binary passes when a ``canUseTool`` callback is supplied. No MCP permission
server is needed, and no ``claude-code-acp``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from nvsh.agent.base import (
    AgentContext,
    AgentRequest,
    EventKind,
    ProposalKind,
    RequestKind,
    Target,
)
from nvsh.agent.claude import ClaudeAgent

FAKES_DIR = Path(__file__).parent / "fakes"
TRANSCRIPT = FAKES_DIR / "claude-2.1.270-thinking-tooluse.jsonl"

#: The control request id the recorded transcript carries.
REQUEST_ID = "11111111-2222-4333-8444-555555555555"


def _request(**kwargs) -> AgentRequest:
    fields = {
        "kind": RequestKind.FAILURE,
        "command": "nvidia-smi",
        "exit_code": 127,
        "prompt": "why did nvidia-smi fail?",
    }
    fields.update(kwargs)
    return AgentRequest(**fields)


def _context() -> AgentContext:
    return AgentContext(platform="dgx-spark", output="command not found", cwd="/workdir")


def _transcript_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env["NVSH_FAKE_TRANSCRIPT"] = str(TRANSCRIPT)
    env["NVSH_FAKE_ARGV"] = str(tmp_path / "argv.jsonl")
    env["NVSH_FAKE_STDIN"] = str(tmp_path / "stdin.jsonl")
    env["NVSH_FAKE_UI_TIMEOUT"] = "10"
    return env


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _drive(agent: ClaudeAgent, answer: dict | None, timeout: float = 30.0) -> list:
    """Run one turn on a worker thread, answering the proposal with ``answer``.

    Threaded so a host that never answers -- or an adapter that never reads
    the prompt -- fails the test instead of hanging the suite.
    """
    events: list = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for event in agent.run(_request(), _context()):
                events.append(event)
                if event.kind is EventKind.PROPOSAL and answer is not None:
                    agent.respond_ui(str(event.args.get("request_id", "")), **answer)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    agent.start()
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        agent.cancel()
        thread.join(5)
        agent.close()
        pytest.fail(f"claude adapter did not finish within {timeout}s")
    agent.close()
    if errors:
        raise errors[0]
    return events


# ---------------------------------------------------------------------------
# Acceptance criterion 1: argv
# ---------------------------------------------------------------------------


def test_argv_carries_the_stream_json_and_permission_flags():
    agent = ClaudeAgent({}, model="sonnet", effort="high")
    argv = agent._argv(_request(), _context())

    assert argv[0] == "claude"
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--input-format") + 1] == "stream-json"
    assert "--include-partial-messages" in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--permission-prompts") + 1] == "host"
    assert argv[argv.index("--permission-prompt-tool") + 1] == "stdio"


def test_argv_never_skips_permissions():
    agent = ClaudeAgent({"model": "opus"}, effort="max", extra_args=["--add-dir", "/srv"])
    argv = agent._argv(_request(), _context())

    assert "--dangerously-skip-permissions" not in argv
    assert "--allow-dangerously-skip-permissions" not in argv
    assert "--permission-mode" not in argv
    # extra_args land verbatim, at the end.
    assert argv[-2:] == ["--add-dir", "/srv"]


def test_model_and_effort_come_from_config_when_not_passed():
    agent = ClaudeAgent({"model": "haiku", "effort": "low"})
    argv = agent._argv(_request(), _context())

    assert argv[argv.index("--model") + 1] == "haiku"
    assert argv[argv.index("--effort") + 1] == "low"


def test_request_target_overrides_the_configured_model_and_effort():
    agent = ClaudeAgent({"model": "haiku", "effort": "low"})
    request = _request(target=Target(backend="claude", model="opus", effort="xhigh"))
    argv = agent._argv(request, _context())

    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "xhigh"


def test_cold_adapter_claims_a_session_id_and_a_warm_one_resumes(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path), session_id="sess-1")
    cold = agent._argv(_request(), _context())
    assert cold[cold.index("--session-id") + 1] == "sess-1"
    assert "--resume" not in cold

    _drive(agent, {"confirmed": True})

    warm = agent._argv(_request(), _context())
    assert warm[warm.index("--resume") + 1] == "sess-1"
    assert "--session-id" not in warm


def test_the_spawned_child_really_gets_those_flags(tmp_path):
    env = _transcript_env(tmp_path)
    agent = ClaudeAgent({}, env=env, model="sonnet", effort="high", session_id="sess-2")
    _drive(agent, {"confirmed": True})

    spawned = _read_jsonl(tmp_path / "argv.jsonl")[0]
    assert "--include-partial-messages" in spawned
    assert spawned[spawned.index("--permission-prompt-tool") + 1] == "stdio"
    assert spawned[spawned.index("--session-id") + 1] == "sess-2"
    assert "--dangerously-skip-permissions" not in spawned


def test_approval_none_turns_prompts_off_instead_of_skipping_permissions():
    agent = ClaudeAgent({}, approval="none")
    argv = agent._argv(_request(), _context())

    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert "--permission-prompt-tool" not in argv
    assert "--dangerously-skip-permissions" not in argv
    assert agent.capabilities().approval == "none"


# ---------------------------------------------------------------------------
# Acceptance criterion 1: content-part mapping
# ---------------------------------------------------------------------------


def test_thinking_tool_use_and_tool_result_map_to_their_event_kinds(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path))
    events = _drive(agent, {"confirmed": True})
    kinds = [event.kind for event in events]

    assert EventKind.THINKING in kinds
    assert EventKind.TOOL_CALL in kinds
    assert EventKind.TOOL_RESULT in kinds
    assert kinds[-1] is EventKind.DONE

    thinking = "".join(e.text for e in events if e.kind is EventKind.THINKING)
    assert "no GPU device node" in thinking

    tool_calls = [e for e in events if e.kind is EventKind.TOOL_CALL]
    assert tool_calls[0].tool == "Bash"
    assert tool_calls[0].args["command"] == "cat /proc/driver/nvidia/version"

    results = [e for e in events if e.kind is EventKind.TOOL_RESULT]
    assert results[0].tool == "Bash"
    assert results[0].result == "NVRM version: 580.95.05"


def test_partial_deltas_are_not_replayed_from_the_aggregate_message(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path))
    events = _drive(agent, {"confirmed": True})

    text = "".join(e.text for e in events if e.kind is EventKind.TEXT_DELTA)
    assert text.count("nvidia-smi typically fails when no GPU device is exposed.") == 1
    assert text.count("The driver is loaded. Done.") == 1


def test_a_message_without_partial_deltas_still_yields_its_text_and_thinking():
    agent = ClaudeAgent({})
    events = agent._events(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "id": "msg_unstreamed",
                    "content": [
                        {"type": "thinking", "thinking": "weighing it up"},
                        {"type": "text", "text": "here is the answer"},
                    ],
                },
            }
        )
    )

    assert [(e.kind, e.text) for e in events] == [
        (EventKind.THINKING, "weighing it up"),
        (EventKind.TEXT_DELTA, "here is the answer"),
    ]


def test_a_failed_result_line_becomes_an_error_event():
    agent = ClaudeAgent({})
    events = agent._events(json.dumps({"type": "result", "subtype": "error", "error": "boom"}))

    assert [e.kind for e in events] == [EventKind.ERROR]
    assert events[0].error == "boom"


def test_non_json_and_uninteresting_lines_are_skipped():
    agent = ClaudeAgent({})
    assert agent._events("not json at all") == []
    assert agent._events("") == []
    assert agent._events(json.dumps({"type": "rate_limit_event"})) == []


# ---------------------------------------------------------------------------
# Acceptance criterion 2: the permission round-trip
# ---------------------------------------------------------------------------


def test_a_permission_request_becomes_a_proposal(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path))
    events = _drive(agent, {"confirmed": True})

    proposals = [e for e in events if e.kind is EventKind.PROPOSAL]
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.args["request_id"] == REQUEST_ID
    assert proposal.tool == "Bash"
    assert proposal.proposal is not None
    assert proposal.proposal.command == "cat /proc/driver/nvidia/version"
    assert proposal.proposal.rationale == "This command requires approval"
    assert proposal.proposal.kind is ProposalKind.FIX


def test_allowing_a_proposal_sends_an_allow_control_response(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path))
    events = _drive(agent, {"confirmed": True})

    sent = [m for m in _read_jsonl(tmp_path / "stdin.jsonl") if m.get("type") == "control_response"]
    assert len(sent) == 1
    response = sent[0]["response"]
    assert response["request_id"] == REQUEST_ID
    assert response["subtype"] == "success"
    assert response["response"]["behavior"] == "allow"
    # The tool input is echoed back verbatim as updatedInput -- nvsh never
    # rewrites what the model asked to run.
    assert response["response"]["updatedInput"]["command"] == "cat /proc/driver/nvidia/version"
    # The turn ran to completion, so the fake really unblocked on that answer.
    assert events[-1].kind is EventKind.DONE


def test_denying_a_proposal_sends_a_deny_control_response(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path))
    events = _drive(agent, {"cancelled": True})

    sent = [m for m in _read_jsonl(tmp_path / "stdin.jsonl") if m.get("type") == "control_response"]
    assert len(sent) == 1
    response = sent[0]["response"]["response"]
    assert response["behavior"] == "deny"
    assert "nvsh" in response["message"]
    assert "updatedInput" not in response
    assert events[-1].kind is EventKind.DONE


def test_an_unreadable_answer_denies_rather_than_allowing(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path))
    _drive(agent, {"value": "something else"})

    sent = [m for m in _read_jsonl(tmp_path / "stdin.jsonl") if m.get("type") == "control_response"]
    assert sent[0]["response"]["response"]["behavior"] == "deny"


def test_the_prompt_is_delivered_as_a_stream_json_user_message(tmp_path):
    agent = ClaudeAgent({}, env=_transcript_env(tmp_path))
    _drive(agent, {"confirmed": True})

    user_messages = [m for m in _read_jsonl(tmp_path / "stdin.jsonl") if m.get("type") == "user"]
    assert len(user_messages) == 1
    content = user_messages[0]["message"]["content"]
    assert "nvidia-smi" in content


def test_a_control_request_nvsh_cannot_answer_is_refused_not_ignored():
    agent = ClaudeAgent({})
    written: list[dict] = []
    agent._send = lambda message: written.append(message) or True  # type: ignore[method-assign]

    events = agent._events(
        json.dumps(
            {
                "type": "control_request",
                "request_id": "req-9",
                "request": {"subtype": "elicitation"},
            }
        )
    )

    assert events == []
    assert written[0]["response"]["subtype"] == "error"
    assert written[0]["response"]["request_id"] == "req-9"


# ---------------------------------------------------------------------------
# Acceptance criterion 2 + 3: capabilities, and no claude-code-acp anywhere
# ---------------------------------------------------------------------------


def test_capabilities_report_the_transport_and_the_approval_owner():
    caps = ClaudeAgent({}).capabilities()

    assert caps.path == "stream-json"
    assert caps.approval == "nvsh"
    assert caps.streaming is True
    assert caps.tool_calling is True
    assert caps.thinking is True
    assert caps.effort is True
    assert caps.persistent_session is True


def test_the_recorded_transcript_declares_where_it_came_from():
    text = TRANSCRIPT.read_text(encoding="utf-8")
    assert "# recorded-from: claude 2.1.270" in text
    payload = [json.loads(line) for line in text.splitlines() if line and not line.startswith("#")]
    kinds = {
        part.get("type")
        for obj in payload
        if obj.get("type") == "assistant"
        for part in obj["message"]["content"]
    }
    assert {"thinking", "tool_use"} <= kinds
    assert any(obj.get("type") == "control_request" for obj in payload)


def test_no_claude_code_acp_anywhere_in_the_nvsh_package_or_the_claude_fixtures():
    """Decisions c43/c54: nvsh drives the official CLI, not an ACP bridge.

    The needle is assembled at runtime so this test file is not itself a hit
    for a plain grep over the tree (the way the module docstring above, which
    has to name what nvsh does *not* use, would be).
    """
    needle = "claude-code" + "-acp"
    package = Path(__file__).parents[1] / "nvsh"
    paths = sorted(package.rglob("*.py")) + sorted(package.rglob("*.ts"))
    paths += [FAKES_DIR / "claude", TRANSCRIPT]
    for path in paths:
        assert needle not in path.read_text(encoding="utf-8"), path


# ---------------------------------------------------------------------------
# reliable-agent-stop (task t8): force_stop() kills a claude that ignores
# its own stream-json interrupt, plus every grandchild it started
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """True while *pid* is a live (non-zombie) process."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            stat = handle.read()
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    # The state field follows the parenthesised comm, which may contain spaces.
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def _wait_gone(pids: list[int], within: float = 3.0) -> list[int]:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _pid_alive(pid)]
        if not alive:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if _pid_alive(pid)]


def _grandchild_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(FAKES_DIR) + os.pathsep + env.get("PATH", "")
    env["NVSH_FAKE_IGNORE_CANCEL"] = "1"
    env["NVSH_FAKE_GRANDCHILD"] = "1"
    env["NVSH_FAKE_PID_FILE"] = str(tmp_path / "pids.json")
    return env


def _wait_for_pid_file(path: Path, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


def test_force_stop_leaves_no_pid_alive_when_claude_ignores_interrupt(tmp_path):
    """Acceptance criterion 1: a fake claude that ignores its stream-json
    interrupt and spawned a grandchild is fully gone -- harness and
    grandchild both -- within 3s of force_stop()."""
    env = _grandchild_env(tmp_path)
    agent = ClaudeAgent({}, env=env)
    agent.start()

    events: list = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for event in agent.run(_request(), _context()):
                events.append(event)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            errors.append(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    pids = _wait_for_pid_file(tmp_path / "pids.json")

    agent.force_stop()
    thread.join(timeout=10)
    assert not thread.is_alive(), "run() did not return after force_stop()"
    if errors:
        raise errors[0]

    assert _wait_gone([pids["harness"], pids["grandchild"]]) == []
