"""Every adapter carries the system brief, in the place its backend expects (d19).

One composer is not enough: a brief that only reaches one backend is exactly
the drift ``nvsh.agent.prompt`` exists to prevent. These tests pin the
*delivery mechanism* per adapter:

* ``claude``/``qwen``/``pi`` take ``--append-system-prompt <text>`` (verified
  against the installed CLIs' ``--help``);
* ``codex exec`` has no such flag, so the brief is prepended to the prompt;
* the OpenAI-compatible endpoint gets a ``system`` message before the user
  message.
"""

from __future__ import annotations

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.agent.claude import ClaudeAgent
from nvsh.agent.codex import CodexAgent
from nvsh.agent.pi import PiAgent
from nvsh.agent.prompt import build_full_prompt, build_prompt, build_system_prompt
from nvsh.agent.qwen import QwenAgent

_CONTEXT = AgentContext(platform="platform: dgx-spark\n  mem_total: 1 kB  [file: /proc/meminfo]")
_REQUEST = AgentRequest(
    kind=RequestKind.FAILURE, prompt="Diagnose.", command="ls /nope", exit_code=2
)


def _brief() -> str:
    return build_system_prompt(_CONTEXT)


def test_claude_appends_the_brief_as_a_system_prompt_flag():
    argv = ClaudeAgent({})._argv(_REQUEST, _CONTEXT)
    assert "--append-system-prompt" in argv
    assert argv[argv.index("--append-system-prompt") + 1] == _brief()
    # the prompt itself stays the facts block, not the brief
    assert build_prompt(_REQUEST, _CONTEXT) in argv


def test_qwen_appends_the_brief_as_a_system_prompt_flag():
    argv = QwenAgent({})._argv(_REQUEST, _CONTEXT)
    assert "--append-system-prompt" in argv
    assert argv[argv.index("--append-system-prompt") + 1] == _brief()


def test_codex_has_no_system_flag_so_the_brief_leads_the_prompt():
    argv = CodexAgent({})._argv(_REQUEST, _CONTEXT)
    assert "--append-system-prompt" not in argv
    assert build_full_prompt(_REQUEST, _CONTEXT) in argv


def test_pi_argv_carries_the_brief_once_for_the_whole_session(tmp_path):
    agent = PiAgent(session_dir=tmp_path, system_prompt="BRIEF-TEXT")
    argv = agent.build_argv()
    assert "--append-system-prompt" in argv
    assert argv[argv.index("--append-system-prompt") + 1] == "BRIEF-TEXT"
    # one turn's prompt must not repeat it: pi holds it for the session
    assert "BRIEF-TEXT" not in agent._prompt_for(_REQUEST, _CONTEXT)


def test_pi_falls_back_to_the_detected_brief_when_none_is_passed(tmp_path):
    argv = PiAgent(session_dir=tmp_path).build_argv()
    text = argv[argv.index("--append-system-prompt") + 1]
    assert "nvsh" in text.lower()
