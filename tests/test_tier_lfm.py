"""Tests for nvsh.tiers.lfm: Tier 2's bounded inspect-interpret-propose loop.

Covers spec targets c15 and h14: at most four inspection rounds, read-only
operations only inside the loop, a fixed context budget, and exactly three
outcomes -- propose, explain, escalate -- with the escalated request carrying
what the tier already inspected.

The cut-off and propose cases run against a real ``http.server`` replaying
scripted OpenAI chat-completion JSON, so the production
:class:`~nvsh.tiers.toolchat.ToolChat` path is exercised and the tier is
pointed at it by an :class:`~nvsh.tiers.runtime.AttachedRuntime` alone --
switching runtime is a config change, not a code change. The remaining
behaviours use an in-memory fake chat.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.ops import table as ops_table
from nvsh.ops.render import render as render_argv
from nvsh.platform._model import Platform
from nvsh.tiers import lfm
from nvsh.tiers.base import Decline, DeclineReason, Explanation, TierDecision
from nvsh.tiers.lfm import (
    COMMAND_CHARS,
    ESCALATE_TOOL,
    EXPLAIN_TOOL,
    EXPLANATION_CHARS,
    LABEL_CHARS,
    MAX_CONTEXT_CHARS,
    MAX_ROUNDS,
    MUTATING_REFUSED,
    OUTPUT_TAIL_CHARS,
    PROPOSE_TOOL,
    REASON_CHARS,
    REDACT_SLACK,
    REQUEST_CHARS,
    RESULT_CHARS,
    LfmTier,
    tools_for,
)
from nvsh.tiers.runtime import AttachedRuntime, RuntimeUnavailable
from nvsh.tiers.toolchat import ChatReply, ToolCall, ToolChat, ToolChatError

MODEL = "test-model"
PLATFORM = Platform(kind="generic")

#: What ``memory_stats`` renders to on a generic machine (nvsh/ops/render.py).
MEMORY_ARGV = ["free", "-m"]


# ---------------------------------------------------------------------------
# fixture chat server
# ---------------------------------------------------------------------------


def _completion(spec: dict) -> dict:
    """One non-streamed OpenAI chat completion from a scripted *spec*."""
    message: dict = {"role": "assistant", "content": spec.get("text", "")}
    tool = spec.get("tool")
    if tool:
        message["tool_calls"] = [
            {
                "id": "call_fixture",
                "type": "function",
                "function": {
                    "name": tool,
                    "arguments": json.dumps(spec.get("arguments", {})),
                },
            }
        ]
    return {"choices": [{"message": message}]}


class _ChatHandler(BaseHTTPRequestHandler):
    """Replays ``self.server.script``, one scripted reply per POST."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        script = self.server.script  # type: ignore[attr-defined]
        spec = script.pop(0) if script else {"text": ""}
        payload = json.dumps(_completion(spec)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Silent: a test server must not write to the captured output."""


@pytest.fixture
def chat_server():
    """A scripted OpenAI-compatible server on an ephemeral localhost port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    server.daemon_threads = True
    server.script = []  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}"  # type: ignore[attr-defined]
    yield server
    server.shutdown()


def _served_tier(server, runner) -> LfmTier:
    """A tier pointed at the fixture server by configuration alone."""
    return LfmTier(
        AttachedRuntime(server.base_url),
        PLATFORM,
        model=MODEL,
        runner=runner,
        chat_factory=lambda base_url: ToolChat(base_url, MODEL, stream=False),
    )


# ---------------------------------------------------------------------------
# in-memory fakes
# ---------------------------------------------------------------------------


class _FakeChat:
    """Replays scripted :class:`ChatReply` objects and records what it was sent."""

    def __init__(self, replies: list[object]) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict]] = []
        self.stopped = 0
        self.base_url = ""

    def factory(self, base_url: str) -> "_FakeChat":
        """A ``chat_factory`` that returns this fake for whatever URL it is given."""
        self.base_url = base_url
        return self

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        self.seen.append(messages)
        if not self._replies:
            return ChatReply(text="", tool_calls=())
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, ChatReply)
        return reply

    def stop(self) -> None:
        self.stopped += 1


class _Runner:
    """A fake ops runner recording every argv it was handed."""

    def __init__(self, output: str = "ok", raises: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self._output = output
        self._raises = raises

    def __call__(self, argv: list[str], timeout: float) -> tuple[int, str]:
        self.calls.append(list(argv))
        self.timeouts.append(timeout)
        if self._raises is not None:
            raise self._raises
        return (0, self._output)


class _Runtime:
    """A fake :class:`~nvsh.tiers.runtime.Runtime` counting what was asked of it."""

    def __init__(self, error: str = "") -> None:
        self.error = error
        self.ensured = 0
        self.stopped = 0

    def ensure(self) -> str:
        self.ensured += 1
        if self.error:
            raise RuntimeUnavailable(self.error)
        return "http://127.0.0.1:65000"

    def stop(self) -> None:
        self.stopped += 1

    def status(self) -> str:
        return "fake runtime"


def _tier(chat: _FakeChat, *, runner=None, runtime=None, **kwargs) -> LfmTier:
    return LfmTier(
        runtime if runtime is not None else _Runtime(),
        PLATFORM,
        model=MODEL,
        runner=runner if runner is not None else _Runner(),
        chat_factory=chat.factory,
        **kwargs,
    )


def _tool_reply(name: str, **arguments: object) -> ChatReply:
    return ChatReply(text="", tool_calls=(ToolCall(name=name, arguments=dict(arguments)),))


def _text_reply(text: str) -> ChatReply:
    return ChatReply(text=text, tool_calls=())


def _ask() -> AgentRequest:
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt="why is this machine slow")


def _failure() -> AgentRequest:
    return AgentRequest(kind=RequestKind.FAILURE, command="docker ps", exit_code=1)


def _ctx(output: str = "") -> AgentContext:
    return AgentContext(platform="generic", output=output)


def _last_tool_content(chat: _FakeChat) -> str:
    """The content of the last ``tool`` message the model was shown."""
    tools = [message for message in chat.seen[-1] if message["role"] == "tool"]
    return str(tools[-1]["content"])


# ---------------------------------------------------------------------------
# criterion 1: cut off at four rounds, through the real chat client
# ---------------------------------------------------------------------------


def _forever_inspecting(server, runner) -> object:
    """Script far more inspections than the loop may run, then select once."""
    server.script.extend([{"tool": "memory_stats", "arguments": {}}] * 12)
    tier = _served_tier(server, runner)
    result = tier.select(_ask(), _ctx())
    tier.close()
    return result


def test_a_model_that_never_stops_inspecting_is_cut_off(chat_server) -> None:
    result = _forever_inspecting(chat_server, _Runner())
    assert isinstance(result, Decline)


def test_the_cut_off_loop_escalates(chat_server) -> None:
    result = _forever_inspecting(chat_server, _Runner())
    assert result.reason is DeclineReason.ESCALATED


def test_the_cut_off_loop_runs_exactly_max_rounds_operations(chat_server) -> None:
    runner = _Runner()
    _forever_inspecting(chat_server, runner)
    assert runner.calls == [MEMORY_ARGV] * MAX_ROUNDS


def test_the_cut_off_decline_carries_every_inspection(chat_server) -> None:
    result = _forever_inspecting(chat_server, _Runner())
    assert len(result.inspections) == MAX_ROUNDS


def test_propose_through_the_fixture_server_returns_a_decision(chat_server) -> None:
    chat_server.script.append(
        {
            "tool": PROPOSE_TOOL,
            "arguments": {"operation": "service_restart", "arguments": {"service": "nginx"}},
        }
    )
    tier = _served_tier(chat_server, _Runner())
    result = tier.select(_ask(), _ctx())
    tier.close()
    assert result == TierDecision(
        operation="service_restart", args={"service": "nginx"}, confidence=None, read_only=False
    )


def test_a_proposed_mutating_operation_is_never_run(chat_server) -> None:
    chat_server.script.append(
        {
            "tool": PROPOSE_TOOL,
            "arguments": {"operation": "service_restart", "arguments": {"service": "nginx"}},
        }
    )
    runner = _Runner()
    tier = _served_tier(chat_server, runner)
    tier.select(_ask(), _ctx())
    tier.close()
    assert runner.calls == []


# ---------------------------------------------------------------------------
# criterion 2: mutating refused, read-only run bounded and redacted
# ---------------------------------------------------------------------------


def _mutating_then_explains() -> tuple[_FakeChat, _Runner]:
    chat = _FakeChat(
        [_tool_reply("service_restart", service="nginx"), _tool_reply(EXPLAIN_TOOL, text="done")]
    )
    return chat, _Runner()


def test_a_mutating_operation_in_the_loop_never_reaches_the_runner() -> None:
    chat, runner = _mutating_then_explains()
    _tier(chat, runner=runner).select(_ask(), _ctx())
    assert runner.calls == []


def test_a_mutating_operation_in_the_loop_is_fed_back_as_an_error() -> None:
    chat, runner = _mutating_then_explains()
    _tier(chat, runner=runner).select(_ask(), _ctx())
    assert _last_tool_content(chat) == f"error: {MUTATING_REFUSED}"


def test_a_refused_call_still_costs_a_round() -> None:
    chat = _FakeChat([_tool_reply("service_restart", service="nginx")] * 12)
    _tier(chat).select(_ask(), _ctx())
    assert len(chat.seen) == MAX_ROUNDS


def test_a_loop_of_refused_calls_ends_in_a_decline() -> None:
    chat = _FakeChat([_tool_reply("service_restart", service="nginx")] * 12)
    assert isinstance(_tier(chat).select(_ask(), _ctx()), Decline)


def test_a_read_only_operation_runs_the_rendered_argv() -> None:
    chat = _FakeChat([_tool_reply("memory_stats"), _tool_reply(EXPLAIN_TOOL, text="fine")])
    runner = _Runner()
    _tier(chat, runner=runner).select(_ask(), _ctx())
    assert runner.calls == [MEMORY_ARGV]


def test_an_inspection_runs_with_the_configured_timeout() -> None:
    chat = _FakeChat([_tool_reply("memory_stats"), _tool_reply(EXPLAIN_TOOL, text="fine")])
    runner = _Runner()
    _tier(chat, runner=runner, op_timeout=2.5).select(_ask(), _ctx())
    assert runner.timeouts == [pytest.approx(2.5)]


def test_an_inspection_result_is_redacted_before_it_re_enters_the_context() -> None:
    secret = "hf_" + "b" * 34  # assembled at runtime: never a literal in the tree
    chat = _FakeChat([_tool_reply("memory_stats"), _tool_reply(EXPLAIN_TOOL, text="fine")])
    _tier(chat, runner=_Runner(output=f"HF_TOKEN={secret}")).select(_ask(), _ctx())
    assert secret not in _last_tool_content(chat)


def test_an_inspection_result_is_bounded() -> None:
    chat = _FakeChat([_tool_reply("memory_stats"), _tool_reply(EXPLAIN_TOOL, text="fine")])
    _tier(chat, runner=_Runner(output="y" * 40000), result_chars=200).select(_ask(), _ctx())
    assert len(_last_tool_content(chat)) == 200


def test_an_ungroundable_argument_is_fed_back_as_one_line() -> None:
    chat = _FakeChat([_tool_reply("service_status", service="ghost"), _text_reply("giving up")])
    _tier(chat, runner=_Runner(output="other.service loaded active\n")).select(_ask(), _ctx())
    assert _last_tool_content(chat) == "error: no such service: ghost"


# ---------------------------------------------------------------------------
# criterion 3: exactly three outcomes, and the escalation carries the work
# ---------------------------------------------------------------------------

_BEHAVIOURS = (
    "garbage_json",
    "unknown_tool",
    "empty_reply",
    "chat_error",
    "runner_timeout",
    "escalate",
    "explain",
    "propose",
    "mutating_bare",
    "missing_argument",
    "raw_shell",
)


def _behaviour(name: str) -> tuple[list[object], _Runner]:
    """The scripted replies and runner for one fixture model behaviour."""
    if name == "runner_timeout":
        return ([_tool_reply("memory_stats")] * 12, _Runner(raises=TimeoutError("timed out")))
    replies: dict[str, list[object]] = {
        "garbage_json": [_text_reply('{"name": "memory_stats", "arg')],
        "unknown_tool": [_tool_reply("frobnicate", level="9")] * 12,
        "empty_reply": [_text_reply("")],
        "chat_error": [ToolChatError("no server there")],
        "escalate": [_tool_reply(ESCALATE_TOOL, reason="this needs a bigger agent")],
        "explain": [_tool_reply(EXPLAIN_TOOL, text="the disk is full")],
        "propose": [_tool_reply(PROPOSE_TOOL, operation="machine_status", arguments={})],
        "mutating_bare": [_tool_reply("container_restart", container="vllm")] * 12,
        "missing_argument": [_tool_reply("service_status")] * 12,
        "raw_shell": [_tool_reply("bash", command="rm -rf /tmp")] * 12,
    }
    return (replies[name], _Runner())


def _mutating_argvs() -> list[list[str]]:
    """Every argv a mutating operation in the table renders to on this platform."""
    rendered = []
    for operation in ops_table.OPERATIONS:
        if operation.read_only:
            continue
        args = {
            spec.name: (spec.choices[0] if spec.kind == "choice" else "vllm")
            for spec in operation.args
        }
        argv = render_argv(operation.name, args, PLATFORM)
        if argv is not None:
            rendered.append(argv)
    return rendered


@pytest.mark.parametrize("name", _BEHAVIOURS)
def test_select_returns_only_the_three_outcome_types(name: str) -> None:
    replies, runner = _behaviour(name)
    result = _tier(_FakeChat(replies), runner=runner).select(_ask(), _ctx())
    assert isinstance(result, (TierDecision, Decline, Explanation))


@pytest.mark.parametrize("name", _BEHAVIOURS)
def test_the_runner_is_never_given_a_mutating_operations_argv(name: str) -> None:
    replies, runner = _behaviour(name)
    _tier(_FakeChat(replies), runner=runner).select(_ask(), _ctx())
    forbidden = _mutating_argvs()
    assert [argv for argv in runner.calls if argv in forbidden] == []


def test_an_escalation_carries_the_inspection_results() -> None:
    chat = _FakeChat([_tool_reply("memory_stats"), _tool_reply(ESCALATE_TOOL, reason="not sure")])
    result = _tier(chat, runner=_Runner(output="MemAvailable 100")).select(_ask(), _ctx())
    assert result.inspections == (("memory_stats", "exit 0\nMemAvailable 100"),)


def test_an_escalation_keeps_the_models_own_reason() -> None:
    chat = _FakeChat([_tool_reply(ESCALATE_TOOL, reason="not sure")])
    result = _tier(chat).select(_ask(), _ctx())
    assert result == Decline(reason=DeclineReason.ESCALATED, detail="not sure")


def test_an_explanation_carries_the_inspection_results() -> None:
    chat = _FakeChat([_tool_reply("memory_stats"), _tool_reply(EXPLAIN_TOOL, text="plenty free")])
    result = _tier(chat, runner=_Runner(output="MemAvailable 100")).select(_ask(), _ctx())
    assert result == Explanation(
        text="plenty free", inspections=(("memory_stats", "exit 0\nMemAvailable 100"),)
    )


def test_plain_words_with_no_tool_call_are_an_explanation() -> None:
    result = _tier(_FakeChat([_text_reply("nothing is wrong here")])).select(_ask(), _ctx())
    assert result == Explanation(text="nothing is wrong here")


def test_an_explanation_is_bounded() -> None:
    result = _tier(_FakeChat([_text_reply("w" * 9000)])).select(_ask(), _ctx())
    assert len(result.text) == 2000


def test_a_silent_model_escalates() -> None:
    result = _tier(_FakeChat([_text_reply("")])).select(_ask(), _ctx())
    assert result.reason is DeclineReason.ESCALATED


# ---------------------------------------------------------------------------
# the fixed context budget
# ---------------------------------------------------------------------------


def _four_maximum_rounds(**kwargs) -> _FakeChat:
    """Four rounds whose every text input is far larger than its own bound."""
    chat = _FakeChat([_tool_reply("memory_stats")] * 12)
    tier = _tier(chat, runner=_Runner(output="y" * 40000), **kwargs)
    tier.select(_failure(), _ctx(output="z" * 40000))
    return chat


def test_the_context_never_exceeds_the_budget_after_four_rounds() -> None:
    chat = _four_maximum_rounds()
    biggest = max(len(json.dumps(messages)) for messages in chat.seen)
    assert biggest <= MAX_CONTEXT_CHARS


def test_older_results_are_replaced_by_a_one_line_stub() -> None:
    # keep_results=1 so the stub is reachable inside the four-round budget:
    # with the default 3, a loop that stops at 4 rounds never drops one.
    chat = _four_maximum_rounds(keep_results=1)
    stubs = [
        message
        for message in chat.seen[-1]
        if message["role"] == "tool" and message["content"] == "(memory_stats: result dropped)"
    ]
    assert len(stubs) == 2


def test_a_failure_reaches_the_model_as_command_and_exit_status() -> None:
    chat = _FakeChat([_text_reply("the daemon is down")])
    _tier(chat).select(_failure(), _ctx(output="cannot connect"))
    assert chat.seen[0][1]["content"] == (
        "command: docker ps\nexit status: 1\noutput tail:\ncannot connect"
    )


def test_the_system_brief_names_the_platform_kind() -> None:
    chat = _FakeChat([_text_reply("fine")])
    _tier(chat).select(_ask(), _ctx())
    assert "generic machine" in chat.seen[0][0]["content"]


# ---------------------------------------------------------------------------
# every text reaches the redactor already bounded
# ---------------------------------------------------------------------------

#: Far larger than any bound, and unbroken: the shape that makes an unbounded
#: redact() pass cost minutes rather than milliseconds.
HUGE = "A" * 2_000_000


class _RedactSpy:
    """Wraps the real redactor and remembers the largest input it was given."""

    def __init__(self, real) -> None:
        self._real = real
        self.largest = 0

    def __call__(self, payload: bytes) -> bytes:
        self.largest = max(self.largest, len(payload))
        return self._real(payload)


def _redact_spy(monkeypatch) -> _RedactSpy:
    spy = _RedactSpy(lfm.redact)
    monkeypatch.setattr(lfm, "redact", spy)
    return spy


def test_a_huge_explain_text_is_bounded_before_it_is_redacted(monkeypatch) -> None:
    spy = _redact_spy(monkeypatch)
    _tier(_FakeChat([_tool_reply(EXPLAIN_TOOL, text=HUGE)])).select(_ask(), _ctx())
    assert spy.largest <= EXPLANATION_CHARS + REDACT_SLACK


def test_a_huge_plain_reply_is_bounded_before_it_is_redacted(monkeypatch) -> None:
    spy = _redact_spy(monkeypatch)
    _tier(_FakeChat([_text_reply(HUGE)])).select(_ask(), _ctx())
    assert spy.largest <= EXPLANATION_CHARS + REDACT_SLACK


def test_a_huge_escalate_reason_is_bounded_before_it_is_redacted(monkeypatch) -> None:
    spy = _redact_spy(monkeypatch)
    _tier(_FakeChat([_tool_reply(ESCALATE_TOOL, reason=HUGE)])).select(_ask(), _ctx())
    assert spy.largest <= REASON_CHARS + REDACT_SLACK


def test_a_huge_inspection_result_is_bounded_before_it_is_redacted(monkeypatch) -> None:
    spy = _redact_spy(monkeypatch)
    chat = _FakeChat([_tool_reply("memory_stats"), _tool_reply(EXPLAIN_TOOL, text="fine")])
    _tier(chat, runner=_Runner(output=HUGE)).select(_ask(), _ctx())
    assert spy.largest <= RESULT_CHARS + REDACT_SLACK


def test_a_huge_failure_output_is_bounded_before_it_is_redacted(monkeypatch) -> None:
    spy = _redact_spy(monkeypatch)
    _tier(_FakeChat([_text_reply("fine")])).select(_failure(), _ctx(output=HUGE))
    assert spy.largest <= OUTPUT_TAIL_CHARS + REDACT_SLACK


def test_a_huge_request_is_bounded_before_it_is_redacted(monkeypatch) -> None:
    spy = _redact_spy(monkeypatch)
    request = AgentRequest(kind=RequestKind.EXPLICIT, prompt=HUGE)
    _tier(_FakeChat([_text_reply("fine")])).select(request, _ctx())
    assert spy.largest <= REQUEST_CHARS + REDACT_SLACK


def test_a_huge_failed_command_is_bounded_before_it_is_redacted(monkeypatch) -> None:
    spy = _redact_spy(monkeypatch)
    request = AgentRequest(kind=RequestKind.FAILURE, command=HUGE, exit_code=1)
    _tier(_FakeChat([_text_reply("fine")])).select(request, _ctx())
    assert spy.largest <= COMMAND_CHARS + REDACT_SLACK


def test_a_secret_across_the_clamp_boundary_does_not_survive() -> None:
    secret = "hf_" + "c" * 34  # assembled at runtime: never a literal in the tree
    # The assignment starts just before the clamp cut and ends past it, so the
    # secret straddles the boundary. REDACT_SLACK is what keeps the whole
    # secret inside the window the redactor sees; the clamp can then only cut
    # into the marker that replaced it.
    text = "w" * (EXPLANATION_CHARS - 10) + f"HF_TOKEN={secret}" + "w" * 1000
    result = _tier(_FakeChat([_tool_reply(EXPLAIN_TOOL, text=text)])).select(_ask(), _ctx())
    assert secret[:12] not in result.text


def test_a_failure_output_keeps_its_tail() -> None:
    chat = _FakeChat([_text_reply("fine")])
    _tier(chat).select(_failure(), _ctx(output="q" * 40000 + "the real error"))
    assert chat.seen[0][1]["content"].endswith("the real error")


# ---------------------------------------------------------------------------
# a private-key block split by the bound never reaches the redactor in halves
# ---------------------------------------------------------------------------

#: The PEM markers, assembled at runtime from pieces: a literal one would be a
#: secret-shaped string in the tree and ``scripts/scan-secrets.py`` would
#: (rightly) fail on it.
_EDGES = "-" * 5
_KEY_WORDS = "RSA PRIVATE KEY"
#: Long enough that the block's far marker falls outside any bound + slack.
_KEY_BODY = "z" * 6000
#: A distinctive stand-in for key material, so a test can assert it is gone.
_KEY_HEAD = "HEADKEYMATERIAL"
_KEY_TAIL = "TAILKEYMATERIAL"


def _pem_marker(edge: str) -> str:
    return f"{_EDGES}{edge} {_KEY_WORDS}{_EDGES}"


def _split_at_head() -> str:
    """A block that opens inside the explanation window and closes far past it."""
    lead = "w" * (EXPLANATION_CHARS - 60)
    return f"{lead}{_pem_marker('BEGIN')}\n{_KEY_HEAD}{_KEY_BODY}\n{_pem_marker('END')}"


def _split_at_tail() -> str:
    """A block that closes inside the output tail window and opens far before it."""
    return (
        f"{_pem_marker('BEGIN')}\n{_KEY_BODY}{_KEY_TAIL}\n" f"{_pem_marker('END')}\nthe real error"
    )


def test_a_key_block_opened_before_the_head_cut_leaves_no_key_material() -> None:
    result = _tier(_FakeChat([_tool_reply(EXPLAIN_TOOL, text=_split_at_head())])).select(
        _ask(), _ctx()
    )
    assert _KEY_HEAD not in result.text


def test_a_key_block_closed_after_the_tail_cut_leaves_no_key_material() -> None:
    chat = _FakeChat([_text_reply("fine")])
    _tier(chat).select(_failure(), _ctx(output=_split_at_tail()))
    assert _KEY_TAIL not in chat.seen[0][1]["content"]


def test_a_whole_key_block_inside_the_window_is_still_redacted() -> None:
    whole = f"{_pem_marker('BEGIN')}\n{_KEY_HEAD}\n{_pem_marker('END')}"
    result = _tier(_FakeChat([_tool_reply(EXPLAIN_TOOL, text=whole)])).select(_ask(), _ctx())
    assert result.text == "<REDACTED:private_key_block>"


# ---------------------------------------------------------------------------
# the echoed tool arguments stay valid JSON
# ---------------------------------------------------------------------------


def _echoed_arguments(messages: list[dict]) -> list[str]:
    """Every ``function.arguments`` string in the assistant turns of *messages*."""
    return [
        call["function"]["arguments"]
        for message in messages
        if message["role"] == "assistant"
        for call in message.get("tool_calls", ())
    ]


def _parses_as_json(raw: str) -> bool:
    try:
        json.loads(raw)
    except ValueError:
        return False
    return True


def _huge_argument_echo() -> list[str]:
    """What the model is shown back after it called a tool with a huge argument."""
    chat = _FakeChat([_tool_reply("memory_stats", note="v" * 50_000), _text_reply("fine")])
    _tier(chat).select(_ask(), _ctx())
    return _echoed_arguments(chat.seen[-1])


def test_a_huge_tool_argument_is_echoed_back_as_valid_json() -> None:
    echoed = _huge_argument_echo()
    assert all(_parses_as_json(arguments) for arguments in echoed)


def test_a_huge_tool_argument_echo_stays_within_the_label_bound() -> None:
    echoed = _huge_argument_echo()
    assert max(len(arguments) for arguments in echoed) <= LABEL_CHARS


# ---------------------------------------------------------------------------
# tools, runtime and lifecycle
# ---------------------------------------------------------------------------


def test_every_operation_in_the_table_is_offered_as_a_tool() -> None:
    offered = {tool["function"]["name"] for tool in tools_for()}
    assert offered == set(ops_table.names()) | {PROPOSE_TOOL, EXPLAIN_TOOL, ESCALATE_TOOL}


def test_the_constructor_asks_the_runtime_for_nothing() -> None:
    runtime = _Runtime()
    _tier(_FakeChat([]), runtime=runtime)
    assert runtime.ensured == 0


def test_the_runtime_is_asked_once_across_two_selects() -> None:
    runtime = _Runtime()
    tier = _tier(_FakeChat([_text_reply("a"), _text_reply("b")]), runtime=runtime)
    tier.select(_ask(), _ctx())
    tier.select(_ask(), _ctx())
    assert runtime.ensured == 1


def test_an_unavailable_runtime_declines_with_its_own_line() -> None:
    tier = _tier(_FakeChat([]), runtime=_Runtime(error="docker is not installed"))
    assert tier.select(_ask(), _ctx()) == Decline(
        reason=DeclineReason.TIER_UNAVAILABLE, detail="docker is not installed"
    )


def test_close_stops_the_chat() -> None:
    chat = _FakeChat([_text_reply("hi")])
    tier = _tier(chat)
    tier.select(_ask(), _ctx())
    tier.close()
    assert chat.stopped == 1


def test_close_releases_the_runtime() -> None:
    runtime = _Runtime()
    tier = _tier(_FakeChat([]), runtime=runtime)
    tier.close()
    assert runtime.stopped == 1


def test_status_is_one_line() -> None:
    assert _tier(_FakeChat([])).status() == f"lfm tier ({MODEL}): fake runtime"
