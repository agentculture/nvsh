"""Tests for evals.tool_jev.track_a_loop (issue #64, deviation d1).

The reference models' Track A runs through the candidates' own
``nvsh.tiers.lfm.LfmTier`` loop with a deferred chat client. Covered here:

- a case that needs two rounds finishes with the round-2 proposal, after
  its round-1 inspection ran against the ground snapshot;
- two cases needing two rounds each cost exactly two batch submissions
  (one per round), and a crash after round 1 resumes at round 2 without
  re-calling round 1;
- driving LfmTier with a plain scripted chat and driving it through
  ``DeferredChat`` + ledger send the same messages each round and reach
  the same decision (equivalence);
- ledger keys carry the round number; the offered set narrows the tools;
  every loop ending (explain, escalate, malformed, out of rounds) maps the
  way measure.py maps a candidate's.

No network: the provider is the in-process FakeProvider. Case text and the
snapshot are synthetic.
"""

from __future__ import annotations

import json

import pytest

from evals.tool_jev import track_a_loop as loop
from evals.tool_jev.cases import Case
from evals.tool_jev.ledger import DONE, INVALID, Ledger
from evals.tool_jev.providers import fake
from evals.tool_jev.providers.base import CallRequest
from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.platform._model import Platform
from nvsh.tiers import lfm
from nvsh.tiers.base import TierDecision
from nvsh.tiers.toolchat import ChatReply, ToolCall

PLATFORM = Platform(kind="jetson")
SNAPSHOT = {
    "services": ["synthetic.service"],
    "containers": [],
    "source": "synthetic test snapshot",
    "created": "2026-09-26",
}
OFFERED = ("service_status", "service_logs", "service_restart")
SERVICE = {"service": "synthetic.service"}


def _call(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True)


STATUS = _call("service_status", SERVICE)
LOGS = _call("service_logs", SERVICE)
RESTART = _call("propose", {"operation": "service_restart", "arguments": SERVICE})


def _case(case_id="loop-1", candidates=OFFERED, text="please restart the synthetic service"):
    return Case(
        id=case_id,
        split="test",
        text=text,
        candidates=candidates,
        expect={"operation": "service_restart", "args": SERVICE},
        read_only=False,
    )


def _round(cases, provider, ledger):
    return loop.run_round(
        cases,
        provider=provider,
        model="fake-model",
        ledger=ledger,
        snapshot=SNAPSHOT,
        platform=PLATFORM,
    )


def _sync(provider, ledger, round_result):
    results = [provider.submit_sync(call.request) for call in round_result.pending]
    assert loop.record_results(ledger, round_result.pending, results) == []


def _batch(provider, ledger, round_result):
    keys = [call.key for call in round_result.pending]
    ref = ledger.begin_submit(keys)
    handle = provider.submit_batch([call.request for call in round_result.pending], ref)
    ledger.mark_submitted(keys, handle.batch_id)
    left = loop.record_results(ledger, round_result.pending, provider.fetch_batch(handle))
    assert left == []


# ---------------------------------------------------------------------------
# one case, two rounds
# ---------------------------------------------------------------------------


def test_two_round_case_inspects_against_the_snapshot_then_proposes(tmp_path):
    provider = fake.FakeProvider(script=[("answer", STATUS), ("answer", RESTART)])
    with Ledger(tmp_path) as ledger:
        first = _round([_case()], provider, ledger)
        assert first.finished == {}
        (call,) = first.pending
        assert call.round == 1 and call.spec.target == "tool_call:r1"
        assert call.request.history == ()
        assert call.request.prompt == lfm.system_brief(PLATFORM)
        _sync(provider, ledger, first)

        second = _round([_case()], provider, ledger)
        (call2,) = second.pending
        assert call2.spec.target == "tool_call:r2"
        assistant, tool = call2.request.history
        assert assistant["tool_calls"][0]["name"] == "service_status"
        assert json.loads(assistant["tool_calls"][0]["arguments"]) == SERVICE
        # The snapshot runner answers lookups only: the inspection ran and
        # came back as "not found" -- exactly what a candidate saw.
        assert tool == {"role": "tool", "tool_call_id": "call_0", "content": "exit 127\n"}
        _sync(provider, ledger, second)

        done = _round([_case()], provider, ledger)
        assert done.pending == []
        record = done.finished["loop-1"]
    assert (record.outcome, record.operation, record.arguments) == (
        "propose",
        "service_restart",
        SERVICE,
    )
    assert record.interface == "tool_call"
    assert record.candidates is None
    assert record.provider == "fake" and record.model == "fake-model"
    assert record.returned_model == "fake-model"
    assert len(provider.received) == 2


def test_ledger_keys_carry_the_round_and_the_exact_content(tmp_path):
    provider = fake.FakeProvider(script=[("answer", STATUS), ("answer", RESTART)])
    with Ledger(tmp_path) as ledger:
        first = _round([_case()], provider, ledger)
        _sync(provider, ledger, first)
        second = _round([_case()], provider, ledger)
        a, b = first.pending[0].spec, second.pending[0].spec
        assert (a.case_id, b.case_id) == ("loop-1", "loop-1")
        assert a.prompt_hash != b.prompt_hash
        assert (
            a.params
            == b.params
            == {
                "reasoning": "medium",
                "max_output_tokens": 512,
                "tool_choice": "auto",
            }
        )
        assert a.subject_role == "reference"
        # Replaying the same round rebuilds the same key.
        assert _round([_case()], provider, ledger).pending[0].key == second.pending[0].key


def test_offered_candidates_narrow_the_tools_like_measure(tmp_path):
    provider = fake.FakeProvider(script=[])
    with Ledger(tmp_path) as ledger:
        (call,) = _round([_case()], provider, ledger).pending
    names = [tool["function"]["name"] for tool in call.request.params["tools"]]
    assert names == list(OFFERED) + [lfm.PROPOSE_TOOL, lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL]
    propose = call.request.params["tools"][-3]["function"]["parameters"]
    assert propose["properties"]["operation"]["enum"] == list(OFFERED)
    assert call.request.offered_candidates == ()  # the loop judges finality, not the adapter


# ---------------------------------------------------------------------------
# batch per round, crash between rounds
# ---------------------------------------------------------------------------


def test_two_cases_two_rounds_cost_exactly_two_batch_submissions(tmp_path):
    cases = [_case("loop-a"), _case("loop-b")]
    provider = fake.FakeProvider(
        script=[("answer", STATUS), ("answer", LOGS), ("answer", RESTART), ("answer", RESTART)]
    )
    with Ledger(tmp_path) as ledger:
        while True:
            current = _round(cases, provider, ledger)
            if not current.pending:
                break
            assert {call.round for call in current.pending} == {len(provider.submitted_refs) + 1}
            _batch(provider, ledger, current)
    assert len(provider.submitted_refs) == 2
    assert {record.operation for record in current.finished.values()} == {"service_restart"}


def test_a_crash_after_round_one_resumes_at_round_two_without_recalling_it(tmp_path):
    cases = [_case("loop-a"), _case("loop-b")]
    provider = fake.FakeProvider(script=[("answer", STATUS), ("answer", STATUS)])
    with Ledger(tmp_path) as ledger:
        _batch(provider, ledger, _round(cases, provider, ledger))
    # power-off here; a fresh process reopens the run directory
    provider.queue("answer", RESTART)
    provider.queue("answer", RESTART)
    with Ledger(tmp_path) as ledger:
        resumed = _round(cases, provider, ledger)
        assert [call.round for call in resumed.pending] == [2, 2]
        assert len(provider.received) == 2  # round 1 was not sent again
        _batch(provider, ledger, resumed)
        final = _round(cases, provider, ledger)
    assert final.pending == [] and len(final.finished) == 2
    assert len(provider.submitted_refs) == 2
    assert len(provider.received) == 4


# ---------------------------------------------------------------------------
# equivalence with a plain scripted chat
# ---------------------------------------------------------------------------


class ScriptedChat:
    """A plain ToolChat stand-in: canned replies, every request recorded."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def complete(self, messages, tools):
        self.sent.append((json.loads(json.dumps(messages)), tools))
        return self.replies.pop(0)

    def stop(self):
        """Nothing in flight."""


def _reply(answer: str) -> ChatReply:
    call = json.loads(answer)
    return ChatReply(text="", tool_calls=(ToolCall(call["name"], call["arguments"]),))


def test_deferred_chat_sends_what_a_plain_chat_sends_and_decides_the_same(tmp_path):
    script = [STATUS, LOGS, RESTART]
    direct_chat = ScriptedChat([_reply(answer) for answer in script])
    tier = lfm.LfmTier(
        loop._NoRuntime(),
        PLATFORM,
        model="fake-model",
        runner=loop.measure.snapshot_runner(SNAPSHOT),
        chat_factory=lambda _url: direct_chat,
    )
    case = _case()
    request = AgentRequest(kind=RequestKind.EXPLICIT, prompt=case.text)
    direct = tier.select(request, AgentContext(output=""))
    assert isinstance(direct, TierDecision)

    provider = fake.FakeProvider(script=[("answer", answer) for answer in script])
    sent: list[CallRequest] = []
    with Ledger(tmp_path) as ledger:
        while True:
            current = _round([case], provider, ledger)
            if not current.pending:
                break
            sent.extend(call.request for call in current.pending)
            _sync(provider, ledger, current)
    record = current.finished[case.id]

    assert len(sent) == len(direct_chat.sent) == 3
    for request_sent, (messages, tools) in zip(sent, direct_chat.sent):
        assert loop.replay_messages(request_sent) == messages
        assert request_sent.params["tools"] == loop.measure.restrict_tools(tools, OFFERED)
    assert (record.operation, record.arguments) == (direct.operation, dict(direct.args))


# ---------------------------------------------------------------------------
# every loop ending maps the way measure.py maps a candidate's
# ---------------------------------------------------------------------------


def _finish(tmp_path, answers):
    provider = fake.FakeProvider(script=answers)
    rounds = 0
    with Ledger(tmp_path) as ledger:
        while True:
            current = _round([_case()], provider, ledger)
            if not current.pending:
                return current.finished["loop-1"], rounds, ledger.entries()
            rounds += 1
            _sync(provider, ledger, current)


def test_explain_ends_as_explain(tmp_path):
    record, rounds, _ = _finish(tmp_path, [("answer", _call("explain", {"text": "it is fine"}))])
    assert (record.outcome, record.operation, rounds) == ("explain", None, 1)


def test_escalate_ends_as_escalate(tmp_path):
    record, _, _ = _finish(tmp_path, [("answer", _call("escalate", {"reason": "too big"}))])
    assert record.outcome == "escalate" and record.invalid_reason is None


def test_a_malformed_answer_is_invalid_with_the_adapter_reason_and_cached(tmp_path):
    record, rounds, entries = _finish(tmp_path, [("malformed", None)])
    assert (record.outcome, record.invalid_reason, rounds) == ("invalid", "malformed", 1)
    assert [entry.state for entry in entries] == [INVALID]


def test_running_out_of_rounds_is_invalid_after_exactly_max_rounds_calls(tmp_path):
    record, rounds, entries = _finish(tmp_path, [("answer", STATUS)] * lfm.MAX_ROUNDS)
    assert rounds == lfm.MAX_ROUNDS
    assert (record.outcome, record.invalid_reason) == ("invalid", "no_decision")
    assert sorted(entry.spec["target"] for entry in entries) == [
        f"tool_call:r{n}" for n in range(1, lfm.MAX_ROUNDS + 1)
    ]
    assert {entry.state for entry in entries} == {DONE}


def test_a_proposal_the_tier_refuses_is_fed_back_and_costs_a_round(tmp_path):
    unknown = _call("propose", {"operation": "no_such_operation", "arguments": {}})
    record, rounds, _ = _finish(tmp_path, [("answer", unknown), ("answer", RESTART)])
    assert (record.outcome, record.operation, rounds) == ("propose", "service_restart", 2)


# ---------------------------------------------------------------------------
# failures are never swallowed into a decision
# ---------------------------------------------------------------------------


class _NoReplay(fake.FakeProvider):
    def result_from_raw(self, request, raw):
        raise RuntimeError("cannot re-read")


def test_a_cache_that_cannot_be_reread_raises_instead_of_scoring(tmp_path):
    provider = _NoReplay(script=[("answer", STATUS)])
    with Ledger(tmp_path) as ledger:
        _sync(provider, ledger, _round([_case()], provider, ledger))
        with pytest.raises(RuntimeError, match="cannot re-read"):
            _round([_case()], provider, ledger)


def test_record_results_refuses_a_result_for_an_unknown_case(tmp_path):
    provider = fake.FakeProvider(script=[("answer", STATUS)])
    with Ledger(tmp_path) as ledger:
        current = _round([_case()], provider, ledger)
        stray = provider.submit_sync(
            CallRequest(case_id="someone-else", split="test", case_text="t")
        )
        with pytest.raises(ValueError):
            loop.record_results(ledger, current.pending, [stray])


# ---------------------------------------------------------------------------
# plan risk r9: a reply with no tool call is read the way ToolChat reads one
# ---------------------------------------------------------------------------


def _spoken(text, *, truncated=False, kind="malformed"):
    return fake.ScriptedOutcome(kind=kind, answer=None, text=text, truncated=truncated)


def test_plain_words_are_an_explanation_as_for_a_candidate(tmp_path):
    record, rounds, entries = _finish(tmp_path, [_spoken("The service is fine; nothing to do.")])
    assert (record.outcome, record.invalid_reason, rounds) == ("explain", None, 1)
    # The adapter's own ledger state is untouched: the bytes stay cached.
    assert [entry.state for entry in entries] == [INVALID]


def test_a_tool_call_printed_in_the_text_is_parsed_like_toolchat(tmp_path):
    printed = '<tool_call>{"name": "escalate", "arguments": {"reason": "too big"}}</tool_call>'
    record, _, _ = _finish(tmp_path, [_spoken(printed)])
    assert (record.outcome, record.invalid_reason) == ("escalate", None)


def test_unparsable_tool_markup_is_invalid_like_a_candidate(tmp_path):
    record, _, _ = _finish(tmp_path, [_spoken("<tool_call>{not json</tool_call>")])
    assert record.outcome == "invalid"
    assert record.invalid_reason == loop.measure.UNPARSED_TOOL_CALL


def test_a_truncated_reply_is_never_an_explanation(tmp_path):
    record, _, _ = _finish(tmp_path, [_spoken("The service is", truncated=True)])
    assert (record.outcome, record.invalid_reason) == ("invalid", loop.TRUNCATED)


def test_a_structural_refusal_is_invalid_whatever_its_text(tmp_path):
    record, _, _ = _finish(tmp_path, [_spoken("I can't help with that.", kind="refusal")])
    assert (record.outcome, record.invalid_reason) == ("invalid", "refusal")


def test_inline_reasoning_alone_is_no_usable_output(tmp_path):
    record, _, _ = _finish(tmp_path, [_spoken("<think>maybe restart it</think>")])
    assert (record.outcome, record.invalid_reason) == ("invalid", "malformed")


def test_every_loop_call_asks_for_auto_tool_choice(tmp_path):
    provider = fake.FakeProvider(script=[("answer", STATUS)])
    with Ledger(tmp_path) as ledger:
        (call,) = _round([_case()], provider, ledger).pending
        assert call.request.params["tool_choice"] == "auto"
        assert call.spec.params["tool_choice"] == "auto"
        # A caller cannot switch the loop back to forced tool use.
        chat = loop.DeferredChat(
            case=_case(),
            provider=provider,
            model="m",
            ledger=ledger,
            params={"tool_choice": "required"},
        )
        assert chat._params["tool_choice"] == "auto"
