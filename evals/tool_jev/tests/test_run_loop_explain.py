"""The explain text a reference's Track A loop ended on (t17's judge input).

``track_a_loop.run_round`` exposes, per finished case, the text of the
``Explanation`` LfmTier returned -- the explain tool's ``text`` argument or a
plain-words reply -- so the judge panel scores exactly what the loop read.
Nothing else becomes explain text. Synthetic cases; FakeProvider; no network.
"""

from __future__ import annotations

import json

from evals.tool_jev import track_a_loop as loop
from evals.tool_jev.cases import Case
from evals.tool_jev.ledger import Ledger
from evals.tool_jev.providers import fake
from nvsh.platform._model import Platform

PLATFORM = Platform(kind="jetson")
SNAPSHOT = {"services": [], "containers": [], "source": "synthetic", "created": "2026-09-26"}


def _case(case_id: str) -> Case:
    return Case(
        id=case_id,
        split="test",
        text="why is the synthetic thing slow",
        candidates=("service_status",),
        expect={"explain": True},
        read_only=None,
    )


def _drive(provider, cases, ledger):
    while True:
        result = loop.run_round(
            cases,
            provider=provider,
            model="fake-model",
            ledger=ledger,
            snapshot=SNAPSHOT,
            platform=PLATFORM,
        )
        if not result.pending:
            return result
        answers = [provider.submit_sync(call.request) for call in result.pending]
        assert loop.record_results(ledger, result.pending, answers) == []


def test_explain_tool_text_and_plain_words_are_exposed(tmp_path):
    tool = json.dumps({"name": "explain", "arguments": {"text": "The disk is full."}})
    provider = fake.FakeProvider(
        script=[
            ("answer", tool),
            fake.ScriptedOutcome("malformed", text="It is a network timeout."),
            ("answer", json.dumps({"name": "escalate", "arguments": {"reason": "unsure"}})),
        ]
    )
    cases = [_case("e1"), _case("e2"), _case("e3")]
    with Ledger(tmp_path) as ledger:
        result = _drive(provider, cases, ledger)
    assert {cid: rec.outcome for cid, rec in result.finished.items()} == {
        "e1": "explain",
        "e2": "explain",
        "e3": "escalate",
    }
    assert result.explanations == {"e1": "The disk is full.", "e2": "It is a network timeout."}


def test_run_case_detail_returns_the_record_and_its_explanation(tmp_path):
    tool = json.dumps({"name": "explain", "arguments": {"text": "Because."}})
    provider = fake.FakeProvider(script=[("answer", tool)])
    with Ledger(tmp_path) as ledger:
        first = loop.run_case_detail(
            _case("d1"),
            provider=provider,
            model="fake-model",
            ledger=ledger,
            snapshot=SNAPSHOT,
            platform=PLATFORM,
        )
        assert isinstance(first.outcome, loop.PendingCall) and first.explanation is None
        (answer,) = [provider.submit_sync(first.outcome.request)]
        loop.record_results(ledger, [first.outcome], [answer])
        done = loop.run_case_detail(
            _case("d1"),
            provider=provider,
            model="fake-model",
            ledger=ledger,
            snapshot=SNAPSHOT,
            platform=PLATFORM,
        )
    assert done.outcome.outcome == "explain"
    assert done.explanation == "Because."
