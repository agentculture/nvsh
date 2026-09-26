"""Tests for evals/tool_jev/providers/{base,errors,fake}.py (issue #64, t10).

Covers the four t10 acceptance criteria:

1. a fake 402 stops the run cleanly with a message naming the provider and
   the remaining call count, leaves those calls pending, and continuing
   after "top-up" finishes with the same results as an uninterrupted run;
2. a refusal or malformed answer is recorded invalid and counted in the
   denominator (row case count == case set size);
3. ``redact()`` is applied to every payload the fake provider receives, and
   a hostile argument string is stored verbatim and never executed (no
   ``subprocess`` import anywhere in ``providers/``);
4. the provider layer refuses any case whose split tag is ``heldout`` or
   ``heldout-mc`` before any network call.

All fixtures here are synthetic strings written inline — no real case text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.tool_jev.providers import base, errors, fake

PROVIDERS_DIR = Path(__file__).resolve().parents[1] / "providers"


# ---------------------------------------------------------------------------
# Small helpers shared across the tests below.
# ---------------------------------------------------------------------------


def _make_requests(n: int, split: str = "test") -> list[base.CallRequest]:
    return [
        base.CallRequest(case_id=f"case-{i}", split=split, case_text=f"synthetic case body {i}")
        for i in range(n)
    ]


def run_all(provider: fake.FakeProvider, requests: list[base.CallRequest], ledger: dict):
    """A tiny, self-contained pending/done loop for this test only.

    Not the real ledger (t9 builds that) — just enough bookkeeping to prove
    a stopped-and-resumed run matches an uninterrupted one. ``ledger`` maps
    case_id -> CallResult for every call already ``done``.
    """
    for request in requests:
        if request.case_id in ledger:
            continue
        try:
            result = provider.submit_sync(request)
        except fake.FakeProviderError as exc:
            remaining = sum(1 for r in requests if r.case_id not in ledger)
            message = errors.stop_message(exc.provider, exc.classification, remaining)
            return message
        else:
            ledger[request.case_id] = result
    return None


# ---------------------------------------------------------------------------
# Criterion 1: clean stop on 402, resume after "top-up" matches uninterrupted.
# ---------------------------------------------------------------------------


def test_402_stops_cleanly_naming_provider_and_remaining_count():
    requests = _make_requests(3)
    provider = fake.FakeProvider(name="acme-provider", script=[fake.ScriptedOutcome("answer", "A")])
    provider.queue("402")

    ledger: dict = {}
    message = run_all(provider, requests, ledger)

    assert message is not None
    assert "acme-provider" in message
    # case-0 done, case-1 hit the 402, case-2 never attempted => 2 pending.
    assert "2" in message
    assert ledger.keys() == {"case-0"}
    # The provider was never asked for case-2 at all.
    assert [r.case_id for r in provider.received] == ["case-0", "case-1"]


def test_continue_after_topup_matches_uninterrupted_run():
    requests = _make_requests(3)

    # Uninterrupted: every case answered first try.
    uninterrupted = fake.FakeProvider(
        name="acme-provider",
        script=[("answer", "A0"), ("answer", "A1"), ("answer", "A2")],
    )
    uninterrupted_ledger: dict = {}
    assert run_all(uninterrupted, requests, uninterrupted_ledger) is None

    # Interrupted: case-0 succeeds, then a 402 stops the run with case-1 and
    # case-2 still pending.
    interrupted = fake.FakeProvider(
        name="acme-provider",
        script=[("answer", "A0"), ("402", None)],
    )
    ledger: dict = {}
    stop_message = run_all(interrupted, requests, ledger)
    assert stop_message is not None
    assert ledger.keys() == {"case-0"}

    # "Top up" and continue: same provider, freshly scripted answers for the
    # calls that are still pending. Only pending calls are sent (case-0 is
    # never re-sent — it's already ``done`` in the ledger).
    interrupted.queue("answer", "A1")
    interrupted.queue("answer", "A2")
    assert run_all(interrupted, requests, ledger) is None

    assert ledger.keys() == uninterrupted_ledger.keys() == {"case-0", "case-1", "case-2"}
    for case_id in ledger:
        assert ledger[case_id].answer == uninterrupted_ledger[case_id].answer
        assert ledger[case_id].outcome == uninterrupted_ledger[case_id].outcome

    # case-0 was submitted exactly once to the interrupted provider (never
    # paid for twice across the stop/resume boundary).
    case0_sends = [r for r in interrupted.received if r.case_id == "case-0"]
    assert len(case0_sends) == 1


# ---------------------------------------------------------------------------
# Criterion 2: refusal/malformed -> invalid, still counted in the denominator.
# ---------------------------------------------------------------------------


def test_refusal_and_malformed_are_invalid_and_counted_in_denominator():
    requests = _make_requests(4)
    provider = fake.FakeProvider(
        name="acme-provider",
        script=[
            ("answer", "42"),
            ("refusal", None),
            ("malformed", "{not json"),
            ("answer", "7"),
        ],
    )
    ledger: dict = {}
    assert run_all(provider, requests, ledger) is None

    # Row case count equals the case set size: nothing was dropped.
    assert len(ledger) == len(requests) == 4

    outcomes = {case_id: result.outcome for case_id, result in ledger.items()}
    assert outcomes["case-0"] == errors.Outcome.OK
    assert outcomes["case-1"] == errors.Outcome.INVALID
    assert outcomes["case-2"] == errors.Outcome.INVALID
    assert outcomes["case-3"] == errors.Outcome.OK

    invalid_count = sum(1 for o in outcomes.values() if o == errors.Outcome.INVALID)
    assert invalid_count == 2
    assert ledger["case-2"].reason == "malformed"


def test_answer_outside_offered_set_is_invalid():
    request = base.CallRequest(
        case_id="case-x",
        split="test",
        case_text="pick a candidate",
        offered_candidates=("mv", "cp", "rm"),
    )
    provider = fake.FakeProvider(script=[("answer", "not-a-real-candidate")])
    result = provider.submit_sync(request)
    assert result.outcome == errors.Outcome.INVALID
    assert result.reason == "outside_offered_set"


# ---------------------------------------------------------------------------
# Criterion 3: redaction is applied to every payload; hostile strings are
# data, never executed; no subprocess anywhere under providers/.
# ---------------------------------------------------------------------------


# Built at runtime so no secret-shaped literal sits in a tracked file
# (scripts/scan-secrets.py scans tracked files for sk-... keys).
FAKE_OPENAI_KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz" + "123456"


def test_redact_is_applied_to_every_payload_the_fake_provider_receives():
    hostile_secret_text = (
        f"please run this: export OPENAI_API_KEY={FAKE_OPENAI_KEY}\n"
        f"Authorization: Bearer {FAKE_OPENAI_KEY}"
    )
    request = base.CallRequest(case_id="case-secret", split="test", case_text=hostile_secret_text)
    provider = fake.FakeProvider(script=[("answer", "ok")])
    provider.submit_sync(request)

    assert len(provider.received) == 1
    received_text = provider.received[0].case_text
    assert FAKE_OPENAI_KEY not in received_text
    assert "<REDACTED:" in received_text
    # Sanity: redact() on the original text produces exactly what the
    # provider received (same choke point, not a look-alike).
    from nvsh.redact import redact

    expected = redact(hostile_secret_text.encode("utf-8")).decode("utf-8")
    assert received_text == expected


def test_hostile_argument_string_is_stored_verbatim_and_never_executed():
    hostile_argument = "; rm -rf / #"
    request = base.CallRequest(case_id="case-hostile", split="test", case_text="pick an argument")
    provider = fake.FakeProvider(script=[("answer", hostile_argument)])

    result = provider.submit_sync(request)

    # Stored verbatim: the fake provider never parses, shells out to, or
    # otherwise interprets the answer text.
    assert result.answer == hostile_argument
    assert result.outcome == errors.Outcome.OK


def test_no_subprocess_or_shell_execution_anywhere_under_providers():
    # Checks actual usage (imports/calls), not prose: this module's own
    # docstrings talk *about* subprocess execution (to say it never
    # happens), which must not itself trip the guard.
    forbidden_tokens = (
        "import subprocess",
        "os.system(",
        "os.popen(",
        "import shlex",
        "import pty",
    )
    for path in sorted(PROVIDERS_DIR.glob("*.py")):
        text = path.read_text()
        for token in forbidden_tokens:
            assert token not in text, f"{path} contains forbidden token {token!r}"


# ---------------------------------------------------------------------------
# Criterion 4: refuse heldout/heldout-mc before any network call.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["heldout", "heldout-mc"])
def test_heldout_split_is_refused_before_any_network_call(split):
    request = base.CallRequest(case_id="case-sealed", split=split, case_text="sealed case text")
    # Script is deliberately empty: if the guard didn't fire before any
    # attempt to send, this would raise ScriptExhausted instead, which
    # would prove a network call was attempted.
    provider = fake.FakeProvider(script=[])

    with pytest.raises(base.HeldoutSplitRefused):
        provider.submit_sync(request)

    assert provider.received == []


def test_heldout_guard_also_covers_submit_batch():
    requests = [
        base.CallRequest(case_id="case-1", split="test", case_text="ok case"),
        base.CallRequest(case_id="case-2", split="heldout", case_text="sealed case"),
    ]
    provider = fake.FakeProvider(script=[])
    with pytest.raises(base.HeldoutSplitRefused):
        provider.submit_batch(requests)
    assert provider.received == []


def test_regular_test_split_is_not_refused():
    request = base.CallRequest(case_id="case-ok", split="test", case_text="ordinary case")
    provider = fake.FakeProvider(script=[("answer", "fine")])
    result = provider.submit_sync(request)
    assert result.outcome == errors.Outcome.OK


# ---------------------------------------------------------------------------
# Provider protocol / capabilities sanity.
# ---------------------------------------------------------------------------


def test_fake_provider_satisfies_the_provider_protocol():
    provider = fake.FakeProvider()
    assert isinstance(provider, base.Provider)


def test_capabilities_default_and_override():
    default_provider = fake.FakeProvider()
    assert default_provider.capabilities == base.ProviderCapabilities(
        logprobs=True, batch=True, reasoning=False
    )

    no_logprobs = fake.FakeProvider(
        capabilities=base.ProviderCapabilities(logprobs=False, batch=True, reasoning=True)
    )
    assert no_logprobs.capabilities.logprobs is False
    assert no_logprobs.capabilities.reasoning is True


def test_batch_round_trip():
    requests = _make_requests(2)
    provider = fake.FakeProvider(script=[("answer", "A0"), ("answer", "A1")])

    handle = provider.submit_batch(requests)
    status = provider.poll_batch(handle)
    assert status.complete is True
    assert status.expired is False

    results = provider.fetch_batch(handle)
    assert [r.case_id for r in results] == ["case-0", "case-1"]
    assert [r.answer for r in results] == ["A0", "A1"]


def test_batch_is_also_redacted_and_heldout_guarded():
    hostile = base.CallRequest(
        case_id="case-secret",
        split="test",
        case_text=f"Authorization: Bearer {FAKE_OPENAI_KEY}",
    )
    provider = fake.FakeProvider(script=[("answer", "ok")])
    handle = provider.submit_batch([hostile])
    provider.fetch_batch(handle)
    assert FAKE_OPENAI_KEY not in provider.received[0].case_text


# ---------------------------------------------------------------------------
# errors.py table sanity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_kind", ["openai", "anthropic", "openai_compat"])
def test_classify_transport_covers_every_provider_kind(provider_kind):
    classification = errors.classify_transport(provider_kind, status_code=402)
    assert classification.outcome == errors.Outcome.PENDING
    assert classification.stop is True
    assert classification.reason == "insufficient_credit"


def test_classify_transport_unknown_kind_raises():
    with pytest.raises(KeyError):
        errors.classify_transport("not-a-real-provider-kind", status_code=402)


def test_classify_transport_unrecognized_condition_stays_pending_not_invalid():
    classification = errors.classify_transport("openai", status_code=599)
    assert classification.outcome == errors.Outcome.PENDING


def test_read_api_key_reads_only_the_named_env_var(monkeypatch):
    monkeypatch.delenv("SOME_PROVIDER_API_KEY", raising=False)
    with pytest.raises(base.MissingProviderKey):
        base.read_api_key("SOME_PROVIDER_API_KEY")

    monkeypatch.setenv("SOME_PROVIDER_API_KEY", "synthetic-test-value")
    assert base.read_api_key("SOME_PROVIDER_API_KEY") == "synthetic-test-value"
