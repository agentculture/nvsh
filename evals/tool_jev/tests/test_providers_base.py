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
        provider.submit_batch(requests, submit_ref="tj-ref-g")
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

    handle = provider.submit_batch(requests, submit_ref="tj-ref-rt")
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
    handle = provider.submit_batch([hostile], submit_ref="tj-ref-h")
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


# ---------------------------------------------------------------------------
# Review fix P1: batch crash recovery through the Provider API itself.
#
# The ledger hands the runner a submit ref (``Ledger.begin_submit``) BEFORE
# the provider call. The provider must carry that ref with the batch and be
# able to find the batch by it, so a batch accepted just before a crash is
# re-attached, never paid for twice.
# ---------------------------------------------------------------------------


def test_submit_batch_carries_the_submit_ref_on_the_handle():
    provider = fake.FakeProvider(script=[("answer", "A0")])
    handle = provider.submit_batch(_make_requests(1), submit_ref="tj-ref-1")
    assert handle.submit_ref == "tj-ref-1"
    assert provider.submitted_refs == ["tj-ref-1"]


def test_find_batch_returns_the_accepted_batch_and_none_for_an_unknown_ref():
    provider = fake.FakeProvider(script=[("answer", "A0"), ("answer", "A1")])
    handle = provider.submit_batch(_make_requests(2), submit_ref="tj-ref-1")

    found = provider.find_batch("tj-ref-1")
    assert found == handle
    assert provider.find_batch("tj-never-submitted") is None
    # Re-attaching through the found handle fetches the original batch.
    assert [r.answer for r in provider.fetch_batch(found)] == ["A0", "A1"]


@pytest.mark.parametrize("bad_ref", ["", "   "])
def test_submit_batch_and_find_batch_refuse_an_empty_submit_ref(bad_ref):
    provider = fake.FakeProvider(script=[])
    with pytest.raises(ValueError):
        provider.submit_batch(_make_requests(1), submit_ref=bad_ref)
    with pytest.raises(ValueError):
        provider.find_batch(bad_ref)
    assert provider.received == []


def test_fake_refuses_to_accept_the_same_submit_ref_twice():
    """A resubmission under the same ref is exactly the double charge P1 is about."""
    provider = fake.FakeProvider(script=[])
    provider.submit_batch(_make_requests(1), submit_ref="tj-ref-1")
    with pytest.raises(fake.DuplicateSubmitRef):
        provider.submit_batch(_make_requests(1), submit_ref="tj-ref-1")


def test_heldout_guard_runs_before_submit_ref_is_recorded():
    provider = fake.FakeProvider(script=[])
    with pytest.raises(base.HeldoutSplitRefused):
        provider.submit_batch(_make_requests(1, split="heldout-mc"), submit_ref="tj-ref-1")
    assert provider.submitted_refs == []
    assert provider.find_batch("tj-ref-1") is None


def test_ledger_orphan_is_recovered_via_find_batch_not_resubmitted(tmp_path):
    """End to end with the real ledger: crash after acceptance, before mark_submitted."""
    from evals.tool_jev.ledger import DONE, CachedResponse, CallSpec, Ledger, prompt_hash

    specs = [
        CallSpec(
            provider="fake",
            model="fake-model",
            subject_role="subject",
            case_id=f"case-{i}",
            target="interface:tool",
            prompt_hash=prompt_hash(f"synthetic prompt {i}"),
        )
        for i in range(2)
    ]
    provider = fake.FakeProvider(script=[("answer", "A0"), ("answer", "A1")])

    with Ledger(tmp_path) as led:
        keys = led.register_many(specs)
        submit_ref = led.begin_submit(keys)
        requests = [base.CallRequest(case_id=k, split="test", case_text="synthetic") for k in keys]
        provider.submit_batch(requests, submit_ref=submit_ref)
        # power-off here: mark_submitted never ran

    with Ledger(tmp_path) as led:
        plan = led.continue_plan()
        assert plan.orphans == {submit_ref: sorted(keys)}
        for ref, orphan_keys in plan.orphans.items():
            handle = provider.find_batch(ref)
            assert handle is not None
            led.mark_submitted(orphan_keys, handle.batch_id)
        for batch_id in led.submitted_batches():
            handle = provider.find_batch(submit_ref)
            assert handle.batch_id == batch_id
            assert provider.poll_batch(handle).complete
            for result in provider.fetch_batch(handle):
                led.record_done(
                    result.case_id,
                    CachedResponse(
                        raw=result.raw,
                        model_id=result.returned_model,
                        response_id=result.response_id,
                        usage=result.usage,
                    ),
                )
        assert led.keys(DONE) == sorted(keys)

    assert provider.submitted_refs == [submit_ref]  # accepted once, never resubmitted
    assert len(provider.received) == 2


# ---------------------------------------------------------------------------
# Review fix P2: CallResult carries what the runner and cache need.
# ---------------------------------------------------------------------------


def test_call_result_defaults_keep_the_fake_easy_to_script():
    result = base.CallResult(case_id="c", outcome=errors.Outcome.OK)
    assert result.candidates is None
    assert result.raw == b""
    assert result.response_id == ""
    assert result.returned_model is None
    assert result.usage == {}
    assert result.interface == "tool_call"


def test_call_result_validates_its_new_fields():
    with pytest.raises(TypeError):
        base.CallResult(case_id="c", outcome=errors.Outcome.OK, raw="not bytes")
    with pytest.raises(TypeError):
        base.CallResult(case_id="c", outcome=errors.Outcome.OK, usage={"input": 1.5})
    with pytest.raises(ValueError):
        base.CallResult(case_id="c", outcome=errors.Outcome.OK, interface="freeform")
    with pytest.raises(TypeError):
        base.CallResult(case_id="c", outcome=errors.Outcome.OK, candidates={"a": "high"})


def test_fake_result_carries_raw_response_id_model_usage_and_interface():
    provider = fake.FakeProvider(
        name="acme",
        model="acme-model-2",
        script=[
            fake.ScriptedOutcome(
                "answer", "disk_usage", usage={"input_tokens": 12, "output_tokens": 3}
            )
        ],
    )
    request = base.CallRequest(
        case_id="c1",
        split="test",
        case_text="synthetic",
        offered_candidates=("disk_usage", "(explain)"),
        interface="choice",
    )
    result = provider.submit_sync(request)
    assert isinstance(result.raw, bytes) and result.raw
    assert b"disk_usage" in result.raw
    assert result.response_id
    assert result.returned_model == "acme-model-2"
    assert result.usage == {"input_tokens": 12, "output_tokens": 3}
    assert result.interface == "choice"
    # No logprobs scripted -> no distribution, never an estimated one.
    assert result.candidates is None


def test_fake_result_carries_a_scripted_distribution_in_offered_order():
    distribution = {"disk_usage": 0.5, "(explain)": 0.5}
    provider = fake.FakeProvider(
        script=[fake.ScriptedOutcome("answer", "disk_usage", candidates=distribution)]
    )
    request = base.CallRequest(
        case_id="c1",
        split="test",
        case_text="synthetic",
        offered_candidates=("disk_usage", "(explain)"),
    )
    result = provider.submit_sync(request)
    assert result.candidates == distribution
    assert list(result.candidates) == ["disk_usage", "(explain)"]


def test_fake_without_logprobs_refuses_a_scripted_distribution():
    provider = fake.FakeProvider(
        capabilities=base.ProviderCapabilities(logprobs=False, batch=True, reasoning=False),
        script=[fake.ScriptedOutcome("answer", "a", candidates={"a": 1.0})],
    )
    with pytest.raises(ValueError):
        provider.submit_sync(base.CallRequest(case_id="c", split="test", case_text="x"))


def test_call_request_rejects_an_unknown_interface():
    with pytest.raises(ValueError):
        base.CallRequest(case_id="c", split="test", case_text="x", interface="freeform")


# ---------------------------------------------------------------------------
# Review fix P2: refusal is structural only, never a substring guess.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I cannot determine disk usage from this output; run df -h.",
        "I can't see a mount point here, so check lsblk first.",
        "As an AI reading this log, the failing step is the nvcc call.",
    ],
)
def test_explanation_that_mentions_inability_is_ok_not_refusal(text):
    classification = errors.classify_answer(text)
    assert classification.outcome == errors.Outcome.OK


def test_structural_refusal_is_invalid_even_with_empty_text():
    classification = errors.classify_answer(None, refused=True)
    assert classification.outcome == errors.Outcome.INVALID
    assert classification.reason == "refusal"
    assert errors.classify_answer("some text", refused=True).reason == "refusal"


def test_no_refusal_phrase_table_remains():
    assert not hasattr(errors, "_REFUSAL_MARKERS")
    assert not hasattr(errors, "_looks_like_refusal")


def test_fake_refusal_kind_is_a_structural_refusal():
    provider = fake.FakeProvider(script=[("refusal", "Here is a normal-looking sentence.")])
    result = provider.submit_sync(base.CallRequest(case_id="c", split="test", case_text="x"))
    assert result.outcome == errors.Outcome.INVALID
    assert result.reason == "refusal"


def test_empty_malformed_and_outside_set_stay_invalid():
    assert errors.classify_answer("").reason == "empty_answer"
    assert errors.classify_answer("x", malformed=True).reason == "malformed"
    assert errors.classify_answer("z", ("a", "b")).reason == "outside_offered_set"


# ---------------------------------------------------------------------------
# Review fix P2: permanent request errors stop, non-retryable, distinct reason.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_kind", ["openai", "anthropic", "openai_compat"])
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_permanent_http_errors_are_request_rejected_not_retryable(provider_kind, status_code):
    c = errors.classify_transport(provider_kind, status_code=status_code)
    assert c.outcome == errors.Outcome.PENDING  # never INVALID, never in the denominator
    assert c.stop is True
    assert c.retryable is False
    assert c.reason.startswith(errors.REQUEST_REJECTED_PREFIX)
    assert c.rejected is True


@pytest.mark.parametrize(
    "error_type", ["invalid_request", "auth_failed", "model_not_found", "unsupported_parameter"]
)
def test_named_permanent_errors_are_request_rejected_with_detail(error_type):
    c = errors.classify_transport("openai", error_type=error_type)
    assert c.outcome == errors.Outcome.PENDING
    assert c.stop is True
    assert c.retryable is False
    assert c.reason == f"request_rejected:{error_type}"


def test_named_error_type_is_more_specific_than_a_generic_400():
    c = errors.classify_transport("openai", status_code=400, error_type="unsupported_parameter")
    assert c.reason == "request_rejected:unsupported_parameter"


@pytest.mark.parametrize("status_code", [402, 408, 429, 500, 502, 503, 504])
def test_transient_http_errors_stay_retryable(status_code):
    c = errors.classify_transport("anthropic", status_code=status_code)
    assert c.outcome == errors.Outcome.PENDING
    assert c.stop is True
    assert c.retryable is True
    assert c.rejected is False


@pytest.mark.parametrize(
    "error_type",
    [
        "insufficient_quota",
        "budget_cap_reached",
        "rate_limited",
        "timeout",
        "network_error",
        "connection_reset",
        "batch_expired",
    ],
)
def test_transient_named_errors_stay_retryable(error_type):
    c = errors.classify_transport("openai_compat", error_type=error_type)
    assert c.retryable is True
    assert not c.reason.startswith(errors.REQUEST_REJECTED_PREFIX)


def test_answer_classifications_are_not_retryable():
    assert errors.classify_answer("ok").retryable is False
    assert errors.classify_answer("").retryable is False


def test_stop_message_says_which_case_it_is():
    rejected = errors.classify_transport("openai", status_code=401)
    transient = errors.classify_transport("openai", status_code=429)

    rejected_msg = errors.stop_message("acme", rejected, 5)
    transient_msg = errors.stop_message("acme", transient, 5)

    assert "acme" in rejected_msg and "5" in rejected_msg
    assert "request_rejected:auth_failed" in rejected_msg
    assert "not retryable" in rejected_msg
    assert "fix the manifest/params, then continue" in rejected_msg

    assert "acme" in transient_msg and "5" in transient_msg
    assert "rate_limited" in transient_msg
    assert "transient" in transient_msg
    assert "fix the manifest" not in transient_msg


@pytest.mark.parametrize("kind", ["400", "401", "403", "404", "422", "model_not_found"])
def test_fake_request_rejected_kinds_raise_a_non_retryable_stop(kind):
    provider = fake.FakeProvider(name="acme", script=[kind])
    with pytest.raises(fake.FakeProviderError) as info:
        provider.submit_sync(base.CallRequest(case_id="c", split="test", case_text="x"))
    assert info.value.classification.retryable is False
    assert info.value.classification.rejected is True


def test_fake_transient_and_rejected_kinds_are_disjoint_and_classified_to_match():
    assert not (fake.TRANSIENT_KINDS & fake.REJECTED_KINDS)
    for kind in sorted(fake.INFRA_KINDS):
        provider = fake.FakeProvider(script=[kind])
        with pytest.raises(fake.FakeProviderError) as info:
            provider.submit_sync(base.CallRequest(case_id="c", split="test", case_text="x"))
        assert info.value.classification.retryable is (kind in fake.TRANSIENT_KINDS), kind
