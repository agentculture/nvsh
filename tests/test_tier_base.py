"""Tests for the tier contract, decision validation and fixture tier."""

from __future__ import annotations

import abc

import pytest

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.tiers.base import Decline, DeclineReason, Tier, TierDecision, decide
from nvsh.tiers.fake import FakeTier


def _req() -> AgentRequest:
    return AgentRequest(kind=RequestKind.EXPLICIT, prompt="restart vllm")


def _ctx() -> AgentContext:
    return AgentContext(platform="jetson")


# ---------------------------------------------------------------------------
# decide(): basic call-count rules
# ---------------------------------------------------------------------------


def test_zero_calls_is_decline_no_call():
    result = decide([])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.NO_CALL


def test_multiple_calls_is_decline():
    result = decide(
        [
            {"name": "service_status", "arguments": {"service": "vllm"}},
            {"name": "service_status", "arguments": {"service": "nginx"}},
        ]
    )
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.MULTIPLE_CALLS


def test_exactly_one_valid_call_becomes_decision():
    result = decide([{"name": "service_restart", "arguments": {"service": "vLLM"}}])
    assert isinstance(result, TierDecision)
    assert result.operation == "service_restart"
    assert result.args == {"service": "vLLM"}
    assert result.read_only is False


def test_exactly_one_valid_read_only_call_marks_read_only():
    result = decide([{"name": "machine_status", "arguments": {}}])
    assert isinstance(result, TierDecision)
    assert result.read_only is True


# ---------------------------------------------------------------------------
# decide(): rejecting bad tier output
# ---------------------------------------------------------------------------


def test_unknown_operation_is_decline():
    result = decide([{"name": "not_a_real_op", "arguments": {}}])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.UNKNOWN_OPERATION


def test_bad_argument_is_decline():
    # service_restart requires "service"; a stderr fragment as the wrong key.
    result = decide([{"name": "service_restart", "arguments": {"stderr": "boom"}}])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.BAD_ARGUMENT


def test_missing_argument_is_decline():
    result = decide([{"name": "service_restart", "arguments": {}}])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.BAD_ARGUMENT


def test_wrong_type_argument_is_decline():
    result = decide([{"name": "service_restart", "arguments": {"service": 5}}])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.BAD_ARGUMENT


def test_raw_shell_string_is_decline():
    result = decide(["sudo systemctl restart vllm"])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.RAW_SHELL


def test_call_named_bash_is_raw_shell_decline():
    result = decide([{"name": "bash", "arguments": {"command": "rm -rf /"}}])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.RAW_SHELL


def test_call_named_shell_is_raw_shell_decline():
    result = decide([{"name": "shell", "arguments": {"cmd": "ls"}}])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.RAW_SHELL


def test_call_named_run_is_raw_shell_decline():
    result = decide([{"name": "run", "arguments": {"command": "ls"}}])
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.RAW_SHELL


def test_service_restart_with_shell_injection_arg_passes_here():
    # Grounding, not decide(), is responsible for catching this -- decide()
    # only rejects what table.validate() rejects, plus raw-shell-string.
    result = decide([{"name": "service_restart", "arguments": {"service": "rm -rf /"}}])
    assert isinstance(result, TierDecision)
    assert result.args == {"service": "rm -rf /"}


# ---------------------------------------------------------------------------
# decide(): malformed / non-list input must never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        42,
        "not a list",
        {"name": "machine_status"},
        [None],
        [42],
        [{"name": "machine_status"}],  # missing "arguments" key entirely
        [{"arguments": {}}],  # missing "name" key
        [{"name": 5, "arguments": {}}],
        [{"name": "machine_status", "arguments": "not a dict"}],
    ],
)
def test_malformed_input_never_raises(raw):
    result = decide(raw)
    assert isinstance(result, Decline)


# ---------------------------------------------------------------------------
# decide(): confidence
# ---------------------------------------------------------------------------


def test_confidence_none_does_not_decline():
    result = decide(
        [{"name": "machine_status", "arguments": {}}], confidence=None, min_confidence=0.5
    )
    assert isinstance(result, TierDecision)
    assert result.confidence is None


def test_confidence_below_floor_is_low_confidence_decline():
    result = decide(
        [{"name": "machine_status", "arguments": {}}], confidence=0.2, min_confidence=0.5
    )
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.LOW_CONFIDENCE


def test_confidence_at_or_above_floor_is_ok():
    result = decide(
        [{"name": "machine_status", "arguments": {}}], confidence=0.5, min_confidence=0.5
    )
    assert isinstance(result, TierDecision)
    assert result.confidence == 0.5


def test_confidence_malformed_type_never_raises():
    result = decide([{"name": "machine_status", "arguments": {}}], confidence="high")
    # A non-numeric confidence must not crash decide(); treat it as absent.
    assert isinstance(result, (TierDecision, Decline))


# ---------------------------------------------------------------------------
# Decline detail is a one-line string
# ---------------------------------------------------------------------------


def test_decline_detail_is_present_and_single_line():
    result = decide([{"name": "nope", "arguments": {}}])
    assert isinstance(result, Decline)
    assert isinstance(result.detail, str)
    assert "\n" not in result.detail


# ---------------------------------------------------------------------------
# Tier abstract contract
# ---------------------------------------------------------------------------


def test_tier_is_abstract_and_cannot_be_instantiated():
    assert issubclass(Tier, abc.ABC)
    with pytest.raises(TypeError):
        Tier()  # type: ignore[abstract]


def test_no_tier_type_has_execute_run_or_exec_attribute():
    import nvsh.tiers.base as base_module

    forbidden = {"execute", "run", "exec"}
    for obj in (Tier, FakeTier):
        for name in dir(obj):
            assert name not in forbidden, f"{obj.__name__} exposes forbidden attribute {name!r}"

    # Also scan the base module's own namespace for stray callables.
    for name in dir(base_module):
        assert name not in forbidden


# ---------------------------------------------------------------------------
# FakeTier
# ---------------------------------------------------------------------------


def test_fake_tier_replays_raw_output_through_decide():
    tier = FakeTier([[{"name": "machine_status", "arguments": {}}]])
    result = tier.select(_req(), _ctx())
    assert isinstance(result, TierDecision)
    assert result.operation == "machine_status"


def test_fake_tier_replays_ready_made_decision():
    decision = TierDecision(operation="machine_status", args={}, confidence=0.9, read_only=True)
    tier = FakeTier([decision])
    result = tier.select(_req(), _ctx())
    assert result is decision


def test_fake_tier_replays_ready_made_decline():
    decline = Decline(reason=DeclineReason.TIER_UNAVAILABLE, detail="offline")
    tier = FakeTier([decline])
    result = tier.select(_req(), _ctx())
    assert result is decline


def test_fake_tier_replays_exception():
    tier = FakeTier([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        tier.select(_req(), _ctx())


def test_fake_tier_records_requests_seen():
    tier = FakeTier([[{"name": "machine_status", "arguments": {}}]])
    req = _req()
    ctx = _ctx()
    tier.select(req, ctx)
    assert tier.requests_seen == [req]


def test_fake_tier_close_sets_closed_flag():
    tier = FakeTier([])
    assert tier.closed is False
    tier.close()
    assert tier.closed is True


def test_fake_tier_advances_through_script_in_order():
    tier = FakeTier(
        [
            [{"name": "machine_status", "arguments": {}}],
            [{"name": "gpu_stats", "arguments": {}}],
        ]
    )
    first = tier.select(_req(), _ctx())
    second = tier.select(_req(), _ctx())
    assert isinstance(first, TierDecision) and first.operation == "machine_status"
    assert isinstance(second, TierDecision) and second.operation == "gpu_stats"


def test_fake_tier_exhausted_script_raises():
    tier = FakeTier([])
    with pytest.raises(Exception):
        tier.select(_req(), _ctx())


def test_fake_tier_is_a_tier_and_has_name():
    tier = FakeTier([])
    assert isinstance(tier, Tier)
    assert isinstance(tier.name, str)
