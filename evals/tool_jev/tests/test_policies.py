"""Tests for evals.tool_jev.policies (issue 64, t8).

Real operation names (``machine_status``, ``gpu_stats`` read_only;
``power_set`` mutating) appear only as fixture data, matching
``tests/test_lfm_finetune_gate.py``'s own convention -- ``gate.decide``
never names an operation in its own logic.
"""

from __future__ import annotations

import json

import pytest

from evals.tool_jev import policies

READ_ONLY_OP = "machine_status"
MUTATING_OP = "power_set"


class FakeProvider:
    """A provider stub that fails the test the moment it is called.

    ``evals/tool_jev/providers`` has no real provider module yet (a sibling
    task builds it); this local stub is only here to prove
    :func:`policies.apply` never reaches for one -- it operates purely on
    an already-recorded ``candidates`` distribution.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("policies.apply must never call a model provider")


def _record(candidates: dict | None) -> dict:
    return {
        "id": "fixture-1",
        "operation": None,
        "arguments": None,
        "candidates": candidates,
    }


@pytest.fixture
def raw_policy():
    return policies.load_policy(policies.builtin_policy_path("raw"))


@pytest.fixture
def shipped_policy():
    return policies.load_policy(policies.builtin_policy_path("scorer-r3b-shipped"))


@pytest.fixture
def mutating_strict_policy():
    return policies.load_policy(policies.builtin_policy_path("mutating-strict-example"))


# ---------------------------------------------------------------------------
# Policy files load and validate
# ---------------------------------------------------------------------------


def test_builtin_policy_files_exist_and_have_name_and_version():
    for policy_name in ("raw", "scorer-r3b-shipped", "mutating-strict-example"):
        path = policies.builtin_policy_path(policy_name)
        assert path.is_file(), path
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == policy_name
        assert data["version"]


def test_raw_policy_has_no_calibration_or_gate(raw_policy):
    assert raw_policy.get("calibration") is None
    assert raw_policy.get("gate") is None


def test_shipped_policy_calibration_and_gate_values(shipped_policy):
    assert shipped_policy["calibration"]["temperature"] == pytest.approx(1.5366)
    assert shipped_policy["calibration"]["vector"] is None
    assert shipped_policy["gate"]["read_only"]["margin"] == pytest.approx(0.2)
    assert shipped_policy["gate"]["mutating"] == {
        "floor": None,
        "margin": None,
        "max_entropy": None,
    }


def test_validate_policy_rejects_missing_name():
    with pytest.raises(policies.PolicyError):
        policies.validate_policy({"version": "1"})


def test_validate_policy_rejects_missing_version():
    with pytest.raises(policies.PolicyError):
        policies.validate_policy({"name": "x"})


def test_validate_policy_rejects_non_mapping():
    with pytest.raises(policies.PolicyError):
        policies.validate_policy(["not", "a", "policy"])


# ---------------------------------------------------------------------------
# Never calls a model
# ---------------------------------------------------------------------------


def test_apply_never_calls_a_model_provider(raw_policy, shipped_policy, mutating_strict_policy):
    fake = FakeProvider()
    record = _record({READ_ONLY_OP: 0.9, MUTATING_OP: 0.1})
    offered = [READ_ONLY_OP, MUTATING_OP]

    for policy in (raw_policy, shipped_policy, mutating_strict_policy):
        policies.apply(policy, record, offered)

    assert fake.calls == 0


# ---------------------------------------------------------------------------
# not_gateable: no recorded distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidates", [None, {}])
def test_apply_reports_not_gateable_without_a_distribution(raw_policy, candidates):
    decision, reason, name, version = policies.apply(
        raw_policy, _record(candidates), [READ_ONLY_OP]
    )
    assert decision == "not_gateable"
    assert reason == "no_distribution"
    assert name == "raw"
    assert version == "1"


def test_apply_not_gateable_under_shipped_gate_too(shipped_policy):
    decision, _reason, _name, _version = policies.apply(
        shipped_policy, _record(None), [READ_ONLY_OP]
    )
    assert decision == "not_gateable"


# ---------------------------------------------------------------------------
# raw policy: bare argmax (no abstention possible)
# ---------------------------------------------------------------------------


def test_raw_policy_never_abstains_even_with_a_thin_margin(raw_policy):
    # top1 beats top2 by only 0.02 -- would fail a 0.2 margin gate, but raw
    # has no gate at all, so it must still propose the argmax.
    record = _record({READ_ONLY_OP: 0.51, MUTATING_OP: 0.49})
    decision, reason, name, version = policies.apply(
        raw_policy, record, [READ_ONLY_OP, MUTATING_OP]
    )
    assert decision == "propose"
    assert reason is None
    assert (name, version) == ("raw", "1")


def test_raw_policy_matches_gate_decide_with_bare_thresholds(raw_policy):
    record = _record({READ_ONLY_OP: 0.6, MUTATING_OP: 0.4})
    offered = [READ_ONLY_OP, MUTATING_OP]
    decision, reason, _name, _version = policies.apply(raw_policy, record, offered)
    expected = policies.gate.decide(record["candidates"], offered, policies.gate.Thresholds())
    assert (decision, reason) == (expected.outcome, expected.reason)


# ---------------------------------------------------------------------------
# shipped policy: read_only and mutating gates reported separately
# ---------------------------------------------------------------------------


def test_shipped_policy_abstains_on_a_thin_read_only_margin(shipped_policy):
    # margin 0.51 - 0.49 = 0.02 < the shipped read_only margin of 0.2.
    record = _record({READ_ONLY_OP: 0.51, "other_read_only_op": 0.49})
    decision, reason, name, version = policies.apply(
        shipped_policy, record, [READ_ONLY_OP, "other_read_only_op"]
    )
    assert decision == "abstain_uncertain"
    assert reason == "margin"
    assert (name, version) == ("scorer-r3b-shipped", "1")


def test_shipped_policy_proposes_a_mutating_op_with_the_same_thin_margin(shipped_policy):
    # Same thin margin, but the argmax is mutating: the shipped policy sets
    # no mutating threshold at all, so it must still propose.
    record = _record({MUTATING_OP: 0.51, "other_mutating_op": 0.49})
    decision, reason, _name, _version = policies.apply(
        shipped_policy, record, [MUTATING_OP, "other_mutating_op"]
    )
    assert decision == "propose"
    assert reason is None


def test_shipped_policy_rescales_before_gating(shipped_policy):
    # Sanity check that calibration actually ran: temperature > 1 flattens
    # a distribution, so a policy applied with a raw (unscaled) distribution
    # and one applied after manual scaling should agree.
    raw_candidates = {READ_ONLY_OP: 0.7, "other_read_only_op": 0.3}
    scaled = policies.calibration_fit.apply_scaling(raw_candidates, temperature=1.5366)
    assert scaled != raw_candidates  # scaling actually changed something
    decision, _reason, _name, _version = policies.apply(
        shipped_policy, _record(raw_candidates), [READ_ONLY_OP, "other_read_only_op"]
    )
    expected = policies.gate.decide(
        scaled,
        [READ_ONLY_OP, "other_read_only_op"],
        policies.gate.Thresholds.from_json({"read_only": {"margin": 0.2}, "mutating": {}}),
    )
    assert decision == expected.outcome


# ---------------------------------------------------------------------------
# mutating-strict-example: stricter mutating gate, independent of read_only
# ---------------------------------------------------------------------------


def test_mutating_strict_policy_abstains_on_a_mutating_op_the_shipped_policy_would_propose(
    mutating_strict_policy,
):
    # After the shared 1.5366 temperature scaling, top1 clears the strict
    # floor (0.6) but not the strict margin (0.3): the shipped policy sets
    # no mutating threshold at all and would propose the same record.
    record = _record({MUTATING_OP: 0.7, "other_mutating_op": 0.3})
    decision, reason, name, version = policies.apply(
        mutating_strict_policy, record, [MUTATING_OP, "other_mutating_op"]
    )
    assert decision == "abstain_uncertain"
    assert reason == "margin"
    assert (name, version) == ("mutating-strict-example", "1")


def test_mutating_strict_policy_keeps_the_shipped_read_only_margin(mutating_strict_policy):
    # read_only gate is identical to scorer-r3b-shipped: same thin margin
    # abstains for the same reason.
    record = _record({READ_ONLY_OP: 0.51, "other_read_only_op": 0.49})
    decision, reason, _name, _version = policies.apply(
        mutating_strict_policy, record, [READ_ONLY_OP, "other_read_only_op"]
    )
    assert decision == "abstain_uncertain"
    assert reason == "margin"


# ---------------------------------------------------------------------------
# version is always returned, for every outcome shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_fixture_name",
    ["raw_policy", "shipped_policy", "mutating_strict_policy"],
)
def test_apply_always_returns_the_policy_version(request, policy_fixture_name):
    policy = request.getfixturevalue(policy_fixture_name)
    record = _record({READ_ONLY_OP: 0.6, MUTATING_OP: 0.4})
    _decision, _reason, _name, version = policies.apply(policy, record, [READ_ONLY_OP, MUTATING_OP])
    assert version == policy["version"]
