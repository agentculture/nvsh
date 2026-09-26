"""Manifest fields the runner (t17) needs: prices, output budgets, reasoning,
Track A grounding, stop rules, judging knobs, per-case-set predictions.

Synthetic inline TOML only; the committed example must keep parsing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from evals.tool_jev.manifest import (
    REASONING_LEVELS,
    JudgingConfig,
    ManifestError,
    StopRules,
    TrackAConfig,
    load_manifest,
    parse_manifest,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "manifest.example.toml"

_REF = """
[[reference]]
provider = "openrouter"
model = "vendor/model-a"
"""


def _parse(text: str):
    return parse_manifest(tomllib.loads(text))


def test_reference_defaults_keep_old_manifests_valid():
    ref = _parse(_REF).references[0]
    assert ref.reasoning == "medium"
    assert ref.max_output_tokens is None
    assert ref.usd_per_mtok_in == 0.0 and ref.usd_per_mtok_out == 0.0


def test_reference_prices_and_output_budget_parse():
    ref = _parse(
        _REF + "max_output_tokens = 2048\nusd_per_mtok_in = 1.25\nusd_per_mtok_out = 10\n"
    ).references[0]
    assert ref.max_output_tokens == 2048
    assert ref.usd_per_mtok_in == 1.25 and ref.usd_per_mtok_out == 10.0


@pytest.mark.parametrize("level", REASONING_LEVELS)
def test_reasoning_accepts_each_level(level):
    assert _parse(_REF + f'reasoning = "{level}"\n').references[0].reasoning == level


@pytest.mark.parametrize(
    "extra",
    [
        'reasoning = "extreme"\n',
        "max_output_tokens = 0\n",
        "max_output_tokens = true\n",
        "usd_per_mtok_in = -1\n",
        'usd_per_mtok_out = "free"\n',
    ],
)
def test_reference_rejects_bad_runner_fields(extra):
    with pytest.raises(ManifestError):
        _parse(_REF + extra)


def test_budget_batch_discount_and_rate_default_and_parse():
    m = _parse("[budget.openai]\nusd_cap = 1.0\nconcurrency_cap = 2\n")
    assert m.budgets[0].batch_discount == 0.5
    assert m.budgets[0].requests_per_minute is None
    m = _parse(
        "[budget.nvidia]\nusd_cap = 0\nconcurrency_cap = 2\n"
        "batch_discount = 0\nrequests_per_minute = 40\n"
    )
    assert m.budgets[0].batch_discount == 0.0
    assert m.budgets[0].requests_per_minute == 40.0
    assert m.budget_for("nvidia").concurrency_cap == 2
    assert m.budget_for("openai") is None


@pytest.mark.parametrize("extra", ["batch_discount = 1.5\n", "requests_per_minute = 0\n"])
def test_budget_rejects_bad_discount_or_rate(extra):
    with pytest.raises(ManifestError):
        _parse("[budget.openai]\nusd_cap = 1.0\nconcurrency_cap = 2\n" + extra)


def test_track_a_stops_and_judging_defaults():
    m = _parse("")
    assert m.track_a == TrackAConfig()
    assert m.track_a.snapshot is None and m.track_a.platform == "unknown"
    assert m.stops == StopRules(min_answers=5, max_truncated_share=0.3)
    assert m.judging == JudgingConfig(seed=64, max_output_tokens=1024, rubric="explain-v1")


def test_track_a_stops_and_judging_parse():
    m = _parse(
        '[track_a]\nsnapshot = "snapshots/ground.json"\nplatform = "jetson"\n'
        'device_cli = "jetson-cli"\n'
        "[stops]\nmin_answers = 3\nmax_truncated_share = 0.5\n"
        '[judging]\nseed = 7\nmax_output_tokens = 900\nrubric = "explain-v1"\n'
    )
    assert m.track_a == TrackAConfig(
        snapshot="snapshots/ground.json", platform="jetson", device_cli="jetson-cli"
    )
    assert m.stops == StopRules(min_answers=3, max_truncated_share=0.5)
    assert m.judging.seed == 7 and m.judging.max_output_tokens == 900


@pytest.mark.parametrize(
    "text",
    [
        '[track_a]\nsnapshot = "/abs/ground.json"\n',
        '[track_a]\nsnapshot = "~/ground.json"\n',
        "[stops]\nmin_answers = 0\n",
        "[stops]\nmax_truncated_share = 0\n",
        "[stops]\nmax_truncated_share = 1.5\n",
        "[judging]\nmax_output_tokens = -3\n",
        "[judging]\nseed = 1.5\n",
    ],
)
def test_runner_tables_reject_bad_values(text):
    with pytest.raises(ManifestError):
        _parse(text)


def test_run_entry_per_case_set_predictions_policies_and_artifact():
    m = _parse(
        '[[candidate]]\nname = "c1"\ntrack = "B"\npredictions_path = "p/c1.jsonl"\n'
        'policies = ["raw", "scorer-r3b-shipped"]\nrepo_id = "org/c1"\nrevision = "abc123"\n'
        "[candidate.predictions]\n"
        '"issue-53-missing-candidate" = "p/c1-mc.jsonl"\n'
        "[candidate.permutation_probes]\n"
        '"issue-53-test" = "p/probe.json"\n'
    )
    entry = m.candidates[0]
    assert entry.policies == ("raw", "scorer-r3b-shipped")
    assert entry.predictions_for("issue-53-missing-candidate") == "p/c1-mc.jsonl"
    assert entry.predictions_for("issue-53-test") == "p/c1.jsonl"
    assert entry.permutation_probes == {"issue-53-test": "p/probe.json"}
    assert (entry.repo_id, entry.revision) == ("org/c1", "abc123")
    assert m.plan_run()[0].policies == ("raw", "scorer-r3b-shipped")


def test_run_entry_defaults_to_raw_policy_only():
    m = _parse('[[baseline]]\nname = "b"\ntrack = "A"\npredictions_path = "p/b.jsonl"\n')
    assert m.baselines[0].policies == ("raw",)
    assert m.baselines[0].predictions == {}


@pytest.mark.parametrize(
    "extra",
    ["policies = []\n", "policies = [1]\n", "repo_id = 3\n", "predictions = 3\n"],
)
def test_run_entry_rejects_bad_runner_fields(extra):
    with pytest.raises(ManifestError):
        _parse('[[baseline]]\nname = "b"\ntrack = "A"\npredictions_path = "p"\n' + extra)


def test_example_manifest_carries_the_runner_fields():
    m = load_manifest(EXAMPLE)
    assert m.track_a.snapshot and not m.track_a.snapshot.startswith("/")
    assert m.stops.min_answers >= 1
    for ref in m.references:
        assert ref.reasoning in REASONING_LEVELS
    priced = [ref for ref in m.references if ref.provider in ("openai", "anthropic")]
    assert priced and all(ref.usd_per_mtok_out > 0 for ref in priced)
    assert all(ref.usd_per_mtok_in == 0 for ref in m.references if ref.provider == "local")
