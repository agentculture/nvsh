"""Tests for the benchmark corpus and runner (task t22).

Acceptance criteria covered:
- the corpus holds the prompts and failure cases from issues #30/#31 and
  scope s13/s16, split into dev and held-out files, each with an expected
  operation+arguments or an expected escalation.
- running the same corpus through the same fixture tier twice gives
  identical accuracy/escalation/calibration numbers; the results file
  records model file hashes, nvsh version, device, engine, CPU/GPU mode
  and concurrent load.
- results include accuracy, argument accuracy, false-mutating-pick count,
  escalation precision/recall, cold and warm latency, idle/peak/reserved
  memory, image size, and a pass/miss line per c20 target.
"""

from __future__ import annotations

import json

import pytest

from nvsh.agent.base import AgentContext, AgentRequest, RequestKind
from nvsh.cli import main
from nvsh.explain.catalog import ENTRIES
from nvsh.ops import ground as ops_ground
from nvsh.ops import table as ops_table
from nvsh.platform._model import Platform
from nvsh.tiers import bench as bench_mod
from nvsh.tiers.base import Decline, DeclineReason
from nvsh.tiers.fake import FakeTier
from nvsh.tiers.router import VerifierVerdict

_PLATFORM = Platform(kind="test")

#: A stand-in for ``nvsh.ops.ground.default_runner`` -- no subprocess is ever
#: spawned in this test file. Mirrors ``tests/test_tier_router.py``'s fixture.
_UNITS = "nginx.service loaded active running\nvllm.service loaded active running\n"


def _runner(argv: list[str], timeout: float) -> tuple[int, str]:
    if argv[:1] == ["systemctl"]:
        return (0, _UNITS)
    if argv[:1] == ["docker"]:
        return (0, "vllm\n")
    return (127, "")


@pytest.fixture(autouse=True)
def xdg_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _counter_clock():
    """A deterministic, injectable "clock": +1.0 per call, starting at 0.0."""
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += 1.0
        return state["t"]

    return clock


def _entry(entry_id, kind, text, expect, source="test"):
    return bench_mod.CorpusEntry(id=entry_id, kind=kind, text=text, expect=expect, source=source)


_ENTRIES = (
    _entry("e1", "explicit", "Show GPU usage", {"operation": "gpu_stats", "args": {}}),
    _entry(
        "e2",
        "explicit",
        "Restart vLLM",
        {"operation": "service_restart", "args": {"service": "vllm.service"}},
    ),
    _entry("e3", "explicit", "Why did vLLM crash?", {"escalate": True}),
    _entry(
        "e4",
        "failure",
        "systemctl restart vllm -> failed",
        {"operation": "service_status", "args": {"service": "vllm.service"}},
    ),
)


def _script_for_entries():
    """A FakeTier script matching ``_ENTRIES`` in order: e1 correct, e2 a
    wrong MUTATING pick (a different operation than expected), e3 no call
    (declines -> escalates), e4 unused (FAILURE requests never reach
    Tier 1 -- see ``TierRouter._order``)."""
    return [
        [{"name": "gpu_stats", "arguments": {}}],
        [{"name": "container_restart", "arguments": {"container": "vllm"}}],
        [],
    ]


def _run_bench(split="dev"):
    tier1 = FakeTier(_script_for_entries())
    return bench_mod.bench(
        _ENTRIES,
        split=split,
        tier1=tier1,
        platform=_PLATFORM,
        options=bench_mod.BenchOptions(
            clock=_counter_clock(),
            runner=_runner,
            nvsh_version="9.9.9",
            engine="fixture",
            mode="cpu",
            concurrent_load=(0.1, 0.2, 0.3),
            timestamp="2026-09-19T00:00:00+00:00",
        ),
    )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def test_shipped_dev_corpus_loads_with_no_problems():
    loaded = bench_mod.load_corpus(bench_mod.dev_corpus_path())
    assert loaded.problems == ()
    assert len(loaded.entries) > 0


def test_shipped_held_out_corpus_ships_empty_with_a_header():
    loaded = bench_mod.load_corpus(bench_mod.held_out_corpus_path())
    assert loaded.entries == ()
    assert loaded.problems == ()
    assert loaded.header is not None
    assert "operator" in loaded.header


def test_dev_corpus_covers_issue_30_issue_31_s13_and_s16():
    loaded = bench_mod.load_corpus(bench_mod.dev_corpus_path())
    sources = {entry.source for entry in loaded.entries}
    assert {"issue-30", "issue-31", "s13", "s16"} <= sources


def test_load_corpus_reports_bad_entries_without_raising(tmp_path):
    corpus_path = tmp_path / "bad.json"
    corpus_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "ok1",
                        "kind": "explicit",
                        "text": "show gpu",
                        "expect": {"operation": "gpu_stats", "args": {}},
                        "source": "test",
                    },
                    {
                        "id": "bad-op",
                        "kind": "explicit",
                        "text": "do the thing",
                        "expect": {"operation": "not_a_real_operation", "args": {}},
                        "source": "test",
                    },
                    {
                        "id": "bad-args",
                        "kind": "explicit",
                        "text": "restart it",
                        "expect": {"operation": "service_restart", "args": {}},
                        "source": "test",
                    },
                    {"id": "no-text-field", "kind": "explicit", "expect": {"escalate": True}},
                ]
            }
        )
    )
    loaded = bench_mod.load_corpus(corpus_path)
    assert [entry.id for entry in loaded.entries] == ["ok1"]
    assert len(loaded.problems) == 3


# ---------------------------------------------------------------------------
# Metrics, computed directly (no router involved)
# ---------------------------------------------------------------------------


def _item(entry, outcome, latency_ms=1.0):
    return bench_mod.ItemResult(entry=entry, outcome=outcome, latency_ms=latency_ms)


def test_operation_accuracy_counts_operation_and_argument_matches():
    correct_entry = _entry("a", "explicit", "x", {"operation": "gpu_stats", "args": {}})
    wrong_args_entry = _entry(
        "b", "explicit", "y", {"operation": "service_restart", "args": {"service": "vllm.service"}}
    )
    wrong_op_entry = _entry("c", "explicit", "z", {"operation": "disk_stats", "args": {}})

    items = [
        _item(
            correct_entry,
            bench_mod.TierOutcome(handled_by="needle", operation="gpu_stats", args={}),
        ),
        _item(
            wrong_args_entry,
            bench_mod.TierOutcome(
                handled_by="needle", operation="service_restart", args={"service": "nginx.service"}
            ),
        ),
        _item(wrong_op_entry, bench_mod.TierOutcome(escalated_to="agent")),
    ]
    accuracy = bench_mod.compute_operation_accuracy(items)
    assert accuracy["total"] == 3
    assert accuracy["operation_correct"] == 2
    assert accuracy["argument_correct"] == 1
    assert accuracy["accuracy"] == pytest.approx(2 / 3)
    assert accuracy["argument_accuracy"] == pytest.approx(1 / 3)


def test_false_mutating_pick_flags_a_mutating_operation_against_a_different_expectation():
    expect_escalate = _entry("a", "explicit", "danger", {"escalate": True})
    wrong_mutating_outcome = bench_mod.TierOutcome(
        handled_by="needle",
        operation="service_restart",
        args={"service": "nginx.service"},
        proposal=_fake_proposal("service_restart"),
    )
    readonly_outcome = bench_mod.TierOutcome(
        handled_by="needle", operation="gpu_stats", args={}, proposal=_fake_proposal("gpu_stats")
    )
    expect_readonly = _entry("b", "explicit", "gpu?", {"operation": "gpu_stats", "args": {}})

    items = [
        _item(expect_escalate, wrong_mutating_outcome),
        _item(expect_readonly, readonly_outcome),
    ]
    result = bench_mod.compute_false_mutating(items)
    assert result["count"] == 1
    assert result["ids"] == ["a"]
    assert result["without_interpretation_shown"] == []


def _fake_proposal(operation_name: str):
    from nvsh.agent.base import Proposal, ProposalKind

    return Proposal(
        command="x", rationale=f"needle read this as: {operation_name}", kind=ProposalKind.FIX
    )


def test_escalation_precision_and_recall():
    should_escalate_and_did = _entry("a", "explicit", "x", {"escalate": True})
    should_escalate_but_didnt = _entry("b", "explicit", "y", {"escalate": True})
    should_not_and_didnt = _entry("c", "explicit", "z", {"operation": "gpu_stats", "args": {}})
    should_not_but_did = _entry("d", "explicit", "w", {"operation": "disk_stats", "args": {}})

    items = [
        _item(should_escalate_and_did, bench_mod.TierOutcome(escalated_to="agent")),
        _item(
            should_escalate_but_didnt,
            bench_mod.TierOutcome(handled_by="needle", operation="gpu_stats"),
        ),
        _item(
            should_not_and_didnt, bench_mod.TierOutcome(handled_by="needle", operation="gpu_stats")
        ),
        _item(should_not_but_did, bench_mod.TierOutcome(escalated_to="agent")),
    ]
    escalation = bench_mod.compute_escalation(items)
    assert escalation == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
    }


def test_calibration_auc_separates_correct_from_wrong_picks():
    entries = [
        _entry(f"c{i}", "explicit", "x", {"operation": "gpu_stats", "args": {}}) for i in range(4)
    ]
    # Correct picks score high, wrong picks score low: perfect separation, AUC 1.0.
    outcomes = [
        bench_mod.TierOutcome(
            operation="gpu_stats", verifier=VerifierVerdict(calibrated=3.0), handled_by="needle"
        ),
        bench_mod.TierOutcome(
            operation="gpu_stats", verifier=VerifierVerdict(calibrated=2.0), handled_by="needle"
        ),
        bench_mod.TierOutcome(
            operation="disk_stats", verifier=VerifierVerdict(calibrated=-3.0), handled_by="needle"
        ),
        bench_mod.TierOutcome(
            operation="disk_stats", verifier=VerifierVerdict(calibrated=-2.0), handled_by="needle"
        ),
    ]
    items = [_item(entry, outcome) for entry, outcome in zip(entries, outcomes)]
    calibration = bench_mod.compute_calibration(items)
    assert calibration["samples"] == 4
    assert calibration["auc"] == pytest.approx(1.0)


def test_calibration_treats_a_wrong_argument_pick_as_incorrect():
    """A pick naming the right operation but the wrong argument (e.g. the
    wrong service) must count as a wrong pick in calibration/threshold
    samples, exactly like ``compute_operation_accuracy`` already does --
    matching the operation name alone would treat it as a positive."""
    entry = _entry(
        "a", "explicit", "restart it", {"operation": "service_restart", "args": {"service": "a"}}
    )
    outcome = bench_mod.TierOutcome(
        operation="service_restart",
        args={"service": "b"},  # wrong argument
        verifier=VerifierVerdict(calibrated=5.0),
        handled_by="needle",
    )
    samples = bench_mod._calibration_samples([_item(entry, outcome)])
    assert samples == [(5.0, False)]


def test_calibration_reports_not_enough_samples_rather_than_a_fake_score():
    calibration = bench_mod.compute_calibration([])
    assert calibration["auc"] is None
    assert calibration["samples"] == 0


def test_suggest_thresholds_only_meaningful_on_dev_split():
    result = _run_bench(split="held-out")
    assert result["threshold_suggestion"] == {
        "escalate_below": None,
        "ask_below": None,
        "youden_j": None,
        "note": "not the dev split",
    }


def test_latency_cold_is_first_call_warm_is_the_rest():
    items = [
        bench_mod.ItemResult(
            entry=_entry("a", "explicit", "x", {"escalate": True}), outcome=None, latency_ms=100.0
        ),
        bench_mod.ItemResult(
            entry=_entry("b", "explicit", "y", {"escalate": True}), outcome=None, latency_ms=10.0
        ),
        bench_mod.ItemResult(
            entry=_entry("c", "explicit", "z", {"escalate": True}), outcome=None, latency_ms=20.0
        ),
    ]
    latency = bench_mod.compute_latency(items)
    assert latency["cold_ms"] == 100.0
    assert latency["warm_median_ms"] == 15.0
    assert latency["warm_p95_ms"] == 20.0


def test_latency_empty_corpus_reports_none_not_zero():
    assert bench_mod.compute_latency([]) == {
        "cold_ms": None,
        "warm_median_ms": None,
        "warm_p95_ms": None,
    }


# ---------------------------------------------------------------------------
# Memory readers: injectable, optional, never crash
# ---------------------------------------------------------------------------


def test_read_proc_status_parses_vmrss_and_vmhwm():
    text = "Name:\tfoo\nVmRSS:\t  12345 kB\nVmHWM:\t  54321 kB\nOther:\tx\n"
    reading = bench_mod.read_proc_status(123, read_text=lambda _pid: text)
    assert reading.vm_rss_kb == 12345
    assert reading.vm_hwm_kb == 54321


def test_read_proc_status_returns_none_when_reader_raises():
    def _boom(_pid):
        raise OSError("no such process")

    assert bench_mod.read_proc_status(999, read_text=_boom) is None


def test_read_proc_status_returns_none_when_text_is_absent():
    assert bench_mod.read_proc_status(1, read_text=lambda _pid: None) is None


def test_read_docker_stats_parses_used_and_limit():
    reading = bench_mod.read_docker_stats("tier2", run_stats=lambda _c: "12.3MiB / 500MiB")
    assert reading.docker_used_mib == pytest.approx(12.3)
    assert reading.docker_limit_mib == pytest.approx(500.0)


def test_read_docker_stats_returns_none_for_unparsable_output():
    assert bench_mod.read_docker_stats("tier2", run_stats=lambda _c: "not a memusage line") is None


def test_read_docker_stats_returns_none_when_runner_raises():
    def _boom(_container):
        raise RuntimeError("docker not installed")

    assert bench_mod.read_docker_stats("tier2", run_stats=_boom) is None


def test_compute_memory_added_is_peak_minus_idle():
    idle = bench_mod.MemoryReading(vm_rss_kb=1024, vm_hwm_kb=2048)
    peak = bench_mod.MemoryReading(vm_rss_kb=3072, vm_hwm_kb=4096)
    memory = bench_mod.compute_memory(idle, peak)
    assert memory["idle_mib"] == pytest.approx(1.0)
    assert memory["peak_mib"] == pytest.approx(3.0)
    assert memory["reserved_mib"] == pytest.approx(4.0)
    assert memory["added_mib"] == pytest.approx(2.0)


def test_compute_memory_not_measured_without_readings():
    memory = bench_mod.compute_memory(None, None)
    assert memory == {"idle_mib": None, "peak_mib": None, "reserved_mib": None, "added_mib": None}


# ---------------------------------------------------------------------------
# Targets: pass/miss/not-measured per c20
# ---------------------------------------------------------------------------


def test_build_targets_reports_not_measured_when_nothing_was_measured():
    targets = bench_mod.build_targets(
        accuracy={"argument_accuracy": None},
        latency={"warm_p95_ms": None},
        escalation={"recall": None},
        false_mutating={"without_interpretation_shown": []},
        memory={"added_mib": None},
    )
    assert len(targets) == 5
    statuses = {t["target"]: t["status"] for t in targets}
    assert statuses["tier1_warm_p95_under_150ms"] == "not measured"
    assert statuses["added_memory_under_1gb"] == "not measured"
    # the mutating-interpretation check is always computable (0 flagged items).
    assert statuses["zero_wrong_mutating_without_interpretation"] == "pass"


def test_build_targets_passes_and_misses_on_real_numbers():
    targets = bench_mod.build_targets(
        accuracy={"argument_accuracy": 0.95},
        latency={"warm_p95_ms": 50.0},
        escalation={"recall": 0.5},
        false_mutating={"without_interpretation_shown": []},
        memory={"added_mib": 100.0},
    )
    statuses = {t["target"]: t["status"] for t in targets}
    assert statuses["tier1_warm_p95_under_150ms"] == "pass"
    assert statuses["correct_operation_and_arguments_ge_90pct"] == "pass"
    assert statuses["should_escalate_recall_ge_80pct"] == "miss"
    assert statuses["added_memory_under_1gb"] == "pass"


# ---------------------------------------------------------------------------
# bench(): drives the real router
# ---------------------------------------------------------------------------


def test_bench_drives_the_real_router_and_scores_each_outcome():
    result = _run_bench()
    accuracy = result["accuracy"]
    # e1, e2 and e4 all name an operation in ``expect`` (e3 expects an
    # escalation, so it is excluded from the operation-accuracy denominator).
    # e4 is a FAILURE request with no Tier 2 configured, so Tier 1 never
    # sees it (TierRouter._order) and it escalates with operation=None.
    assert accuracy["total"] == 3
    assert accuracy["operation_correct"] == 1  # only e1
    assert result["escalation"]["tp"] == 1  # e3 correctly escalated
    assert result["escalation"]["fp"] == 1  # e4 escalated but expected an operation
    assert result["false_mutating_pick"]["count"] == 1  # e2: container_restart, not service_restart


def test_bench_running_twice_with_a_fresh_fixture_tier_gives_identical_results():
    first = _run_bench()
    second = _run_bench()
    first_without_timestamp = {**first, "provenance": {**first["provenance"], "timestamp": None}}
    second_without_timestamp = {**second, "provenance": {**second["provenance"], "timestamp": None}}
    assert first_without_timestamp == second_without_timestamp


def test_bench_results_record_provenance():
    result = _run_bench()
    provenance = result["provenance"]
    assert provenance["nvsh_version"] == "9.9.9"
    assert provenance["device"] == {"kind": "test", "values": []}
    assert provenance["engine"] == "fixture"
    assert provenance["mode"] == "cpu"
    assert provenance["concurrent_load"] == [0.1, 0.2, 0.3]
    assert "model_hashes" in provenance
    assert "image_size_bytes" in provenance
    assert provenance["thresholds_in_force"] == {
        "min_confidence": 0.0,
        "ask_below": 0.0,
        "escalate_below": -2.0,
        "min_mass": 0.05,
    }


def test_bench_options_thresholds_and_min_confidence_reach_provenance():
    """S107: ``bench``'s many optional knobs were grouped into ``BenchOptions``
    (21 parameters -> a handful). This confirms the grouping still threads a
    non-default ``min_confidence``/``thresholds`` through to the router build
    and the reported provenance, not just the defaults ``_run_bench`` uses."""
    tier1 = FakeTier(_script_for_entries())
    result = bench_mod.bench(
        _ENTRIES,
        split="dev",
        tier1=tier1,
        platform=_PLATFORM,
        options=bench_mod.BenchOptions(
            clock=_counter_clock(),
            runner=_runner,
            min_confidence=0.42,
            thresholds=bench_mod.VerifierThresholds(
                ask_below=1.0, escalate_below=-1.0, min_mass=0.1
            ),
        ),
    )
    assert result["provenance"]["thresholds_in_force"] == {
        "min_confidence": 0.42,
        "ask_below": 1.0,
        "escalate_below": -1.0,
        "min_mass": 0.1,
    }


def test_bench_results_include_every_required_metric():
    result = _run_bench()
    assert set(result) == {
        "corpus",
        "accuracy",
        "items",
        "accuracy_by_kind",
        "escalation",
        "false_mutating_pick",
        "calibration",
        "threshold_suggestion",
        "latency",
        "memory",
        "targets",
        "provenance",
    }
    assert {"accuracy", "argument_accuracy"} <= set(result["accuracy"])
    assert {"precision", "recall"} <= set(result["escalation"])
    assert {"cold_ms", "warm_median_ms", "warm_p95_ms"} <= set(result["latency"])
    assert {"idle_mib", "peak_mib", "reserved_mib", "added_mib"} <= set(result["memory"])
    assert "image_size_bytes" in result["provenance"]
    assert len(result["targets"]) == 5
    for target in result["targets"]:
        assert target["status"] in ("pass", "miss", "not measured")


def test_bench_uses_a_throwaway_records_store_not_the_real_log(tmp_path):
    from nvsh.tiers.records import default_records_path

    _run_bench()
    assert not default_records_path().exists()


def test_bench_never_calls_needle_run():
    """Grep-style: the router chain never invokes an execute/run method on
    the tier -- only ``select()``. FakeTier has no ``run``/``execute`` at all,
    so this is enforced structurally."""
    assert not hasattr(FakeTier, "run")
    assert not hasattr(FakeTier, "execute")


# ---------------------------------------------------------------------------
# model hashes / image size
# ---------------------------------------------------------------------------


def test_model_hashes_from_pins():
    pins = {
        "needle3": {
            "weights": {"sha256": "wsha"},
            "engines": {"aarch64": {"sha256": "esha"}},
        }
    }
    hashes = bench_mod.model_hashes(pins, platform_tag="aarch64")
    assert hashes == {"weights_sha256": "wsha", "engine_sha256": "esha"}


def test_model_hashes_null_when_pins_absent():
    assert bench_mod.model_hashes(None) == {"weights_sha256": None, "engine_sha256": None}


def test_image_size_bytes_null_when_no_images_pinned():
    assert bench_mod.image_size_bytes({"images": []}) is None
    assert bench_mod.image_size_bytes(None) is None


def test_image_size_bytes_from_first_pinned_image():
    pins = {"images": [{"name": "tier2", "size_bytes": 12345}]}
    assert bench_mod.image_size_bytes(pins) == 12345


# ---------------------------------------------------------------------------
# CLI wiring: `nvsh tiers bench`
# ---------------------------------------------------------------------------


def test_cli_bench_default_fixture_tier_runs_end_to_end(capsys):
    rc = main(["tiers", "bench", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["corpus"]["split"] == "dev"
    assert payload["corpus"]["count"] > 0
    # UnavailableTier declines everything: every request escalates.
    assert payload["escalation"]["tp"] + payload["escalation"]["fn"] > 0


def test_cli_bench_held_out_reports_zero_entries(capsys):
    rc = main(["tiers", "bench", "--split", "held-out"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "held-out: 0 entries (operator has not added any)" in out


def test_cli_bench_writes_results_to_out_file(tmp_path, capsys):
    out_file = tmp_path / "results.json"
    rc = main(["tiers", "bench", "--out", str(out_file)])
    assert rc == 0
    written = json.loads(out_file.read_text())
    assert written["corpus"]["split"] == "dev"


def test_cli_bench_needle_tier_without_its_files_escalates_everything(
    capsys, tmp_path, monkeypatch
):
    # An empty cache: the tier has no weights or engine, so it declines each
    # item instead of crashing the bench (and never downloads anything).
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    rc = main(["tiers", "bench", "--tier", "needle", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["accuracy"]["operation_correct"] == 0


def test_cli_bench_unknown_tier_is_a_user_error():
    # argparse itself refuses this via `choices=`, one layer above CliError.
    with pytest.raises(SystemExit):
        main(["tiers", "bench", "--tier", "bogus"])


def test_explain_catalog_has_a_bench_entry():
    assert ("tiers", "bench") in ENTRIES
    assert "nvsh tiers bench" in ENTRIES[("tiers", "bench")]


def test_ops_table_still_has_every_operation_the_corpus_names():
    loaded = bench_mod.load_corpus(bench_mod.dev_corpus_path())
    for entry in loaded.entries:
        operation = entry.expect.get("operation")
        if operation is not None:
            assert ops_table.get(operation) is not None, operation


def test_unavailable_tier_declines_never_selects():
    tier = bench_mod.UnavailableTier()
    result = tier.select(AgentRequest(kind=RequestKind.EXPLICIT, prompt="x"), AgentContext())
    assert isinstance(result, Decline)
    assert result.reason == DeclineReason.TIER_UNAVAILABLE
    tier.close()  # never raises


def test_the_fixture_world_grounds_a_service_the_host_does_not_have():
    runner = bench_mod.world_runner({"services": ["vllm.service"]})
    grounded = ops_ground.ground(ops_table.get("service_restart"), {"service": "vllm"}, runner)
    assert grounded.args == {"service": "vllm.service"}


def test_the_fixture_world_runs_nothing_else():
    assert bench_mod.world_runner({})(["rm", "-rf", "/"], 1.0) == (127, "")


def test_the_fixture_platform_carries_the_corpus_device_cli():
    platform = bench_mod.world_platform({"platform": "jetson", "device_cli": "thor"})
    assert platform.kind == "jetson"
    assert platform.get("thor_cli").text == "thor"


def test_a_corpus_without_a_world_is_an_empty_world(tmp_path):
    assert bench_mod.load_world(tmp_path / "missing.json") == {}


def test_results_carry_one_row_per_corpus_entry():
    loaded = bench_mod.load_corpus(bench_mod.dev_corpus_path())
    result = bench_mod.bench(
        loaded.entries, split="dev", tier1=bench_mod.UnavailableTier(), platform=Platform("test")
    )
    assert [row["id"] for row in result["items"]] == [entry.id for entry in loaded.entries]


def test_accuracy_is_also_reported_per_request_kind():
    loaded = bench_mod.load_corpus(bench_mod.dev_corpus_path())
    result = bench_mod.bench(
        loaded.entries, split="dev", tier1=bench_mod.UnavailableTier(), platform=Platform("test")
    )
    assert sorted(result["accuracy_by_kind"]) == ["explicit", "failure"]
