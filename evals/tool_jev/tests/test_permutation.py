"""Tests for ``evals/tool_jev/permutation.py`` (task t23, issue #64/#53).

Fixtures under ``fixtures/permutation/`` are synthetic: same numeric shape as the real
issue-53 probe output (``permutation_probe.py``'s ``kind_report``/``run_probe``), no
real case/request text anywhere -- built by hand from the module docstring's
description of that shape, never copied from the private run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tool_jev import permutation

FIXTURES = Path(__file__).parent / "fixtures" / "permutation"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# resolve_split
# ---------------------------------------------------------------------------


def test_resolve_split_requires_explicit_split_when_probe_records_none():
    probe = _load("probe-r3b-synthetic-test.json")
    with pytest.raises(permutation.PermutationError, match="pass split="):
        permutation.resolve_split(probe)


def test_resolve_split_accepts_explicit_non_heldout_split():
    probe = _load("probe-r3b-synthetic-test.json")
    assert permutation.resolve_split(probe, split="test") == "test"


@pytest.mark.parametrize(
    "bad_split", ["held-out", "held_out", "heldout", "heldout-mc", "HELD-OUT", "Heldout-MC"]
)
def test_resolve_split_refuses_heldout_variants(bad_split):
    probe = _load("probe-r3b-synthetic-test.json")
    with pytest.raises(permutation.PermutationError, match="held-out"):
        permutation.resolve_split(probe, split=bad_split)


def test_resolve_split_prefers_probes_own_recorded_split_over_argument():
    probe = _load("probe-with-recorded-split.json")
    # The probe records "test"; a caller-passed value is ignored in favour of it.
    assert permutation.resolve_split(probe, split="whatever") == "test"


def test_resolve_split_refuses_probes_own_recorded_heldout_split():
    probe = _load("probe-with-recorded-split.json")
    probe = dict(probe, split="heldout")
    with pytest.raises(permutation.PermutationError, match="held-out"):
        permutation.resolve_split(probe, split="test")


# ---------------------------------------------------------------------------
# load_probe_json
# ---------------------------------------------------------------------------


def test_load_probe_json_reads_fixture():
    probe = permutation.load_probe_json(FIXTURES / "probe-r3b-synthetic-test.json")
    assert probe["per_entry"] == 10
    assert len(probe["kinds"]) == 5


def test_load_probe_json_refuses_missing_file(tmp_path):
    with pytest.raises(permutation.PermutationError, match="cannot read"):
        permutation.load_probe_json(tmp_path / "does-not-exist.json")


def test_load_probe_json_refuses_non_probe_shape(tmp_path):
    path = tmp_path / "not-a-probe.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(permutation.PermutationError, match="kinds"):
        permutation.load_probe_json(path)


# ---------------------------------------------------------------------------
# probe_to_entry
# ---------------------------------------------------------------------------


def test_probe_to_entry_shape_matches_report_expectations():
    probe = _load("probe-r3b-synthetic-test.json")
    entry = permutation.probe_to_entry(probe, split="test")
    assert entry["measurable"] is True
    assert entry["split"] == "test"
    assert entry["per_entry"] == 10
    assert entry["seed"] == "53"
    assert entry["entries"] == 6
    assert set(entry["kinds"]) == {"order", "letters", "subset", "paraphrase", "all"}


def test_probe_to_entry_kind_has_rate_and_ci_for_every_kind():
    probe = _load("probe-r3b-synthetic-test.json")
    entry = permutation.probe_to_entry(probe, split="test")
    for name in ("order", "letters", "subset", "paraphrase", "all"):
        kind = entry["kinds"][name]
        assert isinstance(kind["rate"], float)
        assert isinstance(kind["ci_low"], float)
        assert isinstance(kind["ci_high"], float)
        assert isinstance(kind["trials"], int)
        assert isinstance(kind["changes"], int)
        assert isinstance(kind["entries"], int)
        assert isinstance(kind["incomplete_rate"], float)


def test_probe_to_entry_drops_free_text_bootstrap_note():
    probe = _load("probe-r3b-synthetic-test.json")
    entry = permutation.probe_to_entry(probe, split="test")
    for kind in entry["kinds"].values():
        assert "bootstrap_note" not in kind
        assert "label_case" not in kind


def test_probe_to_entry_is_json_serializable():
    probe = _load("probe-r3b-synthetic-test.json")
    entry = permutation.probe_to_entry(probe, split="test")
    # report.py's markdown rendering does json.dumps(permutation, sort_keys=True) on
    # this exact value -- must round-trip cleanly.
    json.dumps(entry, sort_keys=True)


def test_probe_to_entry_refuses_missing_split():
    probe = _load("probe-r3b-synthetic-test.json")
    with pytest.raises(permutation.PermutationError):
        permutation.probe_to_entry(probe)


def test_probe_to_entry_refuses_heldout_split():
    probe = _load("probe-r3b-synthetic-test.json")
    with pytest.raises(permutation.PermutationError, match="held-out"):
        permutation.probe_to_entry(probe, split="heldout")


def test_probe_to_entry_refuses_probe_missing_kinds():
    with pytest.raises(permutation.PermutationError, match="kinds"):
        permutation.probe_to_entry({"per_entry": 10}, split="test")


def test_probe_to_entry_refuses_kind_missing_name():
    probe = {"kinds": [{"rate": 0.1}]}
    with pytest.raises(permutation.PermutationError, match="kind"):
        permutation.probe_to_entry(probe, split="test")


# ---------------------------------------------------------------------------
# not_measurable_entry (a3-heal: generative/tool-call, not a Track B scorer)
# ---------------------------------------------------------------------------


def test_not_measurable_entry_shape():
    reason = (
        "permutation_probe.py only probes a Track B one-position scorer; a3-heal is a "
        "Track A tool-call checkpoint with no candidate listing or letter map to permute"
    )
    entry = permutation.not_measurable_entry(reason)
    assert entry == {"measurable": False, "reason": reason}


def test_not_measurable_entry_requires_a_reason():
    with pytest.raises(permutation.PermutationError, match="reason"):
        permutation.not_measurable_entry("")


# ---------------------------------------------------------------------------
# permutation_key / build_permutation_file (report.py's on-disk shape)
# ---------------------------------------------------------------------------


def test_permutation_key_matches_report_convention():
    assert permutation.permutation_key("r3b", "raw") == "r3b__raw"


def test_build_permutation_file_assembles_report_ready_payload(tmp_path):
    probe = _load("probe-r3b-synthetic-test.json")
    r3b_entry = permutation.probe_to_entry(probe, split="test")
    a3_entry = permutation.not_measurable_entry(
        "permutation_probe.py has no seam for a Track A tool-call/generative checkpoint"
    )
    payload = permutation.build_permutation_file(
        {
            permutation.permutation_key("scorer-r3b", "raw"): r3b_entry,
            permutation.permutation_key("a3-heal", "raw"): a3_entry,
        }
    )
    assert payload["scorer-r3b__raw"]["measurable"] is True
    assert payload["a3-heal__raw"]["measurable"] is False
    assert "reason" in payload["a3-heal__raw"]

    # This is exactly the file evals/tool_jev/report.py's load_permutation reads back.
    out = tmp_path / "permutation.json"
    out.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == payload


def test_build_permutation_file_matches_report_load_permutation_round_trip(tmp_path):
    """Integration with report.py's own loader (no case text anywhere in the result)."""
    from evals.tool_jev import report

    probe = _load("probe-r3b-synthetic-test.json")
    entry = permutation.probe_to_entry(probe, split="test")
    payload = permutation.build_permutation_file(
        {permutation.permutation_key("scorer-r3b", "raw"): entry}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / report.PERMUTATION_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    loaded = report.load_permutation(run_dir)
    assert loaded == payload
    dumped = json.dumps(loaded, sort_keys=True)
    assert "request" not in dumped
    assert "prompt" not in dumped
