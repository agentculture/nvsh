"""Tests for evals.tool_jev.cases -- synthetic fixtures only, no real case text."""

from __future__ import annotations

import json

import pytest

from evals.tool_jev import cases


def _write_split(tmp_path, name, entries, header="synthetic fixture split"):
    path = tmp_path / name
    path.write_text(json.dumps({"header": header, "entries": entries}), encoding="utf-8")
    return path


def _manifest(tmp_path, **paths):
    return {"splits": {tag: str(path) for tag, path in paths.items()}}


# ---------------------------------------------------------------------------
# criterion 1: held-out access and text redaction
# ---------------------------------------------------------------------------


def test_loading_heldout_without_include_heldout_raises(tmp_path):
    heldout = _write_split(
        tmp_path,
        "heldout.json",
        [{"id": "h1", "kind": "explicit", "text": "do the thing", "expect": {"escalate": True}}],
    )
    manifest = _manifest(tmp_path, heldout=heldout)

    with pytest.raises(cases.HeldOutAccessError):
        cases.load_case_set(manifest, "heldout")


def test_loading_heldout_with_include_heldout_never_returns_text(tmp_path):
    heldout = _write_split(
        tmp_path,
        "heldout.json",
        [
            {
                "id": "h1",
                "kind": "explicit",
                "text": "this exact sentence must never come back out",
                "expect": {"operation": "gpu_stats", "args": {}},
            }
        ],
    )
    manifest = _manifest(tmp_path, heldout=heldout)

    loaded = cases.load_case_set(manifest, "heldout", include_heldout=True)

    assert len(loaded) == 1
    case = loaded[0]
    assert case.id == "h1"
    assert case.text is None
    assert case.expect == {"operation": "gpu_stats", "args": {}}


def test_loading_heldout_mc_without_include_heldout_raises(tmp_path):
    heldout_mc = _write_split(
        tmp_path,
        "heldout-mc.json",
        [{"id": "h1-nocand", "kind": "explicit", "text": "x", "expect": {"escalate": True}}],
    )
    manifest = _manifest(tmp_path, **{"heldout-mc": heldout_mc})

    with pytest.raises(cases.HeldOutAccessError):
        cases.load_case_set(manifest, "heldout-mc")


def test_case_construction_refuses_text_on_heldout_split():
    with pytest.raises(ValueError):
        cases.Case(
            id="h1",
            split="heldout",
            text="should never be set",
            candidates=None,
            expect={"escalate": True},
            read_only=None,
        )


def test_unknown_split_tag_raises(tmp_path):
    manifest = _manifest(tmp_path, test=tmp_path / "unused.json")
    with pytest.raises(ValueError):
        cases.load_case_set(manifest, "nope")


# ---------------------------------------------------------------------------
# non-held-out loading (test / test-mc) round-trips text and fields
# ---------------------------------------------------------------------------


def test_loading_test_split_returns_text_and_fields(tmp_path):
    test_split = _write_split(
        tmp_path,
        "test.json",
        [
            {
                "id": "t1",
                "kind": "explicit",
                "text": "show me the gpu stats",
                "class": "imperative",
                "source": "synthetic",
                "expect": {"operation": "gpu_stats", "args": {}},
            }
        ],
    )
    manifest = _manifest(tmp_path, test=test_split)

    loaded = cases.load_case_set(manifest, "test")

    assert len(loaded) == 1
    case = loaded[0]
    assert case.id == "t1"
    assert case.split == "test"
    assert case.text == "show me the gpu stats"
    assert "imperative" in case.tags
    assert "synthetic" in case.tags
    assert case.candidates is None


def test_loading_test_mc_split_carries_candidates(tmp_path):
    test_mc = _write_split(
        tmp_path,
        "test-mc.json",
        [
            {
                "id": "t1-nocand",
                "kind": "explicit",
                "text": "show me the gpu stats",
                "source_id": "t1",
                "candidates": ["memory_stats", "disk_stats"],
                "expect": {"escalate": True},
            }
        ],
    )
    manifest = _manifest(tmp_path, **{"test-mc": test_mc})

    loaded = cases.load_case_set(manifest, "test-mc")

    assert len(loaded) == 1
    case = loaded[0]
    assert case.candidates == ("memory_stats", "disk_stats")
    assert case.source_id == "t1"
    assert "nocand" in case.tags
    assert case.expects_escalate is True
    assert case.read_only is None


def test_manifest_missing_split_raises_keyerror(tmp_path):
    manifest = _manifest(tmp_path, test=tmp_path / "unused.json")
    with pytest.raises(KeyError):
        cases.load_case_set(manifest, "test-mc")


def test_load_manifest_accepts_path_and_bare_mapping(tmp_path):
    test_split = _write_split(tmp_path, "test.json", [])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"splits": {"test": str(test_split)}}), encoding="utf-8")

    from_path = cases.load_manifest(manifest_path)
    assert from_path == {"test": str(test_split)}

    from_bare_mapping = cases.load_manifest({"test": str(test_split)})
    assert from_bare_mapping == {"test": str(test_split)}


# ---------------------------------------------------------------------------
# criterion 3: read_only/mutating comes from nvsh.ops.table, never the model
# ---------------------------------------------------------------------------


def test_read_only_operation_classified_true(tmp_path):
    split = _write_split(
        tmp_path,
        "test.json",
        [
            {
                "id": "t1",
                "kind": "explicit",
                "text": "how hot is the board",
                "expect": {"operation": "thermal_stats", "args": {}},
            }
        ],
    )
    manifest = _manifest(tmp_path, test=split)
    (case,) = cases.load_case_set(manifest, "test")
    assert case.read_only is True


def test_mutating_operation_classified_false(tmp_path):
    split = _write_split(
        tmp_path,
        "test.json",
        [
            {
                "id": "t1",
                "kind": "explicit",
                "text": "restart the nginx service",
                "expect": {"operation": "service_restart", "args": {"service": "nginx"}},
            }
        ],
    )
    manifest = _manifest(tmp_path, test=split)
    (case,) = cases.load_case_set(manifest, "test")
    assert case.read_only is False


def test_unknown_operation_is_conservatively_mutating(tmp_path):
    split = _write_split(
        tmp_path,
        "test.json",
        [
            {
                "id": "t1",
                "kind": "explicit",
                "text": "do something nvsh has never heard of",
                "expect": {"operation": "launch_the_nukes", "args": {}},
            }
        ],
    )
    manifest = _manifest(tmp_path, test=split)
    (case,) = cases.load_case_set(manifest, "test")
    assert case.read_only is False


def test_escalate_and_explain_expectations_have_no_read_only_flag(tmp_path):
    split = _write_split(
        tmp_path,
        "test.json",
        [
            {"id": "t1", "kind": "explicit", "text": "escalate me", "expect": {"escalate": True}},
            {"id": "t2", "kind": "explicit", "text": "explain me", "expect": {"explain": True}},
        ],
    )
    manifest = _manifest(tmp_path, test=split)
    loaded = {case.id: case for case in cases.load_case_set(manifest, "test")}
    assert loaded["t1"].read_only is None
    assert loaded["t1"].expects_escalate is True
    assert loaded["t2"].read_only is None
    assert loaded["t2"].expects_explain is True


# ---------------------------------------------------------------------------
# criterion 2: training_overlap refuses to score a checkpoint on its own ids
# ---------------------------------------------------------------------------


def test_training_overlap_raises_naming_overlapping_ids(tmp_path):
    train_split = _write_split(
        tmp_path,
        "train.json",
        [
            {"id": "a1", "kind": "explicit", "text": "x", "expect": {"escalate": True}},
            {"id": "a2", "kind": "explicit", "text": "y", "expect": {"escalate": True}},
        ],
    )

    with pytest.raises(cases.TrainingOverlapError) as excinfo:
        cases.training_overlap(["a2", "a3"], train_split)

    assert excinfo.value.overlapping_ids == ("a2",)
    assert "a2" in str(excinfo.value)


def test_training_overlap_returns_empty_when_disjoint(tmp_path):
    train_split = _write_split(
        tmp_path,
        "train.json",
        [{"id": "a1", "kind": "explicit", "text": "x", "expect": {"escalate": True}}],
    )

    assert cases.training_overlap(["b1", "b2"], train_split) == ()


def test_training_overlap_names_all_overlapping_ids_sorted(tmp_path):
    train_split = _write_split(
        tmp_path,
        "train.json",
        [
            {"id": "c3", "kind": "explicit", "text": "x", "expect": {"escalate": True}},
            {"id": "c1", "kind": "explicit", "text": "y", "expect": {"escalate": True}},
        ],
    )

    with pytest.raises(cases.TrainingOverlapError) as excinfo:
        cases.training_overlap(["c1", "c2", "c3"], train_split)

    assert excinfo.value.overlapping_ids == ("c1", "c3")
