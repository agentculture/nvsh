"""The Tier 2 (LFM2.5) fine-tune train/val/test splitter: scripts/lfm-finetune/split.py.

Uses a small fixture corpus, not dev.json, so task t6 adding entries to
dev.json cannot break the pinned-split test here (part of #39).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/split.py"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_split", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(entry_id: str, expect: dict, kind: str = "explicit", source: str = "fixture") -> dict:
    return {
        "id": entry_id,
        "kind": kind,
        "text": f"fixture text for {entry_id}",
        "expect": expect,
        "source": source,
    }


def _operation_entry(entry_id: str) -> dict:
    return _entry(entry_id, {"operation": "thermal_stats", "args": {}})


def _escalate_entry(entry_id: str) -> dict:
    return _entry(entry_id, {"escalate": True})


def _explain_entry(entry_id: str) -> dict:
    return _entry(entry_id, {"explain": True})


def _fixture_entries() -> list[dict]:
    """12 operation + 6 escalate + 3 explain entries, ids sorted for readability."""
    entries = [_operation_entry(f"op{i:02d}") for i in range(12)]
    entries += [_escalate_entry(f"esc{i:02d}") for i in range(6)]
    entries += [_explain_entry(f"exp{i:02d}") for i in range(3)]
    return entries


def _fixture_corpus(tmp_path: Path, name: str = "fixture.json") -> Path:
    path = tmp_path / name
    payload = {"header": "Fixture corpus for split tests.", "entries": _fixture_entries()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _all_ids(sides: dict) -> list[str]:
    return [entry["id"] for side in sides.values() for entry in side]


def test_same_seed_gives_identical_split():
    module = _module()
    entries = _fixture_entries()
    sides_a, _ = module.stratified_split(entries, seed=module.DEFAULT_SEED)
    sides_b, _ = module.stratified_split(entries, seed=module.DEFAULT_SEED)
    assert sides_a == sides_b


def test_different_seed_can_give_a_different_split():
    module = _module()
    entries = _fixture_entries()
    sides_default, _ = module.stratified_split(entries, seed=module.DEFAULT_SEED)
    sides_other, _ = module.stratified_split(entries, seed=module.DEFAULT_SEED + 1)
    assert _all_ids(sides_default) != _all_ids(sides_other) or sides_default != sides_other


def test_no_source_id_on_two_sides():
    module = _module()
    sides, _ = module.stratified_split(_fixture_entries(), seed=module.DEFAULT_SEED)
    all_ids = _all_ids(sides)
    assert len(all_ids) == len(set(all_ids))
    assert sorted(all_ids) == sorted(entry["id"] for entry in _fixture_entries())


def test_every_expectation_kind_present_on_every_side():
    module = _module()
    sides, missing_kinds = module.stratified_split(_fixture_entries(), seed=module.DEFAULT_SEED)
    assert missing_kinds == []
    for name in module.SPLIT_NAMES:
        kinds_present = {module.expectation_kind(entry["expect"]) for entry in sides[name]}
        assert kinds_present == set(module.EXPECTATION_KINDS), (name, kinds_present)


def test_a_kind_absent_from_the_corpus_is_reported_not_a_crash():
    module = _module()
    entries = [_operation_entry(f"op{i:02d}") for i in range(10)]
    entries += [_escalate_entry(f"esc{i:02d}") for i in range(6)]
    sides, missing_kinds = module.stratified_split(entries, seed=module.DEFAULT_SEED)
    assert missing_kinds == ["explain"]
    for name in module.SPLIT_NAMES:
        assert all(module.expectation_kind(e["expect"]) != "explain" for e in sides[name])


def test_fractions_must_sum_to_one():
    module = _module()
    with pytest.raises(ValueError, match="sum to 1.0"):
        module.stratified_split(_fixture_entries(), fractions=(0.5, 0.5, 0.5))


def test_the_held_out_split_is_refused(tmp_path):
    module = _module()
    held_out = _fixture_corpus(tmp_path, name="held-out.json")
    with pytest.raises(ValueError, match="held-out"):
        module.build_splits(held_out)


def test_output_entries_carry_their_source_id():
    module = _module()
    sides, _ = module.stratified_split(_fixture_entries(), seed=module.DEFAULT_SEED)
    for side in sides.values():
        for entry in side:
            assert entry["source_id"] == entry["id"]


def test_pins_the_split_for_the_committed_seed():
    """Pins the exact per-side ids for DEFAULT_SEED against a fixed fixture
    corpus (not dev.json), so t6 adding entries to dev.json cannot break
    this test."""
    module = _module()
    sides, _ = module.stratified_split(_fixture_entries(), seed=module.DEFAULT_SEED)
    ids = {name: sorted(entry["id"] for entry in side) for name, side in sides.items()}
    assert ids == {
        "train": [
            "esc00",
            "esc03",
            "esc04",
            "esc05",
            "exp02",
            "op01",
            "op02",
            "op05",
            "op07",
            "op08",
            "op09",
            "op10",
            "op11",
        ],
        "val": ["esc02", "exp01", "op00", "op06"],
        "test": ["esc01", "exp00", "op03", "op04"],
    }


def test_main_writes_train_val_test_json(tmp_path):
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    out_dir = tmp_path / "out"
    exit_code = module.main(["--corpus", str(corpus), "--out-dir", str(out_dir)])
    assert exit_code == 0
    for name in module.SPLIT_NAMES:
        payload = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert "header" in payload
        assert all("source_id" in entry for entry in payload["entries"])


def test_main_writes_the_same_split_as_the_library_call(tmp_path):
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    out_dir = tmp_path / "out"
    module.main(["--corpus", str(corpus), "--out-dir", str(out_dir), "--seed", "7"])
    sides, _ = module.stratified_split(_fixture_entries(), seed=7)
    for name in module.SPLIT_NAMES:
        payload = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert [e["id"] for e in payload["entries"]] == [e["id"] for e in sides[name]]


def test_main_rejects_held_out_json(tmp_path, capsys):
    module = _module()
    held_out = _fixture_corpus(tmp_path, name="held-out.json")
    with pytest.raises(SystemExit):
        module.main(["--corpus", str(held_out), "--out-dir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert "held-out" in captured.err


def test_main_reports_a_missing_kind(tmp_path, capsys):
    module = _module()
    path = tmp_path / "no-explain.json"
    entries = [_operation_entry(f"op{i:02d}") for i in range(10)]
    entries += [_escalate_entry(f"esc{i:02d}") for i in range(6)]
    path.write_text(json.dumps({"header": "no explain", "entries": entries}), encoding="utf-8")
    module.main(["--corpus", str(path), "--out-dir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert "explain" in captured.out


def test_duplicate_ids_are_refused() -> None:
    split = _module()
    entries = [
        {"id": "a", "expect": {"escalate": True}},
        {"id": "a", "expect": {"explain": True}},
    ]
    with pytest.raises(ValueError, match="duplicate entry ids"):
        split.stratified_split(entries)


def test_a_kind_too_small_for_every_side_is_named() -> None:
    split = _module()
    entries = [{"id": f"o{i}", "expect": {"operation": "gpu_stats", "args": {}}} for i in range(9)]
    entries += [{"id": f"x{i}", "expect": {"explain": True}} for i in range(2)]
    sides, _ = split.stratified_split(entries)
    gaps = split.absent_from_sides(sides)
    assert gaps and all(kind == "explain" for kind, _ in gaps)
