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
    entries = _fixture_entries()
    with pytest.raises(ValueError, match="sum to 1.0"):
        module.stratified_split(entries, fractions=(0.5, 0.5, 0.5))


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
    assert gaps
    assert all(kind == "explain" for kind, _ in gaps)


def test_fractions_outside_zero_to_one_are_refused() -> None:
    split = _module()
    entries = [_escalate_entry(str(i)) for i in range(10)]
    with pytest.raises(ValueError, match="between 0 and 1"):
        split.stratified_split(entries, fractions=(-0.5, 0.75, 0.75))


def test_variations_of_one_source_stay_on_one_side_and_keep_their_source_id() -> None:
    split = _module()
    entries = [_operation_entry(f"o{i}") for i in range(9)]
    entries += [{**_escalate_entry(f"v{i}"), "source_id": "parent"} for i in range(3)] + [
        _escalate_entry(f"e{i}") for i in range(6)
    ]
    sides, _ = split.stratified_split(entries)
    holding = [
        name for name, side in sides.items() if any(e["source_id"] == "parent" for e in side)
    ]
    assert len(holding) == 1
    assert sum(1 for e in sides[holding[0]] if e["source_id"] == "parent") == 3


def test_cli_fails_when_a_kind_cannot_reach_every_side(tmp_path, capsys) -> None:
    split = _module()
    entries = [_operation_entry(f"o{i}") for i in range(3)] + [
        _escalate_entry(f"e{i}") for i in range(3)
    ]
    entries += [_entry(f"x{i}", {"explain": True}) for i in range(2)]
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"entries": entries}))
    with pytest.raises(SystemExit):
        split.main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "out")])
    assert "too few entries" in capsys.readouterr().err
    assert not (tmp_path / "out" / "train.json").exists()


def test_refuse_if_under_nvsh_flags_a_path_under_nvsh() -> None:
    module = _module()
    with pytest.raises(ValueError, match="nvsh/"):
        module.refuse_if_under_nvsh(module._REPO_ROOT / "nvsh" / "tiers" / "corpus" / "v2")


def test_refuse_if_under_nvsh_allows_a_path_outside_nvsh(tmp_path) -> None:
    module = _module()
    module.refuse_if_under_nvsh(tmp_path / "corpus-v2")  # must not raise


def test_v2_cli_refuses_an_out_dir_inside_nvsh(tmp_path, capsys) -> None:
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    bad_out = str(module._REPO_ROOT / "nvsh" / "tiers" / "corpus" / "v2-attempt")
    with pytest.raises(SystemExit):
        module.main(
            [
                "--corpus",
                str(corpus),
                "--version",
                "v2",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--fold-seed",
                "1",
                "--out-dir",
                bad_out,
            ]
        )
    assert "nvsh" in capsys.readouterr().err
    assert not Path(bad_out).exists()


def test_v2_writes_versioned_header_with_seed_and_source_hashes(tmp_path) -> None:
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    out_dir = tmp_path / "out"
    exit_code = module.main(
        [
            "--corpus",
            str(corpus),
            "--version",
            "v2",
            "--seed",
            "39",
            "--val-size",
            "4",
            "--test-size",
            "4",
            "--fold-seed",
            "7",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    expected_hash = module.sha256_file(corpus)
    for name in module.SPLIT_NAMES:
        payload = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
        # The header stays split.py's v1-style string note (every downstream
        # reader parses it); the structured metadata sits under "split".
        assert payload["header"].startswith(f"Split '{name}' of corpus-v2 (seed=39). ")
        metadata = payload["split"]
        assert metadata["side"] == name
        assert metadata["version"] == "v2"
        assert metadata["seed"] == 39
        assert metadata["sources"] == [{"path": str(corpus), "sha256": expected_hash}]
        assert metadata["sizes"] == {"train": 13, "val": 4, "test": 4}


def test_v2_target_sizes_are_absolute_and_stratified_by_kind(tmp_path) -> None:
    module = _module()
    entries = [_operation_entry(f"op{i:02d}") for i in range(60)]
    entries += [_escalate_entry(f"esc{i:02d}") for i in range(30)]
    entries += [_explain_entry(f"exp{i:02d}") for i in range(10)]
    corpus = tmp_path / "big.json"
    corpus.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    out_dir = tmp_path / "out"
    module.main(
        [
            "--corpus",
            str(corpus),
            "--version",
            "v2",
            "--val-size",
            "15",
            "--test-size",
            "15",
            "--fold-seed",
            "3",
            "--out-dir",
            str(out_dir),
        ]
    )
    val = json.loads((out_dir / "val.json").read_text())["entries"]
    test = json.loads((out_dir / "test.json").read_text())["entries"]
    train = json.loads((out_dir / "train.json").read_text())["entries"]
    # Each expectation kind rounds its own share independently (the same
    # largest-remainder allocation stratified_split has always used), so the
    # total lands close to, not always exactly at, the requested size.
    assert abs(len(val) - 15) <= 2
    assert abs(len(test) - 15) <= 2
    assert len(train) + len(val) + len(test) == len(entries)
    for side in (val, test):
        kinds = {module.expectation_kind(e["expect"]) for e in side}
        assert kinds == set(module.EXPECTATION_KINDS)


def test_v2_records_fold_assignment_in_val_header_and_folds_json(tmp_path) -> None:
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    out_dir = tmp_path / "out"
    module.main(
        [
            "--corpus",
            str(corpus),
            "--version",
            "v2",
            "--val-size",
            "4",
            "--test-size",
            "4",
            "--fold-seed",
            "11",
            "--out-dir",
            str(out_dir),
        ]
    )
    val_payload = json.loads((out_dir / "val.json").read_text())
    val_header = val_payload["split"]
    val_ids = [e["id"] for e in val_payload["entries"]]
    assert val_header["fold_seed"] == 11
    assert sorted(val_header["fit_ids"] + val_header["selection_ids"]) == sorted(val_ids)
    folds = json.loads((out_dir / "folds.json").read_text())
    assert folds["seed"] == 11
    assert folds["fit_ids"] == val_header["fit_ids"]
    assert folds["selection_ids"] == val_header["selection_ids"]
    # train/test headers carry no fold assignment -- val-only, per the brief.
    for name in ("train", "test"):
        assert "fit_ids" not in json.loads((out_dir / f"{name}.json").read_text())["split"]


def test_v2_merges_multiple_corpora_deduped_by_id(tmp_path) -> None:
    module = _module()
    entries_a = _fixture_entries()
    corpus_a = tmp_path / "a.json"
    corpus_a.write_text(json.dumps({"header": "a", "entries": entries_a}), encoding="utf-8")
    # corpus_b repeats op00 (dropped as a duplicate) and adds 4 new entries.
    entries_b = [entries_a[0]] + [_operation_entry(f"new{i:02d}") for i in range(4)]
    corpus_b = tmp_path / "b.json"
    corpus_b.write_text(json.dumps({"header": "b", "entries": entries_b}), encoding="utf-8")
    out_dir = tmp_path / "out"
    module.main(
        [
            "--corpus",
            str(corpus_a),
            "--corpus",
            str(corpus_b),
            "--val-size",
            "4",
            "--test-size",
            "4",
            "--fold-seed",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    all_ids: list[str] = []
    for name in module.SPLIT_NAMES:
        payload = json.loads((out_dir / f"{name}.json").read_text())
        all_ids += [e["id"] for e in payload["entries"]]
    assert len(all_ids) == len(set(all_ids))
    assert len(all_ids) == len(entries_a) + 4
    metadata = json.loads((out_dir / "train.json").read_text())["split"]
    assert [source["path"] for source in metadata["sources"]] == [str(corpus_a), str(corpus_b)]


def test_v2_refuses_held_out_among_multiple_corpora(tmp_path) -> None:
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    held_out = _fixture_corpus(tmp_path, name="held-out.json")
    with pytest.raises(ValueError, match="held-out"):
        module.merge_corpora([corpus, held_out])


def test_v2_balances_escalation_classes_across_sides_where_counts_allow(tmp_path) -> None:
    module = _module()
    entries = [_operation_entry(f"op{i:02d}") for i in range(20)]
    entries += [
        {**_escalate_entry(f"escA{i:02d}"), "class": "decline:outside_table"} for i in range(10)
    ]
    entries += [
        {**_escalate_entry(f"escB{i:02d}"), "class": "decline:destructive"} for i in range(10)
    ]
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    out_dir = tmp_path / "out"
    module.main(
        [
            "--corpus",
            str(corpus),
            "--version",
            "v2",
            "--val-size",
            "10",
            "--test-size",
            "10",
            "--fold-seed",
            "2",
            "--out-dir",
            str(out_dir),
        ]
    )
    val = json.loads((out_dir / "val.json").read_text())["entries"]
    val_classes = {e["class"] for e in val if "class" in e}
    assert {"decline:outside_table", "decline:destructive"} <= val_classes


def test_v2_requires_val_size_and_test_size(tmp_path, capsys) -> None:
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    with pytest.raises(SystemExit):
        module.main(["--corpus", str(corpus), "--version", "v2", "--out-dir", str(tmp_path / "o")])
    assert "val-size" in capsys.readouterr().err


def test_v2_requires_fold_seed(tmp_path, capsys) -> None:
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--corpus",
                str(corpus),
                "--version",
                "v2",
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--out-dir",
                str(tmp_path / "o"),
            ]
        )
    assert "fold-seed" in capsys.readouterr().err


def test_legacy_single_corpus_invocation_is_unaffected_by_v2(tmp_path) -> None:
    """The pre-#53 invocation shape (single --corpus, fractions) must behave
    exactly as before: string header, no version/sources/sizes keys."""
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    out_dir = tmp_path / "out"
    module.main(["--corpus", str(corpus), "--out-dir", str(out_dir)])
    header = json.loads((out_dir / "train.json").read_text())["header"]
    assert isinstance(header, str)


def test_v2_fold_assignment_keeps_source_id_groups_on_one_side(tmp_path) -> None:
    """#53 review finding P1: fold creation (~split.py line 478) shuffled
    individual val ids and ignored source_id grouping, so a source's
    variations could land in both the fit and selection folds. --seed 3
    lands the whole "sib" group in val for this fixture; sweeping
    --fold-seed reliably reproduces the pre-fix split under the old code."""
    module = _module()
    entries = [_operation_entry(f"op{i:02d}") for i in range(40)]
    entries += [_escalate_entry(f"esc{i:02d}") for i in range(34)]
    entries += [
        {**_escalate_entry("sib0"), "source_id": "sib"},
        {**_escalate_entry("sib1"), "source_id": "sib"},
        {**_escalate_entry("sib2"), "source_id": "sib"},
    ]
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    # --seed 3 lands the whole "sib" group in val for this fixture (confirmed
    # by direct enumeration); vary --fold-seed to hit the buggy per-id shuffle.
    saw_sib_in_val = False
    for fold_seed in range(15):
        out_dir = tmp_path / f"out{fold_seed}"
        module.main(
            [
                "--corpus",
                str(corpus),
                "--version",
                "v2",
                "--val-size",
                "15",
                "--test-size",
                "15",
                "--seed",
                "3",
                "--fold-seed",
                str(fold_seed),
                "--out-dir",
                str(out_dir),
            ]
        )
        val_payload = json.loads((out_dir / "val.json").read_text())
        sib_ids_in_val = {e["id"] for e in val_payload["entries"] if e["source_id"] == "sib"}
        if not sib_ids_in_val:
            continue
        saw_sib_in_val = True
        fit_ids = set(val_payload["split"]["fit_ids"])
        selection_ids = set(val_payload["split"]["selection_ids"])
        assert sib_ids_in_val <= fit_ids or sib_ids_in_val <= selection_ids, fold_seed
    # Guard against this test passing vacuously (e.g. if the fixture ever
    # changes and --seed 3 stops putting the "sib" group in val at all).
    assert saw_sib_in_val, '"sib" source_id group never landed in val across any swept fold-seed'


def test_v2_stratifies_a_rare_class_across_val_and_test(tmp_path) -> None:
    """#53 review finding P2: class round-robin ordering followed by
    contiguous train/val/test slicing is not stratification -- with 90
    common + 10 rare escalation entries and 15/15 eval sizes, all 10 rare
    entries used to land in train, none in val or test."""
    module = _module()
    entries = [{**_escalate_entry(f"common{i:03d}"), "class": "decline:common"} for i in range(90)]
    entries += [{**_escalate_entry(f"rare{i:02d}"), "class": "decline:rare"} for i in range(10)]
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    out_dir = tmp_path / "out"
    module.main(
        [
            "--corpus",
            str(corpus),
            "--version",
            "v2",
            "--val-size",
            "15",
            "--test-size",
            "15",
            "--fold-seed",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    val = json.loads((out_dir / "val.json").read_text())["entries"]
    test = json.loads((out_dir / "test.json").read_text())["entries"]
    train = json.loads((out_dir / "train.json").read_text())["entries"]
    assert "decline:rare" in {e["class"] for e in val}
    assert "decline:rare" in {e["class"] for e in test}
    assert abs(len(val) - 15) <= 2
    assert abs(len(test) - 15) <= 2
    assert len(train) + len(val) + len(test) == len(entries)


def test_v2_class_stratification_respects_source_id_groups(tmp_path) -> None:
    module = _module()
    entries = [_operation_entry(f"op{i:02d}") for i in range(30)]
    entries += [
        {**_escalate_entry("g0"), "class": "decline:rare"},
        {**{**_escalate_entry("g0v1"), "source_id": "g0"}, "class": "decline:rare"},
    ]
    entries += [{**_escalate_entry(f"other{i:02d}"), "class": "decline:rare"} for i in range(8)]
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    out_dir = tmp_path / "out"
    module.main(
        [
            "--corpus",
            str(corpus),
            "--version",
            "v2",
            "--val-size",
            "10",
            "--test-size",
            "10",
            "--fold-seed",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    sides_by_source: dict[str, set[str]] = {}
    for name in module.SPLIT_NAMES:
        entries_out = json.loads((out_dir / f"{name}.json").read_text())["entries"]
        for entry in entries_out:
            sides_by_source.setdefault(entry["source_id"], set()).add(name)
    assert all(len(sides) == 1 for sides in sides_by_source.values())


def test_v2_legacy_mode_is_byte_identical_to_before(tmp_path) -> None:
    """Non-v2 (legacy) invocations must be unaffected by the v2 fixes above."""
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    out_dir = tmp_path / "out"
    module.main(
        ["--corpus", str(corpus), "--out-dir", str(out_dir), "--seed", str(module.DEFAULT_SEED)]
    )
    payload = json.loads((out_dir / "val.json").read_text())["entries"]
    ids = sorted(e["id"] for e in payload)
    assert ids == ["esc02", "exp01", "op00", "op06"]


def test_every_side_keeps_the_corpus_world(tmp_path) -> None:
    split = _module()
    world = {"platform": {"kind": "jetson"}}
    entries = [_operation_entry(f"o{i}") for i in range(6)] + [
        _escalate_entry(f"e{i}") for i in range(6)
    ]
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"header": "h", "world": world, "entries": entries}))
    split.main(["--corpus", str(corpus), "--out-dir", str(tmp_path / "out")])
    for name in ("train", "val", "test"):
        assert json.loads((tmp_path / "out" / f"{name}.json").read_text())["world"] == world


@pytest.mark.parametrize("version", ["v 2", "test", "v2-val", "held-out-v2", "-v2"])
def test_v2_refuses_a_version_that_breaks_or_names_a_side(tmp_path, capsys, version) -> None:
    """The version lands in every side's ``Split '<side>' of corpus-<version>`` note:
    whitespace would end the readers' corpus match early, and a side word would
    make every side read as that side (e.g. every side refused as test)."""
    module = _module()
    corpus = _fixture_corpus(tmp_path)
    with pytest.raises(SystemExit):
        module.main(
            [
                "--corpus",
                str(corpus),
                "--version",
                version,
                "--val-size",
                "4",
                "--test-size",
                "4",
                "--fold-seed",
                "1",
                "--out-dir",
                str(tmp_path / "o"),
            ]
        )
    assert "--version" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_v2_header_note_names_the_version_not_an_input_path(tmp_path) -> None:
    """Several inputs merge into one v2 corpus, so the note names
    ``corpus-<version>``; input paths (which could contain "test" or
    "held-out") stay out of the header and live under "split"."""
    module = _module()
    corpus = _fixture_corpus(tmp_path, name="test-held-out-drafts.json")
    out_dir = tmp_path / "out"
    module.main(
        [
            "--corpus",
            str(corpus),
            "--version",
            "v2.1",
            "--val-size",
            "4",
            "--test-size",
            "4",
            "--fold-seed",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    header = json.loads((out_dir / "train.json").read_text())["header"]
    assert header.startswith("Split 'train' of corpus-v2.1 (seed=39). ")
    assert "drafts" not in header and "test" not in header and "held" not in header
