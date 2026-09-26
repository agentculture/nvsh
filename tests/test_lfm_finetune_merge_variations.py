"""scripts/lfm-finetune/merge_variations.py (issue 39): variations folded into the train split."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "merge_variations.py"

_EXPECT = {"operation": "gpu_stats", "args": {}}


def _module():
    spec = importlib.util.spec_from_file_location("merge_variations", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(header: str = "Development split. Split 'train' of dev.json (seed=39).") -> dict:
    entry = {"id": "g1", "kind": "explicit", "text": "Show GPU load", "expect": _EXPECT}
    return {
        "header": header,
        "world": {"platform": "jetson"},
        "entries": [{**entry, "source_id": "g1"}],
    }


def _variation(text: str, **overrides) -> dict:
    base = {
        "id": "g1~v1",
        "source_id": "g1",
        "side": "train",
        "kind": "explicit",
        "text": text,
        "expect": _EXPECT,
        "seed_format": "split",
        "models": {"GENERATOR": "worker"},
    }
    return {**base, **overrides}


def test_variations_are_appended_with_their_source_id() -> None:
    merged, counts = _module().merge(_split(), [_variation("How busy is the GPU?")])
    assert counts == {"kept": 1, "duplicate": 0, "leaked": 0, "off_split": 0}
    added = merged["entries"][-1]
    assert added["source_id"] == "g1"
    assert "models" not in added
    assert merged["world"] == {"platform": "jetson"}
    assert "Split 'train' of " in merged["header"]


def test_a_repeat_of_the_source_or_an_earlier_variation_is_dropped() -> None:
    variations = [
        _variation("show gpu  LOAD", id="g1~v1"),
        _variation("GPU busy?", id="g1~v2"),
        _variation("gpu busy?", id="g1~v3"),
    ]
    _, counts = _module().merge(_split(), variations)
    assert counts == {"kept": 1, "duplicate": 2, "leaked": 0, "off_split": 0}


def test_a_non_train_split_is_refused() -> None:
    module = _module()
    split = _split("Split 'val' of dev.json (seed=39).")
    with pytest.raises(ValueError, match="train side"):
        module.merge(split, [])


def test_two_variations_with_the_same_id_are_refused_even_with_different_text() -> None:
    module = _module()
    variations = [_variation("How busy is the GPU?"), _variation("Show me GPU usage")]
    split = _split()
    with pytest.raises(ValueError, match="g1~v1"):
        module.merge(split, variations)


def test_a_variation_id_equal_to_a_split_entry_id_is_refused() -> None:
    module = _module()
    variations = [_variation("How busy is the GPU?", id="g1")]
    split = _split()
    with pytest.raises(ValueError, match="g1"):
        module.merge(split, variations)


def test_a_variation_from_another_side_is_refused() -> None:
    module, split, variations = _module(), _split(), [_variation("GPU busy?", side="test")]
    with pytest.raises(ValueError, match="not train"):
        module.merge(split, variations)


def test_a_variation_whose_source_is_missing_is_refused() -> None:
    module, split, variations = _module(), _split(), [_variation("GPU busy?", source_id="g9")]
    with pytest.raises(ValueError, match="not in the split"):
        module.merge(split, variations)


def test_a_variation_with_a_different_answer_is_refused() -> None:
    module = _module()
    split, variations = _split(), [_variation("GPU busy?", expect={"escalate": True})]
    with pytest.raises(ValueError, match="differs"):
        module.merge(split, variations)


def test_a_repeat_differing_only_in_punctuation_is_dropped() -> None:
    variations = [_variation("GPU busy?", id="g1~v1"), _variation("gpu busy", id="g1~v2")]
    _, counts = _module().merge(_split(), variations)
    assert counts == {"kept": 1, "duplicate": 1, "leaked": 0, "off_split": 0}


def _supplement(header: str = "Split 'train' of train-supplement.json.") -> dict:
    entry = {
        "id": "sup-01",
        "kind": "explicit",
        "text": "Stop the trainer",
        "expect": {"escalate": True},
    }
    return {"header": header, "entries": [entry]}


def test_a_supplement_is_appended_as_its_own_train_sources() -> None:
    merged, count = _module().add_supplement(_split(), _supplement())
    assert count == 1
    added = merged["entries"][-1]
    assert added["source_id"] == "sup-01"
    assert added["side"] == "train"
    assert "Split 'train' of " in merged["header"]


def test_a_supplement_not_marked_train_is_refused() -> None:
    module, split, supplement = _module(), _split(), _supplement("A supplement.")
    with pytest.raises(ValueError, match="train side"):
        module.add_supplement(split, supplement)


def test_a_supplement_id_colliding_with_the_split_is_refused() -> None:
    module, split, supplement = _module(), _split(), _supplement()
    supplement["entries"][0]["id"] = "g1"
    with pytest.raises(ValueError, match="collides"):
        module.add_supplement(split, supplement)


def test_the_committed_supplement_loads_and_is_train_only() -> None:
    import json as _json

    from nvsh.tiers.bench import load_corpus

    path = _SCRIPT.parent / "train-supplement.json"
    assert load_corpus(path).problems == ()
    merged, count = _module().add_supplement(_split(), _json.loads(path.read_text()))
    assert count == len(merged["entries"]) - 1


def _side(text: str) -> dict:
    return {"entries": [{"id": "t1", "text": text, "expect": {"escalate": True}}]}


def test_a_variation_repeating_a_validation_or_test_entry_is_dropped_as_leaked() -> None:
    module = _module()
    exclude = module.excluded_texts([_side("Is the GPU busy right now?")])
    variations = [
        _variation("is the GPU busy, right now", id="g1~v1"),
        _variation("GPU busy?", id="g1~v2"),
    ]
    merged, counts = module.merge(_split(), variations, exclude)
    assert counts == {"kept": 1, "duplicate": 0, "leaked": 1, "off_split": 0}
    assert merged["entries"][-1]["text"] == "GPU busy?"


def test_a_supplement_entry_repeating_a_test_entry_is_refused() -> None:
    module = _module()
    exclude = module.excluded_texts([_side("stop the TRAINER.")])
    split, supplement = _split(), _supplement()
    with pytest.raises(ValueError, match="excluded side"):
        module.add_supplement(split, supplement, exclude)


def test_the_committed_supplement_repeats_no_validation_or_test_entry() -> None:
    import json as _json

    corpus_path = Path(__file__).resolve().parents[1] / "nvsh" / "tiers" / "corpus" / "dev.json"
    corpus = _json.loads(corpus_path.read_text())
    entries = corpus["entries"] if isinstance(corpus, dict) else corpus
    sides, _ = _load("split").stratified_split(entries, seed=39)
    other = [{"entries": sides[name]} for name in ("val", "test")]
    module = _module()
    exclude = module.excluded_texts(other)
    supplement = _json.loads((_SCRIPT.parent / "train-supplement.json").read_text())
    merged, count = module.add_supplement(_split(), supplement, exclude)
    assert count == len(merged["entries"]) - 1


def test_filter_to_split_drops_and_counts_variations_whose_source_left_the_split() -> None:
    module = _module()
    split = _split()
    variations = [
        _variation("How busy is the GPU?", id="g1~v1"),  # source_id=g1, in the split
        _variation("GPU is hot?", id="g1~v2", source_id="g9"),  # source_id=g9, not in the split
    ]
    merged, counts = module.merge(split, variations, filter_to_split=True)
    # One kept (the valid variation)
    assert counts["kept"] == 1
    assert counts["off_split"] == 1
    assert counts["duplicate"] == 0
    assert counts["leaked"] == 0
    # Merged has one more entry than the original split
    assert len(merged["entries"]) == len(split["entries"]) + 1
    # The dropped variation's text does not appear
    merged_texts = {entry["text"] for entry in merged["entries"]}
    assert "GPU is hot?" not in merged_texts


def test_without_filter_an_off_split_variation_still_raises() -> None:
    module = _module()
    split = _split()
    variations = [
        _variation("How busy is the GPU?", id="g1~v1"),
        _variation("GPU is hot?", id="g1~v2", source_id="g9"),
    ]
    with pytest.raises(ValueError, match="is not in the split"):
        module.merge(split, variations, filter_to_split=False)


def test_main_prints_off_split_count(tmp_path, capsys) -> None:
    module = _module()
    # Write split file
    split_data = _split()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split_data), encoding="utf-8")
    # Write accepted variations: one valid, one off-split
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        json.dumps(_variation("How busy is the GPU?"))
        + "\n"
        + json.dumps(_variation("GPU is hot?", id="g1~v2", source_id="g9"))
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "out.json"
    module.main(
        [
            "--split",
            str(split_path),
            "--accepted",
            str(accepted_path),
            "--out",
            str(out_path),
            "--filter-to-split",
        ]
    )
    captured = capsys.readouterr()
    assert "kept=1" in captured.out
    assert "off_split=1" in captured.out


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPT.parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_decline_reason_class_survives_the_merge() -> None:
    """t14: reasons-mode gold depends on an escalate entry's `class` field
    (decline:<reason>) surviving from split.py's source entries into a
    merged train file untouched -- merge() already keeps every field but
    models/seed_format/verdicts, so this pins that behaviour."""
    split = _split()
    split["entries"][0]["class"] = "decline:repair"
    merged, _ = _module().merge(split, [])
    assert merged["entries"][0]["class"] == "decline:repair"


def test_a_supplement_entry_keeps_its_own_source_id() -> None:
    supplement = _supplement()
    supplement["entries"][0]["source_id"] = "g1"
    merged, _ = _module().add_supplement(_split(), supplement)
    assert merged["entries"][-1]["source_id"] == "g1"
