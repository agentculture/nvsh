"""scripts/lfm-finetune/merge_variations.py (issue 39): variations folded into the train split."""

from __future__ import annotations

import importlib.util
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
    assert counts == {"kept": 1, "duplicate": 0}
    added = merged["entries"][-1]
    assert added["source_id"] == "g1" and "models" not in added
    assert merged["world"] == {"platform": "jetson"}
    assert "Split 'train' of " in merged["header"]


def test_a_repeat_of_the_source_or_an_earlier_variation_is_dropped() -> None:
    variations = [_variation("show gpu  LOAD"), _variation("GPU busy?"), _variation("gpu busy?")]
    _, counts = _module().merge(_split(), variations)
    assert counts == {"kept": 1, "duplicate": 2}


def test_a_non_train_split_is_refused() -> None:
    with pytest.raises(ValueError, match="train side"):
        _module().merge(_split("Split 'val' of dev.json (seed=39)."), [])


def test_a_variation_from_another_side_is_refused() -> None:
    with pytest.raises(ValueError, match="not train"):
        _module().merge(_split(), [_variation("GPU busy?", side="test")])


def test_a_variation_whose_source_is_missing_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the split"):
        _module().merge(_split(), [_variation("GPU busy?", source_id="g9")])


def test_a_variation_with_a_different_answer_is_refused() -> None:
    with pytest.raises(ValueError, match="differs"):
        _module().merge(_split(), [_variation("GPU busy?", expect={"escalate": True})])


def test_a_repeat_differing_only_in_punctuation_is_dropped() -> None:
    variations = [_variation("GPU busy?"), _variation("gpu busy")]
    _, counts = _module().merge(_split(), variations)
    assert counts == {"kept": 1, "duplicate": 1}


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
    assert added["source_id"] == "sup-01" and added["side"] == "train"
    assert "Split 'train' of " in merged["header"]


def test_a_supplement_not_marked_train_is_refused() -> None:
    with pytest.raises(ValueError, match="train side"):
        _module().add_supplement(_split(), _supplement("A supplement."))


def test_a_supplement_id_colliding_with_the_split_is_refused() -> None:
    supplement = _supplement()
    supplement["entries"][0]["id"] = "g1"
    with pytest.raises(ValueError, match="collides"):
        _module().add_supplement(_split(), supplement)


def test_the_committed_supplement_loads_and_is_train_only() -> None:
    import json as _json

    from nvsh.tiers.bench import load_corpus

    path = _SCRIPT.parent / "train-supplement.json"
    assert load_corpus(path).problems == ()
    merged, count = _module().add_supplement(_split(), _json.loads(path.read_text()))
    assert count == len(merged["entries"]) - 1
