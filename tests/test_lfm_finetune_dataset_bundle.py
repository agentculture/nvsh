"""scripts/lfm-finetune/dataset_bundle.py (issue 39, t17): the nvsh-ops data set folder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "dataset_bundle.py"

_MODELS = {
    "GENERATOR": "worker",
    "CORRECTOR": "cortex",
    "REVIEWER_A": "senses",
    "REVIEWER_B": "associate",
}


def _module():
    spec = importlib.util.spec_from_file_location("dataset_bundle", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(entry_id: str, expect: dict, **extra) -> dict:
    return {
        "id": entry_id,
        "text": f"text {entry_id}",
        "expect": expect,
        "kind": "explicit",
        **extra,
    }


def _inputs(tmp_path: Path, *, variation_models=True, test_extra=()) -> dict:
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "val.json").write_text(
        json.dumps({"entries": [_entry("dev-v1", {"escalate": True})]})
    )
    (splits / "test.json").write_text(
        json.dumps({"entries": [_entry("dev-t1", {"explain": True, "answer": "a"}), *test_extra]})
    )
    train = tmp_path / "train-augmented.json"
    train.write_text(
        json.dumps(
            {
                "entries": [
                    _entry("dev-a", {"operation": "gpu_stats", "args": {}}, side="train"),
                    _entry(
                        "sup-01", {"escalate": True}, side="train", source="supplement-2026-09-23"
                    ),
                    _entry(
                        "dev-a~v1",
                        {"operation": "gpu_stats", "args": {}},
                        side="train",
                        source_id="dev-a",
                    ),
                ]
            }
        )
    )
    accepted = tmp_path / "accepted.jsonl"
    row = {"id": "dev-a~v1"}
    if variation_models:
        row["models"] = _MODELS
    accepted.write_text(json.dumps(row) + "\n")
    rejected = tmp_path / "rejected.jsonl"
    rejected.write_text(json.dumps({"id": "dev-a~v2"}) + "\n")
    licence = tmp_path / "LICENSE"
    licence.write_text("                                 Apache License\n")
    return dict(
        splits=splits,
        train_augmented=train,
        accepted=accepted,
        rejected=rejected,
        licence=licence,
        out=tmp_path / "bundle",
    )


def test_the_bundle_holds_three_splits_a_manifest_and_a_card(tmp_path) -> None:
    counts = _module().build(**_inputs(tmp_path))
    out = tmp_path / "bundle"
    assert counts["train"] == 3 and counts["validation"] == 1 and counts["test"] == 1
    assert (counts["corpus"], counts["supplement"], counts["variation"]) == (1, 1, 1)
    manifest = {row["id"]: row for row in json.loads((out / "manifest.json").read_text())}
    assert manifest["dev-a~v1"]["origin"] == "variation"
    assert manifest["dev-a~v1"]["models"]["GENERATOR"] == "Qwen 3.6 35B-A3B"
    assert manifest["dev-a~v1"]["transformed"] is True
    assert manifest["sup-01"]["source_file"].endswith("train-supplement.json")
    assert manifest["dev-t1"]["split"] == "test"
    card = (out / "README.md").read_text()
    assert "license: apache-2.0" in card and "never trained on" in card
    assert (
        "Of 2 reviewed rewrites, 1\n  were accepted (50%)" in card
        or "Of 2 reviewed rewrites, 1" in card
    )
    assert "Nemotron 3.5 Lightning" in card and "CC-BY-4.0" in card
    lines = (out / "data" / "train.jsonl").read_text().splitlines()
    assert json.loads(lines[2])["source_id"] == "dev-a"


def test_a_variation_without_its_models_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="no accepted record"):
        _module().build(**_inputs(tmp_path, variation_models=False))


def test_a_variation_outside_the_train_side_is_refused(tmp_path) -> None:
    extra = (_entry("dev-t1~v1", {"escalate": True}),)
    with pytest.raises(ValueError, match="never leave the train side"):
        _module().build(**_inputs(tmp_path, test_extra=extra))


def test_the_held_out_split_is_refused(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    held = tmp_path / "held-out.json"
    held.write_text(json.dumps({"entries": []}))
    inputs["train_augmented"] = held
    with pytest.raises(ValueError, match="held-out"):
        _module().build(**inputs)


def test_a_non_apache_licence_is_refused(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    inputs["licence"].write_text("MIT License\n")
    with pytest.raises(ValueError, match="Apache"):
        _module().build(**inputs)
