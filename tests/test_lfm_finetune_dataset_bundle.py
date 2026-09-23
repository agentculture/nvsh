"""scripts/lfm-finetune/dataset_bundle.py (issue 46, t6): teacher models from the run.

ROLE_MODELS used to be a hard-coded dict of gateway alias -> (name, licence).
This module loads that table from a run's own ``--teacher-models`` file
instead, derives which alias played each of the four augment.py roles from
the run's own accepted records (never assumed positional order), stamps
``source``/``origin``/``teachers`` on every manifest record, and can refuse
to build a bundle that names a non-Apache teacher.
"""

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
    "REVIEWER_B": "cortex",
}

# The issue-46 run: reviewer B moved from Nemotron 3.5 Lightning (OpenMDW-1.1)
# to Qwen 3.8 27B (Apache-2.0), which is also the corrector ("cortex").
_ROLE_MODELS = {
    "worker": {"name": "Qwen 3.6 35B-A3B", "licence": "Apache-2.0"},
    "cortex": {"name": "Qwen 3.8 27B", "licence": "Apache-2.0"},
    "senses": {"name": "Gemma 4 26B-A4B", "licence": "Apache-2.0"},
}


def _module():
    spec = importlib.util.spec_from_file_location("dataset_bundle", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(entry_id: str, expect: dict, *, source: str = "issue-30", **extra) -> dict:
    return {
        "id": entry_id,
        "text": f"text {entry_id}",
        "expect": expect,
        "kind": "explicit",
        "source": source,
        **extra,
    }


def _role_models_file(tmp_path: Path, table: dict | None = None) -> Path:
    path = tmp_path / "role-models.json"
    path.write_text(json.dumps(_ROLE_MODELS if table is None else table))
    return path


def _inputs(
    tmp_path: Path,
    *,
    variation_models=True,
    test_extra=(),
    role_models: dict | None = None,
    apache_only: bool = False,
    models: dict | None = None,
) -> dict:
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
                        "sup-01",
                        {"escalate": True},
                        side="train",
                        source="supplement-2026-09-23",
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
        row["models"] = models if models is not None else _MODELS
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
        role_models=_module().load_role_models(_role_models_file(tmp_path, role_models)),
        out=tmp_path / "bundle",
        apache_only=apache_only,
    )


def test_the_bundle_holds_three_splits_a_manifest_and_a_card(tmp_path) -> None:
    counts = _module().build(**_inputs(tmp_path))
    out = tmp_path / "bundle"
    assert counts["train"] == 3
    assert counts["validation"] == 1
    assert counts["test"] == 1
    assert (counts["corpus"], counts["supplement"], counts["variation"]) == (1, 1, 1)
    manifest = {row["id"]: row for row in json.loads((out / "manifest.json").read_text())}
    assert manifest["dev-a~v1"]["origin"] == "variation"
    assert manifest["dev-a~v1"]["teachers"]["GENERATOR"] == "Qwen 3.6 35B-A3B"
    assert manifest["dev-a~v1"]["teachers"]["REVIEWER_B"] == "Qwen 3.8 27B"
    assert manifest["dev-a~v1"]["transformed"] is True
    assert manifest["dev-a~v1"]["source"] == "issue-30"
    assert manifest["sup-01"]["source_file"].endswith("train-supplement.json")
    assert manifest["sup-01"]["source"] == "supplement-2026-09-23"
    assert manifest["dev-t1"]["split"] == "test"
    # every record carries source, origin and a (possibly empty) teachers map
    assert all("source" in row for row in manifest.values())
    assert all("origin" in row for row in manifest.values())
    assert all("teachers" in row for row in manifest.values())
    assert manifest["dev-t1"]["teachers"] == {}
    card = (out / "README.md").read_text()
    assert "license: apache-2.0" in card
    assert "never trained on" in card
    assert "Qwen 3.8 27B" in card
    assert "Apache-2.0" in card
    assert "reviewer b is also the corrector" in card.lower()
    lines = (out / "data" / "train.jsonl").read_text().splitlines()
    assert json.loads(lines[2])["source_id"] == "dev-a"


def test_a_variation_without_its_models_is_refused(tmp_path) -> None:
    module, inputs = _module(), _inputs(tmp_path, variation_models=False)
    with pytest.raises(ValueError, match="no accepted record"):
        module.build(**inputs)


def test_a_variation_missing_one_role_is_refused(tmp_path) -> None:
    incomplete = {k: v for k, v in _MODELS.items() if k != "REVIEWER_B"}
    module, inputs = _module(), _inputs(tmp_path, models=incomplete)
    with pytest.raises(ValueError, match="REVIEWER_B"):
        module.build(**inputs)


def test_a_variation_outside_the_train_side_is_refused(tmp_path) -> None:
    extra = (_entry("dev-t1~v1", {"escalate": True}),)
    module, inputs = _module(), _inputs(tmp_path, test_extra=extra)
    with pytest.raises(ValueError, match="never leave the train side"):
        module.build(**inputs)


def test_the_held_out_split_is_refused(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    held = tmp_path / "held-out.json"
    held.write_text(json.dumps({"entries": []}))
    inputs["train_augmented"] = held
    module = _module()
    with pytest.raises(ValueError, match="held-out"):
        module.build(**inputs)


def test_a_non_apache_licence_is_refused(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    inputs["licence"].write_text("MIT License\n")
    module = _module()
    with pytest.raises(ValueError, match="Apache"):
        module.build(**inputs)


def test_a_train_record_repeating_a_test_entry_is_refused(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    train = json.loads(inputs["train_augmented"].read_text())
    train["entries"][0]["text"] = "TEXT dev-t1."
    inputs["train_augmented"].write_text(json.dumps(train))
    module = _module()
    with pytest.raises(ValueError, match="repeat a validation or test entry"):
        module.build(**inputs)


def test_a_record_with_no_source_field_is_refused(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    train = json.loads(inputs["train_augmented"].read_text())
    del train["entries"][0]["source"]
    inputs["train_augmented"].write_text(json.dumps(train))
    module = _module()
    with pytest.raises(ValueError, match="source"):
        module.build(**inputs)


def test_apache_only_refuses_a_non_apache_teacher(tmp_path) -> None:
    table = dict(_ROLE_MODELS)
    table["cortex"] = {"name": "Nemotron 3.5 Lightning", "licence": "OpenMDW-1.1"}
    module, inputs = _module(), _inputs(tmp_path, role_models=table, apache_only=True)
    with pytest.raises(ValueError, match="Apache"):
        module.build(**inputs)


def test_apache_only_allows_an_all_apache_run(tmp_path) -> None:
    inputs = _inputs(tmp_path, apache_only=True)
    counts = _module().build(**inputs)
    assert counts["variation"] == 1


def test_an_unknown_teacher_alias_is_refused(tmp_path) -> None:
    module, inputs = _module(), _inputs(tmp_path)
    inputs["role_models"] = {"worker": ("Qwen 3.6 35B-A3B", "Apache-2.0")}
    with pytest.raises(ValueError, match="cortex"):
        module.build(**inputs)


def test_load_role_models_rejects_a_malformed_entry(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"worker": {"name": "Qwen"}}))
    with pytest.raises(ValueError, match="licence"):
        _module().load_role_models(path)


def test_main_wires_the_teacher_models_flag_and_apache_only(tmp_path, capsys) -> None:
    inputs = _inputs(tmp_path)
    role_models_path = _role_models_file(tmp_path)
    argv = [
        "--splits",
        str(inputs["splits"]),
        "--train-augmented",
        str(inputs["train_augmented"]),
        "--accepted",
        str(inputs["accepted"]),
        "--rejected",
        str(inputs["rejected"]),
        "--licence",
        str(inputs["licence"]),
        "--teacher-models",
        str(role_models_path),
        "--apache-only",
        "--out",
        str(tmp_path / "bundle-cli"),
    ]
    assert _module().main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["variation"] == 1
