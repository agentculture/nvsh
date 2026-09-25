"""split.py v2 output read end to end by every downstream reader (issue 53).

split.py's v2 mode once wrote ``header`` as a JSON object. Every reader
downstream of it -- the trainers, the dataset builder, the variation merge,
measure.py, the calibration and probe refusals, quantize, augment and the
dataset bundle -- finds a split's side (and seed) in split.py's v1 note,
``Split '<side>' of <corpus> (seed=N).``, inside a *string* header. So a v2
train side was refused for training, and the test side's refusal leaned on
whatever ``json.dumps`` of the dict happened to contain (every side's
``sizes`` named ``test``, so even train looked like the test side).

These tests build one small v2 split with split.py itself and hand each side
to each reader. Every side is also copied to a neutral file name, so the
header alone -- not a ``train.json``/``test.json`` file name -- has to carry
the side.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from nvsh.tiers import bench as tier_bench

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts" / "lfm-finetune"
_WORLD = json.loads((_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json").read_text())["world"]
_SEED = 39
_FOLD_SEED = 7
_VERSION = "v2"
_OPS = ("thermal_stats", "gpu_stats", "disk_stats", "memory_stats")


def _load(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    names = (
        "split",
        "train_scorer",
        "build_dataset",
        "merge_variations",
        "measure",
        "calibration_fit",
        "permutation_probe",
        "quantize",
        "augment",
        "dataset_bundle",
    )
    return {name: _load(f"lfm_v2_readers_{name}", f"{name}.py") for name in names}


def _entry(entry_id: str, text: str, expect: dict, **extra) -> dict:
    return {"id": entry_id, "kind": "explicit", "text": text, "expect": expect, **extra}


def _corpus_entries() -> list[dict]:
    """12 operation + 6 escalate + 3 explain entries, all valid for load_corpus."""
    entries = [
        _entry(f"op{i:02d}", f"show {_OPS[i % 4].replace('_', ' ')} please ({i})", _op(i))
        for i in range(12)
    ]
    entries += [
        _entry(f"esc{i:02d}", f"reflash the bootloader, attempt {i}", {"escalate": True})
        for i in range(6)
    ]
    entries += [
        _entry(f"exp{i:02d}", f"what does nvpmodel do ({i})?", _explain()) for i in range(3)
    ]
    return entries


def _op(index: int) -> dict:
    return {"operation": _OPS[index % 4], "args": {}}


def _explain() -> dict:
    return {"explain": True, "answer": "It sets the power mode."}


@pytest.fixture(scope="module")
def v2_split(modules, tmp_path_factory) -> dict[str, Path]:
    """split.py's v2 sides, plus a neutral-name copy of each (``<side>-copy`` keys)."""
    base = tmp_path_factory.mktemp("v2split")
    corpus = base / "drafted.json"
    payload = {"header": "Fixture corpus.", "entries": _corpus_entries(), "world": _WORLD}
    corpus.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = base / "corpus-v2"
    status = modules["split"].main(
        [
            "--corpus",
            str(corpus),
            "--version",
            _VERSION,
            "--seed",
            str(_SEED),
            "--val-size",
            "4",
            "--test-size",
            "4",
            "--fold-seed",
            str(_FOLD_SEED),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert status == 0
    paths: dict[str, Path] = {}
    neutral = base / "neutral"
    neutral.mkdir()
    for name, alias in (("train", "fold-a"), ("val", "fold-b"), ("test", "fold-c")):
        paths[name] = out_dir / f"{name}.json"
        paths[f"{name}-copy"] = neutral / f"{alias}.json"
        shutil.copyfile(paths[name], paths[f"{name}-copy"])
    return paths


def _raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _held_out_marked(tmp_path: Path, v2_split: dict[str, Path]) -> Path:
    """The val side's entries under a held-out header, at a neutral name."""
    raw = _raw(v2_split["val"])
    raw["header"] = "Held-out split of the fixture corpus, for acceptance only."
    path = tmp_path / "fold-d.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# -- the header itself --


def test_every_v2_side_header_is_a_string_carrying_the_v1_note(v2_split) -> None:
    for side in ("train", "val", "test"):
        header = _raw(v2_split[side])["header"]
        assert isinstance(header, str)
        assert f"Split '{side}' of corpus-{_VERSION} (seed={_SEED})." in header


def test_nvsh_load_corpus_still_loads_a_v2_side(v2_split) -> None:
    for side in ("train", "val", "test"):
        loaded = tier_bench.load_corpus(v2_split[side])
        assert loaded.problems == ()
        assert len(loaded.entries) == len(_raw(v2_split[side])["entries"])


# -- train_scorer --


def test_train_scorer_reads_train_and_val_and_refuses_the_swap(modules, v2_split) -> None:
    train_scorer = modules["train_scorer"]
    for key in ("train", "train-copy"):
        assert train_scorer.read_split(v2_split[key], "train")
    for key in ("val", "val-copy"):
        assert train_scorer.read_split(v2_split[key], "val")
    with pytest.raises(ValueError, match="expected 'train'"):
        train_scorer.read_split(v2_split["val-copy"], "train")
    with pytest.raises(ValueError, match="expected 'val'"):
        train_scorer.read_split(v2_split["train-copy"], "val")
    with pytest.raises(ValueError, match="expected 'val'"):
        train_scorer.read_split(v2_split["test-copy"], "val")


# -- build_dataset --


def test_build_dataset_accepts_the_v2_train_side_and_refuses_val(modules, v2_split) -> None:
    build_dataset = modules["build_dataset"]
    examples, entries = build_dataset.build_with_scorer_entries(
        v2_split["train-copy"], is_split=True
    )
    assert len(examples) == len(_raw(v2_split["train"])["entries"]) == len(entries)
    for key in ("val-copy", "test-copy"):
        with pytest.raises(ValueError, match="only the 'train' side"):
            build_dataset.build_with_scorer_entries(v2_split[key], is_split=True)


def test_build_dataset_scorer_out_keeps_the_train_note(modules, v2_split, tmp_path) -> None:
    build_dataset = modules["build_dataset"]
    out = tmp_path / "nvsh-train.jsonl"
    scorer_out = tmp_path / "scorer-train.json"
    status = build_dataset.main(
        [
            "--split",
            str(v2_split["train-copy"]),
            "--out",
            str(out),
            "--no-verify-render",
            "--scorer-out",
            str(scorer_out),
        ]
    )
    assert status == 0
    header = _raw(scorer_out)["header"]
    assert f"Split 'train' of corpus-{_VERSION} (seed={_SEED})." in header
    # ...and the trainer reads the file build_dataset wrote.
    assert modules["train_scorer"].read_split(scorer_out, "train")


# -- merge_variations --


def test_merge_variations_accepts_the_v2_train_side_only(modules, v2_split) -> None:
    merge_variations = modules["merge_variations"]
    train = _raw(v2_split["train"])
    merged, counts = merge_variations.merge(train, [])
    assert counts["kept"] == 0
    assert "Split 'train' of " in merged["header"]
    with pytest.raises(ValueError, match="train side"):
        merge_variations.merge(_raw(v2_split["val"]), [])


# -- measure --


def test_measure_reads_side_and_seed_from_v2_headers(modules, v2_split) -> None:
    measure = modules["measure"]
    for side in ("train", "val", "test"):
        header = _raw(v2_split[side])["header"]
        assert measure.sides_from_header(header) == {side}
        assert measure.seed_from_header(header) == _SEED


def test_measure_refuses_the_v2_test_side_without_final(modules, v2_split) -> None:
    measure = modules["measure"]
    measure.check_split_allowed(v2_split["val-copy"], acceptance=False, final=False)
    with pytest.raises(measure.MeasureError, match="test side"):
        measure.check_split_allowed(v2_split["test-copy"], acceptance=False, final=False)
    measure.check_split_allowed(v2_split["test-copy"], acceptance=False, final=True)


# -- calibration_fit --


def test_calibration_fit_refuses_only_the_v2_test_side(modules, v2_split, tmp_path) -> None:
    calibration_fit = modules["calibration_fit"]
    for key in ("train-copy", "val-copy"):
        calibration_fit.refuse_if_test_or_held_out(v2_split[key], _raw(v2_split[key])["header"])
    with pytest.raises(calibration_fit.CalibrationError, match="test"):
        calibration_fit.refuse_if_test_or_held_out(
            v2_split["test-copy"], _raw(v2_split["test-copy"])["header"]
        )
    held_out = _held_out_marked(tmp_path, v2_split)
    with pytest.raises(calibration_fit.CalibrationError, match="held-out"):
        calibration_fit.refuse_if_test_or_held_out(held_out, _raw(held_out)["header"])


def test_calibration_fit_folds_cli_takes_val_refuses_test(modules, v2_split, tmp_path) -> None:
    calibration_fit = modules["calibration_fit"]
    out = tmp_path / "folds.json"
    common = ["--seed", "1", "--out", str(out)]
    assert calibration_fit.main(["folds", "--split", str(v2_split["val-copy"]), *common]) == 0
    assert calibration_fit.main(["folds", "--split", str(v2_split["test-copy"]), *common]) == 1


# -- permutation_probe --


def test_permutation_probe_refuses_the_v2_test_side_without_final(
    modules, v2_split, tmp_path, capsys
) -> None:
    probe = modules["permutation_probe"]
    assert probe.main(["--split", str(v2_split["test-copy"])]) == 1
    assert "refusing to probe" in capsys.readouterr().err
    held_out = _held_out_marked(tmp_path, v2_split)
    assert probe.main(["--split", str(held_out)]) == 1
    assert "refusing to probe" in capsys.readouterr().err
    # The val side passes the guard and stops only for want of a model.
    assert probe.main(["--split", str(v2_split["val-copy"])]) == 1
    err = capsys.readouterr().err
    assert "refusing" not in err and "--model is required" in err


# -- quantize, augment, dataset_bundle --


def test_quantize_verifies_each_v2_side_by_its_header(modules, v2_split) -> None:
    quantize = modules["quantize"]
    for side in ("train", "val", "test"):
        path = v2_split[f"{side}-copy"]
        quantize._verify_split_side(path, _raw(path), side)
    with pytest.raises(quantize.QuantizeError, match="expected 'train'"):
        path = v2_split["val-copy"]
        quantize._verify_split_side(path, _raw(path), "train")


def test_augment_infers_each_v2_side_from_its_header(modules, v2_split) -> None:
    augment = modules["augment"]
    for side in ("train", "val", "test"):
        path = v2_split[f"{side}-copy"]
        assert augment._infer_side(path, _raw(path)["header"]) == side
        seeds = augment.load_seeds(path)
        assert {seed.side for seed in seeds} == {side}


def test_dataset_bundle_reads_the_v2_split_seed(modules, v2_split) -> None:
    for side in ("train", "val", "test"):
        assert modules["dataset_bundle"].split_seed(v2_split[f"{side}-copy"]) == _SEED
