"""Post-hoc calibration: scripts/lfm-finetune/calibration_fit.py.

Covers seeded fit/selection folds (disjoint, reproducible), temperature and
vector scaling fit/apply (including recovering a known temperature on
synthetic data), and refusing a test/held-out file as fit input.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/calibration_fit.py"


@pytest.fixture(scope="module")
def calibration_fit():
    spec = importlib.util.spec_from_file_location("lfm_calibration_fit", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lfm_calibration_fit"] = module
    spec.loader.exec_module(module)
    return module


def _split_file(tmp_path: Path, name: str, ids: list[str]) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "header": "development split.",
                "entries": [{"id": entry_id, "expect": {"operation": "op"}} for entry_id in ids],
            }
        ),
        encoding="utf-8",
    )
    return path


def _prediction_line(
    entry_id: str, expected: dict, candidates: dict | None, outcome: str = "propose"
) -> dict:
    row = {
        "id": entry_id,
        "expected": expected,
        "outcome": outcome if candidates is not None else "invalid",
        "operation": None,
        "arguments": None,
        "candidates": candidates,
        "tokens": 1,
        "ttfd_ms": 1.0,
        "latency_ms": 1.0,
    }
    if outcome == "propose" and candidates is not None:
        row["operation"] = expected.get("operation", "noop")
        row["arguments"] = {}
    return row


def _write_predictions(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def test_make_folds_is_seeded_disjoint_and_sorted(calibration_fit):
    ids = [f"id{n}" for n in range(20)]
    fit_a, selection_a = calibration_fit.make_folds(ids, seed=7)
    fit_b, selection_b = calibration_fit.make_folds(ids, seed=7)
    assert fit_a == fit_b
    assert selection_a == selection_b
    assert set(fit_a) & set(selection_a) == set()
    assert set(fit_a) | set(selection_a) == set(ids)
    assert fit_a == sorted(fit_a)
    assert selection_a == sorted(selection_a)


def test_make_folds_different_seed_can_differ(calibration_fit):
    ids = [f"id{n}" for n in range(30)]
    fit_a, _ = calibration_fit.make_folds(ids, seed=1)
    fit_b, _ = calibration_fit.make_folds(ids, seed=2)
    assert fit_a != fit_b


def test_make_folds_dedupes_ids(calibration_fit):
    fit_ids, selection_ids = calibration_fit.make_folds(["a", "a", "b", "c"], seed=3)
    assert set(fit_ids) | set(selection_ids) == {"a", "b", "c"}


def test_read_split_ids(calibration_fit, tmp_path):
    path = _split_file(tmp_path, "val.json", ["a", "b", "c"])
    assert calibration_fit.read_split_ids(path) == ["a", "b", "c"]


def test_cmd_folds_writes_seed_and_disjoint_lists(calibration_fit, tmp_path):
    split_path = _split_file(tmp_path, "val.json", [f"id{n}" for n in range(10)])
    out_path = tmp_path / "folds.json"
    rc = calibration_fit.main(
        ["folds", "--split", str(split_path), "--seed", "39", "--out", str(out_path)]
    )
    assert rc == 0
    folds = json.loads(out_path.read_text(encoding="utf-8"))
    assert folds["seed"] == 39
    assert set(folds["fit_ids"]) & set(folds["selection_ids"]) == set()
    assert set(folds["fit_ids"]) | set(folds["selection_ids"]) == {f"id{n}" for n in range(10)}


def test_cmd_folds_refuses_test_split(calibration_fit, tmp_path):
    split_path = _split_file(tmp_path, "test.json", ["a", "b"])
    out_path = tmp_path / "folds.json"
    rc = calibration_fit.main(
        ["folds", "--split", str(split_path), "--seed", "1", "--out", str(out_path)]
    )
    assert rc == 1
    assert not out_path.exists()


def test_cmd_folds_refuses_held_out_split(calibration_fit, tmp_path):
    split_path = _split_file(tmp_path, "held-out.json", ["a", "b"])
    out_path = tmp_path / "folds.json"
    rc = calibration_fit.main(
        ["folds", "--split", str(split_path), "--seed", "1", "--out", str(out_path)]
    )
    assert rc == 1
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# Scaling primitives
# ---------------------------------------------------------------------------


def test_temperature_scale_identity_at_t1(calibration_fit):
    candidates = {"a": 0.7, "b": 0.3}
    scaled = calibration_fit.temperature_scale(candidates, 1.0)
    assert scaled["a"] == pytest.approx(0.7, abs=1e-9)
    assert scaled["b"] == pytest.approx(0.3, abs=1e-9)


def test_temperature_scale_sums_to_one(calibration_fit):
    candidates = {"a": 0.9, "b": 0.05, "c": 0.05}
    for t in (0.3, 1.0, 3.0):
        scaled = calibration_fit.temperature_scale(candidates, t)
        assert math.isclose(sum(scaled.values()), 1.0, abs_tol=1e-9)


def test_temperature_scale_above_one_flattens_confidence(calibration_fit):
    candidates = {"a": 0.9, "b": 0.1}
    scaled = calibration_fit.temperature_scale(candidates, 4.0)
    assert scaled["a"] < 0.9
    assert scaled["a"] > scaled["b"]


def test_temperature_scale_rejects_non_positive(calibration_fit):
    with pytest.raises(ValueError):
        calibration_fit.temperature_scale({"a": 1.0}, 0.0)


def test_vector_scale_sums_to_one_and_reweights(calibration_fit):
    candidates = {"a": 0.5, "b": 0.5}
    scaled = calibration_fit.vector_scale(candidates, {"a": 3.0, "b": 1.0})
    assert math.isclose(sum(scaled.values()), 1.0, abs_tol=1e-9)
    assert scaled["a"] > scaled["b"]


def test_vector_scale_default_scale_is_one(calibration_fit):
    candidates = {"a": 0.6, "b": 0.4}
    scaled = calibration_fit.vector_scale(candidates, {})
    assert scaled["a"] == pytest.approx(0.6, abs=1e-9)
    assert scaled["b"] == pytest.approx(0.4, abs=1e-9)


# ---------------------------------------------------------------------------
# Fitting temperature on synthetic data
# ---------------------------------------------------------------------------


def _synthetic_rows(calibration_fit, true_temperature: float, n: int = 100) -> list:
    """Rows whose gold frequency exactly matches a well-calibrated ``true_dist``.

    ``distorted`` is what a model would report if its raw confidence needed
    temperature ``true_temperature`` to be corrected back to ``true_dist``
    (``temperature_scale(distorted, true_temperature) == true_dist`` exactly,
    since temperature scaling composes: ``temperature_scale(temperature_scale(p,
    1/T), T) == p``). Every row shares the same distribution, and the golds
    are assigned so their empirical frequency is exactly ``true_dist`` --
    the maximum-likelihood calibration is then exactly ``true_temperature``.
    """
    true_dist = {"a": 0.8, "b": 0.2}
    distorted = calibration_fit.temperature_scale(true_dist, 1.0 / true_temperature)
    n_a = round(n * true_dist["a"])
    rows = [(distorted, "a")] * n_a + [(distorted, "b")] * (n - n_a)
    return rows


@pytest.mark.parametrize("true_temperature", [0.5, 1.0, 2.0, 4.0])
def test_fit_temperature_recovers_known_temperature(calibration_fit, true_temperature):
    rows = _synthetic_rows(calibration_fit, true_temperature)
    fitted = calibration_fit.fit_temperature(rows)
    assert fitted == pytest.approx(true_temperature, rel=0.05)


def test_fit_temperature_on_empty_rows_is_one(calibration_fit):
    assert calibration_fit.fit_temperature([]) == 1.0


def test_fit_vector_default_is_identity_when_already_matched(calibration_fit):
    # Every row's top candidate is already the gold label: the NLL-minimising
    # vector should not need to move far from the all-ones starting point.
    candidates = {"a": 0.9, "b": 0.1}
    rows = [(candidates, "a")] * 20
    vector = calibration_fit.fit_vector(rows, ["a", "b"])
    assert vector["a"] >= vector["b"]


def _reference_nll_temperature(candidates: dict, gold: str, temperature: float) -> float:
    """A from-scratch (non-clipping) log-space NLL, independent of the module.

    Floors the *input* probability once, never the renormalised output, so
    it stays exact (no gradient-killing floor) no matter how small the
    gold's rescaled probability gets -- used as the ground truth a fixed
    ``fit_temperature`` must actually approach.
    """
    logs = {label: math.log(max(p, 1e-9)) / temperature for label, p in candidates.items()}
    top = max(logs.values())
    log_z = top + math.log(sum(math.exp(v - top) for v in logs.values()))
    return log_z - logs[gold]


def test_fit_temperature_does_not_clip_the_output_gold_probability(calibration_fit):
    # 99 well-behaved rows plus one row whose gold sits at a genuinely tiny
    # probability. A fitter that clips the *renormalised* gold probability
    # to EPS finds no cost to sharpening T all the way down (the one bad
    # row's contribution is capped either way), so it wrongly picks a very
    # small T; the true (uncapped) NLL of that choice is far worse than not
    # rescaling at all. The fix must land near the true minimum, not there.
    rows = [({"a": 0.9, "b": 0.1}, "a")] * 99 + [({"a": 1 - 1e-9, "b": 1e-9}, "b")] * 1

    def true_nll(temperature: float) -> float:
        return sum(_reference_nll_temperature(c, g, temperature) for c, g in rows)

    fitted = calibration_fit.fit_temperature(rows)
    assert fitted == pytest.approx(1.0, abs=0.2)
    assert true_nll(fitted) <= true_nll(1.0) + 1e-6


def test_fit_temperature_rolls_up_escalate_reason_candidates_for_gold_match(calibration_fit):
    # The gold is bench's bare "(escalate)" label, but the model's own
    # candidates only offer a reasoned "escalate:repair" -- calibration must
    # score that as escalate-family (metrics.canonical_label), not treat the
    # gold as absent from the line's candidates (which would make the loss
    # constant and let a single row send T wherever a flat search happens to
    # drift, per the review's repro of T -> 20 / escalate mass falling to
    # .53). With the rollup, escalate mass (already .9, the largest share)
    # can only be preserved or sharpened, never diluted, by the fit.
    metrics = calibration_fit.metrics
    gold = metrics.canonical_label("escalate:repair")
    candidates = {"escalate:repair": 0.9, "(explain)": 0.1}
    rows = [(candidates, gold)]

    fitted = calibration_fit.fit_temperature(rows)
    assert fitted <= 1.0

    scaled = calibration_fit.temperature_scale(candidates, fitted)
    escalate_mass = sum(p for label, p in scaled.items() if metrics.canonical_label(label) == gold)
    assert escalate_mass >= 0.9


def test_fit_params_counts_gold_absent_rows_separately(calibration_fit, tmp_path):
    # A row whose gold truly has no matching candidate carries no gradient
    # and must not silently distort the fit as a fixed, T-independent EPS
    # penalty; it is skipped and counted instead.
    predictions_path = tmp_path / "run.predictions.jsonl"
    rows = [
        _prediction_line("a", {"operation": "op_a", "args": {}}, {"op_a": 0.9, "op_b": 0.1}),
        _prediction_line("b", {"operation": "op_c", "args": {}}, {"op_a": 0.9, "op_b": 0.1}),
    ]
    _write_predictions(predictions_path, rows)
    params = calibration_fit.fit_params(predictions_path, {"seed": 1, "fit_ids": ["a", "b"]})
    assert params["skipped_gold_absent"] == 1
    assert params["fit_examples"] == 1


# ---------------------------------------------------------------------------
# End-to-end fit / apply via the CLI
# ---------------------------------------------------------------------------


def _corpus_predictions(calibration_fit, true_temperature: float, ids: list[str]) -> list[dict]:
    """40 ids -> golds whose frequency exactly matches ``true_dist`` (see _synthetic_rows)."""
    true_dist = {"op_a": 0.6, "op_b": 0.25, "(escalate)": 0.1, "(explain)": 0.05}
    distorted = calibration_fit.temperature_scale(true_dist, 1.0 / true_temperature)
    counts = {label: round(p * len(ids)) for label, p in true_dist.items()}
    golds = [label for label, count in counts.items() for _ in range(count)]
    golds = golds[: len(ids)] + ["op_a"] * max(0, len(ids) - len(golds))
    rows = []
    for entry_id, gold in zip(ids, golds):
        if gold in ("(escalate)", "(explain)"):
            key = "escalate" if gold == "(escalate)" else "explain"
            expected = {key: True}
        else:
            expected = {"operation": gold, "args": {}}
        rows.append(_prediction_line(entry_id, expected, dict(distorted)))
    return rows


def test_fit_then_apply_end_to_end(calibration_fit, tmp_path):
    ids = [f"id{n}" for n in range(40)]
    split_path = _split_file(tmp_path, "val.json", ids)
    folds_path = tmp_path / "folds.json"
    rc = calibration_fit.main(
        ["folds", "--split", str(split_path), "--seed", "39", "--out", str(folds_path)]
    )
    assert rc == 0

    predictions_path = tmp_path / "run-1-model.predictions.jsonl"
    _write_predictions(predictions_path, _corpus_predictions(calibration_fit, 2.5, ids))

    params_path = tmp_path / "params.json"
    rc = calibration_fit.main(
        [
            "fit",
            "--predictions",
            str(predictions_path),
            "--folds",
            str(folds_path),
            "--out",
            str(params_path),
        ]
    )
    assert rc == 0
    params = json.loads(params_path.read_text(encoding="utf-8"))
    # The fit fold is a ~70% seeded subsample of the 40 ids, so its empirical
    # label frequency only approximates true_dist -- a generous tolerance.
    assert params["temperature"] == pytest.approx(2.5, rel=0.3)
    assert params["skipped_null"] == 0
    assert set(params["vector"]) == {"op_a", "op_b", "(escalate)", "(explain)"}

    out_path = tmp_path / "rescaled.predictions.jsonl"
    rc = calibration_fit.main(
        [
            "apply",
            "--predictions",
            str(predictions_path),
            "--params",
            str(params_path),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    rescaled_lines = [
        json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rescaled_lines) == len(ids)
    first = rescaled_lines[0]
    assert math.isclose(sum(first["candidates"].values()), 1.0, abs_tol=1e-6)
    # T > 1 sharpens back towards the un-tempered base distribution.
    assert first["candidates"]["op_a"] > first["candidates"]["op_b"]
    assert first["id"] == ids[0]
    assert first["expected"] == {"operation": "op_a", "args": {}}


def test_apply_skips_null_candidates(calibration_fit, tmp_path):
    predictions_path = tmp_path / "run.predictions.jsonl"
    rows = [
        _prediction_line("a", {"operation": "op_a", "args": {}}, {"op_a": 0.8, "op_b": 0.2}),
        _prediction_line("b", {"operation": "op_a", "args": {}}, None, outcome="invalid"),
    ]
    _write_predictions(predictions_path, rows)
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"temperature": 2.0, "vector": {}}), encoding="utf-8")
    out_path = tmp_path / "out.jsonl"
    rc = calibration_fit.main(
        [
            "apply",
            "--predictions",
            str(predictions_path),
            "--params",
            str(params_path),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["candidates"] is not None
    assert lines[1]["candidates"] is None


def test_load_fit_rows_counts_null_and_out_of_fold(calibration_fit, tmp_path):
    predictions_path = tmp_path / "run.predictions.jsonl"
    rows = [
        _prediction_line("a", {"operation": "op_a", "args": {}}, {"op_a": 1.0}),
        _prediction_line("b", {"operation": "op_a", "args": {}}, None, outcome="invalid"),
        _prediction_line("c", {"operation": "op_a", "args": {}}, {"op_a": 1.0}),
    ]
    _write_predictions(predictions_path, rows)
    fit_rows, skipped_null, considered = calibration_fit.load_fit_rows(predictions_path, {"a", "b"})
    assert considered == 2
    assert skipped_null == 1
    assert len(fit_rows) == 1
    assert fit_rows[0][1] == "op_a"


# ---------------------------------------------------------------------------
# Refusing test / held-out predictions as fit input
# ---------------------------------------------------------------------------


def test_split_markers_detects_test_and_held_out_by_name(calibration_fit):
    assert "test" in calibration_fit.split_markers(Path("out/run-test.predictions.jsonl"))
    assert "held-out" in calibration_fit.split_markers(Path("out/held-out.predictions.jsonl"))
    assert calibration_fit.split_markers(Path("out/val.predictions.jsonl")) == set()


def test_split_markers_detects_by_header_text(calibration_fit):
    header = "Split 'test' of dev.json (seed=39)."
    assert "test" in calibration_fit.split_markers(Path("out/run-1-model.json"), header)


def test_fit_refuses_test_predictions_file(calibration_fit, tmp_path):
    ids = ["a", "b"]
    predictions_path = tmp_path / "run-test.predictions.jsonl"
    _write_predictions(predictions_path, _corpus_predictions(calibration_fit, 1.0, ids))
    folds_path = tmp_path / "folds.json"
    calibration_fit.write_json(folds_path, {"seed": 1, "fit_ids": ids, "selection_ids": []})
    params_path = tmp_path / "params.json"
    rc = calibration_fit.main(
        [
            "fit",
            "--predictions",
            str(predictions_path),
            "--folds",
            str(folds_path),
            "--out",
            str(params_path),
        ]
    )
    assert rc == 1
    assert not params_path.exists()


def test_fit_refuses_held_out_predictions_file(calibration_fit, tmp_path):
    ids = ["a", "b"]
    predictions_path = tmp_path / "held-out.predictions.jsonl"
    _write_predictions(predictions_path, _corpus_predictions(calibration_fit, 1.0, ids))
    folds_path = tmp_path / "folds.json"
    calibration_fit.write_json(folds_path, {"seed": 1, "fit_ids": ids, "selection_ids": []})
    params_path = tmp_path / "params.json"
    rc = calibration_fit.main(
        [
            "fit",
            "--predictions",
            str(predictions_path),
            "--folds",
            str(folds_path),
            "--out",
            str(params_path),
        ]
    )
    assert rc == 1
    assert not params_path.exists()


def test_fit_params_raises_calibration_error_directly(calibration_fit, tmp_path):
    predictions_path = tmp_path / "run-test.predictions.jsonl"
    _write_predictions(predictions_path, _corpus_predictions(calibration_fit, 1.0, ["a"]))
    with pytest.raises(calibration_fit.CalibrationError):
        calibration_fit.fit_params(predictions_path, {"seed": 1, "fit_ids": ["a"]})


def test_fit_params_raises_when_no_fit_candidates(calibration_fit, tmp_path):
    predictions_path = tmp_path / "run.predictions.jsonl"
    _write_predictions(
        predictions_path,
        [_prediction_line("a", {"operation": "op_a", "args": {}}, None, outcome="invalid")],
    )
    with pytest.raises(calibration_fit.CalibrationError):
        calibration_fit.fit_params(predictions_path, {"seed": 1, "fit_ids": ["a"]})
