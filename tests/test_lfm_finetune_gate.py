"""The uncertainty gate and its offline sweep: scripts/lfm-finetune/{gate,sweep_gate}.py.

``gate.decide`` never names an operation in its own logic (it reads
``nvsh.ops.table``'s ``read_only`` flag instead), so real operation names
below appear only as fixture data, exactly as ``test_lfm_finetune_metrics.py``
does it. A grep test guards the two scripts themselves for that.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _ROOT / "scripts/lfm-finetune/gate.py"
_SWEEP_PATH = _ROOT / "scripts/lfm-finetune/sweep_gate.py"

# Real operation names, used only as fixture data (see module docstring).
READ_ONLY_OP = "machine_status"
READ_ONLY_OP_2 = "gpu_stats"
MUTATING_OP = "power_set"
MUTATING_ARGS = {"mode": "balanced"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load(_GATE_PATH, "lfm_gate")


@pytest.fixture(scope="module")
def sweep(gate):
    return _load(_SWEEP_PATH, "lfm_sweep_gate")


def _op(name: str, **args) -> dict:
    return {"operation": name, "args": dict(args)}


def _line(
    id_: str,
    expected: dict,
    outcome: str,
    operation: str | None,
    arguments: dict | None,
    candidates: dict | None,
) -> dict:
    return {
        "id": id_,
        "expected": expected,
        "outcome": outcome,
        "operation": operation,
        "arguments": arguments,
        "candidates": candidates,
        "tokens": 0,
        "ttfd_ms": 1.0,
        "latency_ms": 1.0,
    }


# ---------------------------------------------------------------------------
# Thresholds / ThresholdSet
# ---------------------------------------------------------------------------


def test_threshold_set_json_round_trip(gate):
    ts = gate.ThresholdSet(floor=0.6, margin=0.1, max_entropy=0.8)
    assert gate.ThresholdSet.from_json(ts.to_json()) == ts


def test_threshold_set_json_round_trip_all_none(gate):
    ts = gate.ThresholdSet()
    assert ts.to_json() == {"floor": None, "margin": None, "max_entropy": None}
    assert gate.ThresholdSet.from_json(ts.to_json()) == ts


def test_thresholds_json_round_trip(gate):
    thresholds = gate.Thresholds(
        escalate=0.4,
        read_only=gate.ThresholdSet(floor=0.6),
        mutating=gate.ThresholdSet(floor=0.8, margin=0.2, max_entropy=0.5),
    )
    restored = gate.Thresholds.from_json(thresholds.to_json())
    assert restored == thresholds


def test_thresholds_from_json_rejects_non_object(gate):
    with pytest.raises(gate.GateError):
        gate.Thresholds.from_json("nope")


def test_threshold_set_from_json_rejects_bad_value(gate):
    with pytest.raises(gate.GateError):
        gate.ThresholdSet.from_json({"floor": "high"})


# ---------------------------------------------------------------------------
# decide(): the argmax path (every threshold disabled)
# ---------------------------------------------------------------------------


def test_decide_proposes_the_argmax_operation(gate):
    candidates = {READ_ONLY_OP: 0.9, "(explain)": 0.05, "(escalate)": 0.05}
    decision = gate.decide(candidates, list(candidates), gate.Thresholds())
    assert decision.outcome == "propose"
    assert decision.label == READ_ONLY_OP


def test_decide_explains_when_explain_is_argmax(gate):
    candidates = {READ_ONLY_OP: 0.3, "(explain)": 0.6, "(escalate)": 0.1}
    decision = gate.decide(candidates, list(candidates), gate.Thresholds())
    assert decision.outcome == "explain"
    assert decision.label == "(explain)"


def test_decide_escalates_when_escalate_is_argmax(gate):
    candidates = {READ_ONLY_OP: 0.3, "(explain)": 0.1, "(escalate)": 0.6}
    decision = gate.decide(candidates, list(candidates), gate.Thresholds())
    assert decision.outcome == "escalate"
    assert decision.label == "(escalate)"
    assert decision.reason == "argmax"


def test_decide_rolls_up_escalate_reasons_for_argmax(gate):
    # Split escalate mass rolls up to beat the operation's bare 0.45.
    candidates = {READ_ONLY_OP: 0.45, "escalate:low_confidence": 0.3, "escalate:other": 0.25}
    decision = gate.decide(candidates, list(candidates), gate.Thresholds())
    assert decision.outcome == "escalate"
    assert decision.label == "(escalate)"


def test_decide_ties_go_to_the_earlier_offered_label(gate):
    candidates = {READ_ONLY_OP: 0.5, READ_ONLY_OP_2: 0.5}
    offered = [READ_ONLY_OP_2, READ_ONLY_OP]  # gpu_stats offered first
    decision = gate.decide(candidates, offered, gate.Thresholds())
    assert decision.label == READ_ONLY_OP_2


def test_decide_rejects_empty_distribution(gate):
    with pytest.raises(gate.GateError):
        gate.decide({}, [], gate.Thresholds())


# ---------------------------------------------------------------------------
# decide(): semantic escalate by threshold
# ---------------------------------------------------------------------------


def test_decide_escalates_on_threshold_even_when_not_argmax(gate):
    candidates = {READ_ONLY_OP: 0.55, "(escalate)": 0.45}
    decision = gate.decide(candidates, list(candidates), gate.Thresholds(escalate=0.4))
    assert decision.outcome == "escalate"
    assert decision.reason == "threshold"


def test_decide_escalate_threshold_disabled_never_fires(gate):
    candidates = {READ_ONLY_OP: 0.55, "(escalate)": 0.45}
    decision = gate.decide(candidates, list(candidates), gate.Thresholds(escalate=None))
    assert decision.outcome == "propose"


# ---------------------------------------------------------------------------
# decide(): abstain_uncertain, per operation class
# ---------------------------------------------------------------------------


def test_decide_abstains_on_floor_for_read_only(gate):
    candidates = {READ_ONLY_OP: 0.4, READ_ONLY_OP_2: 0.3, "(explain)": 0.3}
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(floor=0.5))
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "floor"
    assert decision.label == READ_ONLY_OP


def test_decide_floor_disabled_never_fires(gate):
    candidates = {READ_ONLY_OP: 0.4, READ_ONLY_OP_2: 0.3, "(explain)": 0.3}
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(floor=None))
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "propose"


def test_decide_abstains_on_margin(gate):
    candidates = {READ_ONLY_OP: 0.45, READ_ONLY_OP_2: 0.40, "(explain)": 0.15}
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(margin=0.5))
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "margin"


def test_decide_abstains_on_entropy(gate):
    candidates = {READ_ONLY_OP: 0.4, READ_ONLY_OP_2: 0.35, "(explain)": 0.25}
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(max_entropy=0.5))
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "entropy"


def test_decide_uses_mutating_thresholds_for_a_mutating_operation(gate):
    candidates = {MUTATING_OP: 0.6, "(explain)": 0.4}
    # A tight read_only floor must not apply to a mutating argmax.
    thresholds = gate.Thresholds(
        read_only=gate.ThresholdSet(floor=0.99),
        mutating=gate.ThresholdSet(floor=0.5),
    )
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "propose"
    assert decision.label == MUTATING_OP


def test_decide_mutating_floor_can_abstain_a_mutating_proposal(gate):
    candidates = {MUTATING_OP: 0.6, "(explain)": 0.4}
    thresholds = gate.Thresholds(mutating=gate.ThresholdSet(floor=0.7))
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "abstain_uncertain"
    assert decision.reason == "floor"


def test_decide_abstain_uncertain_never_fires_for_a_control(gate):
    # explain/escalate argmax never runs through the operation-class thresholds.
    candidates = {"(explain)": 0.6, READ_ONLY_OP: 0.4}
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(floor=0.99))
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "explain"


# ---------------------------------------------------------------------------
# normalized_entropy
# ---------------------------------------------------------------------------


def test_normalized_entropy_zero_for_one_offered(gate):
    assert gate.normalized_entropy({READ_ONLY_OP: 1.0}, 1) == 0.0


def test_normalized_entropy_one_for_uniform_pair(gate):
    entropy = gate.normalized_entropy({READ_ONLY_OP: 0.5, READ_ONLY_OP_2: 0.5}, 2)
    assert entropy == pytest.approx(1.0)


def test_normalized_entropy_zero_for_certain_answer(gate):
    entropy = gate.normalized_entropy({READ_ONLY_OP: 1.0, READ_ONLY_OP_2: 0.0}, 2)
    assert entropy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# No operation names in the two scripts (mirrors the harness's own grep gate)
# ---------------------------------------------------------------------------

_OPERATION_NAME_PATTERN = re.compile(
    r"service_|container_|power_|gpu_stats|disk_stats|memory_stats|machine_status|"
    r"network_info|process_list|swap_status|thermal_stats|nvsh_doctor"
)


@pytest.mark.parametrize("path", [_GATE_PATH, _SWEEP_PATH])
def test_no_operation_names_in_source(path):
    text = path.read_text(encoding="utf-8")
    assert not _OPERATION_NAME_PATTERN.search(text), f"{path} names an operation"


# ---------------------------------------------------------------------------
# sweep_gate: grid parsing
# ---------------------------------------------------------------------------


def test_parse_grid_none(sweep):
    assert sweep.parse_grid("none") == [None]


def test_parse_grid_numbers(sweep):
    assert sweep.parse_grid("0.5, 0.7") == [0.5, 0.7]


def test_parse_grid_mixed(sweep):
    assert sweep.parse_grid("none,0.5") == [None, 0.5]


def test_parse_grid_rejects_garbage(sweep):
    with pytest.raises(sweep.SweepError):
        sweep.parse_grid("bogus")


def test_parse_grid_rejects_empty(sweep):
    with pytest.raises(sweep.SweepError):
        sweep.parse_grid("")


def test_build_threshold_grid_cartesian_size(sweep, gate):
    grid = sweep.build_threshold_grid(
        escalate=[None, 0.5],
        ro_floor=[None, 0.6],
        ro_margin=[None],
        ro_max_entropy=[None],
    )
    # escalate(2) x ro_floor(2) x ro_margin(1) x ro_entropy(1), each squared again
    # by mut_* defaulting to the same ro_* grid (see build_threshold_grid's docstring).
    assert len(grid) == 8
    assert all(isinstance(t, sweep.gate.Thresholds) for t in grid)


def test_build_threshold_grid_mutating_defaults_to_read_only(sweep):
    grid = sweep.build_threshold_grid(
        escalate=[None], ro_floor=[0.6], ro_margin=[None], ro_max_entropy=[None]
    )
    (thresholds,) = grid
    assert thresholds.mutating.floor == 0.6


def test_build_threshold_grid_mutating_override(sweep):
    grid = sweep.build_threshold_grid(
        escalate=[None],
        ro_floor=[0.6],
        ro_margin=[None],
        ro_max_entropy=[None],
        mut_floor=[0.9],
    )
    (thresholds,) = grid
    assert thresholds.read_only.floor == 0.6
    assert thresholds.mutating.floor == 0.9


# ---------------------------------------------------------------------------
# sweep_gate: split safety
# ---------------------------------------------------------------------------


def test_refuse_unless_final_allows_a_plain_path(sweep, tmp_path):
    path = tmp_path / "dev-predictions.jsonl"
    sweep.refuse_unless_final(path, final=False)  # no raise


def test_refuse_unless_final_blocks_test_named_file(sweep, tmp_path):
    path = tmp_path / "test-predictions.jsonl"
    with pytest.raises(sweep.SweepError):
        sweep.refuse_unless_final(path, final=False)


def test_refuse_unless_final_blocks_final_directory(sweep, tmp_path):
    final_dir = tmp_path / "final" / "scorer-b1"
    final_dir.mkdir(parents=True)
    path = final_dir / "scorer-b1.predictions.jsonl"
    with pytest.raises(sweep.SweepError):
        sweep.refuse_unless_final(path, final=False)


def test_refuse_unless_final_allows_final_with_flag(sweep, tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    path = final_dir / "final-scorer-b1.predictions.jsonl"
    sweep.refuse_unless_final(path, final=True)  # no raise


# ---------------------------------------------------------------------------
# sweep_gate: redecide()
# ---------------------------------------------------------------------------


def test_redecide_passes_through_a_null_candidates_line(sweep, gate):
    prediction = sweep.metrics.Prediction.from_dict(
        _line("x1", _op(READ_ONLY_OP), "propose", READ_ONLY_OP, {}, None)
    )
    result = sweep.redecide(prediction, gate.Thresholds())
    assert result is prediction


def test_redecide_reproduces_a_propose_line_under_disabled_thresholds(sweep, gate):
    candidates = {READ_ONLY_OP: 0.9, "(explain)": 0.05, "(escalate)": 0.05}
    prediction = sweep.metrics.Prediction.from_dict(
        _line("x2", _op(READ_ONLY_OP), "propose", READ_ONLY_OP, {}, candidates)
    )
    result = sweep.redecide(prediction, gate.Thresholds())
    assert result.outcome == "propose"
    assert result.operation == READ_ONLY_OP
    assert result.arguments == {}


def test_redecide_can_turn_a_propose_line_into_abstain_uncertain(sweep, gate):
    candidates = {READ_ONLY_OP: 0.6, "(explain)": 0.4}
    prediction = sweep.metrics.Prediction.from_dict(
        _line("x3", _op(READ_ONLY_OP), "propose", READ_ONLY_OP, {}, candidates)
    )
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(floor=0.8))
    result = sweep.redecide(prediction, thresholds)
    assert result.outcome == "abstain_uncertain"
    assert result.operation is None
    assert result.arguments is None


# ---------------------------------------------------------------------------
# sweep_gate: redecide() never invents {} arguments for an invalid original
# (issue 53, P2)
# ---------------------------------------------------------------------------


def test_redecide_keeps_an_invalid_line_invalid_under_disabled_thresholds(sweep, gate):
    """A complete distribution whose winning operation could not be grounded stays invalid.

    Before the fix, disabling every threshold still turned this into a
    fabricated ``propose`` with ``{}`` arguments -- with a complete
    distribution and nothing overriding the argmax, ``gate.decide`` always
    proposes it, and ``redecide`` used to trust that blindly.
    """
    candidates = {READ_ONLY_OP: 0.93, READ_ONLY_OP_2: 0.05, "(explain)": 0.02}
    prediction = sweep.metrics.Prediction.from_dict(
        _line("invalid-1", _op(READ_ONLY_OP), "invalid", None, None, candidates)
    )
    result = sweep.redecide(prediction, gate.Thresholds())
    assert result.outcome == "invalid"
    assert result.operation is None
    assert result.arguments is None
    assert result.invalid_reason == sweep.metrics.invalid_reason(prediction)


def test_redecide_keeps_a_functionally_invalid_propose_invalid(sweep, gate):
    """A "propose" line whose own arguments already fail the operation table stays invalid.

    ``metrics.invalid_reason`` calls this kind of line invalid even though
    its stored ``outcome`` field says ``"propose"``; redecide must not
    launder it into a clean propose just because the gate's argmax agrees
    with its operation.
    """
    candidates = {MUTATING_OP: 0.9, "(escalate)": 0.1}
    prediction = sweep.metrics.Prediction.from_dict(
        _line(
            "invalid-2", _op(MUTATING_OP, **MUTATING_ARGS), "propose", MUTATING_OP, {}, candidates
        )
    )
    assert sweep.metrics.invalid_reason(prediction) is not None  # bad args: {} for power_set
    result = sweep.redecide(prediction, gate.Thresholds())
    assert result.outcome == "invalid"
    assert result.operation is None
    assert result.arguments is None
    assert result.invalid_reason == sweep.metrics.invalid_reason(prediction)


def test_redecide_marks_not_grounded_by_sweep_for_a_different_operation(sweep, gate):
    """The gate picking a different operation than the original never invents arguments.

    The stored line disagrees with its own candidates' argmax (a
    corrupted/tie-broken record) -- an edge case, but redecide must still
    refuse to attach the *original*'s arguments to the operation the gate
    actually picked, or to invent new ones.
    """
    candidates = {READ_ONLY_OP: 0.7, READ_ONLY_OP_2: 0.3}
    prediction = sweep.metrics.Prediction.from_dict(
        _line("mismatch", _op(READ_ONLY_OP_2), "propose", READ_ONLY_OP_2, {}, candidates)
    )
    result = sweep.redecide(prediction, gate.Thresholds())
    assert result.outcome == "invalid"
    assert result.operation is None
    assert result.arguments is None
    assert result.invalid_reason == sweep.NOT_GROUNDED_BY_SWEEP


def test_sweep_reproduces_an_invalid_line_exactly_when_disabled(sweep, gate):
    """The scorer-b1-shaped fixture plus an invalid line: 100% outcome+operation match."""
    lines = _scorer_b1_style_fixture() + [
        _line(
            "b1-06-invalid",
            _op(READ_ONLY_OP),
            "invalid",
            None,
            None,
            {READ_ONLY_OP: 0.93, READ_ONLY_OP_2: 0.05, "(explain)": 0.02},
        ),
    ]
    originals = [sweep.metrics.Prediction.from_dict(line) for line in lines]
    for original in originals:
        redecided = sweep.redecide(original, gate.Thresholds())
        assert redecided.outcome == original.outcome, original.id
        assert redecided.operation == original.operation, original.id


# ---------------------------------------------------------------------------
# sweep_gate: the exact-reproduction acceptance test
# ---------------------------------------------------------------------------


def _scorer_b1_style_fixture() -> list[dict]:
    """A handful of lines shaped like scorer-b1's stored exact predictions.

    Each line's ``outcome``/``operation`` already equals what its own
    ``candidates`` argmax would give -- exactly how a scorer.py-produced
    predictions file records a decision with no gate involved.
    """
    return [
        _line(
            "b1-01",
            _op(READ_ONLY_OP),
            "propose",
            READ_ONLY_OP,
            {},
            {READ_ONLY_OP: 0.97, READ_ONLY_OP_2: 0.02, "(explain)": 0.01},
        ),
        _line(
            "b1-02",
            _op(MUTATING_OP, **MUTATING_ARGS),
            "propose",
            MUTATING_OP,
            dict(MUTATING_ARGS),
            {MUTATING_OP: 0.88, "(escalate)": 0.12},
        ),
        _line(
            "b1-03",
            {"explain": True},
            "explain",
            None,
            None,
            {"(explain)": 0.7, READ_ONLY_OP: 0.3},
        ),
        _line(
            "b1-04",
            {"escalate": True},
            "escalate",
            None,
            None,
            {"(escalate)": 0.65, READ_ONLY_OP: 0.35},
        ),
        _line(
            "b1-05",
            _op(READ_ONLY_OP_2),
            "propose",
            READ_ONLY_OP_2,
            {},
            {READ_ONLY_OP_2: 0.51, READ_ONLY_OP: 0.49},
        ),
    ]


def test_sweep_reproduces_scorer_b1_argmax_exactly_when_disabled(sweep, gate, tmp_path):
    lines = _scorer_b1_style_fixture()
    path = tmp_path / "scorer-b1-fixture.predictions.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    originals = sweep.metrics.read_predictions(path)
    matches, total = sweep.count_argmax_matches(originals)
    assert total == len(lines)
    assert matches == total

    for original in originals:
        redecided = sweep.redecide(original, gate.Thresholds())
        assert redecided.outcome == original.outcome
        assert redecided.operation == original.operation


def test_evaluate_and_run_sweep_shape(sweep, gate):
    originals = [sweep.metrics.Prediction.from_dict(line) for line in _scorer_b1_style_fixture()]
    grid = [gate.Thresholds(), gate.Thresholds(read_only=gate.ThresholdSet(floor=0.99))]
    reports = sweep.run_sweep(originals, grid)
    assert len(reports) == 2
    for report in reports:
        assert set(report) >= {
            "thresholds",
            "right_proposals",
            "wrong_mutating",
            "abstain_uncertain_count",
            "escalation",
            "false_positives",
            "missing_candidate_escalation_recall",
        }


def test_missing_candidate_escalation_recall_improves_with_a_threshold(sweep, gate):
    # A -nocand line the scorer confidently (wrongly) proposed on.
    line = _line(
        "case-nocand",
        _op(READ_ONLY_OP),
        "propose",
        READ_ONLY_OP_2,
        {},
        {READ_ONLY_OP_2: 0.6, "(explain)": 0.4},
    )
    prediction = sweep.metrics.Prediction.from_dict(line)
    assert sweep.metrics.is_missing_candidate(prediction)

    disabled = sweep._missing_candidate_recall(
        [prediction], [sweep.redecide(prediction, gate.Thresholds())]
    )
    assert disabled == {"n": 0, "N": 1, "rate": 0.0}

    tight = gate.Thresholds(read_only=gate.ThresholdSet(floor=0.7))
    gated = sweep._missing_candidate_recall([prediction], [sweep.redecide(prediction, tight)])
    assert gated == {"n": 1, "N": 1, "rate": 1.0}


# ---------------------------------------------------------------------------
# sweep_gate: folds
# ---------------------------------------------------------------------------


def test_filter_by_fold_restricts_to_the_named_fold(sweep):
    originals = [sweep.metrics.Prediction.from_dict(line) for line in _scorer_b1_style_fixture()]
    folds = {"fit_ids": ["b1-01", "b1-02"], "selection_ids": ["b1-03", "b1-04", "b1-05"]}
    fit = sweep.filter_by_fold(originals, folds, "fit")
    assert {p.id for p in fit} == {"b1-01", "b1-02"}
    selection = sweep.filter_by_fold(originals, folds, "selection")
    assert {p.id for p in selection} == {"b1-03", "b1-04", "b1-05"}


def test_filter_by_fold_none_folds_keeps_everything(sweep):
    originals = [sweep.metrics.Prediction.from_dict(line) for line in _scorer_b1_style_fixture()]
    assert sweep.filter_by_fold(originals, None, None) == originals


def test_fold_ids_rejects_unknown_fold_name(sweep):
    with pytest.raises(sweep.SweepError):
        sweep.fold_ids({"fit_ids": [], "selection_ids": []}, "bogus")


# ---------------------------------------------------------------------------
# sweep_gate: CLI (main)
# ---------------------------------------------------------------------------


def _write_predictions(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    lines = _scorer_b1_style_fixture()
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_main_default_grid_is_a_single_disabled_combo(sweep, tmp_path, capsys):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    rc = sweep.main(["--predictions", str(path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 5
    assert len(out["reports"]) == 1
    assert out["reports"][0]["thresholds"] == gate_thresholds_json()


def gate_thresholds_json():
    return {
        "escalate": None,
        "read_only": {"floor": None, "margin": None, "max_entropy": None},
        "mutating": {"floor": None, "margin": None, "max_entropy": None},
    }


def test_main_refuses_a_test_named_file_without_final(sweep, tmp_path, capsys):
    path = _write_predictions(tmp_path, "test-predictions.jsonl")
    rc = sweep.main(["--predictions", str(path)])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_main_allows_test_named_file_with_final(sweep, tmp_path, capsys):
    path = _write_predictions(tmp_path, "test-predictions.jsonl")
    rc = sweep.main(["--predictions", str(path), "--final"])
    assert rc == 0


def test_main_writes_markdown_when_asked(sweep, tmp_path):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    markdown_path = tmp_path / "report.md"
    rc = sweep.main(
        [
            "--predictions",
            str(path),
            "--out",
            str(tmp_path / "out.json"),
            "--markdown",
            str(markdown_path),
        ]
    )
    assert rc == 0
    text = markdown_path.read_text(encoding="utf-8")
    assert text.startswith("| escalate |")


def test_main_grid_flags_expand_the_report(sweep, tmp_path, capsys):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    rc = sweep.main(["--predictions", str(path), "--floor", "none,0.99"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # --floor sets both ro_floor and (by default) mut_floor to the same 2-value
    # grid, so the two floor axes multiply: 2 x 2 = 4 combinations.
    assert len(out["reports"]) == 4


def test_main_fold_filters_the_report_count(sweep, tmp_path, capsys):
    path = _write_predictions(tmp_path, "dev.predictions.jsonl")
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps({"fit_ids": ["b1-01", "b1-02"], "selection_ids": ["b1-03", "b1-04", "b1-05"]}),
        encoding="utf-8",
    )
    rc = sweep.main(["--predictions", str(path), "--folds", str(folds_path), "--fold", "fit"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 2
    assert out["fold"] == "fit"


# PR #65 review: an unknown operation and reasons-mode entropy.


def test_an_unknown_operation_gets_the_mutating_thresholds(gate):
    """metrics.slice_name counts an operation missing from the table as
    mutating; the gate must gate it the same, stricter way."""
    candidates = {"not_in_the_table": 0.6, "(explain)": 0.2, "(escalate)": 0.2}
    thresholds = gate.Thresholds(mutating=gate.ThresholdSet(floor=0.9))
    decision = gate.decide(candidates, list(candidates), thresholds)
    assert decision.outcome == "abstain_uncertain"


def test_entropy_is_normalised_over_the_rolled_up_candidates(gate):
    """Eight escalate:<reason> labels roll up to one; a uniform distribution
    over the rolled candidates must still read entropy 1.0."""
    reasons = [f"escalate:r{i}" for i in range(8)]
    candidates = {READ_ONLY_OP: 1 / 3, "(explain)": 1 / 3}
    candidates.update({r: (1 / 3) / 8 for r in reasons})
    offered = [READ_ONLY_OP, "(explain)", *reasons]
    thresholds = gate.Thresholds(read_only=gate.ThresholdSet(max_entropy=0.95))
    decision = gate.decide(candidates, offered, thresholds)
    assert decision.outcome == "abstain_uncertain"


def test_a_sweep_computes_no_bootstrap_resamples(sweep, gate, monkeypatch):
    """PR #65 review: every grid point reran ~20 x 1000 bootstrap resamples it
    never reads; the sweep only reports point values."""
    originals = [sweep.metrics.Prediction.from_dict(line) for line in _scorer_b1_style_fixture()]

    def no_resampling(*_args, **_kwargs):
        raise AssertionError("a sweep must not bootstrap")

    monkeypatch.setattr(sweep.metrics, "_resample", no_resampling)
    reports = sweep.run_sweep(originals, [gate.Thresholds()])
    assert reports[0]["right_proposals"]["n"] >= 0
