"""Metrics bridge onto ``scripts/lfm-finetune/{metrics,gate}.py`` (issue 64, task t7).

This module never reimplements a formula that already lives in
``scripts/lfm-finetune/metrics.py`` or ``scripts/lfm-finetune/gate.py``. It
loads both by path -- exactly the way ``tests/test_lfm_finetune_metrics.py``
and ``tests/test_lfm_finetune_gate.py`` do -- and calls their functions
directly. The only things genuinely new here are:

* **top-k accuracy** (k > 1; metrics.py only reports top-1 accuracy, via
  ``right_proposals`` for exact proposals and via ``compute_calibration``'s
  top candidate for calibration), built on ``metrics.top_candidate``'s own
  ranking rule generalised to "is gold within the top *k* by probability".
* **log loss** (metrics.py reports Brier, not log loss), built on
  ``metrics.rollup_escalate_candidates`` + ``metrics.expected_label`` --
  the same gold label and rolled distribution ``metrics.brier_one`` uses.
* **per-row normalized entropy and top1-top2 margin** -- these are
  ``gate.normalized_entropy`` and the margin ``gate.decide`` computes
  internally from ``gate._argmax``/``gate._second_place``, called directly
  rather than re-derived, and reported per prediction rather than only as
  an internal step of one threshold decision.

Everything else (right proposals, ECE, Brier, abstain precision/recall,
missing-candidate, wrong-mutating, per-slice) is ``scripts/lfm-finetune/
metrics.py``'s own :func:`compute` output, passed through unchanged.

nothing under ``nvsh/`` may import this module, and this module is never
imported by ``nvsh/`` (see ``evals/tool_jev/__init__.py`` and
``docs/`` for the boundary). No network, no subprocess.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Sequence

#: The two source-of-truth modules this bridge reads, never copies.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_METRICS_PATH = _REPO_ROOT / "scripts" / "lfm-finetune" / "metrics.py"
_GATE_PATH = _REPO_ROOT / "scripts" / "lfm-finetune" / "gate.py"

#: The explicit sentinel every calibration figure reports for a row with no
#: candidate distribution -- never a number, never an estimate.
NOT_MEASURABLE = "not_measurable"

#: Default top-k values this bridge reports accuracy for.
TOP_K_VALUES: tuple[int, ...] = (1, 3, 5)

#: log loss's floor probability, so a gold label absent from the offered
#: candidates (probability 0) gives a large finite number instead of raising.
_LOG_LOSS_EPSILON = 1e-12


def _load_by_path(name: str, path: Path):
    """Load *path* as a standalone module named *name* (not a package import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_metrics_module():
    """``scripts/lfm-finetune/metrics.py``, loaded fresh by path."""
    return _load_by_path("_nvsh_evals_tool_jev_lfm_metrics", _METRICS_PATH)


def load_gate_module():
    """``scripts/lfm-finetune/gate.py``, loaded fresh by path.

    ``gate.py`` loads its own sibling copy of ``metrics.py`` internally
    (under a different module name), so this is independent of whatever
    :func:`load_metrics_module` returns; the two agree because there is
    only one ``metrics.py`` on disk.
    """
    return _load_by_path("_nvsh_evals_tool_jev_lfm_gate", _GATE_PATH)


# ---------------------------------------------------------------------------
# Per-row distribution helpers
# ---------------------------------------------------------------------------


def _rolled_and_offered(metrics_mod, prediction):
    """(rolled candidates, offered labels) for *prediction*, or (None, None).

    ``None`` when the row carries no candidate distribution at all
    (``prediction.candidates is None``) -- the case every calibration figure
    must report as :data:`NOT_MEASURABLE`, never estimate.
    """
    if prediction.candidates is None:
        return None, None
    rolled = metrics_mod.rollup_escalate_candidates(prediction.candidates)
    offered = list(prediction.candidates.keys())
    return rolled, offered


def gold_label(metrics_mod, prediction) -> str:
    """The calibration gold label for *prediction* (``metrics.expected_label``)."""
    return metrics_mod.expected_label(prediction.expected)


def top1(metrics_mod, prediction) -> tuple[str, float] | None:
    """(label, probability) of the rolled top candidate, or ``None`` with no distribution.

    Uses ``metrics.top_candidate``'s own tie-break (sorts first), the same
    one ``compute_calibration`` uses for top-1 accuracy/ECE/Brier.
    """
    rolled, _ = _rolled_and_offered(metrics_mod, prediction)
    if rolled is None:
        return None
    return metrics_mod.top_candidate(rolled)


def top1_correct(metrics_mod, prediction) -> bool | None:
    """Whether the rolled top candidate's label is the gold label, or ``None``."""
    candidate = top1(metrics_mod, prediction)
    if candidate is None:
        return None
    label, _ = candidate
    return label == gold_label(metrics_mod, prediction)


def topk_correct(metrics_mod, prediction, k: int) -> bool | None:
    """Whether the gold label is among the top *k* rolled candidates by probability.

    Ties break the same way ``metrics.top_candidate`` breaks a tie for k=1:
    lower probability first is never preferred, and among equal
    probabilities the label that sorts first is ranked first.
    """
    rolled, _ = _rolled_and_offered(metrics_mod, prediction)
    if rolled is None:
        return None
    ranked = sorted(rolled.items(), key=lambda item: (-item[1], item[0]))
    top_labels = [label for label, _ in ranked[:k]]
    return gold_label(metrics_mod, prediction) in top_labels


def brier(metrics_mod, prediction) -> float | str:
    """``metrics.brier_one`` for this row, or :data:`NOT_MEASURABLE`."""
    rolled, _ = _rolled_and_offered(metrics_mod, prediction)
    if rolled is None:
        return NOT_MEASURABLE
    return metrics_mod.brier_one(rolled, gold_label(metrics_mod, prediction))


def log_loss_one(metrics_mod, prediction) -> float | str:
    """-log(p_gold) over the rolled distribution, or :data:`NOT_MEASURABLE`.

    ``metrics.py`` never reports log loss (only Brier); this reuses the same
    gold label and rolled distribution ``brier_one`` is scored against, and
    floors the gold probability at :data:`_LOG_LOSS_EPSILON` so a gold label
    entirely absent from the offered candidates (probability 0) gives a
    large finite number instead of raising.
    """
    rolled, _ = _rolled_and_offered(metrics_mod, prediction)
    if rolled is None:
        return NOT_MEASURABLE
    p_gold = float(rolled.get(gold_label(metrics_mod, prediction), 0.0))
    return -math.log(max(p_gold, _LOG_LOSS_EPSILON))


def entropy(metrics_mod, gate_mod, prediction) -> float | str:
    """``gate.normalized_entropy`` for this row, or :data:`NOT_MEASURABLE`.

    ``n_offered`` is the number of distinct offered labels *before* the
    escalate-reason roll-up, exactly as ``gate.decide`` computes it.
    """
    rolled, offered = _rolled_and_offered(metrics_mod, prediction)
    if rolled is None:
        return NOT_MEASURABLE
    n_offered = len({metrics_mod.canonical_label(label) for label in offered})
    return gate_mod.normalized_entropy(rolled, n_offered)


def margin(metrics_mod, gate_mod, prediction) -> float | str:
    """The top1-top2 margin ``gate.decide`` computes internally, or :data:`NOT_MEASURABLE`.

    Calls ``gate._argmax``/``gate._second_place`` directly (the same private
    helpers ``gate.decide`` uses) instead of re-deriving the tie-break rule.
    """
    rolled, offered = _rolled_and_offered(metrics_mod, prediction)
    if rolled is None:
        return NOT_MEASURABLE
    top1_label, p_top1 = gate_mod._argmax(rolled, offered)
    return p_top1 - gate_mod._second_place(rolled, top1_label)


# ---------------------------------------------------------------------------
# Per-row summary
# ---------------------------------------------------------------------------


def row_metrics(metrics_mod, gate_mod, prediction, *, top_k: Sequence[int] = TOP_K_VALUES) -> dict:
    """Every per-row bridge metric for one prediction, as a JSON-able dict.

    ``top1``/``topk``/``brier``/``log_loss``/``entropy``/``margin`` are
    :data:`NOT_MEASURABLE` (never a number) when ``prediction.candidates``
    is ``None``. ``missing_candidate``/``wrong_mutating``/``slice`` come
    straight from ``metrics.py`` (``is_missing_candidate``,
    ``_wrong_operation_mutating`` OR ``_wrong_arguments_mutating``, and
    ``slice_name``) and are always defined, distribution or not.
    """
    correct = top1_correct(metrics_mod, prediction)
    return {
        "id": prediction.id,
        "slice": metrics_mod.slice_name(prediction),
        "has_distribution": prediction.candidates is not None,
        "top1_correct": NOT_MEASURABLE if correct is None else correct,
        "topk_correct": {
            k: (
                NOT_MEASURABLE
                if (value := topk_correct(metrics_mod, prediction, k)) is None
                else value
            )
            for k in top_k
        },
        "brier": brier(metrics_mod, prediction),
        "log_loss": log_loss_one(metrics_mod, prediction),
        "entropy": entropy(metrics_mod, gate_mod, prediction),
        "margin": margin(metrics_mod, gate_mod, prediction),
        "missing_candidate": metrics_mod.is_missing_candidate(prediction),
        "wrong_mutating": (
            metrics_mod._wrong_operation_mutating(prediction)
            or metrics_mod._wrong_arguments_mutating(prediction)
        ),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def top_k_accuracy(
    metrics_mod, predictions: Sequence, *, top_k: Sequence[int] = TOP_K_VALUES
) -> dict[int, dict]:
    """Top-k accuracy over operation-expected rows that carry a distribution.

    Restricted to operation-expected rows the same way ``right_proposals``
    is, since "gold in the top k" is only meaningful when gold names one
    of the offered candidates (an escalate/explain-expected row's gold
    label is a control, not a competing operation).
    """
    measured = [
        p
        for p in predictions
        if metrics_mod.expect_kind(p.expected) == "operation" and p.candidates is not None
    ]
    result: dict[int, dict] = {}
    for k in top_k:
        n = sum(1 for p in measured if topk_correct(metrics_mod, p, k))
        result[k] = {"n": n, "N": len(measured), "rate": metrics_mod._ratio(n, len(measured))}
    return result


def mean_log_loss(metrics_mod, predictions: Sequence) -> dict:
    """Mean log loss over rows that carry a distribution; :data:`NOT_MEASURABLE` rows excluded."""
    losses = [log_loss_one(metrics_mod, p) for p in predictions if p.candidates is not None]
    return {
        "n": len(losses),
        "without_distribution": len(predictions) - len(losses),
        "mean": (math.fsum(losses) / len(losses)) if losses else None,
    }


def missing_candidate_summary(metrics_mod, predictions: Sequence) -> dict:
    """Aggregate missing-candidate rate over every row (``metrics.is_missing_candidate``)."""
    n = sum(1 for p in predictions if metrics_mod.is_missing_candidate(p))
    return {"n": n, "N": len(predictions), "rate": metrics_mod._ratio(n, len(predictions))}


def compute(
    predictions: Sequence,
    *,
    metrics_mod=None,
    gate_mod=None,
    top_k: Sequence[int] = TOP_K_VALUES,
    bootstrap_seed: int | None = None,
    bootstrap_resamples: int | None = None,
) -> dict:
    """Every bridge metric for *predictions* (a sequence of ``metrics.Prediction``).

    ``metrics_mod``/``gate_mod`` may be passed in (already-loaded modules,
    e.g. shared across calls in a test); when omitted, fresh copies are
    loaded by path. ``metrics_compute`` in the result is
    ``scripts/lfm-finetune/metrics.py``'s own :func:`compute` output,
    unmodified, so right proposals, ECE, Brier, abstain precision/recall,
    wrong-mutating and per-slice figures are exactly what that module
    returns for the same rows -- never recomputed here.
    """
    metrics_mod = metrics_mod or load_metrics_module()
    gate_mod = gate_mod or load_gate_module()

    kwargs = {}
    if bootstrap_seed is not None:
        kwargs["bootstrap_seed"] = bootstrap_seed
    if bootstrap_resamples is not None:
        kwargs["bootstrap_resamples"] = bootstrap_resamples
    base = metrics_mod.compute(predictions, **kwargs)

    return {
        "rows": [row_metrics(metrics_mod, gate_mod, p, top_k=top_k) for p in predictions],
        "metrics_compute": base,
        "top_k_accuracy": top_k_accuracy(metrics_mod, predictions, top_k=top_k),
        "log_loss": mean_log_loss(metrics_mod, predictions),
        "missing_candidate": missing_candidate_summary(metrics_mod, predictions),
        # Convenience aliases onto metrics_compute's own shape, named the way
        # this bridge's task instruction names them (abstain P/R, wrong-mutating,
        # per-slice) -- not a recomputation, the same dict objects.
        "abstain": base["abstention"],
        "wrong_mutating": base["wrong_mutating"],
        "slices": base["slices"],
    }
