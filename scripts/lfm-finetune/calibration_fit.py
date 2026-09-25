#!/usr/bin/env python3
"""Post-hoc calibration for issue-46 predictions: temperature and vector scaling.

A development-machine tool (part of #53). It is NEVER imported by the nvsh
package -- nothing under nvsh/ may depend on it. Stdlib only.

Three steps, each its own CLI subcommand:

``folds``
    Split a validation split file's entry ids into two disjoint folds -- a
    fit fold and a selection fold -- with a recorded seed, so the same seed
    always yields the identical split (no id appears in both).

``fit``
    Read a predictions file (``metrics.py``'s schema) and a folds file, and
    fit a temperature ``T`` and a per-candidate-label vector on the fit
    fold's rows by minimising negative log-likelihood (NLL) of the expected
    label under the model's own ``candidates`` distribution. Refuses a
    predictions file that looks like the test or held-out split -- see
    :func:`refuse_if_test_or_held_out`.

``apply``
    Rescale a predictions file's ``candidates`` with a fitted temperature
    and/or vector, writing a new predictions file in the same schema.

Temperature scaling raises every candidate probability to the power
``1/T`` and renormalises (equivalently: divide the logits by ``T``).
Vector scaling multiplies each candidate label's probability by its own
scale and renormalises. Both operate only on the labels a line actually
offers (a line's ``candidates`` keys), never by operation name -- the
vector is keyed by whatever label ``metrics.py``'s ``expected_label``
would produce for that entry (an operation name, or ``nvsh.tiers.bench``'s
``"(escalate)"`` / ``"(explain)"``), same as the file already uses.

Lines whose ``candidates`` is ``null`` are skipped and counted, never
treated as a zero-confidence row. Likewise, a fit-fold line whose gold
label has no matching candidate at all -- not even under
``metrics.canonical_label``'s escalate-family rollup -- is skipped and
counted (``skipped_gold_absent``) rather than charged a fixed,
temperature-independent penalty: it carries no gradient for the fit.

Fitting minimises NLL in log-space (see ``_row_nll``): a candidate
probability is floored once, on input, and the gold's probability mass is
the sum of every candidate label that rolls up (via
``metrics.canonical_label``) to the gold's own canonical label -- so an
``escalate:<reason>`` candidate counts towards a bare ``(escalate)`` gold.
The renormalised *output* probability is never clipped, so a badly-off
prediction keeps a real, unbounded gradient instead of bottoming out at a
floor that makes it look no worse than one that is merely wrong.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # runnable from any directory


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


metrics = _sibling("metrics")

#: A probability floor so log() and division never see exactly zero.
EPS = 1e-9
#: Default fraction of a split's ids that go to the fit fold (the rest: selection).
DEFAULT_FIT_FRACTION = 0.7
#: Search bounds for a per-coordinate multiplier, in log space (~[0.05, 20]).
_LOG_BOUND = math.log(20.0)


class CalibrationError(ValueError):
    """A refused input: the wrong split, a malformed folds/params file, and so on."""


# ---------------------------------------------------------------------------
# Refusing test / held-out input (mirrors measure.py's check_split_allowed)
# ---------------------------------------------------------------------------


def _stem_words(path: Path) -> set[str]:
    return {word for word in re.split(r"[^a-z0-9]+", path.stem.lower()) if word}


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def split_markers(path: Path, header: object = None) -> set[str]:
    """Which of ``{"test", "held-out"}`` *path* (and *header*, if given) name.

    Mirrors ``measure.py``'s ``check_split_allowed``: the side is read from
    both the file name and, when available, a JSON header, so renaming a
    file alone does not change what it is judged to be. A predictions
    JSONL file (``metrics.py``'s schema) carries no such header, so a fit
    input built from one is judged on its name alone -- the most robust
    signal that format offers. This is a documented limitation: naming a
    held-out predictions file without "test" or "held-out" in it (or
    renaming split.py's own test.json) would defeat the check.
    """
    words = _stem_words(path)
    compact_stem = _compact(path.stem)
    text = (
        header
        if isinstance(header, str)
        else (json.dumps(header, sort_keys=True) if header else "")
    )
    lowered_text = text.lower()
    markers: set[str] = set()
    if "test" in words or re.search(r"\btest\b", lowered_text):
        markers.add("test")
    if "heldout" in compact_stem or "heldout" in _compact(text):
        markers.add("held-out")
    return markers


def refuse_if_test_or_held_out(path: Path, header: object = None) -> None:
    """Raise :class:`CalibrationError` when *path* looks like test or held-out."""
    markers = split_markers(path, header)
    if markers:
        raise CalibrationError(
            f"{path} looks like the {' and '.join(sorted(markers))} split; "
            "calibration is fit on a validation fold only, never test or held-out data"
        )


# ---------------------------------------------------------------------------
# Folds: seeded, disjoint, sorted
# ---------------------------------------------------------------------------


def read_split_ids(path: Path) -> list[str]:
    """The entry ids of a split file (``{"header": ..., "entries": [...]}``)."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise CalibrationError(f"{path} is not a split file ({{header, entries}})")
    ids = []
    for entry in raw["entries"]:
        if not isinstance(entry, dict) or "id" not in entry:
            raise CalibrationError(f"{path}: an entry is missing its id")
        ids.append(str(entry["id"]))
    return ids


def make_folds(
    ids: Iterable[str],
    seed: int,
    fit_fraction: float = DEFAULT_FIT_FRACTION,
    group_of: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split *ids* into ``(fit_ids, selection_ids)``, seeded, disjoint, each sorted.

    Duplicates in *ids* are folded into one id. The same *ids* and *seed*
    always yield the identical split: ids are sorted before shuffling, so
    the result never depends on input order, dict ordering or
    ``PYTHONHASHSEED``.

    *group_of* (optional) maps an id to the key whose members must all land
    on the same side -- e.g. an entry's ``source_id``, so a source's
    variations never get split across the fit and selection folds (#53
    review finding: the per-id shuffle used to do exactly that). When
    omitted, or when an id is absent from the mapping, that id is its own
    group of one, exactly as before.
    """
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must be between 0 and 1 (exclusive), got {fit_fraction!r}")
    unique_sorted = sorted({str(entry_id) for entry_id in ids})
    if not group_of:
        shuffled = list(unique_sorted)
        random.Random(seed).shuffle(shuffled)
        n_fit = max(0, min(len(shuffled), round(len(shuffled) * fit_fraction)))
        fit_ids = sorted(shuffled[:n_fit])
        selection_ids = sorted(shuffled[n_fit:])
        return fit_ids, selection_ids

    members_of_group: dict[str, list[str]] = {}
    for entry_id in unique_sorted:
        key = str(group_of.get(entry_id, entry_id))
        members_of_group.setdefault(key, []).append(entry_id)
    group_keys = sorted(members_of_group)
    shuffled_keys = list(group_keys)
    random.Random(seed).shuffle(shuffled_keys)
    n_fit_groups = max(0, min(len(shuffled_keys), round(len(shuffled_keys) * fit_fraction)))
    fit_keys = set(shuffled_keys[:n_fit_groups])
    fit_ids = sorted(entry_id for key in fit_keys for entry_id in members_of_group[key])
    selection_ids = sorted(
        entry_id for key in group_keys if key not in fit_keys for entry_id in members_of_group[key]
    )
    return fit_ids, selection_ids


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def temperature_scale(candidates: Mapping[str, float], temperature: float) -> dict[str, float]:
    """``p_i ** (1/T)``, renormalised over *candidates*' own labels."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    powered = {label: max(float(p), EPS) ** (1.0 / temperature) for label, p in candidates.items()}
    total = math.fsum(powered.values())
    if total <= 0:
        return {label: 1.0 / len(powered) for label in powered}
    return {label: value / total for label, value in powered.items()}


def vector_scale(candidates: Mapping[str, float], vector: Mapping[str, float]) -> dict[str, float]:
    """Each label's probability times its own scale (default 1.0), renormalised."""
    scaled = {label: max(float(p), EPS) * vector.get(label, 1.0) for label, p in candidates.items()}
    total = math.fsum(scaled.values())
    if total <= 0:
        return {label: 1.0 / len(scaled) for label in scaled}
    return {label: value / total for label, value in scaled.items()}


def apply_scaling(
    candidates: Mapping[str, float],
    temperature: float = 1.0,
    vector: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Temperature scaling, then vector scaling (if given), over one line's candidates."""
    scaled = temperature_scale(candidates, temperature) if temperature != 1.0 else dict(candidates)
    if vector:
        scaled = vector_scale(scaled, vector)
    return scaled


# ---------------------------------------------------------------------------
# Fitting: golden-section search for T, coordinate descent for the vector
# ---------------------------------------------------------------------------


def _logsumexp(values: Iterable[float]) -> float:
    """``log(sum(exp(v) for v in values))``, computed without overflow."""
    values = list(values)
    top = max(values)
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def _gold_matches(label: str, gold_canonical: str) -> bool:
    """Whether *label* rolls up (via :func:`metrics.canonical_label`) to *gold_canonical*."""
    return metrics.canonical_label(label) == gold_canonical


def _row_log_scores(
    candidates: Mapping[str, float],
    temperature: float,
    vector: Mapping[str, float] | None,
) -> dict[str, float]:
    """Unnormalised log-score per label: ``log(max(p, EPS)) / T [+ log(vector)]``.

    The probability floor is applied exactly once, to the *input* ``p`` --
    never to a transformed/renormalised output -- so a candidate that is
    already vanishingly small does not get an artificial, T-independent
    floor imposed on its *rescaled* probability later. See
    :func:`_row_nll`, which turns these into an exact NLL via
    :func:`_logsumexp` instead of clipping a renormalised probability.
    """
    scores = {label: math.log(max(float(p), EPS)) / temperature for label, p in candidates.items()}
    if vector:
        for label in scores:
            scores[label] += math.log(max(float(vector.get(label, 1.0)), EPS))
    return scores


def _row_nll(
    candidates: Mapping[str, float],
    gold: str,
    temperature: float,
    vector: Mapping[str, float] | None = None,
) -> float | None:
    """The exact NLL of *gold* under a temperature/vector-scaled *candidates*.

    *gold*'s probability mass is the sum of every candidate label that rolls
    up to the same canonical label (``metrics.canonical_label``) -- an
    ``escalate:<reason>`` candidate counts towards a bare ``(escalate)``
    gold, matching how :func:`metrics.rollup_escalate_candidates` scores
    everywhere else. The whole computation is done in log-space via
    :func:`_logsumexp`, so the renormalising constant (shared by every
    label) never needs computing on its own and the gold probability is
    never clipped after the fact -- only the raw input probabilities are
    floored, once, in :func:`_row_log_scores`.

    Returns ``None`` when *gold*'s canonical label has no matching
    candidate at all (the line's model output never considered it):
    such a row carries no gradient for *T*/vector and the caller should
    skip and count it rather than charging a fixed, meaningless penalty.
    """
    gold_canonical = metrics.canonical_label(gold)
    scores = _row_log_scores(candidates, temperature, vector)
    matched = [value for label, value in scores.items() if _gold_matches(label, gold_canonical)]
    if not matched:
        return None
    return _logsumexp(scores.values()) - _logsumexp(matched)


def _nll(
    rows: Sequence[tuple[Mapping[str, float], str]],
    temperature: float,
    vector: Mapping[str, float] | None = None,
) -> float:
    """Total negative log-likelihood of each row's gold label.

    Rows whose gold has no matching candidate (see :func:`_row_nll`) are
    skipped; callers that need that count should filter with
    :func:`_rows_with_gold_present` up front, since presence does not
    depend on *temperature* or *vector*.
    """
    total = 0.0
    for candidates, gold in rows:
        row_nll = _row_nll(candidates, gold, temperature, vector)
        if row_nll is not None:
            total += row_nll
    return total


def _rows_with_gold_present(
    rows: Sequence[tuple[Mapping[str, float], str]],
) -> tuple[list[tuple[Mapping[str, float], str]], int]:
    """``(kept_rows, skipped_count)``: drop rows whose gold has no matching candidate.

    Presence is a structural property of a row's ``candidates`` keys and its
    gold label (via :func:`metrics.canonical_label`), independent of any
    fitted temperature or vector, so it is computed once up front rather
    than inside the fitting objective's hot loop.
    """
    kept = []
    skipped = 0
    for candidates, gold in rows:
        gold_canonical = metrics.canonical_label(gold)
        if any(_gold_matches(label, gold_canonical) for label in candidates):
            kept.append((candidates, gold))
        else:
            skipped += 1
    return kept, skipped


def _golden_section_min(
    f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-5
) -> float:
    """The x in [lo, hi] minimising f, by golden-section search (deterministic)."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0
    a, b = lo, hi
    span = b - a
    if span <= tol:
        return (a + b) / 2.0
    n = max(1, int(math.ceil(math.log(tol / span) / math.log(invphi))))
    c = a + invphi2 * span
    d = a + invphi * span
    yc, yd = f(c), f(d)
    for _ in range(n):
        if yc < yd:
            b, d, yd = d, c, yc
            span = invphi * span
            c = a + invphi2 * span
            yc = f(c)
        else:
            a, c, yc = c, d, yd
            span = invphi * span
            d = a + invphi * span
            yd = f(d)
    return (a + b) / 2.0


def fit_temperature(rows: Sequence[tuple[Mapping[str, float], str]]) -> float:
    """The temperature minimising NLL on *rows*, by 1-D search over log(T)."""
    if not rows:
        return 1.0

    def objective(log_t: float) -> float:
        return _nll(rows, math.exp(log_t))

    return math.exp(_golden_section_min(objective, -_LOG_BOUND, _LOG_BOUND))


def fit_vector(
    rows: Sequence[tuple[Mapping[str, float], str]],
    labels: Sequence[str],
    rounds: int = 25,
    tol: float = 1e-6,
) -> dict[str, float]:
    """A per-label scale minimising NLL on *rows*, by coordinate descent.

    Each round visits every label in sorted order and re-optimises its
    scale alone (golden-section search over its log), holding every other
    label's scale fixed, until no coordinate moves by more than *tol* or
    *rounds* is reached. Deterministic: no randomness, a fixed visit order.
    """
    vector = {label: 1.0 for label in labels}
    if not rows or not labels:
        return vector
    ordered = sorted(labels)
    for _ in range(rounds):
        moved = False
        for label in ordered:

            def objective(log_s: float, label=label) -> float:
                trial = dict(vector)
                trial[label] = math.exp(log_s)
                return _nll(rows, 1.0, trial)

            best = math.exp(_golden_section_min(objective, -_LOG_BOUND, _LOG_BOUND))
            if abs(best - vector[label]) > tol:
                moved = True
            vector[label] = best
        if not moved:
            break
    return vector


def load_fit_rows(
    predictions_path: Path, fit_ids: set[str]
) -> tuple[list[tuple[dict, str]], int, int]:
    """``(rows, skipped_null, considered)`` for the fit fold's predictions.

    *rows* pairs each non-null-candidates prediction's ``candidates`` with
    its gold calibration label (``metrics.expected_label``). *considered*
    counts every prediction whose id is in *fit_ids*; *skipped_null* is how
    many of those had ``candidates: null``.
    """
    predictions = metrics.read_predictions(predictions_path)
    rows: list[tuple[dict, str]] = []
    skipped_null = 0
    considered = 0
    for prediction in predictions:
        if prediction.id not in fit_ids:
            continue
        considered += 1
        if prediction.candidates is None:
            skipped_null += 1
            continue
        gold = metrics.expected_label(prediction.expected)
        rows.append((prediction.candidates, gold))
    return rows, skipped_null, considered


def fit_params(predictions_path: Path, folds: Mapping[str, object]) -> dict:
    """Fit temperature and vector on *predictions_path*'s fit fold; the params dict.

    Refuses *predictions_path* when it looks like the test or held-out
    split (see :func:`refuse_if_test_or_held_out`).
    """
    refuse_if_test_or_held_out(predictions_path)
    fit_ids = {str(entry_id) for entry_id in folds.get("fit_ids", [])}
    if not fit_ids:
        raise CalibrationError("the folds file has no fit_ids")
    rows, skipped_null, considered = load_fit_rows(predictions_path, fit_ids)
    if not rows:
        raise CalibrationError(
            f"no fit-fold prediction in {predictions_path} has a non-null candidates distribution"
        )
    # A row whose gold label has no matching candidate at all (not even under
    # metrics.canonical_label's escalate-family rollup) carries no gradient
    # for temperature/vector fitting -- skip and count it rather than
    # charging it a fixed, T-independent penalty. See _rows_with_gold_present.
    rows, skipped_gold_absent = _rows_with_gold_present(rows)
    if not rows:
        raise CalibrationError(
            f"no fit-fold prediction in {predictions_path} offers its gold label as a candidate"
        )
    labels = sorted({label for candidates, _ in rows for label in candidates})
    temperature = fit_temperature(rows)
    # The vector is fit on top of the already-temperature-scaled rows, so it
    # matches apply_scaling's chain (temperature, then vector) exactly: it
    # corrects what the temperature alone could not (per-label residual bias),
    # rather than being fit against a different starting point than apply()
    # will actually feed it.
    temperature_scaled_rows = [
        (temperature_scale(candidates, temperature), gold) for candidates, gold in rows
    ]
    vector = fit_vector(temperature_scaled_rows, labels)
    return {
        "seed": folds.get("seed"),
        "temperature": temperature,
        "vector": vector,
        "fit_examples": len(rows),
        "skipped_null": skipped_null,
        "skipped_gold_absent": skipped_gold_absent,
        "fit_fold_size": considered,
        "predictions_source": str(predictions_path),
    }


# ---------------------------------------------------------------------------
# Apply: rescale a predictions file
# ---------------------------------------------------------------------------


def apply_to_predictions(predictions_path: Path, params: Mapping[str, object]) -> list[dict]:
    """One rescaled row per line of *predictions_path*, in ``metrics.py``'s schema."""
    temperature = float(params.get("temperature", 1.0))
    vector = params.get("vector") or {}
    predictions = metrics.read_predictions(predictions_path)
    rows = []
    for prediction in predictions:
        row = {field: getattr(prediction, field) for field in metrics.FIELDS}
        row["invalid_reason"] = prediction.invalid_reason
        if prediction.candidates is not None:
            row["candidates"] = apply_scaling(prediction.candidates, temperature, vector)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Evaluate: raw vs temperature vs temperature + vector on one fold
# ---------------------------------------------------------------------------

#: The three variants :func:`evaluate` compares, in this order.
EVALUATED_VARIANTS = ("raw", "temperature", "temperature+vector")


def evaluate(
    predictions_path: Path, params: Mapping[str, object], folds: Mapping[str, object], fold: str
) -> dict:
    """ECE/Brier (``metrics.compute_calibration``, bootstrap CIs) on *fold*'s ids only.

    Three variants of the same lines: the model's own distributions, the
    fitted temperature alone, and temperature then vector -- so the
    selection fold decides whether vector scaling earns its extra
    parameters (issue 53 t17/t19). Evaluating on the fit fold is allowed but
    reported as such; it says nothing about generalisation.
    """
    key = {"fit": "fit_ids", "selection": "selection_ids"}.get(fold)
    if key is None:
        raise CalibrationError(f"--fold must be fit or selection, not {fold!r}")
    ids = set(folds.get(key) or [])
    if not ids:
        raise CalibrationError(f"the folds file has no {key}")
    predictions = [p for p in metrics.read_predictions(predictions_path) if p.id in ids]
    if not predictions:
        raise CalibrationError(f"no predictions line has an id in the {fold} fold")
    temperature = float(params.get("temperature", 1.0))
    vector = params.get("vector") or {}
    scalings = {
        "raw": (1.0, {}),
        "temperature": (temperature, {}),
        "temperature+vector": (temperature, vector),
    }
    report: dict = {"fold": fold, "n": len(predictions), "temperature": temperature, "variants": {}}
    for name in EVALUATED_VARIANTS:
        t, v = scalings[name]
        scaled = [
            (
                p
                if p.candidates is None
                else replace(p, candidates=apply_scaling(p.candidates, t, v))
            )
            for p in predictions
        ]
        block = metrics.compute_calibration(scaled)
        block.pop("bins", None)
        report["variants"][name] = block
    return report


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_folds(args: argparse.Namespace) -> int:
    split_path = Path(args.split)
    with open(split_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    header = raw.get("header") if isinstance(raw, dict) else None
    try:
        refuse_if_test_or_held_out(split_path, header)
        ids = read_split_ids(split_path)
        fit_ids, selection_ids = make_folds(ids, args.seed, args.fit_fraction)
    except (CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_json(
        Path(args.out),
        {
            "seed": args.seed,
            "source": str(split_path),
            "fit_ids": fit_ids,
            "selection_ids": selection_ids,
        },
    )
    print(f"fit={len(fit_ids)} selection={len(selection_ids)}")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    predictions_path = Path(args.predictions)
    with open(args.folds, encoding="utf-8") as handle:
        folds = json.load(handle)
    try:
        params = fit_params(predictions_path, folds)
    except (CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_json(Path(args.out), params)
    print(f"temperature={params['temperature']:.4f} examples={params['fit_examples']}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    predictions_path = Path(args.predictions)
    with open(args.params, encoding="utf-8") as handle:
        params = json.load(handle)
    try:
        rows = apply_to_predictions(predictions_path, params)
    except (CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_jsonl(Path(args.out), rows)
    print(f"rescaled {len(rows)} line(s)")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    with open(args.params, encoding="utf-8") as handle:
        params = json.load(handle)
    with open(args.folds, encoding="utf-8") as handle:
        folds = json.load(handle)
    try:
        report = evaluate(Path(args.predictions), params, folds, args.fold)
    except (CalibrationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        write_json(Path(args.out), report)
    for name in EVALUATED_VARIANTS:
        block = report["variants"][name]
        print(f"{name}: ece={block['ece']:.4f} brier={block['brier']:.4f} n={block['n']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    folds_parser = sub.add_parser("folds", help="split a validation file's ids into fit/selection")
    folds_parser.add_argument("--split", required=True)
    folds_parser.add_argument("--seed", type=int, required=True)
    folds_parser.add_argument("--out", required=True)
    folds_parser.add_argument("--fit-fraction", type=float, default=DEFAULT_FIT_FRACTION)
    folds_parser.set_defaults(func=_cmd_folds)

    fit_parser = sub.add_parser("fit", help="fit temperature and vector scaling on the fit fold")
    fit_parser.add_argument("--predictions", required=True)
    fit_parser.add_argument("--folds", required=True)
    fit_parser.add_argument("--out", required=True)
    fit_parser.set_defaults(func=_cmd_fit)

    apply_parser = sub.add_parser("apply", help="rescale a predictions file with fitted params")
    apply_parser.add_argument("--predictions", required=True)
    apply_parser.add_argument("--params", required=True)
    apply_parser.add_argument("--out", required=True)
    apply_parser.set_defaults(func=_cmd_apply)

    eval_parser = sub.add_parser(
        "evaluate", help="ECE/Brier on one fold: raw vs temperature vs temperature+vector"
    )
    eval_parser.add_argument("--predictions", required=True)
    eval_parser.add_argument("--params", required=True)
    eval_parser.add_argument("--folds", required=True)
    eval_parser.add_argument("--fold", default="selection", choices=("fit", "selection"))
    eval_parser.add_argument("--out", default=None)
    eval_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
