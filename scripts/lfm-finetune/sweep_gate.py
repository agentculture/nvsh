#!/usr/bin/env python3
"""Offline threshold sweep for the uncertainty gate (issue 53): no GPU, no scorer call.

A development-machine tool. It is NEVER imported by the nvsh package --
nothing under ``nvsh/`` may depend on it. Stdlib only.

Re-decides every line of an existing predictions file (``metrics.py``'s
schema) under a grid of :class:`gate.Thresholds`, using each line's already
recorded ``candidates`` distribution -- no model call, so the sweep runs
offline on a laptop from a predictions file a GPU run produced earlier.
For each threshold set it reports ``right_proposals``, ``wrong_mutating``,
escalation recall/precision (including the escalation recall specifically
on missing-candidate lines -- ids ending ``-nocand`` or whose gold label
was never offered), the ``abstain_uncertain`` count and the escalation
false-positive count, by feeding the re-decided lines back through
``metrics.compute`` (issue 53 t5) rather than re-implementing any of that
counting here.

With every grid value left at its default (``none``), the sweep runs
exactly one threshold set -- :class:`gate.Thresholds` with everything
``None`` -- which reproduces each line's own stored argmax decision (see
``gate.decide``'s docstring and this module's ``--final`` sanity mode
below).

**Split safety.** Like ``calibration_fit.py``, this script never sweeps a
predictions file that looks like the test or held-out split, or a "final"
evaluation artifact, unless ``--final`` is passed -- see
:func:`refuse_unless_final`.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # runnable from any directory, for nvsh.*


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


gate = _sibling("gate")
metrics = gate.metrics  # the same loaded metrics module gate.py already imported
calibration_fit = _sibling("calibration_fit")

#: A grid value spelled this way (case-insensitive) means "threshold disabled".
NONE_SPELLING = "none"
#: The default grid for every numeric flag: a single, disabled value.
DEFAULT_GRID = (NONE_SPELLING,)
#: A path whose name or any directory component contains this word is
#: treated as a final/held-out evaluation artifact requiring ``--final``,
#: in addition to calibration_fit's own test/held-out name markers.
FINAL_MARKER = "final"


class SweepError(ValueError):
    """A refused input file, or a malformed folds/grid argument."""


# ---------------------------------------------------------------------------
# Split safety
# ---------------------------------------------------------------------------


def split_markers(path: Path) -> set[str]:
    """Which of ``{"test", "held-out", "final"}`` *path* names.

    Reuses ``calibration_fit.split_markers`` for the "test"/"held-out"
    name check (a predictions JSONL file carries no split header, so this
    is judged on the file name alone, same documented limitation as
    ``calibration_fit``'s own use of it) and additionally flags any path
    with a "final" word in its stem or in one of its directory names --
    scorer-b1's exact-run predictions file (``final-scorer-b1-exact-...``,
    under a ``final/`` directory) is exactly this shape.
    """
    markers = set(calibration_fit.split_markers(path))
    words = {word for word in path.stem.lower().replace("-", " ").replace("_", " ").split()}
    words |= {part.lower() for part in path.parts}
    if FINAL_MARKER in words:
        markers.add(FINAL_MARKER)
    return markers


def refuse_unless_final(path: Path, final: bool) -> None:
    """Raise :class:`SweepError` when *path* looks final/test/held-out and *final* is false."""
    if final:
        return
    markers = split_markers(path)
    if markers:
        raise SweepError(
            f"{path} looks like {' and '.join(sorted(markers))} data; pass --final to sweep it"
        )


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def load_folds(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fold_ids(folds: Mapping, fold: str) -> set[str]:
    key = {"fit": "fit_ids", "selection": "selection_ids"}.get(fold)
    if key is None:
        raise SweepError(f"fold must be 'fit' or 'selection', got {fold!r}")
    ids = folds.get(key)
    if not isinstance(ids, list):
        raise SweepError(f"the folds file has no {key!r}")
    return {str(entry_id) for entry_id in ids}


def filter_by_fold(predictions: Sequence, folds: Mapping | None, fold: str | None) -> list:
    if folds is None:
        return list(predictions)
    ids = fold_ids(folds, fold or "fit")
    return [p for p in predictions if p.id in ids]


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def parse_grid(text: str) -> list[float | None]:
    """A comma-separated list of floats and/or ``"none"`` -> the parsed grid values."""
    values: list[float | None] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == NONE_SPELLING:
            values.append(None)
        else:
            try:
                values.append(float(item))
            except ValueError as exc:
                raise SweepError(f"not a number or 'none': {item!r}") from exc
    if not values:
        raise SweepError("a grid must name at least one value")
    return values


def _grid_or_default(values: Sequence[float | None] | None, fallback: Sequence[float | None]):
    return list(values) if values is not None else list(fallback)


def build_threshold_grid(
    *,
    escalate: Sequence[float | None],
    ro_floor: Sequence[float | None],
    ro_margin: Sequence[float | None],
    ro_max_entropy: Sequence[float | None],
    mut_floor: Sequence[float | None] | None = None,
    mut_margin: Sequence[float | None] | None = None,
    mut_max_entropy: Sequence[float | None] | None = None,
) -> list["gate.Thresholds"]:
    """Every combination of the given grids as a list of :class:`gate.Thresholds`.

    ``mut_*`` defaults to the corresponding ``ro_*`` grid when omitted, so a
    caller who wants one shared floor/margin/entropy grid across both
    operation classes only has to pass the ``ro_*`` values.
    """
    mut_floor = _grid_or_default(mut_floor, ro_floor)
    mut_margin = _grid_or_default(mut_margin, ro_margin)
    mut_max_entropy = _grid_or_default(mut_max_entropy, ro_max_entropy)
    grid = []
    for esc, rf, rm, rx, mf, mm, mx in itertools.product(
        escalate, ro_floor, ro_margin, ro_max_entropy, mut_floor, mut_margin, mut_max_entropy
    ):
        grid.append(
            gate.Thresholds(
                escalate=esc,
                read_only=gate.ThresholdSet(floor=rf, margin=rm, max_entropy=rx),
                mutating=gate.ThresholdSet(floor=mf, margin=mm, max_entropy=mx),
            )
        )
    return grid


# ---------------------------------------------------------------------------
# Re-deciding
# ---------------------------------------------------------------------------


def redecide(prediction, thresholds: "gate.Thresholds"):
    """*prediction*, a ``metrics.Prediction``, with its outcome/operation re-decided.

    Lines with no recorded distribution (``candidates`` is ``null``) cannot
    be gated -- the gate has nothing to score -- so they pass through
    unchanged. Every field but ``outcome``/``operation``/``arguments`` is
    kept as recorded; ``arguments`` is kept when the re-decided outcome is
    still ``propose`` for the same label (the gate never re-grounds
    arguments, it only decides whether to trust the argmax the scorer
    already grounded), and cleared otherwise.
    """
    if prediction.candidates is None:
        return prediction
    offered = list(prediction.candidates.keys())
    decision = gate.decide(prediction.candidates, offered, thresholds)
    if decision.outcome == "propose":
        same_label = prediction.outcome == "propose" and prediction.operation == decision.label
        arguments = dict(prediction.arguments) if same_label and prediction.arguments else {}
        operation = decision.label
    else:
        arguments = None
        operation = None
    return metrics.Prediction(
        id=prediction.id,
        expected=prediction.expected,
        outcome=decision.outcome,
        operation=operation,
        arguments=arguments,
        candidates=prediction.candidates,
        tokens=prediction.tokens,
        ttfd_ms=prediction.ttfd_ms,
        latency_ms=prediction.latency_ms,
        invalid_reason=None,
    )


# ---------------------------------------------------------------------------
# Evaluating one threshold set
# ---------------------------------------------------------------------------


def _missing_candidate_recall(originals: Sequence, redecided: Sequence) -> dict:
    """Among lines whose gold label was missing from the offered candidates.

    the fraction the re-decided outcome escalated (semantically or by
    abstention) -- the gate's whole point on that slice.
    """
    hits = 0
    total = 0
    for original, new in zip(originals, redecided):
        if not metrics.is_missing_candidate(original):
            continue
        total += 1
        if new.outcome in ("escalate", "abstain_uncertain"):
            hits += 1
    rate = (hits / total) if total else None
    return {"n": hits, "N": total, "rate": rate}


def evaluate(originals: Sequence, thresholds: "gate.Thresholds") -> dict:
    """One threshold set's report row, over *originals* (a list of ``metrics.Prediction``)."""
    redecided = [redecide(p, thresholds) for p in originals]
    computed = metrics.compute(redecided)
    return {
        "thresholds": thresholds.to_json(),
        "right_proposals": computed["right_proposals"],
        "wrong_mutating": computed["wrong_mutating"],
        "abstain_uncertain_count": computed["outcome_counts"]["abstain_uncertain"],
        "escalation": {
            "tp": computed["escalation"]["tp"],
            "fn": computed["escalation"]["fn"],
            "fp": computed["escalation"]["fp"],
            "recall": computed["escalation"]["recall"],
            "precision": computed["escalation"]["precision"],
        },
        "false_positives": computed["escalation"]["fp"],
        "missing_candidate_escalation_recall": _missing_candidate_recall(originals, redecided),
    }


def run_sweep(originals: Sequence, grid: Sequence["gate.Thresholds"]) -> list[dict]:
    return [evaluate(originals, thresholds) for thresholds in grid]


# ---------------------------------------------------------------------------
# Exact-reproduction check (the --final sanity mode leans on this too)
# ---------------------------------------------------------------------------


def count_argmax_matches(originals: Sequence) -> tuple[int, int]:
    """``(matches, total)`` re-deciding every line under all-disabled thresholds.

    A line with no recorded distribution is skipped (there is nothing to
    re-decide); it counts toward neither *matches* nor *total*.
    """
    matches = 0
    total = 0
    for original in originals:
        if original.candidates is None:
            continue
        total += 1
        new = redecide(original, gate.Thresholds())
        if new.outcome == original.outcome and new.operation == original.operation:
            matches += 1
    return matches, total


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def report_markdown(reports: Iterable[dict]) -> str:
    lines = [
        "| escalate | ro floor/margin/entropy | mut floor/margin/entropy | right | "
        "wrong mutating | abstain_uncertain | esc recall | esc fp | missing-cand recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in reports:
        t = row["thresholds"]
        ro = t["read_only"]
        mut = t["mutating"]
        right = row["right_proposals"]
        missing = row["missing_candidate_escalation_recall"]
        recall = row["escalation"]["recall"]
        lines.append(
            "| {esc} | {rf}/{rm}/{rx} | {mf}/{mm}/{mx} | {right_n}/{right_N} | {wrong} | "
            "{abstain} | {recall} | {fp} | {mn}/{mN} |".format(
                esc=t["escalate"],
                rf=ro["floor"],
                rm=ro["margin"],
                rx=ro["max_entropy"],
                mf=mut["floor"],
                mm=mut["margin"],
                mx=mut["max_entropy"],
                right_n=right["n"],
                right_N=right["N"],
                wrong=row["wrong_mutating"]["total"],
                abstain=row["abstain_uncertain_count"],
                recall="-" if recall is None else f"{recall:.3f}",
                fp=row["false_positives"],
                mn=missing["n"],
                mN=missing["N"],
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _grid_arg(parser: argparse.ArgumentParser, *names: str) -> None:
    for name in names:
        parser.add_argument(f"--{name}", default=NONE_SPELLING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--fold", choices=("fit", "selection"), default="fit")
    parser.add_argument(
        "--final", action="store_true", help="allow a test/held-out/final predictions file"
    )
    _grid_arg(parser, "escalate", "floor", "margin", "max-entropy")
    parser.add_argument("--mutating-floor")
    parser.add_argument("--mutating-margin")
    parser.add_argument("--mutating-max-entropy")
    parser.add_argument("--out", help="write the JSON report here (default: stdout)")
    parser.add_argument("--markdown", help="also write a markdown table here")
    args = parser.parse_args(argv)

    predictions_path = Path(args.predictions)
    try:
        refuse_unless_final(predictions_path, args.final)
        originals = metrics.read_predictions(predictions_path)
        folds = load_folds(Path(args.folds)) if args.folds else None
        selected = filter_by_fold(originals, folds, args.fold)
        grid = build_threshold_grid(
            escalate=parse_grid(args.escalate),
            ro_floor=parse_grid(args.floor),
            ro_margin=parse_grid(args.margin),
            ro_max_entropy=parse_grid(getattr(args, "max_entropy")),
            mut_floor=parse_grid(args.mutating_floor) if args.mutating_floor else None,
            mut_margin=parse_grid(args.mutating_margin) if args.mutating_margin else None,
            mut_max_entropy=(
                parse_grid(args.mutating_max_entropy) if args.mutating_max_entropy else None
            ),
        )
        reports = run_sweep(selected, grid)
    except (SweepError, metrics.MetricsError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "predictions": str(predictions_path),
        "fold": args.fold if folds is not None else None,
        "n": len(selected),
        "reports": reports,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.markdown:
        Path(args.markdown).write_text(report_markdown(reports) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
