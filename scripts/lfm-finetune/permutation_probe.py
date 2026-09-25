#!/usr/bin/env python3
"""Permutation probe for the Track B candidate scorer (issue 53, t7).

A development-machine tool. It is NEVER imported by the nvsh package --
nothing under ``nvsh/`` may depend on it. Stdlib only (the real scorer is
built through ``measure.py``'s existing ``ScorerSpec``/``build_scorer``
seam, reused here rather than duplicated).

For every entry in a split/corpus file (``tier_bench.load_corpus``'s
schema), this asks the scorer once with the default fixed order and letter
map (``scorer.py``'s ``labels_for`` default -- the *baseline*), then
``--per-entry`` (default 10) seeded perturbations of each of five kinds,
each built through ``scorer.py``'s own seam (:func:`scorer.permute`,
``scorer.prompt_messages``'s ``labels``/``order``/``descriptions``
parameters):

``order``
    Same name -> letter map as the baseline; the listing order is shuffled.
``letters``
    Same listing order as the baseline; the name -> letter map is
    re-drawn (so a candidate may land on a different letter, upper or
    lower).
``subset``
    The baseline order and letter map, restricted to a random subset that
    always keeps the gold candidate. How often that subset also drops the
    baseline's own choice is reported separately (a formerly-correct or
    formerly-wrong choice going missing is not itself an "answer change",
    but it is worth knowing).
``paraphrase``
    The baseline order and letter map; one random alternative description
    (from ``--paraphrases``' JSON, ``{candidate: [alt description, ...]}``)
    replaces each candidate's default description. A candidate the file
    does not cover keeps its default description.
``all``
    Order, letters and subset combined in one draw (:func:`scorer.permute`
    with ``subset=``), the OpenJev-style full shuffle.

An *answer change* is when a trial's op-level choice
(``scorer.same_choice``, compared by candidate name, never by letter)
differs from the baseline's choice for that entry. Every seed is derived
deterministically from ``(--seed, entry.id, kind, trial index)``, so a
report is exactly reproducible.

Trials of the same entry are correlated (one entry, many draws), so the
reported 95% CI is a bootstrap over **entries**, not trials --
:func:`metrics.bootstrap_ci` resampling each kind's per-entry
``(trials, changes)`` pairs, never the flat trial list. This is called out
in the report itself.

The scorer never sees a canonicalised order or letter map: whatever a
kind's draw computes is exactly what is rendered and scored, including for
``order``/``all``, which are the whole point of the probe.

Split guard: refuses a split file that looks like the test or held-out
side (:func:`calibration_fit.split_markers`, mirrored from
``measure.py``'s own guard) unless ``--final`` is given -- the same rule
``measure.py`` and ``calibration_fit.py`` already apply.

``--reasons`` probes a scorer trained with ``build_dataset.py --reasons``:
the pool is ``scorer.candidate_pool(True)`` (the 8 ``escalate:<reason>``
candidates in place of the bare escalate), the baseline letters each
candidate by its position in that pool, every prompt carries the reason
candidates' ``data/reasons.json`` descriptions, and an escalation's gold is
its own reason (``scorer.reason_for_class``). The report records it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # runnable from any directory


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


scorer = _sibling("scorer")
metrics = _sibling("metrics")
calibration_fit = _sibling("calibration_fit")

from nvsh.tiers import bench as tier_bench  # noqa: E402
from nvsh.tiers import lfm as tier_lfm  # noqa: E402

#: The five perturbation kinds, in report order.
KINDS = ("order", "letters", "subset", "paraphrase", "all")

DEFAULT_PER_ENTRY = 10
DEFAULT_SEED = 0


class ProbeError(ValueError):
    """A refused input: the wrong split, a malformed paraphrase file, and so on."""


# ---------------------------------------------------------------------------
# Gold / entry helpers
# ---------------------------------------------------------------------------


def gold_name(
    expect: Mapping[str, object], *, reasons: bool = False, cls: str | None = None
) -> str:
    """The candidate *name* (never a calibration label) the corpus entry expects.

    Mirrors ``metrics.expected_label`` but returns the scorer's own candidate
    name (``scorer.Scored.choice``'s vocabulary: an operation name, or
    ``tier_lfm.EXPLAIN_TOOL`` / ``tier_lfm.ESCALATE_TOOL``), never bench's
    calibration label -- ``scorer.same_choice`` compares choices by name.
    With *reasons*, an escalation's gold is its ``escalate:<reason>``
    candidate, read from the entry's *cls* (``scorer.reason_for_class``, as
    ``build_dataset.py --reasons`` does).
    """
    kind = metrics.expect_kind(expect)
    if kind == "escalate":
        return scorer.reason_for_class(cls) if reasons else tier_lfm.ESCALATE_TOOL
    if kind == "explain":
        return tier_lfm.EXPLAIN_TOOL
    return str(expect["operation"])


def load_paraphrases(path: Path) -> dict[str, list[str]]:
    """``{candidate: [alt description, ...]}`` from *path*; refuses a malformed file.

    Every candidate's list must have at least 2 alternatives (t14's schema),
    but a probe run never requires every candidate to be covered -- a
    candidate the file omits simply keeps its default description.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeError(f"cannot read paraphrase file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProbeError(f"{path} is not a {{candidate: [alt description, ...]}} object")
    result: dict[str, list[str]] = {}
    for name, alts in raw.items():
        if not isinstance(alts, list) or not all(isinstance(alt, str) for alt in alts):
            raise ProbeError(f"{path}: {name!r}'s alternatives must be a list of strings")
        if len(alts) < 2:
            raise ProbeError(f"{path}: {name!r} has fewer than 2 alternative descriptions")
        result[name] = list(alts)
    return result


# ---------------------------------------------------------------------------
# Seeded draws per kind
# ---------------------------------------------------------------------------


def derive_seed(seed: object, entry_id: str, kind: str, index: int) -> str:
    """One deterministic, reproducible seed for (*seed*, *entry_id*, *kind*, *index*)."""
    return f"{seed}:{entry_id}:{kind}:{index}"


def _random_subset_order(
    rng: random.Random, order: Sequence[str], keep: Sequence[str]
) -> list[str]:
    """A random subset of *order*, in *order*'s own order, always keeping *keep*.

    The subset size is itself randomly drawn (from ``len(keep)`` to
    ``len(order)``), so different trials probe different subset sizes.
    """
    keep_set = dict.fromkeys(keep)
    rest = [name for name in order if name not in keep_set]
    size = rng.randint(len(keep_set), len(order))
    extra_count = max(0, min(size - len(keep_set), len(rest)))
    extra = set(rng.sample(rest, extra_count)) if extra_count else set()
    chosen = set(keep_set) | extra
    return [name for name in order if name in chosen]


@dataclass(frozen=True)
class Trial:
    """One perturbation's rendered order/labels/descriptions, ready to score."""

    order: tuple[str, ...]
    labels: dict[str, str]
    descriptions: dict[str, str] | None = None
    #: ``subset`` only: whether this draw dropped the baseline's own choice.
    baseline_choice_dropped: bool | None = None


def build_trial(
    kind: str,
    seed: object,
    pool: Sequence[str],
    baseline_order: Sequence[str],
    baseline_labels: Mapping[str, str],
    gold: str,
    baseline_choice: str | None,
    paraphrases: Mapping[str, Sequence[str]] | None,
) -> Trial:
    """The *kind* perturbation drawn from *seed*, isolating exactly what that kind varies."""
    if kind == "order":
        order = scorer.permute(seed, pool=pool).order
        return Trial(order=order, labels=dict(baseline_labels))
    if kind == "letters":
        labels = scorer.permute(seed, pool=pool).labels
        return Trial(order=tuple(baseline_order), labels=labels)
    if kind == "subset":
        rng = random.Random(seed)  # nosec B311 - probe sampling, not security
        order = _random_subset_order(rng, baseline_order, keep=(gold,))
        labels = {name: baseline_labels[name] for name in order}
        dropped = baseline_choice is not None and baseline_choice not in order
        return Trial(order=tuple(order), labels=labels, baseline_choice_dropped=dropped)
    if kind == "paraphrase":
        rng = random.Random(seed)  # nosec B311 - probe sampling, not security
        descriptions: dict[str, str] = {}
        for name in baseline_order:
            alts = (paraphrases or {}).get(name)
            if alts:
                descriptions[name] = rng.choice(list(alts))
        return Trial(
            order=tuple(baseline_order), labels=dict(baseline_labels), descriptions=descriptions
        )
    if kind == "all":
        rng = random.Random(seed)  # nosec B311 - probe sampling, not security
        keep = (gold,)
        subset_size = rng.randint(len(keep), len(pool))
        perm = scorer.permute(seed, pool=pool, subset=subset_size, keep=keep)
        return Trial(order=perm.order, labels=perm.labels)
    raise ProbeError(f"unknown perturbation kind: {kind!r}")


# ---------------------------------------------------------------------------
# Scoring one entry
# ---------------------------------------------------------------------------


class NextTokenScorer:
    """Structural stand-in for ``scorer.NextTokenScorer`` (documents the seam only)."""


@dataclass
class EntryOutcome:
    """One entry's baseline plus every kind's trials."""

    entry_id: str
    gold: str
    baseline_choice: str | None
    #: kind -> list of (changed, has_lowercase, baseline_choice_dropped-or-None, incomplete)
    #: ``incomplete`` is true when the trial's (or the baseline's) result was missing
    #: labels or made no choice at all -- it is tallied separately, never folded into
    #: ``changed`` (issue 53: a scorer that cannot read a trial's letters is not evidence
    #: of an answer change).
    trials: dict[str, list[tuple[bool, bool, bool | None, bool]]] = field(default_factory=dict)


def _score(
    scorer_obj,
    render: Callable[[list[dict]], str],
    request_text: str,
    order: Sequence[str],
    labels: Mapping[str, str],
    descriptions: Mapping[str, str] | None,
):
    messages = scorer.prompt_messages(
        request_text, order, labels=labels, order=order, descriptions=descriptions
    )
    prompt = render(messages)
    return scorer.score(scorer_obj, prompt, request_text, offered=order, labels=labels, order=order)


def _incomplete(scored) -> bool:
    """True when *scored* made no comparable choice: missing labels, or no label had mass.

    A scorer that cannot read the letters a trial actually offered (e.g. an
    in-process scorer built for a narrower alphabet than the trial's draw --
    issue 53) reports this via ``Scored.incomplete`` (some labels missing) or
    ``Scored.choice is None`` (no label had any mass, or the call failed). A
    trial like this is not evidence either way about the model's answer, so
    it is never compared to the baseline.
    """
    return scored.choice is None or scored.incomplete is not None


def probe_entry(
    scorer_obj,
    render: Callable[[list[dict]], str],
    entry,
    *,
    pool: Sequence[str],
    per_entry: int,
    seed: object,
    paraphrases: Mapping[str, Sequence[str]] | None,
    kinds: Sequence[str] = KINDS,
    reasons: bool = False,
) -> EntryOutcome:
    """Baseline plus *per_entry* trials of each of *kinds* for one corpus entry.

    With *reasons*, *pool* is ``scorer.candidate_pool(True)``: the baseline
    letters each candidate by its position in it (``build_dataset.py
    --reasons``' default map) and every prompt carries the reason
    candidates' descriptions (a paraphrase trial's own overrides on top).
    """
    request_text = tier_lfm.request_message(
        tier_bench.request_for(entry), tier_bench.context_for(entry)
    )
    gold = gold_name(entry.expect, reasons=reasons, cls=entry.phrasing)
    if reasons:
        baseline_labels = scorer.positional_labels(pool, pool)
        base_descriptions: dict[str, str] | None = scorer.reason_descriptions(pool)
    else:
        baseline_labels = scorer.labels_for(pool)
        base_descriptions = None
    baseline_order = list(pool)
    baseline = _score(
        scorer_obj, render, request_text, baseline_order, baseline_labels, base_descriptions
    )
    baseline_incomplete = _incomplete(baseline)
    outcome = EntryOutcome(entry_id=entry.id, gold=gold, baseline_choice=baseline.choice)
    for kind in kinds:
        trials: list[tuple[bool, bool, bool | None, bool]] = []
        for index in range(per_entry):
            if kind == "paraphrase" and not paraphrases:
                break
            trial_seed = derive_seed(seed, entry.id, kind, index)
            trial = build_trial(
                kind,
                trial_seed,
                pool,
                baseline_order,
                baseline_labels,
                gold,
                baseline.choice,
                paraphrases,
            )
            descriptions = trial.descriptions
            if base_descriptions is not None:
                descriptions = {**base_descriptions, **(trial.descriptions or {})}
            scored = _score(
                scorer_obj, render, request_text, trial.order, trial.labels, descriptions
            )
            # A baseline or trial the scorer could not fully read is not evidence of an
            # answer change either way -- tallied separately (kind_report), never scored.
            incomplete = baseline_incomplete or _incomplete(scored)
            changed = False if incomplete else not scorer.same_choice(scored, baseline)
            has_lowercase = any(letter.islower() for letter in trial.labels.values())
            trials.append((changed, has_lowercase, trial.baseline_choice_dropped, incomplete))
        if trials:
            outcome.trials[kind] = trials
    return outcome


# ---------------------------------------------------------------------------
# Aggregation: bootstrap over entries, not trials
# ---------------------------------------------------------------------------


def _rate_stat(pairs: Sequence[tuple[int, int]]) -> float | None:
    """Aggregate change rate over resampled *pairs* of (trials, changes)."""
    total_trials = sum(trials for trials, _ in pairs)
    if total_trials == 0:
        return None
    total_changes = sum(changes for _, changes in pairs)
    return total_changes / total_trials


def kind_report(outcomes: Sequence[EntryOutcome], kind: str, *, bootstrap_seed: int) -> dict | None:
    """One kind's report: trials, changes, rate with a bootstrap CI over entries, label case.

    A trial the scorer could not fully read (``incomplete`` -- missing labels, or no label
    had mass at all) never enters ``trials``/``changes``/``rate``/``label_case``: it is
    tallied on its own under ``incomplete`` instead, so a scorer built for too narrow an
    alphabet cannot masquerade as a real answer-change rate (issue 53).
    """
    per_entry_pairs: list[tuple[int, int]] = []
    lowercase_trials = lowercase_changes = 0
    uppercase_trials = uppercase_changes = 0
    baseline_dropped = 0
    subset_trials_seen = 0
    incomplete_count = 0
    raw_trial_count = 0
    for outcome in outcomes:
        trials = outcome.trials.get(kind)
        if not trials:
            continue
        raw_trial_count += len(trials)
        incomplete_count += sum(1 for *_, incomplete in trials if incomplete)
        scored_trials = [
            (changed, has_lowercase, dropped)
            for changed, has_lowercase, dropped, incomplete in trials
            if not incomplete
        ]
        if scored_trials:
            changes = sum(1 for changed, _, _ in scored_trials if changed)
            per_entry_pairs.append((len(scored_trials), changes))
        for changed, has_lowercase, dropped in scored_trials:
            if has_lowercase:
                lowercase_trials += 1
                lowercase_changes += int(changed)
            else:
                uppercase_trials += 1
                uppercase_changes += int(changed)
            if dropped is not None:
                subset_trials_seen += 1
                baseline_dropped += int(dropped)
    if raw_trial_count == 0:
        return None
    ci = metrics.bootstrap_ci(per_entry_pairs, _rate_stat, seed=bootstrap_seed)
    total_trials = sum(t for t, _ in per_entry_pairs)
    total_changes = sum(c for _, c in per_entry_pairs)
    report = {
        "kind": kind,
        "entries": len(per_entry_pairs),
        "trials": total_trials,
        "changes": total_changes,
        "rate": ci["value"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "bootstrap_note": "95% CI is a bootstrap over entries, not trials (trials of one"
        " entry are correlated)",
        "incomplete": {
            "trials": incomplete_count,
            "of": raw_trial_count,
            "rate": (incomplete_count / raw_trial_count) if raw_trial_count else None,
        },
        "label_case": {
            "lowercase_trials": lowercase_trials,
            "lowercase_changes": lowercase_changes,
            "lowercase_rate": (lowercase_changes / lowercase_trials) if lowercase_trials else None,
            "uppercase_trials": uppercase_trials,
            "uppercase_changes": uppercase_changes,
            "uppercase_rate": (uppercase_changes / uppercase_trials) if uppercase_trials else None,
        },
    }
    if kind == "subset":
        report["baseline_choice_removed"] = {
            "trials": subset_trials_seen,
            "removed": baseline_dropped,
            "rate": (baseline_dropped / subset_trials_seen) if subset_trials_seen else None,
        }
    return report


def run_probe(
    scorer_obj,
    render: Callable[[list[dict]], str],
    entries: Sequence,
    *,
    pool: Sequence[str] | None = None,
    per_entry: int = DEFAULT_PER_ENTRY,
    seed: object = DEFAULT_SEED,
    paraphrases: Mapping[str, Sequence[str]] | None = None,
    bootstrap_seed: int = metrics.DEFAULT_BOOTSTRAP_SEED,
    reasons: bool = False,
) -> dict:
    """The full probe report: every kind's aggregate, plus per-entry detail.

    *reasons* probes a scorer trained with ``build_dataset.py --reasons``: the
    default pool is then ``scorer.candidate_pool(True)`` (no bare escalate).
    """
    resolved_pool = tuple(scorer.candidate_pool(reasons) if pool is None else pool)
    outcomes = [
        probe_entry(
            scorer_obj,
            render,
            entry,
            pool=resolved_pool,
            per_entry=per_entry,
            seed=seed,
            paraphrases=paraphrases,
            reasons=reasons,
        )
        for entry in entries
    ]
    kinds_report = [
        report
        for kind in KINDS
        if (report := kind_report(outcomes, kind, bootstrap_seed=bootstrap_seed)) is not None
    ]
    return {
        "per_entry": per_entry,
        "seed": str(seed),
        "entries": len(entries),
        "kinds": kinds_report,
        "canonicalised_order": False,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: Mapping[str, object]) -> str:
    """A markdown table of the probe report, one row per kind."""
    lines = [
        f"# Permutation probe ({report['entries']} entries, {report['per_entry']} per kind, seed"
        f" {report['seed']})",
        "",
        "The scorer is never canonicalised: each kind's draw is rendered and scored exactly as"
        " drawn.",
        "",
        "| kind | trials | changes | rate | 95% CI | lowercase rate | uppercase rate |"
        " incomplete |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for kind_data in report["kinds"]:  # type: ignore[index]

        def _fmt(value: object) -> str:
            return f"{value:.3f}" if isinstance(value, float) else "n/a"

        case = kind_data["label_case"]
        incomplete = kind_data["incomplete"]
        ci = f"[{_fmt(kind_data['ci_low'])}, {_fmt(kind_data['ci_high'])}]"
        lines.append(
            f"| {kind_data['kind']} | {kind_data['trials']} | {kind_data['changes']} |"
            f" {_fmt(kind_data['rate'])} | {ci}"
            f" | {_fmt(case['lowercase_rate'])} | {_fmt(case['uppercase_rate'])}"
            f" | {incomplete['trials']}/{incomplete['of']} |"
        )
        if kind_data["kind"] == "subset":
            dropped = kind_data["baseline_choice_removed"]
            lines.append(
                f"\nsubset: baseline choice removed in {dropped['removed']}/{dropped['trials']}"
                f" trials ({_fmt(dropped['rate'])})."
            )
    lines.append("")
    lines.append(f"_{report['kinds'][0]['bootstrap_note'] if report['kinds'] else ''}_")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def full_alphabet_labels() -> dict[str, str]:
    """Every letter in ``scorer.LABEL_ALPHABET``, mapped to itself.

    The training scorer's default candidate -> label map (``scorer.labels_for
    (scorer.candidates())``) is fixed to as many letters as there are
    candidates -- A-R today -- because that is what a served/trained model
    was calibrated against. A permutation trial's ``letters``/``all`` draw
    (:func:`scorer.permute`) is not so constrained: it draws each offered
    candidate's letter from the *whole* alphabet without replacement, so any
    candidate can land on any of the 52 letters.

    An in-process :class:`scorer.TransformersScorer` only ever reads token
    variants for the letters in the labels map it was built with
    (``label_variant_ids`` keys purely on letter text, never on which
    candidate holds it -- see its own docstring). Building it from this
    full-alphabet map, instead of the training default, means every letter a
    trial can draw already has its variant token ids precomputed, so no
    trial's answer goes unread regardless of which candidate the draw gave
    it (issue 53: a repro against the default A-R map excluded 12 of the 18
    offered labels and undercounted every answer-change rate).
    """
    return {letter: letter for letter in scorer.LABEL_ALPHABET}


def _build_real_scorer(args: argparse.Namespace):
    """The real Track B scorer, via ``measure.py``'s existing seam. Not exercised by tests."""
    measure = _sibling("measure")  # pragma: no cover - a model server or GPU
    seams = measure.Seams()  # pragma: no cover
    cfg = seams.load_config(Path(args.config) if args.config else None)  # pragma: no cover
    tiers = dict(getattr(cfg, "tiers", {}) or {})  # pragma: no cover
    lfm_settings = dict(tiers.get("lfm") or {})  # pragma: no cover
    lfm_settings["model"] = args.model  # pragma: no cover
    spec = measure.ScorerSpec(  # pragma: no cover
        kind=args.scorer_kind,
        model=args.model,
        revision=args.revision,
        lfm_settings=lfm_settings,
        runtime_platform=seams.detect_platform(),
        memory_floor_mb=int(tiers.get("memory_floor_mb", 1024)),
        tokenizer=args.tokenizer,
    )
    # A served model's own top-K readout already covers whatever letters a trial draws
    # (measure.build_scorer's docstring); only the in-process scorer needs a labels map
    # wide enough for every letter a trial can assign, not just the fixed training slots.
    labels = full_alphabet_labels() if args.scorer_kind == measure.SCORER_IN_PROCESS else None
    return seams.build_scorer(spec, labels)  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="permutation_probe.py",
        description="Answer-change rate under seeded order/letter/subset/paraphrase"
        " permutations (issue 53, t7).",
    )
    parser.add_argument("--split", required=True, help="split or corpus file to probe")
    parser.add_argument("--paraphrases", default=None, help="t14's {candidate: [alt, ...]} JSON")
    parser.add_argument("--per-entry", type=int, default=DEFAULT_PER_ENTRY)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=metrics.DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--final", action="store_true", help="allow a test/held-out split")
    parser.add_argument("--out", default=None, help="JSON report path (default: stdout)")
    parser.add_argument("--markdown", default=None, help="markdown table path")
    parser.add_argument("--model", default=None, help="real scorer: repo id or served model name")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--scorer-kind", default="in-process", choices=("served", "in-process"), dest="scorer_kind"
    )
    parser.add_argument("--config", default=None, help="nvsh config.toml (default: XDG path)")
    parser.add_argument(
        "--reasons",
        action="store_true",
        help="probe the pool build_dataset.py --reasons trains on (8 escalate:<reason>"
        " candidates, no bare escalate), for a scorer trained that way",
    )
    args = parser.parse_args(argv)

    split_path = Path(args.split)
    try:
        raw = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read split file {split_path}: {exc}", file=sys.stderr)
        return 1
    header = raw.get("header") if isinstance(raw, dict) else None
    markers = calibration_fit.split_markers(split_path, header)
    if markers and not args.final:
        print(
            f"error: {split_path} looks like the {' and '.join(sorted(markers))} split; "
            "refusing to probe it without --final",
            file=sys.stderr,
        )
        return 1

    loaded = tier_bench.load_corpus(split_path)
    if not loaded.entries:
        print(f"error: {split_path} has no valid entries", file=sys.stderr)
        return 1

    paraphrases = None
    if args.paraphrases:
        try:
            paraphrases = load_paraphrases(Path(args.paraphrases))
        except ProbeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.model:
        handle = _build_real_scorer(args)  # pragma: no cover - a model server or GPU
        scorer_obj, render, close = handle.scorer, handle.render, handle.close  # pragma: no cover
    else:
        print("error: --model is required (no fake scorer is wired to the CLI)", file=sys.stderr)
        return 1
    try:
        report = run_probe(  # pragma: no cover - reached only with --model
            scorer_obj,
            render,
            loaded.entries,
            per_entry=args.per_entry,
            seed=args.seed,
            paraphrases=paraphrases,
            bootstrap_seed=args.bootstrap_seed,
            reasons=args.reasons,
        )
    finally:
        close()  # pragma: no cover

    payload = json.dumps(report, indent=2, sort_keys=True)  # pragma: no cover
    if args.out:  # pragma: no cover
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    else:  # pragma: no cover
        print(payload)
    if args.markdown:  # pragma: no cover
        Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
