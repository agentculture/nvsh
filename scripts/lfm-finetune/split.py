#!/usr/bin/env python3
"""Seeded train/val/test split for the Tier 2 (LFM2.5) fine-tune benchmark corpus.

A development-machine tool for ``docs/lfm-finetune.md`` (part of #39, #53). It
is NEVER imported by the nvsh package -- nothing under nvsh/ may depend on it.
It does not depend on ``scripts/lfm-finetune/build_dataset.py`` or
``nvsh.tiers.bench.load_corpus`` either: it reads a corpus file's JSON
directly, so this script and its tests never depend on task t1.

Usage (legacy, single corpus, fractions -- unchanged since #39)::

    python scripts/lfm-finetune/split.py --corpus nvsh/tiers/corpus/dev.json \\
        --out-dir out/ --seed 39

Splits the corpus's ``entries`` into three sides -- train, val, test
(decision c31: iterate against val, test is only measured on final runs) --
stratified by *expectation kind* (``"operation"``, ``"escalate"`` or
``"explain"``, from each entry's ``expect`` block), and writes
``train.json``, ``val.json`` and ``test.json`` into ``--out-dir`` in the same
``{"header": ..., "entries": [...]}`` corpus shape. Splitting is seeded
(``random.Random(seed)``) and each expectation-kind group is sorted by id
before it is shuffled, so the split does not depend on JSON object hashing,
dict ordering or the platform's ``PYTHONHASHSEED`` -- the same seed always
yields the identical split. Every output entry carries a ``source_id`` field
(its own ``id``), so a later variation of an entry (e.g. a rephrasing added
by task t6) can record that same ``source_id`` and inherit its original's
side without ever appearing on two sides itself.

Usage (v2, multiple corpora, target sizes -- new for #53)::

    python scripts/lfm-finetune/split.py \\
        --corpus nvsh/tiers/corpus/dev.json --corpus /work/i53/drafted.json \\
        --version v2 --val-size 150 --test-size 150 --seed 39 \\
        --fold-seed 7 --out-dir /work/i53/corpus-v2

v2 mode (triggered by ``--version``, more than one ``--corpus``, or either
``--val-size``/``--test-size``) merges every ``--corpus`` input's entries
(de-duplicated by ``id``, first occurrence wins), takes validation and test
as absolute target sizes rather than fractions (the rest goes to train),
still stratified by expectation kind, and additionally interleaves entries
by their ``class`` field (present on escalation entries) so a contiguous
slice draws from every class roughly evenly where counts allow. Every v2
output's ``header`` is a JSON object (not the plain string the legacy path
writes) naming the corpus ``version``, the ``seed``, each input's path and
sha256, and the resulting side sizes. After ``val.json`` is written, its
ids are handed to ``calibration_fit.make_folds`` (task t4) with
``--fold-seed``, and the resulting ``{fold_seed, fit_ids, selection_ids}``
are folded into ``val.json``'s own header and also written standalone to
``folds.json`` next to it, ready for ``calibration_fit``'s own ``fit``
subcommand. v2 refuses to write to any path that resolves inside ``nvsh/``
(the committed corpus lives there; a v2 corpus never does -- operator
decisions q10/q11) and, like the legacy path, refuses the held-out split as
an input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
_NVSH_DIR = (_REPO_ROOT / "nvsh").resolve()

#: Mirrors build_dataset.py's own constant and refusal: the held-out split is
#: for judging a tuned model, never for building train/val/test splits from.
HELD_OUT_NAME = "held-out.json"

#: A committed default seed: identical input + this seed always yields the
#: identical split (pinned by tests/test_lfm_finetune_split.py).
DEFAULT_SEED = 39

#: train/val/test (decision c31: iterate on val, test only measured on final runs).
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")

#: Every expectation kind the corpus schema knows about, in report order.
#: "explain" is new (see docs/lfm-finetune.md); dev.json has none yet.
EXPECTATION_KINDS = ("operation", "escalate", "explain")


def expectation_kind(expect: dict) -> str:
    """Classify one entry's ``expect`` block: "operation", "escalate" or "explain"."""
    if expect.get("escalate"):
        return "escalate"
    if expect.get("explain"):
        return "explain"
    if "operation" in expect:
        return "operation"
    raise ValueError(f"entry has an expect block nvsh doesn't recognize: {expect!r}")


def _allocate(n: int, fractions: tuple[float, float, float]) -> list[int]:
    """How many of *n* items go to each side, by largest-remainder rounding.

    Once ``n`` reaches the number of sides, every side is guaranteed at
    least one item (borrowed from the largest side), so a kind present in
    the input with enough entries is present on every side.
    """
    raw = [f * n for f in fractions]
    counts = [int(r) for r in raw]
    remainder = n - sum(counts)
    order = sorted(range(len(fractions)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(fractions)]] += 1
    if n >= len(fractions):
        for i, count in enumerate(counts):
            if count == 0:
                donor = max(range(len(counts)), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[i] += 1
    return counts


def stratified_split(
    entries: list[dict],
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Split *entries* into train/val/test, stratified by expectation kind.

    Returns ``(sides, missing_kinds)``. *sides* maps each of
    :data:`SPLIT_NAMES` to a list of entries, each carrying its own
    ``source_id`` (c42). *missing_kinds* lists any of
    :data:`EXPECTATION_KINDS` entirely absent from *entries* -- reported,
    not an error, since a corpus may not yet have every kind (dev.json has
    no "explain" entries yet).
    """
    if any(not math.isfinite(f) or not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError(f"each fraction must be between 0 and 1, got {fractions!r}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fractions!r}")

    id_counts = Counter(entry["id"] for entry in entries)
    duplicates = sorted(entry_id for entry_id, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate entry ids would split one source across sides: {duplicates}")

    # An entry that already carries a source_id (a variation, c42) is kept
    # with every other entry of that source: sources are what get split.
    sources: dict[str, list[dict]] = {}
    for entry in entries:
        sources.setdefault(entry.get("source_id", entry["id"]), []).append(entry)

    by_kind: dict[str, list[str]] = {kind: [] for kind in EXPECTATION_KINDS}
    for source_id, members in sources.items():
        kinds = {expectation_kind(member["expect"]) for member in members}
        if len(kinds) > 1:
            raise ValueError(f"source {source_id!r} mixes expectation kinds {sorted(kinds)}")
        by_kind[kinds.pop()].append(source_id)

    missing_kinds = [kind for kind in EXPECTATION_KINDS if not by_kind[kind]]

    sides: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for kind in EXPECTATION_KINDS:
        # Sort before shuffling: JSON array order is already deterministic,
        # but sorting makes that explicit and platform-independent rather
        # than relying on it.
        group = sorted(by_kind[kind])
        random.Random(seed).shuffle(group)
        counts = _allocate(len(group), fractions)
        offset = 0
        for name, count in zip(SPLIT_NAMES, counts):
            for source_id in group[offset : offset + count]:
                for entry in sources[source_id]:
                    sides[name].append({**entry, "source_id": source_id})
            offset += count

    for name in SPLIT_NAMES:
        sides[name].sort(key=lambda entry: entry["id"])

    return sides, missing_kinds


def absent_from_sides(sides: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """``(kind, side)`` pairs for a kind present in the split but missing from a side.

    A kind with fewer entries than there are sides cannot reach every side;
    this names each gap so the caller reports it instead of passing silently.
    """
    present = {expectation_kind(e["expect"]) for side in sides.values() for e in side}
    return [
        (kind, name)
        for kind in EXPECTATION_KINDS
        if kind in present
        for name in SPLIT_NAMES
        if not any(expectation_kind(e["expect"]) == kind for e in sides[name])
    ]


def build_splits(
    corpus: Path,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> tuple[dict[str, list[dict]], list[str], str]:
    """Load *corpus* (as plain JSON, not via ``load_corpus``) and split it.

    Refuses the held-out split, exactly as ``build_dataset.py``'s ``build()``
    does. Returns ``(sides, missing_kinds, header)``.
    """
    if corpus.name == HELD_OUT_NAME:
        raise ValueError("the held-out split is for judging a tuned model, never for training it")
    with open(corpus, encoding="utf-8") as handle:
        raw = json.load(handle)
    entries = raw.get("entries", []) if isinstance(raw, dict) else raw
    header = raw.get("header", "") if isinstance(raw, dict) else ""
    sides, missing_kinds = stratified_split(entries, seed, fractions)
    return sides, missing_kinds, header


def _load_calibration_fit():
    """Load ``calibration_fit.py`` (task t4) by path -- these scripts are not a package."""
    path = _HERE / "calibration_fit.py"
    spec = importlib.util.spec_from_file_location("lfm_finetune_calibration_fit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def refuse_if_under_nvsh(path: Path) -> None:
    """Raise :class:`ValueError` when *path* resolves inside this repo's ``nvsh/``.

    The committed benchmark corpus lives at ``nvsh/tiers/corpus/``; a v2
    corpus (built from operator decisions q10/q11) is a run-work-dir /
    private-data-repo artifact and must never land there or anywhere else
    under ``nvsh/``.
    """
    resolved = path.resolve()
    if resolved == _NVSH_DIR or _NVSH_DIR in resolved.parents:
        raise ValueError(f"refusing to write inside nvsh/: {path} resolves to {resolved}")


def sha256_file(path: Path) -> str:
    """The hex sha256 digest of *path*'s bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_corpora(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Merge every *paths* corpus file's entries, de-duplicated by ``id``.

    Refuses the held-out split exactly like :func:`build_splits`. The first
    occurrence of a given ``id`` wins; a later duplicate (e.g. the same
    entry present in both ``dev.json`` and a drafted source file) is
    dropped silently -- the caller's ``merged=`` accounting in the printed
    summary is where that shows up. Returns ``(entries, sources)`` where
    *sources* is ``[{"path": str, "sha256": str}, ...]`` in input order, for
    the v2 header.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    sources: list[dict] = []
    for path in paths:
        if path.name == HELD_OUT_NAME:
            raise ValueError(
                "the held-out split is for judging a tuned model, never for training it"
            )
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        entries = raw.get("entries", []) if isinstance(raw, dict) else raw
        for entry in entries:
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            merged.append(entry)
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    return merged, sources


def _class_of(entry: dict) -> object:
    return entry.get("class")


def _class_balanced_order(
    source_ids: list[str], sources: dict[str, list[dict]], seed: int
) -> list[str]:
    """Order *source_ids* so a contiguous slice draws from every ``class`` evenly.

    Groups ids by the first member's ``class`` field (``None`` when absent),
    shuffles each class group independently (seeded, so deterministic), then
    interleaves the groups round-robin. ``_allocate``'s train/val/test slices
    are taken from the front of this order, so each slice draws from every
    class present roughly in proportion, instead of by chance alone. A
    corpus with no ``class`` field on any entry in the group behaves exactly
    like a single shuffled list (unchanged from the legacy split).
    """
    by_class: dict[object, list[str]] = {}
    for source_id in source_ids:
        cls = _class_of(sources[source_id][0])
        by_class.setdefault(cls, []).append(source_id)
    groups: list[list[str]] = []
    for cls in sorted(by_class, key=lambda c: (c is None, str(c))):
        group = sorted(by_class[cls])
        random.Random(seed).shuffle(group)
        groups.append(group)
    ordered: list[str] = []
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                ordered.append(group[index])
    return ordered


def stratified_split_sized(
    entries: list[dict],
    seed: int,
    val_size: int,
    test_size: int,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Like :func:`stratified_split`, but val/test are absolute target counts.

    The rest goes to train. Internally this still uses the same
    largest-remainder allocation per expectation kind that
    :func:`stratified_split` uses (via fractions derived from the target
    sizes over the total entry count), so a kind's own proportional share
    of val/test is preserved; the only difference in ordering is
    :func:`_class_balanced_order`, used here instead of a plain per-kind
    shuffle so entries sharing a ``class`` field (escalation entries)
    spread across sides where counts allow. Because each kind rounds its
    own share independently, the *total* val/test size lands close to, but
    is not always exactly, ``val_size``/``test_size`` -- the acceptance
    target is itself approximate ("test ~150").
    """
    total = len(entries)
    if total == 0:
        raise ValueError("cannot split an empty corpus")
    val_frac = val_size / total
    test_frac = test_size / total
    train_frac = 1.0 - val_frac - test_frac
    if train_frac < 0:
        raise ValueError(
            f"val_size + test_size ({val_size + test_size}) exceeds the corpus size ({total})"
        )
    fractions = (train_frac, val_frac, test_frac)

    id_counts = Counter(entry["id"] for entry in entries)
    duplicates = sorted(entry_id for entry_id, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate entry ids would split one source across sides: {duplicates}")

    sources: dict[str, list[dict]] = {}
    for entry in entries:
        sources.setdefault(entry.get("source_id", entry["id"]), []).append(entry)

    by_kind: dict[str, list[str]] = {kind: [] for kind in EXPECTATION_KINDS}
    for source_id, members in sources.items():
        kinds = {expectation_kind(member["expect"]) for member in members}
        if len(kinds) > 1:
            raise ValueError(f"source {source_id!r} mixes expectation kinds {sorted(kinds)}")
        by_kind[kinds.pop()].append(source_id)

    missing_kinds = [kind for kind in EXPECTATION_KINDS if not by_kind[kind]]

    sides: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for kind in EXPECTATION_KINDS:
        group = _class_balanced_order(sorted(by_kind[kind]), sources, seed)
        counts = _allocate(len(group), fractions)
        offset = 0
        for name, count in zip(SPLIT_NAMES, counts):
            for source_id in group[offset : offset + count]:
                for entry in sources[source_id]:
                    sides[name].append({**entry, "source_id": source_id})
            offset += count

    for name in SPLIT_NAMES:
        sides[name].sort(key=lambda entry: entry["id"])

    return sides, missing_kinds


def _write_side(
    out_dir: Path,
    name: str,
    entries: list[dict],
    header: str,
    corpus_name: str,
    seed: int,
    world: dict | None = None,
) -> Path:
    note = f"Split '{name}' of {corpus_name} (seed={seed})."
    payload: dict = {
        "header": f"{header} {note}".strip(),
        "entries": entries,
    }
    # The corpus world (platform, fixture machine state) travels with every
    # side: without it a builder falls back to an unknown platform and the
    # system brief no longer matches what Tier 2 was measured with.
    if world is not None:
        payload["world"] = world
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def _write_side_v2(
    out_dir: Path,
    name: str,
    entries: list[dict],
    header: dict,
    world: dict | None = None,
) -> Path:
    """Like :func:`_write_side`, but the header is a structured object (v2).

    v2's header carries ``version``, ``seed``, every input's ``sources``
    (path + sha256) and the resulting ``sizes`` on every side, plus
    ``fold_seed``/``fit_ids``/``selection_ids`` on the val side only.
    """
    payload: dict = {"header": {**header, "side": name}, "entries": entries}
    if world is not None:
        payload["world"] = world
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _main_v2(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    corpus_paths: list[Path],
    out_dir: Path,
) -> int:
    if args.val_size is None or args.test_size is None:
        parser.error(
            "v2 mode (--version, multiple --corpus, or --val-size/--test-size) "
            "requires both --val-size and --test-size"
        )
    if args.fold_seed is None:
        parser.error("v2 mode requires --fold-seed (passed to calibration_fit.make_folds)")
    version = args.version or "v2"

    try:
        entries, sources = merge_corpora(corpus_paths)
        sides, missing_kinds = stratified_split_sized(
            entries, args.seed, args.val_size, args.test_size
        )
    except ValueError as exc:
        parser.error(str(exc))
    gaps = absent_from_sides(sides)
    if gaps:
        described = ", ".join(f"{kind!r} on {name}" for kind, name in gaps)
        parser.error(f"too few entries to reach every side: missing {described}")

    world = None
    for path in corpus_paths:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict) and raw.get("world") is not None:
            world = raw["world"]
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {name: len(sides[name]) for name in SPLIT_NAMES}

    calibration_fit = _load_calibration_fit()
    val_ids = [entry["id"] for entry in sides["val"]]
    fit_ids, selection_ids = calibration_fit.make_folds(val_ids, args.fold_seed)

    for name in SPLIT_NAMES:
        header = {
            "version": version,
            "seed": args.seed,
            "sources": sources,
            "sizes": sizes,
        }
        if name == "val":
            header["fold_seed"] = args.fold_seed
            header["fit_ids"] = fit_ids
            header["selection_ids"] = selection_ids
        _write_side_v2(out_dir, name, sides[name], header, world)

    write_json(
        out_dir / "folds.json",
        {
            "seed": args.fold_seed,
            "source": str(out_dir / "val.json"),
            "fit_ids": fit_ids,
            "selection_ids": selection_ids,
        },
    )

    for kind in missing_kinds:
        print(f"note: no {kind!r} entries in the merged corpus; not present on any side")
    for name in SPLIT_NAMES:
        print(f"{name}={len(sides[name])}")
    plural = "y" if len(entries) == 1 else "ies"
    print(f"merged {len(entries)} unique entr{plural} from {len(corpus_paths)} corpus file(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="corpus file to read; repeatable in v2 mode (default: nvsh/tiers/corpus/dev.json)",
    )
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_FRACTIONS[0])
    parser.add_argument("--val-frac", type=float, default=DEFAULT_FRACTIONS[1])
    parser.add_argument("--test-frac", type=float, default=DEFAULT_FRACTIONS[2])
    parser.add_argument(
        "--val-size", type=int, default=None, help="v2: absolute validation size (e.g. 150)"
    )
    parser.add_argument(
        "--test-size", type=int, default=None, help="v2: absolute test size (e.g. 150)"
    )
    parser.add_argument("--version", default=None, help="v2: corpus version name, e.g. v2")
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=None,
        help="v2: seed for calibration_fit.make_folds on the written val ids",
    )
    args = parser.parse_args(argv)

    corpus_paths = [
        Path(p) for p in (args.corpus or [str(_REPO_ROOT / "nvsh/tiers/corpus/dev.json")])
    ]
    out_dir = Path(args.out_dir)
    try:
        refuse_if_under_nvsh(out_dir)
    except ValueError as exc:
        parser.error(str(exc))

    v2_mode = (
        bool(args.version)
        or args.val_size is not None
        or args.test_size is not None
        or len(corpus_paths) > 1
    )
    if v2_mode:
        return _main_v2(parser, args, corpus_paths, out_dir)

    corpus = corpus_paths[0]
    fractions = (args.train_frac, args.val_frac, args.test_frac)
    try:
        sides, missing_kinds, header = build_splits(corpus, args.seed, fractions)
    except ValueError as exc:
        parser.error(str(exc))
    gaps = absent_from_sides(sides)
    if gaps:
        described = ", ".join(f"{kind!r} on {name}" for kind, name in gaps)
        parser.error(f"too few entries to reach every side: missing {described}")

    with open(corpus, encoding="utf-8") as handle:
        raw = json.load(handle)
    world = raw.get("world") if isinstance(raw, dict) else None

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        _write_side(out_dir, name, sides[name], header, corpus.name, args.seed, world)

    for kind in missing_kinds:
        print(f"note: no {kind!r} entries in {corpus.name}; not present on any side")
    for name in SPLIT_NAMES:
        print(f"{name}={len(sides[name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
