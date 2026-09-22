#!/usr/bin/env python3
"""Seeded train/val/test split for the Tier 2 (LFM2.5) fine-tune benchmark corpus.

A development-machine tool for ``docs/lfm-finetune.md`` (part of #39). It is
NEVER imported by the nvsh package -- nothing under nvsh/ may depend on it.
It does not depend on ``scripts/lfm-finetune/build_dataset.py`` or
``nvsh.tiers.bench.load_corpus`` either: it reads a corpus file's JSON
directly, so this script and its tests never depend on task t1.

Usage::

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
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

_REPO_ROOT = Path(__file__).resolve().parents[2]

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
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fractions!r}")

    id_counts = Counter(entry["id"] for entry in entries)
    duplicates = sorted(entry_id for entry_id, count in id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate entry ids would split one source across sides: {duplicates}")

    by_kind: dict[str, list[dict]] = {kind: [] for kind in EXPECTATION_KINDS}
    for entry in entries:
        by_kind[expectation_kind(entry["expect"])].append(entry)

    missing_kinds = [kind for kind in EXPECTATION_KINDS if not by_kind[kind]]

    sides: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    for kind in EXPECTATION_KINDS:
        # Sort before shuffling: JSON array order is already deterministic,
        # but sorting makes that explicit and platform-independent rather
        # than relying on it.
        group = sorted(by_kind[kind], key=lambda entry: entry["id"])
        random.Random(seed).shuffle(group)
        counts = _allocate(len(group), fractions)
        offset = 0
        for name, count in zip(SPLIT_NAMES, counts):
            for entry in group[offset : offset + count]:
                sides[name].append({**entry, "source_id": entry["id"]})
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


def _write_side(
    out_dir: Path, name: str, entries: list[dict], header: str, corpus_name: str, seed: int
) -> Path:
    note = f"Split '{name}' of {corpus_name} (seed={seed})."
    payload = {
        "header": f"{header} {note}".strip(),
        "entries": entries,
    }
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(_REPO_ROOT / "nvsh/tiers/corpus/dev.json"))
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_FRACTIONS[0])
    parser.add_argument("--val-frac", type=float, default=DEFAULT_FRACTIONS[1])
    parser.add_argument("--test-frac", type=float, default=DEFAULT_FRACTIONS[2])
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    fractions = (args.train_frac, args.val_frac, args.test_frac)
    try:
        sides, missing_kinds, header = build_splits(corpus, args.seed, fractions)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        _write_side(out_dir, name, sides[name], header, corpus.name, args.seed)

    for kind in missing_kinds:
        print(f"note: no {kind!r} entries in {corpus.name}; not present on any side")
    for kind, name in absent_from_sides(sides):
        print(f"warning: {kind!r} entries are too few to reach the {name} side")
    for name in SPLIT_NAMES:
        print(f"{name}={len(sides[name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
