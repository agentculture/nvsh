#!/usr/bin/env python3
"""Produce eval slices from a corpus split.

A development-machine tool for the Tier 2 (LFM2.5) fine-tune benchmark.
It reads an existing split (train/val/test) file, extracts every entry
whose ``expect`` block names a concrete operation, and emits a new slice
with a *missing-candidates* prompt: ``candidates`` lists every
registered operation **except** the gold one, and ``expect`` is
``{"escalate": true}``.

Usage::

    python scripts/lfm-finetune/eval_slices.py --split nvsh/tiers/corpus/train.json \\
        --out out/train-eval.json

This script is NEVER imported by the nvsh package and never depends on
any module under ``nvsh/`` beyond the read-only operation table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.ops import table  # noqa: E402


def missing_candidate_slice(split: dict, all_operations: tuple[str, ...]) -> dict:
    """Return a new slice with missing-candidate entries for every operation expect.

    For each entry whose ``expect`` dict has an ``"operation"`` key, emit a
    new entry with:

    - ``id`` = original ``id`` + ``"-nocand"``
    - ``kind`` = same kind
    - ``text`` = byte-identical to the original
    - ``source_id`` = original id
    - ``candidates`` = ``all_operations`` with the gold operation removed
    - ``expect`` = ``{"escalate": True}``

    Entries without an ``"operation"`` key in their ``expect`` dict are
    **not** included.  The input *split* dict is never mutated.
    """
    header = f"Eval slice: missing candidates for {split['header'].strip()}."
    new_entries: list[dict] = []
    for entry in split.get("entries", []):
        expect = entry.get("expect", {})
        if "operation" not in expect:
            continue
        gold_op = expect["operation"]
        candidates = tuple(op for op in all_operations if op != gold_op)
        new_entries.append(
            {
                "id": entry["id"] + "-nocand",
                "kind": entry["kind"],
                "text": entry["text"],
                "source_id": entry["id"],
                "candidates": list(candidates),
                "expect": {"escalate": True},
            }
        )
    return {"header": header, "entries": new_entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, help="Path to a split JSON file")
    parser.add_argument("--out", required=True, help="Path to write the eval-slice JSON")
    args = parser.parse_args(argv)

    split_path = Path(args.split)
    with open(split_path, encoding="utf-8") as handle:
        split = json.load(handle)

    slice_dict = missing_candidate_slice(split, table.names())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(slice_dict, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"entries={len(slice_dict['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
