"""Fold accepted augment.py variations into a train split file (issue 39).

``augment.py`` writes accepted variations of train-side entries as corpus
entries (``kind``, ``text``, ``expect``, ``source_id``, ``side``). This script
appends them to the train split ``split.py`` wrote, so ``build_dataset.py
--split`` builds one training set from both, and every example keeps its
``source_id``.

It refuses anything that could move data across sides: the split must be the
train side (its header says so), every variation must say ``side: train``,
and its ``source_id`` must be an entry of that split with the same expected
answer. A variation whose text repeats one already present (the source or an
earlier variation, compared case- and space-insensitively) is dropped and
counted.

    python scripts/lfm-finetune/merge_variations.py --split train.json \
        --accepted accepted-*.jsonl --out train-augmented.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TRAIN_HEADER = re.compile(r"Split 'train' of ")


def _normal(text: str) -> str:
    """Case, spacing and punctuation folded, so "Restart it." repeats "restart it"."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def merge(split: dict, variations: list[dict]) -> tuple[dict, dict[str, int]]:
    """The split with *variations* appended, and counts of what was kept and dropped."""
    if not _TRAIN_HEADER.search(str(split.get("header", ""))):
        raise ValueError("the split's header does not name the train side")
    sources = {entry["id"]: entry for entry in split["entries"]}
    seen = {_normal(entry["text"]) for entry in split["entries"]}
    kept: list[dict] = []
    counts = {"kept": 0, "duplicate": 0}
    for variation in variations:
        if variation.get("side") != "train":
            raise ValueError(f"{variation.get('id')}: side {variation.get('side')!r} is not train")
        source = sources.get(variation.get("source_id"))
        if source is None:
            raise ValueError(
                f"{variation.get('id')}: source {variation.get('source_id')!r} is not in the split"
            )
        if variation.get("expect") != source["expect"]:
            raise ValueError(f"{variation.get('id')}: expected answer differs from its source")
        key = _normal(variation["text"])
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        entry = {
            k: v for k, v in variation.items() if k not in ("models", "seed_format", "verdicts")
        }
        kept.append(entry)
        counts["kept"] += 1
    merged = {**split, "entries": [*split["entries"], *kept]}
    merged["header"] = f"{split['header']} Plus {counts['kept']} accepted variations (augment.py)."
    return merged, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, type=Path, help="train.json from split.py")
    parser.add_argument("--accepted", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    variations = [
        json.loads(line)
        for path in args.accepted
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        merged, counts = merge(split, variations)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"sources={len(split['entries'])} kept={counts['kept']} duplicate={counts['duplicate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
