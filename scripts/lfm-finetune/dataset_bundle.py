"""Build the upload folder for the nvsh-ops data set (issue 39, task t17; spec c45).

The data set is the second of the three artifacts spec c45 asks for: the data
a tuned LFM2.5 was trained and measured on, with a provenance manifest that
maps every record to its source and names the models that generated and
reviewed it. The folder holds:

- ``data/train.jsonl``: the train side as trained on -- corpus entries, the
  train-only supplement and the accepted variations (``train-augmented.json``
  from ``merge_variations.py``);
- ``data/validation.jsonl`` and ``data/test.jsonl``: the seeded split's other
  two sides, corpus entries only;
- ``manifest.json``: one row per record: split, origin (corpus, supplement or
  variation), source entry, source file and licence, and for a variation the
  generator, corrector and reviewer models;
- ``README.md``: the data set card, with every count computed here;
- ``LICENSE``: nvsh's own Apache-2.0 licence file.

Only nvsh's own data goes in. The Jetson skills requests (derived from
NVIDIA's CC-BY-4.0 / Apache-2.0 skills) are left out; publishing them would
need NVIDIA's attribution (spec c40). ``held-out.json`` is refused by name.
The script never uploads.

    python scripts/lfm-finetune/dataset_bundle.py --splits work/splits \
        --train-augmented work/data/train-augmented.json \
        --accepted work/aug/nvsh-accepted.jsonl --rejected work/aug/nvsh-rejected.jsonl \
        --licence LICENSE --out dataset-bundle
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HELD_OUT_NAME = "held-out.json"
CORPUS_FILE = "nvsh/tiers/corpus/dev.json"
SUPPLEMENT_FILE = "scripts/lfm-finetune/train-supplement.json"
CORPUS_LICENCE = "Apache-2.0"
REPO_URL = "https://github.com/agentculture/nvsh"

#: The gateway role names augment.py records, and the models behind them in
#: this run (named by the operator), with each model's licence.
ROLE_MODELS = {
    "worker": ("Qwen 3.6 35B-A3B", "Apache-2.0"),
    "cortex": ("Qwen 3.8 27B", "Apache-2.0"),
    "senses": ("Gemma 4 26B-A4B", "Apache-2.0"),
    "associate": ("Nemotron 3.5 Lightning", "OpenMDW-1.1"),
}
ROLES = (
    ("GENERATOR", "wrote the variation"),
    ("CORRECTOR", "copyedited it"),
    ("REVIEWER_A", "accepted it (reviewer A)"),
    ("REVIEWER_B", "accepted it (reviewer B)"),
)


def _entries(path: Path) -> list[dict[str, Any]]:
    if path.name == HELD_OUT_NAME:
        raise ValueError(f"{path}: the held-out split is never published with the training data")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw["entries"] if isinstance(raw, dict) else raw)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _model_name(role_value: str) -> str:
    return ROLE_MODELS.get(role_value, (role_value, "unknown"))[0]


def _record(entry: dict[str, Any], split: str) -> dict[str, Any]:
    keep = ("id", "text", "expect", "kind", "source_id", "class")
    record = {key: entry[key] for key in keep if key in entry}
    record.setdefault("source_id", entry["id"])
    record["split"] = split
    return record


def _origin(entry: dict[str, Any]) -> str:
    if "~v" in entry["id"]:
        return "variation"
    if str(entry.get("source", "")).startswith("supplement"):
        return "supplement"
    return "corpus"


def build(
    *,
    splits: Path,
    train_augmented: Path,
    accepted: Path,
    rejected: Path,
    licence: Path,
    out: Path,
) -> dict[str, Any]:
    """Write the data set folder to *out*; return the counts shown in the card."""
    train = _entries(train_augmented)
    sides = {"validation": _entries(splits / "val.json"), "test": _entries(splits / "test.json")}
    accepted_rows = {row["id"]: row for row in _jsonl(accepted)}
    rejected_count = len(_jsonl(rejected))
    if not licence.read_text(encoding="utf-8").lstrip().startswith("Apache License"):
        raise ValueError(f"{licence} is not the Apache License")

    manifest: list[dict[str, Any]] = []
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    origins: collections.Counter[str] = collections.Counter()
    for entry in train:
        if entry.get("side", "train") != "train":
            raise ValueError(f"{entry['id']} in {train_augmented} is not a train-side entry")
        origin = _origin(entry)
        row: dict[str, Any] = {
            "id": entry["id"],
            "split": "train",
            "origin": origin,
            "source_id": entry.get("source_id", entry["id"]),
            "source_file": SUPPLEMENT_FILE if origin == "supplement" else CORPUS_FILE,
            "licence": CORPUS_LICENCE,
            "transformed": origin == "variation",
        }
        if origin == "variation":
            models = accepted_rows.get(entry["id"], {}).get("models")
            if not models:
                raise ValueError(
                    f"variation {entry['id']} has no accepted record naming its models"
                )
            row["models"] = {role: _model_name(models[role]) for role, _ in ROLES}
        origins[origin] += 1
        manifest.append(row)
        rows["train"].append(_record(entry, "train"))
    for split, entries in sides.items():
        for entry in entries:
            if "~v" in entry["id"]:
                raise ValueError(f"{entry['id']}: variations never leave the train side")
            manifest.append(
                {
                    "id": entry["id"],
                    "split": split,
                    "origin": "corpus",
                    "source_id": entry.get("source_id", entry["id"]),
                    "source_file": CORPUS_FILE,
                    "licence": CORPUS_LICENCE,
                    "transformed": False,
                }
            )
            rows[split].append(_record(entry, split))

    ids = [row["id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record ids across the splits")

    if out.exists():
        if any(out.iterdir()):
            raise ValueError(f"{out} is not empty")
        out.rmdir()
    (out / "data").mkdir(parents=True)
    for split, records in rows.items():
        with open(out / "data" / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(licence, out / "LICENSE")

    counts = {
        "train": len(rows["train"]),
        "validation": len(rows["validation"]),
        "test": len(rows["test"]),
        "corpus": origins["corpus"],
        "supplement": origins["supplement"],
        "variation": origins["variation"],
        "accepted": len(accepted_rows),
        "rejected": rejected_count,
        "answers": _answer_counts(rows["train"]),
    }
    (out / "README.md").write_text(card(counts), encoding="utf-8")
    return counts


def _answer_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    kinds: collections.Counter[str] = collections.Counter()
    for record in records:
        expect = record["expect"]
        kinds[
            (
                "explain"
                if expect.get("explain")
                else "escalate" if expect.get("escalate") else "propose"
            )
        ] += 1
    return dict(kinds)


def card(counts: dict[str, Any]) -> str:
    answers = counts["answers"]
    reviewed = counts["accepted"] + counts["rejected"]
    rate = f"{100 * counts['accepted'] / reviewed:.0f}%" if reviewed else "n/a"
    teachers = "\n".join(
        f"| {name} | {licence} | {role} |"
        for (_, role), (name, licence) in zip(ROLES, ROLE_MODELS.values())
    )
    train_parts = (
        f"{counts['corpus']} corpus entries, {counts['supplement']} supplement entries,"
        f" {counts['variation']} synthetic variations"
    )
    return f"""---
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- tool-calling
- nvsh
- jetson
- synthetic
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# nvsh-ops

Requests an operator types at a Jetson or DGX Spark shell, each labelled with
the answer [nvsh](https://github.com/agentculture/nvsh)'s Tier 2 should give:
**propose** one operation from nvsh's table (with its arguments), **explain**
in words, or **escalate** to a full agent. It is built by the pipeline that
trained and measured `jetson-ai-lab/lfm2.5-350m-nvsh-triage` (nvsh issue 39).
The model's run used an earlier snapshot of the train side, with fewer
variations; validation and test are the same.

## Splits

| Split | Records | Contents |
|---|---|---|
| train | {counts['train']} | {train_parts} |
| validation | {counts['validation']} | corpus entries only; runs were chosen on this side |
| test | {counts['test']} | corpus entries only; never trained on or used to choose a run |

The train side's answers: {answers.get('propose', 0)} propose,
{answers.get('escalate', 0)} escalate, {answers.get('explain', 0)} explain.

**Do not train on validation or test** if you want numbers comparable with
nvsh's. nvsh's separate held-out split (`held-out.json`) is not in this data
set.

## Record format

```json
{{"id": "dev-e04~v2", "text": "...", "kind": "explicit",
 "expect": {{"operation": "power_set", "args": {{"mode": "max_performance"}}}},
 "source_id": "dev-e04", "split": "train"}}
```

`expect` is one of `{{"operation": ..., "args": {{...}}}}`, `{{"escalate": true}}` or
`{{"explain": true, "answer": "..."}}`. A variation's `id` is its source's id
plus `~vN`, and `source_id` names the entry it rewrites; it keeps its
source's answer and side.

## Where the records come from

- **Corpus entries**: nvsh's development corpus (`{CORPUS_FILE}` in
  <{REPO_URL}>), written by the nvsh project under Apache-2.0 and split with
  `scripts/lfm-finetune/split.py` (seed 39, stratified by answer, 70/15/15).
- **Supplement entries**: a small authored train-only set
  (`{SUPPLEMENT_FILE}`): requests to stop, shut down or disable a container
  or service, which Tier 2 must escalate because it has no such action,
  next to restart requests as contrasts.
- **Variations**: rewrites of train entries, made by a local pipeline
  (`scripts/lfm-finetune/augment.py`). One model rewrote the request and a
  second copyedited it; **neither saw the expected answer**. Two reviewer
  models were each asked whether the rewrite still calls for exactly that
  answer, and a rewrite was kept only when both said yes. Deterministic
  checks then rejected any rewrite naming an internal operation identifier
  or copying answer wording. Of {reviewed} reviewed rewrites, {counts['accepted']}
  were accepted ({rate}); the train side uses the accepted ones that are not
  duplicates of their source.

| Model | Licence | Role |
|---|---|---|
{teachers}

The teachers' licences do not carry over to their outputs. `manifest.json`
maps every record to its split, origin, source entry, source file, licence,
whether it was transformed, and for a variation the four models above.

## Limits

- The requests are English and short; the operations are those in nvsh's
  table at the time of issue 39.
- Arguments were grounded against a fixture machine. Two `power_set` modes
  cannot be rendered there, which caps proposals on those entries.
- Variations were reviewed by models, not people. The reviewers were shown
  nvsh's list of checks and changes, and every prompt fault found by
  reading samples was fixed before the accepted set was kept (nvsh's
  `docs/lfm-finetune.md` run log), but some wrong labels may remain.
- Jetson skill requests used for method validation are **not** included:
  they derive from NVIDIA's skills repositories and would need their
  CC-BY-4.0 attribution.

## Licence

Apache-2.0, the licence of nvsh (see `LICENSE`).
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", required=True, type=Path, help="split.py --out-dir")
    parser.add_argument("--train-augmented", required=True, type=Path)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--licence", required=True, type=Path, help="nvsh's LICENSE")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        counts = build(
            splits=args.splits,
            train_augmented=args.train_augmented,
            accepted=args.accepted,
            rejected=args.rejected,
            licence=args.licence,
            out=args.out,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
