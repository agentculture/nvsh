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
  variation), source, source entry, source file and licence, and for a
  variation the generator, corrector and reviewer models;
- ``README.md``: the data set card, with every count computed here;
- ``LICENSE``: nvsh's own Apache-2.0 licence file.

Only nvsh's own data goes in. The Jetson skills requests (derived from
NVIDIA's CC-BY-4.0 / Apache-2.0 skills) are left out; publishing them would
need NVIDIA's attribution (spec c40). ``held-out.json`` is refused by name,
and so is a train record that repeats a validation or test entry.
The script never uploads.

Which model answered each of augment.py's four roles is a fact about one
run, not a constant of this script: ``--teacher-models`` points at a JSON
file (``{"<alias-or-model-id>": {"name": ..., "licence": ...}, ...}``, keyed
by whatever an operator set as ``NVSH_AUG_<ROLE>_MODEL``) naming this run's
teachers, and ``--apache-only`` refuses to build a bundle that names any
teacher whose licence is not Apache-2.0.

    python scripts/lfm-finetune/dataset_bundle.py --splits work/splits \
        --train-augmented work/data/train-augmented.json \
        --accepted work/aug/nvsh-accepted.jsonl --rejected work/aug/nvsh-rejected.jsonl \
        --licence LICENSE --teacher-models work/teacher-models.json --apache-only \
        --out dataset-bundle
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

HELD_OUT_NAME = "held-out.json"
CORPUS_FILE = "nvsh/tiers/corpus/dev.json"
SUPPLEMENT_FILE = "scripts/lfm-finetune/train-supplement.json"
CORPUS_LICENCE = "Apache-2.0"
REPO_URL = "https://github.com/agentculture/nvsh"

APACHE_LICENCE = "Apache-2.0"

#: The four functions augment.py's roles fill. Which alias/model answered
#: each one in a given run is not fixed here -- it comes from the run's own
#: accepted records and its ``--teacher-models`` file (``load_role_models``).
ROLES = (
    ("GENERATOR", "wrote the variation"),
    ("CORRECTOR", "copyedited it"),
    ("REVIEWER_A", "accepted it (reviewer A)"),
    ("REVIEWER_B", "accepted it (reviewer B)"),
)


def load_role_models(path: Path) -> dict[str, tuple[str, str]]:
    """The alias/model-id -> (display name, licence) table for one run.

    *path* is a JSON object keyed by whatever an operator set as
    ``NVSH_AUG_<ROLE>_MODEL`` when running ``augment.py`` (often a gateway
    alias, e.g. ``"associate"``), each mapping to ``{"name": ..., "licence":
    ...}``. There is no built-in default: which model answers an alias
    changes run to run, so every bundle names its own teachers.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a JSON object of alias -> {{name, licence}}")
    table: dict[str, tuple[str, str]] = {}
    for alias, info in raw.items():
        if not isinstance(info, dict) or not info.get("name") or not info.get("licence"):
            raise ValueError(f"{path}: {alias!r} needs a non-empty 'name' and 'licence'")
        table[alias] = (str(info["name"]), str(info["licence"]))
    return table


def _entries(path: Path) -> list[dict[str, Any]]:
    if path.name == HELD_OUT_NAME:
        raise ValueError(f"{path}: the held-out split is never published with the training data")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw["entries"] if isinstance(raw, dict) else raw)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _normal(text: str) -> str:
    """Case, spacing and punctuation folded, as merge_variations.py compares texts."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _teacher(role_models: dict[str, tuple[str, str]], alias: str) -> tuple[str, str]:
    if alias not in role_models:
        raise ValueError(f"{alias!r} is not in the run's --teacher-models file")
    return role_models[alias]


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
    role_models: dict[str, tuple[str, str]],
    out: Path,
    apache_only: bool = False,
) -> dict[str, Any]:
    """Write the data set folder to *out*; return the counts shown in the card.

    *role_models* is this run's alias -> (name, licence) table (see
    ``load_role_models``). With *apache_only*, a teacher named by any
    accepted variation whose licence is not Apache-2.0 refuses the build.
    """
    train = _entries(train_augmented)
    sides = {"validation": _entries(splits / "val.json"), "test": _entries(splits / "test.json")}
    accepted_rows = {row["id"]: row for row in _jsonl(accepted)}
    rejected_count = len(_jsonl(rejected))
    if not licence.read_text(encoding="utf-8").lstrip().startswith("Apache License"):
        raise ValueError(f"{licence} is not the Apache License")

    held_apart = {_normal(e["text"]) for entries in sides.values() for e in entries}
    leaked = sum(1 for entry in train if _normal(entry["text"]) in held_apart)
    if leaked:
        raise ValueError(
            f"{leaked} train record(s) repeat a validation or test entry; re-run"
            " merge_variations.py with --exclude before building the data set"
        )

    manifest: list[dict[str, Any]] = []
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    origins: collections.Counter[str] = collections.Counter()
    #: role -> {resolved model name -> licence}, aggregated across every
    #: accepted variation (not just the first one seen); drives the card's
    #: teacher table.
    role_teachers: dict[str, dict[str, str]] = collections.defaultdict(dict)
    #: resolved corrector/reviewer-B names shared by at least one variation;
    #: drives the card's "reviewer B is also the corrector" disclosure. Two
    #: different aliases that resolve to the same model still count.
    shared_corrector_reviewer_names: dict[str, None] = {}
    for entry in train:
        if entry.get("side", "train") != "train":
            raise ValueError(f"{entry['id']} in {train_augmented} is not a train-side entry")
        origin = _origin(entry)
        source = entry.get("source")
        if not source:
            raise ValueError(f"{entry['id']}: every published record needs a 'source' field")
        row: dict[str, Any] = {
            "id": entry["id"],
            "split": "train",
            "origin": origin,
            "source": source,
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
            teachers: dict[str, str] = {}
            for role, _ in ROLES:
                if role not in models:
                    raise ValueError(
                        f"variation {entry['id']}'s accepted record names no {role} teacher"
                    )
                alias = models[role]
                name, teacher_licence = _teacher(role_models, alias)
                if apache_only and teacher_licence != APACHE_LICENCE:
                    raise ValueError(
                        f"{entry['id']}: teacher {name!r} ({teacher_licence}) is not "
                        f"{APACHE_LICENCE}; refused by --apache-only"
                    )
                teachers[role] = name
                role_teachers[role][name] = teacher_licence
            if teachers.get("CORRECTOR") == teachers.get("REVIEWER_B"):
                shared_corrector_reviewer_names.setdefault(teachers["CORRECTOR"], None)
            row["teachers"] = teachers
        else:
            row["teachers"] = {}
        origins[origin] += 1
        manifest.append(row)
        rows["train"].append(_record(entry, "train"))
    for split, entries in sides.items():
        for entry in entries:
            if "~v" in entry["id"]:
                raise ValueError(f"{entry['id']}: variations never leave the train side")
            source = entry.get("source")
            if not source:
                raise ValueError(f"{entry['id']}: every published record needs a 'source' field")
            manifest.append(
                {
                    "id": entry["id"],
                    "split": split,
                    "origin": "corpus",
                    "source": source,
                    "source_id": entry.get("source_id", entry["id"]),
                    "source_file": CORPUS_FILE,
                    "licence": CORPUS_LICENCE,
                    "transformed": False,
                    "teachers": {},
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
    (out / "README.md").write_text(
        card(counts, role_teachers, list(shared_corrector_reviewer_names)), encoding="utf-8"
    )
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


def card(
    counts: dict[str, Any],
    role_teachers: dict[str, dict[str, str]],
    shared_corrector_reviewer_names: list[str],
) -> str:
    """*role_teachers* is role -> {resolved model name -> licence}, aggregated

    across every accepted variation (see ``build``), and
    *shared_corrector_reviewer_names* lists every resolved model name that
    played both the corrector and reviewer-B roles in at least one
    variation -- by resolved identity, not by which alias named it.
    """
    answers = counts["answers"]
    reviewed = counts["accepted"] + counts["rejected"]
    rate = f"{100 * counts['accepted'] / reviewed:.0f}%" if reviewed else "n/a"
    if role_teachers:
        teachers = "\n".join(
            f"| {name} | {licence} | {desc} |"
            for role, desc in ROLES
            for name, licence in role_teachers.get(role, {}).items()
        )
        disclosure = ""
        if shared_corrector_reviewer_names:
            names = ", ".join(shared_corrector_reviewer_names)
            disclosure = (
                f"\n**Reviewer B is also the corrector** in this run ({names}): its"
                " accept/reject verdict is not independent of the copyedit it made.\n"
            )
    else:
        teachers = "| (none) | (none) | this run produced no synthetic variations |"
        disclosure = ""
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
{disclosure}
The teachers' licences do not carry over to their outputs. `manifest.json`
maps every record to its split, origin, source, source file, licence,
whether it was transformed, and a `teachers` map naming the four models
above for a variation (empty for a corpus or supplement record).

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
    parser.add_argument(
        "--teacher-models",
        required=True,
        type=Path,
        help="JSON file: alias/model-id -> {name, licence} for this run's four teacher roles",
    )
    parser.add_argument(
        "--apache-only",
        action="store_true",
        help="refuse to build a bundle that names any non-Apache-2.0 teacher",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        role_models = load_role_models(args.teacher_models)
        counts = build(
            splits=args.splits,
            train_augmented=args.train_augmented,
            accepted=args.accepted,
            rejected=args.rejected,
            licence=args.licence,
            role_models=role_models,
            out=args.out,
            apache_only=args.apache_only,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
