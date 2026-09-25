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
teacher whose licence is not Apache-2.0. Which reviewer decided is read from
the accepted records too: a record accepted under ``--decide-by reviewer_b``
(``decided_by``) or by the clean-slate re-review (a ``reviewer_b`` verdict and
no ``reviewer_a`` one) was decided by reviewer B alone, and the card says
reviewer A was advisory (issue 46, d8/d11). ``release_bundle.py`` reuses
``teacher_summary`` and ``teacher_rows`` for the model card's teacher table.

``--issue`` and ``--model-repo`` name the issue and the model repositories
the data set belongs to (the defaults describe issue 39's LFM2.5 model), and
the split's seed is read from the split file's own header.

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
#: The reviewers' descriptions are those of a run where both decided; see
#: ``role_description`` for a run decided by reviewer B alone.
ROLES = (
    ("GENERATOR", "wrote the variation"),
    ("CORRECTOR", "copyedited it"),
    ("REVIEWER_A", "accepted it (reviewer A)"),
    ("REVIEWER_B", "accepted it (reviewer B)"),
)

#: How a variation's acceptance was decided (augment.py's DECIDE_BY_RULES).
DECIDED_BY_BOTH = "both"
DECIDED_BY_REVIEWER_B = "reviewer_b"

#: Where each issue's run log lives in the nvsh repository.
RUN_LOGS = {
    39: "docs/lfm-finetune.md",
    46: "docs/qwen-tool-jev-finetune.md",
    53: "docs/qwen-tool-jev-calibration.md",
}

_SEED_RE = re.compile(r"\(seed=(\d+)\)")


class TeacherSummary:
    """Which models played each role over a run's accepted variations, and how
    their acceptance was decided.

    ``role_teachers``: role -> {resolved model name -> licence}, aggregated
    over every variation; ``shared_corrector_reviewer_names``: every resolved
    name that was both corrector and reviewer B of one variation;
    ``decisions``: rule (``both`` / ``reviewer_b``) -> variation count;
    ``per_variation``: variation id -> role -> resolved name.
    """

    # A plain class, not a dataclass: the tests (and release_bundle.py) load
    # this file with importlib without registering it in sys.modules, which a
    # dataclass needs.
    def __init__(self) -> None:
        self.role_teachers: dict[str, dict[str, str]] = {}
        self.shared_corrector_reviewer_names: list[str] = []
        self.decisions: dict[str, int] = {}
        self.per_variation: dict[str, dict[str, str]] = {}


def decision_rule(row: dict[str, Any]) -> str:
    """``reviewer_b`` when reviewer B alone decided *row*'s acceptance, else ``both``.

    augment.py stamps ``decided_by`` on a record from a ``--decide-by
    reviewer_b`` run; a clean-slate re-review (``--rereview``, deviation d8)
    keeps only reviewer B's fresh verdict under ``verdicts`` and moves every
    earlier one to ``prior_verdicts``. An accepted record of a ``both`` run
    carries no verdicts at all.
    """
    if row.get("decided_by") == DECIDED_BY_REVIEWER_B:
        return DECIDED_BY_REVIEWER_B
    verdicts = row.get("verdicts")
    if isinstance(verdicts, dict) and "reviewer_b" in verdicts and "reviewer_a" not in verdicts:
        return DECIDED_BY_REVIEWER_B
    return DECIDED_BY_BOTH


def teacher_summary(
    train: list[dict[str, Any]],
    accepted_rows: dict[str, dict[str, Any]],
    role_models: dict[str, tuple[str, str]],
    *,
    apache_only: bool = False,
) -> TeacherSummary:
    """The teachers of every variation (an id with ``~v``) in *train*.

    Each variation must have an accepted record naming all four roles, each
    alias must be in *role_models*, and with *apache_only* every teacher's
    licence must be Apache-2.0.
    """
    summary = TeacherSummary()
    role_teachers: dict[str, dict[str, str]] = collections.defaultdict(dict)
    shared: dict[str, None] = {}
    decisions: collections.Counter[str] = collections.Counter()
    for entry in train:
        if "~v" not in entry["id"]:
            continue
        row = accepted_rows.get(entry["id"], {})
        models = row.get("models")
        if not models:
            raise ValueError(f"variation {entry['id']} has no accepted record naming its models")
        teachers: dict[str, str] = {}
        for role, _ in ROLES:
            if role not in models:
                raise ValueError(
                    f"variation {entry['id']}'s accepted record names no {role} teacher"
                )
            name, teacher_licence = _teacher(role_models, models[role])
            if apache_only and teacher_licence != APACHE_LICENCE:
                raise ValueError(
                    f"{entry['id']}: teacher {name!r} ({teacher_licence}) is not "
                    f"{APACHE_LICENCE}; refused by --apache-only"
                )
            teachers[role] = name
            role_teachers[role][name] = teacher_licence
        if teachers.get("CORRECTOR") == teachers.get("REVIEWER_B"):
            shared.setdefault(teachers["CORRECTOR"], None)
        decisions[decision_rule(row)] += 1
        summary.per_variation[entry["id"]] = teachers
    summary.role_teachers = dict(role_teachers)
    summary.shared_corrector_reviewer_names = list(shared)
    summary.decisions = dict(decisions)
    return summary


def role_description(role: str, decisions: dict[str, int]) -> str:
    """What *role* did in a run whose acceptances were decided as *decisions* says."""
    base = dict(ROLES)[role]
    by_b = decisions.get(DECIDED_BY_REVIEWER_B, 0)
    by_both = decisions.get(DECIDED_BY_BOTH, 0)
    if not by_b or role not in ("REVIEWER_A", "REVIEWER_B"):
        return base
    if role == "REVIEWER_A":
        if not by_both:
            return "reviewer A, advisory: asked and recorded, did not decide"
        return (
            f"accepted it (reviewer A; deciding for {by_both} variations," f" advisory for {by_b})"
        )
    return "accepted it (reviewer B, deciding)" if not by_both else base


def teacher_rows(summary: TeacherSummary) -> list[tuple[str, str, str]]:
    """``(name, licence, role description)`` per role and model, in ROLES order."""
    return [
        (name, licence, role_description(role, summary.decisions))
        for role, _ in ROLES
        for name, licence in summary.role_teachers.get(role, {}).items()
    ]


def decision_sentence(decisions: dict[str, int]) -> str:
    """How a rewrite came to be kept, for a card's prose."""
    by_b = decisions.get(DECIDED_BY_REVIEWER_B, 0)
    by_both = decisions.get(DECIDED_BY_BOTH, 0)
    asked = (
        "Two reviewer models were each asked whether the rewrite still calls for"
        " exactly that answer"
    )
    if not by_b:
        return f"{asked}, and a rewrite was kept only when both said yes."
    advisory = (
        "reviewer B's verdict alone decided whether a rewrite was kept; reviewer"
        " A's verdict, where it was asked, was recorded as advisory and did not decide"
    )
    if not by_both:
        return f"{asked}: {advisory}."
    return f"{asked}. For {by_both} kept variations both said yes; for {by_b}," f" {advisory}."


def split_seed(path: Path) -> int | None:
    """The seed split.py wrote into *path*'s header, if it names one."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    header = raw.get("header", "") if isinstance(raw, dict) else ""
    match = _SEED_RE.search(str(header))
    return int(match.group(1)) if match else None


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
    rejected: Path | list[Path],
    licence: Path,
    role_models: dict[str, tuple[str, str]],
    out: Path,
    apache_only: bool = False,
    issue: int = 39,
    model_repos: list[str] | None = None,
    scorer_train: Path | None = None,
) -> dict[str, Any]:
    """Write the data set folder to *out*; return the counts shown in the card.

    *scorer_train* (issue 53 t21) is the candidate scorer's own training file
    (``build_dataset.py --scorer-out``: per-row label maps and missing-candidate
    rows); it ships byte for byte as ``data/scorer-train.json``.

    *role_models* is this run's alias -> (name, licence) table (see
    ``load_role_models``). With *apache_only*, a teacher named by any
    accepted variation whose licence is not Apache-2.0 refuses the build.
    *rejected* is one file or several (their records are counted together).
    *issue* and *model_repos* name what the card says the data set trained.
    """
    train = _entries(train_augmented)
    sides = {"validation": _entries(splits / "val.json"), "test": _entries(splits / "test.json")}
    accepted_rows = {row["id"]: row for row in _jsonl(accepted)}
    rejected_files = [rejected] if isinstance(rejected, Path) else list(rejected)
    rejected_count = sum(len(_jsonl(path)) for path in rejected_files)
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
    #: Every accepted variation's teachers, aggregated per role (not just the
    #: first one seen), the resolved names shared by corrector and reviewer B
    #: (two aliases resolving to one model still count), and the decision rule.
    summary = teacher_summary(train, accepted_rows, role_models, apache_only=apache_only)
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
            row["teachers"] = summary.per_variation[entry["id"]]
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
    if scorer_train is not None:
        shutil.copyfile(scorer_train, out / "data" / "scorer-train.json")

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
    text = card(
        counts,
        summary,
        issue=issue,
        model_repos=model_repos,
        seed=split_seed(splits / "val.json"),
    )
    if scorer_train is not None:
        text += SCORER_TRAIN_NOTE
    (out / "README.md").write_text(text, encoding="utf-8")
    return counts


#: The card's note on ``data/scorer-train.json`` (issue 53 t21).
SCORER_TRAIN_NOTE = (
    "\n## Candidate-scorer training file\n\n"
    "`data/scorer-train.json` is the file the candidate scorer trained on: the train\n"
    "records above, each with its own offered candidates, letter map and gold label,\n"
    "plus missing-candidate rows (the right operation removed, answer: escalate), as\n"
    "written by nvsh's `scripts/lfm-finetune/build_dataset.py --scorer-out`.\n"
)


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


def _intro(issue: int, model_repos: list[str] | None) -> str:
    if model_repos is None and issue == 39:
        return (
            "It is built by the pipeline that\n"
            "trained and measured `jetson-ai-lab/lfm2.5-350m-nvsh-triage` (nvsh issue 39).\n"
            "The model's run used an earlier snapshot of the train side, with fewer\n"
            "variations; validation and test are the same."
        )
    if not model_repos:
        return f"It is built by nvsh's fine-tune pipeline (nvsh issue {issue})."
    repos = ", ".join(f"`{repo}`" for repo in model_repos)
    return (
        f"It is the data the pipeline behind\n{repos} was trained and measured on"
        f" (nvsh issue {issue})."
    )


#: How each issue's arguments were grounded, for the card's Limits section.
GROUNDING = {
    39: "- Arguments were grounded against a fixture machine. Two `power_set` modes\n"
    "  cannot be rendered there, which caps proposals on those entries.",
    46: "- Arguments were grounded against one fixed snapshot of a DGX Spark (the\n"
    "  measurement's ground snapshot); values that depend on the machine, such\n"
    "  as power modes and service names, may differ on another device.",
}
#: Issue 53 grounded the same way, against a snapshot rebuilt from its own splits.
GROUNDING[53] = GROUNDING[46]


def card(
    counts: dict[str, Any],
    summary: TeacherSummary,
    *,
    issue: int = 39,
    model_repos: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """The data set card. *summary* is the run's ``teacher_summary``: its

    teachers per role (aggregated across every accepted variation), every
    resolved model name that played both the corrector and reviewer-B roles
    in at least one variation -- by resolved identity, not by which alias
    named it -- and how acceptance was decided.
    """
    answers = counts["answers"]
    reviewed = counts["accepted"] + counts["rejected"]
    rate = f"{100 * counts['accepted'] / reviewed:.0f}%" if reviewed else "n/a"
    seeded = f"seed {seed}" if seed is not None else "seeded"
    run_log = RUN_LOGS.get(issue, RUN_LOGS[39])
    grounding = GROUNDING.get(issue, GROUNDING[39])
    if summary.role_teachers:
        teachers = "\n".join(
            f"| {name} | {licence} | {desc} |" for name, licence, desc in teacher_rows(summary)
        )
        disclosure = ""
        if summary.shared_corrector_reviewer_names:
            names = ", ".join(summary.shared_corrector_reviewer_names)
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
in words, or **escalate** to a full agent. {_intro(issue, model_repos)}

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
  `scripts/lfm-finetune/split.py` ({seeded}, stratified by answer, 70/15/15).
- **Supplement entries**: a small authored train-only set
  (`{SUPPLEMENT_FILE}`): requests to stop, shut down or disable a container
  or service, which Tier 2 must escalate because it has no such action,
  next to restart requests as contrasts.
- **Variations**: rewrites of train entries, made by a local pipeline
  (`scripts/lfm-finetune/augment.py`). One model rewrote the request and a
  second copyedited it; **neither saw the expected answer**.
  {decision_sentence(summary.decisions)} Deterministic
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
  table at the time of issue {issue}.
{grounding}
- Variations were reviewed by models, not people. The reviewers were shown
  nvsh's list of checks and changes, and every prompt fault found by
  reading samples was fixed before the accepted set was kept (nvsh's
  `{run_log}` run log), but some wrong labels may remain.
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
    parser.add_argument(
        "--rejected", required=True, type=Path, nargs="+", help="one or more rejected files"
    )
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
    parser.add_argument(
        "--issue", type=int, default=39, help="the nvsh issue the data set belongs to"
    )
    parser.add_argument(
        "--scorer-train", type=Path, help="the candidate scorer's own training file (issue 53)"
    )
    parser.add_argument(
        "--model-repo",
        action="append",
        dest="model_repos",
        help="a model repository trained on this data (repeatable); names it in the card",
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
            issue=args.issue,
            model_repos=args.model_repos,
            scorer_train=args.scorer_train,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
