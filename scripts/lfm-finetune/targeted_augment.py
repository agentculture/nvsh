#!/usr/bin/env python3
"""Targeted train-side augmentation for the v2 training set (issue 53, t15).

A development-machine tool; NEVER imported by the nvsh package.

The Qwen Tool-Jev comparison (docs/benchmarks/2026-09-24-qwen-tool-jev-
comparison.md, "Data-set recommendations for the next Tool-Jev iteration",
items 2-6) names five shapes the frozen training set lacks. Each is one
``--recipes`` entry here:

``missing-argument`` (item 2)
    Rule-based, no teacher. From the *train side's own* explicit operation
    entries whose operation takes an argument and whose request text names
    that argument's value, the value's span is replaced by a vague reference
    ("the service", "a different mode", "it") picked deterministically from
    ``--seed`` and the entry id, so the request no longer says which value
    is meant. Expected answer: escalate, class ``decline:missing_argument``.
    The new entry keeps ``pair_of`` (the original's id) and the original's
    ``source_id``, so the complete request and its stripped twin stay one
    group. Mechanical stripping can yield broken English, so each stripped
    text is put to the reviewer(s) twice: "does it leave the argument
    unspecified?" and "is it a natural request?" -- both must be yes.
``diagnosis-explain`` (item 3)
    Teacher-drafted contrastive pairs about one topic: "what does X mean /
    how does X work" (explain, with a one-sentence answer) against "why is
    X happening on my machine / fix X" (escalate, ``decline:diagnosis``).
    Both halves are reviewed; a pair is kept or dropped whole.
``power-set`` (item 4)
    Explicit-mode positives: for every operation in ``nvsh.ops.table`` with
    a ``choice`` argument and every one of its choices, requests naming that
    choice by a synonym. Nothing here names an operation: the targets come
    from the table (today that is the power-mode operation only).
``disambiguation`` (item 5)
    Per operation, requests whose wording could plausibly be mistaken for
    another listed operation, each with its single correct operation and
    arguments (validated with ``table.validate``) and the ``confusable``
    operation recorded; the reviewer must agree the gold is the most natural
    reading.
``hard-negative`` (item 6)
    Per operation, knowledge questions that mention what the operation deals
    with only in passing but want an answer in words (explain + answer).

Every entry is in corpus format (``id``, ``kind``, ``text``, ``expect``,
``class`` where it has one, ``source: "t15-<recipe>"``, ``source_id``), ids
prefixed per recipe (``t15-marg-0001``, ``t15-dx-...``, ``t15-pset-...``,
``t15-disamb-...``, ``t15-hneg-...``). A contrastive pair shares one
``source_id`` (``t15-dx-p0001``, or the original's for a stripped request);
``merge_variations.py --supplement`` keeps an entry's own ``source_id``. The
output's header names the train side (``Split 'train' of <train file>``), so
``merge_variations.py --supplement`` accepts it; counts, models, seed and the
entries' sha256 go under a top-level ``"t15"`` object.

Before any review call, a candidate is dropped when it names an internal
operation, copies answer-template wording or asks for a hand-off in so many
words (``augment.py``'s own guards), when it exactly repeats a train text,
an earlier kept text, or -- exactly or as a near-duplicate
(``leakage_check.match``) -- a text of an ``--exclude`` file. ``--exclude``
only compares and prints counts; ``leakage_check.py`` against every v2
evaluation side and the held-out remains the gate before the freeze.

Roles are ``augment.py``'s own (``NVSH_AUG_GENERATOR_*``, ``NVSH_AUG_REVIEWER_B_*``
and, with ``--decide-by both``, ``NVSH_AUG_REVIEWER_A_*``), or
``draft_sources.py``'s ``NVSH_DRAFT_*`` names with ``--roles-from draft``.
The client, retry policy, per-call request seeding, JSON-list parsing,
yes/no verdict parsing and guards are imported from ``augment.py`` /
``draft_sources.py``, never copied. ``missing-argument`` alone needs no
generator. Only counts and a sha256 are printed, never an entry's text;
``--review-out`` (optional) keeps every verdict with its text for audit.

Usage::

    NVSH_AUG_GENERATOR_URL=... NVSH_AUG_GENERATOR_MODEL=... \\
    NVSH_AUG_REVIEWER_B_URL=... NVSH_AUG_REVIEWER_B_MODEL=... \\
        python scripts/lfm-finetune/targeted_augment.py --train out/train.json \\
            --out out/t15-supplement.json --per-recipe 20 --seed 53 \\
            --exclude out/val.json out/test.json --review-out out/t15-review.jsonl

    python scripts/lfm-finetune/targeted_augment.py --train out/train.json \\
        --out out/t15-supplement.json --per-recipe 20 --dry-run
"""

from __future__ import annotations

import argparse
import http.client
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment as aug  # noqa: E402
import draft_sources as ds  # noqa: E402
import merge_variations as mv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory
from nvsh.ops import table  # noqa: E402

RECIPES: tuple[str, ...] = (
    "missing-argument",
    "diagnosis-explain",
    "power-set",
    "disambiguation",
    "hard-negative",
)
ID_TAGS: dict[str, str] = {
    "missing-argument": "marg",
    "diagnosis-explain": "dx",
    "power-set": "pset",
    "disambiguation": "disamb",
    "hard-negative": "hneg",
}
#: The recipes a teacher drafts (``missing-argument`` is rule-based).
GENERATED = frozenset(RECIPES) - {"missing-argument"}

#: Most items asked of the generator in one call; more are asked in rounds.
BATCH = 5

_HELD_OUT_NAME = "held-out.json"
_HELD_OUT_MARKER = "held-out split"

EXPLAIN_CLASS = "explain:question"

#: Failures of one model call (after retries) that count as ``error``, never crash a run.
_CALL_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
    KeyError,
    ValueError,
)

# Phrases the prompts below carry verbatim (tests route a fake caller on them).
MARG_UNSPECIFIED_MARKER = "leave the"
MARG_NATURAL_MARKER = "natural, grammatical request"
DX_MARKER = "contrastive pairs"
CHOICE_MARKER = "explicitly ask to set"
DISAMBIGUATION_MARKER = "could plausibly be mistaken"
HARD_NEGATIVE_MARKER = "only in passing"
MOST_NATURAL_MARKER = "most natural reading"


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One would-be entry: its text, answer, class, the review prompts that
    must all be accepted, and any extra fields it carries into the output."""

    text: str
    expect: dict[str, Any]
    cls: str | None
    reviews: list[tuple[str, str]]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Unit:
    """Items kept or dropped together (a contrastive pair is one unit).

    ``group`` is the ``source_id`` every item shares; ``None`` means a fresh
    one (the entry's own id for a single item, ``t15-<tag>-pNNNN`` for a pair).
    """

    recipe: str
    items: list[Item]
    group: str | None = None


def _bump(counts: dict[str, int], key: str, by: int = 1) -> None:
    counts[key] = counts.get(key, 0) + by


def _table_head() -> str:
    return f"Operations:\n{ds.dh.table_text(table)}\n\n"


# ---------------------------------------------------------------------------
# missing-argument (rule-based)
# ---------------------------------------------------------------------------

#: Words that may follow an argument value and belong to its span ("the
#: nginx server", "balanced mode"); the argument's own name is added too.
_SPAN_NOUNS = (
    "service",
    "server",
    "daemon",
    "container",
    "mode",
    "profile",
    "power mode",
    "power profile",
    "unit",
)
_SPAN_ARTICLES = ("the", "a", "an", "my", "this", "that", "our")

#: Vague references by argument kind; ``{noun}`` is the argument's own name.
#: A free-text argument loses its value to a definite-but-unnamed reference,
#: a choice argument to an unnamed alternative.
VAGUE_REFS: dict[str, tuple[str, ...]] = {
    "str": ("the {noun}", "that {noun}", "it"),
    "choice": ("a different {noun}", "another {noun}", "the other {noun}"),
}


def value_forms(value: str) -> list[str]:
    """How a request may spell *value*: verbatim, ``_``/``.``/``-`` as spaces
    or hyphens, and a dotted name's bare stem ("vllm" for "vllm.service").
    Longest first, so the widest span wins."""
    low = value.lower()
    forms = {low, low.replace("_", " "), low.replace("_", "-"), low.replace(".", " ")}
    if "." in low:
        stem = low.split(".", 1)[0]
        if len(stem) >= 3:
            forms.add(stem)
            forms.add(stem.replace("_", " "))
    return sorted((f for f in forms if f.strip()), key=lambda f: (-len(f), f))


def _span_re(arg: Any, value: str) -> re.Pattern[str]:
    forms = "|".join(re.escape(f) for f in value_forms(value))
    nouns = "|".join(
        re.escape(n) for n in sorted({*_SPAN_NOUNS, arg.name.lower()}, key=lambda n: (-len(n), n))
    )
    articles = "|".join(_SPAN_ARTICLES)
    return re.compile(
        rf"(?<![\w.-])(?:(?:{articles})\s+)?(?:{forms})(?:\s+(?:{nouns}))?(?![\w.-])",
        re.IGNORECASE,
    )


def _names_value(text: str, value: str) -> bool:
    return any(
        re.search(rf"(?<![\w.-]){re.escape(form)}(?![\w.-])", text, re.IGNORECASE)
        for form in value_forms(value)
    )


def strip_argument(text: str, arg: Any, value: str, pick: int) -> tuple[str | None, str]:
    """``(stripped text, "")``, or ``(None, reason)`` when *text* cannot be
    stripped: ``no_argument_span`` (the value is not spelled in it),
    ``too_short`` (nothing but the value), ``value_remains`` (a spelling
    survived). *pick* selects the vague reference, so the result is a pure
    function of its inputs."""
    pattern = _span_re(arg, value)
    if not pattern.search(text):
        return None, "no_argument_span"
    if not re.search(r"\w", pattern.sub(" ", text)):
        return None, "too_short"
    refs = VAGUE_REFS["choice" if arg.kind == "choice" else "str"]
    ref = refs[pick % len(refs)].format(noun=arg.name.lower())

    def _replace(match: re.Match[str]) -> str:
        if match.start() == 0 and text[:1].isupper():
            return ref[:1].upper() + ref[1:]
        return ref

    stripped = " ".join(pattern.sub(_replace, text).split())
    if _names_value(stripped, value):
        return None, "value_remains"
    return stripped, ""


def marg_unspecified_prompt(text: str, arg_name: str) -> tuple[str, str]:
    user = (
        f"{_table_head()}Request: {text}\n\n"
        f"The action this request asks for needs a {arg_name} to act on. Does the request "
        f"{MARG_UNSPECIFIED_MARKER} {arg_name} unspecified, so that the assistant cannot tell "
        f"from the request alone which {arg_name} is meant? Answer yes or no, then a short reason."
    )
    return ds.REVIEWER_SYSTEM, user


def marg_natural_prompt(text: str) -> tuple[str, str]:
    user = (
        f"Request: {text}\n\n"
        f"Is this a {MARG_NATURAL_MARKER} that a real user might type to a terminal assistant? "
        "Answer yes or no, then a short reason."
    )
    return ds.REVIEWER_SYSTEM, user


def missing_argument_units(
    train_entries: list[dict[str, Any]], per_op: int, seed: int, rejects: dict[str, int]
) -> list[Unit]:
    """Up to *per_op* stripped requests per operation that takes an argument,
    drawn from *train_entries* in an order shuffled by *seed*; deterministic."""
    by_op: dict[str, list[dict[str, Any]]] = {}
    for entry in train_entries:
        expect = entry.get("expect") or {}
        op = table.get(str(expect.get("operation")))
        if entry.get("kind") != "explicit" or op is None or not op.args:
            continue
        if table.validate(op.name, expect.get("args", {})) is not None:
            continue
        by_op.setdefault(op.name, []).append(entry)

    units: list[Unit] = []
    for op in table.OPERATIONS:
        pool = sorted(by_op.get(op.name, []), key=lambda e: str(e["id"]))
        random.Random(f"{seed}:missing-argument:{op.name}").shuffle(pool)
        made = 0
        for entry in pool:
            if made >= per_op:
                break
            pick = random.Random(f"{seed}:missing-argument:{entry['id']}").randrange(1 << 16)
            stripped, reason = None, "no_argument_span"
            for arg in op.args:
                stripped, reason = strip_argument(
                    entry["text"], arg, entry["expect"]["args"][arg.name], pick
                )
                if stripped is not None:
                    break
            if stripped is None:
                _bump(rejects, reason)
                continue
            made += 1
            item = Item(
                text=stripped,
                expect={"escalate": True},
                cls="decline:missing_argument",
                reviews=[
                    marg_unspecified_prompt(stripped, arg.name),
                    marg_natural_prompt(stripped),
                ],
                extra={"pair_of": str(entry["id"]), "stripped_arg": arg.name},
            )
            group = str(entry.get("source_id") or entry["id"])
            units.append(Unit("missing-argument", [item], group=group))
    return units


# ---------------------------------------------------------------------------
# teacher-drafted recipes
# ---------------------------------------------------------------------------

_DX_ASK = (
    "Write {k} {marker} of user messages about Jetson or DGX Spark topics: a kernel or driver "
    "log message, an error code, a power mode, a GPU or memory metric, a container runtime "
    "message. Both messages of a pair are about the same topic X and worded closely. "
    '"explain" is a definitional question -- what X means or how X works in general -- that '
    'is answered in one sentence without looking at the machine; "answer" is that one-sentence '
    'correct answer. "diagnose" asks why X is happening on the user\'s own machine right now, '
    "or asks to fix it -- that needs investigation, so the assistant must hand it back to a "
    "human instead of answering. Do not ask for the hand-off in so many words. Reply as a "
    'JSON list of objects with keys "topic", "explain", "answer" and "diagnose".'
)

_CHOICE_ASK = (
    "Write {k} different user requests that {marker} the {arg} of the operation {name} to "
    "{value!r}. Each request names that {arg} clearly, but in its own way: a synonym, an "
    "abbreviation, a vendor or tool term, or a plain description -- vary them. Each is a clear, "
    "direct request, never a question about what the {arg} is. Reply as a JSON list of "
    'objects with keys "text" and "args" (an object).'
)

_DISAMBIGUATION_ASK = (
    "Write {k} different user requests that should be handled by the operation {name}, but "
    "whose wording {marker} for another operation in the list (for example because it shares "
    "a keyword with it). For each, give the exact arguments {name} needs (use only allowed "
    "choices; for a free-text argument use a concrete value that appears in the request) and "
    "the other operation it could be confused with. Reply as a JSON list of objects with keys "
    '"text", "args" (an object) and "confusable" (the other operation\'s name).'
)

_HARD_NEGATIVE_ASK = (
    "Write {k} different user questions that mention what the operation {name} deals with "
    "{marker}, but want an explanation in words -- what something means or how it works -- "
    "and not for the operation to be run or anything on the machine to be checked or changed. "
    'Reply as a JSON list of objects with keys "text" and "answer" (a one-sentence correct '
    "answer)."
)


def dx_prompt(k: int) -> tuple[str, str]:
    return ds.GENERATOR_SYSTEM, _table_head() + _DX_ASK.format(k=k, marker=DX_MARKER)


def choice_prompt(op_name: str, arg_name: str, value: str, k: int) -> tuple[str, str]:
    ask = _CHOICE_ASK.format(k=k, marker=CHOICE_MARKER, arg=arg_name, name=op_name, value=value)
    return ds.GENERATOR_SYSTEM, _table_head() + ask


def disambiguation_prompt(op_name: str, k: int) -> tuple[str, str]:
    ask = _DISAMBIGUATION_ASK.format(k=k, marker=DISAMBIGUATION_MARKER, name=op_name)
    return ds.GENERATOR_SYSTEM, _table_head() + ask


def hard_negative_prompt(op_name: str, k: int) -> tuple[str, str]:
    ask = _HARD_NEGATIVE_ASK.format(k=k, marker=HARD_NEGATIVE_MARKER, name=op_name)
    return ds.GENERATOR_SYSTEM, _table_head() + ask


def most_natural_prompt(text: str, expect: dict[str, Any], confusable: str) -> tuple[str, str]:
    other = ds._expect_words({"operation": confusable, "args": {}}, None)
    user = (
        f"{_table_head()}Request: {text}\n\n"
        f"Is this the correct handling, and the {MOST_NATURAL_MARKER} of the request rather "
        f"than '{other}': {ds._expect_words(expect, None)}? Answer yes or no, then a short reason."
    )
    return ds.REVIEWER_SYSTEM, user


def choice_targets() -> list[tuple[Any, Any, str]]:
    """``(operation, argument, choice)`` for every choice argument in the table."""
    return [
        (op, arg, value)
        for op in table.OPERATIONS
        for arg in op.args
        if arg.kind == "choice"
        for value in arg.choices
    ]


def _rounds(k: int) -> list[int]:
    """Batch sizes that ask for *k* items in total, at most :data:`BATCH` each."""
    return [min(BATCH, k - start) for start in range(0, max(k, 0), BATCH)]


def _ask(
    role: aug.RoleConfig,
    prompt: Callable[[int], tuple[str, str]],
    k: int,
    caller: aug.RoleCaller,
    rejects: dict[str, int],
) -> list[dict[str, Any]]:
    """Every object the generator returns over :func:`_rounds` of *k*, capped
    at *k*; a failed call (after retries) counts ``error`` and is skipped."""
    items: list[dict[str, Any]] = []
    for size in _rounds(k):
        system, user = prompt(size)
        try:
            got = ds._generate_list(role, system, user, caller, rejects)
        except _CALL_ERRORS:
            _bump(rejects, "error")
            continue
        items.extend(item for item in got if isinstance(item, dict))
    return items[:k]


def _text(item: dict[str, Any], key: str = "text") -> str:
    value = item.get(key)
    return value.strip() if isinstance(value, str) else ""


def dx_units(
    role: aug.RoleConfig, k: int, caller: aug.RoleCaller, rejects: dict[str, int]
) -> list[Unit]:
    units = []
    for item in _ask(role, dx_prompt, k, caller, rejects):
        explain, answer, diagnose = (
            _text(item, "explain"),
            _text(item, "answer"),
            _text(item, "diagnose"),
        )
        if not (explain and answer and diagnose):
            _bump(rejects, "invalid_item")
            continue
        explain_expect = {"explain": True, "answer": answer}
        escalate_expect = {"escalate": True}
        units.append(
            Unit(
                "diagnosis-explain",
                [
                    Item(
                        explain,
                        explain_expect,
                        EXPLAIN_CLASS,
                        [ds.reviewer_prompt(explain, explain_expect)],
                    ),
                    Item(
                        diagnose,
                        escalate_expect,
                        "decline:diagnosis",
                        [ds.reviewer_prompt(diagnose, escalate_expect, "decline:diagnosis")],
                    ),
                ],
            )
        )
    return units


def choice_units(
    role: aug.RoleConfig, k: int, caller: aug.RoleCaller, rejects: dict[str, int]
) -> list[Unit]:
    units = []
    for op, arg, value in choice_targets():

        def prompt(size: int, op=op, arg=arg, value=value) -> tuple[str, str]:
            return choice_prompt(op.name, arg.name, value, size)

        for item in _ask(role, prompt, k, caller, rejects):
            text, args = _text(item), item.get("args")
            if not text or table.validate(op.name, args) is not None:
                _bump(rejects, "invalid_args")
                continue
            if args.get(arg.name) != value:
                _bump(rejects, "wrong_value")
                continue
            expect = {"operation": op.name, "args": dict(args)}
            units.append(
                Unit("power-set", [Item(text, expect, None, [ds.reviewer_prompt(text, expect)])])
            )
    return units


def disambiguation_units(
    role: aug.RoleConfig, k: int, caller: aug.RoleCaller, rejects: dict[str, int]
) -> list[Unit]:
    units = []
    known = set(table.names())
    for op in table.OPERATIONS:

        def prompt(size: int, op=op) -> tuple[str, str]:
            return disambiguation_prompt(op.name, size)

        for item in _ask(role, prompt, k, caller, rejects):
            text, args = _text(item), item.get("args")
            if not text or table.validate(op.name, args) is not None:
                _bump(rejects, "invalid_args")
                continue
            confusable = item.get("confusable")
            if confusable not in known or confusable == op.name:
                _bump(rejects, "invalid_confusable")
                continue
            expect = {"operation": op.name, "args": dict(args)}
            review = most_natural_prompt(text, expect, confusable)
            units.append(
                Unit(
                    "disambiguation",
                    [Item(text, expect, None, [review], extra={"confusable": confusable})],
                )
            )
    return units


def hard_negative_units(
    role: aug.RoleConfig, k: int, caller: aug.RoleCaller, rejects: dict[str, int]
) -> list[Unit]:
    units = []
    for op in table.OPERATIONS:

        def prompt(size: int, op=op) -> tuple[str, str]:
            return hard_negative_prompt(op.name, size)

        for item in _ask(role, prompt, k, caller, rejects):
            text, answer = _text(item), _text(item, "answer")
            if not text or not answer:
                _bump(rejects, "invalid_item")
                continue
            expect = {"explain": True, "answer": answer}
            review = ds.reviewer_prompt(text, expect)
            units.append(
                Unit(
                    "hard-negative",
                    [Item(text, expect, EXPLAIN_CLASS, [review], extra={"mentions": op.name})],
                )
            )
    return units


_DRAFTERS = {
    "diagnosis-explain": dx_units,
    "power-set": choice_units,
    "disambiguation": disambiguation_units,
    "hard-negative": hard_negative_units,
}


def _units_per_round(recipe: str) -> int:
    """Generator prompts one round of a recipe makes."""
    if recipe == "diagnosis-explain":
        return 1
    if recipe == "power-set":
        return len(choice_targets())
    return len(table.OPERATIONS)


def planned_teacher_calls(recipe: str, k: int) -> int:
    """Generator calls a recipe makes for ``--per-recipe`` *k* (no parse retries)."""
    if recipe == "missing-argument":
        return 0
    return len(_rounds(k)) * _units_per_round(recipe)


def planned_candidates(recipe: str, k: int) -> int:
    """Items a recipe asks the generator for, at most (a pair counts two)."""
    if recipe == "missing-argument":
        return 0
    per_item = 2 if recipe == "diagnosis-explain" else 1
    return max(k, 0) * _units_per_round(recipe) * per_item


# ---------------------------------------------------------------------------
# guards, dedupe, review
# ---------------------------------------------------------------------------


def guard_reason(text: str) -> str | None:
    """``augment.py``'s deterministic guards, reused: the first that fires."""
    if aug.names_internal_operation(text):
        return "identifier"
    if aug.copies_answer_template(text):
        return "template"
    if aug.asks_for_handoff(text):
        return "handoff"
    return None


class _Dedupe:
    """Exact repeats of train texts and of earlier kept texts; exact or
    near-duplicate repeats of a protected (``--exclude``) text."""

    def __init__(self, train_texts: list[str], protected_texts: list[str]):
        self.train = {mv._normal(t) for t in train_texts}
        self.kept: set[str] = set()
        self.protected = protected_texts

    def reason(self, unit: Unit) -> str | None:
        keys = [mv._normal(item.text) for item in unit.items]
        if len(set(keys)) != len(keys):
            return "duplicate"
        for item, key in zip(unit.items, keys):
            if key in self.train:
                return "train_exact"
            if key in self.kept:
                return "duplicate"
            kind = ds.duplicates_against(item.text, self.protected) if self.protected else None
            if kind == "exact":
                return "protected_exact"
            if kind == "near-duplicate":
                return "protected_near"
        return None

    def keep(self, unit: Unit) -> None:
        self.kept.update(mv._normal(item.text) for item in unit.items)


def review_unit(
    unit: Unit, roles: dict[str, aug.RoleConfig], caller: aug.RoleCaller, decide_by: str
) -> tuple[bool, list[dict[str, Any]]]:
    """Ask every review prompt of every item, clean slate; stop at the first
    reject. Reviewer B always decides; with *decide_by* ``both`` reviewer A
    must also say yes."""
    votes: list[dict[str, Any]] = []
    for item in unit.items:
        allowed = ds.ESCALATE_ALLOWED_HEDGES if item.expect.get("escalate") else ()
        for system, user in item.reviews:
            accept_b, reason_b = ds._vote(roles["REVIEWER_B"], system, user, caller, allowed)
            vote: dict[str, Any] = {"reviewer_b": {"accept": accept_b, "reason": reason_b}}
            ok = accept_b
            if decide_by == "both":
                accept_a, reason_a = ds._vote(roles["REVIEWER_A"], system, user, caller, allowed)
                vote["reviewer_a"] = {"accept": accept_a, "reason": reason_a}
                ok = ok and accept_a
            votes.append(vote)
            if not ok:
                return False, votes
    return True, votes


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_train(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A train-side split file; the held-out split and any other side are refused."""
    if path.name == _HELD_OUT_NAME:
        raise ValueError(f"{path}: the held-out split is never a training input")
    doc = json.loads(path.read_text(encoding="utf-8"))
    header = doc.get("header") if isinstance(doc, dict) else None
    header = header if isinstance(header, str) else ""
    if header.casefold().startswith(_HELD_OUT_MARKER):
        raise ValueError(f"{path}: the held-out split is never a training input")
    if not mv._TRAIN_HEADER.search(header):
        raise ValueError(f"{path}: its header does not name the train side")
    return doc, list(doc.get("entries", []))


def load_texts(path: Path) -> list[str]:
    """Every entry text of a split/corpus file (``{entries}``, a list, or JSONL)."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        doc = json.loads(raw)
        records = doc.get("entries", []) if isinstance(doc, dict) else doc
    return [r["text"] for r in records if isinstance(r, dict) and isinstance(r.get("text"), str)]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _check_recipes(recipes: tuple[str, ...]) -> None:
    unknown = [r for r in recipes if r not in RECIPES]
    if unknown:
        raise ValueError(f"unknown recipe(s) {unknown}; choose from {', '.join(RECIPES)}")


def plan(train_path: Path, recipes: tuple[str, ...], per_recipe: int, seed: int) -> dict:
    """What a run would attempt, calling nothing: generator calls and the most
    candidates per recipe (``missing-argument``: the stripped requests, exactly)."""
    _check_recipes(recipes)
    _, entries = load_train(train_path)
    out: dict[str, dict[str, int]] = {}
    for recipe in recipes:
        if recipe == "missing-argument":
            rejects: dict[str, int] = {}
            units = missing_argument_units(entries, per_recipe, seed, rejects)
            out[recipe] = {"teacher_calls": 0, "candidates": len(units), **rejects}
        else:
            out[recipe] = {
                "teacher_calls": planned_teacher_calls(recipe, per_recipe),
                "candidates": planned_candidates(recipe, per_recipe),
            }
    return out


def run(
    train_path: Path,
    out: Path,
    recipes: tuple[str, ...],
    per_recipe: int,
    seed: int,
    roles: dict[str, aug.RoleConfig],
    caller: aug.RoleCaller,
    exclude: list[Path] | tuple[Path, ...] = (),
    decide_by: str = "reviewer_b",
    review_out: Path | None = None,
    max_retries: int = aug.DEFAULT_MAX_RETRIES,
    backoff_base: float = aug.DEFAULT_BACKOFF_BASE,
    sleep_fn: Callable[[float], None] = time.sleep,
    progress: Any = None,
) -> dict[str, Any]:
    """Draft, guard, dedupe and review every recipe; write the supplement to
    *out* (and every verdict to *review_out*); return counts and the sha256."""
    _check_recipes(recipes)
    if decide_by not in aug.DECIDE_BY_RULES:
        raise ValueError(f"decide_by must be one of {aug.DECIDE_BY_RULES}, got {decide_by!r}")
    _, train_entries = load_train(train_path)
    protected = [text for path in exclude for text in load_texts(Path(path))]
    policy = aug.RetryPolicy(max_retries=max_retries, backoff_base=backoff_base, sleep_fn=sleep_fn)

    def call(role: aug.RoleConfig, system: str, user: str) -> str:
        return aug._call_with_retry(caller, role, system, user, policy)

    dedupe = _Dedupe([str(e.get("text", "")) for e in train_entries], protected)
    stream = progress if progress is not None else sys.stderr
    entries: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, Any]] = {}
    for recipe in recipes:
        tag = ID_TAGS[recipe]
        rejects: dict[str, int] = {}
        if recipe == "missing-argument":
            units = missing_argument_units(train_entries, per_recipe, seed, rejects)
        else:
            units = _DRAFTERS[recipe](roles["GENERATOR"], per_recipe, call, rejects)
        n_entry, n_pair, kept_units = 0, 0, 0
        for unit in units:
            reason = next(
                (r for r in (guard_reason(item.text) for item in unit.items) if r), None
            ) or dedupe.reason(unit)
            votes: list[dict[str, Any]] = []
            if reason is None:
                try:
                    accepted, votes = review_unit(unit, roles, call, decide_by)
                except _CALL_ERRORS:
                    reason = "error"
                else:
                    if not accepted:
                        reason = "reviewer"
                        for who in ("reviewer_a", "reviewer_b"):
                            if who in votes[-1] and not votes[-1][who]["accept"]:
                                _bump(rejects, who)
            if reason is not None and reason != "reviewer":
                _bump(rejects, reason)
            ids: list[str | None] = [None] * len(unit.items)
            if reason is None:
                kept_units += 1
                dedupe.keep(unit)
                group = unit.group
                if group is None and len(unit.items) > 1:
                    n_pair += 1
                    group = f"t15-{tag}-p{n_pair:04d}"
                for index, item in enumerate(unit.items):
                    n_entry += 1
                    entry_id = f"t15-{tag}-{n_entry:04d}"
                    ids[index] = entry_id
                    entry: dict[str, Any] = {
                        "id": entry_id,
                        "kind": "explicit",
                        "text": item.text,
                        "expect": item.expect,
                        "source": f"t15-{recipe}",
                        "source_id": group or entry_id,
                    }
                    if item.cls:
                        entry["class"] = item.cls
                    entry.update(item.extra)
                    entries.append(entry)
            review_rows.append(
                {
                    "recipe": recipe,
                    "ids": ids,
                    "texts": [item.text for item in unit.items],
                    "accepted": reason is None,
                    "reason": reason,
                    "votes": votes,
                }
            )
        counts[recipe] = {"kept": n_entry, "units_kept": kept_units, "rejected": rejects}
        print(
            f"t15: {recipe}: kept={n_entry} units_kept={kept_units} "
            f"rejected={json.dumps(rejects, sort_keys=True)}",
            file=stream,
        )

    sha256 = ds._sha256_of_entries(entries)
    meta = {
        "tool": "targeted_augment.py",
        "task": "issue 53 t15",
        "train": {"name": train_path.name, "sha256": _sha256_file(train_path)},
        "exclude": [{"name": Path(p).name, "sha256": _sha256_file(Path(p))} for p in exclude],
        "recipes": list(recipes),
        "per_recipe": per_recipe,
        "seed": seed,
        "decide_by": decide_by,
        "models": {role: cfg.model for role, cfg in sorted(roles.items())},
        "sampling": {
            "per_call_seed": True,
            "note": (
                "reproducible only on endpoints that honour the request seed; the "
                "missing-argument transform itself is deterministic under the seed"
            ),
        },
        "counts": counts,
        "sha256": sha256,
    }
    header = (
        f"Split 'train' of {train_path.name}: a train-only targeted supplement "
        "(issue 53, t15) written by targeted_augment.py; its recipes, seed, counts and "
        'sha256 are under "t15".'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"header": header, "t15": meta, "entries": entries}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    if review_out is not None:
        review_out.parent.mkdir(parents=True, exist_ok=True)
        review_out.write_text(
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in review_rows
            ),
            encoding="utf-8",
        )
    return {"out": str(out), "kept": len(entries), "recipes": counts, "sha256": sha256}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_roles(needed: tuple[str, ...], source: str) -> dict[str, aug.RoleConfig]:
    """*needed* roles from ``augment.py``'s ``NVSH_AUG_*`` variables or
    ``draft_sources.py``'s ``NVSH_DRAFT_*`` ones; each loader reused as is."""
    loader = ds.load_role_config if source == "draft" else aug.load_role_config
    return {role: loader(role) for role in needed}


def _make_caller(seed: int) -> aug.RoleCaller:
    """``draft_sources.py``'s per-call-seeded client (the real HTTP call)."""
    return ds.make_seeded_caller(seed)


def _parse_recipes(value: str) -> tuple[str, ...]:
    recipes = tuple(r.strip() for r in value.split(",") if r.strip())
    try:
        _check_recipes(recipes)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return recipes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, type=Path, help="the v2 train.json (split.py)")
    parser.add_argument("--out", required=True, type=Path, help="supplement JSON to write")
    parser.add_argument(
        "--recipes",
        type=_parse_recipes,
        default=RECIPES,
        help=f"comma-separated subset of {','.join(RECIPES)} (default: all)",
    )
    parser.add_argument(
        "--per-recipe",
        type=int,
        default=5,
        help=(
            "items asked per recipe unit: per operation with an argument (missing-argument), "
            "pairs in total (diagnosis-explain), per choice (power-set), per operation "
            "(disambiguation, hard-negative)"
        ),
    )
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="protected sides whose texts (exact or near-duplicate) must not land; counts only",
    )
    parser.add_argument("--decide-by", choices=aug.DECIDE_BY_RULES, default="reviewer_b")
    parser.add_argument("--roles-from", choices=("augment", "draft"), default="augment")
    parser.add_argument("--review-out", type=Path, default=None, help="verdicts JSONL (has text)")
    parser.add_argument("--max-retries", type=int, default=aug.DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--backoff", type=float, default=aug.DEFAULT_BACKOFF_BASE, dest="backoff_base"
    )
    parser.add_argument("--dry-run", action="store_true", help="plan only; no endpoint is called")
    args = parser.parse_args(argv)

    if args.dry_run:
        try:
            planned = plan(args.train, args.recipes, args.per_recipe, args.seed)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"dry_run": True, "seed": args.seed, "recipes": planned}, sort_keys=True))
        return 0

    needed = ["REVIEWER_B"]
    if args.decide_by == "both":
        needed.insert(0, "REVIEWER_A")
    if GENERATED & set(args.recipes):
        needed.insert(0, "GENERATOR")
    try:
        roles = _load_roles(tuple(needed), args.roles_from)
    except aug.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        summary = run(
            train_path=args.train,
            out=args.out,
            recipes=args.recipes,
            per_recipe=args.per_recipe,
            seed=args.seed,
            roles=roles,
            caller=_make_caller(args.seed),
            exclude=args.exclude,
            decide_by=args.decide_by,
            review_out=args.review_out,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
