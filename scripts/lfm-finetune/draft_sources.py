#!/usr/bin/env python3
"""Draft new sources for the fresh evaluation sides and the sealed held-out (issue 53, t13).

A development-machine tool; NEVER imported by the nvsh package.

Two commands:

``draft OUT --pool {eval,heldout} --seed N``
    Ask a generator model, table-only (never dev.json or any split -- see
    ``draft_heldout.py``'s own docstring for why), for new candidate
    requests: ``--per-op`` per operation (with exact args, validated with
    ``nvsh.ops.table.validate``), ``--per-reason`` per decline reason (one
    of the 8 classes below, expect ``{"escalate": True}``, ``class
    "decline:<reason>"``), and ``--explain`` knowledge questions with a
    one-sentence answer. Every surviving candidate (after dedupe) is then
    judged independently by two reviewer models with a strict yes/no
    prompt; only entries both accept are kept.

``review IN_DRAFT OUT``
    Run the same two-reviewer check (plus the dedupe/near-dup check against
    dev.json) on an existing ``{"header", "entries"}`` draft file -- the
    shape ``draft_heldout.py`` writes. This is how the *sealed* held-out
    draft is reviewed: the lead runs ``draft_heldout.py`` (Qwen3.5-4B,
    table-only) to generate it, then this command to review it, and never
    reads the entries. Entries from ``draft_heldout.py`` carry no
    ``class``, so an escalate entry is judged only on whether no listed
    operation can safely handle it.

Neither command ever prints an entry's text; ``draft.json``/``review.jsonl``
carry it (except ``review.jsonl`` for the held-out pool, which never does,
since it stays sealed even from the review record). ``nvsh/tiers/corpus/
held-out.json`` -- the actual sealed file -- is refused outright as an
``IN_DRAFT`` input; it is never something this script reads.

Every model role is an OpenAI-compatible chat-completions endpoint,
configured entirely from the environment, in ``augment.py``'s own scheme
(``NVSH_DRAFT_<ROLE>_URL`` / ``_MODEL`` / ``_KEY_ENV`` / ``_TIMEOUT`` /
``_TEMPERATURE`` / ``_DISABLE_THINKING`` / ``_REASONING_EFFORT`` for
``GENERATOR``, ``REVIEWER_A``, ``REVIEWER_B``) -- never a literal URL, key
or model name in this file. The HTTP client, retry policy and yes/no
verdict parser are ``augment.py``'s own, reused by import, not copied.

Usage::

    NVSH_DRAFT_GENERATOR_URL=... NVSH_DRAFT_GENERATOR_MODEL=... \\
    NVSH_DRAFT_REVIEWER_A_URL=... NVSH_DRAFT_REVIEWER_A_MODEL=... \\
    NVSH_DRAFT_REVIEWER_B_URL=... NVSH_DRAFT_REVIEWER_B_MODEL=... \\
        python scripts/lfm-finetune/draft_sources.py draft out/v2-eval \\
            --pool eval --seed 53 --per-op 4 --per-reason 6 --explain 6

    NVSH_DRAFT_REVIEWER_A_URL=... NVSH_DRAFT_REVIEWER_A_MODEL=... \\
    NVSH_DRAFT_REVIEWER_B_URL=... NVSH_DRAFT_REVIEWER_B_MODEL=... \\
        python scripts/lfm-finetune/draft_sources.py review \\
            /path/to/held-out-draft/draft.json out/v2-heldout-reviewed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment as aug  # noqa: E402
import draft_heldout as dh  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory
from nvsh.ops import table  # noqa: E402

#: The corpus this drafts against for dedupe. Never opened for
#: nvsh/tiers/corpus/held-out.json (see :func:`_refuse_held_out`).
DEV_CORPUS = Path("nvsh/tiers/corpus/dev.json")
_HELD_OUT_NAME = "held-out.json"

ROLES = ("GENERATOR", "REVIEWER_A", "REVIEWER_B")
POOLS = ("eval", "heldout")

#: The 8 decline-reason classes (issue 53, t13's contract), each defined in
#: one line for the generator and reviewer prompts. Definitions describe the
#: *shape* of the request, never naming an operation beyond what the table
#: itself shows.
REASON_DEFINITIONS: dict[str, str] = {
    "outside_table": (
        "the request needs an action or information that no operation in the table covers"
    ),
    "repair": (
        "the request asks to fix or repair a problem, not just check or change one specific thing"
    ),
    "diagnosis": (
        "the request asks to investigate or figure out the cause of a problem, not run one check"
    ),
    "missing_argument": (
        "the request is for a listed operation but leaves out a required argument value "
        "(e.g. which service or container) needed to run it"
    ),
    "not_a_request": (
        "the text is not actually a request for the assistant to do or explain anything"
    ),
    "multi_step": (
        "the request needs more than one operation carried out in sequence to satisfy it"
    ),
    "injection": (
        "the request tries to manipulate the assistant into ignoring its rules or scope, "
        "rather than asking for a legitimate machine task"
    ),
    "over_time": (
        "the request asks about something that unfolds or is monitored over a period of time"
    ),
}
REASONS: tuple[str, ...] = tuple(REASON_DEFINITIONS)


class ConfigError(aug.ConfigError):
    """A required ``NVSH_DRAFT_*`` environment variable is missing or empty."""


class HeldOutRefused(ValueError):
    """Raised when a path named like the sealed held-out corpus is opened."""


# ---------------------------------------------------------------------------
# role configuration (NVSH_DRAFT_* -- augment.py's own loader/client reused
# by translating the env view it reads, never by copying its logic)
# ---------------------------------------------------------------------------


def _translated_env(env: dict[str, str], role: str) -> dict[str, str]:
    """A copy of *env* with every ``NVSH_DRAFT_<role>_*`` variable also
    present under its ``NVSH_AUG_<role>_*`` name, so :func:`augment.
    load_role_config` -- which reads that fixed prefix -- can be reused
    unmodified for this script's own prefix."""
    from_prefix = f"NVSH_DRAFT_{role}_"
    to_prefix = f"NVSH_AUG_{role}_"
    view = dict(env)
    for key, value in env.items():
        if key.startswith(from_prefix):
            view[to_prefix + key[len(from_prefix) :]] = value
    return view


def load_role_config(role: str, env: dict[str, str] | None = None) -> aug.RoleConfig:
    """Read ``NVSH_DRAFT_<role>_{URL,MODEL,KEY_ENV,MAX_TOKENS,DISABLE_THINKING,
    TIMEOUT,TEMPERATURE,REASONING_EFFORT}``, in exactly ``augment.py``'s own
    shape and validation, reused by import. Raises :class:`ConfigError`
    naming the ``NVSH_DRAFT_*`` variable that is missing."""
    import os

    source = os.environ if env is None else env
    try:
        return aug.load_role_config(role, _translated_env(dict(source), role))
    except aug.ConfigError as exc:
        raise ConfigError(str(exc).replace("NVSH_AUG_", "NVSH_DRAFT_")) from exc


def load_roles(
    roles: tuple[str, ...], env: dict[str, str] | None = None
) -> dict[str, aug.RoleConfig]:
    return {role: load_role_config(role, env) for role in roles}


RoleCaller = aug.RoleCaller
default_caller = aug.default_caller


# ---------------------------------------------------------------------------
# generator prompts (table-only: never dev.json or a split, like draft_heldout.py)
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM = (
    "You write realistic test requests for a shell assistant on NVIDIA Jetson and DGX Spark "
    "machines. The assistant can only act through the operations listed below. Write the way a "
    "busy engineer types at a terminal: short, varied wording, sometimes informal, sometimes with "
    "typos, terse, sometimes as a full sentence, drawing on real Jetson/DGX Spark contexts. Never "
    "copy an operation's name or description word for word. Reply with JSON only."
)

_OP_ASK = (
    "Write {k} different user requests that should be handled by the operation {name}. "
    "For each, give the exact arguments it needs (use only allowed choices; for a free-text "
    "argument use a concrete realistic value that appears in the request). Reply as a JSON "
    'list of objects with keys "text" and "args" (an object).'
)

_DECLINE_ASK = (
    "Write {k} different user requests that the assistant must NOT handle with any of these "
    "operations and must hand back to a human instead, because {definition}. Do not ask for the "
    "hand-off in so many words -- write it the way a real user would phrase the request itself. "
    'Reply as a JSON list of objects with a single key "text".'
)

_EXPLAIN_ASK = (
    "Write {k} different user questions that should be answered in words, without running "
    "anything: questions about what a power mode, a container runtime, CUDA, unified memory, a "
    "log message or one of these tools means or how it works. Reply as a JSON list of objects "
    'with keys "text" and "answer" (a one-sentence correct answer).'
)


def op_prompt(op_name: str, k: int) -> tuple[str, str]:
    head = f"Operations:\n{dh.table_text(table)}\n\n"
    return GENERATOR_SYSTEM, head + _OP_ASK.format(k=k, name=op_name)


def decline_prompt(reason: str, k: int) -> tuple[str, str]:
    head = f"Operations:\n{dh.table_text(table)}\n\n"
    return GENERATOR_SYSTEM, head + _DECLINE_ASK.format(k=k, definition=REASON_DEFINITIONS[reason])


def explain_prompt(k: int) -> tuple[str, str]:
    head = f"Operations:\n{dh.table_text(table)}\n\n"
    return GENERATOR_SYSTEM, head + _EXPLAIN_ASK.format(k=k)


#: Bounded retries on a generator reply that does not parse as JSON.
MAX_PARSE_RETRIES = 3


def _generate_list(
    role: aug.RoleConfig,
    system: str,
    user: str,
    caller: RoleCaller,
    rejects: dict[str, int],
) -> list[dict[str, Any]]:
    """Call *caller* for one generator prompt, retrying a parse failure up
    to :data:`MAX_PARSE_RETRIES` times before giving up on this prompt
    entirely (counted as ``rejects["parse"]``)."""
    for _ in range(MAX_PARSE_RETRIES):
        raw = caller(role, system, user)
        try:
            items = dh.parse_json_list(raw)
        except Exception:  # noqa: BLE001 - any parse failure is a retry
            rejects["parse"] = rejects.get("parse", 0) + 1
            continue
        if isinstance(items, list):
            return items
        rejects["parse"] = rejects.get("parse", 0) + 1
    return []


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


class Candidate:
    """One drafted entry before ids are assigned: everything but ``id``/
    ``source_id``, plus the ``kind_tag`` its id is built from."""

    __slots__ = ("kind_tag", "text", "expect", "cls")

    def __init__(self, kind_tag: str, text: str, expect: dict[str, Any], cls: str | None = None):
        self.kind_tag = kind_tag
        self.text = text
        self.expect = expect
        self.cls = cls


def _op_candidates(
    role: aug.RoleConfig, k: int, caller: RoleCaller, rejects: dict[str, int]
) -> list[Candidate]:
    out: list[Candidate] = []
    for op in table.OPERATIONS:
        if k <= 0:
            continue
        system, user = op_prompt(op.name, k)
        for item in _generate_list(role, system, user, caller, rejects):
            text = str(item.get("text", "")).strip()
            args = item.get("args") or {}
            if not text or not isinstance(args, dict):
                rejects["invalid_args"] = rejects.get("invalid_args", 0) + 1
                continue
            if table.validate(op.name, args) is not None:
                rejects["invalid_args"] = rejects.get("invalid_args", 0) + 1
                continue
            out.append(Candidate(f"op-{op.name}", text, {"operation": op.name, "args": args}))
    return out


def _decline_candidates(
    role: aug.RoleConfig, k: int, caller: RoleCaller, rejects: dict[str, int]
) -> list[Candidate]:
    out: list[Candidate] = []
    if k <= 0:
        return out
    for reason in REASONS:
        system, user = decline_prompt(reason, k)
        for item in _generate_list(role, system, user, caller, rejects):
            text = str(item.get("text", "")).strip()
            if not text:
                rejects["invalid_args"] = rejects.get("invalid_args", 0) + 1
                continue
            out.append(
                Candidate(f"decline-{reason}", text, {"escalate": True}, f"decline:{reason}")
            )
    return out


def _explain_candidates(
    role: aug.RoleConfig, k: int, caller: RoleCaller, rejects: dict[str, int]
) -> list[Candidate]:
    out: list[Candidate] = []
    if k <= 0:
        return out
    system, user = explain_prompt(k)
    for item in _generate_list(role, system, user, caller, rejects):
        text = str(item.get("text", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not text or not answer:
            rejects["invalid_args"] = rejects.get("invalid_args", 0) + 1
            continue
        out.append(Candidate("explain", text, {"explain": True, "answer": answer}))
    return out


# ---------------------------------------------------------------------------
# dedupe (exact + near-duplicate, reusing leakage_check.py's helpers)
# ---------------------------------------------------------------------------


def _leakage_check():
    """Lazily import ``leakage_check`` (same-directory sys.path trick as
    ``augment.py``/``calibrate_reviewer.py``), so it is only ever imported
    once and its shingle/word-Jaccard helpers are reused, never copied."""
    if "leakage_check" not in sys.modules:
        import leakage_check  # noqa: F401
    return sys.modules["leakage_check"]


def load_dev_texts(dev_path: Path = DEV_CORPUS) -> list[str]:
    doc = json.loads(dev_path.read_text(encoding="utf-8"))
    return [e["text"] for e in doc["entries"] if isinstance(e.get("text"), str) and e["text"]]


def duplicates_against(text: str, others: list[str]) -> str | None:
    """``"exact"``, ``"near-duplicate"`` or ``None``, reusing
    ``leakage_check.match`` (never reimplementing the threshold)."""
    lc = _leakage_check()
    for other in others:
        kind = lc.match(text, other)
        if kind is not None:
            return kind
    return None


def dedupe_candidates(
    candidates: list[Candidate], dev_texts: list[str], rejects: dict[str, int]
) -> list[Candidate]:
    """Drop a candidate that exact-or-near-duplicates dev.json or an
    earlier-kept candidate, in stable (input) order."""
    kept: list[Candidate] = []
    kept_texts: list[str] = []
    for candidate in candidates:
        kind = duplicates_against(candidate.text, dev_texts) or duplicates_against(
            candidate.text, kept_texts
        )
        if kind == "exact":
            rejects["dedupe_exact"] = rejects.get("dedupe_exact", 0) + 1
            continue
        if kind == "near-duplicate":
            rejects["dedupe_near"] = rejects.get("dedupe_near", 0) + 1
            continue
        kept.append(candidate)
        kept_texts.append(candidate.text)
    return kept


# ---------------------------------------------------------------------------
# reviewer (strict yes/no, both must accept)
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM = (
    "You are a strict reviewer for a training dataset that teaches a small assistant when to run "
    "one of a fixed set of operations, when to hand a request off to a human, and when to just "
    "answer a question in words. Given the operation table and a request, judge whether the "
    "stated handling is exactly correct. Answer strictly 'yes' or 'no', then a short reason. "
    "Start your reply with the single word 'yes' or 'no'."
)


def _expect_words(expect: dict[str, Any], cls: str | None) -> str:
    if expect.get("escalate"):
        base = "no listed operation can safely handle this request, so it is escalated to a human"
        reason = (cls or "").removeprefix("decline:")
        if reason in REASON_DEFINITIONS:
            base += f", because {REASON_DEFINITIONS[reason]}"
        return base
    if expect.get("explain"):
        answer = expect.get("answer", "")
        return f"this is a knowledge question needing no action, correctly answered as: {answer}"
    name = str(expect.get("operation"))
    op = table.get(name)
    what = op.description.rstrip(".").lower() if op is not None else "the expected action"
    values = ", ".join(f"{k} {v}" for k, v in sorted((expect.get("args") or {}).items()))
    detail = f" -- {values}" if values else ""
    verb = (
        "run this read-only check" if (op is not None and op.read_only) else "propose this change"
    )
    return f"{verb}: {what}{detail}"


def reviewer_prompt(text: str, expect: dict[str, Any], cls: str | None = None) -> tuple[str, str]:
    head = f"Operation table:\n{dh.table_text(table)}\n\n"
    user = (
        f"{head}Request: {text}\n\n"
        f"Is this the correct handling: {_expect_words(expect, cls)}? "
        "Answer yes or no, then a short reason."
    )
    return REVIEWER_SYSTEM, user


def judge(
    entry_text: str,
    expect: dict[str, Any],
    cls: str | None,
    roles: dict[str, aug.RoleConfig],
    caller: RoleCaller,
) -> dict[str, Any]:
    """Ask REVIEWER_A and REVIEWER_B independently; ``votes`` records both,
    ``accepted`` is true only when both said yes."""
    system, user = reviewer_prompt(entry_text, expect, cls)
    accept_a, reason_a = aug.parse_verdict(caller(roles["REVIEWER_A"], system, user))
    accept_b, reason_b = aug.parse_verdict(caller(roles["REVIEWER_B"], system, user))
    return {
        "accepted": accept_a and accept_b,
        "votes": {
            "reviewer_a": {"accept": accept_a, "reason": reason_a},
            "reviewer_b": {"accept": accept_b, "reason": reason_b},
        },
    }


# ---------------------------------------------------------------------------
# ids, header, output
# ---------------------------------------------------------------------------


def _make_id(pool: str, kind_tag: str, n: int) -> str:
    return f"v2-{pool}-{kind_tag}-{n:03d}"


def _entry_from_candidate(entry_id: str, candidate: Candidate) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": entry_id,
        "kind": "explicit",
        "text": candidate.text,
        "expect": candidate.expect,
        "source_id": entry_id,
    }
    if candidate.cls:
        entry["class"] = candidate.cls
    return entry


def _sha256_of_entries(entries: list[dict[str, Any]]) -> str:
    body = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _write_outputs(
    out_dir: Path,
    header: dict[str, Any],
    entries: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    strip_text_from_review: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"header": header, "entries": entries}
    (out_dir / "draft.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = []
    for row in review_rows:
        row = dict(row)
        if strip_text_from_review:
            row.pop("text", None)
        lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    (out_dir / "review.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _refuse_held_out(path: Path) -> None:
    if path.name == _HELD_OUT_NAME:
        raise HeldOutRefused(f"{path}: the sealed held-out corpus is never read by this tool")


# ---------------------------------------------------------------------------
# draft
# ---------------------------------------------------------------------------


def run_draft(
    out_dir: Path,
    pool: str,
    seed: int,
    per_op: int,
    per_reason: int,
    explain: int,
    caller: RoleCaller = default_caller,
    roles: dict[str, aug.RoleConfig] | None = None,
    dev_texts: list[str] | None = None,
) -> dict[str, Any]:
    if pool not in POOLS:
        raise ValueError(f"--pool must be one of {POOLS}, got {pool!r}")
    random.seed(seed)
    roles = roles if roles is not None else load_roles(ROLES)
    dev_texts = dev_texts if dev_texts is not None else load_dev_texts()

    rejects: dict[str, int] = {}
    candidates: list[Candidate] = []
    candidates += _op_candidates(roles["GENERATOR"], per_op, caller, rejects)
    candidates += _decline_candidates(roles["GENERATOR"], per_reason, caller, rejects)
    candidates += _explain_candidates(roles["GENERATOR"], explain, caller, rejects)

    candidates = dedupe_candidates(candidates, dev_texts, rejects)

    counters: dict[str, int] = {}
    rejected_counters: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        outcome = judge(candidate.text, candidate.expect, candidate.cls, roles, caller)
        if outcome["accepted"]:
            counters[candidate.kind_tag] = counters.get(candidate.kind_tag, 0) + 1
            entry_id = _make_id(pool, candidate.kind_tag, counters[candidate.kind_tag])
            entries.append(_entry_from_candidate(entry_id, candidate))
        else:
            if not outcome["votes"]["reviewer_a"]["accept"]:
                rejects["reviewer_a"] = rejects.get("reviewer_a", 0) + 1
            if not outcome["votes"]["reviewer_b"]["accept"]:
                rejects["reviewer_b"] = rejects.get("reviewer_b", 0) + 1
            rejected_counters[candidate.kind_tag] = rejected_counters.get(candidate.kind_tag, 0) + 1
            entry_id = (
                f"v2-{pool}-{candidate.kind_tag}-rejected-"
                f"{rejected_counters[candidate.kind_tag]:03d}"
            )
        review_rows.append({"id": entry_id, "text": candidate.text, "votes": outcome["votes"]})

    by_kind: dict[str, int] = {}
    for entry in entries:
        key = "operation" if "operation" in entry["expect"] else next(iter(entry["expect"]))
        by_kind[key] = by_kind.get(key, 0) + 1
    by_reason: dict[str, int] = {}
    for entry in entries:
        if entry.get("class", "").startswith("decline:"):
            reason = entry["class"].removeprefix("decline:")
            by_reason[reason] = by_reason.get(reason, 0) + 1

    models = {role: cfg.model for role, cfg in roles.items()}
    header = {
        "tool": "draft_sources.py draft",
        "pool": pool,
        "seed": seed,
        "per_op": per_op,
        "per_reason": per_reason,
        "explain": explain,
        "models": models,
        "counts": {"kept": len(entries), "by_kind": by_kind, "by_reason": by_reason, **rejects},
    }
    sha256 = _sha256_of_entries(entries)
    header["sha256"] = sha256

    _write_outputs(
        out_dir, header, entries, review_rows, strip_text_from_review=(pool == "heldout")
    )
    return {
        "pool": pool,
        "seed": seed,
        "kept": len(entries),
        "by_kind": by_kind,
        "by_reason": by_reason,
        "rejects": rejects,
        "sha256": sha256,
    }


# ---------------------------------------------------------------------------
# review (post-hoc review of an existing draft_heldout.py output)
# ---------------------------------------------------------------------------


def run_review(
    in_path: Path,
    out_dir: Path,
    caller: RoleCaller = default_caller,
    roles: dict[str, aug.RoleConfig] | None = None,
    dev_texts: list[str] | None = None,
) -> dict[str, Any]:
    _refuse_held_out(in_path)
    roles = roles if roles is not None else load_roles(("REVIEWER_A", "REVIEWER_B"))
    dev_texts = dev_texts if dev_texts is not None else load_dev_texts()

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    source_entries = doc.get("entries", [])

    rejects: dict[str, int] = {}
    kept_entries: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    kept_texts: list[str] = []
    for entry in source_entries:
        text = entry.get("text", "")
        entry_id = entry.get("id", "")
        kind = duplicates_against(text, dev_texts) or duplicates_against(text, kept_texts)
        if kind == "exact":
            rejects["dedupe_exact"] = rejects.get("dedupe_exact", 0) + 1
            continue
        if kind == "near-duplicate":
            rejects["dedupe_near"] = rejects.get("dedupe_near", 0) + 1
            continue
        outcome = judge(text, entry.get("expect", {}), entry.get("class"), roles, caller)
        review_rows.append({"id": entry_id, "text": text, "votes": outcome["votes"]})
        if outcome["accepted"]:
            kept_entries.append(entry)
            kept_texts.append(text)
        else:
            if not outcome["votes"]["reviewer_a"]["accept"]:
                rejects["reviewer_a"] = rejects.get("reviewer_a", 0) + 1
            if not outcome["votes"]["reviewer_b"]["accept"]:
                rejects["reviewer_b"] = rejects.get("reviewer_b", 0) + 1

    by_kind: dict[str, int] = {}
    for entry in kept_entries:
        key = (
            "operation"
            if "operation" in entry.get("expect", {})
            else next(iter(entry.get("expect", {})), "unknown")
        )
        by_kind[key] = by_kind.get(key, 0) + 1

    models = {role: cfg.model for role, cfg in roles.items()}
    header = {
        "tool": "draft_sources.py review",
        "pool": "heldout",
        "input": str(in_path),
        "input_header": doc.get("header"),
        "models": models,
        "counts": {"kept": len(kept_entries), "by_kind": by_kind, **rejects},
    }
    sha256 = _sha256_of_entries(kept_entries)
    header["sha256"] = sha256

    _write_outputs(out_dir, header, kept_entries, review_rows, strip_text_from_review=True)
    return {
        "pool": "heldout",
        "input": str(in_path),
        "kept": len(kept_entries),
        "by_kind": by_kind,
        "rejects": rejects,
        "sha256": sha256,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_draft_parser(sub: Any) -> None:
    p = sub.add_parser("draft", help="draft new candidate entries from the operation table")
    p.add_argument("out", type=Path, help="output directory for draft.json/review.jsonl")
    p.add_argument("--pool", choices=POOLS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--per-op", type=int, default=3, help="requests per operation")
    p.add_argument("--per-reason", type=int, default=4, help="requests per decline reason")
    p.add_argument("--explain", type=int, default=6, help="knowledge questions to draft")
    p.add_argument("--dry-run", action="store_true", help="build prompts and print counts only")


def _add_review_parser(sub: Any) -> None:
    p = sub.add_parser("review", help="review an existing draft_heldout.py draft file")
    p.add_argument("in_draft", type=Path, help="a {header, entries} draft.json to review")
    p.add_argument("out", type=Path, help="output directory for draft.json/review.jsonl")
    p.add_argument("--dry-run", action="store_true", help="load and count entries only")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    _add_draft_parser(sub)
    _add_review_parser(sub)
    args = parser.parse_args(argv)

    if args.command == "draft":
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "pool": args.pool,
                        "seed": args.seed,
                        "planned": {
                            "operations": len(table.OPERATIONS) * max(args.per_op, 0),
                            "reasons": len(REASONS) * max(args.per_reason, 0),
                            "explain": max(args.explain, 0),
                        },
                    }
                )
            )
            return 0
        try:
            roles = load_roles(ROLES)
        except ConfigError as exc:
            parser.error(str(exc))
            return 2  # pragma: no cover - parser.error already exits
        try:
            result = run_draft(
                out_dir=args.out,
                pool=args.pool,
                seed=args.seed,
                per_op=args.per_op,
                per_reason=args.per_reason,
                explain=args.explain,
                roles=roles,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result))
        return 0

    # command == "review"
    try:
        _refuse_held_out(args.in_draft)
    except HeldOutRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        doc = json.loads(args.in_draft.read_text(encoding="utf-8"))
        print(json.dumps({"dry_run": True, "entries": len(doc.get("entries", []))}))
        return 0
    try:
        roles = load_roles(("REVIEWER_A", "REVIEWER_B"))
    except ConfigError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error already exits
    try:
        result = run_review(in_path=args.in_draft, out_dir=args.out, roles=roles)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
