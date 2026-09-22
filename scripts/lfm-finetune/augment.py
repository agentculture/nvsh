#!/usr/bin/env python3
"""Augmentation pipeline: generate, correct, double review (part of #39).

A development-machine tool for the Tier 2 (LFM2.5) fine-tune. It is NEVER
imported by the nvsh package -- nothing under ``nvsh/`` may depend on it.

Given seed requests whose *answer is already fixed* -- either a split-side
file written by ``scripts/lfm-finetune/split.py`` ({"header", "entries"},
each entry carrying ``id``, ``text``, ``expect``, ``source_id``) or a
``tools.json`` written by ``scripts/lfm-finetune/jetson_skills.py`` (a list
of ``{"skill", "repo", "tool"}`` records) -- this script asks four model
roles to turn each seed into ``--per-source`` paraphrased variations without
ever letting the model choose or drift the answer:

1. **generator** is told the fixed answer and asked only to rephrase the
   seed request (or, for a skill seed, to write a new request that the
   skill's own description says it answers).
2. **corrector** fixes grammar/clarity only; it is told the same fixed
   answer and must not change what the request means.
3. **reviewer_a** and **reviewer_b** each independently answer strictly
   "yes" or "no" to "does this request still mean exactly this answer?".
   A variation is accepted only if *both* say yes. When the fixed answer is
   read-only or an escalation, the reviewer is additionally asked whether
   the request could be read as asking for a change to the machine -- a
   "could" is a reject (h30), since a read-only/escalate answer must never
   drift into a mutating one.

Every model role is an OpenAI-compatible chat-completions endpoint,
configured entirely from the environment -- never a literal URL or key in
this file (``scripts/scan-secrets.py`` enforces that, and Check 2 there
would fail on a committed non-localhost endpoint anyway). For role
``<ROLE>`` in ``GENERATOR``, ``CORRECTOR``, ``REVIEWER_A``, ``REVIEWER_B``::

    NVSH_AUG_<ROLE>_URL              full chat-completions URL (required)
    NVSH_AUG_<ROLE>_MODEL            model id to send (required)
    NVSH_AUG_<ROLE>_KEY_ENV          name of the env var holding the bearer
                                     key (optional -- omitted means no
                                     Authorization header; the key itself is
                                     read at call time via ``grant run``,
                                     never stored here)
    NVSH_AUG_<ROLE>_MAX_TOKENS       reply budget (optional, default 1024).
                                     A reasoning model spends tokens
                                     "thinking" before writing its answer,
                                     so a tiny budget yields an empty reply.
    NVSH_AUG_<ROLE>_DISABLE_THINKING "1"/"true" sends
                                     ``chat_template_kwargs: {"enable_thinking":
                                     false}`` (optional, off by default --
                                     not every server honours it)

A missing required variable is a clear, named error, never a silent
default.

Usage::

    NVSH_AUG_GENERATOR_URL=... NVSH_AUG_GENERATOR_MODEL=... \\
    NVSH_AUG_CORRECTOR_URL=... NVSH_AUG_CORRECTOR_MODEL=... \\
    NVSH_AUG_REVIEWER_A_URL=... NVSH_AUG_REVIEWER_A_MODEL=... \\
    NVSH_AUG_REVIEWER_B_URL=... NVSH_AUG_REVIEWER_B_MODEL=... \\
        python scripts/lfm-finetune/augment.py out/train.json \\
            --per-source 20 --accepted-out accepted.jsonl --rejected-out rejected.jsonl

Accepted variations go to ``--accepted-out`` (one JSON object per line);
rejected ones go to ``--rejected-out`` with the source id, every role's
model id and each reviewer's verdict + reason, so a rejection is auditable
without re-running the pipeline. The run is resumable: a variation id
already present in either output file is skipped. NVIDIA's own eval files
are refused outright as seeds -- see :func:`load_seeds`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.ops.table import get as get_operation  # noqa: E402

# ---------------------------------------------------------------------------
# seed refusal (NVIDIA evals are never a seed)
# ---------------------------------------------------------------------------

#: jetson_skills.py stamps every eval record with these fields (device and
#: BSP shapes alike, once normalized by that script's EvalRecord). A seed
#: file carrying any of them is an eval file, not a seed, and is refused.
_BANNED_EVAL_FIELDS = frozenset({"expected_skill", "names_skill", "ground_truth"})

#: split.py's own held-out guard, mirrored here: the held-out split is for
#: judging a tuned model, never for generating training data from.
_HELD_OUT_NAME = "held-out.json"

ROLES = ("GENERATOR", "CORRECTOR", "REVIEWER_A", "REVIEWER_B")


class SeedRefused(ValueError):
    """Raised when a file that looks like an NVIDIA eval is passed as a seed."""


class ConfigError(ValueError):
    """A required ``NVSH_AUG_*`` environment variable is missing or empty."""


# ---------------------------------------------------------------------------
# role configuration (env vars only -- never a literal endpoint or key)
# ---------------------------------------------------------------------------


#: A reasoning model (as opposed to an instruct-only one) spends its
#: max_tokens budget "thinking" before it ever writes ``content``; a tiny
#: budget then returns an empty reply from every role except a non-reasoning
#: one. This default is generous enough to leave room for an answer after
#: the model's own reasoning trace.
DEFAULT_MAX_TOKENS = 1024

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class RoleConfig:
    role: str
    url: str
    model: str
    key: str | None = field(default=None, repr=False)  # never repr'd/printed
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Some OpenAI-compatible servers honour ``chat_template_kwargs`` to turn
    #: off a reasoning model's thinking pass; off by default since not every
    #: server understands it, and it is per-role because only some of the
    #: four roles here are reasoning models.
    disable_thinking: bool = False


def load_role_config(role: str, env: dict[str, str] | None = None) -> RoleConfig:
    """Read ``NVSH_AUG_<role>_{URL,MODEL,KEY_ENV,MAX_TOKENS,DISABLE_THINKING}``
    from *env* (default ``os.environ``). Raises :class:`ConfigError` naming
    the exact variable that is missing."""
    source = os.environ if env is None else env
    url_var = f"NVSH_AUG_{role}_URL"
    model_var = f"NVSH_AUG_{role}_MODEL"
    key_env_var = f"NVSH_AUG_{role}_KEY_ENV"
    max_tokens_var = f"NVSH_AUG_{role}_MAX_TOKENS"
    disable_thinking_var = f"NVSH_AUG_{role}_DISABLE_THINKING"

    url = source.get(url_var)
    if not url:
        raise ConfigError(f"{url_var} is not set")
    model = source.get(model_var)
    if not model:
        raise ConfigError(f"{model_var} is not set")

    key: str | None = None
    key_env_name = source.get(key_env_var)
    if key_env_name:
        key = source.get(key_env_name)
        if not key:
            raise ConfigError(f"{key_env_var} names {key_env_name!r}, but that variable is not set")

    max_tokens_raw = source.get(max_tokens_var)
    if max_tokens_raw:
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            raise ConfigError(f"{max_tokens_var} must be an integer, got {max_tokens_raw!r}")
    else:
        max_tokens = DEFAULT_MAX_TOKENS

    disable_thinking = source.get(disable_thinking_var, "").strip().lower() in _TRUE_STRINGS

    return RoleConfig(
        role=role,
        url=url,
        model=model,
        key=key,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )


def load_all_roles(env: dict[str, str] | None = None) -> dict[str, RoleConfig]:
    return {role: load_role_config(role, env) for role in ROLES}


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    source_id: str
    seed_format: str  # "split" or "skills" -- which seed file shape this came from
    side: str | None
    seed_text: str  # text to rephrase (split), or the skill's own description
    expect: dict[str, Any]  # split: the entry's own expect block; skill: {"skill": name}
    needs_change_check: bool  # h30: ask the extra "could this be a change?" question
    #: For a split seed only: the source entry's own corpus fields --
    #: ``kind`` ("explicit"/"failure"), and ``source``/``class`` when
    #: present -- carried through unchanged so an accepted/rejected record
    #: still loads via ``nvsh.tiers.bench.load_corpus``. Empty for a skill
    #: seed, which is not a corpus entry at all.
    corpus_fields: dict[str, Any] = field(default_factory=dict)


def _refuse_if_eval(path: Path, records: list[Any]) -> None:
    if path.name == _HELD_OUT_NAME:
        raise SeedRefused(f"{path}: the held-out split is never a seed")
    for record in records:
        if isinstance(record, dict) and _BANNED_EVAL_FIELDS & record.keys():
            fields = sorted(_BANNED_EVAL_FIELDS & record.keys())
            raise SeedRefused(
                f"{path}: looks like an NVIDIA eval file (has {fields}); refused as a seed"
            )


#: split.py writes this header text (``_write_side``'s ``note``) onto every
#: side file it produces: "Split 'train' of dev.json (seed=42).". Matching it
#: lets a seed file be recognized even when it was renamed away from
#: train/val/test.json.
_HEADER_SIDE_RE = re.compile(r"split\s+['\"](train|val|test)['\"]", re.IGNORECASE)


def _infer_side(path: Path, header: str | None = None) -> str | None:
    if path.stem in ("train", "val", "test"):
        return path.stem
    if header:
        match = _HEADER_SIDE_RE.search(header)
        if match:
            return match.group(1).lower()
    return None


def _resolve_side(path: Path, header: str | None, side: str | None) -> str:
    """Reconcile an explicit ``--side`` against the side inferred from *path*
    itself (filename stem or split.py's own header text).

    ``--side`` exists only for a file whose own side cannot be told: when the
    file's side *can* be inferred, ``--side`` must agree with it or the run
    refuses outright, rather than silently relabeling entries onto the wrong
    side.
    """
    inferred = _infer_side(path, header)
    if side is not None:
        if inferred is not None and side != inferred:
            raise ConfigError(
                f"{path}: --side {side!r} conflicts with this file's own side "
                f"{inferred!r}; --side is only for files whose side cannot be inferred"
            )
        return side
    if inferred is None:
        raise ConfigError(
            f"{path}: cannot infer the split side from the filename or header; "
            "pass --side explicitly"
        )
    return inferred


def _needs_change_check(expect: dict[str, Any]) -> bool:
    """True when *expect*'s fixed answer is read-only or an escalation (h30):
    the reviewer must then also refuse any variation that could be read as
    asking for a machine change. Explaining something is read-only in the
    same sense (it never mutates the machine), so "explain" answers get the
    same guard. An unrecognized operation name is treated conservatively as
    needing the guard, rather than silently skipping it.
    """
    if expect.get("escalate"):
        return True
    if "explain" in expect:
        return True
    operation_name = expect.get("operation")
    if operation_name is None:
        return True
    operation = get_operation(operation_name)
    if operation is None:
        return True
    return bool(operation.read_only)


def _seed_from_split_entry(entry: dict[str, Any], side: str) -> Seed:
    expect = entry["expect"]
    if "kind" not in entry:
        raise ValueError(
            f"{entry.get('id', entry.get('source_id', '?'))}: split seed entry is missing "
            "its own corpus 'kind' (explicit/failure)"
        )
    corpus_fields: dict[str, Any] = {"kind": entry["kind"]}
    if "source" in entry:
        corpus_fields["source"] = entry["source"]
    if "class" in entry:
        corpus_fields["class"] = entry["class"]
    return Seed(
        source_id=entry.get("source_id", entry["id"]),
        seed_format="split",
        side=side,
        seed_text=entry["text"],
        expect=expect,
        needs_change_check=_needs_change_check(expect),
        corpus_fields=corpus_fields,
    )


def _seed_from_skill_record(record: dict[str, Any], side: str | None) -> Seed:
    skill_name = record["skill"]
    description = record.get("tool", {}).get("function", {}).get("description", "")
    return Seed(
        source_id=skill_name,
        seed_format="skills",
        side=side,
        seed_text=description,
        expect={"skill": skill_name},
        needs_change_check=False,
    )


def _looks_like_skill_records(records: list[Any]) -> bool:
    return bool(records) and all(
        isinstance(r, dict) and "skill" in r and "tool" in r for r in records
    )


def load_seeds(path: Path, side: str | None = None) -> list[Seed]:
    """Load seeds from *path*: a split-side file ({"header", "entries"}) or
    a jetson_skills.py ``tools.json`` (a list of ``{"skill", "repo",
    "tool"}``). Refuses any file that looks like an NVIDIA eval (h30's
    sibling rule) -- see :func:`_refuse_if_eval`.
    """
    header: str | None = None
    if path.suffix == ".jsonl":
        records: list[Any] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "entries" in raw:
            records = raw["entries"]
            header = raw.get("header")
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError(f"{path}: unrecognized seed file shape")

    _refuse_if_eval(path, records)

    if _looks_like_skill_records(records):
        return [_seed_from_skill_record(r, side) for r in records]

    resolved_side = _resolve_side(path, header, side)
    return [_seed_from_split_entry(r, resolved_side) for r in records]


def _expected_description(seed: Seed) -> str:
    if seed.seed_format == "skills":
        return f"the {seed.expect['skill']!r} capability, and only that capability, handles this"
    return json.dumps(seed.expect, sort_keys=True)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM_SPLIT = (
    "You rewrite user requests for a training dataset. You are given a fixed "
    "answer and an original request that produces it. Rewrite the request in "
    "different words. Do not change what is being asked for: the rewritten "
    "request must still produce exactly the same fixed answer. Reply with "
    "only the rewritten request, nothing else."
)

GENERATOR_SYSTEM_SKILL = (
    "You write realistic user requests for a training dataset that teaches a "
    "router which capability should handle a request. You are given the "
    "description of one capability. Write one natural user request that this "
    "capability -- and only this capability -- would answer. Do not name the "
    "capability outright. Reply with only the request, nothing else."
)

CORRECTOR_SYSTEM = (
    "You copyedit short user requests for a training dataset. Fix grammar, "
    "spelling and clarity only. You are given the fixed answer the request "
    "must produce; it must stay exactly the same after your edit -- do not "
    "change what the request is asking for. Reply with only the corrected "
    "request, nothing else."
)

REVIEWER_SYSTEM = (
    "You are a strict reviewer for a training dataset. You are given a user "
    "request and the fixed answer it must mean. Answer strictly 'yes' or "
    "'no' to whether the request still means exactly this answer, then a "
    "short reason. Start your reply with the single word 'yes' or 'no'."
)

REVIEWER_SYSTEM_CHANGE_CHECK = REVIEWER_SYSTEM + (
    " The fixed answer here is read-only or an escalation, so it must never "
    "involve changing the machine. If the request could reasonably be read "
    "as asking for a change to be made to the machine, answer 'no' even if "
    "it otherwise matches."
)


def generator_prompt(seed: Seed) -> tuple[str, str]:
    """Return ``(system, user)`` for the generator role."""
    if seed.seed_format == "skills":
        user = (
            f"Capability description: {seed.seed_text}\n\n"
            "Write one user request this capability answers."
        )
        return GENERATOR_SYSTEM_SKILL, user
    user = (
        f"Fixed answer (do not change this): {_expected_description(seed)}\n\n"
        f"Original request: {seed.seed_text}\n\n"
        "Rewrite the request above in different words, keeping exactly the same meaning."
    )
    return GENERATOR_SYSTEM_SPLIT, user


def corrector_prompt(seed: Seed, text: str) -> tuple[str, str]:
    user = (
        f"Fixed answer (must stay exactly this): {_expected_description(seed)}\n\n"
        f"Request to copyedit:\n{text}"
    )
    return CORRECTOR_SYSTEM, user


def reviewer_prompt(seed: Seed, text: str) -> tuple[str, str]:
    system = REVIEWER_SYSTEM_CHANGE_CHECK if seed.needs_change_check else REVIEWER_SYSTEM
    user = (
        f"Fixed answer: {_expected_description(seed)}\n\n"
        f"User request: {text}\n\n"
        "Does this request still mean exactly this answer? Answer 'yes' or 'no' "
        "and then a short reason."
    )
    return system, user


# ---------------------------------------------------------------------------
# the OpenAI-compatible client (stdlib urllib only)
# ---------------------------------------------------------------------------


def _extract_content(message: dict[str, Any]) -> str:
    """Only ever read ``message.content`` as the answer.

    A reasoning model may also return ``reasoning_content`` / ``reasoning``
    holding its thinking trace; that field is never treated as the answer,
    even when ``content`` is empty -- an empty ``content`` is simply an
    empty reply, handled by the caller (a generator/corrector empty reply
    is an error; a reviewer empty reply is a reject, per h30's neighbor
    rule: no answer is never treated as an accept).
    """
    return (message.get("content") or "").strip()


def _post_chat_completion(role: RoleConfig, system: str, user: str, timeout: float = 60.0) -> str:
    payload: dict[str, Any] = {
        "model": role.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": role.max_tokens,
    }
    if role.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if role.key:
        headers["Authorization"] = f"Bearer {role.key}"
    request = urllib.request.Request(
        role.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return _extract_content(body["choices"][0]["message"])


#: The call signature every role invocation uses; tests may inject a fake
#: (e.g. to skip real HTTP) but the default is the real client above.
RoleCaller = Callable[[RoleConfig, str, str], str]


def default_caller(role: RoleConfig, system: str, user: str) -> str:
    return _post_chat_completion(role, system, user)


# ---------------------------------------------------------------------------
# reviewer verdict parsing (robust: anything but a clear "yes" is "no")
# ---------------------------------------------------------------------------

#: The first word, allowing it to be wrapped in markdown emphasis/quoting
#: (``**yes**``, ``"yes"``, ``_yes_``) and followed by one piece of
#: sentence punctuation (``yes,`` / ``yes:`` / ``yes.``).
_FIRST_WORD_RE = re.compile(r"""^[\s*_`"']*([A-Za-z]+)[.,:;]?""")

#: A verdict that hedges -- even one that starts with "yes" -- is not a
#: clear accept. Matched as a standalone word so "couldn't"/"unlikely"-style
#: words don't false-positive.
_HEDGE_WORDS = (
    "but",
    "however",
    "although",
    "though",
    "unless",
    "could",
    "might",
    "may",
    "ambiguous",
    "unclear",
    "partially",
)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
_HEDGE_RE = re.compile(r"\b(" + "|".join(_HEDGE_WORDS) + r")\b", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[bool, str]:
    """Parse a reviewer's free-text reply into ``(accepted, reason)``.

    A verdict is an accept ONLY if the first word is exactly "yes" (allowing
    surrounding markdown and one trailing punctuation mark) AND the rest of
    the reply contains no standalone "no" and none of :data:`_HEDGE_WORDS`.
    Anything else -- including a hedged "yes, but ..." -- is a reject, with
    the full reply kept as the reason so a rejection is auditable. An empty
    reply is a reject with reason "empty reply", never an accept.
    """
    stripped = text.strip()
    if not stripped:
        return False, "empty reply"
    match = _FIRST_WORD_RE.match(stripped)
    first_word = match.group(1) if match else ""
    if first_word.lower() != "yes":
        return False, stripped
    rest = stripped[match.end() :]
    if _NO_RE.search(rest) or _HEDGE_RE.search(rest):
        return False, stripped
    reason = rest.strip(" \t\n*_`\"'.,:;-—") or stripped
    return True, reason


def _reviewer_verdict(
    role: RoleConfig, system: str, user: str, caller: RoleCaller
) -> tuple[bool, str]:
    """Call one reviewer and parse its verdict. An empty reply (e.g. a
    reasoning model that spent its whole budget thinking) is never an
    accept -- it is logged as a reject with reason "empty reply"."""
    text = caller(role, system, user)
    return parse_verdict(text)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineCounts:
    generated: int = 0
    corrected: int = 0
    accepted: int = 0
    rejected_by_a: int = 0
    rejected_by_b: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "corrected": self.corrected,
            "accepted": self.accepted,
            "rejected_by_a": self.rejected_by_a,
            "rejected_by_b": self.rejected_by_b,
            "errors": self.errors,
        }


def _existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "id" in record:
                ids.add(record["id"])
    return ids


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _validate_seed_consistency(seeds: list[Seed]) -> None:
    """Refuse before any model call if one ``source_id`` shows up with more
    than one side or more than one ``expect`` block -- two seeds that
    disagree about what their shared id even means must never both be
    accepted under the same variation id ``"{source_id}~vN"``.
    """
    seen: dict[str, Seed] = {}
    for seed in seeds:
        prior = seen.get(seed.source_id)
        if prior is None:
            seen[seed.source_id] = seed
            continue
        if prior.side != seed.side:
            raise ValueError(
                f"source_id {seed.source_id!r} appears with more than one side "
                f"({prior.side!r} and {seed.side!r})"
            )
        if prior.expect != seed.expect:
            raise ValueError(
                f"source_id {seed.source_id!r} appears with more than one expect block "
                f"({prior.expect!r} and {seed.expect!r})"
            )


def _process_variation(
    seed: Seed,
    variation_id: str,
    roles: dict[str, RoleConfig],
    counts: PipelineCounts,
    caller: RoleCaller,
) -> dict[str, Any]:
    """Run one seed through generator -> corrector -> both reviewers.

    Returns the record to append, tagged with ``_accepted`` (bool) so the
    caller knows which file to write it to.
    """
    gen_system, gen_user = generator_prompt(seed)
    generated_text = caller(roles["GENERATOR"], gen_system, gen_user).strip()
    if not generated_text:
        raise ValueError("empty reply from generator")
    counts.generated += 1

    cor_system, cor_user = corrector_prompt(seed, generated_text)
    corrected_text = caller(roles["CORRECTOR"], cor_system, cor_user).strip()
    if not corrected_text:
        raise ValueError("empty reply from corrector")
    counts.corrected += 1

    rev_system, rev_user = reviewer_prompt(seed, corrected_text)
    accept_a, reason_a = _reviewer_verdict(roles["REVIEWER_A"], rev_system, rev_user, caller)
    accept_b, reason_b = _reviewer_verdict(roles["REVIEWER_B"], rev_system, rev_user, caller)

    if not accept_a:
        counts.rejected_by_a += 1
    if not accept_b:
        counts.rejected_by_b += 1

    models = {role: cfg.model for role, cfg in roles.items()}
    verdicts = {
        "reviewer_a": {"accept": accept_a, "reason": reason_a},
        "reviewer_b": {"accept": accept_b, "reason": reason_b},
    }

    accepted = accept_a and accept_b
    # The record keeps the source entry's own corpus fields (kind/source/
    # class for a split seed; nothing for a skill seed, which is not a
    # corpus entry) and separately records which seed file shape produced
    # it, so an accepted split-seed record still loads via
    # nvsh.tiers.bench.load_corpus (bug: it used to overwrite "kind" with
    # "split"/"skill", which load_corpus rejects as an unknown kind).
    record: dict[str, Any] = {
        "id": variation_id,
        "source_id": seed.source_id,
        "side": seed.side,
        "seed_format": seed.seed_format,
        "text": corrected_text,
        "expect": seed.expect,
        "models": models,
    }
    record.update(seed.corpus_fields)
    if accepted:
        counts.accepted += 1
    else:
        record["verdicts"] = verdicts
    return {"accepted": accepted, "record": record}


def run_pipeline(
    seed_files: list[Path],
    roles: dict[str, RoleConfig],
    accepted_out: Path,
    rejected_out: Path,
    per_source: int,
    side: str | None = None,
    limit: int | None = None,
    caller: RoleCaller = default_caller,
) -> PipelineCounts:
    seeds: list[Seed] = []
    for seed_file in seed_files:
        seeds.extend(load_seeds(seed_file, side))
    _validate_seed_consistency(seeds)

    done = _existing_ids(accepted_out) | _existing_ids(rejected_out)
    counts = PipelineCounts()
    processed = 0

    for seed in seeds:
        for n in range(1, per_source + 1):
            if limit is not None and processed >= limit:
                return counts
            variation_id = f"{seed.source_id}~v{n}"
            if variation_id in done:
                continue
            try:
                outcome = _process_variation(seed, variation_id, roles, counts, caller)
            except (urllib.error.URLError, KeyError, ValueError) as exc:
                counts.errors += 1
                print(f"error: {variation_id}: {exc}", file=sys.stderr)
                processed += 1
                continue
            if outcome["accepted"]:
                _append_jsonl(accepted_out, outcome["record"])
            else:
                _append_jsonl(rejected_out, outcome["record"])
            # Reserve the id the moment it is written: two seeds that share a
            # source_id (already proven consistent above, e.g. the same
            # entry loaded from two seed files) must never both write the
            # same variation id in one run.
            done.add(variation_id)
            processed += 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed_files", nargs="+", help="split-side file(s) or a jetson tools.json")
    parser.add_argument(
        "--side",
        default=None,
        help=(
            "train/val/test; only for a file whose side cannot be inferred from its "
            "filename or header -- must agree with an inferable file's own side"
        ),
    )
    parser.add_argument("--per-source", type=int, default=3, help="variations to attempt per seed")
    parser.add_argument("--limit", type=int, default=None, help="cap total variations (dry runs)")
    parser.add_argument("--accepted-out", default="accepted.jsonl")
    parser.add_argument("--rejected-out", default="rejected.jsonl")
    args = parser.parse_args(argv)

    try:
        roles = load_all_roles()
    except ConfigError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error already exits

    try:
        counts = run_pipeline(
            seed_files=[Path(p) for p in args.seed_files],
            roles=roles,
            accepted_out=Path(args.accepted_out),
            rejected_out=Path(args.rejected_out),
            per_source=args.per_source,
            side=args.side,
            limit=args.limit,
        )
    except (SeedRefused, ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for key, value in counts.as_dict().items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
