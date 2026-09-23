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
    NVSH_AUG_<ROLE>_TIMEOUT          per-request timeout in seconds
                                     (optional, default 120)

A missing required variable is a clear, named error, never a silent
default.

This is a process the operator runs and repeats themselves, at large scale
(``--per-source`` x100, tens of thousands of requests total), without an
agent watching. Two things follow from that: it processes variations
through a bounded thread pool (``--workers``) instead of one at a time, and
every role call retries transient failures (429/500/502/503/504, connection
errors, timeouts) with exponential backoff before counting a variation as an
error.

Usage::

    NVSH_AUG_GENERATOR_URL=... NVSH_AUG_GENERATOR_MODEL=... \\
    NVSH_AUG_CORRECTOR_URL=... NVSH_AUG_CORRECTOR_MODEL=... \\
    NVSH_AUG_REVIEWER_A_URL=... NVSH_AUG_REVIEWER_A_MODEL=... \\
    NVSH_AUG_REVIEWER_B_URL=... NVSH_AUG_REVIEWER_B_MODEL=... \\
        python scripts/lfm-finetune/augment.py out/train.json \\
            --per-source 20 --workers 4 \\
            --accepted-out accepted.jsonl --rejected-out rejected.jsonl

``--workers`` 2-4 is the recommended range for one shared gateway fronting
four separate model servers: it is enough to keep all four roles busy at
once, but more workers than that can overload a shared gateway (this is
exactly how the operator saw HTTP 503s and a backing model server restart
in a real run) -- raise it only once the gateway is confirmed to take the
extra load. Run with ``--dry-run`` first on a large seed set to see how
many variations would actually be attempted (seeds x ``--per-source`` minus
ids already written) before spending any model calls on it.

Accepted variations go to ``--accepted-out`` (one JSON object per line);
rejected ones go to ``--rejected-out`` with the source id, every role's
model id and each reviewer's verdict + reason, so a rejection is auditable
without re-running the pipeline. The run is resumable: a variation id
already present in either output file is skipped, and a variation that
exhausted its retries is never written to either file, so it is retried
again on the next resume. NVIDIA's own eval files are refused outright as
seeds -- see :func:`load_seeds`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.ops.table import get as get_operation  # noqa: E402
from nvsh.ops.table import names as operation_names  # noqa: E402

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

#: Replaces the old fixed 60s timeout on every role's HTTP call; a role
#: talking to a slower/busier backend can override it independently.
DEFAULT_TIMEOUT = 120.0

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
    #: Per-request timeout in seconds, from ``NVSH_AUG_<ROLE>_TIMEOUT``.
    timeout: float = DEFAULT_TIMEOUT


def load_role_config(role: str, env: dict[str, str] | None = None) -> RoleConfig:
    """Read
    ``NVSH_AUG_<role>_{URL,MODEL,KEY_ENV,MAX_TOKENS,DISABLE_THINKING,TIMEOUT}``
    from *env* (default ``os.environ``). Raises :class:`ConfigError` naming
    the exact variable that is missing."""
    source = os.environ if env is None else env
    url_var = f"NVSH_AUG_{role}_URL"
    model_var = f"NVSH_AUG_{role}_MODEL"
    key_env_var = f"NVSH_AUG_{role}_KEY_ENV"
    max_tokens_var = f"NVSH_AUG_{role}_MAX_TOKENS"
    disable_thinking_var = f"NVSH_AUG_{role}_DISABLE_THINKING"
    timeout_var = f"NVSH_AUG_{role}_TIMEOUT"

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

    timeout_raw = source.get(timeout_var)
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except ValueError:
            raise ConfigError(f"{timeout_var} must be a number, got {timeout_raw!r}")
    else:
        timeout = DEFAULT_TIMEOUT

    return RoleConfig(
        role=role,
        url=url,
        model=model,
        key=key,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
        timeout=timeout,
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
    #: For a skill seed from ``bodies.json``: an excerpt of the skill's
    #: SKILL.md, so requests can carry the specific details users give.
    context: str = ""
    #: For a skill seed: every skill identifier in the seed file. A request
    #: naming any of them ("I ran jetson-memory-audit") is rejected, since
    #: users describe tasks rather than name the router's tools.
    skill_names: tuple[str, ...] = ()


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
        context=str(record.get("body", "")),
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
        names = tuple(sorted({str(r["skill"]) for r in records}))
        return [replace(_seed_from_skill_record(r, side), skill_names=names) for r in records]

    resolved_side = _resolve_side(path, header, side)
    return [_seed_from_split_entry(r, resolved_side) for r in records]


def _expected_description(seed: Seed) -> str:
    if seed.seed_format == "skills":
        return (
            f"the {seed.expect['skill']!r} capability, and only that capability, handles this."
            f" That capability is described as: {seed.seed_text}"
        )
    return _answer_in_words(seed.expect)


def _answer_in_words(expect: dict[str, Any]) -> str:
    """The expected answer as a person would state it, not as a JSON block.

    Reviewers shown the raw block rejected every request that did not spell
    out the operation's identifier (a real user never says "power_set"), so
    the operation is named with the table's own description and arguments.
    """
    if expect.get("escalate"):
        return (
            "pass it on to a more capable assistant, because it needs investigation"
            " or changes beyond a small fixed set of machine actions"
        )
    if expect.get("explain"):
        return f"reply in words, along the lines of: {expect.get('answer', '')}"
    name = str(expect.get("operation"))
    operation = get_operation(name)
    what = operation.description if operation is not None else "the expected action"
    # Described by what it does, never by its identifier: a reviewer told that
    # users never name identifiers rejected responses that carried one, and a
    # generator shown one copied it into the request. A read-only operation
    # answers the request by being run; a mutating one is proposed for the
    # user to approve (a reviewer read "take this action" as a non-answer).
    what = what.rstrip(".").lower()
    values = ", ".join(
        f"{key} {str(value).replace('_', ' ')}"
        for key, value in sorted(expect.get("args", {}).items())
    )
    detail = f" -- {values}" if values else ""
    if operation is not None and operation.read_only:
        return f"run a read-only check and report what it shows: {what}{detail}"
    return f"propose this change for the user to approve: {what}{detail}"


#: Phrasing styles the generator rotates through, one per variation number,
#: so variations of one request differ in more than word order.
PHRASING_STYLES = (
    "a short imperative, as typed at a shell prompt",
    "a polite question",
    "a description of the symptom or situation, without saying what to do",
    "terse, a few words, like a note to self",
    "with technical jargon an experienced Jetson or Linux user would use",
    "casual and conversational",
    "as part of a longer sentence that gives a reason",
    "as a request from a teammate who is in a hurry",
)


#: Registers for a skill request written from a SKILL.md body: longer and
#: more specific than one written from the one-line description.
DETAILED_STYLES = (
    "as two or three sentences describing the user's situation and goal",
    "as a question from someone who has just hit a problem, with the error they see",
    "as a task request naming the board and software versions involved",
    "as a short paragraph from an engineer explaining what they need and why",
    "as one direct sentence with one concrete detail",
    "as a request from someone new to Jetson who describes what they want in plain words",
)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM_SPLIT = (
    "You rewrite user requests for a training dataset. You are given one "
    "request a user typed to a machine assistant. Rewrite it in different "
    "words as that user might have typed it. Do not change what is being "
    "asked for. Reply with only the rewritten request, nothing else."
)

GENERATOR_SYSTEM_SKILL = (
    "You write realistic user requests for a training dataset that teaches a "
    "router which capability should handle a request. You are given the "
    "description of one capability. Write one natural user request that this "
    "capability -- and only this capability -- would answer. Do not use the "
    "capability's identifier, or the identifier of any tool, skill or script; "
    "describing the task in ordinary words is fine. "
    "Reply with only the request, nothing else."
)

CORRECTOR_SYSTEM = (
    "You copyedit short user requests for a training dataset. Fix grammar and "
    "spelling only; keep the user's wording, tone and meaning, and do not "
    "change what the request is asking for. Reply with only the corrected "
    "request, nothing else."
)


def _capabilities() -> str:
    """What the small assistant can do, from the operation table, in words.

    Without it both reviewers judged escalations against a capable general
    assistant ("freeing disk space is a common task, no need to hand it
    off") and rejected every one. Built from the table, never hard-coded.
    """
    checks, changes = [], []
    for name in operation_names():
        operation = get_operation(name)
        if operation is None:
            continue
        (checks if operation.read_only else changes).append(operation.description.rstrip("."))
    return (
        "This assistant is small and can only do these things. Checks it can run "
        f"and report on: {'; '.join(checks)}. Changes it can propose for the user "
        f"to approve: {'; '.join(changes)}. It can also answer a general question "
        "in words. Anything else -- including anything that needs investigation, "
        "several steps, or a change not in that list -- must be passed on to a "
        "more capable assistant."
    )


REVIEWER_SYSTEM = (
    "You are a strict reviewer for a training dataset. You are given a user "
    "request and the response an assistant should give to it. Answer "
    "strictly 'yes' or 'no' to whether that response is exactly the right "
    "one for the request -- not a different operation, different arguments, "
    "or a different kind of response -- then a short reason. Users never "
    "name internal operations or their argument identifiers, and never ask "
    "for a hand-off in so many words: judge what the request needs. "
    + _capabilities()
    + " Asking the user to approve a change before making it is always the "
    "right way to carry out a request for one of the listed changes. Start "
    "your reply with the single word 'yes' or 'no'."
)

#: Skill seeds have no fixed answer text to compare against, only a capability
#: and its description, so their reviewers are asked the routing question.
REVIEWER_SYSTEM_SKILL = (
    "You are a strict reviewer for a dataset that teaches a router which "
    "capability should handle a user request. You are given one capability's "
    "description and a user request. Answer 'yes' only if this capability is "
    "the right one to handle the request and the request does not use the "
    "capability's identifier (its name as given, e.g. with hyphens or "
    "underscores); otherwise answer 'no'. Naming the technique or task in "
    'ordinary words ("speculative decoding", "headless mode") is how '
    "real users write and is fine. Start your reply with the single word "
    "'yes' or 'no', then a short reason."
)

REVIEWER_SYSTEM_CHANGE_CHECK = REVIEWER_SYSTEM + (
    " The expected response here is not one of the listed changes. If the "
    "request could reasonably be carried out by one of the changes listed "
    "above, answer 'no' even if it otherwise matches (h30)."
)


def generator_prompt(seed: Seed, variation: int = 0) -> tuple[str, str]:
    """Return ``(system, user)`` for the generator role.

    *variation* picks the phrasing style (split seeds), so the Nth variation
    of a request is asked for in a different register from the (N+1)th.
    """
    if seed.seed_format == "skills":
        if seed.context:
            style = DETAILED_STYLES[variation % len(DETAILED_STYLES)]
            user = (
                f"Capability description: {seed.seed_text}\n\n"
                f"Capability documentation (excerpt):\n{seed.context}\n\n"
                "Write one user request this capability answers, written "
                f"{style}. Include the specific details a real user would give"
                " (their board or device, versions, what they tried, what they"
                " see), drawn from situations the documentation covers. Do not"
                " copy sentences from the documentation."
            )
            return GENERATOR_SYSTEM_SKILL, user
        user = (
            f"Capability description: {seed.seed_text}\n\n"
            "Write one user request this capability answers."
        )
        return GENERATOR_SYSTEM_SKILL, user
    style = PHRASING_STYLES[variation % len(PHRASING_STYLES)]
    # The generator never sees the expected answer: shown it, it copied its
    # wording ("Propose this action: ...") into the request. The original
    # request already carries the meaning to keep.
    user = (
        f"Original request: {seed.seed_text}\n\n"
        "Rewrite the request above in different words, keeping exactly the same meaning"
        " and asking for exactly the same thing, no more and no less."
        f" Write it {style}. Do not name internal operations or their identifiers."
    )
    return GENERATOR_SYSTEM_SPLIT, user


def corrector_prompt(seed: Seed, text: str) -> tuple[str, str]:
    # Like the generator, the corrector never sees the expected answer: shown
    # it, it pasted the answer's wording into the request.
    return CORRECTOR_SYSTEM, f"Request to copyedit:\n{text}"


def reviewer_prompt(seed: Seed, text: str) -> tuple[str, str]:
    if seed.seed_format == "skills":
        user = (
            f"Capability: {seed.expect['skill']}\n"
            f"Description: {seed.seed_text}\n\n"
            f"User request: {text}\n\n"
            "Is this capability the right one to handle this request? Answer 'yes' "
            "or 'no' and then a short reason."
        )
        return REVIEWER_SYSTEM_SKILL, user
    system = REVIEWER_SYSTEM_CHANGE_CHECK if seed.needs_change_check else REVIEWER_SYSTEM
    user = (
        f"Response the assistant should give: {_expected_description(seed)}\n\n"
        f"User request: {text}\n\n"
        "Is that response exactly the right one for this request? Answer 'yes' "
        "or 'no' and then a short reason."
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


def _post_chat_completion(
    role: RoleConfig, system: str, user: str, timeout: float | None = None
) -> str:
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
    effective_timeout = role.timeout if timeout is None else timeout
    with urllib.request.urlopen(request, timeout=effective_timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    # An empty or malformed `choices` (a gateway returning a truncated/error
    # body with a 200 status) is treated the same as an empty reply from a
    # role -- an error the caller counts, never an IndexError/KeyError that
    # crashes the run (finding 2).
    choices = body.get("choices")
    if not choices or "message" not in choices[0]:
        raise ValueError(f"malformed reply from {role.model!r}: no choices in response")
    return _extract_content(choices[0]["message"])


#: The call signature every role invocation uses; tests may inject a fake
#: (e.g. to skip real HTTP) but the default is the real client above.
RoleCaller = Callable[[RoleConfig, str, str], str]


def default_caller(role: RoleConfig, system: str, user: str) -> str:
    return _post_chat_completion(role, system, user)


# ---------------------------------------------------------------------------
# retry with exponential backoff + jitter (transient HTTP/connection/timeout
# failures only -- a real workforce-scale run cannot afford to lose a
# variation to one flaky 503 from a shared gateway)
# ---------------------------------------------------------------------------

#: HTTP statuses worth retrying. A non-transient 4xx (400/401/403/404, or any
#: other status not listed here) is never retried -- it means the request
#: itself is wrong, and retrying it would just repeat the same failure.
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

DEFAULT_MAX_RETRIES = 6
DEFAULT_BACKOFF_BASE = 2.0
#: Cap on any single backoff wait, regardless of attempt count or Retry-After.
MAX_BACKOFF_WAIT = 60.0


def _is_transient(exc: BaseException) -> tuple[bool, float | None]:
    """Classify *exc* as transient or not, and pull a ``Retry-After`` seconds
    value out of it when present. A connection error or a read timeout is
    always transient (there is no status code to check); an HTTP response is
    transient only for :data:`_TRANSIENT_STATUS`.
    """
    if isinstance(exc, urllib.error.HTTPError):
        retry_after: float | None = None
        headers = exc.headers
        header_value = headers.get("Retry-After") if headers is not None else None
        if header_value:
            try:
                retry_after = float(header_value)
            except ValueError:
                retry_after = None
        return exc.code in _TRANSIENT_STATUS, retry_after
    if isinstance(exc, TimeoutError):
        return True, None
    if isinstance(exc, urllib.error.URLError):
        # HTTPError is a URLError subclass and is handled above; anything
        # else here is a connection-level failure (refused, DNS, reset...).
        return True, None
    if isinstance(exc, (OSError, http.client.HTTPException)):
        # On Python 3.12, urllib only wraps errors raised by h.request() into
        # URLError -- a connection dropped while reading the response (in
        # h.getresponse()/response.read()) surfaces raw as one of these:
        # ConnectionResetError, http.client.RemoteDisconnected (itself a
        # ConnectionResetError, i.e. an OSError) or http.client.IncompleteRead
        # (an HTTPException). There is no status code or Retry-After here.
        return True, None
    return False, None


def _compute_backoff(
    attempt: int,
    backoff_base: float,
    retry_after: float | None,
    rand_fn: Callable[[], float] = random.random,
) -> float:
    """Wait time before retry number *attempt* (1-based). A server-supplied
    ``Retry-After`` is honoured, bypassing backoff/jitter, but still capped at
    :data:`MAX_BACKOFF_WAIT` -- a misbehaving/hostile gateway must never be
    able to stall a run for longer than that regardless of what it asks for.
    Otherwise: full jitter over an exponential curve, capped at
    :data:`MAX_BACKOFF_WAIT` before the jitter is applied."""
    if retry_after is not None:
        return max(0.0, min(retry_after, MAX_BACKOFF_WAIT))
    raw = backoff_base * (2 ** (attempt - 1))
    capped = min(raw, MAX_BACKOFF_WAIT)
    return capped * rand_fn()


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    #: Injectable so tests never actually sleep.
    sleep_fn: Callable[[float], None] = time.sleep
    rand_fn: Callable[[], float] = random.random


def _call_with_retry(
    caller: RoleCaller,
    role: RoleConfig,
    system: str,
    user: str,
    policy: RetryPolicy,
    on_retry: Callable[[], None] | None = None,
) -> str:
    """Call *caller* once, retrying a transient failure up to
    ``policy.max_retries`` times with backoff. Raises the last exception once
    retries are exhausted, or immediately for a non-transient failure."""
    attempt = 0
    while True:
        try:
            return caller(role, system, user)
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            transient, retry_after = _is_transient(exc)
            if not transient or attempt >= policy.max_retries:
                raise
            attempt += 1
            if on_retry is not None:
                on_retry()
            wait = _compute_backoff(attempt, policy.backoff_base, retry_after, policy.rand_fn)
            policy.sleep_fn(wait)


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
#: A "no" that reads as a verdict: at the start of a line or sentence, or
#: right after a slash or colon ("yes/no: no", "yes? No, it changes ...").
#: A "no" inside a clause ("with no machine changes involved") is not one --
#: counting it rejected clear yeses in a real run.
_NO_RE = re.compile(r"(?:^\s*|[\n.?!:;/]\s*)no\b", re.IGNORECASE)
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
    """Call one reviewer and parse its verdict.

    An empty reply (a reasoning model that spent its whole budget thinking)
    is not a judgement, so it is neither an accept nor a reject: it raises,
    the variation counts as an error, and the next resume tries it again.
    In a real run a third of all rejections were empty replies recorded as
    rejects, which threw those variations away for good.
    """
    text = caller(role, system, user)
    if not text.strip():
        raise ValueError(f"empty reply from reviewer {role.model!r}")
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
    retries: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "corrected": self.corrected,
            "accepted": self.accepted,
            "rejected_by_a": self.rejected_by_a,
            "rejected_by_b": self.rejected_by_b,
            "errors": self.errors,
            "retries": self.retries,
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


#: Wording that only appears in the expected-answer descriptions shown to the
#: corrector and reviewers. A request that carries it was copied from the
#: answer, not written as a user would, and is rejected whatever the reviewers say.
_ANSWER_TEMPLATE_RE = re.compile(
    r"take this action|propose this (?:action|change)|read-only check|report what it shows"
    r"|user to approve|more capable assistant|full agent"
    r"|hand (?:the|this) request|reply in words|\b\w+ = \S+",
    re.IGNORECASE,
)


def copies_answer_template(text: str) -> str:
    """The answer-template wording *text* copies, or "" if none."""
    match = _ANSWER_TEMPLATE_RE.search(text)
    return match.group(0) if match else ""


def names_internal_operation(text: str) -> str:
    """The first operation-table identifier *text* names, or "" if none.

    Checked on every variation after the reviewers: they let "What is the
    operation 'memory_stats' (...)" through once in a real run.
    """
    lowered = text.lower()
    for name in operation_names():
        if re.search(rf"(?<![a-z0-9_]){re.escape(name.lower())}(?![a-z0-9_])", lowered):
            return name
    return ""


def names_skill_identifier(text: str, names: tuple[str, ...]) -> str:
    """The first skill identifier in *names* that *text* names (hyphen or
    underscore form), or "" if none."""
    lowered = text.lower()
    for name in names:
        for form in {name.lower(), name.lower().replace("-", "_")}:
            if re.search(rf"(?<![a-z0-9_-]){re.escape(form)}(?![a-z0-9_-])", lowered):
                return name
    return ""


def _variation_number(variation_id: str) -> int:
    """The N in ``<source_id>~vN``; 0 when the id has no such suffix."""
    _, _, tail = variation_id.rpartition("~v")
    return int(tail) if tail.isdigit() else 0


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
    gen_system, gen_user = generator_prompt(seed, _variation_number(variation_id))
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

    leaked = names_internal_operation(corrected_text) or names_skill_identifier(
        corrected_text, seed.skill_names
    )
    if leaked:
        # Deterministic, whatever the reviewers said: a request that names an
        # internal operation teaches the model that users talk in identifiers.
        verdicts["identifier_check"] = {"accept": False, "reason": f"names {leaked!r}"}
    copied = "" if leaked else copies_answer_template(corrected_text)
    if copied:
        verdicts["template_check"] = {"accept": False, "reason": f"copies {copied!r}"}
        leaked = copied
    accepted = accept_a and accept_b and not leaked
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


@dataclass
class RereviewCounts:
    """Counts for ``--rereview`` (t11, decisions c38/c41): only REVIEWER_B is
    ever called; ``compared``/``agreed`` track how the fresh verdict lines up
    with the stored one, which is what the non-thinking pilot (c41) reports."""

    processed: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    compared: int = 0
    agreed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "processed": self.processed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "errors": self.errors,
            "compared": self.compared,
            "agreed": self.agreed,
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_rereview_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    """Load stored accepted+rejected records from *paths* (the ``--accepted-out``
    / ``--rejected-out`` of a prior run), in file order. Each record is
    returned exactly as stored -- validated only once a re-review is actually
    attempted on it (see :func:`_require_stored_reviewer_a`)."""
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_jsonl(path))
    return records


def _require_stored_reviewer_a(record: dict[str, Any]) -> tuple[bool, str]:
    """The stored reviewer A verdict a re-review re-derives acceptance from.
    Never re-asked -- only REVIEWER_B is called during a re-review.

    ``augment.py`` only ever writes ``verdicts`` onto a *rejected* record
    (see :func:`_process_variation`); a record stored as accepted carries no
    ``verdicts`` at all, because being stored as accepted already means both
    reviewers said yes and every deterministic guard passed. So a record
    with no ``verdicts`` key is treated as an implicit reviewer A accept, and
    this only raises when ``verdicts`` is present but missing reviewer A --
    a genuinely malformed record, not an accepted one.
    """
    verdicts = record.get("verdicts")
    if verdicts is None:
        return True, ""
    reviewer_a = verdicts.get("reviewer_a") if isinstance(verdicts, dict) else None
    if not isinstance(reviewer_a, dict) or "accept" not in reviewer_a:
        raise ValueError(
            f"{record.get('id', '?')}: no stored reviewer_a verdict to re-review against"
        )
    return bool(reviewer_a["accept"]), str(reviewer_a.get("reason", ""))


def _stored_reviewer_b_accept(record: dict[str, Any]) -> bool | None:
    """The old reviewer B verdict, for the pilot's agreement report.

    A record with no ``verdicts`` at all was stored as accepted, which means
    the old reviewer B also said yes (see :func:`_require_stored_reviewer_a`).
    ``None`` only when ``verdicts`` is present but carries no reviewer B
    entry -- agreement reporting is then best-effort, unlike the required
    reviewer A verdict above.
    """
    verdicts = record.get("verdicts")
    if verdicts is None:
        return True
    reviewer_b = verdicts.get("reviewer_b") if isinstance(verdicts, dict) else None
    if not isinstance(reviewer_b, dict) or "accept" not in reviewer_b:
        return None
    return bool(reviewer_b["accept"])


def _rereview_guard_verdicts(text: str, seed: Seed) -> dict[str, dict[str, Any]]:
    """Re-run the same deterministic guards :func:`_process_variation` applies
    after the reviewers -- reused, not copied -- on the stored *text*: an
    internal-operation/skill-identifier leak or a copied answer-template
    phrase must keep a record rejected however the reviewers voted, exactly
    as in a fresh run. Never a switch on a specific operation name; both
    checks are the table-driven functions a fresh run already uses."""
    guard_verdicts: dict[str, dict[str, Any]] = {}
    leaked = names_internal_operation(text) or names_skill_identifier(text, seed.skill_names)
    if leaked:
        guard_verdicts["identifier_check"] = {"accept": False, "reason": f"names {leaked!r}"}
    copied = "" if leaked else copies_answer_template(text)
    if copied:
        guard_verdicts["template_check"] = {"accept": False, "reason": f"copies {copied!r}"}
    return guard_verdicts


def _seed_from_stored_record(record: dict[str, Any]) -> Seed:
    """Rebuild enough of a :class:`Seed` from a stored accepted/rejected
    record to build :func:`reviewer_prompt` again. The generator and
    corrector are never re-run -- the record's own ``text`` (already
    generated and corrected) is reused as the request under review."""
    expect = record["expect"]
    seed_format = record.get("seed_format", "split")
    corpus_fields = {key: record[key] for key in ("kind", "source", "class") if key in record}
    needs_change_check = seed_format != "skills" and _needs_change_check(expect)
    return Seed(
        source_id=str(record.get("source_id", record.get("id", ""))),
        seed_format=seed_format,
        side=record.get("side"),
        seed_text=record.get("text", ""),
        expect=expect,
        needs_change_check=needs_change_check,
        corpus_fields=corpus_fields,
    )


def _process_rereview_candidate(
    record: dict[str, Any], role: RoleConfig, caller: RoleCaller
) -> dict[str, Any]:
    """Re-review one stored candidate: call only REVIEWER_B on the record's
    already-generated/corrected ``text``, then re-derive acceptance from the
    stored reviewer A verdict plus this fresh reviewer B verdict."""
    accept_a, reason_a = _require_stored_reviewer_a(record)
    old_accept_b = _stored_reviewer_b_accept(record)

    seed = _seed_from_stored_record(record)
    system, user = reviewer_prompt(seed, record["text"])
    accept_b, reason_b = _reviewer_verdict(role, system, user, caller)

    guard_verdicts = _rereview_guard_verdicts(record["text"], seed)
    accepted = accept_a and accept_b and not guard_verdicts
    models = dict(record.get("models", {}))
    models[role.role] = role.model
    new_record = dict(record)
    new_record["models"] = models
    new_record["verdicts"] = {
        "reviewer_a": {"accept": accept_a, "reason": reason_a},
        "reviewer_b": {"accept": accept_b, "reason": reason_b},
        **guard_verdicts,
    }
    return {
        "accepted": accepted,
        "record": new_record,
        "old_accept_b": old_accept_b,
        "new_accept_b": accept_b,
    }


def run_rereview(
    candidate_files: list[Path],
    role: RoleConfig,
    accepted_out: Path,
    rejected_out: Path,
    limit: int | None = None,
    caller: RoleCaller = default_caller,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    sleep_fn: Callable[[float], None] = time.sleep,
    rand_fn: Callable[[], float] = random.random,
    workers: int = 1,
) -> RereviewCounts:
    """Re-review stored accepted+rejected candidates with REVIEWER_B only
    (decisions c38/c41). Resumable like :func:`run_pipeline`: an id already
    present in *accepted_out* or *rejected_out* is skipped. *limit* caps how
    many candidates are attempted, which is what the non-thinking pilot
    (about 150 candidates, c41) uses.

    *workers* processes candidates through a bounded thread pool, exactly
    like :func:`run_pipeline` (deviation d4: the thinking-mode reviewer-B
    re-review takes about 44s per candidate, which is too slow to run one
    at a time at pilot scale). ``main()`` passes the existing ``--workers``
    value here; the default of 1 here keeps a direct call serial, matching
    the pre-concurrency behaviour, for a caller that never passes it."""
    candidates = load_rereview_candidates(candidate_files)
    done = _existing_ids(accepted_out) | _existing_ids(rejected_out)
    tasks = [record for record in candidates if record.get("id") not in done]
    if limit is not None:
        tasks = tasks[:limit]

    retry_policy = RetryPolicy(
        max_retries=max_retries, backoff_base=backoff_base, sleep_fn=sleep_fn, rand_fn=rand_fn
    )
    counts = RereviewCounts()
    if not tasks:
        return counts

    lock = threading.Lock()

    def caller_with_retry(role_config: RoleConfig, system: str, user: str) -> str:
        return _call_with_retry(caller, role_config, system, user, retry_policy)

    def _run_one(record: dict[str, Any]) -> None:
        record_id = record.get("id", "?")
        try:
            outcome = _process_rereview_candidate(record, role, caller_with_retry)
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            KeyError,
            ValueError,
        ) as exc:
            with lock:
                counts.errors += 1
            print(f"error: {record_id}: {exc}", file=sys.stderr)
            return

        with lock:
            if record_id in done:  # pragma: no cover - tasks already dedupes against done
                return
            counts.processed += 1
            # One record, one write, one line -- the lock is held across the
            # whole append (and the accept/reject/agreement bookkeeping) so a
            # record is never interleaved with another thread's write, and
            # `done` is updated in the same critical section so no id can
            # ever be written twice, exactly as run_pipeline's _run_one does.
            if outcome["accepted"]:
                counts.accepted += 1
                _append_jsonl(accepted_out, outcome["record"])
            else:
                counts.rejected += 1
                _append_jsonl(rejected_out, outcome["record"])
            done.add(record_id)

            old_accept_b = outcome["old_accept_b"]
            if old_accept_b is not None:
                counts.compared += 1
                if old_accept_b == outcome["new_accept_b"]:
                    counts.agreed += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_run_one, record) for record in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    return counts


def _print_rereview_summary(counts: RereviewCounts, out: Any = None) -> None:
    """One summary line: processed/accepted/rejected/errors, plus agreement
    with the stored reviewer B verdicts (the pilot's own acceptance-rate
    check, decision c41) as ``agreement=<agreed>/<compared>``."""
    stream = out if out is not None else sys.stdout
    print(
        f"rereview: processed={counts.processed} accepted={counts.accepted} "
        f"rejected={counts.rejected} errors={counts.errors} "
        f"agreement={counts.agreed}/{counts.compared}",
        file=stream,
    )


def _plan_tasks(
    seeds: list[Seed],
    per_source: int,
    limit: int | None,
    done: set[str],
) -> list[tuple[Seed, str]]:
    """Build the ordered ``(seed, variation_id)`` work list: skip any id
    already in *done*, reserve every id it does decide to attempt (so two
    seeds sharing a ``source_id`` -- already proven consistent by
    :func:`_validate_seed_consistency` -- never both get a task for the same
    variation id), and stop once *limit* total tasks have been planned.
    *done* itself is never mutated here.
    """
    reserved = set(done)
    tasks: list[tuple[Seed, str]] = []
    # Round-robin: variation 1 of every seed, then variation 2, ... so a run
    # stopped part-way (or a --limit) leaves every seed equally covered, and
    # raising --per-source later only adds further rounds.
    for n in range(1, per_source + 1):
        for seed in seeds:
            if limit is not None and len(tasks) >= limit:
                return tasks
            variation_id = f"{seed.source_id}~v{n}"
            if variation_id in reserved:
                continue
            reserved.add(variation_id)
            tasks.append((seed, variation_id))
    return tasks


def _print_progress(
    done: int, total: int, counts: PipelineCounts, elapsed: float, out: Any
) -> None:
    """One progress line: done/total attempted, accepted, rejected, errors,
    retries so far, rate per minute, and an ETA. ``rejected`` is derived
    (every attempted, non-error variation is exactly accepted or rejected)
    rather than tracked separately."""
    rejected = done - counts.accepted - counts.errors
    rate = done / (elapsed / 60) if elapsed > 0 else 0.0
    remaining = total - done
    eta = f"{remaining / rate:.1f}m" if rate > 0 else "?"
    print(
        f"progress: {done}/{total} attempted accepted={counts.accepted} "
        f"rejected={rejected} errors={counts.errors} retries={counts.retries} "
        f"rate={rate:.1f}/min eta={eta}",
        file=out,
    )


def run_pipeline(
    seed_files: list[Path],
    roles: dict[str, RoleConfig],
    accepted_out: Path,
    rejected_out: Path,
    per_source: int,
    side: str | None = None,
    limit: int | None = None,
    caller: RoleCaller = default_caller,
    workers: int = 2,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    sleep_fn: Callable[[float], None] = time.sleep,
    rand_fn: Callable[[], float] = random.random,
    progress_every: float = 60.0,
    now_fn: Callable[[], float] = time.monotonic,
    progress_out: Any = None,
) -> PipelineCounts:
    seeds: list[Seed] = []
    for seed_file in seed_files:
        seeds.extend(load_seeds(seed_file, side))
    _validate_seed_consistency(seeds)

    done = _existing_ids(accepted_out) | _existing_ids(rejected_out)
    tasks = _plan_tasks(seeds, per_source, limit, done)

    counts = PipelineCounts()
    if not tasks:
        return counts

    retry_policy = RetryPolicy(
        max_retries=max_retries, backoff_base=backoff_base, sleep_fn=sleep_fn, rand_fn=rand_fn
    )
    lock = threading.Lock()
    out = progress_out if progress_out is not None else sys.stderr
    total = len(tasks)
    start = now_fn()
    last_print = start
    completed = 0

    def _run_one(seed: Seed, variation_id: str) -> None:
        # Thread-local: never touched by any other task, so no lock is
        # needed while accumulating it -- only merging it into the shared
        # `counts` below needs the lock.
        local_counts = PipelineCounts()
        local_retries = [0]

        def on_retry() -> None:
            local_retries[0] += 1

        def caller_with_retry(role: RoleConfig, system: str, user: str) -> str:
            return _call_with_retry(caller, role, system, user, retry_policy, on_retry)

        try:
            outcome = _process_variation(seed, variation_id, roles, local_counts, caller_with_retry)
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            KeyError,
            ValueError,
        ) as exc:
            with lock:
                counts.generated += local_counts.generated
                counts.corrected += local_counts.corrected
                counts.retries += local_retries[0]
                counts.errors += 1
            print(f"error: {variation_id}: {exc}", file=sys.stderr)
            return

        with lock:
            counts.generated += local_counts.generated
            counts.corrected += local_counts.corrected
            counts.accepted += local_counts.accepted
            counts.rejected_by_a += local_counts.rejected_by_a
            counts.rejected_by_b += local_counts.rejected_by_b
            counts.retries += local_retries[0]
            if variation_id in done:  # pragma: no cover - _plan_tasks already dedupes
                return
            # One record, one write, one line -- the lock is held across the
            # whole append so a record is never interleaved with another
            # thread's write, and `done` is updated in the same critical
            # section so no id can ever be written twice.
            if outcome["accepted"]:
                _append_jsonl(accepted_out, outcome["record"])
            else:
                _append_jsonl(rejected_out, outcome["record"])
            done.add(variation_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_run_one, seed, variation_id) for seed, variation_id in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            now = now_fn()
            if progress_every > 0 and (now - last_print) >= progress_every:
                with lock:
                    _print_progress(completed, total, counts, now - start, out)
                last_print = now

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
    parser.add_argument(
        "--limit",
        "--sample",
        dest="limit",
        type=int,
        default=None,
        help=(
            "cap total variations attempted (dry runs), or candidates re-reviewed with "
            "--rereview -- --sample is the same option, named for the non-thinking pilot (c41)"
        ),
    )
    parser.add_argument(
        "--rereview",
        action="store_true",
        help=(
            "re-review stored accepted+rejected candidates (seed_files) with REVIEWER_B only, "
            "reusing their generator/corrector text (decisions c38/c41)"
        ),
    )
    parser.add_argument("--accepted-out", default="accepted.jsonl")
    parser.add_argument("--rejected-out", default="rejected.jsonl")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "thread-pool size for parallel variation processing (default 2; 2-4 is "
            "recommended -- more can overload a shared gateway)"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="retries for a transient failure (429/500/502/503/504, connection, timeout)",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=DEFAULT_BACKOFF_BASE,
        dest="backoff_base",
        help="base seconds for exponential backoff between retries (capped at 60s per wait)",
    )
    parser.add_argument(
        "--progress-every",
        type=float,
        default=60.0,
        help="seconds between progress lines on stderr (0 disables progress lines)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many variations would be attempted and exit without calling any endpoint",
    )
    args = parser.parse_args(argv)

    seed_paths = [Path(p) for p in args.seed_files]

    if args.rereview:
        try:
            reviewer_b = load_role_config("REVIEWER_B")
        except ConfigError as exc:
            parser.error(str(exc))
            return 2  # pragma: no cover - parser.error already exits
        if args.dry_run:
            # Codex review finding #7: dry-run must be handled before any
            # dispatch to the reviewer -- report the count, the limit and
            # the reviewer B model, and exit without calling anything or
            # writing accepted/rejected output.
            try:
                candidates = load_rereview_candidates(seed_paths)
            except (ConfigError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            done = _existing_ids(Path(args.accepted_out)) | _existing_ids(Path(args.rejected_out))
            tasks = [record for record in candidates if record.get("id") not in done]
            if args.limit is not None:
                tasks = tasks[: args.limit]
            print(
                f"dry-run: {len(tasks)} candidate(s) would be re-reviewed with REVIEWER_B "
                f"({reviewer_b.model}); no endpoint was called"
            )
            return 0
        try:
            counts = run_rereview(
                candidate_files=seed_paths,
                role=reviewer_b,
                accepted_out=Path(args.accepted_out),
                rejected_out=Path(args.rejected_out),
                limit=args.limit,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
                workers=args.workers,
            )
        except (ConfigError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_rereview_summary(counts)
        return 0

    if args.dry_run:
        try:
            seeds: list[Seed] = []
            for seed_file in seed_paths:
                seeds.extend(load_seeds(seed_file, args.side))
            _validate_seed_consistency(seeds)
            done = _existing_ids(Path(args.accepted_out)) | _existing_ids(Path(args.rejected_out))
            tasks = _plan_tasks(seeds, args.per_source, args.limit, done)
        except (SeedRefused, ConfigError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"dry-run: {len(tasks)} variation(s) would be attempted; no endpoint was called")
        return 0

    try:
        roles = load_all_roles()
    except ConfigError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error already exits

    try:
        counts = run_pipeline(
            seed_files=seed_paths,
            roles=roles,
            accepted_out=Path(args.accepted_out),
            rejected_out=Path(args.rejected_out),
            per_source=args.per_source,
            side=args.side,
            limit=args.limit,
            workers=args.workers,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            progress_every=args.progress_every,
        )
    except (SeedRefused, ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for key, value in counts.as_dict().items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
