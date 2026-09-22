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
import json
import os
import random
import re
import sys
import threading
import time
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
        return (
            f"the {seed.expect['skill']!r} capability, and only that capability, handles this."
            f" That capability is described as: {seed.seed_text}"
        )
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

#: Skill seeds have no fixed answer text to compare against, only a capability
#: and its description, so their reviewers are asked the routing question.
REVIEWER_SYSTEM_SKILL = (
    "You are a strict reviewer for a dataset that teaches a router which "
    "capability should handle a user request. You are given one capability's "
    "description and a user request. Answer 'yes' only if this capability is "
    "the right one to handle the request and the request does not name the "
    "capability outright; otherwise answer 'no'. Start your reply with the "
    "single word 'yes' or 'no', then a short reason."
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
    return _extract_content(body["choices"][0]["message"])


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
    return False, None


def _compute_backoff(
    attempt: int,
    backoff_base: float,
    retry_after: float | None,
    rand_fn: Callable[[], float] = random.random,
) -> float:
    """Wait time before retry number *attempt* (1-based). A server-supplied
    ``Retry-After`` is honoured exactly, bypassing backoff/jitter entirely.
    Otherwise: full jitter over an exponential curve, capped at
    :data:`MAX_BACKOFF_WAIT` before the jitter is applied."""
    if retry_after is not None:
        return max(0.0, retry_after)
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
        except (urllib.error.URLError, TimeoutError) as exc:
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
    for seed in seeds:
        for n in range(1, per_source + 1):
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
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
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
    parser.add_argument("--limit", type=int, default=None, help="cap total variations (dry runs)")
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
