#!/usr/bin/env python3
"""Build a chat-format fine-tune file for Tier 2 (LFM2.5) from the nvsh benchmark corpus.

A development-machine tool for ``docs/lfm-finetune.md``. It is NEVER imported by
the nvsh package -- nothing under nvsh/ may depend on it.

Usage::

    python scripts/lfm-finetune/build_dataset.py --out train.jsonl [--corpus PATH]
    python scripts/lfm-finetune/build_dataset.py --out train.jsonl --split train.json

Each output line is ``{"messages": [...], "tools": [...], "source_id": ...}``
in the shape the Hugging Face chat templates, TRL's ``SFTTrainer`` and
unsloth accept (an extra ``source_id`` key travels along for later grouping;
none of those consumers look at unknown keys). The tools and the system
brief are the ones Tier 2 really sends (``nvsh.tiers.lfm``), so a model
tuned on this file sees at run time exactly what it saw in training.

Only single-turn examples can be built from the corpus: an explicit request
answered by ``propose``, a should-decline request answered by ``escalate``,
and a read-only question answered in words by ``explain`` (the corpus
entry's own ``answer`` text -- never generated here). Multi-round inspection
examples need real Tier 2 records; they are not invented here.

``--split PATH`` reads a train/val/test side written by
``scripts/lfm-finetune/split.py`` instead of a raw corpus (``--corpus`` and
``--split`` are mutually exclusive); each output example carries the split
entry's ``source_id`` so later steps can group by source. The held-out split
is refused either way.

Before writing output, ``build()`` renders one example per outcome (propose,
escalate, explain -- whichever are present) with the base tokenizer's own
chat template and checks it round-trips back to the same tool call
(``verify_round_trip``, ``--base``/``--revision``, default the LFM2.5 base
``train.py`` itself trains): a base whose template cannot reproduce a
record this file wrote must never ship it silently (Codex review finding
#6, issue 46). This needs the training stack's tokenizer cached locally
(``local_files_only`` -- this script is offline-first, never a silent
network fetch); pass ``--no-verify-render`` in an environment without it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.platform._model import Platform  # noqa: E402
from nvsh.tiers import lfm  # noqa: E402
from nvsh.tiers.bench import (  # noqa: E402
    CorpusEntry,
    context_for,
    load_corpus,
    load_world,
    request_for,
    world_platform,
)

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


scorer = _sibling("scorer")

#: The 8 decline classes (issue 53 t14), read from the corpus's own `class`
#: field (``"decline:<reason>"``) and offered, in ``--reasons`` mode, as
#: distinct escalate-reason candidates instead of the bare ``escalate``.
#: Order is fixed so the default (non-randomized) letter map is stable.
REASON_CANDIDATES: tuple[str, ...] = (
    "escalate:outside_table",
    "escalate:repair",
    "escalate:diagnosis",
    "escalate:missing_argument",
    "escalate:not_a_request",
    "escalate:multi_step",
    "escalate:injection",
    "escalate:over_time",
)

#: An escalate entry with no or unknown `class` rolls up to this reason.
DEFAULT_REASON = "escalate:outside_table"

#: Scripts-side (never nvsh/) description overrides for the reason candidates.
REASON_DESCRIPTIONS: dict[str, str] = json.loads(
    (_HERE / "data" / "reasons.json").read_text(encoding="utf-8")
)

#: How many candidates a randomized offer keeps at minimum, and the chance
#: it keeps the full pool instead (--randomize-labels only).
DEFAULT_MIN_SUBSET = 6
DEFAULT_FULL_SET_PROBABILITY = 0.3

#: The request and context builders are ``nvsh.tiers.bench``'s own, and the
#: user message is ``nvsh.tiers.lfm.request_message``: a FAILURE entry is
#: trained on exactly the text Tier 2 sends for it at run time.

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The split a tuned model is judged on. Training on it would make that judgement worthless.
HELD_OUT_NAME = "held-out.json"

#: ``split.py``'s own note, written verbatim into a split file's header by
#: ``_write_side``: ``f"Split '{name}' of {corpus_name} (seed={seed})."``.
#: Matched here to learn which side a ``--split`` file claims to be, without
#: importing split.py (this script and split.py are deliberately independent
#: of each other -- see split.py's module docstring).
_SPLIT_SIDE_RE = re.compile(r"Split '(\w+)' of ")

#: nvsh/tiers/corpus/held-out.json's own header opens with this exact
#: phrase; matched so a held-out file is refused even if renamed away from
#: ``held-out.json``.
_HELD_OUT_HEADER_MARKER = "Held-out split"

#: The only side a ``--split`` file may train from (decision c31: iterate on
#: val, test only measured on final runs -- and both val and test exist to
#: be held out of training so a measured run's fold stays clean, honesty h7).
TRAIN_SIDE = "train"

#: What the assistant says when it hands a request up. Short on purpose: the
#: reason is for the audit log, the behaviour being taught is the call itself.
ESCALATE_REASON = "this needs the full agent"

#: ``--arguments-as`` values (finding 4054701434): whether a training
#: example's ``tool_calls[].function.arguments`` is a JSON object (what
#: Hugging Face's ``tokenizer.apply_chat_template`` documents, and what this
#: file is consumed by) or a JSON string (the OpenAI wire format Tier 2
#: itself replays in its own chat history -- see ``nvsh.tiers.lfm``'s
#: ``_record``, whose ``arguments`` field is already ``json.dumps``'d).
#: See ``docs/lfm-finetune.md`` for which one to pick and why.
ARGUMENTS_AS_OBJECT = "object"
ARGUMENTS_AS_STRING = "string"
ARGUMENTS_AS_CHOICES = (ARGUMENTS_AS_OBJECT, ARGUMENTS_AS_STRING)

#: The base ``verify_round_trip`` checks a real build against by default --
#: kept in sync with ``train.py``'s own ``DEFAULT_BASE``/``DEFAULT_REVISION``
#: (this file's own primary consumer is Tier 2/LFM2.5; pass ``--base``/
#: ``--revision`` to check against another base, such as Qwen3.5, instead).
DEFAULT_BASE = "LiquidAI/LFM2.5-350M"
DEFAULT_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"

#: Qwen3.5's XML function/parameter tool-call form:
#: ``<tool_call><function=NAME><parameter=ARG>VALUE</parameter>...``.
_QWEN_FUNCTION_RE = re.compile(r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>", re.DOTALL)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>]+)>\n(?P<value>.*?)\n</parameter>", re.DOTALL
)

#: LFM2.5's Pythonic call form: ``<|tool_call_start|>[name(kw=val, ...)]<|tool_call_end|>``.
_PYTHONIC_CALL_RE = re.compile(r"<\|tool_call_start\|>(?P<body>.*?)<\|tool_call_end\|>", re.DOTALL)


def _parse_qwen_call(rendered: str) -> tuple[str, dict]:
    """Read Qwen3.5's XML function/parameter tool-call form back to (name, arguments)."""
    match = _QWEN_FUNCTION_RE.search(rendered)
    if not match:
        raise ValueError(f"no <function=...> block in rendered text: {rendered!r}")
    arguments: dict = {}
    for param in _QWEN_PARAMETER_RE.finditer(match.group("body")):
        raw = param.group("value")
        try:
            arguments[param.group("name")] = json.loads(raw)
        except json.JSONDecodeError:
            arguments[param.group("name")] = raw
    return match.group("name"), arguments


def _parse_pythonic_call(rendered: str) -> tuple[str, dict]:
    """Read LFM2.5's Pythonic tool-call form back to (name, arguments).

    The body between the markers is a valid Python call expression, so
    ``ast`` reads it directly rather than hand-rolling a second parser.
    """
    match = _PYTHONIC_CALL_RE.search(rendered)
    if not match:
        raise ValueError(f"no tool-call markers in rendered text: {rendered!r}")
    expr = ast.parse(match.group("body"), mode="eval").body
    (call,) = expr.elts
    arguments = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return call.func.id, arguments


def parse_rendered_call(rendered: str) -> tuple[str, dict]:
    """Parse *rendered*'s tool call, picked by what the text itself looks like.

    Chosen by the rendered form -- Qwen's XML function/parameter tags or
    LFM's Pythonic call markers -- never by which base model produced it
    (rule: never switch on a model/operation name), so a base this file has
    never heard of that happens to render one of these two known shapes is
    still verified correctly.
    """
    if _PYTHONIC_CALL_RE.search(rendered):
        return _parse_pythonic_call(rendered)
    if _QWEN_FUNCTION_RE.search(rendered):
        return _parse_qwen_call(rendered)
    raise ValueError(f"unrecognized tool-call form in rendered text: {rendered!r}")


def _call(name: str, arguments: dict, arguments_as: str) -> dict:
    rendered = (
        json.dumps(arguments, sort_keys=True) if arguments_as == ARGUMENTS_AS_STRING else arguments
    )
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": rendered}},
        ],
    }


def answer_for(entry: CorpusEntry, arguments_as: str = ARGUMENTS_AS_OBJECT) -> dict:
    """The assistant turn the corpus expects for *entry*."""
    if entry.expect.get("explain"):
        answer = entry.expect.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"{entry.id}: an explain entry needs a non-empty 'answer' in its expect block"
                " -- explain text is authored with the corpus entry, never generated here"
            )
        return _call(lfm.EXPLAIN_TOOL, {"text": answer}, arguments_as)
    if entry.expect.get("escalate"):
        return _call(lfm.ESCALATE_TOOL, {"reason": ESCALATE_REASON}, arguments_as)
    arguments = {
        "operation": entry.expect["operation"],
        "arguments": dict(entry.expect.get("args", {})),
    }
    return _call(lfm.PROPOSE_TOOL, arguments, arguments_as)


def user_message_for(entry: CorpusEntry) -> str:
    """The user message Tier 2 itself would see for *entry*'s request.

    Built through ``nvsh.tiers.bench``'s own ``request_for``/``context_for``
    (the same construction ``bench.py``'s ``run_items`` uses) and
    ``nvsh.tiers.lfm``'s ``request_message`` (the function ``LfmTier.select``
    calls itself), for *every* entry kind -- a "failure" entry gets the
    runtime shape (``command:``/``exit status:``/``output tail:``); an
    "explicit" entry gets its text run through the same redaction and
    ``REQUEST_CHARS`` clamp ``request_message`` applies at run time. A model
    tuned on this file must see in training exactly what it sees at run
    time, redaction and clamp included -- training on the raw corpus text
    would teach it a request shape it will never actually receive (honesty
    h7).
    """
    return lfm.request_message(request_for(entry), context_for(entry))


def example_from_entry(
    entry: CorpusEntry,
    platform: Platform,
    arguments_as: str = ARGUMENTS_AS_OBJECT,
    source_id: str | None = None,
) -> dict:
    """One training example: Tier 2's own brief and tools, the request, the expected call."""
    messages = [
        {"role": "system", "content": lfm.system_brief(platform)},
        {"role": "user", "content": user_message_for(entry)},
        answer_for(entry, arguments_as),
    ]
    return {
        "messages": messages,
        "tools": lfm.tools_for(),
        "source_id": entry.id if source_id is None else source_id,
    }


def render_assistant_text(tokenizer, example: dict) -> str:
    """*example*'s assistant turn, rendered by *tokenizer*'s own chat template.

    Diffs the full render against the render of every message but the last
    (with ``add_generation_prompt=True``) to isolate exactly the text the
    template writes for the assistant's tool call -- the same text a served
    model is actually trained to reproduce, in whatever form that base's own
    template uses (LFM2.5's Pythonic call, Qwen's XML function/parameter
    form, or anything else). *tokenizer* only needs ``apply_chat_template``;
    no transformers import happens here, so this stays usable from a fake
    tokenizer in tests that have no training stack installed.
    """
    messages, tools = example["messages"], example["tools"]
    full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, tokenize=False, add_generation_prompt=True
    )
    if not full.startswith(prefix):
        raise ValueError(
            "rendered prefix does not match the full render -- cannot isolate the assistant span"
        )
    return full[len(prefix) :]


def verify_round_trip(example: dict, tokenizer, parse_call) -> None:
    """Render *example* with *tokenizer* and check the call round-trips.

    *parse_call* is a format-specific reader of the rendered assistant span
    (for example, tests/test_lfm_finetune_dataset.py's small Qwen XML
    function/parameter reader, or its Pythonic-call reader for LFM2.5) that
    returns ``(name, arguments)``. The served vLLM tool-call parser is
    checked separately, in the serving smoke task; this only confirms that
    *tokenizer*'s own chat template renders the call this file wrote in a
    form that loses nothing -- a base tokenizer that can't reproduce an
    example losslessly must not ship it silently.
    """
    expected = example["messages"][-1]["tool_calls"][0]["function"]
    expected_arguments = expected["arguments"]
    if isinstance(expected_arguments, str):
        expected_arguments = json.loads(expected_arguments)
    rendered = render_assistant_text(tokenizer, example)
    name, arguments = parse_call(rendered)
    if name != expected["name"] or arguments != expected_arguments:
        raise ValueError(
            f"{example.get('source_id')}: rendered call does not round-trip -- got "
            f"{name}({arguments!r}), expected {expected['name']}({expected_arguments!r})"
        )


def _one_example_per_outcome(examples: list[dict]) -> list[dict]:
    """One example per distinct tool name in *examples* (first one seen)."""
    seen: dict[str, dict] = {}
    for example in examples:
        name = example["messages"][-1]["tool_calls"][0]["function"]["name"]
        seen.setdefault(name, example)
    return list(seen.values())


def verify_build(
    examples: list[dict],
    base: str = DEFAULT_BASE,
    revision: str = DEFAULT_REVISION,
    tokenizer=None,
) -> None:
    """Guard *examples* before they are written (Codex review finding #6).

    One example per outcome present (propose, escalate, explain) must
    round-trip through *base*'s own chat template -- ``verify_round_trip``
    used to be called only by tests, so a base that could not actually
    reproduce a record this file wrote shipped silently. *tokenizer* lets a
    caller that already has one loaded (a test's fake tokenizer, or a
    caller with the training stack already imported) skip the load;
    otherwise *base* at *revision* is loaded from the local Hugging Face
    cache only (``local_files_only=True`` -- this is a dev-machine,
    offline-first pipeline, so a missing cache is a clear error here, never
    a silent network fetch).
    """
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base, revision=revision, local_files_only=True)
    for example in _one_example_per_outcome(examples):
        verify_round_trip(example, tokenizer, parse_rendered_call)


def _source_ids(path: Path) -> dict[str, str]:
    """Map each entry id in *path* to its ``source_id`` (itself, absent one).

    Read straight from the file's raw JSON, the way ``split.py`` reads it --
    :class:`CorpusEntry` (``load_corpus``) doesn't carry ``source_id``, since
    only a split file (never a plain corpus) has one.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    raw_entries = raw.get("entries", []) if isinstance(raw, dict) else raw
    if not isinstance(raw_entries, list):
        return {}
    ids: dict[str, str] = {}
    for item in raw_entries:
        if isinstance(item, dict) and "id" in item:
            entry_id = str(item["id"])
            ids[entry_id] = str(item.get("source_id", entry_id))
    return ids


def _header(source: Path) -> str:
    """*source*'s raw ``"header"`` string, or ``""`` if absent/unreadable."""
    try:
        with open(source, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return ""
    header = raw.get("header") if isinstance(raw, dict) else None
    return header if isinstance(header, str) else ""


def candidate_pool(reasons: bool) -> tuple[str, ...]:
    """Every candidate a build offers: :func:`scorer.candidates`, or with

    *reasons*, its bare ``escalate`` replaced by :data:`REASON_CANDIDATES`
    (16 operations + explain + 8 reasons = 25, still within
    ``scorer.LABEL_ALPHABET``).
    """
    base = scorer.candidates()
    if not reasons:
        return base
    return tuple(name for name in base if name != lfm.ESCALATE_TOOL) + REASON_CANDIDATES


def reason_for_entry(entry: CorpusEntry) -> str:
    """*entry*'s escalate-reason candidate: from its ``class`` field, or :data:`DEFAULT_REASON`.

    The corpus's ``class`` field (``CorpusEntry.phrasing``) carries
    ``"decline:<reason>"`` for an escalate entry; the matching candidate is
    ``"escalate:<reason>"``. A missing or unrecognised class rolls up to
    :data:`DEFAULT_REASON`, never a fabricated one.
    """
    cls = entry.phrasing or ""
    if cls.startswith("decline:"):
        candidate = "escalate:" + cls.split(":", 1)[1]
        if candidate in REASON_CANDIDATES:
            return candidate
    return DEFAULT_REASON


def gold_for(entry: CorpusEntry, reasons: bool) -> str:
    """The candidate name *entry*'s own expected answer names.

    An operation name, ``"explain"``, ``"escalate"``, or -- with *reasons*
    -- ``"escalate:<reason>"``. This is what a scorer (train_scorer.py, t16)
    is trained to pick; it never depends on which candidates are offered.
    """
    if entry.expect.get("explain"):
        return lfm.EXPLAIN_TOOL
    if entry.expect.get("escalate"):
        return reason_for_entry(entry) if reasons else lfm.ESCALATE_TOOL
    return entry.expect["operation"]


def example_seed(perm_seed: int, example_id: str) -> int:
    """A deterministic per-example seed: independent of process or machine.

    Derived from ``sha256(f"{perm_seed}:{example_id}")`` (the first 16 hex
    digits, as an int) so the same (*perm_seed*, *example_id*) pair always
    gives the same seed -- what makes a rendered example "seeded and
    replayable" (t14 acceptance): a consumer holding only the stored
    ``perm_seed`` can hand it straight to :func:`scorer.permute` again.
    """
    digest = hashlib.sha256(f"{perm_seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def subset_size(seed: int, pool_size: int, min_subset: int, full_probability: float) -> int:
    """A deterministic subset size in ``[min_subset, pool_size]``, drawn from *seed*.

    With probability *full_probability* the full pool is offered; otherwise
    a size is drawn uniformly from ``[min_subset, pool_size - 1]``. A pool no
    larger than *min_subset* always offers the full pool.
    """
    rng = random.Random(seed)  # nosec B311 - dataset shaping, not security
    if pool_size <= min_subset or rng.random() < full_probability:
        return pool_size
    return rng.randint(min_subset, pool_size - 1)


def default_permutation(offered: Sequence[str], full: Sequence[str]):
    """Today's fixed order + letter map: *offered*, labelled by its index in *full*.

    Mirrors :func:`scorer.labels_for`: a candidate's letter comes from its
    position in the *full* candidate list, so it does not move when others
    are left out (a missing-candidate example's reduced offer keeps the same
    letters the full offer would have given it). *full* may include the
    reason candidates (:data:`REASON_CANDIDATES`), which are not in
    :func:`scorer.candidates` at all, so this is a small reimplementation
    rather than a call to ``labels_for`` itself.
    """
    letters = scorer.LABEL_ALPHABET
    if len(full) > len(letters):
        raise ValueError(f"{len(full)} candidates but only {len(letters)} labels")
    index = {name: position for position, name in enumerate(full)}
    labels = {name: letters[index[name]] for name in offered}
    return scorer.Permutation(order=tuple(offered), labels=labels)


def build_permutation(
    example_id: str,
    gold: str,
    offered: Sequence[str],
    full: Sequence[str],
    perm_seed: int,
    *,
    randomize: bool,
    min_subset: int,
    full_probability: float,
):
    """This example's ``(Permutation, seed)``. Seeded and replayable (t14).

    Without *randomize*, this is :func:`default_permutation` of *offered*
    against *full* (today's fixed map, so scorer-b1's data stays
    reproducible). With it, a subset size is drawn deterministically from
    the per-example seed and handed to :func:`scorer.permute` along with
    *offered* as the pool, keeping *gold* in the subset.
    """
    seed = example_seed(perm_seed, example_id)
    if not randomize:
        return default_permutation(offered, full), seed
    size = subset_size(seed, len(offered), min_subset, full_probability)
    permutation = scorer.permute(seed, pool=list(offered), subset=size, keep=gold)
    return permutation, seed


def descriptions_for(order: Sequence[str]) -> dict[str, str]:
    """Description overrides for *order*'s reason candidates only; ``{}`` otherwise."""
    return {name: REASON_DESCRIPTIONS[name] for name in order if name in REASON_DESCRIPTIONS}


def enrich_example(
    example: dict,
    example_id: str,
    gold: str,
    offered: Sequence[str],
    full: Sequence[str],
    perm_seed: int,
    *,
    randomize: bool,
    min_subset: int,
    full_probability: float,
) -> dict:
    """*example* plus the shared scorer-training contract (t14/t16): permutation, gold, perm_seed.

    ``descriptions`` is added only when *order* offers a reason candidate
    (never a bare, unconditional key) -- see the t14/t16 shared contract.
    """
    permutation, seed = build_permutation(
        example_id,
        gold,
        offered,
        full,
        perm_seed,
        randomize=randomize,
        min_subset=min_subset,
        full_probability=full_probability,
    )
    enriched = {
        **example,
        "permutation": permutation.to_json(),
        "gold": gold,
        "perm_seed": seed,
    }
    descriptions = descriptions_for(permutation.order)
    if descriptions:
        enriched["descriptions"] = descriptions
    return enriched


def missing_candidate_example(
    entry: CorpusEntry, platform: Platform, arguments_as: str, source_id: str
) -> dict:
    """The eval_slices.py-shaped derived example: *entry* with its gold answer forced to escalate.

    Used as the base for a ``<id>-nocand`` train row (the gold operation is
    removed from the offered candidates by the caller); the assistant turn
    itself must also say escalate, since the operation it would have
    proposed is no longer offered.
    """
    synthetic = replace(entry, id=f"{entry.id}-nocand", expect={"escalate": True})
    return example_from_entry(synthetic, platform, arguments_as, source_id=source_id)


def _select_missing_candidate(perm_seed: int, entry_id: str, rate: float) -> bool:
    """Deterministic, seeded yes/no: does *entry_id* also get a ``-nocand`` example."""
    if rate <= 0:
        return False
    seed = example_seed(perm_seed, f"{entry_id}:missing-candidate")
    return random.Random(seed).random() < rate  # nosec B311 - dataset shaping, not security


def eval_side_markers(path: Path) -> set[str]:
    """Which of ``{"val", "test", "held-out"}`` *path*'s own name claims.

    Mirrors ``calibration_fit.split_markers``'s word-boundary/compact-stem
    approach, but by file name only, never header prose: a corpus header is
    free text that can innocuously mention another split's file name (as
    ``dev.json``'s own header names ``held-out.json``), so header scanning
    would false-positive on the very corpus this build reads every day. A
    ``--split`` file's actual side is instead read from its
    ``"Split '<side>' of ..."`` header note by ``build()`` itself, above;
    this only covers a plain ``--corpus`` path with no such note, e.g. one
    merely named ``test.json`` or ``val.json``.
    """
    stem_words = {word for word in re.split(r"[^a-z0-9]+", path.stem.lower()) if word}
    compact_stem = re.sub(r"[^a-z0-9]", "", path.stem.lower())
    markers: set[str] = set()
    if "test" in stem_words:
        markers.add("test")
    if "heldout" in compact_stem:
        markers.add("held-out")
    if "val" in stem_words:
        markers.add("val")
    return markers


def build(
    source: Path,
    arguments_as: str = ARGUMENTS_AS_OBJECT,
    is_split: bool = False,
    verify_render: bool = False,
    base: str = DEFAULT_BASE,
    revision: str = DEFAULT_REVISION,
    tokenizer=None,
    *,
    reasons: bool = False,
    randomize_labels: bool = False,
    missing_candidate_rate: float = 0.0,
    perm_seed: int = 0,
    min_subset: int = DEFAULT_MIN_SUBSET,
    full_set_probability: float = DEFAULT_FULL_SET_PROBABILITY,
) -> list[dict]:
    """Every example *source* yields. Refuses the held-out split.

    *verify_render* wires ``verify_build`` (finding #6) into the build
    itself: when true, one example per outcome is checked to round-trip
    through *base*'s (at *revision*) own chat template before this
    function returns, and a mismatch raises instead of shipping silently.
    It defaults to false here so plain library calls -- including this
    module's own tests, most of which run with no training stack
    installed -- never need a tokenizer; the CLI turns it on by default
    instead (``--no-verify-render`` to opt back out). *tokenizer* lets a
    caller supply an already-loaded (or fake) tokenizer instead of loading
    *base* from the local cache.

    *source* is either a plain corpus (``--corpus``) or a train/val/test
    split file written by ``split.py`` (``--split``): both share the same
    ``{"header", "entries"}`` shape, so the same loading path builds either
    one -- a split file's ``world`` is simply absent, giving the default
    platform, and its entries' ``source_id`` travels into each example.

    *is_split* is ``True`` when *source* is being used as a ``--split``
    file (a train/val/test side written by ``split.py``, never a plain
    corpus): only the train side may then be used, because training on val
    or test would contaminate the fold a tuned model is later measured on
    (honesty h7). A ``--split`` file whose header does not name a side at
    all is refused too -- a corpus's header never claims to be a split
    side, so a file that is supposed to be one but doesn't say so cannot be
    trusted to be the train side (e.g. a renamed ``test.json`` whose header
    was stripped or hand-edited away from ``split.py``'s own wording).

    Every rendered example also carries the t14/t16 shared scorer-training
    contract (:func:`enrich_example`): ``permutation`` (order + letter map),
    ``gold``, ``perm_seed``, and, where a reason candidate is offered,
    ``descriptions``. *reasons* offers the 8 escalate-reason candidates
    (:data:`REASON_CANDIDATES`) instead of the bare ``escalate``. Without
    *randomize_labels* every example keeps today's fixed order and letter
    map (:func:`default_permutation`); with it, a seeded random subset
    (bounds *min_subset*/*full_set_probability*) and letter permutation are
    drawn per example (:func:`build_permutation`), keeping the gold
    candidate. *perm_seed* is the one global seed mixed with each example's
    own id to derive its seed, so the same *perm_seed* always replays the
    same file.

    *missing_candidate_rate* additionally emits, for that fraction
    (deterministic by seed) of train entries whose gold is an operation, a
    derived ``<id>-nocand`` example (:func:`missing_candidate_example`,
    ``eval_slices.py``'s shape) with that operation removed from the
    offered candidates and the gold forced to escalate (``escalate:
    outside_table`` with *reasons*). This only ever happens from the train
    side: a ``--split`` file's side is already checked above, and a plain
    ``--corpus`` path is refused when it is named or headed like a val,
    test or held-out side (:func:`eval_side_markers`).
    """
    if source.name == HELD_OUT_NAME:
        raise ValueError("the held-out split is for judging a tuned model, never for training it")
    header = _header(source)
    if _HELD_OUT_HEADER_MARKER in header:
        raise ValueError("the held-out split is for judging a tuned model, never for training it")
    if is_split:
        match = _SPLIT_SIDE_RE.search(header)
        side = match.group(1) if match else None
        if side is None:
            raise ValueError(
                f"{source}: --split file's header names no side -- only a train-side split "
                "file written by split.py may be used for training"
            )
        if side != TRAIN_SIDE:
            raise ValueError(
                f"{source}: --split file is the {side!r} side -- only the {TRAIN_SIDE!r} side "
                "may be used for training; val/test are held out for evaluation"
            )
    if missing_candidate_rate > 0 and not is_split:
        markers = eval_side_markers(source)
        if markers:
            raise ValueError(
                f"{source}: looks like the {' and '.join(sorted(markers))} side -- "
                "missing-candidate examples are only ever generated from the train side"
            )
    loaded = load_corpus(source)
    platform = world_platform(load_world(source))
    source_ids = _source_ids(source)
    pool = candidate_pool(reasons)
    fallback_reason = DEFAULT_REASON if reasons else lfm.ESCALATE_TOOL

    examples: list[dict] = []
    for entry in loaded.entries:
        source_id = source_ids.get(entry.id, entry.id)
        example = example_from_entry(entry, platform, arguments_as, source_id)
        gold = gold_for(entry, reasons)
        examples.append(
            enrich_example(
                example,
                entry.id,
                gold,
                pool,
                pool,
                perm_seed,
                randomize=randomize_labels,
                min_subset=min_subset,
                full_probability=full_set_probability,
            )
        )
        operation = entry.expect.get("operation")
        if operation is not None and _select_missing_candidate(
            perm_seed, entry.id, missing_candidate_rate
        ):
            derived = missing_candidate_example(entry, platform, arguments_as, source_id)
            derived_pool = tuple(name for name in pool if name != operation)
            examples.append(
                enrich_example(
                    derived,
                    f"{entry.id}-nocand",
                    fallback_reason,
                    derived_pool,
                    pool,
                    perm_seed,
                    randomize=randomize_labels,
                    min_subset=min_subset,
                    full_probability=full_set_probability,
                )
            )
    if verify_render:
        verify_build(examples, base, revision, tokenizer)
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=None, help="a plain corpus file (default: dev.json)")
    parser.add_argument(
        "--split",
        default=None,
        help="a train/val/test side written by split.py, in place of --corpus",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--arguments-as",
        choices=ARGUMENTS_AS_CHOICES,
        default=ARGUMENTS_AS_OBJECT,
        help=(
            "tool_calls[].function.arguments shape: 'object' (default, what "
            "apply_chat_template documents) or 'string' (the OpenAI wire "
            "format Tier 2 itself replays -- see docs/lfm-finetune.md)"
        ),
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"base whose chat template verify_round_trip checks against (default {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="commit of --base to check against (default matches train.py's own pin)",
    )
    parser.add_argument(
        "--no-verify-render",
        action="store_true",
        help=(
            "skip the per-outcome round-trip check (finding #6); only for an "
            "environment without --base's tokenizer cached locally -- the "
            "check is on by default"
        ),
    )
    parser.add_argument(
        "--reasons",
        action="store_true",
        help=(
            "offer the 8 decline classes as distinct escalate-reason candidates "
            "(escalate:<reason>, from scripts/lfm-finetune/data/reasons.json) "
            "instead of the bare 'escalate', rolling up to escalate in gold labels"
        ),
    )
    parser.add_argument(
        "--randomize-labels",
        action="store_true",
        help=(
            "seed each example's own candidate subset/order/letter map from "
            "--perm-seed and its id; without this, every example keeps today's "
            "fixed order and letter map"
        ),
    )
    parser.add_argument(
        "--perm-seed",
        type=int,
        default=0,
        help="global seed mixed with each example's id (--randomize-labels; default 0)",
    )
    parser.add_argument(
        "--min-subset",
        type=int,
        default=DEFAULT_MIN_SUBSET,
        help=(
            f"smallest randomized candidate subset "
            f"(--randomize-labels; default {DEFAULT_MIN_SUBSET})"
        ),
    )
    parser.add_argument(
        "--full-set-probability",
        type=float,
        default=DEFAULT_FULL_SET_PROBABILITY,
        help=(
            "chance a randomized example offers the full candidate pool "
            f"(--randomize-labels; default {DEFAULT_FULL_SET_PROBABILITY})"
        ),
    )
    parser.add_argument(
        "--missing-candidate-rate",
        type=float,
        default=0.0,
        help=(
            "fraction of train operation entries that also get a derived "
            "<id>-nocand example (gold operation removed, gold escalate); "
            "train side only (default 0, off)"
        ),
    )
    args = parser.parse_args(argv)
    if args.split and args.corpus:
        parser.error("--split and --corpus are mutually exclusive")
    if args.split:
        source = Path(args.split)
    else:
        source = Path(args.corpus or str(_REPO_ROOT / "nvsh/tiers/corpus/dev.json"))
    out = Path(args.out)
    if out.resolve() == source.resolve():
        parser.error("--out must not be the corpus file")
    try:
        examples = build(
            source,
            args.arguments_as,
            is_split=bool(args.split),
            verify_render=not args.no_verify_render,
            base=args.base,
            revision=args.revision,
            reasons=args.reasons,
            randomize_labels=args.randomize_labels,
            missing_candidate_rate=args.missing_candidate_rate,
            perm_seed=args.perm_seed,
            min_subset=args.min_subset,
            full_set_probability=args.full_set_probability,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with open(out, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"written={len(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
