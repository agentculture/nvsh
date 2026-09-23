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
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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


def build(
    source: Path, arguments_as: str = ARGUMENTS_AS_OBJECT, is_split: bool = False
) -> list[dict]:
    """Every example *source* yields. Refuses the held-out split.

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
    loaded = load_corpus(source)
    platform = world_platform(load_world(source))
    source_ids = _source_ids(source)
    return [
        example_from_entry(entry, platform, arguments_as, source_ids.get(entry.id))
        for entry in loaded.entries
    ]


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
        examples = build(source, args.arguments_as, is_split=bool(args.split))
    except ValueError as exc:
        parser.error(str(exc))
    with open(out, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"written={len(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
