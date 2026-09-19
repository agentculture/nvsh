#!/usr/bin/env python3
"""Build a chat-format fine-tune file for Tier 2 (LFM2.5) from the nvsh benchmark corpus.

A development-machine tool for ``docs/lfm-finetune.md``. It is NEVER imported by
the nvsh package -- nothing under nvsh/ may depend on it.

Usage::

    python scripts/lfm-finetune/build_dataset.py --out train.jsonl [--corpus PATH]

Each output line is ``{"messages": [...], "tools": [...]}`` in the shape the
Hugging Face chat templates, TRL's ``SFTTrainer`` and unsloth accept. The tools
and the system brief are the ones Tier 2 really sends (``nvsh.tiers.lfm``), so
a model tuned on this file sees at run time exactly what it saw in training.

Only single-turn examples can be built from the corpus: an explicit request
answered by ``propose``, and a should-decline request answered by ``escalate``.
Multi-round inspection examples need real Tier 2 records; they are not invented
here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.platform._model import Platform  # noqa: E402
from nvsh.tiers import lfm  # noqa: E402
from nvsh.tiers.bench import (  # noqa: E402
    CorpusEntry,
    _context_for,
    _request_for,
    load_corpus,
    load_world,
    world_platform,
)

#: ``nvsh.tiers.bench``'s own request/context builders (private: leading
#: underscore) -- reused here rather than reimplemented so a FAILURE entry's
#: training user message is built the exact same way ``bench.py`` and Tier 2
#: itself (``nvsh.tiers.lfm``'s ``_request_message``, also private) build it
#: at run time. Both are imported anyway per this script's brief; the lead
#: has been told to make them public.

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The split a tuned model is judged on. Training on it would make that judgement worthless.
HELD_OUT_NAME = "held-out.json"

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
    if entry.expect.get("escalate"):
        return _call(lfm.ESCALATE_TOOL, {"reason": ESCALATE_REASON}, arguments_as)
    arguments = {
        "operation": entry.expect["operation"],
        "arguments": dict(entry.expect.get("args", {})),
    }
    return _call(lfm.PROPOSE_TOOL, arguments, arguments_as)


def user_message_for(entry: CorpusEntry) -> str:
    """The user message Tier 2 itself would see for *entry*'s request.

    A "failure" entry is built through ``nvsh.tiers.bench``'s own
    ``_request_for``/``_context_for`` (the same construction ``bench.py``'s
    ``run_items`` uses) and ``nvsh.tiers.lfm``'s ``_request_message`` (the
    function ``LfmTier.select`` calls) -- so its user message has the
    runtime shape (``command:``/``exit status:``/``output tail:``), not the
    corpus entry's raw text verbatim. An "explicit" entry's message is the
    entry's text unchanged, matching what ``_request_message`` itself does
    for a non-FAILURE request (modulo the redaction/length clamp it also
    applies at run time, deliberately not replayed here so training text
    stays exactly what a human wrote in the corpus).
    """
    if entry.kind == "failure":
        return lfm._request_message(_request_for(entry), _context_for(entry))
    return entry.text


def example_from_entry(
    entry: CorpusEntry, platform: Platform, arguments_as: str = ARGUMENTS_AS_OBJECT
) -> dict:
    """One training example: Tier 2's own brief and tools, the request, the expected call."""
    messages = [
        {"role": "system", "content": lfm.system_brief(platform)},
        {"role": "user", "content": user_message_for(entry)},
        answer_for(entry, arguments_as),
    ]
    return {"messages": messages, "tools": lfm.tools_for()}


def build(corpus: Path, arguments_as: str = ARGUMENTS_AS_OBJECT) -> list[dict]:
    """Every example the corpus at *corpus* yields. Refuses the held-out split."""
    if corpus.name == HELD_OUT_NAME:
        raise ValueError("the held-out split is for judging a tuned model, never for training it")
    loaded = load_corpus(corpus)
    platform = world_platform(load_world(corpus))
    return [example_from_entry(entry, platform, arguments_as) for entry in loaded.entries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(_REPO_ROOT / "nvsh/tiers/corpus/dev.json"))
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
    corpus, out = Path(args.corpus), Path(args.out)
    if out.resolve() == corpus.resolve():
        parser.error("--out must not be the corpus file")
    try:
        examples = build(corpus, args.arguments_as)
    except ValueError as exc:
        parser.error(str(exc))
    with open(out, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"written={len(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
