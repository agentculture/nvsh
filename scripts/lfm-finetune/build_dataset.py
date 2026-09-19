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
from nvsh.tiers.bench import CorpusEntry, load_corpus, load_world, world_platform  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The split a tuned model is judged on. Training on it would make that judgement worthless.
HELD_OUT_NAME = "held-out.json"

#: What the assistant says when it hands a request up. Short on purpose: the
#: reason is for the audit log, the behaviour being taught is the call itself.
ESCALATE_REASON = "this needs the full agent"


def _call(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": arguments}},
        ],
    }


def answer_for(entry: CorpusEntry) -> dict:
    """The assistant turn the corpus expects for *entry*."""
    if entry.expect.get("escalate"):
        return _call(lfm.ESCALATE_TOOL, {"reason": ESCALATE_REASON})
    arguments = {
        "operation": entry.expect["operation"],
        "arguments": dict(entry.expect.get("args", {})),
    }
    return _call(lfm.PROPOSE_TOOL, arguments)


def example_from_entry(entry: CorpusEntry, platform: Platform) -> dict:
    """One training example: Tier 2's own brief and tools, the request, the expected call."""
    messages = [
        {"role": "system", "content": lfm.system_brief(platform)},
        {"role": "user", "content": entry.text},
        answer_for(entry),
    ]
    return {"messages": messages, "tools": lfm.tools_for()}


def build(corpus: Path) -> list[dict]:
    """Every example the corpus at *corpus* yields. Refuses the held-out split."""
    if corpus.name == HELD_OUT_NAME:
        raise ValueError("the held-out split is for judging a tuned model, never for training it")
    loaded = load_corpus(corpus)
    platform = world_platform(load_world(corpus))
    return [example_from_entry(entry, platform) for entry in loaded.entries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(_REPO_ROOT / "nvsh/tiers/corpus/dev.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    corpus, out = Path(args.corpus), Path(args.out)
    if out.resolve() == corpus.resolve():
        parser.error("--out must not be the corpus file")
    try:
        examples = build(corpus)
    except ValueError as exc:
        parser.error(str(exc))
    with open(out, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"written={len(examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
