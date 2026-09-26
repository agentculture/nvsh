#!/usr/bin/env python3
"""Draft sealed held-out entries for issue 46 (task t17, decision c51) with Qwen3.5-4B.

A development-machine tool (part of #46); never imported by the nvsh package.
The drafting model (Apache-2.0, not one of the augmentation teachers) sees ONLY
the nvsh operation table -- names, descriptions and argument specs -- never
dev.json or any split. dev.json is read only to drop exact repeats. The draft
goes to OUT_DIR/draft.json for the operator to review in a separate sitting;
this script prints counts and a hash, never entry text.

Usage (the training environment on the path, the model already in the HF cache)::

    PYTHONPATH=<training site-packages>:. python scripts/lfm-finetune/draft_heldout.py \
        OUT_DIR [--seed N]

The seed defaults to 46 (issue 46's sealed draft); issue 53 drafts a fresh held-out with its own
seed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3.5-4B"
#: The snapshot the sealed issue-46 draft was made with (2026-09-23).
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 46


def _arg_text(arg) -> str:
    if arg.kind == "choice":
        return f"{arg.name} (one of: {'/'.join(arg.choices)})"
    return f"{arg.name} (free text)"


def table_text(table) -> str:
    lines = []
    for op in table.OPERATIONS:
        args = ", ".join(_arg_text(a) for a in op.args) or "no arguments"
        effect = "read-only" if op.read_only else "changes the machine"
        lines.append(f"- {op.name}: {op.description} [{effect}; arguments: {args}]")
    return "\n".join(lines)


SYSTEM = (
    "You write realistic test requests for a shell assistant on NVIDIA Jetson and DGX Spark "
    "machines. The assistant can only act through the operations listed below. Write the way a "
    "busy engineer types at a terminal: short, varied wording, sometimes informal, sometimes with "
    "typos, never copying the operation's name or description word for word. Reply with JSON only."
)


_OP_ASK = (
    "Write 3 different user requests that should be handled by the operation {name}. "
    "For each, give the exact arguments it needs (use only allowed choices; for free-text "
    "arguments use a concrete realistic value that appears in the request). Reply as a JSON "
    'list of objects with keys "text" and "args" (an object).'
)
_ESCALATE_ASK = (
    "Write 16 different user requests that the assistant must NOT handle with any of these "
    "operations and must hand back to a human instead: requests needing an operation that is "
    "not listed, destructive or risky actions (deleting data, reflashing, changing users or "
    "firewall), and requests too ambiguous to act on safely. Reply as a JSON list of objects "
    'with a single key "text".'
)
_EXPLAIN_ASK = (
    "Write 16 different user questions that should be answered in words, without running "
    "anything: questions about what a power mode, a container runtime, CUDA, unified memory, "
    "a log message or one of these tools means or how it works. Reply as a JSON list of "
    'objects with keys "text" and "answer" (a one-sentence correct answer).'
)


def prompts(table) -> list[tuple[str, str]]:
    head = f"Operations:\n{table_text(table)}\n\n"
    out = [(f"op:{op.name}", head + _OP_ASK.format(name=op.name)) for op in table.OPERATIONS]
    out.append(("escalate", head + _ESCALATE_ASK))
    out.append(("explain", head + _EXPLAIN_ASK))
    return out


def as_item(item) -> dict:
    """A reply item as a dict: a bare string is a request with no other fields.

    A small model sometimes answers ``["...", "..."]`` instead of objects; an
    operation item then has no ``args`` and fails validation as it should.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"text": item}
    return {}


def parse_json_list(raw: str):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = raw.find("["), raw.rfind("]")
    return json.loads(raw[start : end + 1])


def _take(args: list[str], flag: str) -> int | None:
    """Remove ``flag N`` from *args* and return N; a usage error when N is missing."""
    if flag not in args:
        return None
    at = args.index(flag)
    if at + 1 >= len(args):
        raise SystemExit(f"draft_heldout.py: {flag} needs a value")
    value = int(args[at + 1])
    del args[at : at + 2]
    return value


def parse_args(argv: list[str]) -> tuple[Path, int]:
    """``OUT_DIR [--seed N] [--issue N]`` -> (out_dir, seed); seed defaults to :data:`SEED`."""
    args = list(argv)
    seed = _take(args, "--seed")
    _take(args, "--issue")
    if not args:
        raise SystemExit("usage: draft_heldout.py OUT_DIR [--seed N] [--issue N]")
    return Path(args[0]), SEED if seed is None else seed


def parse_issue(argv: list[str]) -> int:
    """The issue a draft is for (``--issue N``; default 46, the one this tool began with)."""
    issue = _take(list(argv), "--issue")
    return 46 if issue is None else issue


def draft_header(issue: int, snapshot: str, seed: int) -> str:
    """The draft's header, naming the issue it was drafted for (PR #65 review)."""
    origins = {46: " (task t17, decision c51)", 53: " (task t13)"}
    return (
        f"Issue {issue} sealed held-out draft{origins.get(issue, '')}: drafted by"
        f" Qwen/Qwen3.5-4B (snapshot {snapshot}, Apache-2.0, not a pipeline teacher) from the"
        f" nvsh operation table only, seed {seed}, temperature 0.7, thinking off. Awaiting"
        " operator review; the lead agent has not read these entries."
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out_dir, seed = parse_args(argv)
    issue = parse_issue(argv)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nvsh.ops import table

    random.seed(seed)
    torch.manual_seed(seed)
    snap = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    tok = AutoTokenizer.from_pretrained(snap, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        snap, revision=REVISION, dtype=torch.bfloat16, device_map="cuda"
    )
    entries, rejects, raw_log = [], {"parse": 0, "invalid_op_args": 0, "duplicate": 0}, []
    dev_texts = {
        " ".join(e["text"].lower().split())
        for e in json.loads(Path("nvsh/tiers/corpus/dev.json").read_text())["entries"]
    }
    seen = set()
    for key, prompt in prompts(table):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        out = model.generate(**ids, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9)
        raw = tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        raw_log.append({"prompt_key": key, "raw": raw})
        try:
            items = parse_json_list(raw)
        except Exception:
            rejects["parse"] += 1
            continue
        for item in map(as_item, items if isinstance(items, list) else []):
            text = str(item.get("text", "")).strip()
            norm = " ".join(text.lower().split())
            if not text or norm in seen or norm in dev_texts:
                rejects["duplicate"] += 1
                continue
            if key.startswith("op:"):
                op, args = key[3:], item.get("args") or {}
                if table.validate(op, args) is not None:
                    rejects["invalid_op_args"] += 1
                    continue
                expect = {"operation": op, "args": args}
            elif key == "escalate":
                expect = {"escalate": True}
            else:
                expect = {"explain": True, "answer": str(item.get("answer", "")).strip()}
            seen.add(norm)
            entries.append(
                {
                    "id": f"ho46-{len(entries) + 1:03d}",
                    "kind": "explicit",
                    "text": text,
                    "expect": expect,
                    "source": "t17-qwen3.5-4b-draft",
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "header": draft_header(issue, snap.name, seed),
        "entries": entries,
    }
    body = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "draft.json").write_text(body, encoding="utf-8")
    (out_dir / "raw-generations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_log), encoding="utf-8"
    )
    kinds = {}
    for e in entries:
        k = "operation" if "operation" in e["expect"] else next(iter(e["expect"]))
        kinds[k] = kinds.get(k, 0) + 1
    print(
        json.dumps(
            {
                "snapshot": snap.name,
                "entries": len(entries),
                "by_kind": kinds,
                "rejects": rejects,
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
