#!/usr/bin/env python3
"""Calibrate the augmentation reviewer against known-good and known-bad pairs (issue 46, d10).

A development-machine tool (part of #46); never imported by the nvsh package.

The reviewer (``augment.py``'s REVIEWER_B) decides which training candidates
are kept, so it is a grader and must be checked before a full pass (lapse l2).
This builds a small, deterministic probe from train-side re-review candidates:

- **good**: a candidate paired with its own expected answer;
- **bad**: a candidate's text paired with a *wrong* expected answer -- a
  confusable read-only check, a mutating change for a read request, escalate
  for an operation request, an operation for an escalate request -- plus any
  escalate candidate that asks for the hand-off in so many words.

It then asks the reviewer about every item at each reasoning effort, with the
exact prompt and verdict parser ``augment.py`` uses, and prints false accepts
(bad items accepted -- must be 0) and false rejects (good items rejected).
Operations are only ever taken from the table and the candidates' own expect
blocks; nothing here switches on an operation name.

Usage (reviewer B configured through the same ``NVSH_AUG_REVIEWER_B_*``
variables as ``augment.py``)::

    python scripts/lfm-finetune/calibrate_reviewer.py CANDIDATES.jsonl [...] \\
        --efforts medium xhigh --temperature 0.2 --per-kind 6 --out probe.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment as aug  # noqa: E402

from nvsh.ops.table import get as get_operation  # noqa: E402
from nvsh.ops.table import names as operation_names  # noqa: E402

SEED = 46
#: A request that asks for the hand-off in so many words is a bad escalate
#: example (reviewer rule: users never do); it makes a known-bad probe item.
HANDOFF_WORDS = re.compile(
    r"\b(escalat\w*|hand-?off|hand (it|this|that) (off|over)|(human|senior|more capable) "
    r"(agent|assistant|operator|engineer))\b",
    re.IGNORECASE,
)


def _kind(expect: dict[str, Any]) -> str:
    if "operation" in expect:
        operation = get_operation(str(expect["operation"]))
        return "read" if operation is not None and operation.read_only else "change"
    if expect.get("escalate"):
        return "escalate"
    if expect.get("explain"):
        return "explain"
    return "other"


def _default_args(name: str) -> dict[str, str]:
    """The first allowed choice for every choice argument of *name*; a free-text
    argument gets a neutral value. Only used to make a wrong expected answer."""
    operation = get_operation(name)
    args: dict[str, str] = {}
    for arg in operation.args if operation is not None else ():
        args[arg.name] = arg.choices[0] if arg.kind == "choice" and arg.choices else "vllm"
    return args


def build_probe(records: list[dict[str, Any]], per_kind: int, seed: int = SEED) -> list[dict]:
    """Deterministic probe items ``{id, text, expect, label, kind}`` from *records*."""
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(records, key=lambda r: str(r.get("id"))):
        if record.get("seed_format", "split") != "split":
            continue
        pools.setdefault(_kind(record["expect"]), []).append(record)
    reads = [n for n in operation_names() if _kind({"operation": n}) == "read"]
    changes = [n for n in operation_names() if _kind({"operation": n}) == "change"]
    items: list[dict[str, Any]] = []

    def pick(pool: str, count: int) -> list[dict[str, Any]]:
        candidates = [r for r in pools.get(pool, []) if not HANDOFF_WORDS.search(r["text"])]
        return rng.sample(candidates, min(count, len(candidates)))

    for pool in ("read", "change", "escalate", "explain"):
        for record in pick(pool, per_kind):
            items.append(_item(record, record["expect"], "good", f"good-{pool}"))

    for record in pick("read", per_kind):  # a different read-only check
        own = record["expect"]["operation"]
        wrong = rng.choice([n for n in reads if n != own])
        items.append(
            _item(
                record, {"operation": wrong, "args": _default_args(wrong)}, "bad", "bad-other-check"
            )
        )
    for record in pick("read", per_kind):  # a mutating change for a read request
        wrong = rng.choice(changes)
        items.append(
            _item(
                record,
                {"operation": wrong, "args": _default_args(wrong)},
                "bad",
                "bad-change-for-read",
            )
        )
    for record in pick("change", per_kind) + pick("read", per_kind // 2):
        items.append(_item(record, {"escalate": True}, "bad", "bad-escalate-for-op"))
    for record in pick("escalate", per_kind):
        wrong = rng.choice(reads + changes)
        items.append(
            _item(
                record,
                {"operation": wrong, "args": _default_args(wrong)},
                "bad",
                "bad-op-for-escalate",
            )
        )
    for record in pools.get("escalate", []):
        if HANDOFF_WORDS.search(record["text"]):
            items.append(_item(record, record["expect"], "bad", "bad-asks-for-handoff"))
    return items


def _item(record: dict[str, Any], expect: dict[str, Any], label: str, kind: str) -> dict:
    return {
        "id": record["id"],
        "text": record["text"],
        "expect": expect,
        "label": label,
        "kind": kind,
    }


def _ask(role: aug.RoleConfig, item: dict[str, Any], effort: str) -> dict[str, Any]:
    seed = aug.Seed(
        source_id=item["id"],
        seed_format="split",
        side="train",
        seed_text=item["text"],
        expect=item["expect"],
        needs_change_check=aug._needs_change_check(item["expect"]),
    )
    system, user = aug.reviewer_prompt(seed, item["text"])
    payload = {
        "model": role.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": role.temperature,
        "max_tokens": role.max_tokens,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    headers = {"Content-Type": "application/json"}
    if role.key:
        headers["Authorization"] = f"Bearer {role.key}"
    request = urllib.request.Request(
        role.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=role.timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = aug._extract_content(body["choices"][0]["message"])
    accepted, reason = aug.parse_verdict(content)
    return {
        "accepted": accepted,
        "reason": reason,
        "tokens": (body.get("usage") or {}).get("completion_tokens"),
    }


def score(items: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    false_accepts = [
        i["id"] + ":" + i["kind"]
        for i, a in zip(items, answers)
        if i["label"] == "bad" and a["accepted"]
    ]
    false_rejects = [
        i["id"] + ":" + i["kind"]
        for i, a in zip(items, answers)
        if i["label"] == "good" and not a["accepted"]
    ]
    tokens = sorted(a["tokens"] for a in answers if a.get("tokens") is not None)
    return {
        "items": len(items),
        "bad": sum(i["label"] == "bad" for i in items),
        "good": sum(i["label"] == "good" for i in items),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "tokens_mean": round(sum(tokens) / len(tokens)) if tokens else None,
        "tokens_max": tokens[-1] if tokens else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", nargs="+", help="train-side candidate JSONL files")
    parser.add_argument("--efforts", nargs="+", default=["medium", "xhigh"])
    parser.add_argument("--per-kind", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    records = []
    for path in args.candidates:
        records.extend(aug._load_jsonl(Path(path)))
    if any(r.get("side") not in (None, "train") for r in records):
        print("error: calibration uses train-side candidates only", file=sys.stderr)
        return 2
    items = build_probe(records, args.per_kind)
    role = aug.load_role_config("REVIEWER_B")
    report: dict[str, Any] = {
        "temperature": role.temperature,
        "model": role.model,
        "seed": SEED,
        "items": items,
        "efforts": {},
    }
    for effort in args.efforts:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            answers = list(pool.map(lambda item: _ask(role, item, effort), items))
        result = score(items, answers)
        report["efforts"][effort] = {"score": result, "answers": answers}
        print(json.dumps({"effort": effort, **result}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
