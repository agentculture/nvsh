#!/usr/bin/env python3
"""Build a Needle3 LoRA fine-tune training file from the nvsh benchmark corpus.

This is a development-machine tool for the Needle3 LoRA fine-tune recipe.
It is NEVER imported by the nvsh package -- nothing under nvsh/ may depend on it.

Usage::

    python scripts/needle-finetune/build_dataset.py --corpus PATH --bundle PATH --out PATH

The script reads the nvsh benchmark corpus (and an optional export bundle),
converts entries into JSONL examples, deduplicates, and writes the result
for use with ``needle finetune``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nvsh.ops import validate
from nvsh.tiers.needle_worker import tool_schemas

# ---------------------------------------------------------------------------
# The repo root is where we find nvsh/tiers/corpus/dev.json by default.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def example_from_entry(entry: dict, tools: list) -> dict | None:
    """Convert one corpus *entry* into an example dict, or ``None`` to skip.

    Returns ``None`` when the entry has ``kind != "explicit"`` (RULE A) or
    when the expected operation fails validation (RULE C).
    """
    if entry.get("kind") != "explicit":
        return None

    expect = entry.get("expect", {})

    # Escalation entry
    if expect.get("escalate"):
        return {
            "query": entry.get("text", ""),
            "tools": tools,
            "answers": [],
        }

    operation = expect.get("operation")
    if operation is None:
        return None

    args = expect.get("args", {})

    if validate(operation, args) is not None:
        return None

    return {
        "query": entry.get("text", ""),
        "tools": tools,
        "answers": [{"name": operation, "arguments": args}],
    }


def example_from_record(record: dict, tools: list) -> dict | None:
    """Convert one export-bundle *record* into an example dict, or ``None`` to skip.

    A record is used only when:
    - operator_decision == "approved"
    - operation is a non-empty string
    - request_text is a non-empty string
    - validate(operation, args) accepts it
    """
    if record.get("operator_decision") != "approved":
        return None

    operation = record.get("operation")
    if not isinstance(operation, str) or not operation:
        return None

    request_text = record.get("request_text")
    if not isinstance(request_text, str) or not request_text:
        return None

    args = record.get("args") or {}
    if validate(operation, args) is not None:
        return None

    return {
        "query": request_text,
        "tools": tools,
        "answers": [],
    }


def build(corpus_path: Path, bundle_path: Path | None, tools: list) -> tuple[list[dict], dict]:
    """Build the full example list and counts from a corpus (and optional bundle).

    Returns ``(examples, counts)`` where *counts* has keys:

    - ``"written"`` -- examples in the output
    - ``"invalid"`` -- corpus entries skipped by RULE C (validation failure)
    - ``"skipped_records"`` -- bundle records skipped by RULE B checks
    - ``"duplicates"`` -- entries dropped by deduplication
    """
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    entries: list[dict] = corpus.get("entries", [])

    seen: set[str] = set()
    examples: list[dict] = []
    counts: dict[str, int] = {"written": 0, "invalid": 0, "skipped_records": 0, "duplicates": 0}

    # Process corpus entries first (file order).
    for entry in entries:
        example = example_from_entry(entry, tools)
        if example is None:
            # entry was skipped by RULE A or RULE C.
            if entry.get("kind") == "explicit":
                # Explicit entry that failed validation → invalid.
                expect = entry.get("expect", {})
                if expect.get("operation") and not expect.get("escalate"):
                    counts["invalid"] += 1
            continue

        key = (example["query"].strip().lower(), json.dumps(example["answers"], sort_keys=True))
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        examples.append(example)
        counts["written"] += 1

    # Process bundle records (file order).
    if bundle_path is not None:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        records: list[dict] = bundle.get("records", [])
        for record in records:
            example = example_from_record(record, tools)
            if example is None:
                counts["skipped_records"] += 1
                continue
            key = (example["query"].strip().lower(), json.dumps(example["answers"], sort_keys=True))
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)
            examples.append(example)
            counts["written"] += 1

    return examples, counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 2 on input errors."""
    parser = argparse.ArgumentParser(
        description="Build a Needle3 LoRA fine-tune training file from the nvsh benchmark corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_REPO_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json",
        help="Path to the dev corpus JSON file (default: nvsh/tiers/corpus/dev.json).",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Optional export bundle JSON written by ``nvsh tiers export``.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL file path.",
    )
    args = parser.parse_args(argv)

    # RULE B: refuse to train on held-out split.
    if args.corpus.name == "held-out.json":
        print("refusing to train on the held-out split", file=sys.stderr)
        return 2

    # Read corpus file.
    if not args.corpus.exists():
        print(f"corpus file not found: {args.corpus}", file=sys.stderr)
        return 2
    try:
        json.loads(args.corpus.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"corpus file is not valid JSON: {exc}", file=sys.stderr)
        return 2

    # Read bundle file if provided.
    bundle_path: Path | None = None
    if args.bundle is not None:
        if not args.bundle.exists():
            print(f"bundle file not found: {args.bundle}", file=sys.stderr)
            return 2
        try:
            json.loads(args.bundle.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"bundle file is not valid JSON: {exc}", file=sys.stderr)
            return 2
        bundle_path = args.bundle

    # Build examples.
    tools = tool_schemas()
    examples, counts = build(args.corpus, bundle_path, tools)

    # Write output.
    with args.out.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, sort_keys=True, ensure_ascii=False) + "\n")

    # Summary line.
    print(
        f"written={counts['written']} invalid={counts['invalid']} "
        f"skipped_records={counts['skipped_records']} duplicates={counts['duplicates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
