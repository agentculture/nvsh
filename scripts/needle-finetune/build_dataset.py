#!/usr/bin/env python3
"""Build a Needle3 LoRA fine-tune training file from the nvsh benchmark corpus.

This is a development-machine tool for the Needle3 LoRA fine-tune recipe.
It is NEVER imported by the nvsh package -- nothing under nvsh/ may depend on it.

Usage::

    python scripts/needle-finetune/build_dataset.py --corpus PATH --bundle PATH --out PATH

The script reads the nvsh benchmark corpus (and an optional export bundle),
converts entries into JSONL examples, deduplicates, and writes the result
for use with ``needle finetune``.

Corpus entries whose ``expect`` is ``{"escalate": true}`` *or*
``{"explain": true}`` both become the same should-decline training example
(an empty ``answers`` list): Tier 1 (Needle3) has no explain capability --
only Tier 2 can inspect, propose, explain or escalate -- so the honest
mapping for a query the corpus expects to be *explained* is "Tier 1 should
not call a tool here", the same target already used for escalation. Such an
entry is neither an error nor a propose example: it is counted under
``written``, never ``invalid``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.ops import validate  # noqa: E402
from nvsh.tiers.needle_worker import tool_schemas  # noqa: E402

# ---------------------------------------------------------------------------
# The repo root is where we find nvsh/tiers/corpus/dev.json by default.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The split a tuned model is judged on. Training on it would make that judgement worthless.
HELD_OUT_NAME = "held-out.json"


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

    # Escalation entry, or an entry whose expectation is ``explain``.
    #
    # Tier 2 can inspect, propose, explain or escalate; Tier 1 (Needle3) can
    # only propose an operation or say nothing. It has no explain
    # capability, so the honest training target for a corpus entry that
    # expects an explanation is the same should-decline shape already used
    # for escalation: an empty ``answers`` list, i.e. "don't call a tool for
    # this". This is a should-decline *example*, not a validation failure --
    # it is counted under "written", never "invalid".
    if expect.get("escalate") or expect.get("explain"):
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
        "answers": [{"name": operation, "arguments": args}],
    }


def build(corpus_path: Path, bundle_path: Path | None, tools: list) -> tuple[list[dict], dict]:
    """Build the full example list and counts from a corpus (and optional bundle).

    Returns ``(examples, counts)`` where *counts* has keys:

    - ``"written"`` -- examples in the output
    - ``"invalid"`` -- corpus entries skipped by RULE C (validation failure)
    - ``"skipped_records"`` -- bundle records skipped by RULE B checks
    - ``"duplicates"`` -- entries dropped by deduplication
    """
    if corpus_path.name == HELD_OUT_NAME:
        raise ValueError("refusing to train on the held-out split")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {"written": 0, "invalid": 0, "skipped_records": 0, "duplicates": 0}
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for entry in corpus.get("entries", []):
        example = example_from_entry(entry, tools)
        if example is None:
            counts["invalid"] += _is_invalid_entry(entry)
        else:
            _add(example, examples, seen, counts)

    for record in _bundle_records(bundle_path):
        example = example_from_record(record, tools)
        if example is None:
            counts["skipped_records"] += 1
        else:
            _add(example, examples, seen, counts)
    return examples, counts


def _is_invalid_entry(entry: dict) -> int:
    """1 for an explicit entry whose expected operation did not validate, else 0."""
    expect = entry.get("expect", {})
    named = bool(expect.get("operation")) and not expect.get("escalate")
    return int(entry.get("kind") == "explicit" and named)


def _bundle_records(bundle_path: Path | None) -> list[dict]:
    if bundle_path is None:
        return []
    records = json.loads(bundle_path.read_text(encoding="utf-8")).get("records", [])
    return [record for record in records if isinstance(record, dict)]


def _add(example: dict, examples: list[dict], seen: set, counts: dict[str, int]) -> None:
    """Append *example* unless the same request with the same answer is already in."""
    key = (example["query"].strip().lower(), json.dumps(example["answers"], sort_keys=True))
    if key in seen:
        counts["duplicates"] += 1
        return
    seen.add(key)
    examples.append(example)
    counts["written"] += 1


def _same_file(a: Path, b: Path) -> bool:
    """Whether *a* and *b* name the same file: same resolved path, or (for two
    existing paths) the same inode -- catches a symlink `os.path.resolve()`
    itself would already have followed, plus a hardlink, which it would not.
    """
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and a.samefile(b)
    except OSError:
        return False


def _refuse_if_out_aliases_input(out: Path, corpus: Path, bundle: Path | None) -> str | None:
    """``None`` if *out* is safe to truncate, else an error message.

    ``--out`` truncates its destination. Without this check, passing the
    same path (or an equivalent resolved path, or a symlink to it) as
    ``--corpus``/``--bundle`` and ``--out`` reads the source and then
    silently replaces it with generated JSONL.
    """
    if _same_file(out, corpus):
        return f"--out would overwrite --corpus: {out} and {corpus} are the same file"
    if bundle is not None and _same_file(out, bundle):
        return f"--out would overwrite --bundle: {out} and {bundle} are the same file"
    return None


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
    if args.corpus.name == HELD_OUT_NAME:
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

    alias_error = _refuse_if_out_aliases_input(args.out, args.corpus, bundle_path)
    if alias_error is not None:
        print(alias_error, file=sys.stderr)
        return 2

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
