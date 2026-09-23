#!/usr/bin/env python3
"""Try operation descriptions against the real Needle3 engine, before editing the table.

A development-machine tool: it is never imported by the nvsh package.

Needle3 is shown nvsh's operation table as JSON tool schemas, and picks from
what those schemas *say*. So the cheapest way to move Tier 1's accuracy is to
reword a description -- and this script is how to find out whether a
rewording helps without touching ``nvsh/ops/table.py``:

    python scripts/needle-finetune/try_table.py                       # the table as it ships
    python scripts/needle-finetune/try_table.py --overrides new.json  # with rewordings

``new.json`` maps an operation name to a replacement description::

    {"gpu_stats": "Show GPU usage: utilisation and GPU memory.",
     "thermal_stats": "Show temperatures: how hot the machine is."}

It runs the explicit entries of the dev corpus through the same code the
worker child uses (``build_engine`` + ``extract_selection`` + ``decide``), in
this process, and prints one line per request plus the score. It needs
``cactus-needle`` importable and the pinned files fetched
(``nvsh tiers prefetch``). It never reads the held-out split.

Only the *pick* is scored here (operation and raw arguments). Grounding and
rendering are deterministic and are measured by ``nvsh tiers bench``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nvsh.ops import OPERATIONS  # noqa: E402
from nvsh.tiers import needle_home  # noqa: E402
from nvsh.tiers.base import TierDecision, decide  # noqa: E402
from nvsh.tiers.needle_worker import EngineSpec, build_engine, extract_selection  # noqa: E402

DEV_CORPUS = REPO_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json"


def load_entries(path: Path) -> list[dict]:
    if path.name == "held-out.json":
        raise SystemExit("refusing to tune descriptions against the held-out split")
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)["entries"]
    return [entry for entry in entries if entry.get("kind") == "explicit"]


def reworded(overrides: dict[str, str]) -> list:
    unknown = sorted(set(overrides) - {operation.name for operation in OPERATIONS})
    if unknown:
        raise SystemExit(f"overrides name operations that are not in the table: {unknown}")
    return [
        dataclasses.replace(
            operation, description=overrides.get(operation.name, operation.description)
        )
        for operation in OPERATIONS
    ]


def engine_spec() -> EngineSpec:
    staged = needle_home.stage()
    if not isinstance(staged, needle_home.NeedleHome):
        raise SystemExit(
            f"tier files are not ready ({staged.message}); run: nvsh tiers prefetch --yes"
        )
    return EngineSpec(lib=str(staged.lib), weights=str(staged.weights), home=str(staged.home))


def verdict(entry: dict, picked) -> bool:
    """Right pick, or a decline where the corpus expects a should-decline.

    Tier 1 (Needle3) has no explain capability -- only Tier 2 can inspect,
    propose, explain or escalate -- so an ``explain`` expectation is a
    should-decline for Tier 1 scoring, exactly like ``escalate``
    (``nvsh/tiers/bench.py``'s ``_expects_decline``,
    ``scripts/needle-finetune/build_dataset.py``'s dataset builder). A
    proposal on either kind of should-decline entry is a wrong answer, never
    a crash: this is why ``expect["operation"]`` below only runs once both
    should-decline kinds have already returned.
    """
    expect = entry["expect"]
    if expect.get("escalate") or expect.get("explain"):
        return not isinstance(picked, TierDecision)
    return isinstance(picked, TierDecision) and picked.operation == expect["operation"]


def score(engine, entry: dict) -> dict:
    """One request through the engine: what was wanted, what came back, how long it took."""
    started = time.monotonic()
    engine.reset()
    calls, confidence = extract_selection(engine.complete(entry["text"]))
    picked = decide(calls, confidence)
    chose = isinstance(picked, TierDecision)
    return {
        "id": entry["id"],
        "text": entry["text"],
        "want": entry["expect"].get("operation", "ESCALATE"),
        "got": picked.operation if chose else picked.reason.value,
        "args": picked.args if chose else {},
        "confidence": confidence,
        "ms": round((time.monotonic() - started) * 1000),
        "ok": verdict(entry, picked),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEV_CORPUS)
    parser.add_argument("--overrides", type=Path, help="JSON: operation name -> new description")
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)

    overrides = json.loads(args.overrides.read_text(encoding="utf-8")) if args.overrides else {}
    engine = build_engine(engine_spec(), operations=reworded(overrides))
    rows = [score(engine, entry) for entry in load_entries(args.corpus)]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            mark = "ok  " if row["ok"] else "MISS"
            text, want, got = row["text"][:46], row["want"], row["got"]
            print(f"{mark} {text:46} want={want:18} got={got} {row['args'] or ''}")
    picks = [row for row in rows if row["want"] != "ESCALATE"]
    declines = [row for row in rows if row["want"] == "ESCALATE"]
    print(
        f"correct picks {sum(r['ok'] for r in picks)}/{len(picks)}   "
        f"correct declines {sum(r['ok'] for r in declines)}/{len(declines)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
