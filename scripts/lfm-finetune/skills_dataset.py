"""Turn accepted skill-seed variations into chat training examples (issue 39).

The method-validation bar trains LFM2.5-350M to route a Jetson request to one
of NVIDIA's JetPack 7.2 agent skills. ``augment.py`` (skill-seed mode) writes
the requests, generated from each skill's SKILL.md description only; this
script turns the accepted ones into the same request ``measure_skills.py``
sends -- a lone user message with every skill tool, in ``tools.json`` order --
answered by one call to the expected skill.

Before anything is written it runs ``jetson_skills.py``'s contamination scan
against NVIDIA's 104 evals: a request that copies or lightly paraphrases an
eval makes the build fail, because those evals are the test set.

    python scripts/lfm-finetune/skills_dataset.py --accepted accepted.jsonl \
        --tools tools.json --test test.jsonl --out skills-train.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _jetson_skills():
    spec = importlib.util.spec_from_file_location("jetson_skills", _HERE / "jetson_skills.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("jetson_skills", module)
    spec.loader.exec_module(module)
    return module


def load_tools(path: Path) -> tuple[list[dict], dict[str, str]]:
    """The tool list in ``tools.json`` order, and skill name -> tool function name."""
    records = json.loads(path.read_text(encoding="utf-8"))
    tools = [record["tool"] for record in records]
    names = {record["skill"]: record["tool"]["function"]["name"] for record in records}
    return tools, names


def example_for(record: dict, tools: list[dict], names: dict[str, str]) -> dict:
    """One training example: the request, answered by a call to its skill."""
    if record.get("seed_format") != "skills":
        raise ValueError(f"{record.get('id')}: not a skill-seed variation")
    if record.get("side") != "train":
        raise ValueError(f"{record.get('id')}: side {record.get('side')!r} is not train")
    skill = (record.get("expect") or {}).get("skill")
    if skill not in names:
        raise ValueError(f"{record.get('id')}: unknown skill {skill!r}")
    call = {"type": "function", "function": {"name": names[skill], "arguments": {}}}
    return {
        "messages": [
            {"role": "user", "content": record["text"]},
            {"role": "assistant", "content": "", "tool_calls": [call]},
        ],
        "tools": tools,
        "source_id": record["source_id"],
    }


def build(accepted: Path, tools_path: Path, test_path: Path) -> list[dict]:
    """Every example, after the contamination scan passes. Raises on a hit."""
    tools, names = load_tools(tools_path)
    records = [
        json.loads(line)
        for line in accepted.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = [example_for(record, tools, names) for record in records]
    js = _jetson_skills()
    evals = [
        js.EvalRecord(
            id=raw["id"],
            repo=raw["repo"],
            skill=raw["skill"],
            text=raw["text"],
            expected_skill=raw["expected_skill"],
            ground_truth=raw.get("ground_truth"),
            names_skill=raw.get("names_skill", False),
        )
        for raw in (
            json.loads(line)
            for line in test_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    hits = js.scan_contamination(evals, [record["text"] for record in records])
    if hits:
        lines = [f"{hit.eval_id} [{hit.field}]: {hit.reason}" for hit in hits]
        raise ValueError("contamination scan failed:\n  " + "\n  ".join(lines))
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--tools", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path, help="NVIDIA evals (test.jsonl)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        examples = build(args.accepted, args.tools, args.test)
    except ValueError as exc:
        parser.error(str(exc))
    with open(args.out, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"written={len(examples)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
