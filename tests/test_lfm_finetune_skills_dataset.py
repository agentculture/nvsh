"""scripts/lfm-finetune/skills_dataset.py (issue 39): skill variations -> chat examples."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "skills_dataset.py"

_TOOLS = [
    {
        "repo": "device",
        "skill": "jetson-diagnostic",
        "tool": {
            "type": "function",
            "function": {
                "name": "jetson_diagnostic",
                "description": "Read-only Jetson health snapshot.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
    {
        "repo": "bsp",
        "skill": "jetson-customize-fan",
        "tool": {
            "type": "function",
            "function": {
                "name": "jetson_customize_fan",
                "description": "Customize the fan profile in the BSP.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
]

_EVAL = {
    "id": "jetson-diagnostic-001",
    "repo": "device",
    "skill": "jetson-diagnostic",
    "text": "What is this Jetson? Tell me the SKU, how much memory it has, and what's using it.",
    "expected_skill": "jetson-diagnostic",
    "ground_truth": None,
    "names_skill": False,
}


def _module():
    spec = importlib.util.spec_from_file_location("skills_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, accepted: list[dict]) -> tuple[Path, Path, Path]:
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(_TOOLS))
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps(_EVAL) + "\n")
    acc = tmp_path / "accepted.jsonl"
    acc.write_text("".join(json.dumps(r) + "\n" for r in accepted))
    return acc, tools, test


def _variation(text: str, skill: str = "jetson-customize-fan", side: str = "train") -> dict:
    return {
        "id": f"{skill}~v1",
        "source_id": skill,
        "side": side,
        "text": text,
        "expect": {"skill": skill},
        "seed_format": "skills",
    }


def test_an_example_matches_the_measurement_request(tmp_path) -> None:
    acc, tools, test = _write(tmp_path, [_variation("Make the fan quieter on my custom carrier")])
    (example,) = _module().build(acc, tools, test)
    assert example["messages"][0] == {
        "role": "user",
        "content": "Make the fan quieter on my custom carrier",
    }
    assert [t["function"]["name"] for t in example["tools"]] == [
        "jetson_diagnostic",
        "jetson_customize_fan",
    ]
    call = example["messages"][1]["tool_calls"][0]["function"]
    assert call == {"name": "jetson_customize_fan", "arguments": {}}


def test_a_non_train_variation_is_refused(tmp_path) -> None:
    acc, tools, test = _write(tmp_path, [_variation("Make the fan quieter", side="test")])
    module = _module()
    with pytest.raises(ValueError, match="not train"):
        module.build(acc, tools, test)


def test_a_variation_that_paraphrases_an_eval_fails_the_build(tmp_path) -> None:
    text = "What Jetson is this? Tell me its SKU, how much memory it has, and what is using it."
    acc, tools, test = _write(tmp_path, [_variation(text, skill="jetson-diagnostic")])
    module = _module()
    with pytest.raises(ValueError, match="contamination"):
        module.build(acc, tools, test)


def test_an_unknown_skill_is_refused(tmp_path) -> None:
    acc, tools, test = _write(tmp_path, [_variation("Flash it", skill="jetson-nope")])
    module = _module()
    with pytest.raises(ValueError, match="unknown skill"):
        module.build(acc, tools, test)
