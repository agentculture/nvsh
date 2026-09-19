"""The Tier 2 (LFM2.5) fine-tune dataset builder: scripts/lfm-finetune/build_dataset.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nvsh.ops import table as ops_table
from nvsh.tiers import lfm
from nvsh.tiers.bench import dev_corpus_path, held_out_corpus_path, load_corpus

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/build_dataset.py"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_build_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calls(example: dict) -> dict:
    return example["messages"][-1]["tool_calls"][0]["function"]


def test_every_dev_entry_becomes_one_example():
    examples = _module().build(dev_corpus_path())
    assert len(examples) == len(load_corpus(dev_corpus_path()).entries)


def test_examples_carry_the_tools_tier_two_really_sends():
    examples = _module().build(dev_corpus_path())
    assert examples[0]["tools"] == lfm.tools_for()


def test_an_explicit_request_is_answered_by_propose_with_a_table_operation():
    examples = _module().build(dev_corpus_path())
    proposed = [_calls(e) for e in examples if _calls(e)["name"] == lfm.PROPOSE_TOOL]
    assert {call["arguments"]["operation"] for call in proposed} <= set(ops_table.names())


def test_a_should_decline_request_is_answered_by_escalate():
    examples = _module().build(dev_corpus_path())
    names = {_calls(e)["name"] for e in examples}
    assert names == {lfm.PROPOSE_TOOL, lfm.ESCALATE_TOOL}


def test_the_held_out_split_is_refused():
    with pytest.raises(ValueError, match="held-out"):
        _module().build(held_out_corpus_path())


def test_main_writes_one_json_object_per_line(tmp_path):
    out = tmp_path / "train.jsonl"
    _module().main(["--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert all("messages" in json.loads(line) for line in lines)
