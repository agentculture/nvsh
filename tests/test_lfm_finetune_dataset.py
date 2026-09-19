"""The Tier 2 (LFM2.5) fine-tune dataset builder: scripts/lfm-finetune/build_dataset.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nvsh.ops import table as ops_table
from nvsh.tiers import lfm
from nvsh.tiers.bench import (
    _context_for,
    _request_for,
    dev_corpus_path,
    held_out_corpus_path,
    load_corpus,
)

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


def test_a_failure_entrys_user_message_matches_what_the_tier_would_send():
    """4054701428: a "failure" entry's user message must be built the same
    way Tier 2 itself builds one at run time (command/exit status/output
    tail), not the corpus entry's raw text verbatim."""
    entries = load_corpus(dev_corpus_path()).entries
    failure_entry = next(entry for entry in entries if entry.kind == "failure")
    expected = lfm._request_message(_request_for(failure_entry), _context_for(failure_entry))

    examples = _module().build(dev_corpus_path())
    by_entry_id = dict(zip((entry.id for entry in entries), examples))
    actual_user_message = by_entry_id[failure_entry.id]["messages"][1]["content"]

    assert actual_user_message == expected


def test_the_held_out_split_is_refused():
    subject = _module()
    arg0 = held_out_corpus_path()
    with pytest.raises(ValueError, match="held-out"):
        subject.build(arg0)


def test_default_arguments_are_a_json_object():
    """4054701434: default shape -- what apply_chat_template documents."""
    examples = _module().build(dev_corpus_path())
    arguments = _calls(examples[0])["arguments"]
    assert isinstance(arguments, dict)


def test_arguments_as_string_opt_in_matches_the_wire_format():
    """4054701434: --arguments-as string -- the OpenAI wire shape Tier 2
    itself replays in its own chat history (nvsh.tiers.lfm's _record)."""
    module = _module()
    examples = module.build(dev_corpus_path(), module.ARGUMENTS_AS_STRING)
    arguments = _calls(examples[0])["arguments"]
    assert isinstance(arguments, str)


def test_arguments_as_string_is_valid_json_of_the_object_form():
    module = _module()
    object_form = _calls(module.build(dev_corpus_path())[0])["arguments"]
    string_form = _calls(module.build(dev_corpus_path(), module.ARGUMENTS_AS_STRING)[0])[
        "arguments"
    ]
    assert json.loads(string_form) == object_form


def test_main_writes_one_json_object_per_line(tmp_path):
    out = tmp_path / "train.jsonl"
    _module().main(["--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert all("messages" in json.loads(line) for line in lines)
