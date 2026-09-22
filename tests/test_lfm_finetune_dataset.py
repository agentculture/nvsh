"""The Tier 2 (LFM2.5) fine-tune dataset builder: scripts/lfm-finetune/build_dataset.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nvsh.ops import table as ops_table
from nvsh.tiers import lfm
from nvsh.tiers.bench import (
    context_for,
    dev_corpus_path,
    held_out_corpus_path,
    load_corpus,
    request_for,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/build_dataset.py"

#: The set of assistant tool calls a corpus can ask for (task t4, part of #39).
_CLOSED_TOOL_NAMES = {lfm.PROPOSE_TOOL, lfm.ESCALATE_TOOL, lfm.EXPLAIN_TOOL}


def _module():
    spec = importlib.util.spec_from_file_location("lfm_build_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calls(example: dict) -> dict:
    return example["messages"][-1]["tool_calls"][0]["function"]


def _entry(entry_id: str, expect: dict, kind: str = "explicit", source: str = "fixture") -> dict:
    return {
        "id": entry_id,
        "kind": kind,
        "text": f"fixture text for {entry_id}",
        "expect": expect,
        "source": source,
    }


def _write_corpus(tmp_path: Path, entries: list[dict], name: str = "fixture.json") -> Path:
    path = tmp_path / name
    payload = {"header": "Fixture corpus for build_dataset tests.", "entries": entries}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
    # dev.json has no "explain" entries yet, but the closed set of tool
    # names a corpus can produce is {propose, escalate, explain} (task t4).
    assert names <= _CLOSED_TOOL_NAMES
    assert names == {lfm.PROPOSE_TOOL, lfm.ESCALATE_TOOL}


def test_an_explain_entry_is_answered_by_explain_with_its_answer_text(tmp_path):
    corpus = _write_corpus(
        tmp_path, [_entry("exp01", {"explain": True, "answer": "it means the fan is loud"})]
    )
    examples = _module().build(corpus)
    call = _calls(examples[0])
    assert call["name"] == lfm.EXPLAIN_TOOL
    assert call["arguments"]["text"] == "it means the fan is loud"


def test_an_explain_entry_without_an_answer_is_refused(tmp_path):
    corpus = _write_corpus(tmp_path, [_entry("exp01", {"explain": True})])
    with pytest.raises(ValueError, match="answer"):
        _module().build(corpus)


def test_an_explain_entry_with_a_blank_answer_is_refused(tmp_path):
    corpus = _write_corpus(tmp_path, [_entry("exp01", {"explain": True, "answer": "   "})])
    with pytest.raises(ValueError, match="answer"):
        _module().build(corpus)


def test_answer_for_keeps_propose_and_escalate_unchanged():
    """t4 acceptance: answer_for's propose/escalate branches are untouched."""
    entries = load_corpus(dev_corpus_path()).entries
    module = _module()
    propose_entry = next(e for e in entries if e.expect.get("operation"))
    escalate_entry = next(e for e in entries if e.expect.get("escalate"))
    propose_call = module.answer_for(propose_entry)["tool_calls"][0]["function"]
    escalate_call = module.answer_for(escalate_entry)["tool_calls"][0]["function"]
    assert propose_call["name"] == lfm.PROPOSE_TOOL
    assert propose_call["arguments"]["operation"] == propose_entry.expect["operation"]
    assert escalate_call["name"] == lfm.ESCALATE_TOOL


def test_a_failure_entrys_user_message_matches_what_the_tier_would_send():
    """4054701428: a "failure" entry's user message must be built the same
    way Tier 2 itself builds one at run time (command/exit status/output
    tail), not the corpus entry's raw text verbatim."""
    entries = load_corpus(dev_corpus_path()).entries
    failure_entry = next(entry for entry in entries if entry.kind == "failure")
    expected = lfm.request_message(request_for(failure_entry), context_for(failure_entry))

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


def _write_split(tmp_path: Path, entries: list[dict], name: str = "train.json") -> Path:
    """A split.py-shaped file: same corpus shape, every entry carries source_id."""
    path = tmp_path / name
    payload = {
        "header": "Split 'train' of fixture.json (seed=39).",
        "entries": [
            {"source_id": entry["id"], **entry} if "source_id" not in entry else dict(entry)
            for entry in entries
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_split_file_builds_the_same_shape_as_a_corpus(tmp_path):
    split = _write_split(tmp_path, [_entry("op01", {"operation": "thermal_stats", "args": {}})])
    examples = _module().build(split)
    assert len(examples) == 1
    assert _calls(examples[0])["name"] == lfm.PROPOSE_TOOL


def test_a_split_examples_carry_their_source_id(tmp_path):
    entries = [
        {**_entry("var01", {"operation": "thermal_stats", "args": {}}), "source_id": "op01"},
        _entry("esc01", {"escalate": True}),
    ]
    split = _write_split(tmp_path, entries)
    examples = _module().build(split)
    by_id = {e["messages"][1]["content"]: e for e in examples}
    assert by_id["fixture text for var01"]["source_id"] == "op01"
    # an entry with no recorded source_id falls back to its own id, like split.py.
    assert by_id["fixture text for esc01"]["source_id"] == "esc01"


def test_the_held_out_split_is_refused_even_as_a_split_file(tmp_path):
    held_out = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="held-out.json",
    )
    with pytest.raises(ValueError, match="held-out"):
        _module().build(held_out)


def test_cli_split_never_reads_dev_json_whole(tmp_path, monkeypatch):
    """--split builds straight from the split file; dev.json is never opened."""
    split = _write_split(tmp_path, [_entry("op01", {"operation": "thermal_stats", "args": {}})])
    out = tmp_path / "train.jsonl"

    real_open = open

    def _guarded_open(path, *args, **kwargs):
        if str(path).endswith("dev.json"):
            raise AssertionError("dev.json must not be read when --split is given")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _guarded_open)
    _module().main(["--split", str(split), "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["source_id"] == "op01"


def test_cli_refuses_split_named_held_out(tmp_path):
    held_out = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="held-out.json",
    )
    out = tmp_path / "train.jsonl"
    with pytest.raises(SystemExit):
        _module().main(["--split", str(held_out), "--out", str(out)])


def test_cli_refuses_split_with_explicit_corpus(tmp_path):
    split = _write_split(tmp_path, [_entry("op01", {"operation": "thermal_stats", "args": {}})])
    out = tmp_path / "train.jsonl"
    with pytest.raises(SystemExit):
        _module().main(
            ["--split", str(split), "--corpus", str(dev_corpus_path()), "--out", str(out)]
        )


def test_a_split_built_example_matches_the_corpus_built_one(tmp_path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lfm_split_for_builder",
        Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/split.py",
    )
    split = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split)
    split.main(["--corpus", str(dev_corpus_path()), "--out-dir", str(tmp_path)])
    from_split = {ex["source_id"]: ex for ex in _module().build(tmp_path / "train.json")}
    from_corpus = {ex["source_id"]: ex for ex in _module().build(dev_corpus_path())}
    assert from_split
    for source_id, example in from_split.items():
        assert example["messages"] == from_corpus[source_id]["messages"]
