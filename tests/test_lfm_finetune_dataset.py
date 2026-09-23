"""The Tier 2 (LFM2.5) fine-tune dataset builder: scripts/lfm-finetune/build_dataset.py."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from nvsh.ops import table as ops_table
from nvsh.tiers import lfm
from nvsh.tiers.bench import (
    context_for,
    dev_corpus_path,
    held_out_corpus_path,
    load_corpus,
    load_world,
    request_for,
    world_platform,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/build_dataset.py"

#: The set of assistant tool calls a corpus can ask for (task t4, part of #39).
_CLOSED_TOOL_NAMES = {lfm.PROPOSE_TOOL, lfm.ESCALATE_TOOL, lfm.EXPLAIN_TOOL}

#: The Qwen3.5-0.8B tokenizer + chat template (no weights); revision pinned so a
#: cache refresh can't silently change what these tests measure (task t3, c7/h9).
_QWEN_BASE = "Qwen/Qwen3.5-0.8B"
_QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

#: The LFM2.5-350M tokenizer, same pinning reasoning (docs/lfm-finetune.md's own base+commit).
_LFM_BASE = "LiquidAI/LFM2.5-350M"
_LFM_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_build_dataset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cached_tokenizer(repo_id: str, revision: str):
    """*repo_id*'s tokenizer at *revision*, or a clean skip.

    ``local_files_only`` so this never touches the network; a cache miss (no
    training stack, or the model just isn't cached on this box) is a skip,
    never a failure (rule 4: real-tokenizer checks skip cleanly)."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            repo_id, revision=revision, local_files_only=True
        )
    except OSError:
        pytest.skip(f"{repo_id}@{revision} tokenizer not in the local Hugging Face cache")


#: A small, test-side reader for Qwen3.5's XML function/parameter tool-call
#: form (``<tool_call><function=NAME><parameter=ARG>VALUE</parameter>...``).
#: This is NOT the served vLLM parser (that round-trip is the serving smoke
#: task) -- just enough to check that build_dataset.py's rendered example
#: carries the tool name and arguments object it meant to write.
_QWEN_FUNCTION_RE = re.compile(r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>", re.DOTALL)
_QWEN_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>]+)>\n(?P<value>.*?)\n</parameter>", re.DOTALL
)


def _parse_qwen_call(rendered: str) -> tuple[str, dict]:
    match = _QWEN_FUNCTION_RE.search(rendered)
    if not match:
        raise ValueError(f"no <function=...> block in rendered text: {rendered!r}")
    arguments: dict = {}
    for param in _QWEN_PARAMETER_RE.finditer(match.group("body")):
        raw = param.group("value")
        try:
            arguments[param.group("name")] = json.loads(raw)
        except json.JSONDecodeError:
            arguments[param.group("name")] = raw
    return match.group("name"), arguments


def _parse_pythonic_call(rendered: str) -> tuple[str, dict]:
    """A small, test-side reader for LFM2.5's Pythonic tool-call form
    (``<|tool_call_start|>[name(kw=val, ...)]<|tool_call_end|>``): the body
    between the markers is a valid Python call expression, so ``ast`` reads
    it directly rather than hand-rolling a second parser."""
    match = re.search(r"<\|tool_call_start\|>(?P<body>.*?)<\|tool_call_end\|>", rendered, re.DOTALL)
    if not match:
        raise ValueError(f"no tool-call markers in rendered text: {rendered!r}")
    expr = ast.parse(match.group("body"), mode="eval").body
    (call,) = expr.elts
    arguments = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return call.func.id, arguments


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
    # The closed set of tool names a corpus can produce (task t4); dev.json
    # has all three since the explain entries landed (task t6).
    assert names == _CLOSED_TOOL_NAMES


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
    module = _module()
    with pytest.raises(ValueError, match="answer"):
        module.build(corpus)


def test_an_explain_entry_with_a_blank_answer_is_refused(tmp_path):
    corpus = _write_corpus(tmp_path, [_entry("exp01", {"explain": True, "answer": "   "})])
    module = _module()
    with pytest.raises(ValueError, match="answer"):
        module.build(corpus)


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
    # --no-verify-render: this checks the written shape, not the round-trip
    # guard (tested separately below), and the default venv has no
    # training stack to verify against (rule 4).
    _module().main(["--out", str(out), "--no-verify-render"])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert all("messages" in json.loads(line) for line in lines)


def _write_split(
    tmp_path: Path,
    entries: list[dict],
    name: str = "train.json",
    header: str = "Split 'train' of fixture.json (seed=39).",
) -> Path:
    """A split.py-shaped file: same corpus shape, every entry carries source_id."""
    path = tmp_path / name
    payload = {
        "header": header,
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
    module = _module()
    with pytest.raises(ValueError, match="held-out"):
        module.build(held_out)


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
    _module().main(["--split", str(split), "--out", str(out), "--no-verify-render"])
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
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--split", str(held_out), "--out", str(out)])


def test_cli_refuses_split_with_explicit_corpus(tmp_path):
    split = _write_split(tmp_path, [_entry("op01", {"operation": "thermal_stats", "args": {}})])
    out = tmp_path / "train.jsonl"
    module = _module()
    corpus_arg = str(dev_corpus_path())
    with pytest.raises(SystemExit):
        module.main(["--split", str(split), "--corpus", corpus_arg, "--out", str(out)])


def test_cli_refuses_a_val_split_file(tmp_path):
    """Bug repro: --split must be a TRAIN side; val must be refused too."""
    split = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="val.json",
        header="Split 'val' of fixture.json (seed=39).",
    )
    out = tmp_path / "train.jsonl"
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--split", str(split), "--out", str(out)])


def test_cli_refuses_a_test_split_file(tmp_path):
    """Bug repro: --split must be a TRAIN side; test must be refused too."""
    split = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="test.json",
        header="Split 'test' of fixture.json (seed=39).",
    )
    out = tmp_path / "train.jsonl"
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--split", str(split), "--out", str(out)])


def test_build_refuses_a_val_split_file_directly(tmp_path):
    split = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="val.json",
        header="Split 'val' of fixture.json (seed=39).",
    )
    module = _module()
    with pytest.raises(ValueError, match="'val'"):
        module.build(split, is_split=True)


def test_build_refuses_a_test_split_file_directly(tmp_path):
    split = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="test.json",
        header="Split 'test' of fixture.json (seed=39).",
    )
    module = _module()
    with pytest.raises(ValueError, match="'test'"):
        module.build(split, is_split=True)


def test_cli_refuses_a_split_file_renamed_from_test_whose_header_names_no_side(tmp_path):
    """Bug repro: a renamed test.json (header stripped of the split note) is
    still refused -- a --split file that does not positively identify
    itself as the train side cannot be trusted to be one."""
    split = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="renamed.json",
        header="Fixture corpus for build_dataset tests.",
    )
    out = tmp_path / "train.jsonl"
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--split", str(split), "--out", str(out)])


def test_build_refuses_a_split_file_whose_header_names_no_side_directly(tmp_path):
    split = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="renamed.json",
        header="Fixture corpus for build_dataset tests.",
    )
    module = _module()
    with pytest.raises(ValueError, match="names no side"):
        module.build(split, is_split=True)


def test_a_split_file_names_the_train_side_and_is_accepted(tmp_path):
    """The passing case: a genuine train-side split file, is_split=True, builds fine."""
    split = _write_split(tmp_path, [_entry("op01", {"operation": "thermal_stats", "args": {}})])
    examples = _module().build(split, is_split=True)
    assert len(examples) == 1


def test_a_held_out_split_renamed_is_still_refused_by_header(tmp_path):
    """The held-out corpus's own header is refused even under another filename."""
    module = _module()
    held_out_header = module._header(held_out_corpus_path())
    split = _write_split(
        tmp_path,
        [_entry("op01", {"operation": "thermal_stats", "args": {}})],
        name="renamed-held-out.json",
        header=held_out_header,
    )
    with pytest.raises(ValueError, match="held-out"):
        module.build(split, is_split=True)


def test_an_explicit_long_prompt_is_clamped_like_runtime(tmp_path):
    module = _module()
    long_prompt = "x" * 10_000
    entry = _entry("long01", {"escalate": True})
    entry["text"] = long_prompt
    corpus = _write_corpus(tmp_path, [entry])
    examples = module.build(corpus)
    user_message = examples[0]["messages"][1]["content"]
    assert len(user_message) < len(long_prompt)
    assert user_message == lfm.request_message(
        request_for(load_corpus(corpus).entries[0]), context_for(load_corpus(corpus).entries[0])
    )


def test_an_explicit_prompt_with_a_token_is_redacted_like_runtime(tmp_path):
    module = _module()
    entry = _entry("secret01", {"escalate": True})
    entry["text"] = "please HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz help me"
    corpus = _write_corpus(tmp_path, [entry])
    examples = module.build(corpus)
    user_message = examples[0]["messages"][1]["content"]
    loaded_entry = load_corpus(corpus).entries[0]
    assert user_message == lfm.request_message(request_for(loaded_entry), context_for(loaded_entry))
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in user_message


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


def test_a_written_proposal_names_the_operation_before_its_arguments(tmp_path):
    # The chat template renders tool-call arguments in the order they are
    # stored. Sorted keys put "arguments" before "operation", which taught
    # r4 to write the arguments and then stop without choosing an operation.
    out = tmp_path / "train.jsonl"
    _module().main(["--corpus", str(dev_corpus_path()), "--out", str(out), "--no-verify-render"])
    for line in out.read_text(encoding="utf-8").splitlines():
        call = json.loads(line)["messages"][-1]["tool_calls"][0]["function"]
        if call["name"] == lfm.PROPOSE_TOOL:
            assert list(call["arguments"]) == ["operation", "arguments"]


# ---------------------------------------------------------------------------
# t3 (c7/h9): render one example per outcome with the base tokenizer and
# parse the assistant span back to the same tool name and arguments object.
# ---------------------------------------------------------------------------


class _FakeTemplateTokenizer:
    """Stands in for apply_chat_template without any real chat template.

    Renders each message as ``<role>content`` and, for a trailing assistant
    tool-call message, appends ``<assistant>name|json-arguments``. Enough to
    exercise ``render_assistant_text``'s prefix-diff logic (and
    ``verify_round_trip``'s pass/fail paths) without the training stack, so
    that logic is covered even when no real tokenizer is cached."""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        trailing_assistant = messages and messages[-1]["role"] == "assistant"
        leading = messages[:-1] if trailing_assistant and not add_generation_prompt else messages
        rendered = "".join(f"<{m['role']}>{m.get('content', '')}" for m in leading)
        if add_generation_prompt:
            return rendered + "<assistant>"
        if trailing_assistant:
            call = messages[-1]["tool_calls"][0]["function"]
            rendered += f"<assistant>{call['name']}|{json.dumps(call['arguments'])}"
        return rendered


def _parse_fake_call(rendered: str) -> tuple[str, dict]:
    name, _, raw_arguments = rendered.partition("|")
    return name, json.loads(raw_arguments)


def _outcome_examples(module) -> dict[str, dict]:
    """One example each for propose (with args), escalate and explain."""
    entries = load_corpus(dev_corpus_path()).entries
    platform = world_platform(load_world(dev_corpus_path()))
    propose_entry = next(e for e in entries if e.expect.get("args"))
    escalate_entry = next(e for e in entries if e.expect.get("escalate"))
    explain_entry = next(e for e in entries if e.expect.get("explain"))
    return {
        "propose": module.example_from_entry(propose_entry, platform),
        "escalate": module.example_from_entry(escalate_entry, platform),
        "explain": module.example_from_entry(explain_entry, platform),
    }


def test_render_assistant_text_isolates_the_assistant_span_fake_tokenizer():
    module = _module()
    example = _outcome_examples(module)["escalate"]
    span = module.render_assistant_text(_FakeTemplateTokenizer(), example)
    name, arguments = _parse_fake_call(span)
    assert name == lfm.ESCALATE_TOOL
    assert arguments == {"reason": module.ESCALATE_REASON}


def test_verify_round_trip_passes_for_a_correct_render_fake_tokenizer():
    module = _module()
    for example in _outcome_examples(module).values():
        module.verify_round_trip(example, _FakeTemplateTokenizer(), _parse_fake_call)


def test_verify_round_trip_raises_on_a_mismatched_parse():
    module = _module()
    example = _outcome_examples(module)["escalate"]

    def _wrong_parse(rendered: str) -> tuple[str, dict]:
        return "not-escalate", {}

    with pytest.raises(ValueError, match="round-trip"):
        module.verify_round_trip(example, _FakeTemplateTokenizer(), _wrong_parse)


@pytest.mark.parametrize("outcome", ["propose", "escalate", "explain"])
def test_qwen_tokenizer_round_trips_each_outcome(outcome):
    """c7/h9: one rendered example per outcome, real Qwen tokenizer, parsed
    back through the small XML reader to the same tool name and arguments."""
    module = _module()
    tokenizer = _cached_tokenizer(_QWEN_BASE, _QWEN_REVISION)
    example = _outcome_examples(module)[outcome]
    module.verify_round_trip(example, tokenizer, _parse_qwen_call)


@pytest.mark.parametrize("outcome", ["propose", "escalate", "explain"])
def test_lfm_tokenizer_round_trips_each_outcome(outcome):
    """LFM's own workarounds (object arguments, operation before arguments)
    are verified against the real tokenizer here too, not just assumed."""
    module = _module()
    tokenizer = _cached_tokenizer(_LFM_BASE, _LFM_REVISION)
    example = _outcome_examples(module)[outcome]
    module.verify_round_trip(example, tokenizer, _parse_pythonic_call)


def test_qwen_tokenizer_refuses_string_arguments_like_lfm():
    """The 'object arguments' workaround is LFM-only by assumption today;
    verify (not assume) Qwen's own template needs it too -- a mapping to
    iterate as named <parameter> tags, not a JSON-encoded string."""
    module = _module()
    tokenizer = _cached_tokenizer(_QWEN_BASE, _QWEN_REVISION)
    entries = load_corpus(dev_corpus_path()).entries
    platform = world_platform(load_world(dev_corpus_path()))
    propose_entry = next(e for e in entries if e.expect.get("args"))
    example = module.example_from_entry(propose_entry, platform, module.ARGUMENTS_AS_STRING)
    with pytest.raises(Exception):
        tokenizer.apply_chat_template(example["messages"], tools=example["tools"], tokenize=False)


def test_lfm_tokenizer_refuses_string_arguments():
    """The existing LFM restriction (docs/lfm-finetune.md), verified live
    against the real tokenizer rather than only documented."""
    module = _module()
    tokenizer = _cached_tokenizer(_LFM_BASE, _LFM_REVISION)
    entries = load_corpus(dev_corpus_path()).entries
    platform = world_platform(load_world(dev_corpus_path()))
    propose_entry = next(e for e in entries if e.expect.get("args"))
    example = module.example_from_entry(propose_entry, platform, module.ARGUMENTS_AS_STRING)
    with pytest.raises(Exception):
        tokenizer.apply_chat_template(example["messages"], tools=example["tools"], tokenize=False)


def test_qwen_tokenizer_operation_before_arguments_default_order_round_trips():
    """The 'operation before arguments' workaround (dict insertion order) is
    verified for Qwen too: its named <parameter> tags render in dict order,
    and the round trip still recovers the same operation and arguments even
    though a named XML parameter (unlike LFM's positional Pythonic call)
    does not actually depend on that order to parse correctly."""
    module = _module()
    tokenizer = _cached_tokenizer(_QWEN_BASE, _QWEN_REVISION)
    entries = load_corpus(dev_corpus_path()).entries
    platform = world_platform(load_world(dev_corpus_path()))
    propose_entry = next(e for e in entries if e.expect.get("args"))
    example = module.example_from_entry(propose_entry, platform)
    call = example["messages"][-1]["tool_calls"][0]["function"]
    assert list(call["arguments"]) == ["operation", "arguments"]
    module.verify_round_trip(example, tokenizer, _parse_qwen_call)


# ---------------------------------------------------------------------------
# Codex review finding #6: verify_round_trip must guard build() itself, not
# only be callable by tests.
# ---------------------------------------------------------------------------


class _MismatchingPythonicTokenizer:
    """Renders a fixed, wrong Pythonic call for every example's assistant
    turn -- a round-trip mismatch build()'s own verification must catch."""

    def apply_chat_template(
        self, messages, tools=None, tokenize=False, add_generation_prompt=False
    ):
        trailing_assistant = messages and messages[-1]["role"] == "assistant"
        leading = messages[:-1] if trailing_assistant and not add_generation_prompt else messages
        rendered = "".join(f"<{m['role']}>{m.get('content', '')}" for m in leading)
        if add_generation_prompt:
            return rendered + "<assistant>"
        if trailing_assistant:
            rendered += "<assistant><|tool_call_start|>[not_the_right_tool()]<|tool_call_end|>"
        return rendered


def test_build_fails_when_the_tokenizer_render_mismatches():
    """Reproduces finding #6: before this fix, verify_round_trip was never
    called from build() at all, so this mismatch shipped silently."""
    module = _module()
    with pytest.raises(ValueError, match="round-trip"):
        module.build(
            dev_corpus_path(),
            verify_render=True,
            tokenizer=_MismatchingPythonicTokenizer(),
        )


def test_build_does_not_verify_by_default():
    """build()'s own default stays verification-off, so plain library calls
    (most of this file's other tests) never need a tokenizer -- the CLI
    turns verification on by default instead (tested below)."""
    module = _module()
    examples = module.build(dev_corpus_path())
    assert len(examples) == len(load_corpus(dev_corpus_path()).entries)


def test_build_verifies_against_the_default_base_with_a_real_tokenizer():
    """The passing counterpart: build()'s own default base (LFM2.5, matching
    train.py's DEFAULT_BASE/DEFAULT_REVISION) round-trips a normal build."""
    module = _module()
    assert module.DEFAULT_BASE == _LFM_BASE
    assert module.DEFAULT_REVISION == _LFM_REVISION
    tokenizer = _cached_tokenizer(_LFM_BASE, _LFM_REVISION)
    examples = module.build(dev_corpus_path(), verify_render=True, tokenizer=tokenizer)
    assert len(examples) == len(load_corpus(dev_corpus_path()).entries)


def test_qwen_build_with_string_arguments_fails_the_round_trip_guard():
    """Finding #6's own trigger: build with --arguments-as string for Qwen
    must fail, not just the standalone apply_chat_template call that
    test_qwen_tokenizer_refuses_string_arguments_like_lfm already shows."""
    module = _module()
    tokenizer = _cached_tokenizer(_QWEN_BASE, _QWEN_REVISION)
    with pytest.raises(Exception):
        module.build(
            dev_corpus_path(),
            module.ARGUMENTS_AS_STRING,
            verify_render=True,
            base=_QWEN_BASE,
            revision=_QWEN_REVISION,
            tokenizer=tokenizer,
        )


def test_parse_rendered_call_detects_qwen_form():
    module = _module()
    rendered = '<tool_call><function=escalate><parameter=reason>\n"x"\n</parameter></function>'
    name, arguments = module.parse_rendered_call(rendered)
    assert name == "escalate"
    assert arguments == {"reason": "x"}


def test_parse_rendered_call_detects_pythonic_form():
    module = _module()
    rendered = "<|tool_call_start|>[escalate(reason='x')]<|tool_call_end|>"
    name, arguments = module.parse_rendered_call(rendered)
    assert name == "escalate"
    assert arguments == {"reason": "x"}


def test_parse_rendered_call_refuses_an_unrecognized_form():
    module = _module()
    with pytest.raises(ValueError, match="unrecognized"):
        module.parse_rendered_call("<assistant>escalate|{}")


def test_cli_verifies_render_by_default(tmp_path, monkeypatch):
    """The CLI wires verify_render=True into build() unless told not to."""
    module = _module()
    captured = {}

    def _fake_build(source, arguments_as=module.ARGUMENTS_AS_OBJECT, is_split=False, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(module, "build", _fake_build)
    out = tmp_path / "train.jsonl"
    module.main(["--out", str(out)])
    assert captured["verify_render"] is True
    assert captured["base"] == module.DEFAULT_BASE
    assert captured["revision"] == module.DEFAULT_REVISION


def test_cli_no_verify_render_flag_turns_verification_off(tmp_path, monkeypatch):
    module = _module()
    captured = {}

    def _fake_build(source, arguments_as=module.ARGUMENTS_AS_OBJECT, is_split=False, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(module, "build", _fake_build)
    out = tmp_path / "train.jsonl"
    module.main(["--out", str(out), "--no-verify-render"])
    assert captured["verify_render"] is False


def test_cli_base_and_revision_flags_pass_through(tmp_path, monkeypatch):
    module = _module()
    captured = {}

    def _fake_build(source, arguments_as=module.ARGUMENTS_AS_OBJECT, is_split=False, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(module, "build", _fake_build)
    out = tmp_path / "train.jsonl"
    module.main(["--out", str(out), "--base", _QWEN_BASE, "--revision", _QWEN_REVISION])
    assert captured["base"] == _QWEN_BASE
    assert captured["revision"] == _QWEN_REVISION
