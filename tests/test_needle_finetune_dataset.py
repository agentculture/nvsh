"""Tests for scripts/needle-finetune/build_dataset.py (dataset builder).

Uses importlib.util to load the script from disk so the hyphenated directory
name does not interfere with normal import.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nvsh.ops import validate
from nvsh.tiers.needle_worker import tool_schemas

# ---------------------------------------------------------------------------
# Load the module under test
# ---------------------------------------------------------------------------

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "needle-finetune" / "build_dataset.py"
_spec = importlib.util.spec_from_file_location("needle_finetune_build_dataset", _PATH)
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)

example_from_entry = _builder.example_from_entry
example_from_record = _builder.example_from_record
build = _builder.build
main = _builder.main

# The tools list is the canonical set of 16 operation schemas.
tools: list = tool_schemas()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_corpus(entries: list[dict]) -> dict:
    return {"header": "test", "entries": entries}


def _tiny_bundle(records: list[dict]) -> dict:
    return {"records": records}


# ---------------------------------------------------------------------------
# test_an_explicit_entry_becomes_one_answer
# ---------------------------------------------------------------------------


def test_an_explicit_entry_becomes_one_answer() -> None:
    """disk_stats with no args → answers == [{"name": "disk_stats", "arguments": {}}]."""
    entry: dict = {
        "id": "t01",
        "kind": "explicit",
        "text": "am I out of disk?",
        "expect": {"operation": "disk_stats", "args": {}},
        "source": "test",
    }
    result = example_from_entry(entry, tools)
    assert result is not None
    assert result["query"] == "am I out of disk?"
    assert result["answers"] == [{"name": "disk_stats", "arguments": {}}]


# ---------------------------------------------------------------------------
# test_an_entry_with_arguments_keeps_them
# ---------------------------------------------------------------------------


def test_an_entry_with_arguments_keeps_them() -> None:
    """service_restart with args → arguments dict preserved."""
    entry: dict = {
        "id": "t02",
        "kind": "explicit",
        "text": "Restart vLLM",
        "expect": {"operation": "service_restart", "args": {"service": "vllm.service"}},
        "source": "test",
    }
    result = example_from_entry(entry, tools)
    assert result is not None
    assert result["answers"] == [
        {"name": "service_restart", "arguments": {"service": "vllm.service"}}
    ]


# ---------------------------------------------------------------------------
# test_an_escalation_entry_has_no_answers
# ---------------------------------------------------------------------------


def test_an_escalation_entry_has_no_answers() -> None:
    """escalate → answers is []."""
    entry: dict = {
        "id": "t03",
        "kind": "explicit",
        "text": "Why did vLLM crash?",
        "expect": {"escalate": True},
        "source": "test",
    }
    result = example_from_entry(entry, tools)
    assert result is not None
    assert result["answers"] == []


# ---------------------------------------------------------------------------
# test_a_failure_entry_is_skipped
# ---------------------------------------------------------------------------


def test_a_failure_entry_is_skipped() -> None:
    """kind=failure → None."""
    entry: dict = {
        "id": "t04",
        "kind": "failure",
        "text": "docker ps -> exit 1",
        "expect": {"operation": "service_status", "args": {"service": "docker.service"}},
        "source": "test",
    }
    result = example_from_entry(entry, tools)
    assert result is None


# ---------------------------------------------------------------------------
# test_an_unknown_operation_is_counted_invalid
# ---------------------------------------------------------------------------


def test_an_unknown_operation_is_counted_invalid() -> None:
    """operation 'format_disk' → None (invalid)."""
    entry: dict = {
        "id": "t05",
        "kind": "explicit",
        "text": "Format my disk please",
        "expect": {"operation": "format_disk", "args": {}},
        "source": "test",
    }
    result = example_from_entry(entry, tools)
    assert result is None
    # Also verify validate directly rejects it.
    err = validate("format_disk", {})
    assert err is not None


# ---------------------------------------------------------------------------
# test_every_example_carries_the_same_tools_list
# ---------------------------------------------------------------------------


def test_every_example_carries_the_same_tools_list(tmp_path: Path) -> None:
    """Every output example has the identical tools list."""
    entries: list[dict] = [
        {
            "id": "t10",
            "kind": "explicit",
            "text": "Show GPU usage",
            "expect": {"operation": "gpu_stats", "args": {}},
            "source": "test",
        },
        {
            "id": "t11",
            "kind": "explicit",
            "text": "Why did vLLM crash?",
            "expect": {"escalate": True},
            "source": "test",
        },
    ]
    corpus = _tiny_corpus(entries)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))

    examples, counts = build(corpus_path, None, tools)
    for ex in examples:
        assert ex["tools"] is tools


# ---------------------------------------------------------------------------
# test_held_out_file_is_refused
# ---------------------------------------------------------------------------


def test_held_out_file_is_refused(tmp_path: Path) -> None:
    """main(['--corpus', 'held-out.json', '--out', ...]) == 2."""
    (tmp_path / "held-out.json").write_text(json.dumps({"header": "h", "entries": []}))
    rc = main(["--corpus", str(tmp_path / "held-out.json"), "--out", str(tmp_path / "out.jsonl")])
    assert rc == 2


# ---------------------------------------------------------------------------
# test_a_missing_corpus_returns_2
# ---------------------------------------------------------------------------


def test_a_missing_corpus_returns_2(tmp_path: Path) -> None:
    """A missing corpus file → return code 2."""
    rc = main(
        ["--corpus", str(tmp_path / "does-not-exist.json"), "--out", str(tmp_path / "out.jsonl")]
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# test_only_approved_records_with_text_are_used
# ---------------------------------------------------------------------------


def test_only_approved_records_with_text_are_used(tmp_path: Path) -> None:
    """4 records: one good, one declined, one with no request_text,
    one with unknown op → exactly 1 example."""
    records: list[dict] = [
        # Good record
        {
            "id": "r01",
            "operator_decision": "approved",
            "operation": "disk_stats",
            "request_text": "Check disk usage",
            "args": {},
        },
        # Declined
        {
            "id": "r02",
            "operator_decision": "declined",
            "operation": "disk_stats",
            "request_text": "Check disk usage 2",
            "args": {},
        },
        # No request_text
        {
            "id": "r03",
            "operator_decision": "approved",
            "operation": "disk_stats",
            "request_text": "",
            "args": {},
        },
        # Unknown operation
        {
            "id": "r04",
            "operator_decision": "approved",
            "operation": "format_disk",
            "request_text": "Format disk",
            "args": {},
        },
    ]
    bundle = _tiny_bundle(records)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))

    corpus = _tiny_corpus([])
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))

    examples, counts = build(corpus_path, bundle_path, tools)
    assert len(examples) == 1
    assert examples[0]["query"] == "Check disk usage"
    assert counts["skipped_records"] == 3


# ---------------------------------------------------------------------------
# test_duplicates_are_dropped
# ---------------------------------------------------------------------------


def test_duplicates_are_dropped(tmp_path: Path) -> None:
    """Same query in different letter case, same answer → one line."""
    entries: list[dict] = [
        {
            "id": "d01",
            "kind": "explicit",
            "text": "Check disk usage",
            "expect": {"operation": "disk_stats", "args": {}},
            "source": "test",
        },
        {
            "id": "d02",
            "kind": "explicit",
            "text": "check disk usage",
            "expect": {"operation": "disk_stats", "args": {}},
            "source": "test",
        },
    ]
    corpus = _tiny_corpus(entries)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))

    examples, counts = build(corpus_path, None, tools)
    assert len(examples) == 1
    assert counts["duplicates"] == 1


# ---------------------------------------------------------------------------
# test_two_runs_write_identical_bytes
# ---------------------------------------------------------------------------


def test_two_runs_write_identical_bytes(tmp_path: Path) -> None:
    """Running ``main()`` twice on the same input writes byte-identical JSONL files.

    Drives the real ``--out`` file-writing path (not just ``build()``'s
    in-memory result), so an encoding, newline or byte-ordering regression in
    ``main()``'s write loop would fail this test.
    """
    entries: list[dict] = [
        {
            "id": "i01",
            "kind": "explicit",
            "text": "Check disk usage",
            "expect": {"operation": "disk_stats", "args": {}},
            "source": "test",
        },
        {
            "id": "i02",
            "kind": "explicit",
            "text": "Show GPU usage",
            "expect": {"operation": "gpu_stats", "args": {}},
            "source": "test",
        },
    ]
    corpus = _tiny_corpus(entries)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))

    out1 = tmp_path / "out1.jsonl"
    out2 = tmp_path / "out2.jsonl"
    assert main(["--corpus", str(corpus_path), "--out", str(out1)]) == 0
    assert main(["--corpus", str(corpus_path), "--out", str(out2)]) == 0

    bytes1 = out1.read_bytes()
    bytes2 = out2.read_bytes()
    assert bytes1 == bytes2
    assert bytes1.count(b"\n") == 2
    assert bytes1.endswith(b"\n")


# ---------------------------------------------------------------------------
# test_the_real_dev_corpus_builds_without_invalid_entries
# ---------------------------------------------------------------------------


def test_the_real_dev_corpus_builds_without_invalid_entries() -> None:
    """Run build() on the real nvsh/tiers/corpus/dev.json → invalid==0, written>0."""
    repo_root = Path(__file__).resolve().parents[1]
    real_corpus = repo_root / "nvsh" / "tiers" / "corpus" / "dev.json"
    if not real_corpus.exists():
        pytest.skip("dev.json not found")
    examples, counts = build(real_corpus, None, tools)
    assert counts["invalid"] == 0
    assert counts["written"] > 0


# ---------------------------------------------------------------------------
# test_nothing_in_the_nvsh_package_imports_the_script
# ---------------------------------------------------------------------------


def test_nothing_in_the_nvsh_package_imports_the_script() -> None:
    """Every nvsh/*.py file avoids 'needle-finetune' and 'build_dataset'."""
    repo_root = Path(__file__).resolve().parents[1]
    nvsh_py_files = list(repo_root.glob("nvsh/**/*.py"))
    found_finetune: list[str] = []
    found_dataset: list[str] = []
    for py_file in nvsh_py_files:
        text = py_file.read_text(encoding="utf-8")
        if "needle-finetune" in text:
            found_finetune.append(str(py_file))
        if "build_dataset" in text:
            found_dataset.append(str(py_file))

    assert not found_finetune, f"'needle-finetune' found in: {found_finetune}"
    assert not found_dataset, f"'build_dataset' found in: {found_dataset}"


def test_an_approved_record_teaches_the_call_the_operator_approved():
    record = {
        "operator_decision": "approved",
        "operation": "disk_stats",
        "args": {},
        "request_text": "am I out of disk?",
    }
    example = _builder.example_from_record(record, [])
    assert example["answers"] == [{"name": "disk_stats", "arguments": {}}]


def test_build_itself_refuses_the_held_out_split(tmp_path):
    held_out = tmp_path / "held-out.json"
    held_out.write_text('{"entries": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        _builder.build(held_out, None, [])


# ---------------------------------------------------------------------------
# --out must not alias --corpus/--bundle (QODO 4053821279)
# ---------------------------------------------------------------------------


def test_out_equal_to_corpus_is_refused_and_the_corpus_survives(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    original = json.dumps(_tiny_corpus([]))
    corpus_path.write_text(original, encoding="utf-8")

    rc = main(["--corpus", str(corpus_path), "--out", str(corpus_path)])

    assert rc == 2
    assert corpus_path.read_text(encoding="utf-8") == original


def test_out_equal_to_bundle_is_refused_and_the_bundle_survives(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(_tiny_corpus([])), encoding="utf-8")
    bundle_path = tmp_path / "bundle.json"
    original = json.dumps(_tiny_bundle([]))
    bundle_path.write_text(original, encoding="utf-8")

    rc = main(
        ["--corpus", str(corpus_path), "--bundle", str(bundle_path), "--out", str(bundle_path)]
    )

    assert rc == 2
    assert bundle_path.read_text(encoding="utf-8") == original


def test_out_via_a_symlink_to_corpus_is_also_refused(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    original = json.dumps(_tiny_corpus([]))
    corpus_path.write_text(original, encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(corpus_path)

    rc = main(["--corpus", str(corpus_path), "--out", str(alias)])

    assert rc == 2
    assert corpus_path.read_text(encoding="utf-8") == original


def test_a_genuinely_different_out_path_still_writes(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            _tiny_corpus(
                [
                    {
                        "id": "t01",
                        "kind": "explicit",
                        "text": "am I out of disk?",
                        "expect": {"operation": "disk_stats", "args": {}},
                        "source": "test",
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "out.jsonl"

    rc = main(["--corpus", str(corpus_path), "--out", str(out_path)])

    assert rc == 0
    assert out_path.exists()
