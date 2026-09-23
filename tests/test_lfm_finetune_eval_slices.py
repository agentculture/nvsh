"""The eval-slice builder: scripts/lfm-finetune/eval_slices.py.

Uses a small in-memory fixture, not a real corpus, so no file I/O is
needed beyond pytest's tmp_path fixtures (part of #39).
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/eval_slices.py"


def _module():
    spec = importlib.util.spec_from_file_location("eval_slices", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(
    entry_id: str,
    kind: str = "explicit",
    expect: dict | None = None,
    source: str = "fixture",
    text: str = "",
) -> dict:
    if expect is None:
        expect = {"escalate": True}
    if not text:
        text = f"fixture text for {entry_id}"
    return {
        "id": entry_id,
        "kind": kind,
        "text": text,
        "expect": expect,
        "source": source,
    }


def _op_entry(entry_id: str) -> dict:
    return _entry(entry_id, expect={"operation": "gpu_stats", "args": {}})


def _esc_entry(entry_id: str) -> dict:
    return _entry(entry_id, expect={"escalate": True})


def _split(header: str = "Fixture split") -> dict:
    return {
        "header": header,
        "entries": [
            _op_entry("op01"),
            _op_entry("op02"),
            _esc_entry("esc01"),
        ],
    }


def test_every_slice_entry_text_equals_its_source_entry_text():
    """Each slice entry's text must be byte-identical to its source."""
    module = _module()
    split = _split()
    slice_dict = module.missing_candidate_slice(split, ("gpu_stats",))
    sources = {e["id"]: e["text"] for e in split["entries"]}
    for entry in slice_dict["entries"]:
        assert entry["text"] == sources[entry["source_id"]]


def test_every_slice_candidates_equals_all_ops_minus_gold():
    """candidates must be exactly all_operations minus the gold operation."""
    module = _module()
    ops = ("op_a", "op_b", "op_c")
    split = {
        "header": "Fixture split",
        "entries": [
            _entry("op01", expect={"operation": "gpu_stats", "args": {}}),
        ],
    }
    slice_dict = module.missing_candidate_slice(split, ops)
    assert len(slice_dict["entries"]) == 1
    entry = slice_dict["entries"][0]
    assert sorted(entry["candidates"]) == sorted(["op_a", "op_b", "op_c"])
    assert "gpu_stats" not in entry["candidates"]


def test_entries_without_operation_expectation_are_skipped():
    """Only operation-expect entries appear in the slice."""
    module = _module()
    split = _split()
    slice_dict = module.missing_candidate_slice(split, ("gpu_stats",))
    # _split has 2 op entries + 1 esc entry → slice has exactly 2 entries
    assert len(slice_dict["entries"]) == 2
    source_ids = {e["id"] for e in slice_dict["entries"]}
    assert "esc01-nocand" not in source_ids
    assert "op01-nocand" in source_ids
    assert "op02-nocand" in source_ids


def test_every_slice_entry_expects_escalate():
    """Every slice entry's expect must be {escalate: True}."""
    module = _module()
    split = _split()
    slice_dict = module.missing_candidate_slice(split, ("gpu_stats",))
    for entry in slice_dict["entries"]:
        assert entry["expect"] == {"escalate": True}


def test_input_dict_is_unchanged_after_the_call():
    """The call must not mutate the input split."""
    module = _module()
    split = _split()
    original = copy.deepcopy(split)
    module.missing_candidate_slice(split, ("gpu_stats",))
    assert split == original


def test_main_writes_the_file_and_prints_entries_n(tmp_path, capsys):
    """The CLI writes JSON and prints entries=<n>."""
    module = _module()
    split = _split()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    out_path = tmp_path / "out" / "slice.json"

    exit_code = module.main(["--split", str(split_path), "--out", str(out_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "entries=2" in captured.out

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "header" in payload
    assert len(payload["entries"]) == 2
    for entry in payload["entries"]:
        assert "nocand" in entry["id"]
        assert entry["expect"] == {"escalate": True}
        assert "candidates" in entry
        assert "source_id" in entry
        assert "text" in entry
