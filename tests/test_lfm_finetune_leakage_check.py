"""leakage_check.py (issue 46, t19): no val/test/held-out text or near-duplicate in training."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lfm-finetune" / "leakage_check.py"


def _module():
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("leakage_check", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(path: Path, texts: dict[str, str]) -> Path:
    entries = [{"id": k, "text": v, "expect": {"escalate": True}} for k, v in texts.items()]
    path.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    return path


def test_finds_exact_and_near_duplicates_by_id_only(tmp_path, capsys) -> None:
    module = _module()
    train = _split(
        tmp_path / "train.json",
        {
            "t1": "Show me the current GPU utilisation please",
            "t2": "show me the current gpu utilisation, please!",  # exact after normalizing
            "t3": "please show me the current GPU utilisation",  # same words, reordered
            "t4": "How hot is the board right now?",
        },
    )
    held = _split(tmp_path / "held.json", {"h1": "Show me the current GPU utilisation please"})
    rc = module.main(["--train", str(train), "--protected", str(held)])
    out = capsys.readouterr().out
    assert rc == 1
    report = json.loads(out)
    assert report["hits"] == 3
    assert {h["train_id"] for h in report["matches"]} == {"t1", "t2", "t3"}
    assert "GPU" not in out  # never prints a protected (or training) text


def test_drops_hits_into_a_filtered_file(tmp_path, capsys) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "Show the GPU stats now", "t4": "how hot is it"})
    held = _split(tmp_path / "held.json", {"h1": "Show the GPU stats now"})
    out_file = tmp_path / "filtered.json"
    rc = module.main(
        ["--train", str(train), "--protected", str(held), "--out-filtered", str(out_file)]
    )
    assert rc == 0
    kept = json.loads(out_file.read_text(encoding="utf-8"))
    assert [e["id"] for e in kept["entries"]] == ["t4"]
    assert kept["header"] == "h"


def test_a_clean_training_file_passes(tmp_path, capsys) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "Restart the vllm service for me"})
    held = _split(tmp_path / "held.json", {"h1": "What does unified memory mean on GB10?"})
    assert module.main(["--train", str(train), "--protected", str(held)]) == 0
    assert json.loads(capsys.readouterr().out)["hits"] == 0


def test_short_texts_only_match_exactly(tmp_path, capsys) -> None:
    module = _module()
    train = _split(tmp_path / "train.json", {"t1": "gpu stats", "t2": "stats gpu"})
    held = _split(tmp_path / "held.json", {"h1": "gpu stats"})
    module.main(["--train", str(train), "--protected", str(held)])
    report = json.loads(capsys.readouterr().out)
    assert [h["train_id"] for h in report["matches"]] == ["t1"]


def test_two_protected_files_with_the_same_name_are_both_checked(tmp_path, capsys) -> None:
    # Codex review: files keyed by basename let one test.json replace another.
    module = _module()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _split(tmp_path / "a" / "test.json", {"x1": "Restart the vllm service now please"})
    second = _split(tmp_path / "b" / "test.json", {"y1": "How hot is the Jetson board right now"})
    train = _split(
        tmp_path / "train.json",
        {
            "t1": "Restart the vllm service now please",
            "t2": "How hot is the Jetson board right now",
        },
    )
    assert module.main(["--train", str(train), "--protected", str(first), str(second)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["hits"] == 2
    assert len(report["protected"]) == 2


def test_filtering_drops_only_the_matching_rows(tmp_path, capsys) -> None:
    module = _module()
    train = tmp_path / "train.jsonl"
    rows = [
        {"id": "dup", "text": "Show the GPU stats now please"},
        {"id": "dup", "text": "What is unified memory on GB10"},
    ]
    train.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    held = _split(tmp_path / "held.json", {"h1": "Show the GPU stats now please"})
    out_file = tmp_path / "out.jsonl"
    module.main(["--train", str(train), "--protected", str(held), "--out-filtered", str(out_file)])
    kept = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in kept] == ["What is unified memory on GB10"]


def test_an_entry_without_a_text_is_refused(tmp_path, capsys) -> None:
    module = _module()
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"id": "m1", "messages": []}) + "\n", encoding="utf-8")
    held = _split(tmp_path / "held.json", {"h1": "anything at all here"})
    assert module.main(["--train", str(train), "--protected", str(held)]) == 2
    assert "m1" in capsys.readouterr().err
