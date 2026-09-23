"""scripts/lfm-finetune/train_scorer.py (issue 46, Track B): the scorer's trainer.

The pure helpers run everywhere; the loss and the training loop run with a
tiny fake torch model only when torch is importable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "lfm-finetune" / "train_scorer.py"
_WORLD = json.loads((_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json").read_text())["world"]


def _module():
    spec = importlib.util.spec_from_file_location("lfm_train_scorer", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def _split(tmp_path: Path, side: str, entries: list[dict], name: str | None = None) -> Path:
    path = tmp_path / (name or f"{side}.json")
    payload = {
        "header": f"Development corpus. Split '{side}' of dev.json (seed=46).",
        "entries": entries,
        "world": _WORLD,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(entry_id: str, text: str, expect: dict) -> dict:
    return {"id": entry_id, "kind": "explicit", "text": text, "expect": expect, "source_id": "s"}


_ENTRIES = [
    _entry("e1", "How hot is this machine?", {"operation": "thermal_stats", "args": {}}),
    _entry("e2", "Rewrite my kernel", {"escalate": True}),
    _entry("e3", "What is swap?", {"explain": True, "answer": "Disk used as memory."}),
]


class _Tokenizer:
    """One token per character of the rendered prompt; letters encode to their code point."""

    chat_template = "{{ messages }}"
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["content"][:8] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 250 + 1 for char in text]


# -- reading the split files --


def test_gold_candidate_maps_each_expectation_kind() -> None:
    module = _module()
    assert module.gold_candidate({"operation": "gpu_stats", "args": {}}) == "gpu_stats"
    assert module.gold_candidate({"escalate": True}) == "escalate"
    assert module.gold_candidate({"explain": True, "answer": "x"}) == "explain"
    with pytest.raises(ValueError, match="expect"):
        module.gold_candidate({"surprise": 1})


def test_a_gold_operation_outside_the_table_is_refused() -> None:
    with pytest.raises(ValueError, match="not a candidate"):
        _module().gold_candidate({"operation": "no_such_operation"})


def test_read_split_returns_request_text_and_gold_for_each_entry(tmp_path) -> None:
    module = _module()
    examples = module.read_split(_split(tmp_path, "train", _ENTRIES), module.TRAIN_SIDE)
    assert [example.gold for example in examples] == ["thermal_stats", "escalate", "explain"]
    assert examples[0].request == "How hot is this machine?"
    assert [example.entry_id for example in examples] == ["e1", "e2", "e3"]


def test_training_refuses_any_side_but_train(tmp_path) -> None:
    module = _module()
    for side in ("val", "test"):
        with pytest.raises(ValueError, match="side"):
            module.read_split(_split(tmp_path, side, _ENTRIES), module.TRAIN_SIDE)


def test_validation_reads_only_the_val_side(tmp_path) -> None:
    module = _module()
    assert module.read_split(_split(tmp_path, "val", _ENTRIES), module.VAL_SIDE)
    with pytest.raises(ValueError, match="side"):
        module.read_split(_split(tmp_path, "test", _ENTRIES), module.VAL_SIDE)


def test_the_held_out_split_is_refused_by_name_and_by_header(tmp_path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="held-out"):
        module.read_split(
            _split(tmp_path, "train", _ENTRIES, name="held-out.json"), module.TRAIN_SIDE
        )
    path = tmp_path / "renamed.json"
    path.write_text(json.dumps({"header": "Held-out split: sealed.", "entries": _ENTRIES}))
    with pytest.raises(ValueError, match="held-out"):
        module.read_split(path, module.TRAIN_SIDE)


def test_a_split_with_no_entries_is_refused(tmp_path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="no entries"):
        module.read_split(_split(tmp_path, "train", []), module.TRAIN_SIDE)


# -- encoding --


def test_encode_puts_the_gold_index_on_the_rendered_prompt(tmp_path) -> None:
    module = _module()
    examples = module.read_split(_split(tmp_path, "train", _ENTRIES), module.TRAIN_SIDE)
    rows = module.encode(_Tokenizer(), examples, max_length=4096)
    candidates = module.scorer.candidates()
    assert [row["target"] for row in rows] == [
        candidates.index("thermal_stats"),
        candidates.index("escalate"),
        candidates.index("explain"),
    ]
    assert all(row["input_ids"] for row in rows)


def test_encode_refuses_a_prompt_over_max_length_rather_than_cutting_it(tmp_path) -> None:
    module = _module()
    examples = module.read_split(_split(tmp_path, "train", _ENTRIES), module.TRAIN_SIDE)
    with pytest.raises(ValueError, match="over 5"):
        module.encode(_Tokenizer(), examples, max_length=5)


def test_file_sha256_names_the_exact_split_bytes(tmp_path) -> None:
    module = _module()
    path = _split(tmp_path, "train", _ENTRIES)
    first = module.file_sha256(path)
    assert len(first) == 64
    path.write_text(path.read_text() + " ")
    assert module.file_sha256(path) != first


# -- the loss and the loop (torch, when importable) --


def _toy(torch, module, n_labels: int = 4):
    """A tiny causal-LM stand-in: an embedding and a head, logits_to_keep honoured."""

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(64, 16)
            self.head = torch.nn.Linear(16, 64)
            self.kept: list = []

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            self.kept.append(logits_to_keep)
            hidden = self.embed(input_ids).cumsum(dim=1)
            if isinstance(logits_to_keep, int):
                keep = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
            else:
                keep = logits_to_keep
            return type("Out", (), {"logits": self.head(hidden[:, keep, :])})()

    rows = [
        # The first token decides the label; lengths differ so padding is exercised.
        {"input_ids": [1 + (i % 8), 9, 2 + (i % 5)] + [3] * (i % 3), "target": i % 8 % n_labels}
        for i in range(24)
    ]
    label_ids = [40 + index for index in range(n_labels)]
    return _Model, rows, label_ids


def test_label_logits_are_restricted_to_the_candidate_tokens_at_each_last_position() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    model_cls, rows, label_ids = _toy(torch, module)
    model = model_cls()
    input_ids, mask = module.pad_right([row["input_ids"] for row in rows[:4]], pad_id=0)
    logits = module.label_logits(model, input_ids, mask, torch.tensor(label_ids))
    assert tuple(logits.shape) == (4, len(label_ids))
    kept = model.kept[-1]
    assert not isinstance(kept, int) and len(kept) <= 4  # never every position's logits
    # Each row's scores are the head's output at that row's own last real token.
    for row_index, row in enumerate(rows[:4]):
        alone = torch.tensor([row["input_ids"]])
        full = model_cls.forward(model, alone, None, 0).logits[0, -1, label_ids]
        assert torch.allclose(logits[row_index], full, atol=1e-5)


def test_pad_right_masks_the_padding() -> None:
    pytest.importorskip("torch")
    module = _module()
    input_ids, mask = module.pad_right([[5, 6, 7], [8]], pad_id=0)
    assert input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_training_loss_decreases_on_a_toy_set() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    model_cls, rows, label_ids = _toy(torch, module)
    torch.manual_seed(0)
    history = module.train_loop(
        model_cls(), rows, label_ids, epochs=30, lr=0.05, batch=4, seed=46, pad_id=0
    )
    assert history[-1]["loss"] < 0.5 * history[0]["loss"]


def test_training_is_reproducible_under_one_seed() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    model_cls, rows, label_ids = _toy(torch, module)

    def run(seed: int) -> list[float]:
        module.seed_everything(seed)
        model = model_cls()
        history = module.train_loop(
            model, rows, label_ids, epochs=3, lr=0.05, batch=4, seed=seed, pad_id=0
        )
        return [step["loss"] for step in history]

    assert run(46) == run(46)
    assert run(46) != run(47)


def test_evaluate_reports_accuracy_and_confidence() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    model_cls, rows, label_ids = _toy(torch, module)
    model = model_cls()
    module.train_loop(model, rows, label_ids, epochs=30, lr=0.05, batch=4, seed=46, pad_id=0)
    report = module.evaluate(model, rows, label_ids, batch=4, pad_id=0)
    assert report["n"] == len(rows)
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["accuracy"] > 1 / len(label_ids)
    assert 0.0 < report["mean_confidence"] <= 1.0
    assert report["loss"] >= 0.0
