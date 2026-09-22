"""Pure helpers of scripts/lfm-finetune/train.py (issue 39); no training stack needed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "train.py"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_train", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    """Stands in for apply_chat_template: the last message's content is the assistant turn."""

    def apply_chat_template(self, messages, tools=None, **kwargs):
        prompt = [1] * (len(messages) - 1) * 3
        answer = [7, 8, 9]
        return {"input_ids": prompt + answer, "assistant_masks": [0] * len(prompt) + [1] * 3}


def test_labels_train_only_on_the_assistant_turn() -> None:
    module = _module()
    assert module.labels_from_mask([5, 6, 7], [0, 0, 1]) == [-100, -100, 7]


def test_an_empty_mask_is_refused() -> None:
    with pytest.raises(ValueError, match="assistant mask is empty"):
        _module().labels_from_mask([5, 6], [0, 0])


def test_tokenize_example_masks_the_prompt() -> None:
    example = {"messages": [{"role": "user"}, {"role": "assistant"}], "source_id": "s"}
    row = _module().tokenize_example(_Tokenizer(), example, max_length=100)
    assert row["labels"] == [-100, -100, -100, 7, 8, 9]
    assert row["attention_mask"] == [1] * 6


def test_an_example_over_max_length_is_refused_not_cut() -> None:
    example = {"messages": [{"role": "user"}, {"role": "assistant"}], "source_id": "s"}
    with pytest.raises(ValueError, match="over 4"):
        _module().tokenize_example(_Tokenizer(), example, max_length=4)


def test_read_examples_refuses_a_line_not_ending_in_the_assistant(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n")
    with pytest.raises(ValueError, match="not the assistant"):
        _module().read_examples(path)


def test_read_examples_refuses_an_empty_file(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no examples"):
        _module().read_examples(path)
