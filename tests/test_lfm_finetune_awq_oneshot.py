"""scripts/lfm-finetune/awq_oneshot.py (issue 46, risk r14): the t16 spike's proven AWQ recipe.

Runs only inside the separate AWQ venv in production (llm-compressor +
transformers 5.17.0); here it is loaded by importlib with fakes, exactly as
the nvsh dev environment (no torch/transformers/llmcompressor) requires.
Every heavy import stays lazy inside build_recipe()/main() so this module
loads with none of those packages installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "awq_oneshot.py"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_awq_oneshot", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Calibration text: reading and chat-template rendering
# ---------------------------------------------------------------------------


def test_read_calibration_texts_returns_one_text_per_nonblank_line(tmp_path) -> None:
    module = _module()
    path = tmp_path / "calib.txt"
    path.write_text("first\n\nsecond\n", encoding="utf-8")
    assert module.read_calibration_texts(path) == ["first", "second"]


def test_read_calibration_texts_handles_an_empty_file(tmp_path) -> None:
    module = _module()
    path = tmp_path / "calib.txt"
    path.write_text("", encoding="utf-8")
    assert module.read_calibration_texts(path) == []


class _FakeTokenizer:
    """Records the kwargs apply_chat_template was called with."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return f"<rendered:{messages[0]['content']}>"


def test_render_calibration_text_requests_a_generation_prompt_with_thinking_off() -> None:
    module = _module()
    tokenizer = _FakeTokenizer()
    rendered = module.render_calibration_text(tokenizer, "turn the fan on")
    assert rendered == "<rendered:turn the fan on>"
    call = tokenizer.calls[0]
    assert call["add_generation_prompt"] is True
    assert call["enable_thinking"] is False
    assert call["tokenize"] is False
    assert call["messages"] == [{"role": "user", "content": "turn the fan on"}]


# ---------------------------------------------------------------------------
# The proven recipe (spike t16): targets, scheme, ignore list
# ---------------------------------------------------------------------------


def test_awq_ignore_list_excludes_vision_attention_and_mtp() -> None:
    module = _module()
    assert module.AWQ_IGNORE == [
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn.*",
        "re:.*mtp.*",
    ]
    assert module.AWQ_TARGETS == ["Linear"]
    assert module.AWQ_SCHEME == "W4A16"


def test_build_recipe_matches_the_proven_spike_recipe() -> None:
    pytest.importorskip("llmcompressor")
    module = _module()
    recipe = module.build_recipe()
    assert len(recipe) == 1
    modifier = recipe[0]
    assert modifier.targets == module.AWQ_TARGETS
    assert modifier.scheme == module.AWQ_SCHEME
    assert modifier.ignore == module.AWQ_IGNORE


def test_main_requires_all_four_arguments() -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--model-dir", "x"])
