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
import types
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


def test_read_calibration_texts_returns_one_text_per_nonblank_jsonl_line(tmp_path) -> None:
    module = _module()
    path = tmp_path / "calib.jsonl"
    path.write_text('"first"\n\n"second"\n', encoding="utf-8")
    assert module.read_calibration_texts(path) == ["first", "second"]


def test_read_calibration_texts_handles_an_empty_file(tmp_path) -> None:
    module = _module()
    path = tmp_path / "calib.jsonl"
    path.write_text("", encoding="utf-8")
    assert module.read_calibration_texts(path) == []


def test_read_calibration_texts_keeps_an_embedded_newline_as_one_record(tmp_path) -> None:
    """Codex finding #7: a JSON-encoded string's embedded newline is data, not a line break."""
    module = _module()
    path = tmp_path / "calib.jsonl"
    path.write_text('"request\\nerror details"\n"second request"\n', encoding="utf-8")
    assert module.read_calibration_texts(path) == ["request\nerror details", "second request"]


def _quantize_module():
    """quantize.py, loaded the same way ``_module()`` loads awq_oneshot.py (they are siblings)."""
    spec = importlib.util.spec_from_file_location(
        "lfm_quantize_for_awq_oneshot_test", _SCRIPT.parent / "quantize.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_calibration_round_trips_a_record_with_an_embedded_newline(tmp_path) -> None:
    """Codex finding #7: quantize.py writes JSONL, awq_oneshot.py reads it back -- a record

    containing a newline must round-trip as exactly one AWQ sample, and the sample count
    must equal the record count (a naive plain-text join would split it into two and lose
    the original count).
    """
    quantize = _quantize_module()
    awq = _module()
    texts = ["request\nerror details", "second request"]
    path = quantize.write_calibration_jsonl(texts, tmp_path / "calibration.jsonl")
    read_back = awq.read_calibration_texts(path)
    assert read_back == texts
    assert len(read_back) == len(texts)


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


# ---------------------------------------------------------------------------
# sanitize_generation_config (live-check fix): transformers >= 5.17 refuses
# to save a loaded generation config whose do_sample is False but temperature
# is still set -- exactly the deviation-d3 generation_config.json a
# stock-copy source dir carries.
# ---------------------------------------------------------------------------


class _FakeGenerationConfig:
    """Stands in for transformers.GenerationConfig: keyword-args become attributes."""

    def __init__(self, **kwargs) -> None:
        self.temperature = kwargs.get("temperature")
        self.do_sample = kwargs.get("do_sample")
        for key, value in kwargs.items():
            setattr(self, key, value)


def _install_fake_transformers(monkeypatch, generation_config_cls=_FakeGenerationConfig) -> None:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.GenerationConfig = generation_config_cls
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_sanitize_generation_config_drops_temperature_and_do_sample(monkeypatch) -> None:
    """The exact combination the live check hit: temperature 0.0, do_sample False."""
    module = _module()
    _install_fake_transformers(monkeypatch)

    class _LoadedGenConfig:
        temperature = 0.0
        do_sample = False
        eos_token_id = [248046, 248044]
        bos_token_id = 1
        pad_token_id = None

    class _FakeModel:
        generation_config = _LoadedGenConfig()

    model = _FakeModel()
    module.sanitize_generation_config(model)

    sanitized = model.generation_config
    assert sanitized.temperature is None
    assert sanitized.do_sample is None
    # Not the invalid combination that made save_pretrained refuse: a config
    # with do_sample False must never still carry an explicit temperature.
    assert not (sanitized.temperature is not None and sanitized.do_sample is False)


def test_sanitize_generation_config_keeps_the_token_ids(monkeypatch) -> None:
    module = _module()
    _install_fake_transformers(monkeypatch)

    class _LoadedGenConfig:
        temperature = 0.0
        do_sample = False
        eos_token_id = [248046, 248044]
        bos_token_id = 1
        pad_token_id = None  # absent ids are never invented

    class _FakeModel:
        generation_config = _LoadedGenConfig()

    model = _FakeModel()
    module.sanitize_generation_config(model)

    assert model.generation_config.eos_token_id == [248046, 248044]
    assert model.generation_config.bos_token_id == 1
    assert getattr(model.generation_config, "pad_token_id", None) is None


def test_sanitize_generation_config_handles_no_existing_config(monkeypatch) -> None:
    module = _module()
    _install_fake_transformers(monkeypatch)

    class _FakeModel:
        generation_config = None

    model = _FakeModel()
    module.sanitize_generation_config(model)  # must not raise
    assert model.generation_config.temperature is None


# ---------------------------------------------------------------------------
# main()'s order: quantize (oneshot) -> sanitize generation config -> save
# ---------------------------------------------------------------------------


def test_main_sanitizes_the_generation_config_between_oneshot_and_save(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple] = []

    class _LoadedGenConfig:
        temperature = 0.0
        do_sample = False
        eos_token_id = [2]

    class _FakeModel:
        def __init__(self) -> None:
            self.generation_config = _LoadedGenConfig()

        def save_pretrained(self, out_dir, save_compressed=True) -> None:
            calls.append(
                (
                    "model.save_pretrained",
                    self.generation_config.temperature,
                    self.generation_config.do_sample,
                )
            )

    class _FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<prompt>"

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2, 3] for _ in texts]}

        def save_pretrained(self, out_dir) -> None:
            calls.append(("tokenizer.save_pretrained",))

    class _FakeDataset:
        def __init__(self, rows) -> None:
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def map(self, fn, remove_columns=None):
            return self

    def fake_oneshot(**kwargs) -> None:
        calls.append(("oneshot",))

    _install_fake_transformers(monkeypatch)
    sys.modules["transformers"].AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeModel()
    )
    sys.modules["transformers"].AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeTokenizer()
    )

    fake_llmcompressor = types.ModuleType("llmcompressor")
    fake_llmcompressor.oneshot = fake_oneshot
    monkeypatch.setitem(sys.modules, "llmcompressor", fake_llmcompressor)
    fake_modifiers_pkg = types.ModuleType("llmcompressor.modifiers")
    monkeypatch.setitem(sys.modules, "llmcompressor.modifiers", fake_modifiers_pkg)
    fake_awq_module = types.ModuleType("llmcompressor.modifiers.awq")
    fake_awq_module.AWQModifier = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "llmcompressor.modifiers.awq", fake_awq_module)

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    module = _module()
    calib_file = tmp_path / "calib.jsonl"
    calib_file.write_text('"hello there"\n', encoding="utf-8")
    out_dir = tmp_path / "awq-out"

    rc = module.main(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--calibration-file",
            str(calib_file),
            "--out-dir",
            str(out_dir),
            "--num-calibration-samples",
            "1",
        ]
    )

    assert rc == 0
    assert [call[0] for call in calls] == [
        "oneshot",
        "model.save_pretrained",
        "tokenizer.save_pretrained",
    ]
    save_call = calls[1]
    # The save must never see the invalid combination that made the live
    # check's save_pretrained refuse: do_sample False with a set temperature.
    assert not (save_call[1] is not None and save_call[2] is False)
