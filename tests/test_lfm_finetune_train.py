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
    module = _module()
    with pytest.raises(ValueError, match="assistant mask is empty"):
        module.labels_from_mask([5, 6], [0, 0])


def test_tokenize_example_masks_the_prompt() -> None:
    example = {"messages": [{"role": "user"}, {"role": "assistant"}], "source_id": "s"}
    row = _module().tokenize_example(_Tokenizer(), example, max_length=100)
    assert row["labels"] == [-100, -100, -100, 7, 8, 9]
    assert row["attention_mask"] == [1] * 6


def test_an_example_over_max_length_is_refused_not_cut() -> None:
    example = {"messages": [{"role": "user"}, {"role": "assistant"}], "source_id": "s"}
    module = _module()
    with pytest.raises(ValueError, match="over 4"):
        module.tokenize_example(_Tokenizer(), example, max_length=4)


def test_read_examples_refuses_a_line_not_ending_in_the_assistant(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n")
    module = _module()
    with pytest.raises(ValueError, match="not the assistant"):
        module.read_examples(path)


def test_read_examples_refuses_an_empty_file(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("\n")
    module = _module()
    with pytest.raises(ValueError, match="no examples"):
        module.read_examples(path)


# --- Templates without generation markers (issue 46: Qwen3.5-0.8B) ---------

#: A template shaped like Qwen3.5's: no generation-block tags, and an
#: ``enable_thinking`` switch. The word "generation" appears only in
#: ``add_generation_prompt``, which must not count as a marker.
_NO_MARKER_TEMPLATE = (
    "{%- for m in messages %}...{%- endfor %}"
    "{%- if add_generation_prompt %}{%- if enable_thinking is defined %}{%- endif %}{%- endif %}"
)
_MARKER_TEMPLATE = "{%- for m in messages %}{%- generation -%}...{%- endgeneration -%}{%- endfor %}"

_EOS = 2
_NEWLINE = 3
_THINK_BLOCK = [40, 41, 42]


class _NoMarkerTokenizer:
    """A Qwen-shaped stand-in: no assistant mask, an empty think block before the answer."""

    chat_template = _NO_MARKER_TEMPLATE
    eos_token_id = _EOS

    def __init__(self, answer=(7, 8, 9)) -> None:
        self.answer = list(answer)
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tools=None, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("return_assistant_tokens_mask"):
            raise AssertionError("a template without generation markers cannot build a mask")
        head = [1, 1, 1]
        if kwargs.get("add_generation_prompt"):
            ids = head + [5] + _THINK_BLOCK
        else:
            ids = head + [5] + _THINK_BLOCK + self.answer + [_EOS, _NEWLINE]
        return {"input_ids": ids}


def _single_turn() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": ""},
        ],
        "source_id": "s",
    }


def test_generation_markers_are_detected_by_the_tag_not_the_word() -> None:
    module = _module()
    assert module.has_generation_markers(_MARKER_TEMPLATE)
    assert not module.has_generation_markers(_NO_MARKER_TEMPLATE)
    assert not module.has_generation_markers("{{- add_generation_prompt }}")


def test_thinking_is_switched_off_only_where_the_template_has_the_switch() -> None:
    module = _module()
    assert module.thinking_off_kwargs(_NO_MARKER_TEMPLATE) == {"enable_thinking": False}
    assert module.thinking_off_kwargs(_MARKER_TEMPLATE) == {}


def test_answer_span_is_the_answer_plus_the_end_of_turn_token() -> None:
    module = _module()
    prompt = [1, 1, 5, 40]
    full = prompt + [7, 8, _EOS, _NEWLINE]
    assert module.answer_span(prompt, full, _EOS) == (4, 7)


def test_answer_span_refuses_a_prompt_that_is_not_a_prefix() -> None:
    module = _module()
    with pytest.raises(ValueError, match="not a prefix"):
        module.answer_span([1, 2], [1, 3, 7, _EOS], _EOS)


def test_answer_span_refuses_an_answer_without_an_end_of_turn_token() -> None:
    module = _module()
    with pytest.raises(ValueError, match="end-of-turn"):
        module.answer_span([1], [1, 7, 8], _EOS)


def test_answer_span_refuses_an_empty_answer() -> None:
    module = _module()
    with pytest.raises(ValueError, match="empty"):
        module.answer_span([1], [1, _EOS], _EOS)


def test_without_markers_labels_cover_exactly_the_answer_and_end_of_turn() -> None:
    tokenizer = _NoMarkerTokenizer()
    row = _module().tokenize_example(tokenizer, _single_turn(), max_length=100)
    prompt_length = 4 + len(_THINK_BLOCK)
    assert row["labels"] == [-100] * prompt_length + [7, 8, 9, _EOS, -100]
    assert row["attention_mask"] == [1] * len(row["input_ids"])


def test_without_markers_rendering_switches_thinking_off() -> None:
    tokenizer = _NoMarkerTokenizer()
    _module().tokenize_example(tokenizer, _single_turn(), max_length=100)
    assert tokenizer.calls
    assert all(call.get("enable_thinking") is False for call in tokenizer.calls)


def test_without_markers_an_example_over_max_length_is_refused() -> None:
    with pytest.raises(ValueError, match="over 5"):
        _module().tokenize_example(_NoMarkerTokenizer(), _single_turn(), max_length=5)


def test_with_markers_the_assistant_mask_is_still_used() -> None:
    class _MarkerTokenizer(_Tokenizer):
        chat_template = _MARKER_TEMPLATE

        def apply_chat_template(self, messages, tools=None, **kwargs):
            assert kwargs.get("return_assistant_tokens_mask") is True
            assert "enable_thinking" not in kwargs
            return super().apply_chat_template(messages, tools, **kwargs)

    example = {"messages": [{"role": "user"}, {"role": "assistant"}], "source_id": "s"}
    row = _module().tokenize_example(_MarkerTokenizer(), example, max_length=100)
    assert row["labels"] == [-100, -100, -100, 7, 8, 9]


# --- Real tokenizers, when cached (the training environment) --------------

_QWEN = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
_LFM = ("LiquidAI/LFM2.5-350M", "9e6c6ccf47cd318696e137d381a7ded8fe4df09f")

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": kind} for key, kind in properties.items()},
                "required": list(properties),
            },
        },
    }
    for name, properties in (
        ("propose", {"operation": "string", "arguments": "object"}),
        ("explain", {"text": "string"}),
        ("escalate", {"reason": "string"}),
    )
]

#: One assistant answer per way a Tier 2 turn may end, in build_dataset.py's shape.
_ANSWERS = {
    "propose": {"operation": "restart_container", "arguments": {"name": "inference"}},
    "explain": {"text": "The GPU is idle because nothing is running on it."},
    "escalate": {"reason": "this needs the full agent"},
}


def _real_tokenizer(name: str, revision: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            name, revision=revision, local_files_only=True
        )
    except (OSError, ValueError) as error:  # not in the Hugging Face cache
        pytest.skip(f"{name} tokenizer not cached: {error}")


def _tool_example(tool: str) -> dict:
    call = {"type": "function", "function": {"name": tool, "arguments": _ANSWERS[tool]}}
    return {
        "messages": [
            {"role": "system", "content": "You are nvsh's local assistant."},
            {"role": "user", "content": "Restart the inference container"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
        ],
        "tools": _TOOLS,
        "source_id": tool,
    }


@pytest.mark.parametrize("tool", sorted(_ANSWERS))
def test_real_qwen_mask_is_answer_only(tool: str) -> None:
    tokenizer = _real_tokenizer(*_QWEN)
    module = _module()
    assert not module.has_generation_markers(tokenizer.chat_template)
    row = module.tokenize_example(tokenizer, _tool_example(tool), max_length=4096)
    kept = [token for token in row["labels"] if token != module.IGNORE_INDEX]
    assert kept, "the answer mask is empty"
    assert kept[-1] == tokenizer.eos_token_id
    answer = tokenizer.decode(kept)
    assert answer.startswith("<tool_call>\n<function=" + tool + ">")
    assert answer.endswith("</tool_call><|im_end|>")
    assert "<think>" not in answer and "<|im_start|>" not in answer
    # Masked tokens form one contiguous run at the end of the answer turn.
    first = row["labels"].index(kept[0])
    assert row["labels"][first : first + len(kept)] == kept


@pytest.mark.parametrize("tool", sorted(_ANSWERS))
def test_real_qwen_renders_an_empty_think_block_before_the_answer(tool: str) -> None:
    tokenizer = _real_tokenizer(*_QWEN)
    row = _module().tokenize_example(tokenizer, _tool_example(tool), max_length=4096)
    text = tokenizer.decode(row["input_ids"])
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>" in text


def test_real_lfm_still_uses_the_assistant_mask() -> None:
    tokenizer = _real_tokenizer(*_LFM)
    module = _module()
    assert module.has_generation_markers(tokenizer.chat_template)
    row = module.tokenize_example(tokenizer, _tool_example("propose"), max_length=4096)
    kept = [token for token in row["labels"] if token != module.IGNORE_INDEX]
    assert kept
    assert "<think>" not in tokenizer.decode(row["input_ids"])


def test_the_gpu_memory_fraction_is_the_budget_over_the_device_total() -> None:
    module = _module()
    assert module.gpu_memory_fraction("8", 128 * 2**30) == 0.0625
    assert module.gpu_memory_fraction(" 12.5 ", 100 * 2**30) == 0.125


@pytest.mark.parametrize("value", ["0", "-4", "lots", "", "nan", "inf", "129"])
def test_a_gpu_memory_budget_that_is_not_a_positive_fit_is_refused(value: str) -> None:
    module = _module()
    with pytest.raises(ValueError, match="NVSH_TRAIN_GPU_MEMORY_GB"):
        module.gpu_memory_fraction(value, 128 * 2**30)


class _FakeCuda:
    def __init__(self, total: int | None) -> None:
        self.total = total
        self.fractions: list[float] = []

    def is_available(self) -> bool:
        return self.total is not None

    def get_device_properties(self, device: int):
        return type("Props", (), {"total_memory": self.total})()

    def set_per_process_memory_fraction(self, fraction: float, device: int = 0) -> None:
        self.fractions.append(fraction)


class _FakeTorch:
    def __init__(self, total: int | None) -> None:
        self.cuda = _FakeCuda(total)


def test_the_gpu_memory_cap_is_set_from_the_environment(capsys) -> None:
    module = _module()
    torch = _FakeTorch(128 * 2**30)
    assert module.cap_gpu_memory(torch, {"NVSH_TRAIN_GPU_MEMORY_GB": "8"}) == 0.0625
    assert torch.cuda.fractions == [0.0625]
    assert "0.0625" in capsys.readouterr().err


def test_no_gpu_memory_budget_leaves_torch_alone() -> None:
    module = _module()
    torch = _FakeTorch(128 * 2**30)
    assert module.cap_gpu_memory(torch, {}) is None
    assert torch.cuda.fractions == []


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_an_empty_or_whitespace_gpu_memory_budget_is_treated_as_unset(value: str) -> None:
    """Codex finding #4: the pipeline exports the env var empty; that must not abort training."""
    module = _module()
    torch = _FakeTorch(128 * 2**30)
    assert module.cap_gpu_memory(torch, {"NVSH_TRAIN_GPU_MEMORY_GB": value}) is None
    assert torch.cuda.fractions == []


def test_a_gpu_memory_budget_over_the_device_is_refused_before_any_cap() -> None:
    module = _module()
    torch = _FakeTorch(128 * 2**30)
    with pytest.raises(ValueError, match="exceeds"):
        module.cap_gpu_memory(torch, {"NVSH_TRAIN_GPU_MEMORY_GB": "200"})
    assert torch.cuda.fractions == []


class _GenConfig:
    def __init__(self, temperature, do_sample):
        self.temperature = temperature
        self.do_sample = do_sample


class _Model:
    def __init__(self, generation_config):
        self.generation_config = generation_config


def test_a_greedy_generation_config_is_made_save_valid() -> None:
    """transformers 5.5 and 5.17 refuse to save temperature 0 with do_sample False
    (the deviation-d3 file); gen_config.py rewrites the served file after the save."""
    module = _module()
    model = _Model(_GenConfig(0.0, False))
    module.save_valid_generation_config(model)
    assert model.generation_config.temperature is None
    assert model.generation_config.do_sample is False


def test_a_sampling_or_missing_generation_config_is_left_alone() -> None:
    module = _module()
    sampling = _Model(_GenConfig(0.7, True))
    module.save_valid_generation_config(sampling)
    assert sampling.generation_config.temperature == 0.7
    bare = _Model(None)
    module.save_valid_generation_config(bare)
    assert bare.generation_config is None


# ---------------------------------------------------------------------------
# LoRA targets (issue 46, risk r9: Qwen3.5's 18 Gated-DeltaNet layers)
# ---------------------------------------------------------------------------


def test_default_targets_are_unsloths_attention_and_mlp_list() -> None:
    module = _module()
    assert module.lora_targets("attn-mlp") == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_gdn_targets_add_the_linear_attention_projections() -> None:
    module = _module()
    targets = module.lora_targets("attn-mlp-gdn")
    assert targets[:7] == module.lora_targets("attn-mlp")
    assert set(targets[7:]) == {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}


def test_unknown_targets_are_refused() -> None:
    module = _module()
    with pytest.raises(ValueError, match="targets"):
        module.lora_targets("everything")


def test_targets_option_defaults_to_attn_mlp() -> None:
    module = _module()
    args = module._parser().parse_args(["--train", "t.jsonl", "--out", "o"])
    assert args.targets == "attn-mlp"
