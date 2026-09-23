"""Fine-tune LFM2.5 on the examples build_dataset.py writes (issue 39).

Training data comes from ``build_dataset.py --split <train.json>``: one JSON
object per line with ``messages``, ``tools`` and ``source_id``. Each line is
rendered and tokenized with the base model's own chat template
(``apply_chat_template(..., return_assistant_tokens_mask=True)``) and the loss
is masked to the assistant turn, so the model is trained on exactly the tool
call Tier 2 must produce and nothing else. A template without Jinja
generation-block markers (Qwen3.5, issue 46) cannot build that mask, so there
the answer is located in the rendered ids instead: the prompt is rendered
with the generation prompt and the answer is what the full rendering adds
after it, up to and including the end-of-turn token. Where the template has
an ``enable_thinking`` switch it is rendered with thinking off, which puts an
empty think block before the answer, as at inference. Tokenizing here, before a
``datasets.Dataset`` is built, matters: ``Dataset.from_list`` merges every
tool's schema and would add every other tool's parameters as ``null``
(see docs/lfm-finetune.md, run log).

This script is never imported by nvsh; it needs a training environment with
torch, unsloth and transformers (docs/lfm-finetune.md). Its pure helpers are
importable without them so they can be tested in CI.

    python scripts/lfm-finetune/train.py --train train.jsonl --val val.jsonl \
        --out runs/r1 --epochs 20 --lr 5e-4 --rank 32 --alpha 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

#: The base model and commit the stock measurement names (t11).
DEFAULT_BASE = "LiquidAI/LFM2.5-350M"
DEFAULT_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"

#: The environment variable that caps the trainer's GPU memory, in GiB.
GPU_MEMORY_ENV = "NVSH_TRAIN_GPU_MEMORY_GB"

#: The label value the loss ignores.
IGNORE_INDEX = -100

#: A Jinja generation-block tag, the marker return_assistant_tokens_mask needs.
#: Matched as a tag so ``add_generation_prompt`` does not count.
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")

#: The template variable that switches a base's thinking block on and off.
_THINKING_SWITCH = re.compile(r"\benable_thinking\b")


def read_examples(path: Path) -> list[dict]:
    """Every example in a build_dataset.py output file, checked for shape."""
    examples: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            messages = example.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{number}: no messages")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{number}: the last message is not the assistant's")
            examples.append(example)
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples


def labels_from_mask(input_ids: list[int], mask: list[int]) -> list[int]:
    """Labels that train only on masked (assistant) tokens."""
    if len(input_ids) != len(mask):
        raise ValueError("input_ids and mask differ in length")
    if not any(mask):
        raise ValueError("the assistant mask is empty: the template marked no assistant tokens")
    return [token if keep else IGNORE_INDEX for token, keep in zip(input_ids, mask)]


def has_generation_markers(template: str) -> bool:
    """Whether a chat template marks the assistant turn with generation-block tags."""
    return bool(_GENERATION_TAG.search(template))


def thinking_off_kwargs(template: str) -> dict:
    """Template arguments that switch thinking off, if the template has the switch."""
    return {"enable_thinking": False} if _THINKING_SWITCH.search(template) else {}


def answer_span(prompt_ids: list[int], full_ids: list[int], end_of_turn: int) -> tuple[int, int]:
    """Start and end of the answer in *full_ids*: what follows the prompt, through end-of-turn.

    *prompt_ids* is the rendering up to and including the generation prompt,
    *full_ids* the rendering with the answer. Refuses a prompt that does not
    tokenize as a prefix of the full rendering rather than guess the boundary.
    """
    start = len(prompt_ids)
    if full_ids[:start] != prompt_ids:
        raise ValueError(
            "the prompt's rendering is not a prefix of the full rendering:"
            " the answer cannot be located"
        )
    try:
        end = full_ids.index(end_of_turn, start) + 1
    except ValueError:
        raise ValueError("the answer has no end-of-turn token") from None
    if end - 1 == start:
        raise ValueError("the answer is empty: nothing between the prompt and end-of-turn")
    return start, end


def _template_text(tokenizer) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        return "\n".join(str(text) for text in template.values())
    return template if isinstance(template, str) else None


def tokenize_example(tokenizer, example: dict, max_length: int) -> dict:
    """input_ids, attention_mask and labels for one example, loss on the assistant turn.

    A template with generation markers (LFM2.5) masks the assistant turn
    itself. One without them (Qwen3.5) has the answer located by rendering
    the prompt alone, and the labels cover exactly the answer and its
    end-of-turn token. A tokenizer exposing no template text is taken to
    have markers, the original path.
    """
    template = _template_text(tokenizer)
    extra = thinking_off_kwargs(template) if template is not None else {}
    if template is None or has_generation_markers(template):
        rendered = tokenizer.apply_chat_template(
            example["messages"],
            tools=example.get("tools"),
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **extra,
        )
        input_ids = list(rendered["input_ids"])
        mask = list(rendered["assistant_masks"])
    else:
        prompt = tokenizer.apply_chat_template(
            example["messages"][:-1],
            tools=example.get("tools"),
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            **extra,
        )
        rendered = tokenizer.apply_chat_template(
            example["messages"],
            tools=example.get("tools"),
            tokenize=True,
            return_dict=True,
            **extra,
        )
        input_ids = list(rendered["input_ids"])
        start, end = answer_span(list(prompt["input_ids"]), input_ids, tokenizer.eos_token_id)
        mask = [int(start <= index < end) for index in range(len(input_ids))]
    if len(input_ids) > max_length:
        raise ValueError(
            f"example {example.get('source_id')!r} is {len(input_ids)} tokens, over {max_length};"
            " raise --max-length rather than cutting the assistant turn off"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels_from_mask(input_ids, mask),
    }


def gpu_memory_fraction(gb: str, total_bytes: int) -> float:
    """The share of a device of *total_bytes* that a budget of *gb* GiB is.

    capped.sh's MemoryMax does not see CUDA allocations, and on unified memory
    (GB10, Jetson) they come out of the same pool the serving stack uses, so
    the trainer caps itself (issue 46, c49). Refuses a budget that is not a
    positive number or is larger than the device.
    """
    try:
        value = float(gb)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{GPU_MEMORY_ENV}={gb!r} is not a positive number of GiB")
    budget = value * 2**30
    if budget > total_bytes:
        raise ValueError(
            f"{GPU_MEMORY_ENV}={gb!r} exceeds the device's {total_bytes / 2**30:.1f} GiB"
        )
    return budget / total_bytes


def cap_gpu_memory(torch, environ=os.environ) -> float | None:
    """Apply $NVSH_TRAIN_GPU_MEMORY_GB to CUDA device 0 before a model loads.

    Returns the fraction set, or None when the variable is unset or there is
    no CUDA device. Raises ValueError for a budget gpu_memory_fraction refuses.
    """
    gb = environ.get(GPU_MEMORY_ENV)
    if gb is None:
        return None
    if not torch.cuda.is_available():
        print(f"{GPU_MEMORY_ENV} set but no CUDA device; no GPU cap applied", file=sys.stderr)
        return None
    fraction = gpu_memory_fraction(gb, torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(f"{GPU_MEMORY_ENV}={gb.strip()}: GPU memory fraction {fraction:.4f}", file=sys.stderr)
    return fraction


def save_valid_generation_config(model) -> None:
    """Clear a greedy temperature so transformers will save the model.

    The served generation_config.json says temperature 0 with do_sample False
    (deviation d3: vLLM reads temperature, and Qwen ships no file of its own),
    and transformers 5.5 and 5.17 both refuse to save that combination. A heal
    run merges from a checkpoint that carries the file, so clear the
    temperature before the save; pipeline.sh runs gen_config.py write on the
    merged dir afterwards, which puts it back for serving.
    """
    config = getattr(model, "generation_config", None)
    if config is None:
        return
    if getattr(config, "do_sample", None) is False and getattr(config, "temperature", None) == 0:
        config.temperature = None


def merge_adapter(base: str, revision: str, adapter: Path, out: Path) -> None:  # pragma: no cover
    """Merge a saved LoRA adapter into a fresh copy of the base and save it to *out*.

    Uses plain transformers + peft rather than unsloth's merged saver, which
    copies the base weights out of the Hugging Face cache with their
    read-only permissions and then fails to overwrite them (run log, t14).
    The tokenizer, and so the chat template, is saved from the base
    unchanged; stage_cache.py checks that byte for byte.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    save_valid_generation_config(merged)
    merged.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(base, revision=revision).save_pretrained(str(out))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", type=Path, help="validation examples (loss only)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--no-merge", action="store_true", help="save the adapter only")
    parser.add_argument(
        "--merge-only",
        type=Path,
        metavar="ADAPTER",
        help="skip training; merge this saved adapter into the base and write --out/merged",
    )
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU stack
    args = _parser().parse_args(argv)
    if args.merge_only is not None:
        merge_adapter(args.base, args.revision, args.merge_only, args.out / "merged")
        print(f"merged {args.merge_only} into {args.out / 'merged'}")
        return 0

    # Imported here so the helpers above work without a training environment.
    # unsloth must come first: importing it patches transformers.
    # isort: off
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    # isort: on

    try:
        cap_gpu_memory(torch)
    except ValueError as exc:
        print(f"train.py: {exc}", file=sys.stderr)
        return 2
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.base, revision=args.revision, max_seq_length=args.max_length, load_in_4bit=False
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, random_state=args.seed
    )

    def dataset(path: Path) -> Dataset:
        rows = [tokenize_example(tokenizer, ex, args.max_length) for ex in read_examples(path)]
        return Dataset.from_list(rows)

    train_ds = dataset(args.train)
    val_ds = dataset(args.val) if args.val else None
    args.out.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        seed=args.seed,
        logging_steps=5,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX
        ),
    )
    started = time.time()
    trainer.train()
    seconds = time.time() - started

    model.save_pretrained(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    if not args.no_merge:
        merge_adapter(args.base, args.revision, args.out / "adapter", args.out / "merged")

    log = {
        "base": args.base,
        "revision": args.revision,
        "train_file": str(args.train),
        "val_file": str(args.val) if args.val else None,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds) if val_ds is not None else 0,
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "rank": args.rank,
            "alpha": args.alpha,
            "batch": args.batch,
            "seed": args.seed,
            "max_length": args.max_length,
        },
        "seconds": round(seconds, 1),
        "max_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "history": trainer.state.log_history,
    }
    (args.out / "train-log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
