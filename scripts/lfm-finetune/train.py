"""Fine-tune LFM2.5 on the examples build_dataset.py writes (issue 39).

Training data comes from ``build_dataset.py --split <train.json>``: one JSON
object per line with ``messages``, ``tools`` and ``source_id``. Each line is
rendered and tokenized with the base model's own chat template
(``apply_chat_template(..., return_assistant_tokens_mask=True)``) and the loss
is masked to the assistant turn, so the model is trained on exactly the tool
call Tier 2 must produce and nothing else. Tokenizing here, before a
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
import sys
import time
from pathlib import Path

#: The base model and commit the stock measurement names (t11).
DEFAULT_BASE = "LiquidAI/LFM2.5-350M"
DEFAULT_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"

#: The label value the loss ignores.
IGNORE_INDEX = -100


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


def tokenize_example(tokenizer, example: dict, max_length: int) -> dict:
    """input_ids, attention_mask and labels for one example, loss on the assistant turn."""
    rendered = tokenizer.apply_chat_template(
        example["messages"],
        tools=example.get("tools"),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = list(rendered["input_ids"])
    mask = list(rendered["assistant_masks"])
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
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU stack
    args = _parser().parse_args(argv)

    # Imported here so the helpers above work without a training environment.
    # unsloth must come first: importing it patches transformers.
    # isort: off
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    # isort: on

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
        model.save_pretrained_merged(
            str(args.out / "merged"), tokenizer, save_method="merged_16bit"
        )

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
