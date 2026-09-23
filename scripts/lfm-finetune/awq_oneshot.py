#!/usr/bin/env python3
"""INT4 AWQ (W4A16) one-shot quantization of a merged Qwen3.5 checkpoint.

Issue 46, risk r14 (t16 spike). This script runs as a **subprocess of a
separate AWQ venv's python** (llm-compressor 0.14.0, transformers 5.17.0,
compressed-tensors 0.19.0), never in-process inside ``quantize.py``: the
shared training venv pins an older transformers for training, too old for
llm-compressor's ``AWQModifier`` on this architecture. ``quantize.py``
invokes it with the python named by its ``AWQ_PY`` environment variable.

Every heavy import (``transformers``, ``llmcompressor``, ``datasets``) is
lazy, inside :func:`main` / :func:`build_recipe`, so this file can be loaded
by ``importlib`` with none of them installed -- which is exactly what
``tests/test_lfm_finetune_awq_oneshot.py`` does, since the nvsh dev
environment has neither.

    <awq venv python> scripts/lfm-finetune/awq_oneshot.py \\
        --model-dir <merged checkpoint> --calibration-file <one text per line> \\
        --out-dir <awq output dir> --num-calibration-samples 128 \\
        --max-seq-length 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: The proven recipe's targets and exclusions (spike t16): every Linear layer
#: except the LM head, the vision tower, the linear-attention path and the
#: MTP heads -- none of those three quantizes cleanly on this architecture.
AWQ_TARGETS = ["Linear"]
AWQ_SCHEME = "W4A16"
AWQ_IGNORE = ["lm_head", "re:.*visual.*", "re:.*linear_attn.*", "re:.*mtp.*"]

#: The spike's calibration rendering: max sequence length for tokenization.
MAX_SEQ_LENGTH = 512


def read_calibration_texts(path: Path) -> list[str]:
    """One calibration text per non-empty line of *path*, in file order."""
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_calibration_text(tokenizer, text: str) -> str:
    """Render *text* as a single-user-turn chat prompt (spike t16).

    ``add_generation_prompt=True`` and ``enable_thinking=False``: calibration
    samples are what the served model actually sees before it answers, not a
    completed turn with a thinking block already in it.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_recipe():
    """The proven ``AWQModifier`` recipe (spike t16). Imports llm-compressor lazily."""
    from llmcompressor.modifiers.awq import AWQModifier

    return [AWQModifier(targets=list(AWQ_TARGETS), scheme=AWQ_SCHEME, ignore=list(AWQ_IGNORE))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--calibration-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-calibration-samples", required=True, type=int)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    args = parser.parse_args(argv)

    # Heavy imports stay lazy: this module must stay importable (by importlib,
    # for tests) with none of these packages on the path.
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype="auto")

    texts = read_calibration_texts(args.calibration_file)[: args.num_calibration_samples]
    rendered = [render_calibration_text(tokenizer, text) for text in texts]
    dataset = Dataset.from_list([{"text": rendered_text} for rendered_text in rendered])
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length),
        remove_columns=["text"],
    )

    recipe = build_recipe()
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(rendered),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, save_compressed=True)
    tokenizer.save_pretrained(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
