"""Train the Track B candidate scorer: a LoRA on label-restricted cross-entropy (issue 46).

Training data is the same split files ``split.py`` writes (``train.json``,
and ``val.json`` for validation), read with ``nvsh.tiers.bench``'s own
loaders. Each entry becomes one prompt from ``scorer.py``
(:func:`scorer.prompt_messages` over every candidate, the request text
being ``nvsh.tiers.lfm.request_message``, which is what ``build_dataset.py``
trains Track A on) and one target: the index of its gold candidate.

The loss reads the logits of the label tokens only, at the one position
after the prompt: the model is asked for ``logits_to_keep`` of just each
row's last real position, and a cross-entropy over the candidate labels'
columns is taken there. Qwen3.5's vocabulary is 248,320 tokens; the full
vocabulary is never computed over the whole sequence, which is what keeps a
run inside the memory spark2 has spare beside its serving stack.

Runs are seeded (Python, torch and the LoRA initialisation) and the batch
order is drawn from the seed, so one seed on one machine repeats. The
train-log records each split file's sha256 next to the hyperparameters.

This script is never imported by nvsh; it needs torch, transformers and
peft (docs/lfm-finetune.md). Its helpers import them lazily.

    python scripts/lfm-finetune/train_scorer.py --train train.json --val val.json \
        --out runs/b1 --epochs 3 --lr 2e-4 --rank 16 --alpha 32
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))  # runnable from any directory

from nvsh.tiers import lfm  # noqa: E402
from nvsh.tiers.bench import context_for, load_corpus, request_for  # noqa: E402


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


scorer = _sibling("scorer")
# The GPU memory cap is shared with train.py (issue 46, c49); train.py imports
# nothing heavy at module level, so loading it here stays cheap.
_train = _sibling("train")
gpu_memory_fraction = _train.gpu_memory_fraction
cap_gpu_memory = _train.cap_gpu_memory

#: The base model and commit issue 46 pins.
DEFAULT_BASE = "Qwen/Qwen3.5-0.8B"
DEFAULT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

TRAIN_SIDE = "train"
VAL_SIDE = "val"

#: ``split.py``'s note in a side's header: ``Split '<side>' of <corpus> (seed=N).``
_SPLIT_SIDE_RE = re.compile(r"Split '(\w+)' of ")

#: The held-out split's file name and the phrase its header opens with.
HELD_OUT_NAME = "held-out.json"
_HELD_OUT_MARKER = "held-out split"


@dataclass(frozen=True)
class Example:
    """One entry of a split: the request as the model reads it, and its gold candidate."""

    entry_id: str
    request: str
    gold: str


def gold_candidate(expect: dict) -> str:
    """The candidate an entry's ``expect`` block names (split.py's three kinds)."""
    if expect.get("escalate"):
        return lfm.ESCALATE_TOOL
    if expect.get("explain"):
        return lfm.EXPLAIN_TOOL
    if "operation" in expect:
        name = expect["operation"]
        if name not in scorer.candidates():
            raise ValueError(f"{name!r} is not a candidate")
        return name
    raise ValueError(f"an expect block nvsh does not recognise: {expect!r}")


def read_split(path: Path, side: str) -> list[Example]:
    """Every entry of a split file that must be *side*; refuses the held-out split.

    A file whose header names no side, or another side, is refused: the
    trainer reads the train side, validation reads the val side, and the
    test and held-out sides are never read here.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    header = raw.get("header") if isinstance(raw, dict) else None
    header = header if isinstance(header, str) else ""
    if path.name == HELD_OUT_NAME or header.casefold().startswith(_HELD_OUT_MARKER):
        raise ValueError(f"{path}: the held-out split is never read by the trainer")
    match = _SPLIT_SIDE_RE.search(header)
    found = match.group(1) if match else None
    if found != side:
        raise ValueError(f"{path}: its header names side {found!r}, expected {side!r}")
    loaded = load_corpus(path)
    if not loaded.entries:
        raise ValueError(f"{path}: no entries")
    return [
        Example(
            entry_id=entry.id,
            request=lfm.request_message(request_for(entry), context_for(entry)),
            gold=gold_candidate(entry.expect),
        )
        for entry in loaded.entries
    ]


def encode(tokenizer, examples: list[Example], max_length: int) -> list[dict]:
    """``{"input_ids", "target"}`` per example: the rendered prompt and its gold index."""
    names = scorer.candidates()
    rows: list[dict] = []
    for example in examples:
        prompt = scorer.render_prompt(tokenizer, scorer.prompt_messages(example.request))
        input_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if len(input_ids) > max_length:
            raise ValueError(
                f"entry {example.entry_id!r} is {len(input_ids)} tokens, over {max_length};"
                " raise --max-length rather than cutting the request"
            )
        rows.append({"input_ids": input_ids, "target": names.index(example.gold)})
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- the loss and the loop (torch) --


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def pad_right(sequences: list[list[int]], pad_id: int):
    """``(input_ids, attention_mask)`` tensors, padded on the right."""
    import torch

    width = max(len(sequence) for sequence in sequences)
    ids = [sequence + [pad_id] * (width - len(sequence)) for sequence in sequences]
    mask = [[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences]
    return torch.tensor(ids), torch.tensor(mask)


def label_logits(model, input_ids, attention_mask, label_ids):
    """``(batch, labels)`` logits of the label tokens at each row's last real position.

    Only the distinct last positions in the batch are kept (a tensor
    ``logits_to_keep``), so the head runs on at most *batch* positions.
    Padding is on the right: every layer is causal (Qwen3.5's linear-attention
    layers included), so the pad tokens after a row's last real token cannot
    change what the model computes there.
    """
    import torch

    last = attention_mask.sum(dim=1) - 1
    keep, where = torch.unique(last, return_inverse=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep).logits
    rows = logits[torch.arange(input_ids.shape[0], device=logits.device), where]
    return rows.index_select(-1, label_ids.to(rows.device)).float()


def _batches(rows: list[dict], batch: int, order: list[int]):
    for start in range(0, len(order), batch):
        yield [rows[index] for index in order[start : start + batch]]


def _step_inputs(model, chunk: list[dict], pad_id: int):
    import torch

    device = next(model.parameters()).device
    input_ids, mask = pad_right([row["input_ids"] for row in chunk], pad_id)
    targets = torch.tensor([row["target"] for row in chunk], device=device)
    return input_ids.to(device), mask.to(device), targets


def train_loop(
    model,
    rows: list[dict],
    label_ids: list[int],
    *,
    epochs: int,
    lr: float,
    batch: int,
    seed: int,
    pad_id: int,
) -> list[dict]:
    """AdamW over label-restricted cross-entropy; returns ``[{epoch, step, loss}]``."""
    import torch

    labels = torch.tensor(label_ids)
    order_rng = random.Random(seed)  # nosec B311 - batch order, not security
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history: list[dict] = []
    step = 0
    for epoch in range(epochs):
        order = list(range(len(rows)))
        order_rng.shuffle(order)
        for chunk in _batches(rows, batch, order):
            input_ids, mask, targets = _step_inputs(model, chunk, pad_id)
            logits = label_logits(model, input_ids, mask, labels)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1
            history.append({"epoch": epoch + 1, "step": step, "loss": loss.item()})
    return history


def evaluate(model, rows: list[dict], label_ids: list[int], *, batch: int, pad_id: int) -> dict:
    """Mean loss, accuracy and mean top probability over *rows* (all candidates offered)."""
    import torch

    labels = torch.tensor(label_ids)
    model.eval()
    total_loss = 0.0
    right = 0
    confidence = 0.0
    with torch.no_grad():
        for chunk in _batches(rows, batch, list(range(len(rows)))):
            input_ids, mask, targets = _step_inputs(model, chunk, pad_id)
            logits = label_logits(model, input_ids, mask, labels)
            total_loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
            probabilities = torch.softmax(logits, dim=-1)
            top, picked = probabilities.max(dim=-1)
            right += int((picked == targets).sum())
            confidence += float(top.sum())
    model.train()
    n = len(rows)
    return {
        "n": n,
        "loss": total_loss / n,
        "accuracy": right / n,
        "mean_confidence": confidence / n,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, type=Path, help="split.py's train side")
    parser.add_argument("--val", type=Path, help="split.py's val side (loss and accuracy)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU stack
    args = _parser().parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        cap_gpu_memory(torch)
    except ValueError as exc:
        print(f"train_scorer.py: {exc}", file=sys.stderr)
        return 2
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    labels = scorer.labels_for(scorer.candidates())
    ids = scorer.label_token_ids(tokenizer, labels)
    label_ids = [ids[name] for name in scorer.candidates()]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_rows = encode(tokenizer, read_split(args.train, TRAIN_SIDE), args.max_length)
    val_rows = (
        encode(tokenizer, read_split(args.val, VAL_SIDE), args.max_length) if args.val else []
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.revision, dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank, lora_alpha=args.alpha, target_modules="all-linear", task_type="CAUSAL_LM"
        ),
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    started = time.time()
    history = train_loop(
        model,
        train_rows,
        label_ids,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        seed=args.seed,
        pad_id=pad_id,
    )
    seconds = time.time() - started
    val = (
        evaluate(model, val_rows, label_ids, batch=args.batch, pad_id=pad_id) if val_rows else None
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    log = {
        "base": args.base,
        "revision": args.revision,
        "objective": "cross-entropy over the candidate label tokens at the decision position",
        "candidates": list(scorer.candidates()),
        "labels": labels,
        "label_token_ids": ids,
        "train_file": str(args.train),
        "train_sha256": file_sha256(args.train),
        "val_file": str(args.val) if args.val else None,
        "val_sha256": file_sha256(args.val) if args.val else None,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
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
        "max_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available()
            else None
        ),
        "val": val,
        "history": history,
    }
    (args.out / "train-log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
