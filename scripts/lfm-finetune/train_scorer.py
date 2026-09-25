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

**Per-row label maps (issue 53).** A split entry may carry its own
``"permutation"`` (``scorer.Permutation.to_json()``: the offered candidates
in listing order and their letters), a ``"gold"`` candidate name (an
operation, ``explain``, ``escalate`` or ``escalate:<reason>``), a
``"descriptions"`` override and its ``"perm_seed"``. Such a row's prompt is
rendered from its own map, its label columns are the letters in its own
order and its target is the gold's position there; a row without one keeps
today's fixed full candidate map and prompt. Rows in a batch may offer
different numbers of labels: the label logits are padded per row, and a
padded column is ``-inf`` so it gets no probability and no loss.

**The label readout.** ``--label-readout variants`` (the default) is the
shared definition in ``scorer.py``: a label's logit is the log-sum-exp of
every token whose stripped text is the letter (``"A"``, ``" A"``,
``"\tA"``), :func:`scorer.label_logits_from_vocab`, so training optimises
the distribution the scorer reads. ``--label-readout single`` reads the one
id of the bare letter; with rows that carry no permutation it is exactly
scorer-b1's loss, kept so that run stays reproducible.

**Calibration terms.** ``--label-smoothing`` (over each row's offered labels
only) and ``--brier-weight`` (a multi-class Brier score over the offered
labels, added to the cross-entropy) are both 0, off, by default; with both
off the loss is plain cross-entropy.

Runs are seeded (Python, torch and the LoRA initialisation) and the batch
order is drawn from the seed, so one seed on one machine repeats. The
train-log records each split file's sha256 next to the hyperparameters, the
readout and its token ids, the permutation seeds seen and how many rows were
permuted; every permuted row's map is in ``row-maps.json`` beside it (path
and sha256 in the log), since a large set's maps would swamp the log.

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
import math
import random
import re
import sys
import time
import dataclasses
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
    """One entry of a split: the request as the model reads it, and its gold candidate.

    ``permutation`` is the row's own offered candidates and letters (``None``:
    the fixed full map), ``descriptions`` its prompt-description overrides and
    ``perm_seed`` the seed its permutation was drawn with, all as stored.
    """

    entry_id: str
    request: str
    gold: str
    permutation: scorer.Permutation | None = None
    descriptions: dict[str, str] | None = None
    perm_seed: int | None = None
    cls: str | None = None


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
    raw_entries = raw.get("entries") if isinstance(raw, dict) else raw
    stored = {
        str(item["id"]): item
        for item in (raw_entries if isinstance(raw_entries, list) else [])
        if isinstance(item, dict) and "id" in item
    }
    return [_example(entry, stored.get(entry.id, {})) for entry in loaded.entries]


def _example(entry, item: dict) -> Example:
    """One :class:`Example` from a loaded entry and its raw item's per-row fields."""
    expected = gold_candidate(entry.expect)
    gold = item.get("gold", expected)
    if not isinstance(gold, str) or gold.split(":", 1)[0] != expected:
        raise ValueError(f"entry {entry.id!r}: gold {gold!r} disagrees with its expect block")
    permutation = None
    if item.get("permutation") is not None:
        permutation = scorer.Permutation.from_json(item["permutation"])
        letters = [permutation.labels.get(name) for name in permutation.order]
        if None in letters or len(set(letters)) != len(letters):
            raise ValueError(
                f"entry {entry.id!r}: its permutation must give each a distinct letter"
            )
        if gold not in permutation.order:
            raise ValueError(f"entry {entry.id!r}: gold {gold!r} is not offered by its permutation")
    descriptions = item.get("descriptions")
    perm_seed = item.get("perm_seed")
    return Example(
        entry_id=entry.id,
        request=lfm.request_message(request_for(entry), context_for(entry)),
        gold=gold,
        permutation=permutation,
        descriptions=dict(descriptions) if descriptions is not None else None,
        perm_seed=perm_seed,
        cls=item.get("class") if isinstance(item.get("class"), str) else None,
    )


def reasons_mode(examples: list[Example]) -> bool:
    """True when any row offers an ``escalate:<reason>`` candidate (``--reasons`` data)."""
    return any(
        example.permutation is not None
        and any(name in scorer.REASON_CANDIDATES for name in example.permutation.order)
        for example in examples
    )


def match_validation(train: list[Example], val: list[Example]) -> list[Example]:
    """*val*, scored on the same candidate pool the train rows use (PR #65 review).

    In reasons mode a validation row with no stored map gets the reasons pool's
    default map, its reason descriptions, and an escalate gold named by its
    class (:func:`scorer.reason_for_class`) -- what ``measure.py --reasons``
    scores -- rather than the fixed map with a bare ``escalate`` the model never
    trained on. Otherwise *val* is returned unchanged.
    """
    if not reasons_mode(train):
        return val
    pool = scorer.candidate_pool(reasons=True)
    permutation = scorer.Permutation(order=tuple(pool), labels=scorer.positional_labels(pool, pool))
    matched = []
    for example in val:
        if example.permutation is not None:
            matched.append(example)
            continue
        gold = example.gold
        if gold == lfm.ESCALATE_TOOL:
            gold = scorer.reason_for_class(example.cls)
        matched.append(
            dataclasses.replace(
                example,
                gold=gold,
                permutation=permutation,
                descriptions={**scorer.reason_descriptions(pool), **(example.descriptions or {})},
            )
        )
    return matched


def example_messages(example: Example) -> list[dict]:
    """The prompt messages *example* is trained on: its own map, else today's fixed one.

    ``measure.py`` (``scorer_request``) must render byte-identical messages
    for the same candidate list and order; a test pins the two together.
    """
    permutation = example.permutation
    return scorer.prompt_messages(
        example.request,
        labels=permutation.labels if permutation is not None else None,
        order=permutation.order if permutation is not None else None,
        descriptions=example.descriptions,
    )


def encode(tokenizer, examples: list[Example], max_length: int) -> list[dict]:
    """``{"input_ids", "target", "letters"}`` per example, from that example's own map.

    ``letters`` are the row's label letters in its listing order and
    ``target`` the gold's position in that order. An example with no
    permutation renders today's prompt over every candidate, fixed map.
    """
    fixed_names = scorer.candidates()
    fixed_labels = scorer.labels_for(fixed_names)
    rows: list[dict] = []
    for example in examples:
        permutation = example.permutation
        order = permutation.order if permutation is not None else fixed_names
        labels = permutation.labels if permutation is not None else fixed_labels
        prompt = scorer.render_prompt(tokenizer, example_messages(example))
        input_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        if len(input_ids) > max_length:
            raise ValueError(
                f"entry {example.entry_id!r} is {len(input_ids)} tokens, over {max_length};"
                " raise --max-length rather than cutting the request"
            )
        if example.gold not in order:
            raise ValueError(f"entry {example.entry_id!r}: gold {example.gold!r} is not offered")
        rows.append(
            {
                "input_ids": input_ids,
                "target": list(order).index(example.gold),
                "letters": [labels[name] for name in order],
            }
        )
    return rows


READOUTS = ("variants", "single")


def letter_ids(tokenizer, letters, readout: str) -> dict[str, tuple[int, ...]]:
    """Letter -> the token ids its label logit reads, under *readout*.

    ``variants``: every token whose stripped text is the letter
    (:func:`scorer.label_variant_ids`, the shared readout definition).
    ``single``: the one id of the bare letter (:func:`scorer.label_token_ids`),
    scorer-b1's readout.
    """
    wanted = {letter: letter for letter in sorted(set(letters))}
    if readout == "variants":
        return dict(scorer.label_variant_ids(tokenizer, wanted))
    if readout == "single":
        return {letter: (i,) for letter, i in scorer.label_token_ids(tokenizer, wanted).items()}
    raise ValueError(f"unknown label readout {readout!r}; expected one of {READOUTS}")


def attach_columns(rows: list[dict], ids: dict[str, tuple[int, ...]]) -> list[dict]:
    """Copies of *rows*, each with ``columns``: its letters' token ids, in its order."""
    return [{**row, "columns": [tuple(ids[letter]) for letter in row["letters"]]} for row in rows]


def permutation_record(sides: dict[str, list[Example]]) -> tuple[dict, list[dict]]:
    """``(summary, maps)``: the permutation seeds and counts, and every permuted row's map.

    The summary goes in train-log.json; the maps (one per permuted row, with
    its side, entry id, seed, order, letters, gold, target and which
    descriptions it overrides) go in their own file.
    """
    seeds: set = set()
    permuted: dict[str, int] = {}
    fixed: dict[str, int] = {}
    maps: list[dict] = []
    for side, examples in sides.items():
        permuted[side] = fixed[side] = 0
        for example in examples:
            if example.permutation is None:
                fixed[side] += 1
                continue
            permuted[side] += 1
            if example.perm_seed is not None:
                seeds.add(example.perm_seed)
            order = list(example.permutation.order)
            maps.append(
                {
                    "side": side,
                    "entry_id": example.entry_id,
                    "perm_seed": example.perm_seed,
                    "order": order,
                    "labels": {name: example.permutation.labels[name] for name in order},
                    "gold": example.gold,
                    "target": order.index(example.gold),
                    "descriptions": sorted(example.descriptions or {}),
                }
            )
    summary = {
        "perm_seeds": sorted(seeds, key=lambda seed: (str(type(seed)), seed)),
        "permuted_rows": permuted,
        "fixed_rows": fixed,
    }
    return summary, maps


def write_row_maps(path: Path, maps: list[dict]) -> dict:
    """Write *maps* to *path*; ``{"file", "sha256", "rows"}`` for the train-log."""
    path.write_text(json.dumps(maps, indent=1) + "\n", encoding="utf-8")
    return {"file": str(path), "sha256": file_sha256(path), "rows": len(maps)}


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
    rows = last_logits(model, input_ids, attention_mask)
    return rows.index_select(-1, label_ids.to(rows.device)).float()


def column_logits(vocab_rows, columns: list[list[tuple[int, ...]]]):
    """``(logits, mask)``: each row's label logits from its ``(vocab,)`` logits, padded.

    ``columns[i]`` is row *i*'s label columns, each a tuple of token ids. When
    every column is one id (the ``single`` readout) the logits are those ids'
    entries, read exactly as scorer-b1's loss read them; otherwise each is the
    log-sum-exp of its ids (:func:`scorer.label_logits_from_vocab`). Columns
    past a row's own count are ``-inf`` and ``False`` in *mask*.
    """
    import torch

    device = vocab_rows.device
    width = max(len(row) for row in columns)
    mask = torch.tensor(
        [[j < len(row) for j in range(width)] for row in columns], dtype=torch.bool, device=device
    )
    if all(len(ids) == 1 for row in columns for ids in row):
        index = torch.tensor(
            [[ids[0] for ids in row] + [0] * (width - len(row)) for row in columns], device=device
        )
        logits = vocab_rows.gather(-1, index).float()
    else:
        groups: dict[tuple, list[int]] = {}
        for i, row in enumerate(columns):
            groups.setdefault(tuple(tuple(ids) for ids in row), []).append(i)
        pieces: list = [None] * len(columns)
        for key, members in groups.items():
            part = scorer.label_logits_from_vocab(vocab_rows[members].float(), key)
            part = torch.nn.functional.pad(part, (0, width - len(key)), value=-math.inf)
            for j, member in enumerate(members):
                pieces[member] = part[j]
        logits = torch.stack(pieces)
    return logits.masked_fill(~mask, -math.inf), mask


def label_loss(logits, mask, targets, *, label_smoothing: float = 0.0, brier_weight: float = 0.0):
    """Mean label-restricted loss: cross-entropy, plus the optional calibration terms.

    With both terms 0 (the default) this is ``cross_entropy(logits, targets)``
    unchanged. *label_smoothing* mixes that share of a uniform target over the
    row's offered (unmasked) labels; *brier_weight* adds that multiple of the
    multi-class Brier score ``sum_j (p_j - y_j)^2`` over the offered labels.
    """
    import torch

    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label smoothing must be in [0, 1), not {label_smoothing}")
    if brier_weight < 0.0:
        raise ValueError(f"brier weight must be at least 0, not {brier_weight}")
    if label_smoothing == 0.0 and brier_weight == 0.0:
        return torch.nn.functional.cross_entropy(logits, targets)
    log_p = torch.log_softmax(logits, dim=-1)
    offered_log_p = torch.where(mask, log_p, torch.zeros_like(log_p))
    per_row = -offered_log_p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if label_smoothing:
        uniform = -offered_log_p.sum(dim=-1) / mask.sum(dim=-1)
        per_row = (1.0 - label_smoothing) * per_row + label_smoothing * uniform
    if brier_weight:
        gold = torch.nn.functional.one_hot(targets, logits.shape[-1]).to(log_p.dtype)
        brier = ((log_p.exp() - gold) ** 2 * mask).sum(dim=-1)
        per_row = per_row + brier_weight * brier
    return per_row.mean()


def last_logits(model, input_ids, attention_mask):
    """``(batch, vocab)`` logits at each row's last real position (see :func:`label_logits`)."""
    import torch

    last = attention_mask.sum(dim=1) - 1
    keep, where = torch.unique(last, return_inverse=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep).logits
    return logits[torch.arange(input_ids.shape[0], device=logits.device), where]


def _batches(rows: list[dict], batch: int, order: list[int]):
    for start in range(0, len(order), batch):
        yield [rows[index] for index in order[start : start + batch]]


def _step_inputs(model, chunk: list[dict], pad_id: int):
    import torch

    device = next(model.parameters()).device
    input_ids, mask = pad_right([row["input_ids"] for row in chunk], pad_id)
    targets = torch.tensor([row["target"] for row in chunk], device=device)
    return input_ids.to(device), mask.to(device), targets


def _default_columns(label_ids) -> list[tuple[int, ...]] | None:
    """*label_ids* (one id, or a tuple of ids, per column) as columns; ``None`` stays ``None``."""
    if label_ids is None:
        return None
    return [(ids,) if isinstance(ids, int) else tuple(ids) for ids in label_ids]


def _chunk_logits(model, chunk: list[dict], default, pad_id: int):
    """``(logits, mask, targets)`` for one batch: each row's own columns, else *default*."""
    input_ids, attention_mask, targets = _step_inputs(model, chunk, pad_id)
    columns = [row.get("columns", default) for row in chunk]
    if any(row is None for row in columns):
        raise ValueError("a row has no label columns and no default label_ids was given")
    logits, mask = column_logits(last_logits(model, input_ids, attention_mask), columns)
    return logits, mask, targets


def train_loop(
    model,
    rows: list[dict],
    label_ids=None,
    *,
    epochs: int,
    lr: float,
    batch: int,
    seed: int,
    pad_id: int,
    label_smoothing: float = 0.0,
    brier_weight: float = 0.0,
) -> list[dict]:
    """AdamW over the label-restricted loss; returns ``[{epoch, step, loss}]``.

    Each row reads its own ``columns`` (:func:`attach_columns`); *label_ids*
    (one id or a tuple of ids per column) serves rows that carry none, the
    pre-issue-53 fixed map. The loss is :func:`label_loss`.
    """
    import torch

    default = _default_columns(label_ids)
    order_rng = random.Random(seed)  # nosec B311 - batch order, not security
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history: list[dict] = []
    step = 0
    for epoch in range(epochs):
        order = list(range(len(rows)))
        order_rng.shuffle(order)
        for chunk in _batches(rows, batch, order):
            logits, mask, targets = _chunk_logits(model, chunk, default, pad_id)
            loss = label_loss(
                logits,
                mask,
                targets,
                label_smoothing=label_smoothing,
                brier_weight=brier_weight,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1
            history.append({"epoch": epoch + 1, "step": step, "loss": loss.item()})
    return history


def evaluate(model, rows: list[dict], label_ids=None, *, batch: int, pad_id: int) -> dict:
    """Mean cross-entropy, accuracy and mean top probability over each row's offered labels."""
    import torch

    default = _default_columns(label_ids)
    model.eval()
    total_loss = 0.0
    right = 0
    confidence = 0.0
    with torch.no_grad():
        for chunk in _batches(rows, batch, list(range(len(rows)))):
            logits, _mask, targets = _chunk_logits(model, chunk, default, pad_id)
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


def _smoothing(text: str) -> float:
    value = float(text)
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"label smoothing must be in [0, 1), not {value}")
    return value


def _non_negative(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"must be at least 0, not {value}")
    return value


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
    parser.add_argument(
        "--label-readout",
        choices=READOUTS,
        default="variants",
        help="variants: log-sum-exp of every token that is the letter (scorer.py's definition);"
        " single: the bare letter's one id (scorer-b1's loss)",
    )
    calibration = parser.add_argument_group("calibration loss (both off by default)")
    calibration.add_argument(
        "--label-smoothing",
        type=_smoothing,
        default=0.0,
        help="label smoothing over each row's offered labels, in [0, 1)",
    )
    calibration.add_argument(
        "--brier-weight",
        type=_non_negative,
        default=0.0,
        help="weight of a multi-class Brier term over the offered labels, added to CE",
    )
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
    variants = scorer.label_variant_ids(tokenizer, labels)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_examples = read_split(args.train, TRAIN_SIDE)
    val_examples = read_split(args.val, VAL_SIDE) if args.val else []
    val_examples = match_validation(train_examples, val_examples)
    train_rows = encode(tokenizer, train_examples, args.max_length)
    val_rows = encode(tokenizer, val_examples, args.max_length)
    readout_ids = letter_ids(
        tokenizer,
        [letter for row in train_rows + val_rows for letter in row["letters"]],
        args.label_readout,
    )
    train_rows = attach_columns(train_rows, readout_ids)
    val_rows = attach_columns(val_rows, readout_ids)
    perm_summary, row_maps = permutation_record(
        {TRAIN_SIDE: train_examples, VAL_SIDE: val_examples}
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
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        seed=args.seed,
        pad_id=pad_id,
        label_smoothing=args.label_smoothing,
        brier_weight=args.brier_weight,
    )
    seconds = time.time() - started
    val = evaluate(model, val_rows, batch=args.batch, pad_id=pad_id) if val_rows else None

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    objective = "cross-entropy over each row's offered label tokens at the decision position"
    if args.label_smoothing or args.brier_weight:
        objective += ", with calibration terms"
    log = {
        "base": args.base,
        "revision": args.revision,
        "objective": objective,
        "candidates": list(scorer.candidates()),
        "labels": labels,
        "label_token_ids": ids,
        "label_variant_ids": {name: list(found) for name, found in variants.items()},
        "label_readout": args.label_readout,
        "letter_ids": {letter: list(found) for letter, found in readout_ids.items()},
        "calibration_loss": {
            "label_smoothing": args.label_smoothing,
            "brier_weight": args.brier_weight,
        },
        "permutations": {
            **perm_summary,
            "row_maps": write_row_maps(args.out / "row-maps.json", row_maps),
        },
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
