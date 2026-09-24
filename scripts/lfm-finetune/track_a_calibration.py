#!/usr/bin/env python3
"""Exact Track A candidate distributions, scored in process (issue 46, deviation d6).

A development-machine tool for the Qwen3.5-0.8B tool-decision experiment
(part of #46). It is NEVER imported by the nvsh package -- nothing under nvsh/
may depend on it.

Usage::

    python scripts/lfm-finetune/track_a_calibration.py --model runs/r1/merged \\
        --split work/splits/val.json --predictions out/predictions-r1.jsonl \\
        --out out/predictions-r1-exact.jsonl

Why: Track A (the generative model) decides by writing a tool call, and its
calibration (ECE/Brier in ``metrics.py``) needs a probability for every
candidate. Qwen3.5 splits most operation names into several tokens
(``propose`` is ``prop`` + ``ose``, ``escalate`` is ``escal`` + ``ate``), and
a served model returns log-probabilities only along the path it generated, so
measure.py can read an exact distribution from a served reply only in the
cases its path covers. Deviation d6 measures it here instead, exactly: every
candidate's continuation is teacher-forced under the model in process.

For each entry of ``--split`` (the entries measure.py runs), the prompt is
rendered the way Track A saw it in training: the conversation comes from
``build_dataset.py``'s own ``example_from_entry`` (system brief, tools, user
message) and is rendered with the tokenizer's chat template with the
generation prompt and thinking off, the switches ``train.py`` uses. The
candidates are ``scorer.py``'s: every operation in ``nvsh/ops/table.py``,
then ``explain`` and ``escalate``. Each candidate's continuation is cut from
the template's own rendering of that candidate as a real assistant turn
(built by ``build_dataset.py``'s ``answer_for``, arguments as an object, the
pipeline's default), never from hand-written tag strings: for an operation,
the ``propose`` call through the operation's name and the character that
ends it (a newline in Qwen3.5's XML form); for a control, the call through
the tool's name and the character that ends it (the closing ``>``). That
character is what proves the choice -- ``gpu`` is not yet ``gpu_stats``.

The continuation's token log-probabilities are summed (one forward pass per
candidate) and a softmax over the candidates' sums gives the distribution,
keyed by ``metrics.py``'s labels (``scorer.calibration_label``). A cut whose
tokens are not exactly the tokens of the full rendering is refused rather
than scored: the model must be scored on the tokens it was trained on.

``--predictions`` is a file measure.py wrote for the same model and split;
each line's ``candidates`` is replaced with the exact distribution and every
other field is kept. The split's side is checked with measure.py's own
refusal (test needs ``--final``, held-out needs ``--acceptance``). A sidecar
JSON next to ``--out`` records the model directory and its revision (the
same content hash ``stage_cache.py`` stages it under), the split's and the
input file's sha256, and the torch/transformers versions.

Its torch/transformers imports are lazy, so the helpers work without a
training environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.platform._model import Platform  # noqa: E402
from nvsh.tiers import bench as tier_bench  # noqa: E402
from nvsh.tiers import lfm  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


build_dataset = _sibling("build_dataset")
train = _sibling("train")
scorer = _sibling("scorer")
metrics = _sibling("metrics")
measure = _sibling("measure")
stage_cache = _sibling("stage_cache")

#: What an explain candidate's turn says when the entry carries no answer of
#: its own. The cut ends before the text, so it never reaches the scored tokens.
_EXPLAIN_FILLER = "-"

#: How the sidecar names the method, so a reader knows these are not served logprobs.
METHOD = "exact in-process teacher-forced scoring over every candidate (deviation d6)"


class ContinuationScorer(Protocol):
    """The one method scoring needs: a continuation's summed token log-probability."""

    def logprob(self, prompt_ids: Sequence[int], continuation_ids: Sequence[int]) -> float:
        """Natural-log probability of *continuation_ids* following *prompt_ids*."""


@dataclass(frozen=True)
class Continuation:
    """One candidate's scored span: the shared prompt's ids, then the text and ids that decide."""

    prompt_ids: list[int]
    text: str
    continuation_ids: list[int]


# -- the conversation --


def _candidate_entry(entry: tier_bench.CorpusEntry, name: str) -> tier_bench.CorpusEntry:
    """*entry* with its expectation swapped for candidate *name*'s, so answer_for renders it."""
    if name == lfm.EXPLAIN_TOOL:
        answer = entry.expect.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = _EXPLAIN_FILLER
        expect: dict = {"explain": True, "answer": answer}
    elif name == lfm.ESCALATE_TOOL:
        expect = {"escalate": True}
    else:
        expect = {"operation": name, "args": {}}
    return dataclasses.replace(entry, expect=expect)


def _whole_word(text: str, name: str, start: int) -> int:
    """The first whole-word *name* in *text* at or after *start* (measure.py's rule), or -1."""
    return measure._find_name(text, name, start)


def decisive_end(turn: str, name: str) -> int:
    """Where candidate *name*'s deciding span ends in its rendered assistant *turn*.

    The span runs through the tool's name (``propose`` for an operation) and,
    for an operation, on through the operation's name, then one character
    more: the one the template writes to close the name.
    """
    tool = name if name in scorer.CONTROLS else lfm.PROPOSE_TOOL
    at = _whole_word(turn, tool, 0)
    if at < 0:
        raise ValueError(f"the rendered turn for {name!r} does not name {tool!r}: {turn!r}")
    end = at + len(tool)
    if tool != name:
        at = _whole_word(turn, name, end)
        if at < 0:
            raise ValueError(f"the rendered proposal does not name {name!r}: {turn!r}")
        end = at + len(name)
    if end >= len(turn):
        raise ValueError(f"nothing closes {name!r} in the rendered turn: {turn!r}")
    return end + 1


def _ids(rendered) -> list[int]:
    return list(rendered["input_ids"])


def continuations(
    tokenizer, entry: tier_bench.CorpusEntry, platform: Platform
) -> dict[str, Continuation]:
    """Every candidate's :class:`Continuation` for *entry*, keyed by metrics.py's label.

    The prompt is tokenized as train.py tokenizes it (the chat template with
    the generation prompt, thinking off where the template has the switch);
    each continuation's ids are the cut text's, and must be exactly the
    leading tokens of that candidate's full rendered turn.
    """
    template = train._template_text(tokenizer)
    extra = train.thinking_off_kwargs(template) if template is not None else {}
    result: dict[str, Continuation] = {}
    prompt_ids: list[int] | None = None
    prompt_text = ""
    for name in scorer.candidates():
        example = build_dataset.example_from_entry(_candidate_entry(entry, name), platform)
        messages, tools = example["messages"], example["tools"]
        if prompt_ids is None:
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tools=tools, tokenize=False, add_generation_prompt=True, **extra
            )
            prompt_ids = _ids(
                tokenizer.apply_chat_template(
                    messages[:-1],
                    tools=tools,
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=True,
                    **extra,
                )
            )
        full_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, **extra)
        if not full_text.startswith(prompt_text):
            raise ValueError(f"{entry.id}: the prompt is not a prefix of {name!r}'s rendering")
        turn = full_text[len(prompt_text) :]
        text = turn[: decisive_end(turn, name)]
        full_ids = _ids(
            tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, return_dict=True, **extra
            )
        )
        cut_ids = list(tokenizer.encode(prompt_text + text, add_special_tokens=False))
        if cut_ids[: len(prompt_ids)] != prompt_ids or full_ids[: len(cut_ids)] != cut_ids:
            raise ValueError(
                f"{entry.id}: {name!r}'s deciding span does not end on a token boundary of"
                " its full rendering; refusing to score tokens the model was not trained on"
            )
        result[scorer.calibration_label(name)] = Continuation(
            prompt_ids=list(prompt_ids), text=text, continuation_ids=cut_ids[len(prompt_ids) :]
        )
    return result


# -- the distribution --


def normalise(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Softmax over summed log-probabilities, shifted by the maximum so nothing underflows."""
    top = max(logprobs.values())
    weights = {label: math.exp(value - top) for label, value in logprobs.items()}
    total = math.fsum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def score_entry(
    tokenizer, model: ContinuationScorer, entry: tier_bench.CorpusEntry, platform: Platform
) -> dict[str, float]:
    """The exact candidate distribution for *entry*, under metrics.py's labels."""
    spans = continuations(tokenizer, entry, platform)
    return normalise(
        {
            label: float(model.logprob(span.prompt_ids, span.continuation_ids))
            for label, span in spans.items()
        }
    )


class TransformersContinuationScorer:
    """An in-process causal LM behind :meth:`logprob`, one forward pass per continuation.

    Only the positions that predict the continuation keep their logits
    (``logits_to_keep``), so the full vocabulary is never computed for the
    whole prompt. Log-softmax runs in float32.
    """

    def __init__(self, model) -> None:
        self._model = model

    def logprob(self, prompt_ids: Sequence[int], continuation_ids: Sequence[int]) -> float:
        import torch

        count = len(continuation_ids)
        if not prompt_ids or not count:
            raise ValueError("a continuation needs a prompt and at least one token")
        device = next(self._model.parameters()).device
        ids = torch.tensor([list(prompt_ids) + list(continuation_ids)], device=device)
        with torch.no_grad():
            logits = self._model(input_ids=ids, logits_to_keep=count + 1).logits[0, :-1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        targets = torch.tensor(list(continuation_ids), device=logprobs.device)
        return float(logprobs.gather(1, targets.unsqueeze(1)).sum())


def load_model(model_dir: Path):  # pragma: no cover - needs the training stack and weights
    """``(tokenizer, scorer, versions)`` for the model in *model_dir*, local files only."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, local_files_only=True
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    return tokenizer, TransformersContinuationScorer(model), versions


# -- the predictions file --


def read_lines(path: Path) -> list[dict]:
    """The predictions file's lines as decoded objects, in order (blank lines skipped)."""
    lines = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {number}: not JSON ({exc.msg})") from None
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}: line {number}: not a predictions line (no id)")
            lines.append(row)
    return lines


def match_entries(
    lines: Sequence[dict], entries: Sequence[tier_bench.CorpusEntry]
) -> list[tier_bench.CorpusEntry]:
    """The split entry for each line, in line order; the two must cover each other exactly."""
    by_id = {entry.id: entry for entry in entries}
    seen: set[str] = set()
    matched = []
    for line in lines:
        entry_id = line["id"]
        if entry_id in seen:
            raise ValueError(f"duplicate predictions line for {entry_id!r}")
        if entry_id not in by_id:
            raise ValueError(f"predictions line {entry_id!r} is not an entry of the split")
        seen.add(entry_id)
        matched.append(by_id[entry_id])
    missing = [entry.id for entry in entries if entry.id not in seen]
    if missing:
        raise ValueError(
            f"no predictions line for {len(missing)} split entries ({', '.join(missing[:3])}...)"
            " -- was it written by measure.py for this split?"
        )
    return matched


def sidecar_path(out: Path) -> Path:
    """Where the provenance JSON for *out* goes: next to it."""
    return out.with_name(out.name + ".provenance.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path, help="a model directory (merged)")
    parser.add_argument("--split", required=True, type=Path, help="the split measure.py ran")
    parser.add_argument("--predictions", required=True, type=Path, help="measure.py's file")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--acceptance", action="store_true", help="allow held-out.json")
    parser.add_argument("--final", action="store_true", help="allow test.json (a final run)")
    return parser


def run(args: argparse.Namespace) -> int:
    raw = measure.read_split(args.split)
    measure.check_split_allowed(
        args.split, acceptance=args.acceptance, final=args.final, header=raw.get("header")
    )
    if args.out.resolve() in (args.predictions.resolve(), args.split.resolve()):
        raise measure.MeasureError(measure.EXIT_USER, "--out must not be one of the inputs")
    if not args.model.is_dir():
        raise measure.MeasureError(measure.EXIT_USER, f"--model {args.model} is not a directory")
    loaded = tier_bench.load_corpus(args.split)
    if not loaded.entries:
        raise measure.MeasureError(measure.EXIT_USER, f"{args.split.name} has no valid entries")
    try:
        lines = read_lines(args.predictions)
        entries = match_entries(lines, loaded.entries)
    except (OSError, ValueError) as exc:
        raise measure.MeasureError(measure.EXIT_USER, str(exc)) from exc
    platform = tier_bench.world_platform(tier_bench.load_world(args.split))

    tokenizer, model, versions = load_model(args.model)
    filled = []
    for line, entry in zip(lines, entries):
        try:
            candidates = score_entry(tokenizer, model, entry, platform)
        except ValueError as exc:
            raise measure.MeasureError(measure.EXIT_ENV, str(exc)) from exc
        filled.append({**line, "candidates": candidates})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for line in filled:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    provenance = {
        "method": METHOD,
        "model": measure.home_relative(str(args.model.resolve())),
        "model_revision": stage_cache.revision_of(args.model),
        "split": measure.home_relative(str(args.split.resolve())),
        "split_sha256": _sha256(args.split),
        "predictions": measure.home_relative(str(args.predictions.resolve())),
        "predictions_sha256": _sha256(args.predictions),
        "candidates": [scorer.calibration_label(name) for name in scorer.candidates()],
        "lines": len(filled),
        "final": args.final,
        "acceptance": args.acceptance,
        **{name: versions.get(name) for name in ("torch", "transformers")},
    }
    sidecar_path(args.out).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"written={len(filled)} out={args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return run(args)
    except measure.MeasureError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
