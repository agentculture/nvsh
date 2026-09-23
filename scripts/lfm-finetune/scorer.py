"""Track B candidate scorer: one next-token read over distinct label tokens (issue 46).

Track B does not generate a tool call. Every candidate -- each operation in
``nvsh/ops/table.py`` plus ``explain`` and ``escalate`` -- is listed in the
prompt under its own one-letter label, and the model's next-token
log-probabilities over those labels are read once and normalised over the
offered candidates. The result is a distribution (what the calibration
metrics score), its argmax (the choice) and, when the choice is an
operation, arguments from nvsh's own deterministic grounding (decision c52):
the scorer never generates an argument value.

The serving-side pattern is nvsh's own: ``ToolChat.score_next_token`` reads
one token's log-probabilities and ``yes_no_probability`` sums a word's
token variants and renormalises over the words asked about
(``nvsh/tiers/toolchat.py``). Anything with that ``score_next_token``
method scores here, so a served model (through ``ToolChat``) and an
in-process one (:class:`TransformersScorer`) share one code path.

Arguments come from ``nvsh.ops.ground.ground``, the function Tier 1's
router and Tier 2 call on a model's pick. It only checks a value against
what the machine lists and returns the machine's own spelling, so the
values offered to it here are the request's own words: exactly one word
that grounds becomes the argument, none or several is reported instead of
guessed. A ``choice`` argument is matched against its declared choices the
same way. Nothing here is keyed on a particular operation's name.

This script is never imported by nvsh. Its torch imports are lazy, so the
helpers work without a training environment.
"""

from __future__ import annotations

import math
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.ops import ground as ops_ground  # noqa: E402
from nvsh.ops import table as ops_table  # noqa: E402
from nvsh.ops._model import Operation  # noqa: E402
from nvsh.tiers import lfm  # noqa: E402

#: The two candidates that are not operations: answer in words, or hand the request up.
CONTROLS = (lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)

#: Label alphabet, in candidate order. Each is one token in the Qwen3.5 and
#: LFM2.5 vocabularies (checked by :func:`label_token_ids`, never assumed).
LABEL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase

#: How many next-token log-probabilities to ask a served model for, above the
#: candidate count, so a label is not lost behind a few unrelated tokens.
TOP_MARGIN = 4

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]*")

_INSTRUCTION = (
    "You pick the one action that handles the operator's request on this machine."
    " Answer with the action's letter only.\n\nActions:\n"
)


class NextTokenScorer(Protocol):
    """The one method the scorer needs: ``ToolChat.score_next_token``'s shape."""

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        """Token -> natural-log probability for the token after *prompt*."""


@dataclass(frozen=True)
class Scored:
    """One request's scores.

    ``distribution`` maps every offered candidate to its probability (sums
    to 1), or is empty when the model put no mass on any label. ``choice``
    is its argmax (``None`` when empty). ``mass`` is how much of the raw
    next-token distribution the labels held. For an operation choice,
    ``arguments`` are the grounded values, or ``None`` with ``grounding``
    saying why they could not be grounded.
    """

    distribution: dict[str, float]
    choice: str | None
    confidence: float
    mass: float
    arguments: dict[str, str] | None = None
    grounding: str | None = None


# -- candidates and labels --


def candidates() -> tuple[str, ...]:
    """Every table operation, in table order, then ``explain`` and ``escalate``."""
    return ops_table.names() + CONTROLS


def labels_for(offered: Sequence[str]) -> dict[str, str]:
    """Candidate -> label for *offered*, each keeping its label from the full list.

    A candidate's label does not move when others are left out, so the
    missing-candidate slice asks the same letters the model was trained on.
    """
    full = candidates()
    if len(full) > len(LABEL_ALPHABET):
        raise ValueError(f"{len(full)} candidates but only {len(LABEL_ALPHABET)} labels")
    labels: dict[str, str] = {}
    for name in offered:
        if name not in full:
            raise ValueError(f"{name!r} is not a candidate")
        labels[name] = LABEL_ALPHABET[full.index(name)]
    return labels


def label_token_ids(tokenizer, labels: Mapping[str, str]) -> dict[str, int]:
    """Candidate -> the single token id of its label; refuses multi-token or shared labels."""
    ids: dict[str, int] = {}
    for name, label in labels.items():
        encoded = list(tokenizer.encode(label, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(f"label {label!r} for {name!r} is not a single token: {encoded}")
        ids[name] = encoded[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError("two labels share a token id; the scores would not be distinct")
    return ids


def _description(name: str) -> str:
    operation = ops_table.get(name)
    if operation is not None:
        return operation.description
    for tool in lfm.tools_for(()):
        if tool["function"]["name"] == name:
            return tool["function"]["description"]
    raise ValueError(f"{name!r} is not a candidate")


# -- the prompt --


def prompt_messages(request_text: str, offered: Sequence[str] | None = None) -> list[dict]:
    """System message listing the offered candidates under their labels, then the request."""
    labels = labels_for(candidates() if offered is None else offered)
    lines = [f"{label}) {name}: {_description(name)}" for name, label in labels.items()]
    return [
        {"role": "system", "content": _INSTRUCTION + "\n".join(lines)},
        {"role": "user", "content": request_text},
    ]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """The prompt text up to the answer, thinking off when the template has that switch.

    Qwen3.5's template renders an empty think block for ``enable_thinking``
    False, so the label is the very next token; a template without the
    switch is rendered as it is.
    """
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


# -- the distribution --


def _valid_logprob(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isnan(value)
        and value <= 0
    )


def distribution(
    logprobs: Mapping[str, float], labels: Mapping[str, str]
) -> tuple[dict[str, float], float]:
    """``(candidate -> probability, mass)`` over *labels*. Never raises.

    Token variants of one label (``"A"``, ``" A"``) are summed, as
    ``yes_no_probability`` sums ``" yes"`` and ``"Yes"``. Tokens that are no
    label, and junk values, are skipped. ``({}, 0.0)`` when no label has mass.
    """
    by_label = {label: name for name, label in labels.items()}
    masses = {name: 0.0 for name in labels}
    for token, logprob in logprobs.items():
        if not isinstance(token, str) or not _valid_logprob(logprob):
            continue
        name = by_label.get(token.strip())
        if name is not None:
            masses[name] += math.exp(logprob)
    mass = sum(masses.values())
    if mass <= 0:
        return ({}, 0.0)
    return ({name: value / mass for name, value in masses.items()}, mass)


# -- arguments, from nvsh's grounding --


def _memoised(runner: ops_ground.Runner) -> ops_ground.Runner:
    """*runner*, with each lookup run once for this request."""
    seen: dict[tuple[str, ...], tuple[int, str]] = {}

    def run(argv: list[str], timeout: float) -> tuple[int, str]:
        key = tuple(argv)
        if key not in seen:
            seen[key] = runner(argv, timeout)
        return seen[key]

    return run


def _spellings(choice: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((choice, choice.replace("_", " "), choice.replace("_", "-"))))


def _choice_value(name: str, choices: Sequence[str], text: str) -> str:
    folded = f" {' '.join(_WORD_RE.findall(text)).casefold()} "
    found = [
        choice
        for choice in choices
        if any(f" {spelling.casefold()} " in folded for spelling in _spellings(choice))
    ]
    if len(found) == 1:
        return found[0]
    if found:
        raise LookupError(f"{name} is ambiguous: {', '.join(found)}")
    raise LookupError(f"no {name} named in the request")


def _grounded_value(operation: Operation, name: str, text: str, runner: ops_ground.Runner) -> str:
    """The one value the request's words ground to for argument *name*."""
    found: list[str] = []
    for word in dict.fromkeys(_WORD_RE.findall(text)):
        grounded = ops_ground.ground(operation, {name: word}, runner)
        if isinstance(grounded, ops_ground.GroundDecline):
            if grounded.code == "lookup_failed":
                raise LookupError(grounded.message)
            continue
        value = grounded.args[name]
        if value not in found:
            found.append(value)
    if len(found) == 1:
        return found[0]
    if found:
        raise LookupError(f"{name} is ambiguous: {', '.join(found)}")
    raise LookupError(f"no {name} named in the request grounds on this machine")


def ground_arguments(
    operation: Operation, request_text: str, runner: ops_ground.Runner
) -> dict[str, str] | str:
    """Arguments for *operation* from the request's words, or one line saying why not.

    ``choice`` arguments match a declared choice; every other argument is
    whichever request word ``nvsh.ops.ground.ground`` accepts. Each machine
    lookup runs at most once per request. The result is checked with the
    table's own ``validate``.
    """
    runner = _memoised(runner)
    args: dict[str, str] = {}
    try:
        for spec in operation.args:
            if spec.kind == "choice":
                args[spec.name] = _choice_value(spec.name, spec.choices, request_text)
            else:
                args[spec.name] = _grounded_value(operation, spec.name, request_text, runner)
    except LookupError as problem:
        return str(problem)
    invalid = ops_table.validate(operation.name, args)
    return invalid.message if invalid is not None else args


# -- scoring --


def score(
    scorer: NextTokenScorer,
    prompt: str,
    request_text: str,
    *,
    offered: Sequence[str] | None = None,
    runner: ops_ground.Runner = ops_ground.default_runner,
) -> Scored:
    """Score every offered candidate for one request. Never raises.

    *prompt* is the rendered prompt (:func:`render_prompt` of
    :func:`prompt_messages`) and *request_text* the request those messages
    carry, which grounding reads. Ties go to the earlier candidate.
    """
    labels = labels_for(candidates() if offered is None else offered)
    try:
        logprobs = scorer.score_next_token(prompt, top=len(labels) + TOP_MARGIN)
    except Exception:  # noqa: BLE001 -- a scorer that fails makes no choice
        logprobs = {}
    probabilities, mass = distribution(logprobs if isinstance(logprobs, dict) else {}, labels)
    if not probabilities:
        return Scored(distribution={}, choice=None, confidence=0.0, mass=0.0)
    choice = max(labels, key=lambda name: probabilities[name])  # first maximum wins
    scored = Scored(probabilities, choice, probabilities[choice], mass)
    operation = ops_table.get(choice)
    if operation is None:
        return scored
    grounded = ground_arguments(operation, request_text, runner)
    if isinstance(grounded, str):
        return Scored(probabilities, choice, probabilities[choice], mass, None, grounded)
    return Scored(probabilities, choice, probabilities[choice], mass, grounded, None)


class TransformersScorer:
    """An in-process model behind ``score_next_token``, over the label tokens only.

    One forward pass keeps only the last position's logits
    (``logits_to_keep``), so the full 248k-token vocabulary is computed for
    one position and never for the whole prompt. The returned
    log-probabilities are the true next-token ones (softmax over the whole
    vocabulary at that position), keyed by label, so :func:`distribution`
    treats them exactly as a served model's.
    """

    def __init__(
        self, model, tokenizer, labels: Mapping[str, str], label_ids: Mapping[str, int]
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._labels = dict(labels)
        self._ids = dict(label_ids)

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        """Label -> log-probability at the position after *prompt*; *top* is ignored."""
        del top
        import torch

        device = next(self._model.parameters()).device
        ids = torch.tensor([self._tokenizer.encode(prompt, add_special_tokens=False)])
        with torch.no_grad():
            logits = self._model(input_ids=ids.to(device), logits_to_keep=1).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        return {label: float(logprobs[self._ids[name]]) for name, label in self._labels.items()}
