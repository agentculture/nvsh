"""Track B candidate scorer: one next-token read over distinct label tokens (issue 46).

Track B does not generate a tool call. Every candidate -- each operation in
``nvsh/ops/table.py`` plus ``explain`` and ``escalate`` -- is listed in the
prompt under its own one-letter label, and the model's next-token
log-probabilities over those labels are read once and normalised over the
offered candidates. The result is a distribution (what the calibration
metrics score), its argmax (the choice) and, when the choice is an
operation, arguments from nvsh's own deterministic grounding (decision c52):
the scorer never generates an argument value.

The distribution is also offered as ``Scored.candidates``, keyed the way
``metrics.py``'s predictions file keys it: operation names, and
``nvsh.tiers.bench``'s ``"(explain)"`` / ``"(escalate)"`` for the two
controls, so a predictions line written from it calibrates against the
entry's expected label.

**The readout definition (issue 53).** A candidate's raw mass is the sum
of the next-token probabilities of every vocabulary token whose text,
stripped of surrounding whitespace, is its label: ``"A"``, ``" A"`` and
``"\tA"`` all count for ``A`` (the Qwen3.5 vocabulary has exactly those
three for every label). The candidate probabilities are those masses
normalised over the offered candidates. :func:`distribution` is the one
function that computes them, and every path feeds it: a served model's
top log-probabilities keyed by token text, the in-process
:class:`TransformersScorer` (which reads the same token ids,
:func:`label_variant_ids`, and keys them the same way), and training,
whose :func:`label_logits_from_vocab` is the log of the same masses, so a
softmax over it is this distribution.

Why the variant sum, and not one token id per label: the 2026-09-25 probe
served scorer-b1 (bf16 GGUF, llama-server) against the in-process scorer.
Five of six prompts agreed within 1e-4, but on dev-f02 the two differed
by 0.137, because the in-process scorer then read the single id of the
bare label while the served one summed ``"A"`` and ``" A"``. The served
sum is the definition kept: it is what a server can report without a
tokenizer, and a model that answers ``" A"`` has still chosen ``A``.

A served model returns only its top next-token log-probabilities, so a
served request asks for :data:`READOUT_TOP` of them (the probe: the top 22
missed 10-15 labels on every prompt, the top 5000 was complete on 6/6).
A label below that cutoff is still simply absent. Such a result is never
renormalised over the labels that did come back (that turns a label holding
1% of the raw mass into a certainty): it is marked incomplete, with the
missing candidates named and no distribution, so metrics leaves it out of
calibration and counts it. The in-process scorer reads every label's
logit, so its distributions are always complete.

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

import json
import math
import random
import re
import string
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

from nvsh.ops import ground as ops_ground  # noqa: E402
from nvsh.ops import table as ops_table  # noqa: E402
from nvsh.ops._model import Operation  # noqa: E402
from nvsh.tiers import bench as tier_bench  # noqa: E402
from nvsh.tiers import lfm  # noqa: E402

#: The two candidates that are not operations: answer in words, or hand the request up.
CONTROLS = (lfm.EXPLAIN_TOOL, lfm.ESCALATE_TOOL)

#: Each control's label in a predictions file's ``candidates`` (metrics.py's convention).
CALIBRATION_LABELS = {
    lfm.EXPLAIN_TOOL: tier_bench.EXPLAIN_LABEL,
    lfm.ESCALATE_TOOL: tier_bench.ESCALATE_LABEL,
}

#: The 8 decline classes (issue 53 t14), offered in reasons mode (``build_dataset.py
#: --reasons``, ``measure.py --reasons``) as distinct escalate-reason candidates in
#: place of the bare ``escalate``. Order is fixed so the default letter map is stable.
REASON_CANDIDATES: tuple[str, ...] = (
    "escalate:outside_table",
    "escalate:repair",
    "escalate:diagnosis",
    "escalate:missing_argument",
    "escalate:not_a_request",
    "escalate:multi_step",
    "escalate:injection",
    "escalate:over_time",
)

#: An escalate entry with no or unknown ``class`` rolls up to this reason.
DEFAULT_REASON = "escalate:outside_table"

#: Scripts-side (never nvsh/) prompt descriptions for the reason candidates.
REASON_DESCRIPTIONS: dict[str, str] = json.loads(
    (Path(__file__).resolve().parent / "data" / "reasons.json").read_text(encoding="utf-8")
)

#: Label alphabet, in candidate order. Each is one token in the Qwen3.5 and
#: LFM2.5 vocabularies (checked by :func:`label_token_ids`, never assumed).
LABEL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase

#: How many next-token log-probabilities a served request asks for. The
#: 2026-09-25 probe found the top 22 missing 10-15 labels on every prompt and
#: the top 5000 complete on 6/6; on all 64 spent-test prompts the Q4_K_M GGUF still missed labels
#: on 5 at top 5000 and on none at top 20000 (0.25 s a request on CPU). Anything still
#: missing is marked incomplete.
READOUT_TOP = 20000

#: The pre-issue-53 margin above the candidate count. :func:`score` no longer
#: uses it (it asks for :data:`READOUT_TOP`); measure.py's ``--max-logprobs``
#: check still reads it until that check moves to :data:`READOUT_TOP`.
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
    to 1), or is empty when the model put no mass on any label or the
    result is incomplete. ``choice`` is the most likely candidate (``None``
    when no label had mass) and ``confidence`` its share of the label mass.
    ``mass`` is how much of the raw next-token distribution the labels
    held. For an operation choice, ``arguments`` are the grounded values, or
    ``None`` with ``grounding`` saying why they could not be grounded.

    ``incomplete`` says why there is no distribution although some label
    had mass: ``missing`` offered candidates whose label the scorer did not
    return at all. The choice is then the most likely label that did come
    back, and ``confidence`` its raw next-token probability -- a lower bound
    on its share, never a renormalised one.
    """

    distribution: dict[str, float]
    choice: str | None
    confidence: float
    mass: float
    arguments: dict[str, str] | None = None
    grounding: str | None = None
    incomplete: str | None = None
    missing: tuple[str, ...] = ()

    @property
    def candidates(self) -> dict[str, float] | None:
        """The distribution under metrics.py's labels, or ``None`` when there is none."""
        if not self.distribution or self.incomplete is not None:
            return None
        return {calibration_label(name): p for name, p in self.distribution.items()}


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


def candidate_pool(reasons: bool = False) -> tuple[str, ...]:
    """Every candidate one request is offered: :func:`candidates`, or in reasons mode

    its bare ``escalate`` replaced by :data:`REASON_CANDIDATES` (16 operations +
    explain + 8 reasons = 25, still within :data:`LABEL_ALPHABET`). The one pool
    ``build_dataset.py --reasons`` trains on and ``measure.py --reasons`` measures.
    """
    base = candidates()
    if not reasons:
        return base
    return tuple(name for name in base if name != lfm.ESCALATE_TOOL) + REASON_CANDIDATES


def reason_for_class(cls: str | None) -> str:
    """The reason candidate a corpus ``class`` (``"decline:<reason>"``) names.

    A missing or unrecognised class rolls up to :data:`DEFAULT_REASON`, never
    a fabricated one.
    """
    if cls and cls.startswith("decline:"):
        candidate = "escalate:" + cls.split(":", 1)[1]
        if candidate in REASON_CANDIDATES:
            return candidate
    return DEFAULT_REASON


def reason_descriptions(order: Sequence[str]) -> dict[str, str]:
    """Description overrides for *order*'s reason candidates only; ``{}`` otherwise."""
    return {name: REASON_DESCRIPTIONS[name] for name in order if name in REASON_DESCRIPTIONS}


def positional_labels(offered: Sequence[str], full: Sequence[str]) -> dict[str, str]:
    """Candidate -> letter for *offered*, each lettered by its position in *full*.

    :func:`labels_for`'s rule over any pool, including the reasons pool (whose
    names :func:`labels_for` refuses): a candidate's letter does not move when
    others are left out.
    """
    if len(full) > len(LABEL_ALPHABET):
        raise ValueError(f"{len(full)} candidates but only {len(LABEL_ALPHABET)} labels")
    index = {name: position for position, name in enumerate(full)}
    missing = [name for name in offered if name not in index]
    if missing:
        raise ValueError(f"not in the candidate pool: {', '.join(missing)}")
    return {name: LABEL_ALPHABET[index[name]] for name in offered}


@dataclass(frozen=True)
class Permutation:
    """A seeded order + letter map (:func:`permute`): one round-trippable, serialisable value.

    ``order`` is the candidates to offer, in listing order (so it also
    carries the offered subset); ``labels`` maps each of them to its
    letter. Both a permutation probe (t7) and training/dataset build
    (t14/t16) store this per example and must render exactly the same
    prompt from it later, hence :meth:`to_json`/:meth:`from_json`.
    """

    order: tuple[str, ...]
    labels: dict[str, str]

    def to_json(self) -> dict:
        """A plain JSON-safe value: ``{"order": [...], "labels": {...}}``."""
        return {"order": list(self.order), "labels": dict(self.labels)}

    @classmethod
    def from_json(cls, data: Mapping) -> "Permutation":
        """The inverse of :meth:`to_json`."""
        return cls(order=tuple(data["order"]), labels=dict(data["labels"]))


def permute(
    seed,
    pool: Sequence[str] | None = None,
    *,
    subset: int | None = None,
    keep: str | Sequence[str] | None = None,
) -> Permutation:
    """A seeded random order + letter permutation (optionally a random subset) of *pool*.

    *pool* defaults to :func:`candidates`. The same *seed* (any hashable
    accepted by :class:`random.Random`, typically an ``int`` or ``str`)
    always gives the same :class:`Permutation`, independent of process or
    machine (``random.Random`` seeds deterministically from an int or str).

    *subset*, when given, keeps only that many candidates (drawn from the
    seeded order); *keep* names one candidate, or several, that the subset
    must include regardless of where the seeded order put them -- the gold
    answer, for a subset probe. Letters are drawn from :data:`LABEL_ALPHABET`
    without replacement, so at most ``len(LABEL_ALPHABET)`` candidates can be
    offered at once, exactly as :func:`labels_for` already requires.
    """
    names = list(candidates() if pool is None else pool)
    if len(names) > len(LABEL_ALPHABET):
        raise ValueError(f"{len(names)} candidates but only {len(LABEL_ALPHABET)} labels")
    rng = random.Random(seed)
    order = list(names)
    rng.shuffle(order)
    if subset is not None:
        keep_names = (keep,) if isinstance(keep, str) else tuple(keep or ())
        for name in keep_names:
            if name not in order:
                raise ValueError(f"{name!r} to keep is not in the candidate pool")
        if subset < len(keep_names):
            raise ValueError(f"subset {subset} is smaller than the {len(keep_names)} to keep")
        if subset > len(order):
            raise ValueError(f"subset {subset} is larger than the {len(order)}-candidate pool")
        kept = [name for name in order if name in keep_names]
        rest = [name for name in order if name not in keep_names]
        order = sorted(kept + rest[: subset - len(kept)], key=order.index)
    letters = list(LABEL_ALPHABET)
    rng.shuffle(letters)
    labels = {name: letters[index] for index, name in enumerate(order)}
    return Permutation(order=tuple(order), labels=labels)


def same_choice(a, b) -> bool:
    """True when *a* and *b* chose the same candidate, compared by name only.

    Each of *a*, *b* is a :class:`Scored` result or a bare choice name (what
    ``Scored.choice`` already is). Comparing by name, never by letter, is
    what lets a permutation probe (t7) re-score the same request under many
    seeded label/order permutations and count only real, op-level answer
    changes -- a candidate permuted onto a different letter but still chosen
    is not a change.
    """
    left = a.choice if isinstance(a, Scored) else a
    right = b.choice if isinstance(b, Scored) else b
    return left == right


def calibration_label(name: str) -> str:
    """*name*'s label in a predictions file: bench's label for a control, else the name."""
    return CALIBRATION_LABELS.get(name, name)


def label_token_ids(tokenizer, labels: Mapping[str, str]) -> dict[str, int]:
    """Candidate -> the single token id of its bare label; refuses multi-token or shared labels.

    This is the id a prompt followed by the label encodes to, so it checks
    the label alphabet. The readout sums more than this one id: see
    :func:`label_variant_ids`.
    """
    ids: dict[str, int] = {}
    for name, label in labels.items():
        encoded = list(tokenizer.encode(label, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(f"label {label!r} for {name!r} is not a single token: {encoded}")
        ids[name] = encoded[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError("two labels share a token id; the scores would not be distinct")
    return ids


def label_variant_ids(tokenizer, labels: Mapping[str, str]) -> dict[str, tuple[int, ...]]:
    """Candidate -> every token id whose text, stripped of whitespace, is its label.

    The in-process half of the readout definition (module docstring): the
    whole vocabulary is scanned once (about 0.4 s for Qwen3.5's 248k
    tokens), so the ids are the ones a served model's :func:`distribution`
    sums, found rather than assumed. Ids are sorted. Refuses a label with
    no token at all. Distinct labels never share an id: a token strips to
    one text.
    """
    by_label = {label: name for name, label in labels.items()}
    found: dict[str, list[int]] = {name: [] for name in labels}
    for token_id in sorted(set(tokenizer.get_vocab().values())):
        name = by_label.get(tokenizer.decode([token_id]).strip())
        if name is not None:
            found[name].append(token_id)
    empty = [name for name, ids in found.items() if not ids]
    if empty:
        raise ValueError(f"no token in the vocabulary is the label of: {', '.join(empty)}")
    return {name: tuple(ids) for name, ids in found.items()}


def label_logits_from_vocab(logits, variant_ids: Sequence[Sequence[int]]):
    """``(..., labels)`` label logits from ``(..., vocab)`` logits, one log-sum-exp per label.

    The training half of the readout definition: for each label, the log of
    its summed variant probabilities (up to the shared log-partition), so a
    softmax over the result is exactly :func:`distribution` of the same
    next-token log-probabilities. Differentiable. *variant_ids* is in
    candidate order, usually ``[label_variant_ids(...)[name] for name in ...]``.
    """
    import torch

    columns = [
        torch.logsumexp(
            logits.index_select(-1, torch.tensor(list(ids), device=logits.device)), dim=-1
        )
        for ids in variant_ids
    ]
    return torch.stack(columns, dim=-1)


def _description(name: str, descriptions: Mapping[str, str] | None = None) -> str:
    """*name*'s prompt description: *descriptions*' entry for it, else the default lookup.

    An override in *descriptions* is used verbatim and never reads
    ``nvsh/``, so it also covers a candidate that is not in
    ``ops_table``/``lfm`` at all -- a paraphrase probe's reworded text, or a
    reason candidate such as ``"escalate:repair"`` -- as long as its text is
    supplied. Without an override, an unknown name still raises.
    """
    if descriptions is not None and name in descriptions:
        return descriptions[name]
    operation = ops_table.get(name)
    if operation is not None:
        return operation.description
    for tool in lfm.tools_for(()):
        if tool["function"]["name"] == name:
            return tool["function"]["description"]
    raise ValueError(f"{name!r} is not a candidate")


def _label_map(
    offered: Sequence[str] | None,
    labels: Mapping[str, str] | None,
    order: Sequence[str] | None,
) -> dict[str, str]:
    """Candidate -> label for one request: *labels*/*order* if given, else today's default.

    *order* names the candidates to offer, in listing order (also carrying
    the offered subset); default order is *offered* (or every candidate).
    *labels* is used verbatim when given, keyed exactly by that order --
    including names :func:`labels_for` would refuse, such as a reason
    candidate with no table/lfm entry. With no *labels*, :func:`labels_for`
    keeps assigning today's fixed alphabetical map, so the no-argument
    default stays byte-identical to before this seam existed.
    """
    names = list(order) if order is not None else list(candidates() if offered is None else offered)
    if labels is None:
        return labels_for(names)
    missing = [name for name in names if name not in labels]
    if missing:
        raise ValueError(f"no label given for: {', '.join(missing)}")
    return {name: labels[name] for name in names}


# -- the prompt --


def prompt_messages(
    request_text: str,
    offered: Sequence[str] | None = None,
    *,
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    descriptions: Mapping[str, str] | None = None,
) -> list[dict]:
    """System message listing the offered candidates under their labels, then the request.

    With no *labels*/*order*/*descriptions*, this is exactly today's prompt
    (labels from :func:`labels_for`, descriptions from the default lookup),
    kept byte-identical so scorer-b1 stays reproducible. *labels* is an
    explicit candidate -> letter map (e.g. from :func:`permute`) and *order*
    the listing order (also the offered subset) it was drawn for; give
    either without the other and the missing one is filled in the usual way.
    *descriptions* overrides a candidate's description text (paraphrase
    probes, reason candidates) without reading ``nvsh/``; a name with no
    override falls back to the default lookup.
    """
    resolved = _label_map(offered, labels, order)
    names = list(order) if order is not None else list(resolved)
    lines = [f"{resolved[name]}) {name}: {_description(name, descriptions)}" for name in names]
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

    The one readout definition (module docstring), for every path: each
    token whose text stripped of whitespace is a label (``"A"``, ``" A"``,
    ``"\tA"``) adds its probability to that label, as ``yes_no_probability``
    sums ``" yes"`` and ``"Yes"``, and the masses are normalised over the
    offered candidates. Tokens that are no label, and junk values, are
    skipped. ``({}, 0.0)`` when no label has mass.
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


def missing_labels(logprobs: Mapping[str, float], labels: Mapping[str, str]) -> tuple[str, ...]:
    """Candidates in *labels* whose label (in any token variant) *logprobs* does not carry.

    A label reported with a valid log-probability is present even when that
    probability is zero; junk values count as absent, as in :func:`distribution`.
    """
    seen = {
        token.strip()
        for token, logprob in logprobs.items()
        if isinstance(token, str) and _valid_logprob(logprob)
    }
    return tuple(name for name, label in labels.items() if label not in seen)


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
    # A name that ends a sentence keeps the full stop ("restart vllm."), which
    # then grounds to nothing while another unit the sentence mentions does:
    # a wrong target (issue 53, t18). Sentence punctuation is not a name.
    words = (word.rstrip(".:") for word in _WORD_RE.findall(text))
    for word in dict.fromkeys(word for word in words if word):
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
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    runner: ops_ground.Runner = ops_ground.default_runner,
) -> Scored:
    """Score every offered candidate for one request. Never raises.

    *prompt* is the rendered prompt (:func:`render_prompt` of
    :func:`prompt_messages`) and *request_text* the request those messages
    carry, which grounding reads. Ties go to the earlier candidate.

    *labels* and *order* are :func:`prompt_messages`' seam: pass the same
    ones used to render *prompt* (e.g. from a stored :class:`Permutation`)
    so the readout sums the letters the prompt actually offered. With
    neither, this reads today's fixed label map, unchanged. ``Scored.choice``
    is always the candidate's name, never its letter, so op-level comparison
    (:func:`same_choice`) needs nothing extra even under a permuted map.
    """
    labels = _label_map(offered, labels, order)
    try:
        logprobs = scorer.score_next_token(prompt, top=READOUT_TOP)
    except Exception:  # noqa: BLE001 -- a scorer that fails makes no choice
        logprobs = {}
    if not isinstance(logprobs, dict):
        logprobs = {}
    probabilities, mass = distribution(logprobs, labels)
    if not probabilities:
        return Scored(distribution={}, choice=None, confidence=0.0, mass=0.0)
    choice = max(labels, key=lambda name: probabilities[name])  # first maximum wins
    missing = missing_labels(logprobs, labels)
    if missing:
        # Renormalising what came back would fabricate certainty (review finding #2).
        scored = Scored(
            {},
            choice,
            probabilities[choice] * mass,  # the raw next-token probability
            mass,
            incomplete=f"{len(missing)} of {len(labels)} candidate labels missing from the"
            " scorer's top log-probabilities",
            missing=missing,
        )
    else:
        scored = Scored(probabilities, choice, probabilities[choice], mass)
    operation = ops_table.get(choice)
    if operation is None:
        return scored
    grounded = ground_arguments(operation, request_text, runner)
    if isinstance(grounded, str):
        return replace(scored, grounding=grounded)
    return replace(scored, arguments=grounded)


class TransformersScorer:
    """An in-process model behind ``score_next_token``, over the label tokens only.

    One forward pass keeps only the last position's logits
    (``logits_to_keep``), so the full 248k-token vocabulary is computed for
    one position and never for the whole prompt. The returned
    log-probabilities are the true next-token ones (softmax over the whole
    vocabulary at that position) of every label variant
    (:func:`label_variant_ids`), keyed by the token's text exactly as a
    served model keys them, so :func:`distribution` sums the same tokens on
    both paths (issue 53: reading only the bare label's id made dev-f02
    differ from the served scorer by 0.137).

    *label_ids* (:func:`label_token_ids`) is kept for its callers and
    checked: each must be one of its label's variants.
    """

    def __init__(
        self, model, tokenizer, labels: Mapping[str, str], label_ids: Mapping[str, int]
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._labels = dict(labels)
        self._variants = label_variant_ids(tokenizer, self._labels)
        for name, token_id in label_ids.items():
            if name in self._variants and token_id not in self._variants[name]:
                raise ValueError(
                    f"token id {token_id} for {name!r} is not a variant of its label"
                    f" {self._labels[name]!r}"
                )
        self._texts = {
            token_id: tokenizer.decode([token_id])
            for ids in self._variants.values()
            for token_id in ids
        }

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        """Variant token text -> log-probability after *prompt*; *top* is ignored.

        Two ids that decode to the same text have their probabilities added.
        """
        del top
        import torch

        device = next(self._model.parameters()).device
        ids = torch.tensor([self._tokenizer.encode(prompt, add_special_tokens=False)])
        with torch.no_grad():
            logits = self._model(input_ids=ids.to(device), logits_to_keep=1).logits[0, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        result: dict[str, float] = {}
        for token_id, text in self._texts.items():
            value = float(logprobs[token_id])
            if text in result:
                high, low = max(result[text], value), min(result[text], value)
                if low != -math.inf:
                    value = high + math.log1p(math.exp(low - high))
                else:
                    value = high
            result[text] = value
        return result
