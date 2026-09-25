"""scripts/lfm-finetune/scorer.py (issue 46, Track B): the candidate scorer.

A fake next-token scorer stands in for the model everywhere; the real
Qwen3.5-0.8B tokenizer is used only when it is in the Hugging Face cache,
and torch only when it is importable.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from nvsh.ops import table as ops_table
from nvsh.tiers.bench import world_runner

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "lfm-finetune" / "scorer.py"
_WORLD = json.loads((_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json").read_text())["world"]

QWEN = "Qwen/Qwen3.5-0.8B"
QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_scorer", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


class _FakeScorer:
    """Answers score_next_token with fixed logprobs; records the prompt and top it was asked."""

    def __init__(self, logprobs: dict[str, float]) -> None:
        self.logprobs = logprobs
        self.prompts: list[str] = []
        self.tops: list[int] = []

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        self.prompts.append(prompt)
        self.tops.append(top)
        return dict(self.logprobs)


class _PlainTokenizer:
    """A chat template with no thinking switch: joins the contents."""

    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "\n".join(message["content"] for message in messages) + "\nANSWER:"


def _favouring(module, candidate: str, *, offered=None) -> dict[str, float]:
    """Logprobs where *candidate*'s label is most likely and every other label has some mass."""
    labels = module.labels_for(offered or module.candidates())
    return {
        label: (-0.5 if name == candidate else -5.0 - index * 0.1)
        for index, (name, label) in enumerate(labels.items())
    }


def _operation_with_arg(kind: str, name: str | None = None) -> str:
    """The first table operation declaring an argument of *kind* (and *name*, if given)."""
    for operation in ops_table.OPERATIONS:
        for spec in operation.args:
            if spec.kind == kind and (name is None or spec.name == name):
                return operation.name
    raise AssertionError(f"the table has no operation with a {kind} argument")


# -- candidates and labels --


def test_candidates_are_every_table_operation_then_explain_and_escalate() -> None:
    module = _module()
    assert module.candidates() == ops_table.names() + ("explain", "escalate")


def test_every_candidate_gets_a_distinct_label() -> None:
    module = _module()
    labels = module.labels_for(module.candidates())
    assert set(labels) == set(module.candidates())
    assert len(set(labels.values())) == len(labels)


def test_a_label_stays_with_its_candidate_when_others_are_not_offered() -> None:
    module = _module()
    full = module.labels_for(module.candidates())
    offered = module.candidates()[1:]
    assert module.labels_for(offered) == {name: full[name] for name in offered}


def test_an_unknown_candidate_is_refused() -> None:
    module = _module()
    with pytest.raises(ValueError, match="not a candidate"):
        module.labels_for(("no_such_operation",))


# -- the distribution --


def test_distribution_sums_to_one_over_every_candidate() -> None:
    module = _module()
    logprobs = _favouring(module, "escalate")
    logprobs.update({"the": -0.5, "<junk>": float("nan"), "Z9": 3.0})
    distribution, mass = module.distribution(logprobs, module.labels_for(module.candidates()))
    assert set(distribution) == set(module.candidates())
    assert math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-9)
    assert 0 < mass <= 1
    assert max(distribution, key=distribution.get) == "escalate"


def test_token_variants_of_one_label_are_summed() -> None:
    module = _module()
    labels = module.labels_for(("explain", "escalate"))
    plain = {labels["explain"]: math.log(0.2), labels["escalate"]: math.log(0.2)}
    spaced = dict(plain, **{" " + labels["explain"]: math.log(0.2)})
    distribution, mass = module.distribution(spaced, labels)
    assert math.isclose(distribution["explain"], 2 / 3)
    assert math.isclose(mass, 0.6)
    assert module.distribution(plain, labels)[0]["explain"] == pytest.approx(0.5)


def test_no_label_mass_gives_no_distribution() -> None:
    module = _module()
    assert module.distribution({"the": -0.1}, module.labels_for(module.candidates())) == ({}, 0.0)


# -- scoring one request --


def test_score_returns_a_normalised_distribution_and_its_argmax() -> None:
    module = _module()
    fake = _FakeScorer(_favouring(module, "explain"))
    scored = module.score(fake, "prompt text", "What is my GPU doing?", runner=world_runner(_WORLD))
    assert math.isclose(sum(scored.distribution.values()), 1.0, rel_tol=1e-9)
    assert scored.choice == "explain"
    assert scored.confidence == max(scored.distribution.values())
    assert scored.arguments is None
    assert scored.grounding is None
    assert fake.prompts == ["prompt text"]
    assert fake.tops == [module.READOUT_TOP]


def test_score_over_an_offered_subset_scores_only_that_subset() -> None:
    module = _module()
    offered = module.candidates()[2:]
    fake = _FakeScorer(_favouring(module, module.candidates()[0]))  # favours one not offered
    scored = module.score(fake, "p", "anything", offered=offered, runner=world_runner(_WORLD))
    assert set(scored.distribution) == set(offered)
    assert math.isclose(sum(scored.distribution.values()), 1.0, rel_tol=1e-9)


def test_a_scorer_with_no_label_mass_makes_no_choice() -> None:
    module = _module()
    scored = module.score(_FakeScorer({"the": -0.1}), "p", "x", runner=world_runner(_WORLD))
    assert scored.choice is None
    assert scored.distribution == {}
    assert scored.confidence == 0.0


def test_a_failing_scorer_makes_no_choice_and_does_not_raise() -> None:
    class _Broken:
        def score_next_token(self, prompt, *, top=20):
            raise OSError("no server")

    scored = _module().score(_Broken(), "p", "x", runner=world_runner(_WORLD))
    assert scored.choice is None


def test_argmax_ties_break_by_candidate_order() -> None:
    module = _module()
    labels = module.labels_for(module.candidates())
    fake = _FakeScorer({label: -1.0 for label in labels.values()})
    scored = module.score(fake, "p", "x", runner=world_runner(_WORLD))
    assert scored.choice == module.candidates()[0]


# -- calibration labels: metrics.py's convention (review finding #1) --


_METRICS = _ROOT / "scripts" / "lfm-finetune" / "metrics.py"


def _metrics():
    spec = importlib.util.spec_from_file_location("lfm_metrics_for_scorer", _METRICS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prediction_line(scored, expected: dict, entry_id: str = "e1") -> dict:
    """A predictions-file line written straight from a Scored result."""
    operation = scored.choice if ops_table.get(scored.choice or "") is not None else None
    return {
        "id": entry_id,
        "expected": expected,
        "outcome": "propose" if operation else scored.choice,
        "operation": operation,
        "arguments": (scored.arguments or {}) if operation else None,
        "candidates": scored.candidates,
        "tokens": 0,
        "ttfd_ms": 1.0,
        "latency_ms": 1.0,
    }


def test_candidates_use_bench_labels_for_the_two_controls() -> None:
    module = _module()
    from nvsh.tiers import bench

    fake = _FakeScorer(_favouring(module, "escalate"))
    scored = module.score(fake, "p", "x", runner=world_runner(_WORLD))
    assert set(scored.candidates) == (
        set(ops_table.names()) | {bench.ESCALATE_LABEL, bench.EXPLAIN_LABEL}
    )
    assert scored.candidates[bench.ESCALATE_LABEL] == scored.distribution["escalate"]
    assert "escalate" not in scored.candidates
    assert "explain" not in scored.candidates


def test_a_confident_correct_escalate_calibrates_perfectly_through_metrics() -> None:
    module = _module()
    metrics = _metrics()
    labels = module.labels_for(("explain", "escalate"))
    fake = _FakeScorer({labels["escalate"]: 0.0, labels["explain"]: float("-inf")})
    scored = module.score(
        fake, "p", "x", offered=("explain", "escalate"), runner=world_runner(_WORLD)
    )
    assert scored.choice == "escalate"
    assert scored.confidence == 1.0
    line = _prediction_line(scored, {"escalate": True})
    result = metrics.compute([metrics.Prediction.from_dict(json.loads(json.dumps(line)))])
    assert result["calibration"]["n"] == 1
    assert result["calibration"]["ece"] == 0
    assert result["calibration"]["brier"] == 0


# -- incomplete served distributions (review finding #2) --


def test_a_served_result_missing_labels_is_marked_incomplete_not_renormalised() -> None:
    module = _module()
    labels = module.labels_for(module.candidates())
    first = module.candidates()[0]
    # Only one label came back, with little raw mass; the rest fell below the cutoff.
    fake = _FakeScorer({labels[first]: math.log(0.01), "the": math.log(0.9)})
    scored = module.score(fake, "p", "x", runner=world_runner(_WORLD))
    assert scored.candidates is None
    assert scored.distribution == {}
    assert scored.incomplete
    assert "missing" in scored.incomplete
    assert set(scored.missing) == set(module.candidates()) - {first}
    assert scored.choice == first
    assert scored.confidence == pytest.approx(0.01)  # the raw probability, not 1.0
    assert scored.mass == pytest.approx(0.01)


def test_an_incomplete_result_is_left_out_of_calibration_and_counted() -> None:
    module = _module()
    metrics = _metrics()
    labels = module.labels_for(("explain", "escalate"))
    fake = _FakeScorer({labels["escalate"]: math.log(0.01)})
    scored = module.score(
        fake, "p", "x", offered=("explain", "escalate"), runner=world_runner(_WORLD)
    )
    assert scored.missing == ("explain",)
    line = _prediction_line(scored, {"escalate": True})
    calibration = metrics.compute([metrics.Prediction.from_dict(line)])["calibration"]
    assert calibration["n"] == 0
    assert calibration["without_distribution"] == 1


def test_a_result_with_every_label_is_complete() -> None:
    module = _module()
    scored = module.score(
        _FakeScorer(_favouring(module, "explain")), "p", "x", runner=world_runner(_WORLD)
    )
    assert scored.incomplete is None
    assert scored.missing == ()
    assert scored.candidates is not None


def test_a_scorer_with_no_label_mass_has_no_candidates() -> None:
    module = _module()
    scored = module.score(_FakeScorer({"the": -0.1}), "p", "x", runner=world_runner(_WORLD))
    assert scored.candidates is None


# -- arguments come from nvsh's deterministic grounding (decision c52) --


def test_a_service_argument_is_grounded_from_the_request_text() -> None:
    module = _module()
    operation = _operation_with_arg("str", "service")
    fake = _FakeScorer(_favouring(module, operation))
    scored = module.score(fake, "p", "Restart vLLM please", runner=world_runner(_WORLD))
    assert scored.choice == operation
    assert scored.arguments == {"service": "vllm.service"}
    assert scored.grounding is None


def test_a_container_argument_is_grounded_from_the_request_text() -> None:
    module = _module()
    operation = _operation_with_arg("str", "container")
    grounded = module.ground_arguments(
        ops_table.get(operation), "bounce the trainer container", world_runner(_WORLD)
    )
    assert grounded == {"container": "trainer"}


def test_a_choice_argument_is_matched_from_the_request_text() -> None:
    module = _module()
    operation = ops_table.get(_operation_with_arg("choice"))
    spec = operation.args[0]
    wanted = spec.choices[-1]
    text = f"please switch to {wanted.replace('_', ' ')} now"
    assert module.ground_arguments(operation, text, world_runner(_WORLD)) == {spec.name: wanted}


def test_an_argument_nothing_in_the_request_grounds_is_reported_not_guessed() -> None:
    module = _module()
    operation = _operation_with_arg("str", "service")
    fake = _FakeScorer(_favouring(module, operation))
    scored = module.score(fake, "p", "restart the web thing", runner=world_runner(_WORLD))
    assert scored.choice == operation
    assert scored.arguments is None
    assert "service" in scored.grounding


def test_two_grounded_values_are_ambiguous_not_a_pick() -> None:
    module = _module()
    operation = ops_table.get(_operation_with_arg("str", "service"))
    result = module.ground_arguments(operation, "restart nginx or docker", world_runner(_WORLD))
    assert isinstance(result, str)
    assert "ambiguous" in result


def test_an_operation_with_no_arguments_grounds_to_empty_without_a_lookup() -> None:
    module = _module()
    operation = next(op for op in ops_table.OPERATIONS if not op.args)

    def runner(argv, timeout):
        raise AssertionError("no lookup for an operation without arguments")

    assert module.ground_arguments(operation, "anything at all", runner) == {}


def test_grounding_looks_each_kind_up_once() -> None:
    module = _module()
    calls: list[list[str]] = []
    inner = world_runner(_WORLD)

    def counting(argv, timeout):
        calls.append(list(argv))
        return inner(argv, timeout)

    operation = ops_table.get(_operation_with_arg("str", "service"))
    module.ground_arguments(operation, "restart a b c d e f vllm", counting)
    assert len(calls) == 1


# -- the prompt --


def test_the_prompt_lists_every_offered_candidate_with_its_label() -> None:
    module = _module()
    offered = module.candidates()[1:]
    messages = module.prompt_messages("Restart vLLM", offered)
    labels = module.labels_for(offered)
    system = messages[0]["content"]
    for name in offered:
        assert f"{labels[name]}) {name}:" in system
    assert f") {module.candidates()[0]}:" not in system
    assert messages[-1] == {"role": "user", "content": "Restart vLLM"}


def test_render_prompt_turns_thinking_off_only_when_the_template_has_the_switch() -> None:
    module = _module()
    plain = _PlainTokenizer()
    module.render_prompt(plain, [{"role": "user", "content": "x"}])
    assert "enable_thinking" not in plain.kwargs
    assert plain.kwargs["add_generation_prompt"] is True

    thinking = _PlainTokenizer()
    thinking.chat_template = "{% if enable_thinking is defined %}{% endif %}"
    module.render_prompt(thinking, [{"role": "user", "content": "x"}])
    assert thinking.kwargs["enable_thinking"] is False


# -- the tokenizer (real, when cached) --


class _OneTokenPerChar:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_label_token_ids_refuses_a_label_that_is_not_a_single_token() -> None:
    module = _module()
    tokenizer = _OneTokenPerChar()
    with pytest.raises(ValueError, match="single token"):
        module.label_token_ids(tokenizer, {"a": "AB"})


def test_label_token_ids_refuses_two_labels_sharing_a_token() -> None:
    module = _module()

    class _Same:
        def encode(self, text, add_special_tokens=False):
            return [1]

    tokenizer = _Same()
    with pytest.raises(ValueError, match="share"):
        module.label_token_ids(tokenizer, {"a": "A", "b": "B"})


def _qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            QWEN, revision=QWEN_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("the Qwen3.5-0.8B tokenizer is not in the Hugging Face cache")


def test_label_tokens_are_distinct_single_tokens_for_the_qwen_tokenizer() -> None:
    module = _module()
    tokenizer = _qwen_tokenizer()
    ids = module.label_token_ids(tokenizer, module.labels_for(module.candidates()))
    assert len(ids) == len(module.candidates())
    assert len(set(ids.values())) == len(ids)


def test_the_label_is_the_token_right_after_the_qwen_prompt() -> None:
    module = _module()
    tokenizer = _qwen_tokenizer()
    labels = module.labels_for(module.candidates())
    prompt = module.render_prompt(tokenizer, module.prompt_messages("Restart vLLM"))
    assert "<think>\n\n</think>" in prompt
    ids = module.label_token_ids(tokenizer, labels)
    for name, label in labels.items():
        rendered = tokenizer.encode(prompt + label, add_special_tokens=False)
        assert rendered[:-1] == tokenizer.encode(prompt, add_special_tokens=False)
        assert rendered[-1] == ids[name]


# -- the in-process scorer (torch, when importable) --


class _VocabTokenizer:
    """A tokenizer with a small vocabulary: each label as itself, spaced and tabbed, plus junk.

    Encoding a label gives its bare token, as the Qwen3.5 tokenizer does.
    """

    pad_token_id = 0

    def __init__(self, labels) -> None:
        self.texts = ["<pad>", "the", "Restart", "\n"]
        for label in labels:
            self.texts += [label, " " + label, "\t" + label]
        self.texts += ["AB", " the"]

    def get_vocab(self) -> dict[str, int]:
        return {f"tok{index}": index for index in range(len(self.texts))}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.texts[index] for index in ids)

    def encode(self, text, add_special_tokens=False):
        if text in self.texts:
            return [self.texts.index(text)]
        return [1, 2, 3]

    def __len__(self) -> int:
        return len(self.texts)


def _fixture_logits(size: int):
    """Deterministic, uneven next-token logits over a vocabulary of *size*."""
    return [((index * 7919) % 97) / 13.0 - 3.0 for index in range(size)]


def _logprobs_of(logits) -> list[float]:
    top = max(logits)
    total = top + math.log(sum(math.exp(value - top) for value in logits))
    return [value - total for value in logits]


def _torch_model(torch, logits, calls: list):
    """A model whose last-position logits are *logits*; records each forward's logits_to_keep."""

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.fixed = torch.tensor([logits], dtype=torch.float32)

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            calls.append({"logits_to_keep": logits_to_keep, "length": input_ids.shape[1]})
            positions = input_ids.shape[1] if not logits_to_keep else logits_to_keep
            out = self.fixed.unsqueeze(1).expand(input_ids.shape[0], positions, -1)
            return type("Out", (), {"logits": out + self.anchor})()

    return _Model()


# -- the one readout definition (issue 53, t1) --


def test_readout_top_is_a_named_constant_of_at_least_20000() -> None:
    module = _module()
    assert isinstance(module.READOUT_TOP, int)
    assert module.READOUT_TOP >= 20000


def test_label_variant_ids_are_every_vocabulary_token_that_strips_to_the_label() -> None:
    module = _module()
    labels = module.labels_for(module.candidates())
    tokenizer = _VocabTokenizer(labels.values())
    variants = module.label_variant_ids(tokenizer, labels)
    assert set(variants) == set(labels)
    for name, label in labels.items():
        texts = sorted(tokenizer.texts[index] for index in variants[name])
        assert texts == sorted([label, " " + label, "\t" + label])
    every = [index for ids in variants.values() for index in ids]
    assert len(every) == len(set(every))


def test_label_variant_ids_refuses_a_label_with_no_token() -> None:
    module = _module()
    labels = module.labels_for(("explain", "escalate"))
    tokenizer = _VocabTokenizer([labels["explain"]])
    with pytest.raises(ValueError, match="no token"):
        module.label_variant_ids(tokenizer, labels)


def test_distribution_counts_the_same_variants_the_vocabulary_scan_finds() -> None:
    module = _module()
    labels = module.labels_for(("explain", "escalate"))
    tokenizer = _VocabTokenizer(labels.values())
    variants = module.label_variant_ids(tokenizer, labels)
    logprobs = {tokenizer.texts[index]: math.log(0.1) for index in variants["explain"]}
    logprobs[labels["escalate"]] = math.log(0.1)
    distribution, mass = module.distribution(logprobs, labels)
    assert distribution["explain"] == pytest.approx(0.75)  # three variants against one
    assert mass == pytest.approx(0.4)


def test_the_served_request_asks_for_readout_top_and_one_token(monkeypatch) -> None:
    from nvsh.tiers import toolchat

    module = _module()
    sent: list[tuple[str, dict]] = []

    def fake_request(self, path, body, parse):
        sent.append((path, dict(body)))
        return _favouring(module, "explain")

    monkeypatch.setattr(toolchat.ToolChat, "_request", fake_request)
    chat = toolchat.ToolChat("http://127.0.0.1:8000/v1", "scorer-b1", stream=False)
    scored = module.score(chat, "p", "x", runner=world_runner(_WORLD))
    assert scored.choice == "explain"
    assert len(sent) == 1
    path, body = sent[0]
    assert path == "/completions"
    assert body["max_tokens"] == 1
    assert body["logprobs"] == module.READOUT_TOP


def test_a_readout_top_result_still_missing_labels_is_incomplete_not_renormalised() -> None:
    module = _module()
    labels = module.labels_for(module.candidates())
    first, second = module.candidates()[:2]
    fake = _FakeScorer({labels[first]: math.log(0.3), " " + labels[second]: math.log(0.2)})
    scored = module.score(fake, "p", "x", runner=world_runner(_WORLD))
    assert fake.tops == [module.READOUT_TOP]
    assert scored.incomplete is not None
    assert scored.candidates is None
    assert scored.distribution == {}
    assert set(scored.missing) == set(module.candidates()) - {first, second}
    assert scored.confidence == pytest.approx(0.3)


# -- the in-process scorer (torch, when importable) --


def test_the_in_process_scorer_returns_label_logprobs_that_normalise() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    labels = module.labels_for(module.candidates())
    tokenizer = _VocabTokenizer(labels.values())
    label_ids = module.label_token_ids(tokenizer, labels)
    calls: list = []
    model = _torch_model(torch, _fixture_logits(len(tokenizer)), calls)
    scorer = module.TransformersScorer(model, tokenizer, labels, label_ids)
    logprobs = scorer.score_next_token("anything")
    assert {text.strip() for text in logprobs} == set(labels.values())
    assert len(logprobs) == 3 * len(labels)  # every variant of every label
    assert all(value <= 0 for value in logprobs.values())
    distribution, _ = module.distribution(logprobs, labels)
    assert math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-6)
    scored = module.score(scorer, "anything", "x", runner=world_runner(_WORLD))
    assert scored.incomplete is None
    assert scored.missing == ()
    assert math.isclose(sum(scored.candidates.values()), 1.0, rel_tol=1e-6)


def test_the_in_process_scorer_runs_one_forward_pass_over_one_position() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    labels = module.labels_for(module.candidates())
    tokenizer = _VocabTokenizer(labels.values())
    calls: list = []
    model = _torch_model(torch, _fixture_logits(len(tokenizer)), calls)
    scorer = module.TransformersScorer(
        model, tokenizer, labels, module.label_token_ids(tokenizer, labels)
    )
    module.score(scorer, "a prompt", "x", runner=world_runner(_WORLD))
    assert calls == [{"logits_to_keep": 1, "length": 3}]


def test_the_in_process_scorer_refuses_a_label_id_outside_its_variants() -> None:
    pytest.importorskip("torch")
    module = _module()
    labels = module.labels_for(("explain", "escalate"))
    tokenizer = _VocabTokenizer(labels.values())
    wrong = {"explain": 1, "escalate": tokenizer.texts.index(labels["escalate"])}
    with pytest.raises(ValueError, match="variant"):
        module.TransformersScorer(object(), tokenizer, labels, wrong)


def test_in_process_and_served_paths_give_the_same_distribution_on_a_fixture() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    labels = module.labels_for(module.candidates())
    tokenizer = _VocabTokenizer(labels.values())
    logits = _fixture_logits(len(tokenizer))
    scorer = module.TransformersScorer(
        _torch_model(torch, logits, []),
        tokenizer,
        labels,
        module.label_token_ids(tokenizer, labels),
    )
    in_process = module.score(scorer, "p", "x", runner=world_runner(_WORLD))

    # The served path: the same next-token distribution, as a server's top log-probabilities
    # keyed by token text (every token fits under READOUT_TOP).
    served_logprobs: dict[str, float] = {}
    for index, value in enumerate(_logprobs_of(logits)):
        text = tokenizer.texts[index]
        served_logprobs[text] = (
            value
            if text not in served_logprobs
            else math.log(math.exp(served_logprobs[text]) + math.exp(value))
        )
    served = module.score(_FakeScorer(served_logprobs), "p", "x", runner=world_runner(_WORLD))

    assert in_process.incomplete is None and served.incomplete is None
    assert set(in_process.distribution) == set(served.distribution)
    for name in labels:
        assert in_process.distribution[name] == pytest.approx(served.distribution[name], abs=1e-6)
    assert in_process.mass == pytest.approx(served.mass, abs=1e-6)


def test_the_training_readout_matches_the_distribution() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    labels = module.labels_for(module.candidates())
    tokenizer = _VocabTokenizer(labels.values())
    variants = module.label_variant_ids(tokenizer, labels)
    logits = torch.tensor([_fixture_logits(len(tokenizer))] * 2)
    label_logits = module.label_logits_from_vocab(logits, [variants[name] for name in labels])
    assert tuple(label_logits.shape) == (2, len(labels))
    trained = torch.softmax(label_logits, dim=-1)[0].tolist()
    logprobs = {
        tokenizer.texts[index]: value
        for index, value in enumerate(_logprobs_of(logits[0].tolist()))
    }
    distribution, _ = module.distribution(logprobs, labels)
    for position, name in enumerate(labels):
        assert trained[position] == pytest.approx(distribution[name], abs=1e-6)


def test_the_training_readout_keeps_gradients() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    logits = torch.zeros(1, 6, requires_grad=True)
    label_logits = module.label_logits_from_vocab(logits, [(1, 2), (4,)])
    torch.nn.functional.cross_entropy(label_logits, torch.tensor([0])).backward()
    assert logits.grad is not None
    assert label_logits[0, 0].item() == pytest.approx(math.log(2))


def test_label_variants_for_the_qwen_tokenizer_include_the_bare_and_spaced_label() -> None:
    module = _module()
    tokenizer = _qwen_tokenizer()
    labels = module.labels_for(module.candidates())
    variants = module.label_variant_ids(tokenizer, labels)
    bare = module.label_token_ids(tokenizer, labels)
    for name, label in labels.items():
        assert bare[name] in variants[name]
        texts = {tokenizer.decode([index]) for index in variants[name]}
        assert {label, " " + label} <= texts


# -- the permutation seam (issue 53, t2) --


def test_default_prompt_text_is_pinned_so_scorer_b1_stays_reproducible() -> None:
    module = _module()
    messages = module.prompt_messages("Restart vLLM", ("explain", "escalate"))
    assert messages == [
        {
            "role": "system",
            "content": (
                "You pick the one action that handles the operator's request on this machine."
                " Answer with the action's letter only.\n\nActions:\n"
                "Q) explain: Answer the operator in plain words, with no command.\n"
                "R) escalate: Hand this request to the full agent, saying why."
            ),
        },
        {"role": "user", "content": "Restart vLLM"},
    ]


def test_score_with_no_seam_arguments_is_unchanged_from_before_the_seam() -> None:
    module = _module()
    fake = _FakeScorer(_favouring(module, "explain"))
    scored = module.score(fake, "prompt text", "x", runner=world_runner(_WORLD))
    assert scored.choice == "explain"
    assert set(scored.distribution) == set(module.candidates())


def test_prompt_messages_accepts_an_explicit_label_map_and_order() -> None:
    module = _module()
    order = ("escalate", "explain")
    labels = {"escalate": "Z", "explain": "Y"}
    messages = module.prompt_messages("Restart vLLM", labels=labels, order=order)
    system = messages[0]["content"]
    assert "Z) escalate:" in system
    assert "Y) explain:" in system
    # the order given is the listing order
    assert system.index("Z) escalate:") < system.index("Y) explain:")


def test_score_accepts_the_same_explicit_label_map_and_order() -> None:
    module = _module()
    order = ("escalate", "explain")
    labels = {"escalate": "Z", "explain": "Y"}
    fake = _FakeScorer({"Z": math.log(0.9), "Y": math.log(0.1)})
    scored = module.score(fake, "p", "x", labels=labels, order=order, runner=world_runner(_WORLD))
    assert set(scored.distribution) == {"escalate", "explain"}
    assert scored.choice == "escalate"


def test_labels_for_default_is_unaffected_by_the_new_seam() -> None:
    module = _module()
    assert module.labels_for(module.candidates()) == {
        name: module.LABEL_ALPHABET[index] for index, name in enumerate(module.candidates())
    }


def test_permute_is_deterministic_for_the_same_seed() -> None:
    module = _module()
    first = module.permute("seed-1")
    second = module.permute("seed-1")
    assert first == second
    assert set(first.order) == set(module.candidates())
    assert set(first.labels) == set(module.candidates())
    assert len(set(first.labels.values())) == len(first.labels)  # distinct letters


def test_permute_gives_a_different_order_or_map_for_a_different_seed() -> None:
    module = _module()
    first = module.permute("seed-1")
    second = module.permute("seed-2")
    assert first != second


def test_permute_can_take_a_random_subset_that_keeps_a_gold_candidate() -> None:
    module = _module()
    gold = module.candidates()[3]
    permutation = module.permute("seed-3", subset=4, keep=gold)
    assert len(permutation.order) == 4
    assert gold in permutation.order
    assert set(permutation.labels) == set(permutation.order)


def test_permute_subset_is_deterministic_for_the_same_seed() -> None:
    module = _module()
    gold = module.candidates()[3]
    first = module.permute("seed-4", subset=5, keep=gold)
    second = module.permute("seed-4", subset=5, keep=gold)
    assert first == second


def test_permute_refuses_a_subset_smaller_than_what_must_be_kept() -> None:
    module = _module()
    gold = module.candidates()[0]
    with pytest.raises(ValueError, match="smaller"):
        module.permute("seed-5", subset=0, keep=gold)


def test_permutation_round_trips_through_json() -> None:
    module = _module()
    permutation = module.permute("seed-6", subset=4, keep=module.candidates()[0])
    restored = module.Permutation.from_json(json.loads(json.dumps(permutation.to_json())))
    assert restored == permutation


def test_a_permuted_prompt_and_score_round_trip_through_the_stored_permutation() -> None:
    module = _module()
    permutation = module.permute("seed-7", subset=4, keep=module.candidates()[0])
    stored = module.Permutation.from_json(json.loads(json.dumps(permutation.to_json())))
    messages = module.prompt_messages("Restart vLLM", labels=stored.labels, order=stored.order)
    gold_letter = stored.labels[stored.order[0]]
    logprobs = {
        letter: (-0.5 if name == stored.order[0] else -5.0 - index * 0.1)
        for index, (name, letter) in enumerate(stored.labels.items())
    }
    assert logprobs[gold_letter] == -0.5
    fake = _FakeScorer(logprobs)
    scored = module.score(
        fake,
        "p",
        "Restart vLLM",
        labels=stored.labels,
        order=stored.order,
        runner=world_runner(_WORLD),
    )
    assert scored.choice == stored.order[0]
    assert set(scored.distribution) == set(stored.order)
    assert messages[0]["content"].startswith(
        "You pick the one action that handles the operator's request on this machine."
    )


def test_candidate_descriptions_can_be_overridden_without_reading_nvsh() -> None:
    module = _module()
    messages = module.prompt_messages(
        "x",
        ("explain", "escalate"),
        descriptions={"explain": "A paraphrased explain description."},
    )
    system = messages[0]["content"]
    assert "A paraphrased explain description." in system
    assert "Hand this request to the full agent" in system  # escalate: unoverridden default


def test_a_reason_candidate_outside_ops_table_and_lfm_needs_an_override() -> None:
    module = _module()
    with pytest.raises(ValueError, match="not a candidate"):
        module.prompt_messages("x", ("escalate:repair",))


def test_a_reason_candidate_with_an_override_can_be_offered() -> None:
    module = _module()
    order = ("escalate:repair", "explain")
    labels = {"escalate:repair": "Z", "explain": "Y"}
    messages = module.prompt_messages(
        "x",
        labels=labels,
        order=order,
        descriptions={"escalate:repair": "Escalate because a repair is needed."},
    )
    system = messages[0]["content"]
    assert "Z) escalate:repair: Escalate because a repair is needed." in system


def test_score_permits_a_reason_candidate_with_an_override_and_explicit_label() -> None:
    module = _module()
    order = ("escalate:repair", "explain")
    labels = {"escalate:repair": "Z", "explain": "Y"}
    fake = _FakeScorer({"Z": math.log(0.9), "Y": math.log(0.1)})
    scored = module.score(fake, "p", "x", labels=labels, order=order, runner=world_runner(_WORLD))
    assert scored.choice == "escalate:repair"


def test_same_choice_compares_by_operation_name_not_letter() -> None:
    module = _module()
    order = tuple(reversed(module.candidates()))
    permuted_labels = {name: letter for name, letter in zip(order, reversed(module.LABEL_ALPHABET))}
    plain = module.score(
        _FakeScorer(_favouring(module, "explain")), "p", "x", runner=world_runner(_WORLD)
    )
    fake = _FakeScorer(
        {permuted_labels[name]: (-0.5 if name == "explain" else -5.0) for name in order}
    )
    permuted = module.score(
        fake, "p", "x", labels=permuted_labels, order=order, runner=world_runner(_WORLD)
    )
    assert module.same_choice(plain, permuted)
    assert module.same_choice(plain.choice, permuted.choice)
    assert not module.same_choice(plain, "escalate")


def test_score_returns_which_operation_was_chosen_for_op_level_comparison() -> None:
    module = _module()
    fake = _FakeScorer(_favouring(module, "explain"))
    scored = module.score(fake, "p", "x", runner=world_runner(_WORLD))
    assert scored.choice == "explain"  # the operation name, never a letter
