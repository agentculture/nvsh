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
    with pytest.raises(ValueError, match="not a candidate"):
        _module().labels_for(("no_such_operation",))


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
    assert fake.tops[0] >= len(module.candidates())


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
    with pytest.raises(ValueError, match="single token"):
        module.label_token_ids(_OneTokenPerChar(), {"a": "AB"})


def test_label_token_ids_refuses_two_labels_sharing_a_token() -> None:
    module = _module()

    class _Same:
        def encode(self, text, add_special_tokens=False):
            return [1]

    with pytest.raises(ValueError, match="share"):
        module.label_token_ids(_Same(), {"a": "A", "b": "B"})


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


def test_the_in_process_scorer_returns_label_logprobs_that_normalise() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    labels = module.labels_for(module.candidates())
    label_ids = {name: 10 + index for index, name in enumerate(labels)}

    class _Tok:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [3, 4, 5]

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(1, 64)

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            hidden = torch.ones(input_ids.shape[0], input_ids.shape[1], 1)
            logits = self.head(hidden)
            return type("Out", (), {"logits": logits[:, -logits_to_keep:, :]})()

    scorer = module.TransformersScorer(_Model(), _Tok(), labels, label_ids)
    logprobs = scorer.score_next_token("anything")
    assert set(logprobs) == set(labels.values())
    assert all(value <= 0 for value in logprobs.values())
    distribution, _ = module.distribution(logprobs, labels)
    assert math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-6)
    scored = module.score(scorer, "anything", "x", runner=world_runner(_WORLD))
    assert scored.incomplete is None
    assert scored.missing == ()
    assert math.isclose(sum(scored.candidates.values()), 1.0, rel_tol=1e-6)
