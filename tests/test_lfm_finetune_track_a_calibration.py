"""scripts/lfm-finetune/track_a_calibration.py (issue 46, deviation d6): exact Track A scoring.

A character-level fake tokenizer with a Qwen-shaped template and a fake
continuation scorer stand in for the model everywhere; the real Qwen3.5-0.8B
tokenizer is used only when it is in the Hugging Face cache, and torch only
when it is importable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from nvsh.ops import table as ops_table
from nvsh.tiers import bench as tier_bench

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "lfm-finetune" / "track_a_calibration.py"
_DEV = _ROOT / "nvsh" / "tiers" / "corpus" / "dev.json"

QWEN = "Qwen/Qwen3.5-0.8B"
QWEN_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_track_a_calibration", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def _labels() -> set[str]:
    return set(ops_table.names()) | {tier_bench.EXPLAIN_LABEL, tier_bench.ESCALATE_LABEL}


class _FakeTokenizer:
    """One token per character, and a template shaped like Qwen3.5's XML tool-call form."""

    chat_template = "{% if enable_thinking is false %}<think></think>{% endif %}"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def _render(self, messages, add_generation_prompt, enable_thinking):
        parts = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n")
            if message.get("tool_calls"):
                if enable_thinking is False:
                    parts.append("<think>\n\n</think>\n\n")
                for call in message["tool_calls"]:
                    function = call["function"]
                    parts.append(f"<tool_call>\n<function={function['name']}>\n")
                    for key, value in function["arguments"].items():
                        text = value if isinstance(value, str) else json.dumps(value)
                        parts.append(f"<parameter={key}>\n{text}\n</parameter>\n")
                    parts.append("</function>\n</tool_call>")
            else:
                parts.append(str(message.get("content", "")))
            parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if enable_thinking is False:
                parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        enable_thinking=None,
        **kwargs,
    ):
        text = self._render(messages, add_generation_prompt, enable_thinking)
        if not tokenize:
            return text
        ids = self.encode(text)
        return {"input_ids": ids} if return_dict else ids


class _FakeScorer:
    """Log-probability of a continuation: -1 per token, and a bonus when it spells *favour*."""

    def __init__(self, favour: str | None = None, bonus: float = 0.0) -> None:
        self.favour = favour
        self.bonus = bonus
        self.calls: list[tuple[list[int], list[int]]] = []

    def logprob(self, prompt_ids, continuation_ids) -> float:
        self.calls.append((list(prompt_ids), list(continuation_ids)))
        text = "".join(chr(i) for i in continuation_ids)
        value = -1.0 * len(continuation_ids)
        if self.favour is not None and self.favour in text:
            value += self.bonus
        return value


def _entries(module):
    return tier_bench.load_corpus(_DEV).entries


def _first(entries, kind: str):
    for entry in entries:
        if kind == "operation" and entry.expect.get("operation"):
            return entry
        if kind != "operation" and entry.expect.get(kind):
            return entry
    raise AssertionError(f"dev.json has no {kind} entry")


# --- the distribution --------------------------------------------------------


def test_distribution_sums_to_one_over_exactly_the_candidate_labels() -> None:
    module = _module()
    entry = _first(_entries(module), "operation")
    platform = tier_bench.world_platform(tier_bench.load_world(_DEV))
    distribution = module.score_entry(_FakeTokenizer(), _FakeScorer(), entry, platform)
    assert set(distribution) == _labels()
    assert math.isclose(math.fsum(distribution.values()), 1.0, abs_tol=1e-9)
    assert all(0.0 <= value <= 1.0 for value in distribution.values())


def test_a_more_likely_continuation_gets_more_mass() -> None:
    module = _module()
    entry = _first(_entries(module), "operation")
    platform = tier_bench.world_platform(tier_bench.load_world(_DEV))
    tokenizer = _FakeTokenizer()
    plain = module.score_entry(tokenizer, _FakeScorer(), entry, platform)
    target = ops_table.names()[3]
    favoured = module.score_entry(
        tokenizer, _FakeScorer(favour=f"{target}\n", bonus=200.0), entry, platform
    )
    assert favoured[target] > plain[target]
    assert max(favoured, key=favoured.get) == target
    controls = module.score_entry(
        tokenizer, _FakeScorer(favour="=escalate>", bonus=200.0), entry, platform
    )
    assert max(controls, key=controls.get) == tier_bench.ESCALATE_LABEL


def test_the_softmax_is_over_the_summed_logprobs() -> None:
    module = _module()
    assert module.normalise({"a": math.log(1.0), "b": math.log(3.0)}) == pytest.approx(
        {"a": 0.25, "b": 0.75}
    )
    # Very negative sums (long continuations) must not underflow to 0/0.
    spread = module.normalise({"a": -2000.0, "b": -2001.0})
    assert math.isclose(sum(spread.values()), 1.0)
    assert spread["a"] > spread["b"] > 0


def test_every_continuation_stops_at_the_character_that_proves_the_choice() -> None:
    module = _module()
    entry = _first(_entries(module), "operation")
    platform = tier_bench.world_platform(tier_bench.load_world(_DEV))
    tokenizer = _FakeTokenizer()
    scorer = _FakeScorer()
    module.score_entry(tokenizer, scorer, entry, platform)
    texts = {tokenizer.decode(cont) for _, cont in scorer.calls}
    assert len(scorer.calls) == len(_labels())
    # One shared prompt for every candidate, rendered with thinking off.
    prompts = {tuple(prompt) for prompt, _ in scorer.calls}
    assert len(prompts) == 1
    assert tokenizer.decode(next(iter(prompts))).endswith("<think>\n\n</think>\n\n")
    for name in ops_table.names():
        assert f"<tool_call>\n<function=propose>\n<parameter=operation>\n{name}\n" in texts
    assert "<tool_call>\n<function=explain>" in texts
    assert "<tool_call>\n<function=escalate>" in texts


def test_the_prompt_is_the_training_conversation(monkeypatch) -> None:
    """System brief, tools and user message come from build_dataset.py's own builder."""
    module = _module()
    entry = _first(_entries(module), "explain")
    platform = tier_bench.world_platform(tier_bench.load_world(_DEV))
    seen = {}

    class _Recording(_FakeTokenizer):
        def apply_chat_template(self, messages, tools=None, **kwargs):
            if kwargs.get("add_generation_prompt"):
                seen["messages"], seen["tools"] = messages, tools
            return super().apply_chat_template(messages, tools=tools, **kwargs)

    module.score_entry(_Recording(), _FakeScorer(), entry, platform)
    example = module.build_dataset.example_from_entry(entry, platform)
    assert seen["messages"] == example["messages"][:-1]
    assert seen["tools"] == example["tools"]


# --- the torch scorer --------------------------------------------------------


def test_the_transformers_scorer_sums_the_continuation_token_logprobs() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    vocab = 7
    table = torch.randn(vocab, vocab)

    class _Output:
        def __init__(self, logits):
            self.logits = logits

    class _Model(torch.nn.Module):
        """Next-token logits depend on the current token only; honours logits_to_keep."""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, logits_to_keep=0):
            logits = table[input_ids]
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:]
            return _Output(logits)

    prompt, continuation = [1, 2, 3], [4, 5]
    got = module.TransformersContinuationScorer(_Model()).logprob(prompt, continuation)
    logprobs = torch.log_softmax(table, dim=-1)
    want = float(logprobs[3, 4] + logprobs[4, 5])
    assert got == pytest.approx(want, abs=1e-5)


# --- predictions file and CLI ------------------------------------------------


def _split(tmp_path: Path, name: str, header: str, count: int = 3) -> Path:
    raw = json.loads(_DEV.read_text(encoding="utf-8"))
    kinds = ["operation", "explain", "escalate"]
    entries = []
    for kind in kinds:
        for item in raw["entries"]:
            expect = item.get("expect", {})
            if (kind == "operation" and expect.get("operation")) or expect.get(kind):
                entries.append(item)
                break
    path = tmp_path / name
    path.write_text(
        json.dumps({"header": header, "world": raw["world"], "entries": entries[:count]}),
        encoding="utf-8",
    )
    return path


def _predictions(tmp_path: Path, split: Path) -> Path:
    lines = []
    for item in json.loads(split.read_text())["entries"]:
        lines.append(
            {
                "id": item["id"],
                "expected": item["expect"],
                "outcome": "escalate",
                "operation": None,
                "arguments": None,
                "candidates": None,
                "tokens": 12,
                "ttfd_ms": 3.5,
                "latency_ms": 9.0,
                "extra": "kept",
            }
        )
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "merged"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def _fake_loader(module, monkeypatch, scorer=None):
    calls = []

    def load(model_dir):
        calls.append(model_dir)
        return _FakeTokenizer(), scorer or _FakeScorer(), {"torch": "t-1", "transformers": "x-2"}

    monkeypatch.setattr(module, "load_model", load)
    return calls


def test_cli_fills_candidates_and_keeps_every_other_field(tmp_path, monkeypatch) -> None:
    module = _module()
    _fake_loader(module, monkeypatch)
    split = _split(tmp_path, "val.json", "Development split. Split 'val' of dev.json (seed=39).")
    predictions = _predictions(tmp_path, split)
    out = tmp_path / "out.jsonl"
    model = _model_dir(tmp_path)
    code = module.main(
        [
            "--model",
            str(model),
            "--split",
            str(split),
            "--predictions",
            str(predictions),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    before = [json.loads(line) for line in predictions.read_text().splitlines()]
    after = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["id"] for line in after] == [line["id"] for line in before]
    for old, new in zip(before, after):
        assert {k: v for k, v in new.items() if k != "candidates"} == {
            k: v for k, v in old.items() if k != "candidates"
        }
        assert set(new["candidates"]) == _labels()
        assert math.isclose(sum(new["candidates"].values()), 1.0, abs_tol=1e-6)
    # metrics.py accepts the file it wrote.
    metrics = module.metrics.read_predictions(out)
    assert all(p.candidates is not None for p in metrics)

    sidecar = json.loads(module.sidecar_path(out).read_text())
    assert sidecar["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert sidecar["model_revision"] == module.stage_cache.revision_of(model)
    assert sidecar["model"].endswith("merged")
    assert sidecar["torch"] == "t-1"
    assert sidecar["transformers"] == "x-2"
    assert sidecar["lines"] == len(after)


def test_cli_refuses_the_test_side_without_final(tmp_path, monkeypatch, capsys) -> None:
    module = _module()
    calls = _fake_loader(module, monkeypatch)
    split = _split(tmp_path, "test.json", "Split 'test' of dev.json (seed=39).")
    predictions = _predictions(tmp_path, split)
    argv = ["--model", str(_model_dir(tmp_path)), "--split", str(split)]
    argv += ["--predictions", str(predictions), "--out", str(tmp_path / "o.jsonl")]
    assert module.main(argv) == 1
    assert "--final" in capsys.readouterr().err
    assert calls == []  # refused before any model loads
    assert module.main(argv + ["--final"]) == 0


def test_cli_refuses_held_out_without_acceptance(tmp_path, monkeypatch, capsys) -> None:
    module = _module()
    calls = _fake_loader(module, monkeypatch)
    split = _split(tmp_path, "held-out.json", "Held-out split for acceptance.")
    predictions = _predictions(tmp_path, split)
    argv = ["--model", str(_model_dir(tmp_path)), "--split", str(split)]
    argv += ["--predictions", str(predictions), "--out", str(tmp_path / "o.jsonl")]
    assert module.main(argv) == 1
    assert "--acceptance" in capsys.readouterr().err
    assert calls == []
    assert module.main(argv + ["--acceptance"]) == 0


def test_cli_refuses_final_on_a_val_split(tmp_path, monkeypatch) -> None:
    module = _module()
    _fake_loader(module, monkeypatch)
    split = _split(tmp_path, "val.json", "Split 'val' of dev.json (seed=39).")
    predictions = _predictions(tmp_path, split)
    argv = ["--model", str(_model_dir(tmp_path)), "--split", str(split), "--final"]
    argv += ["--predictions", str(predictions), "--out", str(tmp_path / "o.jsonl")]
    assert module.main(argv) == 1


def test_cli_refuses_predictions_that_do_not_match_the_split(tmp_path, monkeypatch, capsys):
    module = _module()
    _fake_loader(module, monkeypatch)
    split = _split(tmp_path, "val.json", "Split 'val' of dev.json (seed=39).")
    predictions = _predictions(tmp_path, split)
    lines = predictions.read_text().splitlines()
    predictions.write_text("\n".join(lines[:-1]) + "\n")
    argv = ["--model", str(_model_dir(tmp_path)), "--split", str(split)]
    argv += ["--predictions", str(predictions), "--out", str(tmp_path / "o.jsonl")]
    assert module.main(argv) == 1
    assert "no predictions line" in capsys.readouterr().err


def test_cli_refuses_to_overwrite_its_input(tmp_path, monkeypatch) -> None:
    module = _module()
    _fake_loader(module, monkeypatch)
    split = _split(tmp_path, "val.json", "Split 'val' of dev.json (seed=39).")
    predictions = _predictions(tmp_path, split)
    argv = ["--model", str(_model_dir(tmp_path)), "--split", str(split)]
    argv += ["--predictions", str(predictions), "--out", str(predictions)]
    assert module.main(argv) == 1


# --- the real Qwen3.5 template -----------------------------------------------


def _qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            QWEN, revision=QWEN_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip(f"{QWEN}@{QWEN_REVISION} tokenizer not in the local Hugging Face cache")


@pytest.mark.parametrize("kind", ["operation", "explain", "escalate"])
def test_continuations_are_prefixes_of_the_templates_own_gold_turn(kind) -> None:
    tokenizer = _qwen_tokenizer()
    module = _module()
    entries = _entries(module)
    entry = _first(entries, kind)
    platform = tier_bench.world_platform(tier_bench.load_world(_DEV))
    example = module.build_dataset.example_from_entry(entry, platform)
    extra = module.train.thinking_off_kwargs(module.train._template_text(tokenizer))
    gold_full = tokenizer.apply_chat_template(
        example["messages"], tools=example["tools"], tokenize=False, **extra
    )
    gold_label = module.metrics.expected_label(entry.expect)
    continuations = module.continuations(tokenizer, entry, platform)
    assert set(continuations) == _labels()
    gold = continuations[gold_label]
    prompt_text = tokenizer.decode(gold.prompt_ids)
    assert gold_full.startswith(prompt_text + gold.text)
    assert gold_full.startswith(tokenizer.decode(gold.prompt_ids + gold.continuation_ids))
    # The text ends right after the deciding name and the character that closes it.
    if kind == "operation":
        assert gold.text.endswith(f"{entry.expect['operation']}\n")
    else:
        assert gold.text.endswith(f"{module.metrics.expected_label(entry.expect)[1:-1]}>")
    # Every candidate's tokens are distinct (none a prefix of another) and share the prompt.
    seqs = [tuple(c.continuation_ids) for c in continuations.values()]
    assert len(set(seqs)) == len(seqs)
    for one in seqs:
        assert not any(other != one and other[: len(one)] == one for other in seqs)
    assert len({tuple(c.prompt_ids) for c in continuations.values()}) == 1
