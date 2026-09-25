"""scripts/lfm-finetune/train_scorer.py (issue 53, t16): per-row label maps and calibration loss.

Each training row carries its own offered candidates and letter map (a
stored ``scorer.Permutation``), so its label columns and cross-entropy
target come from that row, not from one fixed list. ``--label-readout``
picks the label-probability definition (``variants``, the shared t1
definition, by default; ``single`` reproduces scorer-b1 exactly), and
``--label-smoothing`` / ``--brier-weight`` add calibration terms, both off
by default. The pure helpers run everywhere; the loss tests need torch.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "lfm-finetune" / "train_scorer.py"
_WORLD = json.loads((_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json").read_text())["world"]


def _module():
    spec = importlib.util.spec_from_file_location("lfm_train_scorer_rows", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def _split(tmp_path: Path, side: str, entries: list[dict], name: str | None = None) -> Path:
    path = tmp_path / (name or f"{side}.json")
    payload = {
        "header": f"Development corpus. Split '{side}' of dev.json (seed=46).",
        "entries": entries,
        "world": _WORLD,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(entry_id: str, text: str, expect: dict, **extra) -> dict:
    return {
        "id": entry_id,
        "kind": "explicit",
        "text": text,
        "expect": expect,
        "source_id": "s",
        **extra,
    }


class _Tokenizer:
    """One token per character of the rendered prompt."""

    chat_template = "{{ messages }}"
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 250 + 1 for char in text]


def _perm(module, seed, **kwargs) -> dict:
    return module.scorer.permute(seed, **kwargs).to_json()


# -- per-row maps from the split file --


def test_read_split_keeps_each_rows_stored_permutation_seed_and_descriptions(tmp_path) -> None:
    module = _module()
    perm = _perm(module, 7, subset=5, keep="thermal_stats")
    entries = [
        _entry(
            "p1",
            "How hot is this machine?",
            {"operation": "thermal_stats", "args": {}},
            permutation=perm,
            perm_seed=7,
            descriptions={"thermal_stats": "Read the temperatures."},
        ),
        _entry("f1", "What is swap?", {"explain": True, "answer": "Disk used as memory."}),
    ]
    permuted, fixed = module.read_split(_split(tmp_path, "train", entries), module.TRAIN_SIDE)
    assert permuted.permutation == module.scorer.Permutation.from_json(perm)
    assert permuted.perm_seed == 7
    assert permuted.descriptions == {"thermal_stats": "Read the temperatures."}
    assert permuted.gold == "thermal_stats"
    assert fixed.permutation is None and fixed.perm_seed is None and fixed.descriptions is None


def test_a_stored_gold_must_agree_with_the_expect_block(tmp_path) -> None:
    module = _module()
    reason = _entry("r1", "Rewrite my kernel", {"escalate": True}, gold="escalate:repair")
    [example] = module.read_split(_split(tmp_path, "train", [reason]), module.TRAIN_SIDE)
    assert example.gold == "escalate:repair"
    wrong = _entry("w1", "How hot?", {"operation": "thermal_stats", "args": {}}, gold="gpu_stats")
    with pytest.raises(ValueError, match="gold"):
        module.read_split(_split(tmp_path, "train", [wrong]), module.TRAIN_SIDE)


def test_a_gold_the_rows_permutation_does_not_offer_is_refused(tmp_path) -> None:
    module = _module()
    perm = {"order": ["gpu_stats", "explain"], "labels": {"gpu_stats": "k", "explain": "B"}}
    entry = _entry("x", "How hot?", {"operation": "thermal_stats", "args": {}}, permutation=perm)
    with pytest.raises(ValueError, match="not offered"):
        module.read_split(_split(tmp_path, "train", [entry]), module.TRAIN_SIDE)


def test_a_permutation_that_reuses_a_letter_is_refused(tmp_path) -> None:
    module = _module()
    perm = {"order": ["thermal_stats", "explain"], "labels": {"thermal_stats": "Q", "explain": "Q"}}
    entry = _entry("x", "How hot?", {"operation": "thermal_stats", "args": {}}, permutation=perm)
    with pytest.raises(ValueError, match="letter"):
        module.read_split(_split(tmp_path, "train", [entry]), module.TRAIN_SIDE)


# -- encoding: per-row letters and targets --


def test_two_rows_with_different_maps_get_different_targets_for_the_same_gold(tmp_path) -> None:
    module = _module()
    first = _perm(module, 1)
    second = _perm(module, 2)
    assert first["order"].index("thermal_stats") != second["order"].index("thermal_stats")
    entries = [
        _entry("a", "How hot?", {"operation": "thermal_stats", "args": {}}, permutation=first),
        _entry("b", "How hot?", {"operation": "thermal_stats", "args": {}}, permutation=second),
    ]
    examples = module.read_split(_split(tmp_path, "train", entries), module.TRAIN_SIDE)
    rows = module.encode(_Tokenizer(), examples, max_length=100_000)
    for row, perm in zip(rows, (first, second)):
        assert row["target"] == perm["order"].index("thermal_stats")
        assert row["letters"] == [perm["labels"][name] for name in perm["order"]]
        assert row["letters"][row["target"]] == perm["labels"]["thermal_stats"]
    assert rows[0]["target"] != rows[1]["target"]
    assert rows[0]["input_ids"] != rows[1]["input_ids"]  # the prompts list them differently


def test_a_permuted_row_renders_the_prompt_from_its_own_map(tmp_path) -> None:
    module = _module()
    perm = _perm(module, 5, subset=3, keep="escalate")
    descriptions = {"escalate": "Hand it up."}
    entry = _entry(
        "a",
        "Rewrite my kernel",
        {"escalate": True},
        permutation=perm,
        descriptions=descriptions,
    )
    examples = module.read_split(_split(tmp_path, "train", [entry]), module.TRAIN_SIDE)
    tokenizer = _Tokenizer()
    [row] = module.encode(tokenizer, examples, max_length=100_000)
    expected = module.scorer.render_prompt(
        tokenizer,
        module.scorer.prompt_messages(
            "Rewrite my kernel",
            labels=perm["labels"],
            order=perm["order"],
            descriptions=descriptions,
        ),
    )
    assert row["input_ids"] == tokenizer.encode(expected)
    assert len(row["letters"]) == 3


def test_a_row_without_a_permutation_keeps_the_fixed_map_and_prompt(tmp_path) -> None:
    module = _module()
    entry = _entry("a", "How hot?", {"operation": "thermal_stats", "args": {}})
    examples = module.read_split(_split(tmp_path, "train", [entry]), module.TRAIN_SIDE)
    tokenizer = _Tokenizer()
    [row] = module.encode(tokenizer, examples, max_length=100_000)
    names = module.scorer.candidates()
    fixed = module.scorer.labels_for(names)
    assert row["target"] == names.index("thermal_stats")
    assert row["letters"] == [fixed[name] for name in names]
    prompt = module.scorer.render_prompt(tokenizer, module.scorer.prompt_messages("How hot?"))
    assert row["input_ids"] == tokenizer.encode(prompt)


def test_attach_columns_maps_each_rows_letters_to_their_token_ids() -> None:
    module = _module()
    rows = [
        {"input_ids": [1], "target": 0, "letters": ["B", "A"]},
        {"input_ids": [2], "target": 1, "letters": ["A", "C", "B"]},
    ]
    ids = {"A": (10, 11), "B": (20,), "C": (30, 31, 32)}
    attached = module.attach_columns(rows, ids)
    assert attached[0]["columns"] == [(20,), (10, 11)]
    assert attached[1]["columns"] == [(10, 11), (30, 31, 32), (20,)]
    assert "columns" not in rows[0]  # the input rows are left alone


def test_letter_ids_single_and_variants_readouts() -> None:
    module = _module()

    class _Vocab(_Tokenizer):
        _vocab = {"A": 5, " A": 6, "\tA": 7, "B": 8, " B": 9, "x": 10}

        def get_vocab(self):
            return dict(self._vocab)

        def decode(self, ids):
            inverse = {v: k for k, v in self._vocab.items()}
            return "".join(inverse[i] for i in ids)

        def encode(self, text, add_special_tokens=False):
            return [self._vocab[text]]

    tokenizer = _Vocab()
    assert module.letter_ids(tokenizer, ["A", "B"], "variants") == {"A": (5, 6, 7), "B": (8, 9)}
    assert module.letter_ids(tokenizer, ["B", "A"], "single") == {"A": (5,), "B": (8,)}
    with pytest.raises(ValueError, match="readout"):
        module.letter_ids(tokenizer, ["A"], "other")


def test_the_label_readout_defaults_to_variants_and_calibration_terms_to_off() -> None:
    module = _module()
    args = module._parser().parse_args(["--train", "t.json", "--out", "o"])
    assert args.label_readout == "variants"
    assert args.label_smoothing == 0.0
    assert args.brier_weight == 0.0
    single = module._parser().parse_args(
        ["--train", "t.json", "--out", "o", "--label-readout", "single"]
    )
    assert single.label_readout == "single"
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--train", "t", "--out", "o", "--label-readout", "x"])


def test_permutation_record_logs_seeds_counts_and_per_row_maps(tmp_path) -> None:
    module = _module()
    perm = _perm(module, 9)
    entries = [
        _entry(
            "a",
            "How hot?",
            {"operation": "thermal_stats", "args": {}},
            permutation=perm,
            perm_seed=9,
            descriptions={"thermal_stats": "Temps."},
        ),
        _entry("b", "Rewrite my kernel", {"escalate": True}),
    ]
    train = module.read_split(_split(tmp_path, "train", entries), module.TRAIN_SIDE)
    val = module.read_split(_split(tmp_path, "val", entries[:1]), module.VAL_SIDE)
    summary, maps = module.permutation_record({"train": train, "val": val})
    assert summary["perm_seeds"] == [9]
    assert summary["permuted_rows"] == {"train": 1, "val": 1}
    assert summary["fixed_rows"] == {"train": 1, "val": 0}
    assert maps == [
        {
            "side": side,
            "entry_id": "a",
            "perm_seed": 9,
            "order": perm["order"],
            "labels": perm["labels"],
            "gold": "thermal_stats",
            "target": perm["order"].index("thermal_stats"),
            "descriptions": ["thermal_stats"],
        }
        for side in ("train", "val")
    ]
    path = tmp_path / "row-maps.json"
    written = module.write_row_maps(path, maps)
    assert written["file"] == str(path)
    assert written["rows"] == 2
    assert len(written["sha256"]) == 64
    assert json.loads(path.read_text()) == maps


# -- the loss (torch) --


class _OldReference:
    """The pre-t16 loss path, copied verbatim, so ``single`` + fixed map is checked against it."""

    @staticmethod
    def label_logits(torch, model, input_ids, attention_mask, label_ids):
        last = attention_mask.sum(dim=1) - 1
        keep, where = torch.unique(last, return_inverse=True)
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep
        ).logits
        rows = logits[torch.arange(input_ids.shape[0], device=logits.device), where]
        return rows.index_select(-1, label_ids.to(rows.device)).float()

    @staticmethod
    def train_loop(torch, module, model, rows, label_ids, *, epochs, lr, batch, seed, pad_id):
        labels = torch.tensor(label_ids)
        order_rng = random.Random(seed)  # nosec B311
        optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        model.train()
        history = []
        for _ in range(epochs):
            order = list(range(len(rows)))
            order_rng.shuffle(order)
            for start in range(0, len(order), batch):
                chunk = [rows[i] for i in order[start : start + batch]]
                input_ids, mask = module.pad_right([row["input_ids"] for row in chunk], pad_id)
                targets = torch.tensor([row["target"] for row in chunk])
                logits = _OldReference.label_logits(torch, model, input_ids, mask, labels)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                history.append(loss.item())
        return history


def _toy(torch):
    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(64, 16)
            self.head = torch.nn.Linear(16, 64)

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            hidden = self.embed(input_ids).cumsum(dim=1)
            if isinstance(logits_to_keep, int):
                keep = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
            else:
                keep = logits_to_keep
            return type("Out", (), {"logits": self.head(hidden[:, keep, :])})()

    return _Model


def _fixed_rows(n_labels: int = 4) -> list[dict]:
    return [
        {"input_ids": [1 + (i % 8), 9, 2 + (i % 5)] + [3] * (i % 3), "target": i % 8 % n_labels}
        for i in range(24)
    ]


def test_single_readout_on_the_fixed_map_matches_the_old_code_path_exactly() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    model_cls = _toy(torch)
    label_ids = [40, 41, 42, 43]
    rows = _fixed_rows()
    letters = ["A", "B", "C", "D"]
    single = module.attach_columns(
        [{**row, "letters": letters} for row in rows],
        {letter: (label_ids[i],) for i, letter in enumerate(letters)},
    )

    torch.manual_seed(3)
    old = _OldReference.train_loop(
        torch, module, model_cls(), rows, label_ids, epochs=3, lr=0.05, batch=4, seed=46, pad_id=0
    )
    torch.manual_seed(3)
    new = module.train_loop(
        model_cls(), single, None, epochs=3, lr=0.05, batch=4, seed=46, pad_id=0
    )
    assert [step["loss"] for step in new] == old  # bit-identical, not just close
    torch.manual_seed(3)
    legacy_call = module.train_loop(
        model_cls(), rows, label_ids, epochs=3, lr=0.05, batch=4, seed=46, pad_id=0
    )
    assert [step["loss"] for step in legacy_call] == old


def test_variant_readout_is_the_log_sum_exp_of_each_labels_variants() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    vocab = torch.randn(2, 64)
    columns = [[(40, 41), (42,)], [(42,), (43, 44, 45), (40, 41)]]
    logits, mask = module.column_logits(vocab, columns)
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert torch.allclose(logits[0, 0], torch.logsumexp(vocab[0, [40, 41]], dim=0))
    assert torch.allclose(logits[1, 1], torch.logsumexp(vocab[1, [43, 44, 45]], dim=0))
    assert logits[0, 2] == float("-inf")
    expected = module.scorer.label_logits_from_vocab(vocab[1:2].float(), columns[1])
    assert torch.allclose(logits[1], expected[0])


def test_single_readout_reads_the_one_id_and_masks_the_padding() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    vocab = torch.randn(2, 64)
    logits, mask = module.column_logits(vocab, [[(40,), (41,)], [(42,), (43,), (44,)]])
    assert logits[0, :2].tolist() == vocab[0, [40, 41]].tolist()
    assert logits[1].tolist() == vocab[1, [42, 43, 44]].tolist()
    assert logits[0, 2] == float("-inf") and not bool(mask[0, 2])


def test_a_mixed_batch_gives_each_row_the_loss_it_has_alone() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    vocab = torch.randn(3, 64, requires_grad=True)
    columns = [[(40, 41), (42,)], [(43,), (44, 45), (46,), (47,)], [(48,), (49,), (50,)]]
    targets = torch.tensor([1, 2, 0])
    logits, mask = module.column_logits(vocab, columns)
    loss = module.label_loss(logits, mask, targets)
    assert torch.isfinite(loss)
    alone = [
        torch.nn.functional.cross_entropy(
            module.column_logits(vocab[i : i + 1], [columns[i]])[0], targets[i : i + 1]
        )
        for i in range(3)
    ]
    assert torch.allclose(loss, torch.stack(alone).mean())
    loss.backward()
    assert torch.isfinite(vocab.grad).all()


def test_the_default_loss_is_plain_cross_entropy() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    logits = torch.randn(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    targets = torch.tensor([0, 3, 1, 4])
    plain = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.equal(module.label_loss(logits, mask, targets), plain)
    assert torch.equal(
        module.label_loss(logits, mask, targets, label_smoothing=0.0, brier_weight=0.0), plain
    )


def test_label_smoothing_matches_torch_on_a_full_row_and_ignores_masked_columns() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    logits = torch.randn(4, 5)
    full = torch.ones(4, 5, dtype=torch.bool)
    targets = torch.tensor([0, 3, 1, 2])
    smoothed = module.label_loss(logits, full, targets, label_smoothing=0.1)
    expected = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.1)
    assert torch.allclose(smoothed, expected)
    assert not torch.allclose(smoothed, torch.nn.functional.cross_entropy(logits, targets))
    # Smoothing spreads only over a row's offered labels: a masked column changes nothing.
    offered = logits[:, :3]
    mask = torch.tensor([[True] * 3 + [False] * 2] * 4)
    padded = logits.masked_fill(~mask, float("-inf"))
    in_row = module.label_loss(padded, mask, torch.tensor([0, 2, 1, 2]), label_smoothing=0.2)
    reference = torch.nn.functional.cross_entropy(
        offered, torch.tensor([0, 2, 1, 2]), label_smoothing=0.2
    )
    assert torch.allclose(in_row, reference)


def test_the_brier_term_adds_the_multi_class_brier_score_over_offered_labels() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    logits = torch.tensor([[2.0, 0.5, -1.0, float("-inf")], [0.0, 1.0, 0.3, 0.2]])
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    targets = torch.tensor([0, 2])
    ce = module.label_loss(logits, mask, targets)
    with_brier = module.label_loss(logits, mask, targets, brier_weight=0.5)
    probabilities = [torch.softmax(logits[0, :3], -1), torch.softmax(logits[1], -1)]
    briers = [
        sum((p - (1.0 if j == int(t) else 0.0)) ** 2 for j, p in enumerate(row.tolist()))
        for row, t in zip(probabilities, targets)
    ]
    assert with_brier.item() == pytest.approx(ce.item() + 0.5 * sum(briers) / 2, rel=1e-6)


def test_train_loop_on_permuted_rows_learns_and_the_calibration_terms_change_it() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    model_cls = _toy(torch)
    # Rows offer two to four labels each, over different ids; the first token picks the gold.
    rows = []
    for i in range(24):
        width = 2 + i % 3
        columns = [(40 + (i + j) % 10, 50 + (i + j) % 10) for j in range(width)]
        rows.append(
            {"input_ids": [1 + i % 8, 9, 2 + i % 5], "target": i % width, "columns": columns}
        )

    def run(**loss) -> list[float]:
        torch.manual_seed(0)
        history = module.train_loop(
            model_cls(), rows, None, epochs=20, lr=0.05, batch=4, seed=46, pad_id=0, **loss
        )
        return [step["loss"] for step in history]

    plain = run()
    assert plain[-1] < plain[0]
    assert run(label_smoothing=0.1) != plain
    assert run(brier_weight=0.5) != plain
    torch.manual_seed(0)
    model = model_cls()
    module.train_loop(model, rows, None, epochs=20, lr=0.05, batch=4, seed=46, pad_id=0)
    report = module.evaluate(model, rows, None, batch=4, pad_id=0)
    assert report["n"] == 24
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["loss"] >= 0.0


def test_calibration_terms_out_of_range_are_refused() -> None:
    torch = pytest.importorskip("torch")
    module = _module()
    logits, mask, targets = torch.randn(1, 3), torch.ones(1, 3, dtype=torch.bool), torch.tensor([0])
    with pytest.raises(ValueError, match="smoothing"):
        module.label_loss(logits, mask, targets, label_smoothing=1.0)
    with pytest.raises(ValueError, match="brier"):
        module.label_loss(logits, mask, targets, brier_weight=-0.1)


@pytest.mark.parametrize(
    "flags", [["--label-smoothing", "1.0"], ["--label-smoothing", "-0.1"], ["--brier-weight", "-1"]]
)
def test_the_parser_refuses_calibration_terms_out_of_range(flags: list[str]) -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--train", "t.json", "--out", "o", *flags])
    ok = module._parser().parse_args(
        ["--train", "t", "--out", "o", "--label-smoothing", "0.1", "--brier-weight", "0.5"]
    )
    assert (ok.label_smoothing, ok.brier_weight) == (0.1, 0.5)


# -- PR #65 review: validation in reasons mode --


def test_reasons_mode_training_validates_on_the_reasons_pool(tmp_path) -> None:
    """With reasons-mode training rows, a validation row with no stored map is
    scored on the 25-candidate reasons pool (as measure.py --reasons does), its
    escalate gold named by its class -- not on the fixed 18-label map."""
    module = _module()
    pool = module.scorer.candidate_pool(reasons=True)
    train_perm = module.scorer.Permutation(
        order=tuple(pool), labels=module.scorer.positional_labels(pool, pool)
    ).to_json()
    train = module.read_split(
        _split(
            tmp_path,
            "train",
            [
                _entry(
                    "t1",
                    "restart it",
                    {"escalate": True},
                    **{
                        "class": "decline:missing_argument",
                        "permutation": train_perm,
                        "gold": "escalate:missing_argument",
                    },
                ),
            ],
        ),
        module.TRAIN_SIDE,
    )
    val = module.read_split(
        _split(
            tmp_path,
            "val",
            [
                _entry(
                    "v1", "restart it", {"escalate": True}, **{"class": "decline:missing_argument"}
                ),
                _entry("v2", "how busy is the gpu", {"operation": "gpu_stats", "args": {}}),
            ],
        ),
        module.VAL_SIDE,
    )
    assert module.reasons_mode(train)
    matched = module.match_validation(train, val)
    assert matched[0].gold == "escalate:missing_argument"
    assert tuple(matched[0].permutation.order) == tuple(pool)
    assert matched[1].gold == "gpu_stats"
    rows = module.encode(_Tokenizer(), matched, 100000)
    assert len(rows[0]["letters"]) == len(pool)


def test_plain_training_leaves_validation_unchanged(tmp_path) -> None:
    module = _module()
    train = module.read_split(
        _split(tmp_path, "train", [_entry("t1", "gpu?", {"operation": "gpu_stats", "args": {}})]),
        module.TRAIN_SIDE,
    )
    val = module.read_split(
        _split(tmp_path, "val", [_entry("v1", "gpu?", {"operation": "gpu_stats", "args": {}})]),
        module.VAL_SIDE,
    )
    assert not module.reasons_mode(train)
    assert module.match_validation(train, val) == val
