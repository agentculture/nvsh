"""build_dataset.py --scorer-out -> train_scorer.py (issue 53, deviation d2).

t14 enriches the rendered Track A examples (``nvsh-train.jsonl``) with each
row's permutation, gold and seed, but Track B (``train_scorer.py``) reads a
corpus-format split file by entry id. ``--scorer-out`` writes that file from
the same computation: the original entries plus their derived ``-nocand``
entries, each carrying the permutation/gold/perm_seed/descriptions the
rendered example got. These tests prove the rows that reach the scorer's
trainer carry the randomized maps and the missing-candidate rows, and that
the file and the rendered output never diverge. Nothing here needs torch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts" / "lfm-finetune"
_WORLD = json.loads((_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json").read_text())["world"]

#: Table operations the fixture uses (all argument-free).
_OPS = ("thermal_stats", "gpu_stats", "disk_stats", "memory_stats")


def _load(name: str, script: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_dataset():
    return _load("lfm_d2_build_dataset", "build_dataset.py")


@pytest.fixture(scope="module")
def train_scorer():
    return _load("lfm_d2_train_scorer", "train_scorer.py")


def _entries() -> list[dict]:
    entries = []
    for index, op in enumerate(_OPS * 3):
        entries.append(
            {
                "id": f"op{index:02d}",
                "kind": "explicit",
                "text": f"please show {op.replace('_', ' ')} now ({index})",
                "expect": {"operation": op, "args": {}},
                "class": "imperative",
                "source_id": f"src-op{index:02d}",
                "source": "fixture",
            }
        )
    entries.append(
        {
            "id": "esc00",
            "kind": "explicit",
            "text": "reflash the bootloader and rebuild the kernel",
            "expect": {"escalate": True},
            "class": "decline:repair",
            "source_id": "src-esc00",
        }
    )
    entries.append(
        {
            "id": "exp00",
            "kind": "explicit",
            "text": "what does nvpmodel do?",
            "expect": {"explain": True, "answer": "It sets the power mode."},
            "source_id": "src-exp00",
        }
    )
    return entries


def _split(tmp_path: Path) -> Path:
    path = tmp_path / "train-augmented.json"
    payload = {
        "header": "Development corpus. Split 'train' of dev.json (seed=46).",
        "entries": _entries(),
        "world": _WORLD,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_main(build_dataset, tmp_path: Path, *extra: str) -> tuple[Path, Path]:
    split = _split(tmp_path)
    out = tmp_path / "nvsh-train.jsonl"
    scorer_out = tmp_path / "scorer-train.json"
    status = build_dataset.main(
        [
            "--split",
            str(split),
            "--out",
            str(out),
            "--no-verify-render",
            "--scorer-out",
            str(scorer_out),
            *extra,
        ]
    )
    assert status == 0
    return out, scorer_out


class _Tokenizer:
    """One token per character of the rendered prompt."""

    chat_template = "{{ messages }}"
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 250 + 1 for char in text]


# -- the file build_dataset writes --


def test_scorer_out_writes_every_original_entry_and_its_derived_nocand_entries(
    build_dataset, tmp_path
) -> None:
    _, scorer_out = _run_main(
        build_dataset, tmp_path, "--randomize-labels", "--missing-candidate-rate", "1.0"
    )
    written = json.loads(scorer_out.read_text(encoding="utf-8"))
    ids = [entry["id"] for entry in written["entries"]]
    originals = [entry["id"] for entry in _entries()]
    assert set(originals) <= set(ids)
    nocand = [entry_id for entry_id in ids if entry_id.endswith("-nocand")]
    op_ids = [entry["id"] for entry in _entries() if "operation" in entry["expect"]]
    assert sorted(nocand) == sorted(f"{entry_id}-nocand" for entry_id in op_ids)
    for entry in written["entries"]:
        assert {"permutation", "gold", "perm_seed"} <= set(entry)


def test_scorer_out_keeps_each_original_entrys_own_fields(build_dataset, tmp_path) -> None:
    _, scorer_out = _run_main(build_dataset, tmp_path, "--randomize-labels")
    written = {e["id"]: e for e in json.loads(scorer_out.read_text(encoding="utf-8"))["entries"]}
    for original in _entries():
        stored = written[original["id"]]
        for key, value in original.items():
            assert stored[key] == value, (original["id"], key)


def test_a_nocand_entry_expects_escalate_and_keeps_its_source(build_dataset, tmp_path) -> None:
    _, scorer_out = _run_main(
        build_dataset, tmp_path, "--randomize-labels", "--missing-candidate-rate", "1.0"
    )
    written = {e["id"]: e for e in json.loads(scorer_out.read_text(encoding="utf-8"))["entries"]}
    derived = written["op00-nocand"]
    assert derived["expect"] == {"escalate": True}
    assert derived["gold"] == "escalate"
    assert derived["source_id"] == "src-op00"
    assert derived["kind"] == "explicit"
    assert derived["text"] == written["op00"]["text"]
    assert "thermal_stats" not in derived["permutation"]["order"]


def test_reasons_mode_classes_a_nocand_entry_outside_table(build_dataset, tmp_path) -> None:
    _, scorer_out = _run_main(
        build_dataset,
        tmp_path,
        "--randomize-labels",
        "--missing-candidate-rate",
        "1.0",
        "--reasons",
    )
    written = {e["id"]: e for e in json.loads(scorer_out.read_text(encoding="utf-8"))["entries"]}
    derived = written["op00-nocand"]
    assert derived["class"] == "decline:outside_table"
    assert derived["gold"] == "escalate:outside_table"
    assert written["esc00"]["gold"] == "escalate:repair"
    assert "escalate:outside_table" in derived["descriptions"]


def test_scorer_out_matches_the_rendered_examples_row_for_row(build_dataset, tmp_path) -> None:
    """One computation, two writers: every map/gold/seed is identical in both files."""
    out, scorer_out = _run_main(
        build_dataset,
        tmp_path,
        "--randomize-labels",
        "--perm-seed",
        "53",
        "--missing-candidate-rate",
        "1.0",
        "--reasons",
    )
    rendered = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    entries = json.loads(scorer_out.read_text(encoding="utf-8"))["entries"]
    originals = [entry for entry in entries if not entry["id"].endswith("-nocand")]
    assert len(rendered) == len(originals)
    for example, entry in zip(rendered, originals):
        assert example["source_id"] == entry["source_id"]
        for key in ("permutation", "gold", "perm_seed"):
            assert example[key] == entry[key], (entry["id"], key)
        assert example.get("descriptions") == entry.get("descriptions")


def test_nocand_rows_reach_the_scorer_file_but_never_the_rendered_output(
    build_dataset, tmp_path
) -> None:
    """A rendered -nocand row would keep the full tools and propose enum, so it
    would pair its propose twin's exact messages/tools with the opposite target."""
    out, scorer_out = _run_main(
        build_dataset, tmp_path, "--randomize-labels", "--missing-candidate-rate", "1.0"
    )
    rendered = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rendered) == len(_entries())
    assert sorted(e["source_id"] for e in rendered) == sorted(e["source_id"] for e in _entries())
    by_source = {example["source_id"]: example for example in rendered}
    for original in _entries():
        if "operation" in original["expect"]:
            example = by_source[original["source_id"]]
            assert example["gold"] == original["expect"]["operation"]
            assert example["messages"][-1]["tool_calls"][0]["function"]["name"] == "propose"
    entries = json.loads(scorer_out.read_text(encoding="utf-8"))["entries"]
    assert sum(entry["id"].endswith("-nocand") for entry in entries) == 12


def test_missing_candidate_rate_without_scorer_out_warns_it_does_nothing(
    build_dataset, tmp_path, capsys
) -> None:
    split = _split(tmp_path)
    out = tmp_path / "nvsh-train.jsonl"
    argv = ["--split", str(split), "--out", str(out), "--no-verify-render"]
    assert build_dataset.main([*argv, "--missing-candidate-rate", "1.0"]) == 0
    assert "--scorer-out" in capsys.readouterr().err
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(_entries())


def test_scorer_out_keeps_the_split_header_and_records_provenance(build_dataset, tmp_path) -> None:
    _, scorer_out = _run_main(
        build_dataset,
        tmp_path,
        "--randomize-labels",
        "--perm-seed",
        "53",
        "--missing-candidate-rate",
        "1.0",
    )
    written = json.loads(scorer_out.read_text(encoding="utf-8"))
    assert "Split 'train' of dev.json (seed=46)." in written["header"]
    provenance = written["provenance"]
    assert provenance["source"].endswith("train-augmented.json")
    assert len(provenance["source_sha256"]) == 64
    assert provenance["randomize_labels"] is True
    assert provenance["reasons"] is False
    assert provenance["perm_seed"] == 53
    assert provenance["missing_candidate_rate"] == 1.0
    assert provenance["counts"] == {"entries": 26, "original": 14, "missing_candidate": 12}
    assert written["world"] == _WORLD


def test_scorer_out_is_optional(build_dataset, tmp_path) -> None:
    split = _split(tmp_path)
    out = tmp_path / "nvsh-train.jsonl"
    assert build_dataset.main(["--split", str(split), "--out", str(out), "--no-verify-render"]) == 0
    assert not (tmp_path / "scorer-train.json").exists()


@pytest.mark.parametrize("target", ["split", "out"])
def test_scorer_out_refuses_to_overwrite_the_source_or_the_rendered_output(
    build_dataset, tmp_path, target
) -> None:
    split = _split(tmp_path)
    out = tmp_path / "nvsh-train.jsonl"
    clash = split if target == "split" else out
    with pytest.raises(SystemExit):
        build_dataset.main(
            [
                "--split",
                str(split),
                "--out",
                str(out),
                "--no-verify-render",
                "--scorer-out",
                str(clash),
            ]
        )


def test_build_with_scorer_entries_returns_the_same_rendered_examples_as_build(
    build_dataset, tmp_path
) -> None:
    split = _split(tmp_path)
    options = {"randomize_labels": True, "perm_seed": 5, "missing_candidate_rate": 0.5}
    examples, entries = build_dataset.build_with_scorer_entries(split, is_split=True, **options)
    assert examples == build_dataset.build(split, is_split=True, **options)
    derived = [entry for entry in entries if entry["id"].endswith("-nocand")]
    assert 0 < len(derived) < len(examples)
    assert len(entries) == len(examples) + len(derived)


def test_a_derived_id_colliding_with_a_real_entry_is_refused(build_dataset, tmp_path) -> None:
    split = _split(tmp_path)
    payload = json.loads(split.read_text(encoding="utf-8"))
    clash = dict(payload["entries"][0], id="op00-nocand")
    payload["entries"].append(clash)
    split.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="op00-nocand"):
        build_dataset.build_with_scorer_entries(split, is_split=True, missing_candidate_rate=1.0)


# -- end to end: what train_scorer reads from it --


def test_train_scorer_reads_randomized_maps_and_nocand_rows_from_scorer_out(
    build_dataset, train_scorer, tmp_path
) -> None:
    _, scorer_out = _run_main(
        build_dataset,
        tmp_path,
        "--randomize-labels",
        "--perm-seed",
        "53",
        "--missing-candidate-rate",
        "1.0",
    )
    examples = train_scorer.read_split(scorer_out, "train")
    assert len(examples) == 26
    fixed_names = train_scorer.scorer.candidates()
    fixed_labels = train_scorer.scorer.labels_for(fixed_names)
    assert all(example.permutation is not None for example in examples)
    permuted = [
        e
        for e in examples
        if tuple(e.permutation.order) != fixed_names
        or any(e.permutation.labels[name] != fixed_labels.get(name) for name in e.permutation.order)
    ]
    assert permuted, "no row carries a non-default letter map"
    nocand = [e for e in examples if e.entry_id.endswith("-nocand")]
    assert len(nocand) == 12
    originals = {entry["id"]: entry for entry in _entries()}
    for example in nocand:
        assert example.gold == "escalate"
        gold_op = originals[example.entry_id.removesuffix("-nocand")]["expect"]["operation"]
        assert gold_op not in example.permutation.order
        assert "escalate" in example.permutation.order
    seeds = {example.perm_seed for example in examples}
    assert len(seeds) == len(examples)  # one derived seed per row


def test_train_scorer_reads_reason_rows_with_their_descriptions(
    build_dataset, train_scorer, tmp_path
) -> None:
    _, scorer_out = _run_main(
        build_dataset,
        tmp_path,
        "--randomize-labels",
        "--missing-candidate-rate",
        "1.0",
        "--reasons",
    )
    examples = {e.entry_id: e for e in train_scorer.read_split(scorer_out, "train")}
    assert examples["esc00"].gold == "escalate:repair"
    assert examples["op00-nocand"].gold == "escalate:outside_table"
    assert "escalate:outside_table" in (examples["op00-nocand"].descriptions or {})


def test_encoded_targets_are_each_rows_gold_position_in_its_own_map(
    build_dataset, train_scorer, tmp_path
) -> None:
    _, scorer_out = _run_main(
        build_dataset,
        tmp_path,
        "--randomize-labels",
        "--perm-seed",
        "53",
        "--missing-candidate-rate",
        "1.0",
        "--reasons",
    )
    examples = train_scorer.read_split(scorer_out, "train")
    rows = train_scorer.encode(_Tokenizer(), examples, max_length=100_000)
    assert len(rows) == len(examples)
    for example, row in zip(examples, rows):
        order = list(example.permutation.order)
        assert row["target"] == order.index(example.gold)
        assert row["letters"] == [example.permutation.labels[name] for name in order]
        gold_letter = example.permutation.labels[example.gold]
        assert row["letters"][row["target"]] == gold_letter
        messages = train_scorer.scorer.prompt_messages(
            example.request,
            labels=example.permutation.labels,
            order=example.permutation.order,
            descriptions=example.descriptions,
        )
        prompt = messages[0]["content"]
        assert f"{gold_letter}) {example.gold}:" in prompt


def test_the_default_build_still_gives_train_scorer_the_fixed_map(
    build_dataset, train_scorer, tmp_path
) -> None:
    """Without --randomize-labels every row keeps today's fixed order and letters."""
    _, scorer_out = _run_main(build_dataset, tmp_path)
    examples = train_scorer.read_split(scorer_out, "train")
    fixed_names = train_scorer.scorer.candidates()
    fixed_labels = train_scorer.scorer.labels_for(fixed_names)
    assert len(examples) == 14
    for example in examples:
        assert tuple(example.permutation.order) == fixed_names
        assert dict(example.permutation.labels) == fixed_labels
