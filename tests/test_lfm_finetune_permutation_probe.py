"""scripts/lfm-finetune/permutation_probe.py (issue 53, t7): the permutation probe.

A fake next-token scorer stands in for the model everywhere; no tokenizer or
GPU is needed. Two fakes exercise the invariant the probe is built to catch:

* ``_IdentityScorer`` always answers whichever offered candidate's *name*
  appears in the prompt (the name is rendered on every line regardless of
  order, letter or description text), so op-level answer changes must stay
  at 0 for every kind.
* ``_FirstListedScorer`` always answers the first offered candidate's
  current label, so ``order`` (and ``all``, which also reorders) must show
  answer changes, while ``letters`` (order unchanged) must not.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest

from nvsh.tiers.bench import CorpusEntry

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "lfm-finetune" / "permutation_probe.py"

#: A handful of real, zero-argument operations so descriptions resolve.
POOL = ("machine_status", "memory_stats", "gpu_stats", "disk_stats", "thermal_stats")
_LINE_RE = re.compile(r"^([A-Za-z])\) (\S+):", re.MULTILINE)


def _module():
    spec = importlib.util.spec_from_file_location("lfm_permutation_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _module()


def _render(messages: list[dict]) -> str:
    """A trivial renderer: joins message contents (no chat template needed)."""
    return "\n".join(message["content"] for message in messages)


def _lines(prompt: str) -> dict[str, str]:
    """``name -> label`` parsed back out of a rendered prompt."""
    return {name: label for label, name in _LINE_RE.findall(prompt)}


class _IdentityScorer:
    """Always answers *target*'s current label, wherever it landed.

    Returns a logprob for every offered label (not just the winner), the
    way a real top-K readout does for a small candidate pool -- so the
    result is always complete, never ``incomplete`` (issue 53).
    """

    def __init__(self, target: str) -> None:
        self.target = target

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        names = _lines(prompt)
        return {
            label: math.log(0.9 if name == self.target else 0.01) for name, label in names.items()
        }


class _FirstListedScorer:
    """Always answers whichever candidate's line comes first in the prompt.

    Returns a logprob for every offered label, so the result is always
    complete (issue 53) -- only the winner's identity, not the label set,
    depends on listing order.
    """

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        names = _lines(prompt)
        if not names:
            return {}
        first_label = _LINE_RE.search(prompt).group(1)
        return {label: math.log(0.9 if label == first_label else 0.01) for label in names.values()}


def _entry(entry_id: str, operation: str, text: str = "show me the machine") -> CorpusEntry:
    return CorpusEntry(
        id=entry_id, kind="explicit", text=text, expect={"operation": operation}, source="test"
    )


# ---------------------------------------------------------------------------
# gold_name / paraphrase loading
# ---------------------------------------------------------------------------


def test_gold_name_operation_escalate_explain(probe):
    assert probe.gold_name({"operation": "machine_status"}) == "machine_status"
    assert probe.gold_name({"escalate": True}) == "escalate"
    assert probe.gold_name({"explain": True}) == "explain"


def test_load_paraphrases_requires_two_alternatives(tmp_path, probe):
    path = tmp_path / "paraphrases.json"
    path.write_text(json.dumps({"machine_status": ["only one"]}), encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="fewer than 2"):
        probe.load_paraphrases(path)


def test_load_paraphrases_ok(tmp_path, probe):
    path = tmp_path / "paraphrases.json"
    data = {"machine_status": ["what state is this box in", "how is the machine doing"]}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert probe.load_paraphrases(path) == data


# ---------------------------------------------------------------------------
# The core probe: an identity-tracking fake shows (near) zero change
# ---------------------------------------------------------------------------


def test_identity_scorer_never_changes_answer(probe):
    entries = [_entry("e1", "machine_status")]
    scorer_obj = _IdentityScorer("machine_status")
    report = probe.run_probe(
        scorer_obj, _render, entries, pool=POOL, per_entry=12, seed="probe-seed"
    )
    kinds = {row["kind"]: row for row in report["kinds"]}
    for kind in ("order", "letters", "subset", "all"):
        assert kind in kinds, kind
        assert kinds[kind]["changes"] == 0, kind
        assert kinds[kind]["rate"] == 0.0, kind
    # No paraphrase file was given, so that kind is simply absent.
    assert "paraphrase" not in kinds


def test_identity_scorer_survives_paraphrase(probe):
    entries = [_entry("e1", "machine_status")]
    scorer_obj = _IdentityScorer("machine_status")
    paraphrases = {
        "machine_status": ["what state is this box in", "how is the machine doing"],
    }
    report = probe.run_probe(
        scorer_obj,
        _render,
        entries,
        pool=POOL,
        per_entry=8,
        seed="probe-seed",
        paraphrases=paraphrases,
    )
    kinds = {row["kind"]: row for row in report["kinds"]}
    assert kinds["paraphrase"]["changes"] == 0
    assert kinds["paraphrase"]["trials"] == 8


# ---------------------------------------------------------------------------
# A position-tracking fake must show changes exactly where order moves
# ---------------------------------------------------------------------------


def test_first_listed_scorer_changes_under_order_not_letters(probe):
    entries = [_entry("e1", "machine_status")]
    scorer_obj = _FirstListedScorer()
    report = probe.run_probe(
        scorer_obj, _render, entries, pool=POOL, per_entry=20, seed="probe-seed-2"
    )
    kinds = {row["kind"]: row for row in report["kinds"]}
    # "order" reorders the listing, so the first line is (almost always) a
    # different candidate than the baseline's first -- some real changes.
    assert kinds["order"]["changes"] > 0
    # "letters" keeps the listing order fixed; the first LINE is still the
    # same candidate, so the op-level choice never changes.
    assert kinds["letters"]["changes"] == 0
    # "all" also reorders, so it must show changes too.
    assert kinds["all"]["changes"] > 0


# ---------------------------------------------------------------------------
# Subset: always keeps the gold candidate; reports baseline-choice removal
# ---------------------------------------------------------------------------


def test_subset_keeps_gold_and_reports_baseline_removed(probe):
    entries = [_entry("e1", "machine_status")]
    scorer_obj = _FirstListedScorer()
    report = probe.run_probe(
        scorer_obj, _render, entries, pool=POOL, per_entry=25, seed="probe-seed-3"
    )
    kinds = {row["kind"]: row for row in report["kinds"]}
    subset = kinds["subset"]
    assert "baseline_choice_removed" in subset
    removed = subset["baseline_choice_removed"]
    assert removed["trials"] == subset["trials"]
    # With a 5-candidate pool and random subset sizes, the baseline's own
    # choice (whatever the first-listed fake picked) is dropped sometimes.
    assert 0 <= removed["removed"] <= removed["trials"]


def test_build_trial_subset_always_keeps_gold(probe):
    baseline_order = list(POOL)
    baseline_labels = probe.scorer.labels_for(POOL)
    for i in range(30):
        trial = probe.build_trial(
            "subset",
            probe.derive_seed("s", "e1", "subset", i),
            POOL,
            baseline_order,
            baseline_labels,
            "gpu_stats",
            "machine_status",
            None,
        )
        assert "gpu_stats" in trial.order
        assert set(trial.labels) == set(trial.order)


# ---------------------------------------------------------------------------
# Bootstrap CI is over entries, not trials
# ---------------------------------------------------------------------------


def test_kind_report_bootstraps_over_entries(probe):
    # Two entries: one always changes (10/10), one never changes (0/10).
    # A trial-level bootstrap over the flattened 20 trials would give a
    # narrower CI around 0.5 than an entry-level bootstrap, which can only
    # ever draw {0.0, 0.5, 1.0} (the only entry-count combinations from 2
    # entries) -- so the CI bounds must include 0.0 or 1.0, not be pinned
    # tightly to 0.5.
    outcome_changed = probe.EntryOutcome(
        entry_id="a",
        gold="machine_status",
        baseline_choice="machine_status",
        trials={"order": [(True, False, None, False)] * 10},
    )
    outcome_same = probe.EntryOutcome(
        entry_id="b",
        gold="machine_status",
        baseline_choice="machine_status",
        trials={"order": [(False, False, None, False)] * 10},
    )
    report = probe.kind_report([outcome_changed, outcome_same], "order", bootstrap_seed=0)
    assert report["trials"] == 20
    assert report["changes"] == 10
    assert report["rate"] == 0.5
    assert report["ci_low"] in (0.0, 0.5)
    assert report["ci_high"] in (0.5, 1.0)
    assert "entries" in report["bootstrap_note"]


def test_kind_report_label_case_counts(probe):
    outcome = probe.EntryOutcome(
        entry_id="a",
        gold="machine_status",
        baseline_choice="machine_status",
        trials={
            "letters": [
                (True, True, None, False),
                (False, False, None, False),
                (True, False, None, False),
            ]
        },
    )
    report = probe.kind_report([outcome], "letters", bootstrap_seed=0)
    case = report["label_case"]
    assert case["lowercase_trials"] == 1
    assert case["lowercase_changes"] == 1
    assert case["uppercase_trials"] == 2
    assert case["uppercase_changes"] == 1


def test_kind_report_returns_none_when_no_trials(probe):
    outcome = probe.EntryOutcome(
        entry_id="a", gold="machine_status", baseline_choice="machine_status", trials={}
    )
    assert probe.kind_report([outcome], "paraphrase", bootstrap_seed=0) is None


# ---------------------------------------------------------------------------
# Incomplete trials (issue 53): tallied separately, never scored as an answer
# ---------------------------------------------------------------------------


def test_kind_report_counts_incomplete_separately_from_changes(probe):
    outcome = probe.EntryOutcome(
        entry_id="a",
        gold="machine_status",
        baseline_choice="machine_status",
        trials={
            "letters": [
                (True, False, None, False),  # a real, scored answer change
                (False, False, None, False),  # a real, scored non-change
                (True, True, None, True),  # incomplete: must not count as a change
                (False, True, None, True),  # incomplete: must not count as a non-change either
            ]
        },
    )
    report = probe.kind_report([outcome], "letters", bootstrap_seed=0)
    # Only the two scored trials feed trials/changes/rate/label_case.
    assert report["trials"] == 2
    assert report["changes"] == 1
    assert report["rate"] == 0.5
    assert report["label_case"]["lowercase_trials"] == 0
    assert report["label_case"]["uppercase_trials"] == 2
    # The incomplete pair is tallied on its own, out of all 4 raw trials.
    assert report["incomplete"] == {"trials": 2, "of": 4, "rate": 0.5}


def test_kind_report_reports_when_every_trial_is_incomplete(probe):
    outcome = probe.EntryOutcome(
        entry_id="a",
        gold="machine_status",
        baseline_choice="machine_status",
        trials={"letters": [(True, False, None, True)] * 5},
    )
    report = probe.kind_report([outcome], "letters", bootstrap_seed=0)
    assert report is not None
    assert report["trials"] == 0
    assert report["changes"] == 0
    assert report["rate"] is None
    assert report["incomplete"] == {"trials": 5, "of": 5, "rate": 1.0}


class _NarrowAlphabetScorer:
    """Mimics the pre-fix in-process scorer: only ever reads labels A, B and C.

    A trial whose letter draw puts an offered candidate outside A-C is
    missing that label entirely, the same way ``TransformersScorer`` built
    from the fixed A-R training map cannot read a letter it was never told
    about (issue 53).
    """

    def __init__(self, target: str) -> None:
        self.target = target

    def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
        names = _lines(prompt)
        result = {}
        for name, label in names.items():
            if label.upper() not in ("A", "B", "C"):
                continue
            result[label] = math.log(0.9) if name == self.target else math.log(0.01)
        return result


def test_narrow_alphabet_scorer_incomplete_trials_are_not_scored_as_changes(probe):
    """A scorer too narrow for the letters a trial draws must never look like an answer change.

    With a 3-candidate pool the baseline always lands on A/B/C (fully
    readable), but ``letters`` redraws each candidate's letter from the
    full 52-letter alphabet, so most trials assign at least one candidate a
    letter outside A-C and become incomplete under the fix. Before the fix
    (no ``incomplete`` tracking) those trials would have been compared to
    the baseline anyway and inflated the answer-change rate.
    """
    pool = ("machine_status", "memory_stats", "gpu_stats")
    entries = [_entry("e1", "machine_status")]
    scorer_obj = _NarrowAlphabetScorer("machine_status")
    report = probe.run_probe(
        scorer_obj, _render, entries, pool=pool, per_entry=40, seed="narrow-alphabet"
    )
    kinds = {row["kind"]: row for row in report["kinds"]}
    letters = kinds["letters"]
    assert letters["incomplete"]["of"] == 40
    assert letters["incomplete"]["trials"] > 0
    # Every trial the scorer *could* fully read still agrees with the identity-like
    # baseline choice, so the scored (non-incomplete) rate is exactly 0 -- none of
    # the incomplete trials leaked into "changes".
    assert letters["trials"] + letters["incomplete"]["trials"] == 40
    assert letters["changes"] == 0


# ---------------------------------------------------------------------------
# Determinism: the same seed always draws the same perturbations
# ---------------------------------------------------------------------------


def test_same_seed_is_reproducible(probe):
    entries = [_entry("e1", "machine_status"), _entry("e2", "gpu_stats")]
    scorer_obj = _FirstListedScorer()
    report_a = probe.run_probe(scorer_obj, _render, entries, pool=POOL, per_entry=6, seed="fixed")
    report_b = probe.run_probe(scorer_obj, _render, entries, pool=POOL, per_entry=6, seed="fixed")
    assert report_a == report_b


def test_different_seed_can_change_the_draws(probe):
    baseline_order = list(POOL)
    baseline_labels = probe.scorer.labels_for(POOL)
    trial_a = probe.build_trial(
        "order",
        probe.derive_seed(1, "e1", "order", 0),
        POOL,
        baseline_order,
        baseline_labels,
        "machine_status",
        "machine_status",
        None,
    )
    trial_b = probe.build_trial(
        "order",
        probe.derive_seed(2, "e1", "order", 0),
        POOL,
        baseline_order,
        baseline_labels,
        "machine_status",
        "machine_status",
        None,
    )
    assert trial_a.order != trial_b.order or trial_a != trial_b


# ---------------------------------------------------------------------------
# Never canonicalises order: the scorer sees exactly the drawn order/labels
# ---------------------------------------------------------------------------


def test_scorer_sees_the_drawn_order_verbatim(probe):
    """The prompt the fake scorer is asked to score carries the perturbed order, not the default."""
    entries = [_entry("e1", "machine_status")]
    seen_prompts: list[str] = []

    class _Recording(_FirstListedScorer):
        def score_next_token(self, prompt: str, *, top: int = 20) -> dict[str, float]:
            seen_prompts.append(prompt)
            return super().score_next_token(prompt, top=top)

    probe.run_probe(_Recording(), _render, entries, pool=POOL, per_entry=5, seed="verbatim")
    # 1 baseline + 5 trials * 4 kinds (no paraphrases given) = 21 prompts.
    assert len(seen_prompts) == 1 + 5 * 4
    # At least one non-baseline prompt must differ in its first listed name
    # from the baseline's -- i.e. the raw drawn order reached the scorer,
    # nothing normalised it back to the default listing first.
    first_names = [_LINE_RE.search(p).group(2) for p in seen_prompts]
    assert len(set(first_names)) > 1


# ---------------------------------------------------------------------------
# full_alphabet_labels (issue 53, P1): the in-process scorer must read every
# letter a permutation trial can draw, not just the fixed A-R training map.
# ---------------------------------------------------------------------------


def test_full_alphabet_labels_covers_every_letter(probe):
    labels = probe.full_alphabet_labels()
    assert set(labels) == set(probe.scorer.LABEL_ALPHABET)
    assert all(name == letter for name, letter in labels.items())
    assert len(labels) == len(probe.scorer.LABEL_ALPHABET) == 52


class _VocabTokenizer:
    """A small fake tokenizer: each label is its own token, spaced and tabbed, plus junk."""

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


def _fake_torch_model(torch, size: int):
    """A model whose last-position logits are a fixed, deterministic ramp over *size*."""

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.fixed = torch.tensor([[float(index) for index in range(size)]])

        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            positions = input_ids.shape[1] if not logits_to_keep else logits_to_keep
            out = self.fixed.unsqueeze(1).expand(input_ids.shape[0], positions, -1)
            return type("Out", (), {"logits": out + self.anchor})()

    return _Model()


def test_build_real_scorer_passes_full_alphabet_only_for_in_process(probe, monkeypatch):
    """The CLI glue: in-process asks for every letter; served needs none (issue 53)."""
    measure = probe._sibling("measure")
    original_sibling = probe._sibling

    def fake_sibling(name):
        # _build_real_scorer calls _sibling("measure") itself, which by default reloads
        # the module from scratch -- pin it to the one instance this test patches.
        return measure if name == "measure" else original_sibling(name)

    monkeypatch.setattr(probe, "_sibling", fake_sibling)

    seen: list[tuple[object, object]] = []

    def fake_build_scorer(spec, labels=None):
        seen.append((spec, labels))
        return "handle"

    monkeypatch.setattr(measure.Seams, "load_config", lambda self, path: object())
    monkeypatch.setattr(measure.Seams, "detect_platform", lambda self: "fake-platform")
    # Seams.build_scorer's default lambda calls the *module-level* build_scorer by name,
    # resolved at call time -- patching that global is what actually reroutes it here.
    monkeypatch.setattr(measure, "build_scorer", fake_build_scorer)

    args = argparse.Namespace(
        model="scorer-b1",
        revision="main",
        tokenizer=None,
        config=None,
        scorer_kind=measure.SCORER_IN_PROCESS,
    )
    result = probe._build_real_scorer(args)
    assert result == "handle"
    assert len(seen) == 1
    _spec, labels = seen[0]
    assert labels == probe.full_alphabet_labels()

    seen.clear()
    args.scorer_kind = measure.SCORER_SERVED
    probe._build_real_scorer(args)
    _spec, labels = seen[0]
    assert labels is None


def test_full_alphabet_scorer_reads_letters_outside_the_default_a_to_r_map(probe):
    """The P1 repro: a scorer built for only A-R cannot read letters a trial draws outside it.

    ``scorer.labels_for(scorer.candidates())`` (the training default measure.build_scorer
    used to always pass) only spans as many letters as there are candidates -- A-R for
    today's 18. :func:`probe.full_alphabet_labels` must cover every letter instead, so a
    ``TransformersScorer`` built from it reads a trial's letters regardless of which one a
    permutation drew.
    """
    torch = pytest.importorskip("torch")
    scorer = probe.scorer

    default_labels = scorer.labels_for(scorer.candidates())
    assert "S" not in default_labels.values()  # confirms the bug's premise: A-R only

    full_labels = probe.full_alphabet_labels()
    tokenizer = _VocabTokenizer(full_labels.values())
    ids = scorer.label_token_ids(tokenizer, full_labels)
    model = _fake_torch_model(torch, len(tokenizer))
    in_process = scorer.TransformersScorer(model, tokenizer, full_labels, ids)

    logprobs = in_process.score_next_token("anything")
    seen = {text.strip() for text in logprobs}
    # Letters well outside the default A-R training map are read.
    for letter in ("S", "Z", "a", "z"):
        assert letter in seen, letter


# ---------------------------------------------------------------------------
# CLI: refuses test/held-out without --final
# ---------------------------------------------------------------------------


def _split_file(tmp_path: Path, name: str, header: str | None = None) -> Path:
    path = tmp_path / name
    payload = {
        "entries": [
            {
                "id": "e1",
                "kind": "explicit",
                "text": "show me the machine",
                "expect": {"operation": "machine_status"},
                "source": "test",
            }
        ]
    }
    if header is not None:
        payload["header"] = header
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_refuses_test_split_without_final(tmp_path, probe, capsys):
    path = _split_file(tmp_path, "test.json")
    exit_code = probe.main(["--split", str(path), "--model", "unused"])
    assert exit_code == 1
    assert "test" in capsys.readouterr().err


def test_cli_refuses_held_out_split_without_final(tmp_path, probe, capsys):
    path = _split_file(tmp_path, "held-out.json")
    exit_code = probe.main(["--split", str(path), "--model", "unused"])
    assert exit_code == 1
    assert "held-out" in capsys.readouterr().err


def test_cli_allows_val_split_without_final(tmp_path, probe, capsys, monkeypatch):
    """A plain val split is not test/held-out, so it proceeds past the split guard.

    No real model is wired up in tests, so it should fail later (no fake
    scorer path in the CLI) rather than on the split guard.
    """
    path = _split_file(tmp_path, "val.json")
    exit_code = probe.main(["--split", str(path)])
    assert exit_code == 1
    assert "--model" in capsys.readouterr().err


def test_cli_final_allows_test_split_to_reach_model_check(tmp_path, probe, capsys):
    path = _split_file(tmp_path, "test.json")
    exit_code = probe.main(["--split", str(path), "--final"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "looks like the test" not in err
    assert "--model" in err
