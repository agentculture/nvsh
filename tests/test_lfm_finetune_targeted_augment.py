"""Tests for scripts/lfm-finetune/targeted_augment.py (issue 53, t15).

A fake generator/reviewer caller replaces every model call -- no network,
no real model. Loaded the same way sibling lfm-finetune tests load a
same-directory script (see test_lfm_finetune_draft_sources.py).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DIR = _ROOT / "scripts" / "lfm-finetune"
_WORLD = json.loads((_ROOT / "nvsh" / "tiers" / "corpus" / "dev.json").read_text())["world"]


def _load(name: str, module_name: str | None = None):
    spec = importlib.util.spec_from_file_location(module_name or name, _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ta = _load("targeted_augment")
table = ta.table


def _role(name: str):
    return ta.aug.RoleConfig(role=name, url="http://example.invalid/v1/chat", model=f"{name}-model")


ROLES = {name: _role(name) for name in ("GENERATOR", "REVIEWER_A", "REVIEWER_B")}

_TRAIN_ENTRIES = [
    {
        "id": "g170",
        "source_id": "g170",
        "kind": "explicit",
        "text": "restart nginx",
        "expect": {"operation": "service_restart", "args": {"service": "nginx.service"}},
        "class": "terse",
    },
    {
        "id": "g168",
        "source_id": "g168",
        "kind": "explicit",
        "text": "the nginx server is hung, restart it",
        "expect": {"operation": "service_restart", "args": {"service": "nginx.service"}},
        "class": "symptom",
    },
    {
        "id": "g137",
        "source_id": "g137",
        "kind": "explicit",
        "text": "Is vllm.service running?",
        "expect": {"operation": "service_status", "args": {"service": "vllm.service"}},
        "class": "question",
    },
    {
        "id": "g174",
        "source_id": "g174",
        "kind": "explicit",
        "text": "Restart the trainer container",
        "expect": {"operation": "container_restart", "args": {"container": "trainer"}},
        "class": "imperative",
    },
    {
        "id": "g193",
        "source_id": "g193",
        "kind": "explicit",
        "text": "Can you switch to balanced mode?",
        "expect": {"operation": "power_set", "args": {"mode": "balanced"}},
        "class": "question",
    },
    {
        "id": "g189",
        "source_id": "g189",
        "kind": "explicit",
        "text": "max perf",
        "expect": {"operation": "power_set", "args": {"mode": "max_performance"}},
        "class": "terse",
    },
    {
        "id": "g010",
        "source_id": "g010",
        "kind": "explicit",
        "text": "How busy is the GPU?",
        "expect": {"operation": "gpu_stats", "args": {}},
        "class": "question",
    },
    {
        "id": "f01",
        "source_id": "f01",
        "kind": "failure",
        "text": "systemctl restart vllm -> failed",
        "expect": {"operation": "service_status", "args": {"service": "vllm.service"}},
    },
]


def _train_file(tmp_path: Path, entries: list[dict] | None = None, name: str = "train.json"):
    path = tmp_path / name
    payload = {
        "header": "Split 'train' of corpus-v2 (seed=53). Corpus v2, merged by split.py.",
        "world": _WORLD,
        "entries": entries if entries is not None else _TRAIN_ENTRIES,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# a scripted generator + reviewer caller
# ---------------------------------------------------------------------------


def _asks_for(user: str, name: str) -> bool:
    """True when a generator prompt asks about operation *name*."""
    return re.search(rf"the operation {name}\b", user) is not None


def _choice_arg(op):
    return next(a for a in op.args if a.kind == "choice")


class FakeCaller:
    """Scripted generator/reviewer behaviour, routed by role and by phrases the
    script's own prompts carry -- never real HTTP."""

    def __init__(self, reject_b_containing: str | None = None, generator=None):
        self.calls: list[tuple[str, str]] = []
        self.reject_b_containing = reject_b_containing
        self.generator = generator

    def __call__(self, role, system: str, user: str) -> str:
        self.calls.append((role.role, user))
        if role.role == "GENERATOR":
            if self.generator is not None:
                return self.generator(user)
            return self._generate(user)
        if self.reject_b_containing and self.reject_b_containing in user:
            return "no, that is wrong"
        return "yes, correct"

    @staticmethod
    def _generate(user: str) -> str:
        if ta.DX_MARKER in user:
            return json.dumps(
                [
                    {
                        "topic": "Xid 79",
                        "explain": "What does Xid 79 mean in the kernel log?",
                        "answer": "It means the GPU fell off the bus.",
                        "diagnose": "Why do I keep getting Xid 79 on this box?",
                    }
                ]
            )
        if ta.CHOICE_MARKER in user:
            for op in table.OPERATIONS:
                for arg in op.args:
                    if arg.kind != "choice":
                        continue
                    for value in arg.choices:
                        if f"to {value!r}" in user:
                            text = f"please go to {value.replace('_', ' ')} now"
                            return json.dumps([{"text": text, "args": {arg.name: value}}])
            return "[]"
        if ta.DISAMBIGUATION_MARKER in user:
            if _asks_for(user, "service_status"):
                return json.dumps(
                    [
                        {
                            "text": "docker status",
                            "args": {"service": "docker.service"},
                            "confusable": "container_list",
                        }
                    ]
                )
            return "[]"
        if ta.HARD_NEGATIVE_MARKER in user:
            if _asks_for(user, "thermal_stats"):
                return json.dumps(
                    [
                        {
                            "text": "what does throttling mean when the board gets hot?",
                            "answer": "The clocks drop to keep the chip under its limit.",
                        }
                    ]
                )
            return "[]"
        return "[]"


# ---------------------------------------------------------------------------
# missing-argument: rule-based stripping
# ---------------------------------------------------------------------------


def _op(name: str):
    op = table.get(name)
    assert op is not None
    return op


def test_strip_removes_the_argument_value() -> None:
    arg = _op("service_restart").args[0]
    for pick in range(5):
        stripped, reason = ta.strip_argument(
            "the nginx server is hung, restart it", arg, "nginx.service", pick
        )
        assert reason == ""
        assert stripped is not None
        assert "nginx" not in stripped.lower()


def test_strip_finds_a_dotted_service_name_and_its_bare_stem() -> None:
    arg = _op("service_status").args[0]
    stripped, _ = ta.strip_argument("Is vllm.service running?", arg, "vllm.service", 0)
    assert stripped is not None and "vllm" not in stripped.lower()
    stripped, _ = ta.strip_argument("restart docker", arg, "docker.service", 0)
    assert stripped is not None and "docker" not in stripped.lower()


def test_strip_handles_a_choice_value_with_underscores() -> None:
    arg = _choice_arg(_op("power_set"))
    value = arg.choices[0]
    text = f"Set the power mode to {value.replace('_', ' ')}"
    stripped, _ = ta.strip_argument(text, arg, value, 0)
    assert stripped is not None
    assert value.replace("_", " ") not in stripped.lower()


def test_strip_reports_no_span_when_the_value_is_not_in_the_text() -> None:
    arg = _choice_arg(_op("power_set"))
    stripped, reason = ta.strip_argument("max perf", arg, "max_performance", 0)
    assert stripped is None
    assert reason == "no_argument_span"


def test_strip_rejects_a_text_left_empty() -> None:
    arg = _op("service_restart").args[0]
    stripped, reason = ta.strip_argument("nginx", arg, "nginx.service", 2)
    assert stripped is None
    assert reason in {"too_short", "no_argument_span"}


def test_missing_argument_units_are_deterministic_under_a_seed() -> None:
    first = ta.missing_argument_units(_TRAIN_ENTRIES, per_op=5, seed=7, rejects={})
    second = ta.missing_argument_units(_TRAIN_ENTRIES, per_op=5, seed=7, rejects={})
    assert [u.items[0].text for u in first] == [u.items[0].text for u in second]
    assert [u.items[0].extra for u in first] == [u.items[0].extra for u in second]
    assert first  # something stripped


def test_missing_argument_units_link_their_original_and_escalate() -> None:
    rejects: dict[str, int] = {}
    units = ta.missing_argument_units(_TRAIN_ENTRIES, per_op=5, seed=1, rejects=rejects)
    originals = {e["id"]: e for e in _TRAIN_ENTRIES}
    for unit in units:
        (item,) = unit.items
        assert item.expect == {"escalate": True}
        assert item.cls == "decline:missing_argument"
        original = originals[item.extra["pair_of"]]
        assert unit.group == original["source_id"]
        assert original["kind"] == "explicit"  # a failure entry is never stripped
        value = next(iter(original["expect"]["args"].values()))
        assert value.split(".")[0].replace("_", " ") not in item.text.lower()
    # "max perf" names no choice value verbatim; a no-arg operation is never eligible
    assert rejects.get("no_argument_span", 0) >= 1
    assert "g010" not in {u.items[0].extra["pair_of"] for u in units}
    assert "f01" not in {u.items[0].extra["pair_of"] for u in units}


def test_missing_argument_caps_per_operation() -> None:
    units = ta.missing_argument_units(_TRAIN_ENTRIES, per_op=1, seed=3, rejects={})
    ops = [
        next(e for e in _TRAIN_ENTRIES if e["id"] == u.items[0].extra["pair_of"])["expect"][
            "operation"
        ]
        for u in units
    ]
    assert len(ops) == len(set(ops))


# ---------------------------------------------------------------------------
# the end-to-end run, one recipe at a time
# ---------------------------------------------------------------------------


def _run(tmp_path: Path, recipes, caller=None, per_recipe=1, **kwargs):
    out = tmp_path / "supplement.json"
    summary = ta.run(
        train_path=_train_file(tmp_path),
        out=out,
        recipes=tuple(recipes),
        per_recipe=per_recipe,
        seed=53,
        roles=ROLES,
        caller=caller or FakeCaller(),
        **kwargs,
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    return summary, doc


def test_missing_argument_recipe_writes_escalate_entries_with_pair_links(tmp_path) -> None:
    caller = FakeCaller()
    summary, doc = _run(tmp_path, ["missing-argument"], caller=caller, per_recipe=5)
    entries = doc["entries"]
    assert entries
    for entry in entries:
        assert entry["id"].startswith("t15-marg-")
        assert entry["source"] == "t15-missing-argument"
        assert entry["expect"] == {"escalate": True}
        assert entry["class"] == "decline:missing_argument"
        assert entry["source_id"] == entry["pair_of"]  # the original's own source_id here
    assert summary["recipes"]["missing-argument"]["kept"] == len(entries)
    # only reviewers are called for the rule-based recipe; never the generator
    assert {role for role, _ in caller.calls} == {"REVIEWER_B"}
    asked = [user for _, user in caller.calls]
    assert any(ta.MARG_UNSPECIFIED_MARKER in u for u in asked)
    assert any(ta.MARG_NATURAL_MARKER in u for u in asked)


def test_a_reviewer_b_rejection_drops_the_item(tmp_path) -> None:
    caller = FakeCaller(reject_b_containing=ta.MARG_NATURAL_MARKER)
    summary, doc = _run(tmp_path, ["missing-argument"], caller=caller, per_recipe=5)
    assert doc["entries"] == []
    counts = summary["recipes"]["missing-argument"]
    assert counts["kept"] == 0
    assert counts["rejected"]["reviewer_b"] >= 1


def test_decide_by_both_also_asks_reviewer_a(tmp_path) -> None:
    caller = FakeCaller()
    _run(tmp_path, ["missing-argument"], caller=caller, per_recipe=5, decide_by="both")
    assert {role for role, _ in caller.calls} == {"REVIEWER_A", "REVIEWER_B"}


def test_diagnosis_explain_pairs_share_a_source_id(tmp_path) -> None:
    summary, doc = _run(tmp_path, ["diagnosis-explain"])
    entries = doc["entries"]
    assert len(entries) == 2
    explain, diagnose = entries
    assert explain["expect"] == {"explain": True, "answer": "It means the GPU fell off the bus."}
    assert explain["class"] == "explain:question"
    assert diagnose["expect"] == {"escalate": True}
    assert diagnose["class"] == "decline:diagnosis"
    assert explain["source_id"] == diagnose["source_id"]
    assert explain["source_id"].startswith("t15-dx-p")
    assert {e["source"] for e in entries} == {"t15-diagnosis-explain"}
    assert summary["recipes"]["diagnosis-explain"]["kept"] == 2


def test_a_rejected_half_drops_the_whole_pair(tmp_path) -> None:
    caller = FakeCaller(reject_b_containing="Why do I keep getting Xid 79")
    summary, doc = _run(tmp_path, ["diagnosis-explain"], caller=caller)
    assert doc["entries"] == []
    assert summary["recipes"]["diagnosis-explain"]["rejected"]["reviewer_b"] == 1


def test_power_set_recipe_covers_every_choice_from_the_table(tmp_path) -> None:
    summary, doc = _run(tmp_path, ["power-set"])
    targets = ta.choice_targets()
    assert targets  # the table has at least one operation with a choice argument
    got = {
        (e["expect"]["operation"], json.dumps(e["expect"]["args"], sort_keys=True))
        for e in doc["entries"]
    }
    want = {(op.name, json.dumps({arg.name: value})) for op, arg, value in targets}
    assert got == want
    for entry in doc["entries"]:
        assert table.validate(entry["expect"]["operation"], entry["expect"]["args"]) is None
        assert entry["id"].startswith("t15-pset-")


def test_power_set_rejects_invalid_or_off_target_args(tmp_path) -> None:
    def generator(user: str) -> str:
        _, arg, value = next(t for t in ta.choice_targets() if f"to {t[2]!r}" in user)
        other = next(c for c in arg.choices if c != value)
        return json.dumps(
            [
                {"text": "set it to turbo", "args": {arg.name: "turbo"}},  # not a choice
                {"text": "go to the other one", "args": {arg.name: other}},  # off target
                {"text": "go there", "args": {}},  # missing
            ]
        )

    summary, doc = _run(
        tmp_path, ["power-set"], caller=FakeCaller(generator=generator), per_recipe=3
    )
    assert doc["entries"] == []
    rejected = summary["recipes"]["power-set"]["rejected"]
    assert rejected["invalid_args"] >= 2 * len(ta.choice_targets())
    assert rejected["wrong_value"] == len(ta.choice_targets())


def test_disambiguation_keeps_a_validated_gold_with_its_confusable(tmp_path) -> None:
    caller = FakeCaller()
    summary, doc = _run(tmp_path, ["disambiguation"], caller=caller)
    (entry,) = doc["entries"]
    assert entry["expect"] == {"operation": "service_status", "args": {"service": "docker.service"}}
    assert entry["confusable"] == "container_list"
    assert entry["id"].startswith("t15-disamb-")
    assert any(ta.MOST_NATURAL_MARKER in user for role, user in caller.calls if role != "GENERATOR")


def test_disambiguation_rejects_invalid_args_and_a_bad_confusable(tmp_path) -> None:
    def generator(user: str) -> str:
        if not _asks_for(user, "service_status"):
            return "[]"
        return json.dumps(
            [
                {"text": "docker status", "args": {}, "confusable": "container_list"},
                {"text": "docker state", "args": {"service": "docker.service"}, "confusable": "x"},
                {
                    "text": "docker health",
                    "args": {"service": "docker.service"},
                    "confusable": "service_status",
                },
            ]
        )

    summary, doc = _run(
        tmp_path, ["disambiguation"], caller=FakeCaller(generator=generator), per_recipe=3
    )
    assert doc["entries"] == []
    rejected = summary["recipes"]["disambiguation"]["rejected"]
    assert rejected["invalid_args"] == 1
    assert rejected["invalid_confusable"] == 2


def test_hard_negative_is_an_explain_entry(tmp_path) -> None:
    _, doc = _run(tmp_path, ["hard-negative"])
    (entry,) = doc["entries"]
    assert entry["expect"]["explain"] is True
    assert entry["expect"]["answer"]
    assert entry["class"] == "explain:question"
    assert entry["id"].startswith("t15-hneg-")


def test_a_text_naming_an_internal_operation_is_rejected(tmp_path) -> None:
    def generator(user: str) -> str:
        if not _asks_for(user, "thermal_stats"):
            return "[]"
        return json.dumps([{"text": "what does thermal_stats show?", "answer": "Temperatures."}])

    summary, doc = _run(tmp_path, ["hard-negative"], caller=FakeCaller(generator=generator))
    assert doc["entries"] == []
    assert summary["recipes"]["hard-negative"]["rejected"]["identifier"] == 1


def test_exclude_drops_a_repeat_of_a_protected_text(tmp_path) -> None:
    protected = tmp_path / "val.json"
    protected.write_text(
        json.dumps(
            {
                "header": "Split 'val' of corpus-v2 (seed=53).",
                "entries": [
                    {
                        "id": "v1",
                        "kind": "explicit",
                        "text": "What does Xid 79 mean in the kernel log?",
                        "expect": {"explain": True, "answer": "x"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary, doc = _run(tmp_path, ["diagnosis-explain"], exclude=[protected])
    assert doc["entries"] == []
    assert summary["recipes"]["diagnosis-explain"]["rejected"]["protected_exact"] == 1


def test_a_repeat_of_a_train_text_is_dropped(tmp_path) -> None:
    def generator(user: str) -> str:
        if not _asks_for(user, "thermal_stats"):
            return "[]"
        return json.dumps([{"text": "How busy is the GPU?", "answer": "Look at utilisation."}])

    summary, doc = _run(tmp_path, ["hard-negative"], caller=FakeCaller(generator=generator))
    assert doc["entries"] == []
    assert summary["recipes"]["hard-negative"]["rejected"]["train_exact"] == 1


def test_a_generator_failure_is_counted_not_raised(tmp_path) -> None:
    def generator(user: str) -> str:
        raise ValueError("boom")

    summary, _ = _run(tmp_path, ["hard-negative"], caller=FakeCaller(generator=generator))
    assert summary["recipes"]["hard-negative"]["rejected"]["error"] == len(table.OPERATIONS)


def test_the_held_out_file_is_refused_as_train(tmp_path) -> None:
    path = _train_file(tmp_path, name="held-out.json")
    with pytest.raises(ValueError, match="held-out"):
        ta.load_train(path)


def test_a_non_train_side_is_refused(tmp_path) -> None:
    path = tmp_path / "val.json"
    path.write_text(
        json.dumps({"header": "Split 'val' of corpus-v2 (seed=53).", "entries": _TRAIN_ENTRIES}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train side"):
        ta.load_train(path)


def test_unknown_recipe_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="recipe"):
        _run(tmp_path, ["nope"])


# ---------------------------------------------------------------------------
# the output feeds the existing merge -> train pipeline
# ---------------------------------------------------------------------------


def test_output_is_accepted_by_merge_and_loads_as_train(tmp_path) -> None:
    _, doc = _run(
        tmp_path,
        ["missing-argument", "diagnosis-explain", "power-set", "disambiguation", "hard-negative"],
        per_recipe=2,
    )
    assert doc["entries"]
    mv = _load("merge_variations")
    split = json.loads(_train_file(tmp_path).read_text(encoding="utf-8"))
    merged, added = mv.add_supplement(split, doc)
    assert added == len(doc["entries"])
    by_id = {e["id"]: e for e in merged["entries"]}
    for entry in doc["entries"]:
        assert by_id[entry["id"]]["source_id"] == entry["source_id"]  # pairs stay grouped
    merged_path = tmp_path / "train-merged.json"
    merged_path.write_text(json.dumps(merged), encoding="utf-8")

    from nvsh.tiers.bench import load_corpus

    assert load_corpus(merged_path).problems == ()

    ts = _load("train_scorer", "lfm_train_scorer_t15")
    examples = ts.read_split(merged_path, "train")
    assert len(examples) == len(merged["entries"])

    bd = _load("build_dataset", "lfm_build_dataset_t15")
    out = tmp_path / "train.jsonl"
    assert bd.main(["--split", str(merged_path), "--out", str(out), "--no-verify-render"]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(merged["entries"])
    out_r = tmp_path / "train-r.jsonl"
    assert (
        bd.main(
            ["--split", str(merged_path), "--out", str(out_r), "--no-verify-render", "--reasons"]
        )
        == 0
    )


def test_output_carries_counts_and_sha256(tmp_path) -> None:
    summary, doc = _run(tmp_path, ["diagnosis-explain"])
    meta = doc["t15"]
    assert meta["sha256"] == summary["sha256"]
    assert meta["counts"] == summary["recipes"]
    assert meta["seed"] == 53
    assert "Split 'train' of " in doc["header"]
    for word in ("test", "held-out", "val"):
        assert word not in doc["header"].lower().split()


# ---------------------------------------------------------------------------
# CLI: counts and hashes only, dry-run never calls
# ---------------------------------------------------------------------------


def test_main_prints_no_entry_text(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(ta, "_make_caller", lambda seed: FakeCaller())
    monkeypatch.setattr(ta, "_load_roles", lambda needed, source: {n: ROLES[n] for n in needed})
    out = tmp_path / "sup.json"
    code = ta.main(
        [
            "--train",
            str(_train_file(tmp_path)),
            "--out",
            str(out),
            "--recipes",
            "missing-argument,diagnosis-explain,hard-negative",
            "--per-recipe",
            "2",
            "--seed",
            "5",
        ]
    )
    assert code == 0
    printed = capsys.readouterr()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["entries"]
    for entry in doc["entries"]:
        assert entry["text"] not in printed.out
        assert entry["text"] not in printed.err
    result = json.loads(printed.out)
    assert result["sha256"] == doc["t15"]["sha256"]


def test_dry_run_makes_no_calls_and_needs_no_roles(tmp_path, monkeypatch, capsys) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("dry-run must not build a caller or load roles")

    monkeypatch.setattr(ta, "_make_caller", refuse)
    monkeypatch.setattr(ta, "_load_roles", refuse)
    out = tmp_path / "sup.json"
    code = ta.main(
        [
            "--train",
            str(_train_file(tmp_path)),
            "--out",
            str(out),
            "--per-recipe",
            "3",
            "--dry-run",
        ]
    )
    assert code == 0
    assert not out.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert set(plan["recipes"]) == set(ta.RECIPES)
    assert plan["recipes"]["missing-argument"]["teacher_calls"] == 0
    assert plan["recipes"]["missing-argument"]["candidates"] >= 1
    assert plan["recipes"]["power-set"]["teacher_calls"] == len(ta.choice_targets())


def test_main_reports_a_missing_role_variable(tmp_path, monkeypatch, capsys) -> None:
    for key in list(__import__("os").environ):
        if key.startswith("NVSH_AUG_") or key.startswith("NVSH_DRAFT_"):
            monkeypatch.delenv(key)
    code = ta.main(["--train", str(_train_file(tmp_path)), "--out", str(tmp_path / "o.json")])
    assert code == 2
    assert "NVSH_AUG_" in capsys.readouterr().err


def test_naturalness_verdict_allows_user_describing_modals() -> None:
    """t15 run: every 'is this natural?' yes carried 'a real user might ask it'
    or 'though it depends on context' and was parsed as a hedge (56 of 69 lost)."""
    item = ta.Item(
        text="Is that service up?",
        expect={"escalate": True},
        cls="decline:missing_argument",
        reviews=[
            ta.marg_unspecified_prompt("Is that service up?", "service"),
            ta.marg_natural_prompt("Is that service up?"),
        ],
    )
    unit = ta.Unit(recipe="missing-argument", items=[item])

    def caller(role, system, user):
        if ta.MARG_NATURAL_MARKER in user:
            return "yes, a real user might ask it, though 'that service' depends on context."
        return "yes, 'that service' does not name a specific service."

    ok, _ = ta.review_unit(unit, ROLES, caller, "reviewer_b")
    assert ok

    def hedged(role, system, user):
        return "yes, but it is not really natural."

    ok, _ = ta.review_unit(unit, ROLES, hedged, "reviewer_b")
    assert not ok


def test_missing_argument_verdicts_allow_a_mid_sentence_but() -> None:
    """t15 rerun (lapse l8): 20 of 37 reviewer-B rejects were clear yeses whose
    reason used 'but' ("says 'that service' but does not name it")."""
    item = ta.Item(
        text="Set the power mode to another mode",
        expect={"escalate": True},
        cls="decline:missing_argument",
        reviews=[
            ta.marg_unspecified_prompt("Set the power mode to another mode", "mode"),
            ta.marg_natural_prompt("Set the power mode to another mode"),
        ],
    )
    unit = ta.Unit(recipe="missing-argument", items=[item])

    def caller(role, system, user):
        if ta.MARG_NATURAL_MARKER in user:
            return "yes, it is a terse but natural terminal request."
        return "yes, the request says 'another mode' but does not name which mode is meant."

    ok, _ = ta.review_unit(unit, ROLES, caller, "reviewer_b")
    assert ok
    ok, _ = ta.review_unit(unit, ROLES, lambda r, s, u: "yes, but the mode is named.", "reviewer_b")
    assert not ok


def test_a_mid_sentence_but_still_hedges_outside_missing_argument() -> None:
    item = ta.Item(
        text="restart nginx",
        expect={"operation": "service_restart", "args": {"service": "nginx"}},
        cls=None,
        reviews=[ta.ds.reviewer_prompt("restart nginx", {"operation": "service_restart"})],
    )
    unit = ta.Unit(recipe="disambiguation", items=[item])
    reply = "yes, service_restart fits but container_restart fits as well"
    ok, _ = ta.review_unit(unit, ROLES, lambda r, s, u: reply, "reviewer_b")
    assert not ok


def test_user_modals_still_hedge_outside_the_naturalness_question() -> None:
    item = ta.Item(
        text="restart nginx",
        expect={"operation": "service_restart", "args": {"service": "nginx"}},
        cls=None,
        reviews=[ta.ds.reviewer_prompt("restart nginx", {"operation": "service_restart"})],
    )
    unit = ta.Unit(recipe="disambiguation", items=[item])
    ok, _ = ta.review_unit(unit, ROLES, lambda r, s, u: "yes, though it might be docker", "both")
    assert not ok


def test_hard_negative_prompt_forbids_writing_the_identifier() -> None:
    """t15 run: 104 of 129 hard negatives named the identifier and were dropped."""
    _, user = ta.hard_negative_prompt("gpu_stats", 8)
    assert "Never write gpu_stats or any other operation identifier" in user


def test_diagnosis_explain_prompt_forbids_writing_an_identifier() -> None:
    """t15 run: 10 diagnosis-explain candidates named an operation identifier
    and were dropped by the guard (same cause as D30's hard negatives)."""
    _, user = ta.dx_prompt(4)
    assert "Never write an operation identifier" in user


# ---------------------------------------------------------------------------
# check-then-change (issue 53, deviation d7)
# ---------------------------------------------------------------------------


def _ctc_generator(user: str) -> str:
    if ta.CTC_MARKER not in user or not _asks_for(user, "power_set"):
        return "[]"
    return json.dumps(
        [
            {
                "check": "How hot is the board right now?",
                "operation": "thermal_stats",
                "args": {},
                "conditional": "check the board temperature and if it's over 80C go to low power",
            },
            {  # the check half must be read-only
                "check": "switch to balanced",
                "operation": "power_set",
                "args": {"mode": "balanced"},
                "conditional": "check swap and if it's full switch to balanced",
            },
            {  # the check half must validate against the table
                "check": "status of it",
                "operation": "service_status",
                "args": {},
                "conditional": "check the web service and restart it if it's down",
            },
        ]
    )


def test_check_then_change_pairs_a_read_only_check_with_an_escalation(tmp_path) -> None:
    """t18 (d7): r3 proposed power_set on check-then-change requests whose gold
    is escalate; each pair teaches the check alone vs the conditional change."""
    summary, doc = _run(
        tmp_path, ["check-then-change"], caller=FakeCaller(generator=_ctc_generator), per_recipe=3
    )
    entries = doc["entries"]
    assert len(entries) == 2
    check, conditional = entries
    assert check["expect"] == {"operation": "thermal_stats", "args": {}}
    assert conditional["expect"] == {"escalate": True}
    assert conditional["class"] == "decline:multi_step"
    assert check["source_id"] == conditional["source_id"]
    assert check["source_id"].startswith("t15-ctc-p")
    assert {e["source"] for e in entries} == {"t15-check-then-change"}
    rejected = summary["recipes"]["check-then-change"]["rejected"]
    assert rejected["invalid_item"] == 2


def test_check_then_change_prompt_names_the_change_but_forbids_identifiers() -> None:
    _, user = ta.check_then_change_prompt("power_set", 3)
    assert ta.CTC_MARKER in user
    assert "the operation power_set" in user
    assert "Never write an operation identifier" in user
    # t18 pilot: the generator split check and conditional into two objects
    assert "ONE object per pair" in user


def test_check_then_change_asks_once_per_mutating_operation() -> None:
    mutating = [op for op in table.OPERATIONS if not op.read_only]
    assert mutating
    assert ta.planned_teacher_calls("check-then-change", 3) == len(mutating)
    assert ta.planned_candidates("check-then-change", 3) == 3 * len(mutating) * 2
