"""Tests for scripts/lfm-finetune/draft_sources.py (issue 53, t13).

A fake generator/reviewer caller replaces every model call -- no network,
no real model. Loaded the same way sibling lfm-finetune tests load a
same-directory script (see test_lfm_finetune_draft_heldout.py /
test_lfm_finetune_augment.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "draft_sources.py"


def _module():
    spec = importlib.util.spec_from_file_location("draft_sources", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ds = _module()
table = ds.table


def _role(name: str) -> "ds.aug.RoleConfig":
    return ds.aug.RoleConfig(role=name, url="http://example.invalid/v1/chat", model=f"{name}-model")


ROLES = {name: _role(name) for name in ds.ROLES}


# ---------------------------------------------------------------------------
# a scripted generator + reviewer caller
# ---------------------------------------------------------------------------


class FakeCaller:
    """Routes a call to a scripted generator/reviewer behaviour by role and
    by matching text embedded in the prompt -- never real HTTP."""

    def __init__(self, reject_reviewer_b_containing: str | None = None):
        self.calls: list[tuple[str, str]] = []
        self.reject_reviewer_b_containing = reject_reviewer_b_containing

    def __call__(self, role, system: str, user: str) -> str:  # noqa: D401 - callable
        self.calls.append((role.role, user))
        if role.role == "GENERATOR":
            return self._generate(user)
        if role.role == "REVIEWER_B" and self.reject_reviewer_b_containing:
            if self.reject_reviewer_b_containing in user:
                return "no, this is the wrong handling"
        return "yes, this is correct"

    def _generate(self, user: str) -> str:
        for op in table.OPERATIONS:
            if f"operation {op.name}." in user:
                args = {
                    a.name: (a.choices[0] if a.kind == "choice" else "vllm.service")
                    for a in op.args
                }
                item = {"text": f"please run {op.name} now", "args": args}
                return json.dumps([item])
        for reason, definition in ds.REASON_DEFINITIONS.items():
            if definition in user:
                return json.dumps([{"text": f"escalate scenario for {reason}"}])
        if "knowledge questions" in user:
            return json.dumps(
                [{"text": "What is unified memory?", "answer": "Memory shared between CPU/GPU."}]
            )
        return "[]"


# ---------------------------------------------------------------------------
# generator-side: argument validation
# ---------------------------------------------------------------------------


def test_op_candidates_validates_args_and_drops_invalid() -> None:
    def bad_caller(role, system, user):
        return json.dumps(
            [
                {"text": "check the vllm service status", "args": {"service": "vllm.service"}},
                {"text": "check some service status", "args": {}},  # missing required arg
            ]
        )

    rejects: dict[str, int] = {}
    out = ds._op_candidates(ROLES["GENERATOR"], 1, bad_caller, rejects)
    # only the operation with args (service_status) can be invalid here; every
    # no-arg operation's single call also runs, contributing valid candidates.
    service_status_candidates = [c for c in out if c.expect["operation"] == "service_status"]
    assert len(service_status_candidates) == 1
    assert rejects["invalid_args"] >= 1


def test_op_candidates_uses_table_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real_validate = table.validate

    def spy(name, args):
        calls.append((name, args))
        return real_validate(name, args)

    monkeypatch.setattr(table, "validate", spy)
    rejects: dict[str, int] = {}
    ds._op_candidates(ROLES["GENERATOR"], 1, FakeCaller(), rejects)
    assert calls  # nvsh.ops.table.validate was actually consulted


# ---------------------------------------------------------------------------
# reviewer: both must accept
# ---------------------------------------------------------------------------


def test_judge_requires_both_reviewers_to_accept() -> None:
    caller = FakeCaller(reject_reviewer_b_containing="please run machine_status now")
    outcome = ds.judge(
        "please run machine_status now",
        {"operation": "machine_status", "args": {}},
        None,
        ROLES,
        caller,
    )
    assert outcome["accepted"] is False
    assert outcome["votes"]["reviewer_a"]["accept"] is True
    assert outcome["votes"]["reviewer_b"]["accept"] is False


def test_judge_accepts_when_both_reviewers_say_yes() -> None:
    caller = FakeCaller()
    outcome = ds.judge(
        "please run machine_status now",
        {"operation": "machine_status", "args": {}},
        None,
        ROLES,
        caller,
    )
    assert outcome["accepted"] is True


def test_escalate_reviewer_prompt_names_the_reason() -> None:
    _, user = ds.reviewer_prompt("some text", {"escalate": True}, "decline:outside_table")
    assert ds.REASON_DEFINITIONS["outside_table"] in user


def test_explain_reviewer_prompt_carries_the_answer() -> None:
    _, user = ds.reviewer_prompt("what is X", {"explain": True, "answer": "X is Y."}, None)
    assert "X is Y." in user


# ---------------------------------------------------------------------------
# dedupe: exact + near-duplicate, against dev.json and within the draft
# ---------------------------------------------------------------------------


def test_dedupe_drops_exact_match_against_dev_json() -> None:
    candidates = [ds.Candidate("op-thermal_stats", "How hot is this machine?", {})]
    rejects: dict[str, int] = {}
    kept = ds.dedupe_candidates(candidates, ["How hot is this machine?"], rejects)
    assert kept == []
    assert rejects["dedupe_exact"] == 1


def test_dedupe_drops_near_duplicate_against_dev_json() -> None:
    dev_text = "please restart the vllm service on this machine right now"
    near = "please restart the vllm service on this box right now"
    candidates = [ds.Candidate("op-service_restart", near, {})]
    rejects: dict[str, int] = {}
    kept = ds.dedupe_candidates(candidates, [dev_text], rejects)
    assert kept == []
    assert rejects["dedupe_near"] == 1


def test_dedupe_drops_within_draft_duplicate() -> None:
    candidates = [
        ds.Candidate("op-gpu_stats", "Show me GPU usage please", {}),
        ds.Candidate("op-gpu_stats", "Show me GPU usage please", {}),
    ]
    rejects: dict[str, int] = {}
    kept = ds.dedupe_candidates(candidates, [], rejects)
    assert len(kept) == 1
    assert rejects["dedupe_exact"] == 1


def test_dedupe_keeps_genuinely_different_text() -> None:
    candidates = [ds.Candidate("explain", "What does CUDA mean?", {})]
    rejects: dict[str, int] = {}
    kept = ds.dedupe_candidates(candidates, ["How hot is this machine?"], rejects)
    assert len(kept) == 1
    assert rejects == {}


# ---------------------------------------------------------------------------
# ids: stable, v2-<pool>-<kind>-<n>
# ---------------------------------------------------------------------------


def test_ids_are_stable_across_identical_runs(tmp_path: Path) -> None:
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    kwargs = dict(
        pool="eval",
        seed=53,
        per_op=1,
        per_reason=1,
        explain=1,
        roles=ROLES,
        dev_texts=[],
    )
    result1 = ds.run_draft(out_dir=out1, caller=FakeCaller(), **kwargs)
    result2 = ds.run_draft(out_dir=out2, caller=FakeCaller(), **kwargs)
    doc1 = json.loads((out1 / "draft.json").read_text())
    doc2 = json.loads((out2 / "draft.json").read_text())
    assert [e["id"] for e in doc1["entries"]] == [e["id"] for e in doc2["entries"]]
    assert result1["sha256"] == result2["sha256"]


def test_id_shape_and_source_id(tmp_path: Path) -> None:
    ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=1,
        per_op=1,
        per_reason=0,
        explain=0,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
    )
    doc = json.loads((tmp_path / "draft.json").read_text())
    assert doc["entries"], "expected at least one operation entry"
    for entry in doc["entries"]:
        assert entry["id"].startswith("v2-eval-op-")
        assert entry["source_id"] == entry["id"]
        assert entry["kind"] == "explicit"


def test_decline_entries_carry_the_class(tmp_path: Path) -> None:
    ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=1,
        per_op=0,
        per_reason=1,
        explain=0,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
    )
    doc = json.loads((tmp_path / "draft.json").read_text())
    assert doc["entries"], "expected decline entries"
    for entry in doc["entries"]:
        assert entry["expect"] == {"escalate": True}
        assert entry["class"].startswith("decline:")
        assert entry["class"].removeprefix("decline:") in ds.REASONS


# ---------------------------------------------------------------------------
# header fields
# ---------------------------------------------------------------------------


def test_header_records_pool_seed_models_counts_hash(tmp_path: Path) -> None:
    ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=7,
        per_op=1,
        per_reason=1,
        explain=1,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
    )
    doc = json.loads((tmp_path / "draft.json").read_text())
    header = doc["header"]
    assert header["pool"] == "eval"
    assert header["seed"] == 7
    assert header["models"] == {name: cfg.model for name, cfg in ROLES.items()}
    assert "kept" in header["counts"]
    assert len(header["sha256"]) == 64


# ---------------------------------------------------------------------------
# counts-only stdout / files: never entry text
# ---------------------------------------------------------------------------


def test_run_draft_return_value_carries_no_entry_text(tmp_path: Path) -> None:
    result = ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=1,
        per_op=1,
        per_reason=1,
        explain=1,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
    )
    dumped = json.dumps(result)
    for banned in ("please run", "escalate scenario", "unified memory", "What is"):
        assert banned not in dumped


def test_main_dry_run_prints_counts_only(capsys: pytest.CaptureFixture[str]) -> None:
    rc = ds.main(["draft", "unused-out", "--pool", "eval", "--seed", "1", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert "text" not in json.dumps(payload)


def test_heldout_pool_review_jsonl_never_carries_text(tmp_path: Path) -> None:
    ds.run_draft(
        out_dir=tmp_path,
        pool="heldout",
        seed=1,
        per_op=1,
        per_reason=0,
        explain=0,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
    )
    lines = (tmp_path / "review.jsonl").read_text().splitlines()
    assert lines
    for line in lines:
        row = json.loads(line)
        assert "text" not in row
        assert set(row) >= {"id", "votes"}


def test_eval_pool_review_jsonl_keeps_text_for_the_lead(tmp_path: Path) -> None:
    ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=1,
        per_op=1,
        per_reason=0,
        explain=0,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
    )
    lines = (tmp_path / "review.jsonl").read_text().splitlines()
    assert lines
    assert any("text" in json.loads(line) for line in lines)


# ---------------------------------------------------------------------------
# review subcommand: post-hoc review of a draft_heldout.py-shaped file
# ---------------------------------------------------------------------------


def _write_heldout_style_draft(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"header": "drafted by X", "entries": entries}), encoding="utf-8")


def test_review_keeps_only_entries_both_reviewers_accept(tmp_path: Path) -> None:
    in_path = tmp_path / "draft.json"
    _write_heldout_style_draft(
        in_path,
        [
            {
                "id": "ho46-001",
                "kind": "explicit",
                "text": "please run machine_status now",
                "expect": {"operation": "machine_status", "args": {}},
                "source": "t17",
            },
            {
                "id": "ho46-002",
                "kind": "explicit",
                "text": "reformat the whole disk please",
                "expect": {"escalate": True},
                "source": "t17",
            },
        ],
    )
    caller = FakeCaller(reject_reviewer_b_containing="reformat the whole disk please")
    result = ds.run_review(
        in_path=in_path,
        out_dir=tmp_path / "out",
        caller=caller,
        roles={"REVIEWER_A": ROLES["REVIEWER_A"], "REVIEWER_B": ROLES["REVIEWER_B"]},
        dev_texts=[],
    )
    assert result["kept"] == 1
    doc = json.loads((tmp_path / "out" / "draft.json").read_text())
    assert [e["id"] for e in doc["entries"]] == ["ho46-001"]


def test_review_escalate_prompt_has_no_reason_clause_without_class() -> None:
    _, user = ds.reviewer_prompt("text", {"escalate": True}, None)
    assert "no listed operation can safely handle" in user
    for definition in ds.REASON_DEFINITIONS.values():
        assert definition not in user


def test_review_dedupes_against_dev_json(tmp_path: Path) -> None:
    in_path = tmp_path / "draft.json"
    _write_heldout_style_draft(
        in_path,
        [
            {
                "id": "ho46-001",
                "kind": "explicit",
                "text": "How hot is this machine?",
                "expect": {"operation": "thermal_stats", "args": {}},
                "source": "t17",
            }
        ],
    )
    result = ds.run_review(
        in_path=in_path,
        out_dir=tmp_path / "out",
        caller=FakeCaller(),
        roles={"REVIEWER_A": ROLES["REVIEWER_A"], "REVIEWER_B": ROLES["REVIEWER_B"]},
        dev_texts=["How hot is this machine?"],
    )
    assert result["kept"] == 0
    assert result["rejects"]["dedupe_exact"] == 1


def test_review_output_never_carries_text_in_review_jsonl(tmp_path: Path) -> None:
    in_path = tmp_path / "draft.json"
    _write_heldout_style_draft(
        in_path,
        [
            {
                "id": "ho46-001",
                "kind": "explicit",
                "text": "please run machine_status now",
                "expect": {"operation": "machine_status", "args": {}},
                "source": "t17",
            }
        ],
    )
    ds.run_review(
        in_path=in_path,
        out_dir=tmp_path / "out",
        caller=FakeCaller(),
        roles={"REVIEWER_A": ROLES["REVIEWER_A"], "REVIEWER_B": ROLES["REVIEWER_B"]},
        dev_texts=[],
    )
    lines = (tmp_path / "out" / "review.jsonl").read_text().splitlines()
    for line in lines:
        assert "text" not in json.loads(line)


def test_review_refuses_the_sealed_held_out_corpus(tmp_path: Path) -> None:
    sealed = tmp_path / "held-out.json"
    sealed.write_text(json.dumps({"header": "x", "entries": []}), encoding="utf-8")
    with pytest.raises(ds.HeldOutRefused):
        ds.run_review(in_path=sealed, out_dir=tmp_path / "out")


def test_main_review_refuses_the_sealed_held_out_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sealed = tmp_path / "held-out.json"
    sealed.write_text(json.dumps({"header": "x", "entries": []}), encoding="utf-8")
    rc = ds.main(["review", str(sealed), str(tmp_path / "out")])
    assert rc == 2
    assert "held-out" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# environment: NVSH_DRAFT_<ROLE>_* (augment.py's own scheme, different prefix)
# ---------------------------------------------------------------------------


def test_load_role_config_reads_the_nvsh_draft_prefix() -> None:
    env = {
        "NVSH_DRAFT_GENERATOR_URL": "http://example.invalid/v1/chat",
        "NVSH_DRAFT_GENERATOR_MODEL": "some-model",
        "NVSH_DRAFT_GENERATOR_TEMPERATURE": "0.4",
    }
    cfg = ds.load_role_config("GENERATOR", env)
    assert cfg.url == "http://example.invalid/v1/chat"
    assert cfg.model == "some-model"
    assert cfg.temperature == 0.4


def test_load_role_config_error_names_the_nvsh_draft_variable() -> None:
    with pytest.raises(ds.ConfigError) as excinfo:
        ds.load_role_config("GENERATOR", {})
    assert "NVSH_DRAFT_GENERATOR_URL" in str(excinfo.value)
    assert "NVSH_AUG_" not in str(excinfo.value)


def test_reason_definitions_cover_the_eight_classes() -> None:
    assert set(ds.REASONS) == {
        "outside_table",
        "repair",
        "diagnosis",
        "missing_argument",
        "not_a_request",
        "multi_step",
        "injection",
        "over_time",
    }


def test_decline_prompt_embeds_the_reason_definition() -> None:
    _, user = ds.decline_prompt("missing_argument", 3)
    assert ds.REASON_DEFINITIONS["missing_argument"] in user


def test_an_empty_reviewer_reply_is_asked_again_not_counted_as_a_no() -> None:
    replies = {"REVIEWER_A": ["", "", "yes"], "REVIEWER_B": ["yes"]}
    asked: list[str] = []

    def caller(role, system, user):
        name = "REVIEWER_A" if role is ROLES["REVIEWER_A"] else "REVIEWER_B"
        asked.append(name)
        return replies[name].pop(0)

    out = ds.judge("restart it", {"escalate": True}, "decline:missing_argument", ROLES, caller)
    assert out["accepted"] is True
    assert asked.count("REVIEWER_A") == 3


def test_a_reviewer_that_stays_empty_is_a_reject_after_the_retries() -> None:
    def caller(role, system, user):
        return "" if role is ROLES["REVIEWER_A"] else "yes"

    out = ds.judge("restart it", {"escalate": True}, "decline:missing_argument", ROLES, caller)
    assert out["accepted"] is False
    assert out["votes"]["reviewer_a"]["reason"] == "empty reply"


# ---------------------------------------------------------------------------
# per-call request seeding (issue 53 P2: random.seed(seed) alone never
# reached the generator/reviewer HTTP calls -- see call_seed/
# make_seeded_caller)
# ---------------------------------------------------------------------------


def test_call_seed_is_the_same_for_the_same_call_key_across_runs() -> None:
    # Two independent computations with the same (run_seed, role, prompt_key,
    # attempt) simulate two separate runs recording the same seed.
    a = ds.call_seed(53, "GENERATOR", "prompt-abc", 0)
    b = ds.call_seed(53, "GENERATOR", "prompt-abc", 0)
    assert a == b


def test_call_seed_differs_for_different_call_keys() -> None:
    base = ds.call_seed(53, "GENERATOR", "prompt-abc", 0)
    assert base != ds.call_seed(53, "REVIEWER_A", "prompt-abc", 0)  # different role
    assert base != ds.call_seed(53, "GENERATOR", "prompt-xyz", 0)  # different prompt
    assert base != ds.call_seed(53, "GENERATOR", "prompt-abc", 1)  # different attempt
    assert base != ds.call_seed(99, "GENERATOR", "prompt-abc", 0)  # different run seed


def test_make_seeded_caller_passes_the_same_seed_for_the_same_call_across_runs() -> None:
    """A fake raw caller (in place of real HTTP) observes the seed each call
    would carry; the same run seed on two independent caller instances --
    standing in for two separate runs -- must produce the same sequence of
    per-call seeds for the same sequence of calls."""

    def make_recorder():
        seen: list[int] = []

        def raw(role, system, user, seed_value):
            seen.append(seed_value)
            return "yes"

        return seen, raw

    seen1, raw1 = make_recorder()
    seen2, raw2 = make_recorder()
    caller1 = ds.make_seeded_caller(53, raw_caller=raw1)
    caller2 = ds.make_seeded_caller(53, raw_caller=raw2)

    caller1(ROLES["REVIEWER_A"], "system prompt", "please check the vllm service")
    caller1(ROLES["REVIEWER_B"], "system prompt", "please check the vllm service")
    caller2(ROLES["REVIEWER_A"], "system prompt", "please check the vllm service")
    caller2(ROLES["REVIEWER_B"], "system prompt", "please check the vllm service")

    assert seen1 == seen2
    assert seen1[0] != seen1[1]  # different roles -> different seeds


def test_make_seeded_caller_gives_repeated_calls_distinct_seeds() -> None:
    """A prompt asked twice in the same run (e.g. a parse-failure retry)
    gets a distinct, but still reproducible, seed per attempt."""
    seen: list[int] = []

    def raw(role, system, user, seed_value):
        seen.append(seed_value)
        return "[]"

    caller = ds.make_seeded_caller(53, raw_caller=raw)
    caller(ROLES["GENERATOR"], "system prompt", "same prompt text")
    caller(ROLES["GENERATOR"], "system prompt", "same prompt text")

    assert len(seen) == 2
    assert seen[0] != seen[1]
    # Reproducible: a fresh caller with the same run seed replays the same
    # per-attempt sequence.
    replay: list[int] = []

    def raw_replay(role, system, user, seed_value):
        replay.append(seed_value)
        return "[]"

    caller_replay = ds.make_seeded_caller(53, raw_caller=raw_replay)
    caller_replay(ROLES["GENERATOR"], "system prompt", "same prompt text")
    caller_replay(ROLES["GENERATOR"], "system prompt", "same prompt text")
    assert replay == seen


def test_run_draft_defaults_to_a_seeded_caller_when_none_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Left unspecified, run_draft's caller must be built from the run seed
    (not plain augment.default_caller), so real requests carry a per-call
    seed."""
    built_with_seed = []

    def fake_make_seeded_caller(run_seed, raw_caller=None):
        built_with_seed.append(run_seed)
        return FakeCaller()

    monkeypatch.setattr(ds, "make_seeded_caller", fake_make_seeded_caller)
    ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=53,
        per_op=1,
        per_reason=0,
        explain=0,
        roles=ROLES,
        dev_texts=[],
    )
    assert built_with_seed == [53]


def test_run_draft_records_the_sampling_header(tmp_path: Path) -> None:
    ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=1,
        per_op=1,
        per_reason=0,
        explain=0,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
    )
    doc = json.loads((tmp_path / "draft.json").read_text())
    sampling = doc["header"]["sampling"]
    assert sampling["per_call_seed"] is True
    assert isinstance(sampling["note"], str) and sampling["note"]


def test_post_chat_completion_seeded_sends_a_top_level_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real (non-fake) network path must add "seed" to the request body
    on top of augment.chat_payload's own fields, never replacing them."""
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(ds.urllib.request, "urlopen", fake_urlopen)
    role = ROLES["GENERATOR"]
    result = ds._post_chat_completion_seeded(role, "sys", "usr", 12345)
    assert result == "hi"
    assert captured["body"]["seed"] == 12345
    assert captured["body"]["model"] == role.model
    assert captured["body"]["messages"][1]["content"] == "usr"


def test_only_reasons_limits_the_decline_draft(tmp_path: Path) -> None:
    """``--only-reasons`` (issue 53 top-ups): only the named classes are drafted."""
    ds.run_draft(
        out_dir=tmp_path,
        pool="eval",
        seed=1,
        per_op=0,
        per_reason=1,
        explain=0,
        caller=FakeCaller(),
        roles=ROLES,
        dev_texts=[],
        only_reasons=["not_a_request", "missing_argument"],
    )
    doc = json.loads((tmp_path / "draft.json").read_text())
    classes = {entry["class"] for entry in doc["entries"]}
    assert classes and classes <= {"decline:not_a_request", "decline:missing_argument"}
    assert doc["header"]["reasons"] == ["missing_argument", "not_a_request"]


def test_only_reasons_refuses_an_unknown_reason() -> None:
    with pytest.raises(ValueError, match="unknown decline reason"):
        ds.check_reasons(["repair", "nonsense"])
    assert ds.check_reasons(None) == ds.REASONS


def test_only_reasons_cli_dry_run_counts_the_named_reasons(capsys) -> None:
    ds.main(
        [
            "draft",
            "out",
            "--pool",
            "eval",
            "--seed",
            "1",
            "--per-reason",
            "5",
            "--only-reasons",
            "repair,injection",
            "--dry-run",
        ]
    )
    assert json.loads(capsys.readouterr().out)["planned"]["reasons"] == 10


def test_reviewer_policy_hands_off_small_talk_not_answers_it() -> None:
    """dev.json labels small talk ``decline:not_a_request`` -> escalate; the
    reviewer must be told the same policy or it rejects every such entry."""
    system = ds.REVIEWER_SYSTEM
    assert "only for technical knowledge questions" in system
    assert "greetings, thanks" in system and "handed off, not answered" in system


def test_missing_argument_definition_requires_an_offered_action() -> None:
    definition = ds.REASON_DEFINITIONS["missing_argument"]
    assert "table DOES offer" in definition
    assert "does not offer is not this reason" in definition
