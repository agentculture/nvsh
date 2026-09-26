"""Tests for ``evals/tool_jev/judge.py`` (task t16, issue #64).

Every fixture here is synthetic: invented shell requests and explain texts
written for this file, never corpus or held-out text. No test touches the
network: :func:`_no_network` makes any socket connect raise, so a DeepEval
code path that tried to reach a provider (or Confident AI) fails loudly.

Acceptance criteria (plan task t16):

1. pass 2 makes zero network calls and its prompts equal pass 1's byte for
   byte -- ``test_pass2_prompts_equal_pass1_byte_for_byte`` and
   ``test_pass2_makes_no_network_call``.
2. no judge prompt contains a subject's model name or provider, and a
   judge's score of its own answer is excluded from that subject's panel
   score -- ``test_no_judge_prompt_names_a_subject``,
   ``test_self_score_is_kept_apart_from_panel_score``.
3. a metric that needs a second, dependent call is refused with a clear
   message (guards parked v6) -- the ``test_dependent_call_*`` tests.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

import evals.tool_jev  # noqa: F401  (env guard: must precede any deepeval import)
from evals.tool_jev import judge
from evals.tool_jev import report as report_module
from evals.tool_jev.ledger import CallSpec, Ledger, ledger_key, prompt_hash

# The operator's judge panel (from the spec), as (provider, model) pairs.
PANEL = (
    judge.JudgeId("anthropic", "claude-opus-5-5"),
    judge.JudgeId("openai", "gpt-6-sol"),
    judge.JudgeId("nvidia", "moonshotai/kimi-k3"),
    judge.JudgeId("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
    judge.JudgeId("openrouter", "qwen/qwen3.8-max-0902"),
)

# Two extra synthetic references that are subjects but not judges.
OTHER_REFS = (("openrouter", "mistral-giant-9"), ("openai", "gpt-6-mini"))

REQUESTS = {
    "c1": "docker ps failed with: permission denied while trying to connect to the socket",
    "c2": "df -h shows / at 100% and apt install stopped with no space left on device",
}


def _explain(i: int, case_id: str) -> str:
    # Distinct per subject, and a few that name their author, to prove scrubbing.
    return f"Answer {i} for {case_id}: the user is not in the docker group; add it and re-login."


def _answers() -> list[judge.ExplainAnswer]:
    out: list[judge.ExplainAnswer] = []
    subjects = [(j.provider, j.model) for j in PANEL] + list(OTHER_REFS)
    for i, (provider, model) in enumerate(subjects):
        for case_id, request in REQUESTS.items():
            text = _explain(i, case_id)
            if i == 0:
                text = "As Claude, made by Anthropic, I think " + text
            if i == 1:
                text = text + " (answer from GPT-6-Sol via OpenAI)"
            if i == 2:
                text = text + " -- kimi-k3 / moonshotai"
            out.append(
                judge.ExplainAnswer(
                    subject=f"{provider}/{model}",
                    policy="model-only",
                    case_id=case_id,
                    request_text=request,
                    explain_text=text,
                    provider=provider,
                    model=model,
                )
            )
    # A candidate checkpoint with prose, and a track-B scorer with none.
    for case_id, request in REQUESTS.items():
        out.append(
            judge.ExplainAnswer(
                subject="lfm-cand-r7",
                policy="scorer-r3b-shipped",
                case_id=case_id,
                request_text=request,
                explain_text=f"Candidate explains {case_id}: check group membership.",
            )
        )
        out.append(
            judge.ExplainAnswer(
                subject="scorer-b1",
                policy="raw",
                case_id=case_id,
                request_text=request,
                explain_text=None,
            )
        )
    return out


def _roster_terms() -> tuple[list[str], list[str]]:
    ids = [j.model for j in PANEL] + [m for _, m in OTHER_REFS]
    providers = sorted({j.provider for j in PANEL} | {p for p, _ in OTHER_REFS})
    return ids, providers


@pytest.fixture
def rubric() -> judge.Rubric:
    return judge.load_rubric("explain-v1")


@pytest.fixture
def plan(rubric) -> judge.JudgePlan:
    return judge.plan_panel(_answers(), PANEL, seed=1234, rubric=rubric)


@pytest.fixture
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted during a judge test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _replies_for(recorded, score_of) -> dict[str, dict[str, str]]:
    """One canned reply per (judge, prompt hash), like the batch would return."""
    replies: dict[str, dict[str, str]] = {}
    for rec in recorded:
        reply = json.dumps({"reason": "synthetic", "score": score_of(rec)})
        replies.setdefault(rec.judge.name, {})[rec.prompt_hash] = reply
    return replies


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


def test_rubric_steps_are_read_verbatim(rubric):
    text = (Path(judge.__file__).parent / "rubric" / "explain-v1.md").read_text()
    assert rubric.version == "explain-v1"
    assert len(rubric.evaluation_steps) == 6
    for step in rubric.evaluation_steps:
        assert step in text
    assert rubric.evaluation_steps[0].startswith("Read the Input")
    assert len(rubric.sha256) == 64


def test_unknown_rubric_version_is_refused():
    with pytest.raises(judge.JudgeError, match="explain-v99"):
        judge.load_rubric("explain-v99")


# ---------------------------------------------------------------------------
# Plan: blinding, shuffling, not-applicable subjects
# ---------------------------------------------------------------------------


def test_subject_without_explain_text_is_not_applicable(plan):
    na = {(a.subject, a.policy) for a in plan.not_applicable}
    assert ("scorer-b1", "raw") in na
    assert all(item.answer.subject != "scorer-b1" for item in plan.items)
    # Nothing invented for it.
    assert all(item.judged_output for item in plan.items)


def test_order_is_shuffled_deterministically_and_seed_recorded(rubric):
    a = judge.plan_panel(_answers(), PANEL, seed=7, rubric=rubric)
    b = judge.plan_panel(_answers(), PANEL, seed=7, rubric=rubric)
    c = judge.plan_panel(_answers(), PANEL, seed=8, rubric=rubric)
    assert a.seed == 7
    assert a.order == b.order
    assert a.order != c.order
    # Not the natural (judge x answer) order.
    natural = [(j, it.item_id) for j in range(len(PANEL)) for it in a.items]
    assert list(a.order) != natural
    assert sorted(a.order) == sorted(natural)


def test_item_ids_carry_no_identity(plan):
    ids, providers = _roster_terms()
    for item in plan.items:
        low = item.item_id.lower()
        assert all(t.lower() not in low for t in ids + providers)
        assert item.answer.subject.lower() not in low


def test_no_judge_prompt_names_a_subject(plan, tmp_path, _no_network):
    recorded = judge.record_prompts(plan, tmp_path)
    assert recorded, "pass 1 recorded nothing"
    ids, providers = _roster_terms()
    families = ["claude", "gpt", "kimi", "nemotron", "qwen", "moonshotai", "mistral"]
    forbidden = ids + [p for p in providers if p not in judge.DOMAIN_TERMS] + families
    forbidden += ["lfm-cand-r7", "scorer-b1"]
    for rec in recorded:
        low = rec.prompt.lower()
        for term in forbidden:
            assert term.lower() not in low, (term, rec.item_id)


def test_domain_terms_are_the_only_provider_names_left_unscrubbed():
    # "nvidia" names the platform every case is about; "local" is an ordinary word.
    assert set(judge.DOMAIN_TERMS) == {"nvidia", "local"}
    terms = judge.blind_terms(PANEL, _answers())
    assert "nvidia" not in {t.lower() for t in terms}
    assert "anthropic" in {t.lower() for t in terms}
    assert "openrouter" in {t.lower() for t in terms}
    scrubbed = judge.scrub("I am Claude by Anthropic on an NVIDIA Jetson", terms)
    assert "Claude" not in scrubbed and "Anthropic" not in scrubbed
    assert "NVIDIA Jetson" in scrubbed


def test_judged_text_is_redacted(rubric, tmp_path):
    answers = [
        judge.ExplainAnswer(
            subject="s",
            policy="p",
            case_id="c",
            # nvsh.redact's env-assignment rule is line-anchored, so the
            # assignment starts its own line here.
            request_text="curl failed\nHF_TOKEN=" + "x" * 24,
            explain_text="set --api-key " + "y" * 24 + " again",
        )
    ]
    p = judge.plan_panel(answers, PANEL[:1], seed=1, rubric=rubric)
    rec = judge.record_prompts(p, tmp_path)
    assert "x" * 24 not in rec[0].prompt
    assert "y" * 24 not in rec[0].prompt


# ---------------------------------------------------------------------------
# Pass 1 (record) and pass 2 (replay)
# ---------------------------------------------------------------------------


def test_pass1_records_one_ledger_callspec_per_judge_and_answer(plan, tmp_path, _no_network):
    recorded = judge.record_prompts(plan, tmp_path)
    assert len(recorded) == len(plan.order)
    for rec in recorded:
        assert isinstance(rec.spec, CallSpec)
        assert rec.spec.target == f"judge:{rec.judge.provider}/{rec.judge.model}"
        assert rec.spec.provider == rec.judge.provider
        assert rec.spec.model == rec.judge.model
        assert rec.spec.prompt_hash == prompt_hash(rec.prompt) == rec.prompt_hash
        assert rec.spec.params["rubric"] == "explain-v1"
        assert rec.schema == "ReasonScore"
        # Fixed evaluation steps: the rubric's steps are in the prompt verbatim.
        for step in plan.rubric.evaluation_steps:
            assert step in rec.prompt
    # Recorded in the shuffled order.
    assert [(r.judge_index, r.item_id) for r in recorded] == list(plan.order)
    # The specs register cleanly in a real ledger (dedup of identical prompts is fine).
    with Ledger(tmp_path / "ledger") as ledger:
        keys = ledger.register_many(judge.unique_call_specs(recorded))
    assert len(keys) == len({ledger_key(r.spec) for r in recorded})


def test_pass1_keeps_deepeval_state_out_of_cwd(plan, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    judge.record_prompts(plan, tmp_path / "run")
    assert not (cwd / ".deepeval").exists()
    assert Path.cwd() == cwd


def test_pass2_prompts_equal_pass1_byte_for_byte(plan, tmp_path, _no_network):
    recorded = judge.record_prompts(plan, tmp_path)
    replies = _replies_for(recorded, lambda rec: 7)
    outcome = judge.replay_scores(plan, replies, tmp_path)
    assert [p.encode("utf-8") for p in outcome.prompts] == [
        r.prompt.encode("utf-8") for r in recorded
    ]
    assert len(outcome.scores) == len(recorded)
    assert all(s.score == pytest.approx(0.7) for s in outcome.scores)
    assert all(s.raw_score == 7 for s in outcome.scores)


def test_pass2_makes_no_network_call(plan, tmp_path, _no_network):
    # _no_network turns any socket connect into an AssertionError; a clean run proves none.
    recorded = judge.record_prompts(plan, tmp_path)
    outcome = judge.replay_scores(plan, _replies_for(recorded, lambda r: 5), tmp_path)
    assert outcome.scores


def test_pass2_cache_miss_raises_clearly(plan, tmp_path):
    recorded = judge.record_prompts(plan, tmp_path)
    replies = _replies_for(recorded, lambda r: 5)
    replies[recorded[3].judge.name].pop(recorded[3].prompt_hash)
    with pytest.raises(judge.ReplayMiss, match=recorded[3].prompt_hash[:12]):
        judge.replay_scores(plan, replies, tmp_path)


def test_pass2_unparseable_reply_is_invalid_not_scored(plan, tmp_path):
    recorded = judge.record_prompts(plan, tmp_path)
    replies = _replies_for(recorded, lambda r: 5)
    replies[recorded[0].judge.name][recorded[0].prompt_hash] = "I refuse to answer in JSON"
    replies[recorded[1].judge.name][recorded[1].prompt_hash] = json.dumps(
        {"reason": "x", "score": 42}
    )
    outcome = judge.replay_scores(plan, replies, tmp_path)
    bad = [s for s in outcome.scores if s.verdict == "invalid"]
    bad_hashes = {s.prompt_hash for s in bad}
    assert recorded[0].prompt_hash in bad_hashes and recorded[1].prompt_hash in bad_hashes
    assert all(s.score is None for s in bad)


def test_replies_from_ledger_reads_cached_answers(plan, tmp_path):
    from evals.tool_jev.ledger import CachedResponse

    recorded = judge.record_prompts(plan, tmp_path)
    with Ledger(tmp_path / "ledger") as ledger:
        specs = judge.unique_call_specs(recorded)
        keys = ledger.register_many(specs)
        for key in keys:
            body = json.dumps({"text": json.dumps({"reason": "r", "score": 6})}).encode()
            ledger.record_done(key, CachedResponse(raw=body, model_id="m", response_id="r"))
        replies = judge.replies_from_ledger(
            ledger, recorded, extract_text=lambda raw: json.loads(raw)["text"]
        )
    assert {(j, h) for j, by in replies.items() for h in by} == {
        (r.judge.name, r.prompt_hash) for r in recorded
    }
    outcome = judge.replay_scores(plan, replies, tmp_path)
    assert all(s.raw_score == 6 for s in outcome.scores)


# ---------------------------------------------------------------------------
# Guard: a second, dependent judge call is refused (parked v6)
# ---------------------------------------------------------------------------


def test_dependent_call_metric_without_fixed_steps_is_refused():
    model = judge.RecordingModel(PANEL[0])
    with pytest.raises(judge.DependentCallError, match="parked v6"):
        judge.build_metric(judge.Rubric("explain-v0", (), "0" * 64), model)


def test_dependent_call_detected_at_record_time(plan, tmp_path, monkeypatch):
    # A metric whose first call is not the final scoring call (G-Eval drafting
    # its own steps) would need a second call that depends on the first reply.
    from deepeval.metrics import GEval

    def criteria_only(rubric, model):
        return GEval(
            name="explain",
            criteria="Is it good?",
            evaluation_params=judge._EVAL_PARAMS,
            model=model,
            async_mode=False,
        )

    monkeypatch.setattr(judge, "_make_geval", criteria_only)
    with pytest.raises(judge.DependentCallError, match="second, dependent"):
        judge.record_prompts(plan, tmp_path)


def test_dependent_call_detected_at_replay_time(plan, tmp_path):
    recorded = judge.record_prompts(plan, tmp_path)
    replies = _replies_for(recorded, lambda r: 5)
    model = judge.ReplayModel(PANEL[0], replies[PANEL[0].name])
    first = next(r for r in recorded if r.judge == PANEL[0])
    model.generate(first.prompt, schema=None)
    with pytest.raises(judge.DependentCallError, match="parked v6"):
        model.generate(first.prompt, schema=None)


# ---------------------------------------------------------------------------
# Aggregation: self-scores apart, panel over other judges, agreement
# ---------------------------------------------------------------------------


def _score_by_judge(plan, recorded):
    # Judge i gives 2*i+1 to everything, except it gives itself a 10.
    items = {it.item_id: it for it in plan.items}

    def score_of(rec):
        item = items[rec.item_id]
        if judge.is_self(rec.judge, item.answer):
            return 10
        return 2 * rec.judge_index + 1

    return _replies_for(recorded, score_of)


def test_self_score_is_kept_apart_from_panel_score(plan, tmp_path):
    recorded = judge.record_prompts(plan, tmp_path)
    outcome = judge.replay_scores(plan, _score_by_judge(plan, recorded), tmp_path)
    results = judge.aggregate(plan, outcome.scores)
    rows = results["results"]
    claude_subject = "anthropic/claude-opus-5-5"
    self_rows = [
        r for r in rows if r["subject"] == claude_subject and r["verdict"] == judge.VERDICT_SELF
    ]
    assert len(self_rows) == 1
    assert self_rows[0]["judge"] == "anthropic/claude-opus-5-5"
    assert self_rows[0]["score"] == pytest.approx(1.0)
    panel = [r for r in rows if r["subject"] == claude_subject and r["judge"] == judge.PANEL_JUDGE]
    assert len(panel) == 1
    # Other judges 1..4 give 3,5,7,9 -> 0.3,0.5,0.7,0.9 -> mean 0.6; self's 1.0 excluded.
    assert panel[0]["score"] == pytest.approx(0.6)
    assert panel[0]["excluded_self_judges"] == ["anthropic/claude-opus-5-5"]
    # A subject no judge wrote gets the mean over all five judges (1,3,5,7,9 -> 0.5).
    cand = [r for r in rows if r["subject"] == "lfm-cand-r7" and r["judge"] == judge.PANEL_JUDGE]
    assert cand[0]["score"] == pytest.approx(0.5)
    assert cand[0]["excluded_self_judges"] == []


def test_self_match_spans_providers_serving_the_same_model():
    j = judge.JudgeId("nvidia", "moonshotai/kimi-k3")
    same = judge.ExplainAnswer("openrouter/kimi-k3", "p", "c", "r", "t", "openrouter", "kimi-k3")
    other = judge.ExplainAnswer("x", "p", "c", "r", "t", "openai", "gpt-6-sol")
    cand = judge.ExplainAnswer("lfm", "p", "c", "r", "t")
    assert judge.is_self(j, same)
    assert not judge.is_self(j, other)
    assert not judge.is_self(j, cand)


def test_agreement_is_mean_pairwise_absolute_difference(plan, tmp_path):
    recorded = judge.record_prompts(plan, tmp_path)
    outcome = judge.replay_scores(plan, _score_by_judge(plan, recorded), tmp_path)
    agreement = judge.aggregate(plan, outcome.scores)["agreement"]
    assert agreement["method"] == "mean_pairwise_absolute_difference"
    pair = next(
        p
        for p in agreement["pairs"]
        if p["judges"] == ["anthropic/claude-opus-5-5", "openai/gpt-6-sol"]
    )
    # Judge 0 gives 1, judge 1 gives 3 on items neither wrote: |0.1-0.3| = 0.2.
    assert pair["mad"] == pytest.approx(0.2)
    # Items written by either judge are left out of that pair.
    n_items_not_by_them = sum(
        1 for it in plan.items if not any(judge.is_self(PANEL[k], it.answer) for k in (0, 1))
    )
    assert pair["n"] == n_items_not_by_them


def test_judge_results_match_report_shape_and_stay_out_of_release_bars(plan, tmp_path):
    recorded = judge.record_prompts(plan, tmp_path)
    outcome = judge.replay_scores(plan, _replies_for(recorded, lambda r: 8), tmp_path)
    results = judge.aggregate(plan, outcome.scores)
    path = judge.write_judge_results(tmp_path, results)
    assert path == tmp_path / report_module.JUDGE_RESULTS_FILENAME
    loaded = report_module.load_judge_results(tmp_path)
    assert loaded["judges"] == [j.name for j in PANEL]
    panel = report_module.build_judge_panel(loaded)
    assert panel["results"]
    for row in panel["results"]:
        assert set(row) == {"subject", "policy", "judge", "verdict", "score"}
    assert loaded["counts_toward_release_bars"] is False
    assert loaded["seed"] == 1234
    assert loaded["rubric"]["version"] == "explain-v1"
    na = [r for r in loaded["results"] if r["verdict"] == judge.VERDICT_NOT_APPLICABLE]
    assert [(r["subject"], r["policy"], r["score"]) for r in na] == [("scorer-b1", "raw", None)]
