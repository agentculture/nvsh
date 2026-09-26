# Handoff: DeepEval release gate (issue 64)

State at the end of the 2026-09-26 session. The guide is
[`deepeval-gate.md`](deepeval-gate.md); this file is only what the next
session needs to pick up.

## Where the work is

- Branch `spec/deepeval-release-gate-issue-64`, local only (not pushed, no
  PR). Version not bumped yet (still 0.20.0).
- devague frame and plan slug: `deepeval-release-gate-for-tool-jev-issue-64`
  (`devague plan status`, `devague deviate --list`, `devague lapse --list`).
- Tests: evals suite 520 passing
  (`uv run pytest -c evals/pytest.ini --rootdir=. -q`); root suite 4710
  passing, 55 skipped.

## Done

| Task | What |
| --- | --- |
| t1 | Scaffold, env guard, isolated pytest, CI `evals` job |
| t2 t5 t6 t7 t8 t9 t10 | cases, manifest, trace, metrics bridge, policies, ledger, provider base |
| t11 t12 t13 t14 t15 t18 | request contract, OpenAI, Anthropic, OpenAI-compatible adapters, DeepEval layer, report |
| t16 t23 | judge panel, permutation entries |
| d1 | `track_a_loop.py`: references' Track A through the candidates' multi-round `LfmTier` loop; `CallRequest.history` rendered natively by all adapters |
| t20 | the one approved a3-heal.q4_k_m run on the issue-53 test set and slice |
| t21 | local references confirmed on the lobes gateway (no new serving) |
| review fixes | codex wave-1 (A, B), wave-2 (C, deepeval destinations, and the four adapter fixes merged with d1: Anthropic unresolved batch lookup, `custom_id`-keyed context, OpenAI batch-line and failed-response classification), t15 rework, probe cwd |

## In flight at handoff

Nothing. Every started task is merged; no worktree or agent is running.

## Next, in order

1. Done 2026-09-26: codex review of the d1 loop, all findings fixed
   (`962852a`), and plan risk r9 fixed (`e937544`): loop calls send
   `tool_choice: auto` for every provider, a reply with no tool call reaches
   `LfmTier` as its visible text (`Provider.reply_text`), truncated replies
   are invalid, adapters answer with the first well-formed call, Anthropic
   native blocks replay in order, and argument redaction keeps the JSON
   secret-field rule. Deviation d3 changed the OpenRouter lineup.
2. **t17 runner** (opus): `python -m evals.tool_jev run|continue|status|smoke`,
   folding in deviation d2's driver loop. It must supply what no module owns
   yet:
   - the judge call path (free text, neither `tool_call` nor `choice`);
   - reply-text extraction from each provider's cached raw bytes;
   - the explain-text source per subject (candidates' saved predictions
     carry no prose; scorers are not applicable);
   - the run-dir layout that `report.py` reads (`manifest.json`, `traces/`,
     `metrics/`, `permutation.json`, `judge_results.json`);
   - per-(subject, policy) metrics through
     `deepeval_layer.apply_policy_to_prediction`;
   - stop-and-ask on an **unresolved** Anthropic orphan batch (risk r6);
     never resubmit it;
   - read judge answers and explain text with `Provider.reply_text` (visible
     text only; `truncated` flags a reply cut at the output budget);
   - the reasoning parameter name for build.nvidia.com and local models is
     an unverified assumption (`reasoning_effort`); settle it in the smoke run.
3. **d2 packaging**: Dockerfile + compose (pinned `python:3.12-slim` + uv,
   host network, private run dir read-write, case data read-only,
   `restart: unless-stopped`, logs), started with
   `grant run --inject ... -- docker compose up -d`.
4. **t19** `evals/README.md`, then **t22** 10-case smoke run (measure tokens,
   set per-provider caps), **t24** full run, **t25** delivery (version bump,
   CHANGELOG, follow-up issues for tier wiring and execution eval, PR "part of
   #64"), then `/validate-delivery` and `/summarize-delivery`.

## Open items

- Plan risks open: r10 (512-token output cap vs medium reasoning; settle in
  the smoke run), r5 (build.nvidia.com free-tier throttling), r6 (Anthropic
  in-progress batch cannot be matched: the adapter now raises
  `BatchLookupUnresolved`; the runner must stop and ask). r1-r4, r7-r9 are
  resolved; r9 leaves two live checks for the smoke run (Anthropic thinking
  replay acceptance, OpenAI reasoning items not replayed).
- Lapses l1-l4 are filed and proposed; the operator adjudicates them at
  delivery.
- Issue [#66](https://github.com/agentculture/nvsh/issues/66) (mid-line env
  assignment not redacted) is fixed in its own PR, not this one.
- Budgets: the operator topped up OpenRouter (\$20 credit, \$20 key limit),
  Anthropic and OpenAI. After deviation d3 the OpenRouter estimate is about
  \$15 per fresh full run plus about \$0.6 for the smoke run. The operator
  will add OpenRouter funds as needed and accepts hitting the ceiling as a
  live test of the money-stop path.

## Rules that bit this run

- Evals pytest needs `--rootdir=.` or it runs the whole repository suite.
- `scripts/scan-secrets.py` scans only tracked files: run it after `git add`,
  and never write `sk-...` literals or `token = "..."` assignments in tests.
- Anything that imports deepeval must not run with the repository as its
  working directory (deepeval creates `.deepeval/` in the cwd).
- Commit only when the evals run is green; merge through the gated merge.
- Briefs quote the plan verbatim and share one request contract; integration
  gaps between separately built modules were found only by cross-module tests
  and codex reviews after each wave, so keep both.
