# Delivery Summary — DeepEval release gate for Tool-Jev (issue 64)

plan: `deepeval-release-gate-for-tool-jev-issue-64` · run: `complete` · date: `2026-09-26`
baseline: `devague summary skeleton`

## Intent

> nvsh has a repeatable DeepEval release gate for its Tool-Jev models: one command scores any fine-tuned checkpoint and reference models (frontier models via the OpenAI and Anthropic platform APIs, open models via OpenRouter and build.nvidia.com, and local models) on the same fixed cases, model-only and through explicit nvsh harness policies, so a3-heal and scorer-r3b can be judged for release and wiring into the tier stack, and the next fine-tune is judged the same way

After: One documented command replays the saved outputs of a3-heal, scorer-r3b and baselines plus fresh reference-model runs, and writes a JSON result, per-case traces with raw and final side by side, and a markdown comparison page; adding a new checkpoint is one manifest entry

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Scaffold evals/: dependency group, env guard, isolated pytest, CI lint + evals job
- `t2` — Case model and case-set loader with split guards
- `t5` — Run manifest: candidates, baselines, references, policies, case sets
- `t6` — Trace schema: raw and per-policy final records side by side, private run dir
- `t7` — Metrics bridge onto scripts/lfm-finetune (raw quality + corpus metrics)
- `t8` — Harness policies as versioned JSON configs applied offline
- `t9` — Durable call ledger and response cache (resume across stops and resets)
- `t10` — Provider base, error taxonomy, fake provider, redaction and no-exec guard
- `t11` — Request contract shared with the candidates (both interfaces)
- `t12` — OpenAI adapter: sync + Batch API
- `t13` — Anthropic adapter: sync + Message Batches API
- `t14` — OpenAI-compatible adapter for OpenRouter, build.nvidia.com and local servers
- `t15` — DeepEval layer: test cases, exact metrics, local export
- `t16` — Blind all-to-all judge panel with two-pass DeepEval record/replay
- `t17` — Runner: run / continue / status with budgets and clean stops
- `t18` — Report: JSON result + markdown comparison page from one run
- `t19` — evals/README.md: operator and next-cycle agent guide
- `t20` — Live: one a3-heal.`q4_k_m` run on issue-53 test + missing-candidate slice (c38)
- `t21` — Live: local reference serving (Qwen3.8 27B, gemma-4-26b-a4b)
- `t22` — Live: 10-case smoke across the roster, measure tokens, set budget caps
- `t23` — Permutation stability for the release candidates (test side only)
- `t24` — Live: full gate run, reproduce recorded figures, benchmark page
- `t25` — Delivery hygiene: non-goals verified, follow-up issue, version bump

2 tasks were rejected during planning — see `devague plan show`.

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `evals/` tree, `evals` uv dependency group, `evals/pytest.ini`, env guard (telemetry/dotenv off, Confident key refused), CI lint + evals job |
| `t2` | delivered | case model + case-set loader with split guards (held-out refused) |
| `t5` | delivered | run manifest: candidates, baselines, references, judges (with optional per-judge reasoning, `d8`), policies, case sets, budgets, `[track_a]`, `[stops]`, `[judging]` |
| `t6` | delivered | trace schema with raw and per-policy final side by side, Track A inspections (`d9`); reference lineup changed by `d3`, `d7` |
| `t7` | delivered | metrics bridge onto `scripts/lfm-finetune` (top-k, Brier, ECE, abstain P/R, wrong mutating) plus Track A task done (`d9`) |
| `t8` | delivered | versioned JSON harness policies applied offline |
| `t9` | delivered | durable call ledger + response cache, resume across stops and restarts |
| `t10` | delivered | provider base, error taxonomy, fake provider, key-aware redaction, no-exec guard |
| `t11` | delivered | request contract for both interfaces; Track A through the candidates' LfmTier loop (`d1`) |
| `t12` | delivered | OpenAI adapter, sync + Batch API (missing-scope 401 classified) |
| `t13` | delivered | Anthropic adapter, sync + Message Batches (hashed `custom_id` for long case ids) |
| `t14` | delivered | OpenAI-compatible adapter for OpenRouter, build.nvidia.com, local (per-provider timeout) |
| `t15` | delivered | DeepEval layer: per-case test cases, exact metrics, local export |
| `t16` | delivered | blind all-to-all judge panel, two-pass record/replay; ran with three judges (`d7`), qwen at low reasoning (`d8`) |
| `t17` | delivered | runner `run`/`continue`/`status`/`smoke`/`drive` with budgets, reservations, clean stops, per-provider pools, per-model interleave and timeout pause; docker driver + Discord alerts (`d2`, `d5`, `d6`) |
| `t18` | delivered | JSON result + markdown comparison page with Right and Task done columns (`d9`) |
| `t19` | delivered | `evals/README.md` operator guide with a tested no-network fixture run |
| `t20` | delivered | one a3-heal.`q4_k_m` run on the issue-53 test + missing-candidate slice; baselines measured too (`d4`) |
| `t21` | delivered | local references served via the lobes gateway (Qwen3.8-27B, Gemma-4-26B-A4B) |
| `t22` | delivered | 10-case smoke: 0 truncated, 0 invalid at medium reasoning / 2048 tokens; budget caps set |
| `t23` | delivered | permutation stability entries for the release candidates (test side only; Track A not measurable, stated) |
| `t24` | delivered | full run `8b59d0afa150` complete: 15 references, 3-judge panel over 1,581 explanations, report `docs/benchmarks/2026-09-29-deepeval-gate-run-8b59d0afa150.md`; \$35.78 spent |
| `t25` | delivered | version 0.21.0, CHANGELOG, harness prompt files, guide; follow-up issues #69 #70 #71; PR #68 |

## Mid-work Decisions

Decisions no deviation record covers, captured directly:

- The run's live fixes, each with a regression test: Anthropic `custom_id` hash token; `missing_scope` for a restricted OpenAI key; per-provider `timeout_seconds`; 60 s transient pauses and a 300 s round deadline; one sync pool per provider; per-model interleave with a model-scoped timeout pause.
- The private manifest gained `[budget.nvidia] timeout_seconds = 180`, and the Anthropic and OpenRouter caps were raised to \$20 and \$16 after the operator added funds (\$5 on Anthropic); config only, recorded in the run's manifest history.
- The worktree guard refuses an empty `/tmp/.git` left by another session's codex sandbox; the evals suite was run with its temporary files outside `/tmp` (issue #71 item 2).

Approved deviation records:

- `d1` — Reference models' Track A runs through the candidates' own multi-round LfmTier loop (up to 4 rounds, read-only inspections executed against the recorded ground snapshot, then propose/explain/escalate) instead of t11's single-turn `tool_call` request: each case is replayed through the real LfmTier with a deferred chat client (cached replies replayed; the first uncached reply becomes a pending ledger call keyed by its exact prompt hash and round); OpenAI and Anthropic batch one loop round at a time. New module evals/`tool_jev`/`track_a_loop.py` built before the runner (t17) — The candidates (a3-heal) were measured by measure.py through LfmTier's multi-round loop; single-turn reference requests score an inspect-first model as malformed, breaking c33/h23 (same request contract incl. ground snapshot). Operator approved 2026-09-26: batch per round; cost ~$40-50 per fresh run batched (was ~$27), wall clock hours to days
- `d2` — Add an autonomous driver: a docker compose service (pinned python:3.12-slim + uv image, `network_mode` host for the localhost lobes gateway, private run dir mounted read-write and case data read-only, restart unless-stopped, logs via docker logs plus a log file in the run dir, deepeval telemetry off) that runs the gate's 'continue' loop until done, survives the session ending and reboots, and continues on its own. Keys by grant passthrough: 'grant run --inject ... -- docker compose up -d' with only variable names in the compose file, no key file on disk. On a money stop (402, quota, budget cap) it keeps other providers going and re-checks the stopped provider every 30 min with one probe call; a rejected request stops only that model's calls and is logged; batches keep being polled — Operator request 2026-09-26: a full run spans hours to days (per-round batching, d1) and must not depend on this Claude session; operator chose docker compose, grant passthrough for keys, wait-and-recheck on stops
- `d3` — OpenRouter reference lineup changed to fit its budget: add qwen/qwen3.8-27b and deepseek/deepseek-v4.1-flash; move moonshotai/kimi-k3 to build.nvidia.com only (kimi host-to-host comparison dropped); keep qwen/qwen3.8-max-0902 as a reference and a judge (every judge also answers). 16 references; judges unchanged. Estimated OpenRouter cost about $15 per fresh full run plus about $0.6 for the smoke run, within the $20 key limit; the operator accepts hitting the ceiling as a test of the money-stop path and will top up as needed — Operator request 2026-09-26: smaller OpenRouter models to fit $20; live OpenRouter price list showed kimi-k3 ($3/$15 per M) was most of the OpenRouter cost; no deepseek v4.1 pro exists, only v4.1-flash
- `d4` — Run the two baselines locally once on the issue-53 test set (198) and its missing-candidate slice (83) with the same measure.py command shape as the approved a3-heal run: stock Qwen3.5-0.8B (Track A) and scorer-b1 (Track B); no held-out runs; no API cost — Operator decision 2026-09-26: neither baseline had saved issue-53 test predictions (scorer-b1's are on the issue-46 test, whose ids are in issue-53's training split), so the gate would have no baseline rows
- `d5` — Add Discord webhook alerts to the unattended driver (extends deviation d2): every 10% of each provider's calls answered, every whole dollar of total spend, each provider/model stop, and the run's end (complete or stop-and-ask); counts, dollars, names and stop reasons only, never case text; webhook URL read from `NVSH_EVALS_ALERT_WEBHOOK` (grant-injected), sent milestones recorded in the run dir so restarts never repeat them — Operator request 2026-09-26: know the unattended run's progress, spend (so each dollar used is visible) and finish state without watching the terminal
- `d6` — Extend deviation d5's alerts with a status summary every 30 minutes (per provider: percent answered, calls to go, invalid, spend of cap; plus active stops), posted by a heartbeat thread in the driver so it keeps flowing during long steps — Operator request 2026-09-26: a 30-minute cadence status update
- `d7` — Close the full run without build.nvidia.com's slow tail: drop z-ai/glm-5.3 from the references (its unfinished missing-candidate slice, 240 calls, and its finished test-set rows leave the report) and drop moonshotai/kimi-k3 and nvidia/nemotron-3-ultra-550b-a55b from the judge panel (both stay references), leaving a three-judge panel (claude-opus-5-5, gpt-6-sol, qwen/qwen3.8-max-0902). Budget caps unchanged: judges run until a cap stops them; the operator adds funds, the cap is raised and the run continues from the ledger — Operator decision 2026-09-28: the free build.nvidia.com tier answered about 110 calls an hour; glm-5.3's 240 calls plus 1,581 judge calls each for kimi-k3 and nemotron-ultra meant about 30 more hours; kimi-k3 answered malformed 257 of 784 times as a subject, a weak judge; paid judges estimated Anthropic $11-19 (cap left $8.25), OpenAI $5-9 ($11.10 left), OpenRouter $7.6-12.5 ($8.70 left)
- `d8` — Judge panel parameters per judge: a \[\[judge\]\] entry may set its own reasoning effort; qwen/qwen3.8-max-0902 judges at reasoning low (its subject calls stay medium), claude-opus-5-5 and gpt-6-sol judges stay medium; only qwen's judge calls are re-keyed (its 63 judge calls so far, about $0.30, are superseded) — Operator decision 2026-09-29: at medium reasoning qwen3.8-max as a judge truncated 12 of 63 replies (19%) at the 1024-token judge output cap; the cap and effort are shared by all judges and changing them globally would re-key the already-paid OpenAI and Anthropic judge batches (1578 + 684 calls); per the no-spend-on-capped-output rule try lower reasoning first
- `d9` — Track A reports a second column, 'task done', beside the strict 'right proposal': a read-only-expected case also counts as done when the subject ran the expected read-only operation with the expected arguments as a loop inspection and then ended in explain. The strict metric is unchanged; Track A records carry the inspections they ran — Operator decision 2026-09-29: in the LfmTier loop read-only operations execute as inspections, and frontier references answered by running the expected read-only command and explaining its result (gpt-6-sol 53/53, claude-opus-5-5 61/61, qwen3.8-max 17/17 such cases), which the strict proposal metric scores as misses (opus 5/83 strict vs about 66/83 task done); a3-heal was trained to propose instead

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t11` (`d1`) | The candidates (a3-heal) were measured by measure.py through LfmTier's multi-round loop; single-turn reference requests score an inspect-first model as malformed, breaking c33/h23 (same request contract incl. ground snapshot). Operator approved 2026-09-26: batch per round; cost ~$40-50 per fresh run batched (was ~$27), wall clock hours to days | `acceptable` |
| `t17` (`d2`) | Operator request 2026-09-26: a full run spans hours to days (per-round batching, d1) and must not depend on this Claude session; operator chose docker compose, grant passthrough for keys, wait-and-recheck on stops | `acceptable` |
| `t6` (`d3`) | Operator request 2026-09-26: smaller OpenRouter models to fit $20; live OpenRouter price list showed kimi-k3 ($3/$15 per M) was most of the OpenRouter cost; no deepseek v4.1 pro exists, only v4.1-flash | `acceptable` |
| `t20` (`d4`) | Operator decision 2026-09-26: neither baseline had saved issue-53 test predictions (scorer-b1's are on the issue-46 test, whose ids are in issue-53's training split), so the gate would have no baseline rows | `acceptable` |
| `t17` (`d5`) | Operator request 2026-09-26: know the unattended run's progress, spend (so each dollar used is visible) and finish state without watching the terminal | `acceptable` |
| `t17` (`d6`) | Operator request 2026-09-26: a 30-minute cadence status update | `acceptable` |
| `t24` (`d7`) | Operator decision 2026-09-28: the free build.nvidia.com tier answered about 110 calls an hour; glm-5.3's 240 calls plus 1,581 judge calls each for kimi-k3 and nemotron-ultra meant about 30 more hours; kimi-k3 answered malformed 257 of 784 times as a subject, a weak judge; paid judges estimated Anthropic $11-19 (cap left $8.25), OpenAI $5-9 ($11.10 left), OpenRouter $7.6-12.5 ($8.70 left) | `acceptable` |
| `t16` (`d8`) | Operator decision 2026-09-29: at medium reasoning qwen3.8-max as a judge truncated 12 of 63 replies (19%) at the 1024-token judge output cap; the cap and effort are shared by all judges and changing them globally would re-key the already-paid OpenAI and Anthropic judge batches (1578 + 684 calls); per the no-spend-on-capped-output rule try lower reasoning first | `acceptable` |
| `t18` (`d9`) | Operator decision 2026-09-29: in the LfmTier loop read-only operations execute as inspections, and frontier references answered by running the expected read-only command and explaining its result (gpt-6-sol 53/53, claude-opus-5-5 61/61, qwen3.8-max 17/17 such cases), which the strict proposal metric scores as misses (opus 5/83 strict vs about 66/83 task done); a3-heal was trained to propose instead | `acceptable` |

## Evidence

- tests: `uv run pytest -c evals/pytest.ini --rootdir=. -q` (temporary files outside `/tmp`) at `458d911` — pass (799)
- tests: `uv run pytest -n auto -q` — pass (4710, 55 skipped, at `f71c765`; nothing under `nvsh/` changed since)
- tests: `evals/tool_jev/tests/test_task_done.py`, `test_judge_reasoning.py`, `test_run_fairness.py` — pass
- live run: `8b59d0afa150` status complete; `result.json`, `report.md`, `judge_results.json` written; report copied to `docs/benchmarks/2026-09-29-deepeval-gate-run-8b59d0afa150.md`
- replay: the candidates' and baselines' saved outputs reproduce their recorded figures through the gate (a3-heal 77/83, 4 wrong mutating; scorer-r3b 79/83, 0; scorer-b1 68/83, 1; stock 6/83)
- validate-delivery records: obligations `o1`-`o6`, evidence `e1`-`e6`, deltas `b1`-`b3` (all proposed)
- commits: `d8416bc..458d911` on `spec/deepeval-release-gate-issue-64`
- PRs / issues: #64, PR #68, follow-ups #69 #70 #71; #66 separate

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| the gate never ships in or is imported by the nvsh runtime | high | test `test_scaffold.py::test_wheel_contains_no_evals_files` · `e1` |
| the eval suite is isolated from the root suite and green | high | 799 evals / 4710 root passed · `e2` |
| runs resume without re-sending or re-billing a call | high | test `test_run.py::test_interrupt_then_continue_gives_identical_outputs_and_sends_nothing_twice` · many live redeploys of `8b59d0afa150` · `e3` |
| one command produces result.json and the comparison page | high | fixture test `test_readme.py::test_fixture_run_commands_reach_a_real_run_outside_the_repo` (`e4`) and the live run (`e6`) |
| the gate reproduces the candidates' recorded figures | medium | scratch replay on a run-dir copy (`e5`, observation) and the committed report's candidate rows |
| first gate result: `scorer-r3b.q4_k_m` 79/83 right, 0 wrong mutating on test and slice, best escalation recall of any subject | high | report `docs/benchmarks/2026-09-29-deepeval-gate-run-8b59d0afa150.md` · `e6` |
| `a3-heal.q4_k_m` Track A task done | low | its saved predictions record no inspections, so task done equals strict (a lower bound; issue #71) |
| candidates' explanations judged | unverified | saved predictions carry no explanation text; judge rows not_applicable (`b3`, issue #71) |
| release / publication verdict | unverified | not this work (non-goal c16); wiring is issue #69 |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `provenance-missing` | Scope claim c4 said a3-heal has saved held-out and missing-candidate predictions, relying on an explorer's inventory; the challenge probe found only final (test) predictions for a3-heal and a3-heal.`q4_k_m` |
| `l5` | `control-absent` | commit 4a55743 landed with one evals test red: the gate command piped pytest into tail, so the commit ran on tail's exit status, not pytest's; fixed in a146ab6, and the gate now checks pytest's own exit code |

pending approval (not yet evidence): `l2`, `l3`, `l4`

## Remaining Work / Follow-up

- wire `scorer-r3b.q4_k_m` into the tier stack (issue-54 style) — issue #69
- execution-based evaluation layer (non-goal c15) — issue #70
- record candidates' Track A inspections and explanations; fix the worktree guard's empty-`.git` false positive; NVIDIA hosting; Track A contract wording — issue #71
- adjudicate proposed lapses `l2`-`l4` and proposed records `o1`-`o6`, `e1`-`e6`, `b1`-`b3` — operator
- rotate the OpenAI and build.nvidia.com keys pasted during the run — operator
- stop the compose projects `nvsh-evals-gate` and `nvsh-evals-rehearsal` — main agent, after this summary is committed
