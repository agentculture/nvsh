# Delivery Summary — DeepEval release gate for Tool-Jev (issue 64)

plan: `deepeval-release-gate-for-tool-jev-issue-64` · run: `partial (interim; the full gate run t24 is still in flight)` · date: `2026-09-26`
baseline: `devague summary skeleton`

## Intent

> nvsh has a repeatable DeepEval release gate for its Tool-Jev models: one command scores any fine-tuned checkpoint and reference models (frontier models via the OpenAI and Anthropic platform APIs, open models via OpenRouter and build.nvidia.com, and local models) on the same fixed cases, model-only and through explicit nvsh harness policies, so a3-heal and scorer-r3b can be judged for release and wiring into the tier stack, and the next fine-tune is judged the same way

After: One documented command replays the saved outputs of a3-heal, scorer-r3b and baselines plus fresh reference-model runs, and writes a JSON result, per-case traces with raw and final side by side, and a markdown comparison page; adding a new checkpoint is one manifest entry

## Intent note

This is an **interim** summary written while the full gate run (t24, run
`8b59d0afa150`) is still running unattended under the docker compose driver.
No gate verdict exists yet; it is listed under Remaining Work. This file is
updated in place when the run completes.

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
| `t1` | delivered | `evals/` tree, `evals` uv dependency group, `evals/pytest.ini`, env guard (telemetry/dotenv forced off, Confident key refused), CI lint + evals job |
| `t2` | delivered | case model + case-set loader with split guards (held-out refused) |
| `t5` | delivered | run manifest (candidates, baselines, 16 references, policies, case sets, budgets, `[track_a]`, `[stops]`, `[judging]`) |
| `t6` | delivered | trace schema with raw and per-policy final side by side; private run dir (reference lineup changed by `d3`) |
| `t7` | delivered | metrics bridge onto `scripts/lfm-finetune` (top-k, Brier, ECE, abstain P/R, wrong mutating) |
| `t8` | delivered | versioned JSON harness policies applied offline |
| `t9` | delivered | durable call ledger + response cache, resume across stops |
| `t10` | delivered | provider base, error taxonomy, fake provider, key-aware redaction, no-exec guard |
| `t11` | delivered | request contract for both interfaces; Track A replaced by the candidates' LfmTier loop (`d1`, `track_a_loop.py`) |
| `t12` | delivered | OpenAI adapter, sync + Batch API (missing-scope 401 classified) |
| `t13` | delivered | Anthropic adapter, sync + Message Batches (hashed `custom_id` for long case ids) |
| `t14` | delivered | OpenAI-compatible adapter for OpenRouter, build.nvidia.com, local (per-provider timeout) |
| `t15` | delivered | DeepEval layer: per-case test cases, exact metrics, local export |
| `t16` | delivered | blind all-to-all judge panel, two-pass record/replay (not yet exercised on the full run) |
| `t17` | delivered | runner `run`/`continue`/`status`/`smoke`/`drive` with budgets, reservations, clean stops; docker driver + Discord alerts (`d2`, `d5`, `d6`); two codex review rounds fixed |
| `t18` | delivered | JSON result + markdown comparison page (fixture-verified; not yet produced for the full run) |
| `t19` | delivered | `evals/README.md` operator guide with a tested no-network fixture run |
| `t20` | delivered | one a3-heal.`q4_k_m` run on the issue-53 test + missing-candidate slice; baselines measured too (`d4`) |
| `t21` | delivered | local references served via the lobes gateway (Qwen3.8-27B, Gemma-4-26B-A4B) |
| `t22` | delivered | 10-case smoke: 0 truncated, 0 invalid at medium reasoning / 2048 tokens; budget caps set |
| `t23` | delivered | permutation stability entries for the release candidates (test side only) |
| `t24` | partial | full run in flight: Anthropic and OpenAI answered in full; OpenRouter (qwen3.8-max), local (Qwen3.8-27B) and build.nvidia.com still sending; no result.json, no benchmark page yet |
| `t25` | partial | version 0.21.0, CHANGELOG, harness prompt files, guides done; follow-up issues and the PR not yet opened |

## Mid-work Decisions

- `d1` — Reference models' Track A runs through the candidates' own multi-round LfmTier loop (up to 4 rounds, read-only inspections executed against the recorded ground snapshot, then propose/explain/escalate) instead of t11's single-turn `tool_call` request: each case is replayed through the real LfmTier with a deferred chat client (cached replies replayed; the first uncached reply becomes a pending ledger call keyed by its exact prompt hash and round); OpenAI and Anthropic batch one loop round at a time. New module evals/`tool_jev`/`track_a_loop.py` built before the runner (t17) — The candidates (a3-heal) were measured by measure.py through LfmTier's multi-round loop; single-turn reference requests score an inspect-first model as malformed, breaking c33/h23 (same request contract incl. ground snapshot). Operator approved 2026-09-26: batch per round; cost ~$40-50 per fresh run batched (was ~$27), wall clock hours to days
- `d2` — Add an autonomous driver: a docker compose service (pinned python:3.12-slim + uv image, `network_mode` host for the localhost lobes gateway, private run dir mounted read-write and case data read-only, restart unless-stopped, logs via docker logs plus a log file in the run dir, deepeval telemetry off) that runs the gate's 'continue' loop until done, survives the session ending and reboots, and continues on its own. Keys by grant passthrough: 'grant run --inject ... -- docker compose up -d' with only variable names in the compose file, no key file on disk. On a money stop (402, quota, budget cap) it keeps other providers going and re-checks the stopped provider every 30 min with one probe call; a rejected request stops only that model's calls and is logged; batches keep being polled — Operator request 2026-09-26: a full run spans hours to days (per-round batching, d1) and must not depend on this Claude session; operator chose docker compose, grant passthrough for keys, wait-and-recheck on stops
- `d3` — OpenRouter reference lineup changed to fit its budget: add qwen/qwen3.8-27b and deepseek/deepseek-v4.1-flash; move moonshotai/kimi-k3 to build.nvidia.com only (kimi host-to-host comparison dropped); keep qwen/qwen3.8-max-0902 as a reference and a judge (every judge also answers). 16 references; judges unchanged. Estimated OpenRouter cost about $15 per fresh full run plus about $0.6 for the smoke run, within the $20 key limit; the operator accepts hitting the ceiling as a test of the money-stop path and will top up as needed — Operator request 2026-09-26: smaller OpenRouter models to fit $20; live OpenRouter price list showed kimi-k3 ($3/$15 per M) was most of the OpenRouter cost; no deepseek v4.1 pro exists, only v4.1-flash
- `d4` — Run the two baselines locally once on the issue-53 test set (198) and its missing-candidate slice (83) with the same measure.py command shape as the approved a3-heal run: stock Qwen3.5-0.8B (Track A) and scorer-b1 (Track B); no held-out runs; no API cost — Operator decision 2026-09-26: neither baseline had saved issue-53 test predictions (scorer-b1's are on the issue-46 test, whose ids are in issue-53's training split), so the gate would have no baseline rows
- `d5` — Add Discord webhook alerts to the unattended driver (extends deviation d2): every 10% of each provider's calls answered, every whole dollar of total spend, each provider/model stop, and the run's end (complete or stop-and-ask); counts, dollars, names and stop reasons only, never case text; webhook URL read from `NVSH_EVALS_ALERT_WEBHOOK` (grant-injected), sent milestones recorded in the run dir so restarts never repeat them — Operator request 2026-09-26: know the unattended run's progress, spend (so each dollar used is visible) and finish state without watching the terminal
- `d6` — Extend deviation d5's alerts with a status summary every 30 minutes (per provider: percent answered, calls to go, invalid, spend of cap; plus active stops), posted by a heartbeat thread in the driver so it keeps flowing during long steps — Operator request 2026-09-26: a 30-minute cadence status update

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t11` (`d1`) | The candidates (a3-heal) were measured by measure.py through LfmTier's multi-round loop; single-turn reference requests score an inspect-first model as malformed, breaking c33/h23 (same request contract incl. ground snapshot). Operator approved 2026-09-26: batch per round; cost ~$40-50 per fresh run batched (was ~$27), wall clock hours to days | `acceptable` |
| `t17` (`d2`) | Operator request 2026-09-26: a full run spans hours to days (per-round batching, d1) and must not depend on this Claude session; operator chose docker compose, grant passthrough for keys, wait-and-recheck on stops | `acceptable` |
| `t6` (`d3`) | Operator request 2026-09-26: smaller OpenRouter models to fit $20; live OpenRouter price list showed kimi-k3 ($3/$15 per M) was most of the OpenRouter cost; no deepseek v4.1 pro exists, only v4.1-flash | `acceptable` |
| `t20` (`d4`) | Operator decision 2026-09-26: neither baseline had saved issue-53 test predictions (scorer-b1's are on the issue-46 test, whose ids are in issue-53's training split), so the gate would have no baseline rows | `acceptable` |
| `t17` (`d5`) | Operator request 2026-09-26: know the unattended run's progress, spend (so each dollar used is visible) and finish state without watching the terminal | `acceptable` |
| `t17` (`d6`) | Operator request 2026-09-26: a 30-minute cadence status update | `acceptable` |
| `t24` | not complete at the time of writing; the run is resumable and continues unattended | `needs-follow-up` |
| `t24` | build.nvidia.com timed out repeatedly at the default 60 s, idling four NVIDIA models; the private manifest's `[budget.nvidia]` gained `timeout_seconds = 180` mid-run (config only, recorded in the run's manifest history; no call re-keyed, free provider) | `acceptable` |

## Evidence

- tests: `uv run pytest -c evals/pytest.ini --rootdir=. -q` at `f71c765` — pass (784)
- tests: `uv run pytest -n auto -q` at `f71c765` — pass (4710, 55 skipped)
- tests: `evals/tool_jev/tests/test_scaffold.py::test_wheel_contains_no_evals_files`, `::test_root_suite_collects_zero_evals_tests`, `test_run.py::test_interrupt_then_continue_gives_identical_outputs_and_sends_nothing_twice`, `test_readme.py::test_fixture_run_commands_reach_a_real_run_outside_the_repo` — pass
- replay: the candidates' and baselines' saved outputs scored through the gate's own path (`deepeval_layer.evaluate_traces`, raw policy) on a copy of the live run dir reproduce the recorded figures exactly (a3-heal test 77/83 right, 71/79 escalated, 4 wrong mutating; scorer-r3b 79/83, 75/79, 0; scorer-b1 68/83, 1; stock 6/83, 86 invalid) — pass
- validate-delivery records: obligations `o1`-`o6`, evidence `e1`-`e5` (all proposed; `o6`, the full run's output, has no evidence yet)
- commits: `d8416bc..f71c765` (88 commits on `spec/deepeval-release-gate-issue-64`)
- PRs / issues: #64 (this work), #66 (redaction, separate PR)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| the gate never ships in or is imported by the nvsh runtime | high | test `test_scaffold.py::test_wheel_contains_no_evals_files` · `e1` |
| the eval suite is isolated from the root suite and green | high | 784 / 4710 passed at `f71c765` · `e2` |
| runs resume without re-sending or re-billing a call | high | test `test_run.py::test_interrupt_then_continue_gives_identical_outputs_and_sends_nothing_twice` · live redeploys of run `8b59d0afa150` · `e3` |
| one command produces result.json and the comparison page | medium | fixture only: `test_readme.py::test_fixture_run_commands_reach_a_real_run_outside_the_repo` · `e4`; not yet on the full run |
| the gate reproduces the candidates' recorded figures | medium | scratch replay on a run-dir copy (`e5`, observation, not a committed test) |
| the full gate result (all 16 references, judge panel, benchmark page) | unverified | t24 in flight — not claimed done |
| release verdict for a3-heal.`q4_k_m` and scorer-r3b.`q4_k_m` | unverified | depends on t24 — not claimed |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `provenance-missing` | Scope claim c4 said a3-heal has saved held-out and missing-candidate predictions, relying on an explorer's inventory; the challenge probe found only final (test) predictions for a3-heal and a3-heal.`q4_k_m` |
| `l5` | `control-absent` | commit 4a55743 landed with one evals test red: the gate command piped pytest into tail, so the commit ran on tail's exit status, not pytest's; fixed in a146ab6, and the gate now checks pytest's own exit code |

pending approval (not yet evidence): `l2`, `l3`, `l4`

## Remaining Work / Follow-up

- `t24` — let the unattended run finish (Discord `COMPLETE` or `result.json`), then re-validate `o6` and update this file — main agent
- `t24` — copy the aggregate report page (no case text) into `docs/benchmarks/` — main agent
- `t25` — follow-up issues (wire the released model into the tier stack, issue-54 style; an execution eval), then the PR "part of #64" via `cicd` — main agent
- adjudicate proposed lapses `l2`-`l4` and proposed records `o1`-`o6`, `e1`-`e5` — operator
- rotate the OpenAI and build.nvidia.com keys after the run — operator
- interim observation to check on the full result: through the Track A loop (`d1`) the frontier references mostly answer `explain` (e.g. 131 of 198 for claude-opus-5-5) and score 5-13/83 right, while on Track B (choice) they score 78-82/83 right; decide whether that is model behaviour or a Track A contract effect before reading Track A rows as a verdict — main agent + operator
