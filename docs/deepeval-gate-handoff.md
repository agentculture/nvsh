# Handoff: DeepEval release gate (issue 64)

State at the end of the 2026-09-26 session. The design guide is
[`deepeval-gate.md`](deepeval-gate.md) and the operator guide is
[`../evals/README.md`](../evals/README.md); this file is only what the next
session needs to pick up.

## Where the work is

- Branch `spec/deepeval-release-gate-issue-64`, local only (not pushed, no
  PR). Version already bumped to 0.21.0 with its CHANGELOG entry.
- devague frame and plan slug: `deepeval-release-gate-for-tool-jev-issue-64`
  (`devague plan status`, `devague deviate --list`, `devague lapse --list`).
- Tests: evals suite 784 passing
  (`uv run pytest -c evals/pytest.ini --rootdir=. -q`); root suite 4710
  passing, 55 skipped; lint, secret scan and harness smoke clean.

## Done

| Task | What |
| --- | --- |
| t1 t2 t5-t16 t18 t20 t21 t23 | Library, adapters, DeepEval layer, judge panel, report, permutation entries, the a3-heal run |
| d1 | References' Track A through the candidates' `LfmTier` loop; codex-reviewed; plan risk r9 (prose replies) fixed |
| t17 | Runner (`run`, `continue`, `status`, `smoke`, `drive`); two codex review rounds fixed (billing, reservations, stops, locks) |
| d2 | Docker compose driver `evals/docker/` (`drive --start --idle-when-done`) |
| t19 | `evals/README.md` with a no-network fixture run checked by a test |
| d4 | Baselines stock Qwen3.5-0.8B and `scorer-b1.q4_k_m` measured on the issue-53 test set and slice |
| t22 | Smoke run: 0 truncated and 0 invalid at medium reasoning, 2048 output tokens; r10 resolved |
| d5 d6 | Discord alerts: 10% progress per provider, each whole dollar, stops, the end, a 30-minute summary |
| t25 (part) | Version 0.21.0, CHANGELOG, harness prompt files and guide updated |

Fixes the live runs found, each with a regression test: Anthropic
`custom_id` too long for real case ids (hash token), a restricted OpenAI key
missing `api.files.write` (reported as `missing_scope`), a 60 s timeout too
short for the local 27B model (`timeout_seconds`), transient pauses that
lasted a whole pass, and one shared sync thread pool that let the slow
local provider starve OpenRouter and build.nvidia.com.

## In flight at handoff

- **The full gate run (t24)** runs unattended in the docker compose project
  `nvsh-evals-gate` (`docker ps`, `docker compose -p nvsh-evals-gate logs`).
  Its run directory is the private `runs/smoke-1` (the smoke run expanded to
  full with `run --expand`, so the smoke answers are reused). Check it with
  `status --run-dir <that dir>`; it posts to the operator's Discord.
- **A local-only rehearsal** runs in `nvsh-evals-rehearsal` (private
  `runs/rehearsal-local`, local references and judges only, free). It is a
  dress rehearsal of the report path; its result is not the gate result.
- The private data root, manifests and run directories live outside the
  repository in the operator's `lfm-train` work tree under `i64-gate/`; the
  session memory note names the exact paths.

## Next, in order

1. Wait for the full run's `COMPLETE` (Discord, or `result.json` in the run
   dir). If it stops to ask (an unresolvable batch lookup, a torn billing
   line), read `drive.log` and ask the operator before anything is resent.
2. `/validate-delivery` against the plan, with the full run's result.
3. `/summarize-delivery`, which writes `docs/deliveries/<date>-<slug>.md` and
   quotes deviations d1-d6 and lapses l1-l5.
4. Copy the aggregate report page (no case text) into `docs/benchmarks/`.
5. The rest of t25: follow-up issues (wiring the gate into the tier stack in
   the issue-54 style; an execution eval), then the PR "part of #64" through
   the `cicd` skill.
6. Stop both compose projects (`docker compose -p <project> down`) once the
   result is committed.

## Open items

- Plan risks open: r5 (build.nvidia.com free-tier rate limit, about 36
  requests a minute, about 2800 calls to send) and r6 (an in-progress
  Anthropic batch cannot be matched to its submit ref: stop and ask, never
  resubmit). All others are resolved.
- Lapses: l1-l4 proposed, l5 approved by the operator.
- Keys: the OpenAI key (restricted, with Files and Batch scopes, expires
  about 2026-10-26) and the build.nvidia.com key were pasted into the chat
  transcript; the operator rotates both when the run is done. Both, and the
  Discord webhook, are stored hidden in `grant`.
- Issue [#66](https://github.com/agentculture/nvsh/issues/66) (mid-line env
  assignment not redacted) is fixed in its own PR, not this one.
- Early read from the saved runs (not the gate verdict): `scorer-r3b.q4_k_m`
  makes no wrong mutating proposal on the test set or the slice;
  `a3-heal.q4_k_m` is 77/83 right on the test set but proposes instead of
  escalating on the missing-candidate slice (11 wrong mutating).

## Rules that bit this run

- Gate a commit on pytest's own exit code, never on a pipe's (`| tail`
  swallowed a red test once, lapse l5).
- Never wait with `pgrep -f "<pattern>"` when the pattern is in the waiting
  command's own text: it matches itself and never ends. Wait on a PID.
- Anthropic `custom_id` is at most 64 characters; OpenAI restricted keys need
  the Files and Batch scopes for the Batch API.
- Evals pytest needs `--rootdir=.`; scan-secrets scans only tracked files
  (run it after `git add`, and never write secret-shaped literals in tests).
- Anything that imports deepeval must not run with the repository as its
  working directory.
- Never print or probe a key's value: read keys only inside the process.
