# DeepEval release gate for Tool-Jev (issue 64)

This guide describes the DeepEval evaluation layer that decides whether a
Tool-Jev checkpoint is released. It covers what is built, how the parts fit,
the decisions behind them, and what is still missing. It is kept current
while the work is in progress; the live handoff is
[`deepeval-gate-handoff.md`](deepeval-gate-handoff.md).

- Spec: [`specs/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md`](specs/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md)
- Plan: [`plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md`](plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md)
  and its approved split
  [`plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64-split.md`](plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64-split.md)
- Issue: [#64](https://github.com/agentculture/nvsh/issues/64)

**Status: in progress.** The library under `evals/tool_jev/` (including the
multi-round Track A loop) is built and tested with fixtures only. The runner, the autonomous driver, the smoke run
and the full gate run are not built or run yet, so no gate result exists.
Do not describe the gate as producing results until the runner lands.

## What the gate answers

For each release candidate it separates two questions, as issue 64 asks:

- **Did the model improve?** The model-only row: the raw decision and its
  full candidate distribution, scored with exact metrics.
- **Did the harness prevent the model's mistakes?** The model+harness rows:
  the same raw outputs passed through named harness policies (calibration,
  read-only and mutating gates), scored the same way.

Frontier and open models answer the same cases as reference rows. An LLM
judge panel scores explain text only, and never feeds a release bar.

## Subjects

| Role | Subjects |
| --- | --- |
| Release candidates | `a3-heal.q4_k_m` (Track A, issue 46), `scorer-r3b.q4_k_m` (Track B, issue 53) |
| Baselines | stock Qwen3.5-0.8B, `scorer-b1` (issue 46) |
| References, OpenAI | `gpt-6-luna`, `gpt-6-sol` |
| References, Anthropic | `claude-opus-5-5`, `claude-sonnet-5` |
| References, OpenRouter | `qwen/qwen3.8-max-0902`, `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro-0813`, `moonshotai/kimi-k3` |
| References, build.nvidia.com | `moonshotai/kimi-k3`, `z-ai/glm-5.3`, `nvidia/nemotron-3-ultra-550b-a55b`, `nvidia/nemotron-3-super-120b-a12b`, `google/gemma-4-31b-it` |
| References, local | `unsloth/Qwen3.8-27B-NVFP4`, `nvidia/Gemma-4-26B-A4B-NVFP4` (the lobes gateway on localhost; 4-bit NVFP4 builds) |
| Judge panel (explain text) | `claude-opus-5-5`, `gpt-6-sol`, `moonshotai/kimi-k3` (build.nvidia.com), `nvidia/nemotron-3-ultra-550b-a55b`, `qwen/qwen3.8-max-0902` |

`kimi-k3` is deliberately on two hosts, which shows host-to-host variance.

## Case sets and privacy

- Comparisons between the candidates and against the references use the
  **issue-53 test set (198 cases) and its missing-candidate slice (83)**.
  The two candidates were measured on disjoint test sets, and all 64
  issue-46 test cases are in the issue-53 training split, so r3b is never
  scored on issue-46 cases.
- `a3-heal.q4_k_m` had one approved new run on the issue-53 test set and
  its slice (decision c38): see
  [`benchmarks/2026-09-26-lfm-final-a3-heal.q4_k_m.md`](benchmarks/2026-09-26-lfm-final-a3-heal.q4_k_m.md)
  and
  [`benchmarks/2026-09-26-lfm-final-a3-heal.q4_k_m-missing-candidate.md`](benchmarks/2026-09-26-lfm-final-a3-heal.q4_k_m-missing-candidate.md).
  No other candidate re-runs are allowed without a recorded deviation.
- **The sealed held-out sets never leave the machine.** Reference models
  see the test side only. Held-out figures come only from the saved
  final-run prediction files, behind an explicit flag, and the provider
  layer refuses any held-out case before a network call.
- **Case text never enters git.** Per-case traces, provider responses and
  the response cache live in a private run directory outside the
  repository. The repository receives only the aggregate page, policy
  configs, the example manifest and synthetic test fixtures.
- Every text that leaves the process is redacted by `nvsh.redact` through
  one choke point in `providers/base.py`. Issue
  [#66](https://github.com/agentculture/nvsh/issues/66) tracks a gap in that
  redactor (an env assignment in the middle of a line is not redacted).

## Layout

The suite lives in `evals/`, outside the root `tests/` testpaths, in its own
uv dependency group (`deepeval==4.2.6`). Nothing under `nvsh/` imports it,
and the wheel does not ship it.

| Module | Purpose |
| --- | --- |
| `evals/tool_jev/__init__.py` | Env guard: forces `DEEPEVAL_TELEMETRY_OPT_OUT=1` and `DEEPEVAL_DISABLE_DOTENV=1`, refuses any other value, refuses `CONFIDENT_API_KEY` |
| `cases.py` | Case model, split tags, held-out guard (no held-out text ever returned), training-overlap refusal |
| `manifest.py` + `manifest.example.toml` | Run manifest: candidates, baselines, references, judges, case sets (split + private-root-relative path), budgets |
| `trace.py` | Raw record and per-policy final records side by side; writer refuses paths inside a git worktree; keeps candidate order |
| `metrics_bridge.py` | Reuses `scripts/lfm-finetune/metrics.py` and `gate.py`; `not_measurable` when no distribution |
| `policies.py` + `policies/*.json` | Harness policies applied offline: `raw`, `scorer-r3b-shipped` (T=1.5366, read-only margin 0.2), a stricter mutating example |
| `ledger.py` | Durable call ledger and response cache: atomic writes, resume after a stop or reset, batch re-attach by submit ref |
| `track_a_loop.py` | Deviation d1: replays each case through the candidates' `LfmTier` loop with a deferred chat client; one ledger call per uncached round |
| `request.py` | Request contract from the candidates' own prompt code (Track A tool call, Track B choice), `canonical_content`, parsers |
| `providers/base.py`, `errors.py`, `fake.py` | Provider protocol, redaction + held-out choke point, error taxonomy, scriptable fake |
| `providers/openai.py` | OpenAI Responses API, sync + Batch API |
| `providers/anthropic.py` | Anthropic Messages, sync + Message Batches |
| `providers/openai_compat.py` | OpenRouter, build.nvidia.com and local servers (sync, logprobs, rate limiter) |
| `deepeval_layer.py` | DeepEval test cases and exact metrics graded per policy on the policy-applied prediction; deepeval state kept in the run dir |
| `judge.py` + `rubric/explain-v1.md` | Blind all-to-all judge panel, G-Eval with fixed steps, two-pass record/replay |
| `permutation.py` | Permutation-stability entries (r3b's saved probe; a3-heal not measurable by the probe) |
| `report.py` | `result.json` and the markdown page with issue 64's table; no case text |

Run the suite's own tests (CI runs them with no secrets):

```bash
uv sync --group evals
uv run pytest -c evals/pytest.ini --rootdir=. -q   # --rootdir=. is required
```

## Key contracts

- **Request contract.** A `CallRequest` carries the system text in `prompt`
  and the only case text in `case_text`; tools and labels go in `params`.
  Adapters build payloads only through `request.canonical_content`, so the
  content is byte-identical across providers.
- **Answer contract.** For a tool call every adapter returns the raw first
  call as JSON `{name, arguments}`; `request.parse_tool_call` interprets it.
  For a choice answer the text is parsed by `request.parse_choice`; a
  distribution exists only when the provider returned token logprobs for the
  first answer token.
- **Outcomes.** A bad model answer (structural refusal, malformed or empty,
  outside the offered set) is `invalid` and counts. An infrastructure stop
  (402, quota, budget cap, 429, timeout, network, reset, expired batch) stays
  `pending` and retryable. A rejected request (400, 401, 403, 404, 422,
  unsupported parameter) stays `pending` but is not retryable until the
  manifest or parameters change.
- **Resume.** The ledger writes the cache before the state change, uses
  write-then-rename with fsync, and re-attaches submitted batches by submit
  ref instead of resubmitting.

## Decisions and deviations

- References are comparison subjects scored by the same exact metrics; the
  judge panel scores explain text only, blind, all-to-all, with self-scores
  excluded from each subject's panel score.
- OpenAI and Anthropic calls use their Batch APIs; every run can stop and
  continue for every provider; reasoning effort is medium.
- **d1 (approved):** reference models' Track A runs through the candidates'
  own multi-round `LfmTier` loop (up to 4 rounds, read-only inspections
  against the recorded ground snapshot), batched one round at a time.
- **d2 (approved):** an autonomous docker compose driver runs the gate so it
  survives the session and reboots; keys by `grant run --inject` passthrough,
  no key file on disk; on a money stop it re-checks that provider every
  30 minutes while the others continue.

## Cost

One fresh full run at medium reasoning was estimated at about \$44 without
batching and about \$27 with OpenAI and Anthropic batching, before d1.
d1's multi-round Track A raises that to roughly \$40-50 batched.
build.nvidia.com's hosted catalog is free (rate-limited). Cached reruns cost
nothing. The planned 10-case smoke run replaces these estimates with
measured token counts before the budget caps are set.

## Reproducing and extending

- A new checkpoint is one `[[candidate]]` table in the private manifest;
  the dataset does not change.
- The saved candidate outputs are replayed; no GPU is needed for them.
- The runner, the driver and the README with the full command sequence are
  still to be written (plan tasks t17 and t19, deviation d2).
