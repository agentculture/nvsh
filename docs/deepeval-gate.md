# DeepEval release gate for Tool-Jev (issue 64)

This guide describes the DeepEval evaluation layer that decides whether a
Tool-Jev checkpoint is released. It covers what is built, how the parts fit,
the decisions behind them, and what is still missing. The handoff for the next session is
[`deepeval-gate-handoff.md`](deepeval-gate-handoff.md).

- Spec: [`specs/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md`](specs/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md)
- Plan: [`plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md`](plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64.md)
  and its approved split
  [`plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64-split.md`](plans/2026-09-26-deepeval-release-gate-for-tool-jev-issue-64-split.md)
- Issue: [#64](https://github.com/agentculture/nvsh/issues/64)

**Status: first full run complete.** The library, the runner
(`python -m evals.tool_jev run|continue|status|smoke|drive`), the autonomous
docker compose driver and the operator guide [`evals/README.md`](../evals/README.md)
are built. The first full gate run (t24, run `8b59d0afa150`) finished on
2026-09-29; its report is
[`benchmarks/2026-09-29-deepeval-gate-run-8b59d0afa150.md`](benchmarks/2026-09-29-deepeval-gate-run-8b59d0afa150.md)
and the headline is in [First result](#first-result) below.

## First result

Run `8b59d0afa150`, issue-53 test set (198 cases, 83 expect an operation, 79
expect an escalation) and its missing-candidate slice (83 cases, escalation
expected). 15 references (glm-5.3 dropped, deviation d7), judge panel
claude-opus-5-5, gpt-6-sol and qwen3.8-max (low reasoning as a judge,
deviation d8). Spend: Anthropic $17.41, OpenAI $3.36, OpenRouter $15.01,
build.nvidia.com and local free; $35.78 in all.

| Subject | Right | Escalation recall | Wrong mutating | Slice: escalation recall | Slice: wrong mutating |
| --- | --- | --- | --- | --- | --- |
| `scorer-r3b.q4_k_m` (shipped policy) | 79/83 | 94.9% | 0 | 85.5% | 0 |
| `a3-heal.q4_k_m` | 77/83 | 89.9% | 4 | 12.0% | 11 |
| `scorer-b1.q4_k_m` (baseline) | 68/83 | 79.7% | 1 | 27.7% | 0 |
| stock Qwen3.5-0.8B (baseline) | 6/83 | 3.8% | 5 | 3.6% | 4 |
| best references, choice interface | 81-82/83 | 72.2% at most | 0-3 | 65.1% at most | 0-1 |

- `scorer-r3b.q4_k_m` is the only subject with no wrong mutating proposal on
  both sets, and it escalates best of every subject; its calibration is the
  best measured (ECE 0.028 raw, 0.016 with the shipped policy).
- `a3-heal.q4_k_m` matches it on the test set but proposes instead of
  escalating when the right operation is not offered.
- References score 78-82/83 through the choice interface. Through the Track
  A loop they mostly run the expected read-only command as an inspection and
  explain its result, which the strict metric scores as a miss; the **Task
  done** column (deviation d9) credits that (claude-opus-5-5 5/83 strict,
  66/83 task done; gpt-6-luna 13 and 74). The candidates' saved predictions
  record no inspections, so their task done equals their strict score.
- The judge panel scored 1,581 reference explanations (mean pairwise
  disagreement 0.17 on a 0-1 scale); the candidates' saved predictions carry
  no explanation text, so their explanations are not judged. Judge scores
  never feed a release bar.

On this result the operator published `scorer-r3b.q4_k_m` and its bf16
checkpoint on the Hub (2026-09-29,
[`...-scorer-v2-gguf`](https://huggingface.co/jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev-scorer-v2-gguf),
[`...-scorer-v2`](https://huggingface.co/jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev-scorer-v2);
recorded on #60); `a3-heal.q4_k_m` stays private. Nothing is wired into the
tiers yet. Follow-ups:
[#69](https://github.com/agentculture/nvsh/issues/69) (wire
`scorer-r3b.q4_k_m` into the tier stack),
[#70](https://github.com/agentculture/nvsh/issues/70) (an execution-based
layer) and [#71](https://github.com/agentculture/nvsh/issues/71) (gaps this
run found).

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
| References, OpenRouter | `qwen/qwen3.8-max-0902`, `qwen/qwen3.8-27b`, `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro-0813`, `deepseek/deepseek-v4.1-flash` |
| References, build.nvidia.com | `moonshotai/kimi-k3`, `z-ai/glm-5.3`, `nvidia/nemotron-3-ultra-550b-a55b`, `nvidia/nemotron-3-super-120b-a12b`, `google/gemma-4-31b-it` |
| References, local | `unsloth/Qwen3.8-27B-NVFP4`, `nvidia/Gemma-4-26B-A4B-NVFP4` (the lobes gateway on localhost; 4-bit NVFP4 builds) |
| Judge panel (explain text) | `claude-opus-5-5`, `gpt-6-sol`, `moonshotai/kimi-k3` (build.nvidia.com), `nvidia/nemotron-3-ultra-550b-a55b`, `qwen/qwen3.8-max-0902` |

Deviation d3 moved OpenRouter to cheaper open models to fit its budget:
`qwen/qwen3.8-27b` and `deepseek/deepseek-v4.1-flash` were added, and
`moonshotai/kimi-k3` now runs on build.nvidia.com only (so the kimi
host-to-host comparison is dropped). `qwen/qwen3.8-max-0902` stays as a
reference and a judge, since every judge also answers.
`Qwen3.8-27B` still appears twice, hosted on OpenRouter and as a local
4-bit NVFP4 build, which shows hosted-versus-local variance.

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
| `run.py`, `runplan.py`, `runstate.py`, `runstatus.py`, `__main__.py` | The runner: plan, pass engine, durable run state and append-only billing, status, CLI |
| `drive.py`, `alerts.py`, `evals/docker/` | The unattended driver loop, Discord alerts, and its compose packaging |

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
- **d3 (approved):** OpenRouter runs cheaper open models to fit its budget:
  `qwen/qwen3.8-max-0902` (a reference and a judge), `qwen/qwen3.8-27b`,
  `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro-0813` and
  `deepseek/deepseek-v4.1-flash`; `moonshotai/kimi-k3` runs on build.nvidia.com
  only. 16 references.
- **d4 (approved):** the baselines (stock Qwen3.5-0.8B and
  `scorer-b1.q4_k_m`) were measured once on the issue-53 test set and its
  missing-candidate slice, since neither had predictions there.
- **d5 and d6 (approved):** the driver posts Discord alerts: every 10% of each
  provider's calls, every whole dollar of spend, each stop, the run's end, and
  a status summary every 30 minutes. The webhook is a secret passed by name.
- **d7 (approved):** to close the run without build.nvidia.com's slow free
  tier (about 110 calls an hour), `z-ai/glm-5.3` left the references and
  kimi-k3 and nemotron-ultra left the judge panel (both stay references).
- **d8 (approved):** a `[[judge]]` entry may set its own reasoning effort;
  qwen3.8-max judges at low, because at medium it truncated 19% of its judge
  replies at the 1024-token cap (2% at low). Only its judge calls re-keyed.
- **d9 (approved):** the main table has **Right** (strict proposal) and
  **Task done**: a read-only-expected case also counts when the subject ran
  the expected operation with the expected arguments as a loop inspection
  and then explained. Track A records carry the inspections they ran.
- **Risk r10 (resolved):** the smoke run saw no truncated reply at medium
  reasoning with a 2048-token output budget, so no model was replaced.

## Cost

The 10-case smoke run (2026-09-26, medium reasoning, 2048-token output
budget) measured the reference calls at about \$0.25 on Anthropic (batched)
and \$0.22 on OpenRouter, which projects to about \$6.9 and \$6.0 for a full
fresh run of the references, plus the judges. OpenAI is batched at half price;
build.nvidia.com's hosted catalog and the local models cost nothing. The
private manifest caps each provider at about 1.5 times its projection, and
the runner reserves each call's worst-case cost before sending it, so a cap
is never overshot. Cached reruns cost nothing.

The first full run cost \$35.78: Anthropic \$17.41 (references \$6.75, the
opus judge the rest), OpenAI \$3.36 (gpt-6-sol judged all 1,581 explanations
for about \$2.46), OpenRouter \$15.01 (qwen3.8-max as a judge about \$9.5).
The judge phase was the larger part; raising the Anthropic and OpenRouter caps
to \$20 and \$16 finished it.

## Reproducing and extending

- A new checkpoint is one `[[candidate]]` table in the private manifest;
  the dataset does not change.
- The saved candidate outputs are replayed; no GPU is needed for them.
- The command sequence, keys, stops and costs are in
  [`evals/README.md`](../evals/README.md).
