# DeepEval release gate for Tool-Jev (issue 64)

> nvsh has a repeatable DeepEval release gate for its Tool-Jev models: one command scores any fine-tuned checkpoint and reference models (frontier models via the OpenAI and Anthropic platform APIs, open models via OpenRouter and build.nvidia.com, and local models) on the same fixed cases, model-only and through explicit nvsh harness policies, so a3-heal and scorer-r3b can be judged for release and wiring into the tier stack, and the next fine-tune is judged the same way
> instruction: evals/`tool_jev` manifest lists a3-heal.`q4_k_m` and scorer-r3b.`q4_k_m` as candidates and stock, scorer-b1 as baselines

## Audience

- The nvsh operator deciding whether a Tool-Jev checkpoint is released publicly and wired into the tier stack, and the agent running the next fine-tune cycle who must judge a new checkpoint the same way
  - instruction: evals/`tool_jev`/README.md names both readers and gives each one command

## Before → After

- Before: Today each cycle's comparison page is assembled by hand from chained measure.py, `calibration_fit` apply and `sweep_gate` --final outputs; raw and gated results live in separate files; no reference model has ever answered the Tool-Jev cases; there is no DeepEval anywhere in the repo
  - instruction: cite docs/benchmarks/2026-09-25-tool-jev-calibration-cycle.md and grep -ri deepeval returning nothing
- After: One documented command replays the saved outputs of a3-heal, scorer-r3b and baselines plus fresh reference-model runs, and writes a JSON result, per-case traces with raw and final side by side, and a markdown comparison page; adding a new checkpoint is one manifest entry
  - instruction: run the command from a clean checkout against the fixture manifest and against the real manifest; both write the three artifacts

## Why it matters

- Issue 53 D49 makes DeepEval (issue 64) the operator's final gate before any Tool-Jev repository goes public, and the gate has to separate 'the model improved' from 'the harness prevented the model's mistakes' and place a 0.8B model against frontier and open models
  - instruction: quote issue 53 delivery claim table ('DeepEval evaluation (#64) is the operator's final gate') in the spec

## Requirements

- Reference providers are the OpenAI API and the Anthropic API on their own platforms (not Bedrock), OpenRouter, build.nvidia.com, and local OpenAI-compatible model servers
  - instruction: one provider adapter module per API with keys read from env (injected by grant run --inject); OpenRouter key name `OPEN_ROUTER_API_KEY`
  - honesty: Each provider is reached through its own documented endpoint (api.openai.com, api.anthropic.com, openrouter.ai, integrate.api.nvidia.com, a localhost OpenAI-compatible server) and a run records which provider and model id answered every case
- The suite has two recorded layers per case: raw = the model's decision and full candidate distribution as measured, final = an explicit harness policy applied to that raw record; both are persisted side by side in one per-case trace so 'model wrong, harness abstained' and 'model right, harness overrode' are distinguishable (issue 64)
  - instruction: trace JSONL: {case, raw:{...}, final:{`policy_name`:{decision, reason}}, `ground_truth`:{...}}
  - honesty: Every per-case trace holds both the raw record (outcome, full candidates distribution or null) and one final record per harness policy, and the raw record is never modified by applying a policy
- Deterministic metrics are reused from the existing tooling, not reimplemented: metrics.py (top-1, Brier, 10-bin ECE, per-slice, abstain P/R, `is_missing_candidate`), gate.py (normalized entropy, top-1/top-2 margin, `read_only` vs mutating ThresholdSets), `calibration_fit.py` and `permutation_probe.py`; entropy and margin get surfaced as raw-quality metrics, not only as gate inputs
  - instruction: import scripts/lfm-finetune modules by path (as tests do) rather than copying code; add entropy/margin to the raw report
  - honesty: Every numeric metric the gate reports for an existing candidate equals what the existing scripts/lfm-finetune function computes on the same file; no parallel reimplementation diverges
- DeepEval is the per-case harness and exporter (custom BaseMetric per exact check, LLMTestCase/Golden per case, local JSON export via `DEEPEVAL_RESULTS_FOLDER`); corpus-level numbers (ECE, Brier, coverage, abstain P/R, permutation change rate) are computed from the per-case records outside DeepEval's metric model, because DeepEval metrics are per-test-case only
  - instruction: custom BaseMetric per exact check; corpus metrics computed from collected per-case records; `DEEPEVAL_RESULTS_FOLDER` set to the run's output dir
  - honesty: The gate runs with `DEEPEVAL_TELEMETRY_OPT_OUT`=1 set before import, no `CONFIDENT_API_KEY`, and makes no network call other than to the configured reference providers
- Harness policies are named, versioned configs (calibration temperature, escalate floor, `read_only` and mutating floor/margin/`max_entropy`, allowed-operation set) applied to the same raw outputs; at least the raw policy and the shipped scorer-r3b policy (T=1.5366, `read_only` margin 0.2) are compared, and read-only vs mutating gates report separately
  - instruction: policies/\*.json with gate.py Thresholds plus calibration params; raw policy = identity
  - honesty: Applying a policy to saved outputs needs no model call, and the shipped r3b policy (temperature 1.5366, `read_only` margin 0.2) reproduces the recorded gated test and held-out figures
- Reference models (frontier and open) answer the same cases through one provider adapter layer and are scored by the same deterministic metrics; where a provider returns no per-candidate probabilities (e.g. no logprobs), calibration metrics are reported as not measurable rather than reconstructed, and Track A generative rows are labelled native vs teacher-forced per issue 64
  - instruction: OpenAI and OpenRouter/NVIDIA may return top logprobs; Anthropic returns none; record capability per provider
  - honesty: A provider that returns no token logprobs yields rows whose calibration metrics are marked not measurable, never estimated from sampling or reconstruction
- The suite lives in a top-level evals/ tree outside pytest testpaths, with deepeval in its own dependency group or requirements file; the main uv run pytest suite is unaffected (deepeval's auto-loaded pytest11 plugin is disabled wherever deepeval is installed alongside it); `DEEPEVAL_TELEMETRY_OPT_OUT`=1 is set before import and `CONFIDENT_API_KEY` is never used, so results stay local
  - instruction: evals/ outside testpaths; deepeval in a separate dependency group not synced by default, or pytest -p no:plugins
  - honesty: uv run pytest -n auto in a synced checkout collects no evals/ tests and loads no deepeval plugin
- CI runs the eval suite's own unit tests on small committed fixture predictions (no GPU, no model, no API keys) and lints evals/ like scripts/; live runs against checkpoints and provider APIs are operator-run locally
  - instruction: tests.yml: add evals to black/isort/flake8 path lists and a job running the evals unit tests on fixtures
  - honesty: The evals unit tests pass on ubuntu-latest CI with no secrets configured, and black/isort/flake8 run over evals/
- Provider credentials come only from environment variables or grant injection at run time, never from committed files; cases are redacted with nvsh/redact.py before any leave the machine; provider endpoints are code defaults or operator config outside the repo, never committed JSON with non-localhost URLs
  - instruction: run python3 scripts/scan-secrets.py; unit test that the provider layer calls redact before sending
  - honesty: No provider key or non-localhost endpoint appears in any tracked file (scan-secrets passes), and every case text sent to a provider passed through nvsh/redact.py first
- Per-case traces, raw provider responses and the response cache contain private case text (the issue-46/53 test sets are not in git) and stay outside the repository in a private run directory; git gets only the aggregate comparison page, policy configs, the manifest without private paths, and synthetic fixtures for CI
  - instruction: scan-secrets plus a unit test that the page writer emits no case text; fixtures are generated, never copied from real splits
  - honesty: No committed file contains a case request text or id-to-text mapping from the issue-46/53 splits
- Reference models answer through the same request contract the candidates were measured with (the measure.py prompt, offered candidate list and ground snapshot), adapted per provider only in transport; the interface used (tool call vs candidate choice with logprobs) is recorded per row
  - instruction: reuse measure.py's prompt composition by import; a unit test pins the prompt bytes for a fixture case across providers
  - honesty: For a fixture case, every provider adapter sends identical system and user content (modulo transport framing)
- Every reference call records the provider, requested model id, provider-returned model id/version, request parameters and response id; raw responses are cached by (provider, model, case, prompt hash) so a rerun replays the cache byte-identically and a fresh call is always a new dated run; parameters a model rejects (temperature, logprobs on reasoning models) are recorded in a per-model capability entry
  - instruction: cache-hit rerun test with a fake provider; capability matrix in the manifest
  - honesty: A rerun with a warm cache makes zero network calls and produces identical JSON
- Provider runs are bounded and resumable: a per-run call budget and concurrency cap, retries with backoff, and every timeout, refusal, malformed or empty answer is recorded per case as invalid and counted in the denominator, never dropped
  - instruction: fake-provider tests for timeout, 429, refusal and malformed JSON; the page shows invalid counts per model
  - honesty: Every model row's case count equals the case set size; invalid answers appear as their own count

## Honesty conditions

- The command runs end to end without a GPU for the replay part, and each release candidate row is labelled with its exact artifact (repo id and revision or local file hash)
- No release bar in the gate's output depends on a judge score; judge scores appear in a separate column or section marked as judge-scored
- uv build produces a wheel with no evals/ files and no new Requires-Dist, and grep finds no deepeval or evals import under nvsh/
- The markdown page and JSON are regenerated from one run's outputs with no hand-edited numbers
- The gate never opens the sealed held-out files itself; held-out figures come only from the saved final-run prediction files, and a flag is required to include them
- The provider layer refuses to send any case whose source split is held-out, checked by split tag before any network call
- Both readers can run the gate from the README alone, without reading the fine-tune guides
- The three artifacts exist after one command and a second checkpoint can be added by one manifest entry without editing code or dataset
- The before-state facts are checkable in the repo at the base commit (grep deepeval empty; benchmark page built from chained tools)
- The gate's output answers the release question for each candidate separately for model-only and model+harness
- The reproduced figures match the recorded benchmark pages exactly, or every difference is explained and filed
- The gate refuses to score a checkpoint on a case id that appears in that checkpoint's training split, checked from split files before scoring
- No code path in evals/ spawns a process with model-returned content
- The run record lists every host that received case text

## Success signals

- One command produces a machine-readable result and a markdown comparison page with issue 64's table (variant, harness policy, top-1, ECE, Brier, coverage, abstain P/R, missing-candidate, wrong mutations) plus slice reports and per-case traces; the next fine-tune is evaluated by adding one manifest entry, without changing the dataset
  - instruction: docs/benchmarks/`<date>`-tool-jev-deepeval-gate.md generated by the command
- On the release candidates the gate reproduces the recorded final figures from the saved outputs exactly (scorer-r3b.`q4_k_m` test 79/83 right, 0 wrong mutating, held-out missing-candidate escalation 76.7%; a3-heal.`q4_k_m` test 32/32 right, 2 wrong mutating), compares at least 2 harness policies and at least 4 reference models on the same test cases, and a rerun with the same inputs produces byte-identical deterministic results
  - instruction: diff the gate's candidate rows against docs/benchmarks figures; run twice and diff the JSON (excluding timestamps); reference runs are cached so a rerun does not call providers again

## Scope / boundaries

- An LLM judge never replaces exact action, safety, calibration or abstention measurements; any judge-scored metric (if used at all) is limited to outputs with no exact ground truth, such as explain text, and is reported separately from the release bars
  - instruction: judge metric only on explain-kind cases, separate report section
- Nothing under nvsh/ imports deepeval, the eval suite or scripts/lfm-finetune; the runtime keeps dependencies = \[\] and deepeval never ships in the wheel or as a PyPI extra
  - instruction: check with uv build + unzip -l and grep -rn deepeval nvsh/
- The sealed held-out sets are never read by the development side, and the release candidates are not re-run on test or held-out beyond the one approved a3-heal run on the issue-53 test (c38): the gate replays saved final-run outputs; any other new inference by a release candidate on test or held-out is a recorded deviation
  - instruction: held-out predictions are read from saved files only when --include-heldout is passed
- Reference models run on the test side only (and its missing-candidate slice); the sealed held-out sets never leave the machine
  - instruction: provider layer asserts case split in {test, test-mc}; unit test for refusal
- The two candidates have no shared measured cases: a3-heal was measured on the issue-46 test (64) and r3b on the issue-53 test (198), zero shared ids, and all 64 issue-46 test ids are in the issue-53 training split (d4), so r3b is never scored on issue-46 cases and any a3-heal vs r3b comparison uses the issue-53 test
  - instruction: id-overlap probe over the saved prediction files and q53-run-d7 splits (train 64/64 overlap, val 0, test 0)
- The gate executes nothing: operations and arguments a model returns are untrusted data that are compared and rendered as text only, never passed to a shell, nvsh.ops.render execution, or any subprocess
  - instruction: grep evals/ for subprocess/os.system; a test that a hostile argument string is stored verbatim

## Non-goals

- Execution-based evaluation (running the selected operation in an isolated shell or container and checking machine state, Harbor/Terminal-Bench style) is a later layer, not this frame
- Making any Hugging Face repo public is not done by this work; the gate informs the operator's per-repository visibility decision (issue 46 c48, issue 60)
- Wiring the released model into nvsh's runtime tiers (Verifier seam, issue 54) is a follow-up spec after the gate passes, not this work

## Assumptions

- The first gate run replays already-saved per-case outputs with no GPU for: stock, a3, scorer-b1 (issue 46 test, missing-candidate, sealed held-out) and scorer-r3b bf16 and `Q4_K_M` (issue 53 test, missing-candidate, held-out, held-out missing-candidate, raw/calibrated/gated); the shipped a3-heal and a3-heal.`q4_k_m` have saved issue-46 TEST predictions only
- Permutation stability for the release candidates needs new inference (no permutation outputs are saved); it runs on the test side only via `permutation_probe.py`, as issue 53 did, never on the sealed held-out
- Sending the issue-53 test side (and missing-candidate slice) to OpenAI, Anthropic, OpenRouter (which forwards to third-party hosts) and NVIDIA is acceptable exposure: the set is already spent for final-run claims, and provider retention is accepted for it
  - instruction: state provider data-retention settings used (e.g. OpenRouter data-collection off) in the run record

## Scope exploration

- `s1` — `scripts/lfm-finetune/metrics.py + measure.py --predictions`: measure.py --predictions writes one JSONL per model in metrics.py's Prediction schema (metrics.py:19-45,190-243): id, expected, outcome, operation, arguments, full normalized candidates distribution recorded before any threshold, latency; gated/calibrated decisions are separate files from separate tools, never one merged raw+final record
  - seeds: `c3`
- `s2` — `private lfm-train work tree (issue 46 and 53 run outputs)`: final/ dirs hold .predictions.jsonl with an 18-key candidates distribution per case for a3, a3-heal, scorer-b1, stock (q46) and scorer-r3b(.`q4_k_m`) incl. gated .cal.jsonl + .gate.json (q53); `HELDOUT_SPLIT` in every pipeline env file points at the sealed held-out file itself, so saved held-out predictions are against the sealed sets; no permutation outputs are saved; scorer-r1..r3 have val only
  - seeds: `c4`
- `s3` — `scripts/lfm-finetune/{metrics,gate,sweep_gate,calibration_fit,permutation_probe}.py`: every issue-64 raw metric already exists deterministically (metrics.py:553-705, gate.py:227-287, `permutation_probe.py` order/letters/subset/paraphrase with bootstrap CIs); entropy/margin live only inside gate.decide; gate.decide and calibration apply are pure functions over saved candidates, so several policies can replay one raw file offline; all have tests (tests/`test_lfm_finetune_`\*.py)
  - seeds: `c5`
- `s4` — `DeepEval 4.2.6 (PyPI, deepeval.com docs)`: Apache-2.0, py>=3.9; custom BaseMetric needs no OpenAI key; no native dataset-level metric (no ECE/Brier); evaluate(hyperparameters=...) comparisons are a Confident-AI feature, locally only embedded in `test_run_`\*.json; hard deps include openai, grpcio, opentelemetry, posthog telemetry
  - seeds: `c6`
- `s5` — `issue 64 body (DeepEval integration section)`: 'Use deterministic metrics wherever we have a ground truth. LLM-as-judge should not replace exact action, safety, calibration or abstention measurements'; also 'avoid defining success as agreement with Jev'
  - seeds: `c7`
- `s6` — `scripts/lfm-finetune/gate.py`: Thresholds/ThresholdSet (gate.py:99-140) are JSON-serializable per class `read_only`/mutating, dispatched by nvsh.ops.table's `read_only` flag never by name; allowed-op filtering happens upstream in what candidates are offered (measure.py `_entry_candidates`:1395), so it is not replayable on saved outputs without re-scoring
  - seeds: `c8`
- `s7` — `nvsh/agent/openai_compat.py`: stdlib-only streaming chat/completions client with `api_key_env`/`api_key_file` (never a literal key); it is a runtime agent adapter for prose turns, not a scorer with logprobs, so eval provider adapters are new code outside nvsh/
  - seeds: `c9`
- `s8` — `pyproject.toml + scripts/lfm-finetune/gate.py:4-5`: dependencies=\[\] (pyproject:16); optional-dependencies ship as installable extras; wheel packages only nvsh; gate.py states it is never imported by nvsh; requirements-train.txt is the precedent for heavy out-of-uv deps
  - seeds: `c10`
- `s9` — `.github/workflows/tests.yml + pyproject pytest config`: testpaths=\['tests'\] (pyproject:100); lint runs black/isort/flake8 on 'nvsh tests scripts' only (tests.yml:61-67); no GPU runner anywhere; version-check blocks any PR without a version bump; deepeval registers a pytest11 plugin named 'plugins' that auto-loads into any pytest run and starts telemetry at import (deepeval issue 1419)
  - seeds: `c11`
- `s10` — `sonar-project.properties + publish.yml`: Sonar analyzes only nvsh/ and tests/; publish.yml triggers on nvsh/\*\* or pyproject.toml (the mandatory version bump touches pyproject, so the PR still runs test-publish)
  - seeds: `c12`
- `s11` — `scripts/scan-secrets.py + local credential check`: scan-secrets fails any tracked JSON with a non-localhost url/endpoint key (scan-secrets.py:185-239) and key-shaped strings; on this box only `NGC_API_KEY` is set; no OpenAI, Anthropic or OpenRouter key was found in env or grant
  - seeds: `c13`
- `s12` — `docs/benchmarks/2026-09-25-tool-jev-calibration-cycle.md`: the existing before/after page (scorer-b1 vs scorer-r3b.`q4_k_m`) is hand-assembled from chained CLI outputs (measure -> `calibration_fit` apply -> `sweep_gate` --final); the gate would make that page reproducible from one run
  - seeds: `c14`
- `s13` — `issue 64 body (Later / out of scope)`: 'Do not block this issue on full terminal execution'
  - seeds: `c15`
- `s14` — `docs/deliveries (issue 46 and 53)`: all nine jetson-ai-lab/qwen3.5-0.8b-nvsh-\* repos are private; `hub_upload.py` refuses non-private targets; issue 53 D49 names DeepEval (64) as the operator's final gate before publishing
  - seeds: `c16`
- `s15` — `docs/tool-jev-calibration-rule.md + docs/qwen-tool-jev-calibration.md:706`: held-out sha256 recorded, 'Never read by the lead'; 'Any rerun of a candidate ... is recorded' as a deviation; issue 46 t24 'final ONCE per checkpoint'; measure.py refuses silent overwrite under an existing label
  - seeds: `c17`
- `s16` — `scripts/lfm-finetune/permutation_probe.py`: issue 53 ran 9900 trials on test only (2.07% for r3b vs 18.8% for b1); no permutation files exist in the private run tree
  - seeds: `c18`
- `s17` — `nvsh/tiers/manager.py, router.py, registry.py`: runtime has only Needle (Tier 1) and LFM (Tier 2) slots (manager.py:170-220, router.py:337-347); no Tool-Jev/scorer loader; the Verifier protocol (router.py:99-113) is the natural seam for a calibrated scorer and is tracked as issue 54; nvsh tiers bench (bench.py) has a corpus + c20 targets (p95<150ms, >=90% accuracy, 1 GiB) but --tier accepts only fixture/needle
  - seeds: `c19`
- `s18` — `challenge pass / counter-evidence lens: saved final predictions + q53-run-d7 splits`: q46 test 64 ids vs q53 test 198 ids overlap 0; q46 test ids 64/64 inside q53-d7 train.json; reference models on 'the same cases' must mean one named case set per comparison
  - seeds: `c31`
- `s19` — `challenge pass / counter-evidence lens: q46 final/a3-heal, a3-heal.q4_k_m`: only final-\*-a3-heal\*.predictions.jsonl exist; held-out and missing-candidate predictions exist for a3 and a3-exact (pre-heal), not for the shipped heal build
  - seeds: `c4`
- `s20` — `challenge pass / security lens: repo tracked files`: git ls-files has no split or dataset JSON for issue 46/53 (only plan docs and nvsh/tiers/corpus); the splits live only in the private work tree and private HF dataset repos, so traces would be the first place their text enters git
  - seeds: `c32`
- `s21` — `challenge pass / adjacent-systems lens: scripts/lfm-finetune/measure.py request path (explorer report)`: Track A runs through RecordingChat/ToolChat like the daemon, Track B through scorer.py's candidate readout; frontier APIs expose neither natively, so the fairness of the comparison rests on reusing that prompt; I did not read the composer myself
  - seeds: `c33`
- `s22` — `challenge pass / reproducibility lens: provider /models catalogs (live 2026-09-26)`: gpt-6-luna/sol carry no dated suffix; OpenRouter lists ~aliased 'latest' ids next to dated ones; Anthropic returns no logprobs; outputs from hosted models drift over time, so c29's byte-identical rerun only holds from cache
  - seeds: `c34`
- `s23` — `challenge pass / failure-mode + operations lens: provider layer (not yet built)`: 13 remote model rows x 198 test (+83 mc) cases is ~3,650 calls per full run; rate limits and refusals are certain at that volume
  - seeds: `c35`
- `s24` — `challenge pass / security lens: provider responses`: OpenRouter routes to third-party hosts; any returned argument could be adversarial; nvsh's own rule is propose-don't-run
  - seeds: `c36`
- `s25` — `challenge pass / operations + concurrency lens: serve_for_measure.sh, issue 58`: local serving shares GPUs with the lobes (cortex max-num-seqs 2 OOM guard) and a concurrent stop/start on one measuring port can kill the replacement server (issue 58); local reference runs and any new a3-heal inference must be serialized
- `s26` — `challenge pass / migration + reversibility lens: pyproject, evals/, CI`: clean pass: evals/ is additive, no schema or state migration, removal is deleting a tree and a dependency group; residual risk only in uv.lock churn from deepeval's transitive deps
- `s27` — `challenge pass / observability lens: gate outputs`: clean pass beyond C5: per-case traces are the inspection surface; residual gap is that no alert fires when a provider silently swaps the model behind an unversioned id (covered only by recording the returned model id)
- `s28` — `cost assessment 2026-09-26: provider price lists + OpenRouter balance`: Per fresh full run (281 test+mc cases x 2 interfaces = 562 calls/subject; 684 explain answers x 5 judges; ~1.65k input tokens/case, measured from the Track A tool schema + Track B scorer prompt), low (reasoning off, ~100 out) / high (~1.5k reasoning out): OpenAI $3.7/$17.8, Anthropic $11.3/$57.1, OpenRouter $5.4/$26.1, NVIDIA $0 (free dev tier, ~40 RPM; $7.4/$35.3 at OpenRouter-equivalent prices), total paid $20/$101 (x1.5 margin $31/$152). Official prices: OpenAI gpt-6-sol $2/$10, luna $0.10/$0.50; Anthropic opus-5-5 $4/$20, sonnet-5 $2/$10 (+~30% tokenizer), batch -50%. OpenRouter key: $5 credits, $10 key limit — below even the low estimate with margin. Cached reruns cost $0

## Decisions

- Release candidates are a3-heal.`q4_k_m` (Track A, issue 46) and scorer-r3b.`q4_k_m` (Track B, issue 53); scorer-b1 and stock appear as baseline rows
- Reference models (frontier via OpenAI/Anthropic platform APIs, open via OpenRouter and build.nvidia.com, local) are comparison subjects scored by the same exact metrics; an LLM judge is used only on explain text and reported separately from the release bars
- Provider keys live in grant as hidden secrets `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `OPEN_ROUTER_API_KEY` (note the underscore in `OPEN_ROUTER`), injected only at run time via grant run --inject; build.nvidia.com uses the `NGC_API_KEY` already in the environment
- Reference roster: OpenAI gpt-6-luna, gpt-6-sol; Anthropic claude-opus-5-5, claude-sonnet-5; OpenRouter qwen/qwen3.8-max-0902, deepseek/deepseek-v4-flash, deepseek/deepseek-v4-pro-0813, moonshotai/kimi-k3; build.nvidia.com moonshotai/kimi-k3, z-ai/glm-5.3, nvidia/nemotron-3-ultra-550b-a55b, nvidia/nemotron-3-super-120b-a12b, google/gemma-4-31b-it; local Qwen3.8 27B and Gemma 4 26B-A4B (gemma-4-26b-a4b-it), both served locally; kimi-k3 deliberately on two hosts
  - instruction: roster lives in the eval manifest; ids checked against each provider's live /models list on 2026-09-26
- a3-heal.`q4_k_m` gets exactly one new inference run on the issue-53 test (198) plus its missing-candidate slice, an approved exception to c17 recorded here; the issue-46 and issue-53 sealed held-outs stay untouched and a3-heal gets no held-out run
  - instruction: serialize with any local serving (issue 58); label the run so measure.py refuses a silent rerun
- Explain text is judged by a blind panel, all-to-all: every subject (both candidates, the baselines and all 15 references) answers; the panel claude-opus-5-5 (Anthropic), gpt-6-sol (OpenAI), moonshotai/kimi-k3 (build.nvidia.com), nvidia/nemotron-3-ultra-550b-a55b (build.nvidia.com) and qwen/qwen3.8-max-0902 (OpenRouter) scores every answer with model identity removed and order shuffled; a judge's scores of its own answers are kept apart from the panel score, which is aggregated over the other judges; panel scores stay outside the release bars (c7)
  - instruction: judge prompt and rubric versioned in evals/; judge calls cached like subject calls (c34) and bounded by the same budget (c35); report per-judge scores and inter-judge agreement

## Hard questions

- Which two models are 'the 2 candidates': a3-heal.`q4_k_m` (Track A, issue 46) + scorer-r3b.`q4_k_m` (Track B, issue 53) — the current per-track picks — or scorer-b1 + scorer-r3b (the before/after pair the issue-53 report sets side by side)? (resolved: a3-heal.`q4_k_m` (Track A, issue 46) + scorer-r3b.`q4_k_m` (Track B, issue 53); scorer-b1 and stock are baseline rows)
- Role of the frontier/open/local models: comparison subjects scored by the same exact metrics, LLM judges (only for non-exact outputs like explain text), or both? And does 'harness' mean nvsh's own agent adapters (claude/codex/pi/qwen via nvsh ask) as additional subjects? (resolved: Subjects + narrow judge: reference models answer the same cases scored by the same exact metrics; an LLM judge only on explain text, reported separately from release bars)
- Reference models need to see cases to be compared: may they see the sealed held-out (sending it to external APIs exposes it and ends its value as sealed), or do references run on the spent test side only, or on a new reference set? (resolved: Reference models see the test side only (plus its missing-candidate slice); the sealed held-out never leaves the machine)
- Does 'then add it to the stack' mean wiring the released model into nvsh's runtime tiers (a Verifier/tier slot, overlapping issue 54) inside this work, or adding the DeepEval gate to the tooling stack, with the tier wiring as a follow-up? (resolved: This work builds the repeatable DeepEval gate; wiring the released model into nvsh runtime tiers (Verifier seam, issue 54) is a follow-up spec after the gate)

## Open questions

- What 'add it to the stack' means: the release candidate wired into nvsh's runtime tiers, or the DeepEval gate added to the project's tooling/CI stack

## Open parks

- [unknown_nonblocking] Calibration of quantized builds from served routes: issue 46 could not measure scorer-b1 Q4/AWQ distributions (d17/l6), while issue 53 measured r3b Q4 complete (198/198); whether every candidate's deployed artifact has complete distributions in the saved files is unverified per artifact
- [unknown_nonblocking] Which interface each reference model answers through (generative tool call like Track A, or choosing among offered candidates like Track B, or both) and how its choice maps onto the 18-key candidate space

## Resolved vagueness

- [unknown_nonblocking] Provider API keys for OpenAI, Anthropic and OpenRouter are not present on this machine (only `NGC_API_KEY`); live reference runs wait on the operator supplying them via grant — resolved: Operator added `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPEN_ROUTER_API_KEY` to grant (hidden) on 2026-09-26
- [unknown_nonblocking] Local reference models Qwen3.8 27B and Gemma 4 need serving on spark/spark2 (weights, engine, memory next to the lobes); the catalogs list gemma-4-26b-a4b-it and gemma-4-31b-it but no 27B, so the exact Gemma weights are undecided — resolved: Gemma: google/gemma-4-31b-it on build.nvidia.com AND gemma-4-26b-a4b-it served locally; local Qwen3.8 27B also served locally (serving mechanics are a plan task)
- [unknown_nonblocking] Judge model for explain text: using a model that is also a subject (same family) biases the judge; which judge and whether it is excluded from its own family's rows is undecided — resolved: Blind all-to-all judge panel of 5 expensive models; self-scores excluded from the panel aggregate
