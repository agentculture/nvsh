# Build Plan — Tool-Jev calibration cycle (issue 53)

slug: `tool-jev-calibration-cycle-issue-53` · status: `exported` · from frame: `tool-jev-calibration-cycle-issue-53`

> The Tool-Jev scorer now makes calibrated, label-permutation-robust decisions on its deployed quantized build: an improvement cycle over code and data measured the permutation dependence of scorer-b1, added an uncertainty gate separate from semantic escalate, trained with per-example randomized candidate labels on a grown dataset, and reports reliability per slice for the actual shipped artifact (issue #53)

## Tasks

### t1 — Readout core in scorer.py: one label-probability definition and a complete top-k

- instruction: Files: scripts/lfm-finetune/scorer.py, tests/`test_lfm_finetune_scorer.py` only. Decide the definition from the 2026-09-25 probe (dev-f02 gap: in-process single token id vs served variant sum 'A'/' A'); prefer the served variant-sum definition and make in-process sum the same token ids, then document why in the docstring.
- covers: c39, h31, c19, h22
- acceptance:
  - One function computes candidate probabilities from label logits/logprobs and is used by the in-process scorer, the served scorer and `train_scorer` (tests/`test_lfm_finetune_scorer.py` asserts both paths give the same distribution on a fixture)
  - The served request asks for top >= 5000 via a named constant (`READOUT_TOP`) instead of len(labels)+`TOP_MARGIN`; incomplete results are still marked incomplete, never renormalised
  - Track B still sends `max_tokens`=1 and one forward position (test)

### t2 — Permutation seam in scorer.py: seeded order/letter permutation, subsets and description overrides

- instruction: Files: scripts/lfm-finetune/scorer.py, tests/`test_lfm_finetune_scorer.py`. Keep `label_token_ids`' single-token and distinct-id checks for every letter in `LABEL_ALPHABET` actually used.
- depends on: t1
- covers: c2
- acceptance:
  - `prompt_messages`/score accept an explicit label map (candidate -> letter) and an order; `labels_for` keeps today's fixed map as the default (existing tests unchanged)
  - A seeded helper produces a random order + letter permutation + optional subset; the same seed gives the same map (test)
  - Candidate descriptions can be overridden from a scripts-side mapping (for paraphrase probes and reason candidates) without reading nvsh/; default stays `ops_table`/lfm (test)
  - Results are compared at the operation level, never the letter (test: permuted letters, same op chosen => no change counted)

### t3 — Served readout cap: servers started for measurement return >= 5000 logprobs

- instruction: Files: scripts/lfm-finetune/`serve_for_measure.sh`, scripts/lfm-finetune/pipeline\*.env.example. Do not touch scorer.py (t1 owns `READOUT_TOP`).
- covers: c38, c7
- acceptance:
  - `serve_for_measure.sh` defaults `MEASURE_MAX_LOGPROBS` to >= 5000 for vLLM (--max-logprobs) and passes no cap that truncates llama-server `n_probs` below it; pipeline env examples updated
  - bash -n passes; existing `serve_for_measure` tests (if any) pass

### t4 — Post-hoc calibration module: seeded validation folds, temperature and vector scaling fit/apply

- instruction: New files only: scripts/lfm-finetune/`calibration_fit.py`, tests/`test_lfm_finetune_calibration_fit.py`. Stdlib only (math, json, random); reuse metrics.py readers by import, do not edit metrics.py.
- covers: c5, h9, c41, h33
- acceptance:
  - New scripts/lfm-finetune/`calibration_fit.py`: split validation ids into fit/selection folds with a recorded seed; no id in both (test)
  - Fits a temperature (and a per-candidate vector) by minimising NLL on the fit fold from predictions.jsonl candidates; apply() rescales a predictions file; recovers a known temperature on synthetic data (test)
  - Refuses test/held-out files as fit input (test); writes the fitted parameters as JSON stored with the run

### t5 — metrics.py: per-slice ECE/Brier, bootstrap CIs, `abstain_uncertain` outcome, reliability bins in markdown

- instruction: Files: scripts/lfm-finetune/metrics.py, tests/`test_lfm_finetune_metrics.py`. measure.py's report page calls the new renderer in t8, not here.
- covers: c6, h10, c14, h18, c16, h19
- acceptance:
  - compute() reports ECE/Brier/bins per slice: read-only vs mutating (Operation.`read_only`), candidate count, missing-candidate, confidence bucket (test)
  - Every rate and ECE carries n and a seeded bootstrap 95% CI (test)
  - `abstain_uncertain` is counted separately from semantic escalate; escalation bars count both; escalation reasons roll up to escalate (test)
  - A markdown renderer prints the reliability bin table per slice; grep shows no operation name in metrics code

### t6 — Uncertainty gate module and offline threshold sweep

- instruction: New files: scripts/lfm-finetune/gate.py, scripts/lfm-finetune/`sweep_gate.py`, tests/`test_lfm_finetune_gate.py`. Use metrics.py (t5) for outcome counting.
- depends on: t5
- covers: c3, h7, c4, h8
- acceptance:
  - New scripts/lfm-finetune/gate.py: decide(distribution, offered, thresholds) -> propose | explain | escalate | `abstain_uncertain` using `p_escalate`, `p_top1` floor, top1-top2 margin and normalized entropy, thresholds keyed by Operation.`read_only` only (tests, incl. grep for no operation names)
  - `sweep_gate` CLI reads predictions.jsonl (no GPU) and reports, per threshold set, accuracy, wrong-mutating, missing-candidate escalation and abstention on a given fold; refuses test/held-out unless --final (test)
  - The sweep reproduces scorer-b1's stored argmax decisions exactly when every threshold is disabled (test on a fixture)

### t7 — Permutation probe runner with answer-change rate and CI

- instruction: New files: scripts/lfm-finetune/`permutation_probe.py`, tests/`test_lfm_finetune_permutation_probe.py`. Paraphrases come from a scripts-side JSON of alternative descriptions (authored in t12), passed by path.
- depends on: t2, t5
- covers: h25
- acceptance:
  - New scripts/lfm-finetune/`permutation_probe.py`: for each entry, >= 10 seeded perturbations of each kind (order, letters, subset keeping gold, paraphrased descriptions) via the t2 seam; reports op-level answer-change rate with n and bootstrap 95% CI per kind (test with a fake scorer)
  - Never canonicalises runtime order; refuses test/held-out without --final (test)

### t8 — measure.py: wire the new readout, calibration apply and per-slice report into measure runs; fix P71

- instruction: Files: scripts/lfm-finetune/measure.py, tests/`test_lfm_finetune_measure`\*.py. Closes #57 in the PR body.
- depends on: t1, t4, t5
- covers: h11
- acceptance:
  - measure runs accept --calibration <params.json> and render per-slice reliability tables into the committed benchmark markdown (test)
  - A served run records complete/incomplete counts; incomplete lines are counted, never renormalised (test)
  - A failed start-up no longer leaves a results page that blocks a clean re-run with the same label (issue #57; test)

### t9 — Pre-registered decision rule for checkpoint choice, committed before any training

- instruction: Docs only. Operator confirms the rule text before t14 starts.
- covers: c42, h34
- acceptance:
  - docs/tool-jev-calibration-rule.md states, before the first training command: the selection fold, the ordered criteria (0 wrong mutating at gate thresholds; permutation change rate; ECE after temperature; missing-candidate escalation; clean accuracy floor vs scorer-b1), tie-breaks, and what makes the c31 calibration-aware stage run
  - The commit adding it precedes the first training run's commit/log timestamp (checked in validate-delivery)

### t10 — Live guide and ledger with a documentation subagent

- instruction: A long-lived doc subagent in its own worktree owns only this file and never runs devague; the lead forwards every finding as it happens (issue 46 style, docs/qwen-tool-jev-finetune.md is the model).
- covers: c43, h35
- acceptance:
  - docs/qwen-tool-jev-calibration.md exists from the start of the run: code map, corpus v2 design, every run/obstacle/fix (symptom -> cause -> fix), results, reproduce steps
  - Updated in the same step as each run, obstacle or fix; markdownlint, scan-secrets and harness-smoke clean; no home or machine paths

### t11 — Documentation relabel: Track A is a generative tool router, Track B the Jev-style scorer

- instruction: Files: those three only (not the new guide, owned by t10). Track A's distribution is reconstructed offline by `track_a_calibration.py`; say so.
- covers: c20, h23
- acceptance:
  - docs/qwen-tool-jev-finetune.md, docs/benchmarks/2026-09-24-qwen-tool-jev-comparison.md and the model-card text in `release_bundle.py` describe Track A as a specialized generative tool router and Track B as the Jev-style candidate scorer; no text calls them equally Jev-like

### t12 — Corpus v2 tooling: versioned re-split kept outside nvsh/, with validation folds

- instruction: Files: scripts/lfm-finetune/split.py, tests/`test_lfm_finetune_split`\*.py. The v2 files themselves live in the run's work dir and the private data repo, never committed (operator decisions q10/q11).
- depends on: t4
- acceptance:
  - split.py can write a versioned corpus v2 (header names version, seed, source hashes) to a path outside nvsh/ and never writes nvsh/tiers/corpus (test)
  - Target sizes are parameters: validation >= 150, test ~150, stratified by answer kind; the fold assignment from `calibration_fit` is recorded in the split header (test)
  - git diff shows no change under nvsh/tiers/corpus/ (checked in validate-delivery)

### t13 — Draft new sources for the fresh evaluation sides and the sealed held-out

- instruction: Run task. Measure only on a quiet GPU; teachers on the lobes (restore any lobe you stop). Forward every step to the t10 doc agent.
- depends on: t12
- covers: c13, h17, c17, h20
- acceptance:
  - Enough new reviewed entries (all-Apache teachers, reviewer B, `leakage_check.py`) to give validation >= 150 and test ~150 after the v2 re-split; counts per answer kind and per escalation reason recorded
  - A new held-out is drafted with `draft_heldout.py`; the lead records only counts and hashes; it is sealed before any training of this cycle and stored only in the private data repo
  - The issue-46 test side is not used for any claim of this cycle

### t14 — Dataset build: per-example candidate sets and letter maps, missing-candidate and no-valid-option examples, reason candidates

- instruction: Files: scripts/lfm-finetune/`build_dataset.py`, `merge_variations.py`, new scripts/lfm-finetune/data/{reasons,paraphrases}.json, their tests. Candidate count must stay within the single-token label check.
- depends on: t2
- covers: c12, h16, c45, h37
- acceptance:
  - `build_dataset.py` stores offered candidates, order and letter map on every rendered example, seeded and replayable (test)
  - Deterministic missing-candidate / no-valid-option train examples are generated from train entries only (gold removed -> escalate), `eval_slices` shape, never from eval sides (test)
  - An optional reasons mode renders the 8 decline classes as distinct escalate-reason candidates with descriptions from a scripts-side JSON (not nvsh/), rolling up to escalate in gold labels (test)
  - The scripts-side paraphrase JSON for the permutation probe exists and is covered by a schema test

### t15 — Augment for the comparison doc's five other recommendations, then freeze the v2 training set

- instruction: Run task. More data is an acceptable fix when a slice is short (operator rule); add it before the freeze.
- depends on: t13, t14
- covers: c11, h15
- acceptance:
  - New train-side variations for missing-argument escalation, diagnosis-vs-explanation pairs, explicit-mode `power_set` positives, disambiguation pairs and hard negatives, through augment.py teachers + reviewer B; acceptance counts recorded
  - `leakage_check.py` is clean against every v2 eval side and the held-out; the frozen set's hashes are recorded; any later change is a /deviate record

### t16 — `train_scorer.py`: per-row label ids and targets from the rendered prompt; optional calibration-aware loss

- instruction: Files: scripts/lfm-finetune/`train_scorer.py`, tests/`test_lfm_finetune_train_scorer`\*.py. Keep the default path byte-identical in behaviour for the fixed map so scorer-b1 stays reproducible.
- depends on: t1, t2
- covers: c10, h14
- acceptance:
  - Each row's label token ids and CE target come from that row's stored letter map and offered set; two rows with different maps give different target indices for the same gold op (test)
  - The randomisation seed and per-row maps are logged in train-log.json
  - A --calibration-loss option adds label smoothing and/or a Brier term beside CE; off by default (test)

### t17 — Baseline on scorer-b1 before any retraining: permutation probe, readout fidelity, offline gate and temperature

- instruction: Run task on a quiet GPU. Q4 fits use Q4's own validation predictions, not bf16's.
- depends on: t3, t6, t7, t8, t12, t13
- covers: c2, h6, c40, h32
- acceptance:
  - Permutation probe of scorer-b1 on v2 validation: answer-change rate + CI per perturbation kind (order, letters, subset, paraphrase), before any new training
  - Served bf16 GGUF readout at top >= 5000 matches in-process exact within 0.01 on validation (or the gap is reported); `Q4_K_M` served distributions recorded on validation
  - Offline gate sweep and temperature fit on scorer-b1's Q4 validation predictions (fit fold / selection fold); results in the guide

### t18 — Training runs and selection: randomized-label variant, reasons ablation, lr variants, conditional calibration-aware stage

- instruction: Run task (spark2 training, spark measuring; never overlap training and measuring on one Spark). Any retry beyond the pre-registered set is a /deviate record.
- depends on: t9, t15, t16, t17
- covers: c31, h28
- acceptance:
  - At least: randomized-label run on v2 (reasons off), same with reasons on (c28 ablation), and a lower-lr variant; each measured in-process on the selection fold with the gate and temperature
  - The checkpoint is chosen by the pre-registered rule (t9); reasons kept only if they improve validation
  - The calibration-aware stage runs only if post-hoc + data miss ECE <= 0.10 on validation, and is compared head-to-head with post-hoc on validation; if not triggered, the validation evidence is recorded

### t19 — Quantize the chosen checkpoint and calibrate the deployed build

- instruction: Run task. AWQ optional; if built, calibrate it the same way or say it was not.
- depends on: t18
- covers: c5
- acceptance:
  - `Q4_K_M` GGUF of the chosen checkpoint served by llama-server with top >= 5000 returns complete distributions on validation
  - Temperature (and vector if chosen) and gate thresholds are fitted on the Q4 build's own fit/selection folds and stored with the run

### t20 — Final measurement on the fresh test side and held-out, Spark and Orin

- instruction: Run task. Bars missed are reported as missed, never rounded up; a missed bar blamed on data becomes a data task, not an explanation (operator rule).
- depends on: t19
- covers: c25, c26, h26, c27, h27, c9, h13, h30, h11
- acceptance:
  - Permutation answer-change rate on fresh test with >= 10 permutations per entry, point estimate and 95% CI, runtime order not canonicalised (bar <= 2.3%)
  - Deployed `Q4_K_M`: ECE with bootstrap CI and reliability diagrams per slice after the fitted temperature (bar <= 0.10); complete distributions >= 95%
  - Wrong mutating at the chosen thresholds on test and held-out (bar 0); missing-candidate escalation incl. uncertainty abstain (bar >= 80%)
  - Same deployed build measured on AGX Orin (validation side) with complete distributions; each final checkpoint exposed to test/held-out once

### t21 — Private uploads without touching PR #52's artifacts

- instruction: Ask the operator before any upload; `HF_TOKEN_FT` grant, never make anything public.
- depends on: t20
- covers: c44, h36, c18, h21
- acceptance:
  - New private repos (or tagged revisions) for the new scorer, GGUF and data; every bundle has a clean `scan_bundle` report; the six PR #52 repos are byte-identical before and after

### t22 — Cycle report: comparison against scorer-b1 and OpenJev, before/after, follow-up links

- instruction: Docs task; markdownlint clean.
- depends on: t20, t11
- covers: c1, h1, c34, h2, c35, h3, c36, h4, c37, h5, c23, h24
- acceptance:
  - A new benchmark file `docs/benchmarks/<date>-tool-jev-calibration-cycle.md`: every announcement clause with its measured number, CI and n, before (scorer-b1, sourced) vs after, per-slice reliability tables, OpenJev comparison limited to what OpenJev publishes
  - Names the artifact, thresholds and slices so a reader without #46 context can use it (#54's input); links #54-#60

### t23 — Close the loop: validate-delivery, summarize-delivery, version bump, PR

- instruction: Standing flow: follow all waves, /deviate and decide, /validate-delivery and /summarize-delivery before the PR.
- depends on: t21, t22, t10
- acceptance:
  - /validate-delivery evidence filed for every success signal and honesty condition; /summarize-delivery record committed; version bumped; PR via /cicd says 'part of #53' with CI and Sonar green

## Risks

- [unknown_nonblocking] Shared compute: training and serving on one DGX Spark causes `NV_ERR_NO_MEMORY` (issue 46 ledger); teachers run on the lobes. Measure only on a quiet GPU, train on spark2, restore any lobe stopped. (task t18)
- [unknown_nonblocking] vLLM at --max-logprobs >= 5000 and Orin's llama.cpp build at `n_probs` 5000 were not probed (frame park v2); if either fails, h30 reports the incomplete count and the readout is revisited via /deviate. (task t17)
- [unknown_nonblocking] Lowercase letter labels (a-z) may behave differently from uppercase once letters are randomised (frame park v3); t17/t18 report answer-change by label case. (task t18)
- [unknown_nonblocking] Corpus v2 volume: validation >= 150 and test ~150 plus the train side need roughly 250+ new reviewed entries beyond today's 431, plus a new held-out; teacher throughput (~23 tok/s per request on cortex) may make t13 the longest step. (task t13)
- [unknown_nonblocking] AGX Orin must be reachable over ssh for t20's edge measurement; if it is not, the Orin row is reported as not measured, not inferred from the Spark. (task t20)
