# Build Plan — LFM2.5-350M Tier 2 fine-tune (issue 39)

slug: `lfm2-5-350m-tier-2-fine-tune-issue-39` · status: `exported` · from frame: `lfm2-5-350m-tier-2-fine-tune-issue-39`

> A fine-tuned LFM2.5-350M ends Tier 2 turns with propose or escalate where those are right, and still explains read-only questions in words: it beats the stock 350M on right proposals and escalations, makes no wrong mutating proposals, and keeps a warm median under half a second on the DGX Spark

## Tasks

### t1 — Corpus accepts an explain expectation (nvsh/tiers/bench.py)

- instruction: Touch only nvsh/tiers/bench.py and tests/`test_tier_bench.py`. Extend `_validate_expect` (bench.py:153-161) and CorpusEntry's docstring; do not name any operation in code. Bump the version with the version-bump skill in the PR that lands this (h20).
- covers: c30, h20
- acceptance:
  - `load_corpus` accepts an entry whose expect is {"explain": true} with 0 problems; an expect with neither escalate, explain nor a table-valid operation is still reported as a problem
  - Tier 1 scoring (nvsh tiers bench) treats an explain entry as should-decline; existing propose/escalate scoring and every existing test in tests/`test_tier_bench.py` are unchanged
  - tests/`test_tier_bench.py` gains tests for the explain form; uv run pytest -n auto passes

### t2 — Needle builder handles explain entries (scripts/needle-finetune/`build_dataset.py`)

- instruction: Touch only scripts/needle-finetune/`build_dataset.py` and tests/`test_needle_finetune_dataset.py`. Read expect handling at lines 52-66 first. Tier 1 cannot explain, so the honest mapping is decline.
- covers: c30
- acceptance:
  - an explain entry in the corpus is neither an error nor a propose example: it becomes the builder's should-decline form (or is skipped with a counted reason), documented in the module docstring
  - tests/`test_needle_finetune_dataset.py` covers an explain entry; all existing tests pass

### t3 — Seeded train/val/test split script (scripts/lfm-finetune/split.py)

- instruction: New files only: scripts/lfm-finetune/split.py and tests/`test_lfm_finetune_split.py`. Stratify by expectation kind; use random.Random(seed) and sort before shuffling so order is platform independent. Read entries as JSON, not via `load_corpus`, so this task does not depend on t1.
- covers: c28, h16, c31, c8, h7, c42
- acceptance:
  - same seed gives an identical split; no source id on two sides; every expectation kind (operation, escalate, explain) present on every side
  - passing held-out.json raises ValueError exactly as `build_dataset.py` does
  - output files carry each entry's source id so later variations can inherit the side (c42)
  - tests/`test_lfm_finetune_split.py` pins the split for the committed seed
  - the pinning test uses a small fixture corpus, not dev.json, so t6 adding entries cannot break it

### t4 — LFM builder: explain branch and split input (scripts/lfm-finetune/`build_dataset.py`)

- instruction: Touch only scripts/lfm-finetune/`build_dataset.py` and tests/`test_lfm_finetune_dataset.py`. Explain text comes from the corpus entry (an 'answer' field authored with it in t6); never generate it.
- depends on: t1
- covers: c7, c8, h7
- acceptance:
  - an explain entry yields an assistant explain(text) tool call; `answer_for` keeps its propose and escalate branches unchanged
  - the builder accepts a split file (train side) and never reads dev.json whole when --split is given; held-out.json is still refused
  - tests/`test_lfm_finetune_dataset.py` updated: the closed set becomes {propose, escalate, explain}; all tests pass

### t5 — Stock-vs-tuned Tier 2 measurement script (scripts/lfm-finetune/measure.py)

- instruction: New files only: scripts/lfm-finetune/measure.py and tests/`test_lfm_finetune_measure.py`. Print the same columns as docs/benchmarks/2026-09-19-dgx-spark-tier2.md. No endpoints or keys in the file.
- depends on: t1
- covers: c29, h17, h14, h15, c35, h24, h21, h29, h1
- acceptance:
  - builds an LfmTier for a given \[tiers.lfm\] model and calls nvsh.tiers.bench.bench with options.tier2 on a named split file; scoring uses `_is_correct`, `compute_escalation` and `compute_false_mutating`
  - refuses held-out.json unless --acceptance is passed and a test file unless --final is passed; counts explain outcomes on explain entries
  - reports per-variation and per-source columns; writes a dated file under docs/benchmarks/ with the command line, seed, repo id and commit, plus docker ps and nvidia-smi captured before each run
  - refuses to start if a container named `nvsh-tier2-<uid>` is already running, rather than stopping it
  - tests/`test_lfm_finetune_measure.py` covers all of this with the fake tier and a fake docker runner

### t6 — Explain entries and updated counts (nvsh/tiers/corpus/dev.json + docs)

- instruction: Touch only nvsh/tiers/corpus/dev.json, docs/tiers-improving-accuracy.md and docs/benchmarks/2026-09-19-dgx-spark-tier2.md (a dated note, not a rewrite). Entries are questions a person would answer in words; aim for at least as many as the smallest existing expectation group per side after the split.
- depends on: t1
- covers: c7, h6
- acceptance:
  - dev.json gains read-only-question entries with expect {"explain": true} and an authored answer; no existing entry changes (a diff shows only additions)
  - `load_corpus`(dev.json) reports 0 problems; docs/tiers-improving-accuracy.md and the benchmark note quote the new counts

### t7 — Augmentation pipeline: generate, correct, double review (scripts/lfm-finetune/augment.py)

- instruction: New files only: scripts/lfm-finetune/augment.py and its test. The generator is told the fixed answer and asked only to rephrase; the corrector and reviewers answer 'does this still mean exactly this answer?'. Keys are supplied at run time via grant run.
- depends on: t3
- covers: c42, c43, h30, c44, h31
- acceptance:
  - each variation inherits its source's split side and expected answer; NVIDIA's evals are refused as seeds
  - a variation enters accepted.jsonl only if both reviewers accept it; rejected.jsonl keeps source id, model ids and each verdict; the run prints counts per stage
  - endpoints and model ids come only from environment variables; the committed file contains no URL other than localhost and no key; scan-secrets passes
  - tests/`test_lfm_finetune_augment.py` drives the pipeline with fake OpenAI-compatible endpoints

### t8 — Jetson skills validation set: tools, 104-eval test set, provenance (scripts/lfm-finetune/`jetson_skills.py`)

- instruction: New files only: scripts/lfm-finetune/`jetson_skills.py`, its test and fixtures. Handle both eval shapes: device {`skill_name`, evals:\[{id, prompt, ...}\]}, BSP \[{id, question, `expected_skill`, ...}\]. Nothing from these repos goes into nvsh/.
- covers: c37, h27, c40, h28
- acceptance:
  - reads both NVIDIA repos at pinned commit shas into scratch space; builds one tool per skill (38) from SKILL.md name and description; the test set is exactly the 104 evals, with evals that name the skill outright flagged
  - a contamination scan (exact and near-duplicate) fails the build if any eval prompt or `ground_truth` appears in a training file
  - writes a manifest mapping every derived record to repository, file, licence (CC-BY-4.0 or Apache-2.0) and transformed yes/no, and a dataset README with the attribution
  - tests/`test_lfm_finetune_jetson_skills.py` runs against a small fixture copy of both eval formats (device dict form, BSP list form)

### t9 — Skill-routing measurement (scripts/lfm-finetune/`measure_skills.py`)

- instruction: New files only: scripts/lfm-finetune/`measure_skills.py` and its test. Use the same vLLM + lfm2 launcher path for serving as measure.py; endpoint from the environment.
- depends on: t8
- covers: c37, h25
- acceptance:
  - sends each eval to an OpenAI-compatible endpoint with the 38 skill tools and scores the called skill against the expected one; reports overall, skill-named-in-prompt and other columns
  - requires --margin before any tuned run is scored and writes it into the dated results file ahead of the numbers
  - tests with a fake endpoint

### t10 — Training environment spike on the DGX Spark GB10

- instruction: Run-only task on this box; no repo code. If aarch64 support fails, record why and switch to Route 3 (remote HF) as the recorded fallback (c19).
- covers: c3, h3
- acceptance:
  - a scratch venv outside the repo has torch with CUDA on aarch64 plus unsloth (or TRL+PEFT if unsloth fails); a 20-step LoRA run on 10 examples shows the GPU in use (nvidia-smi) and a falling loss
  - every command that worked, and every one that failed with its symptom and fix, is written into the guide draft; routes not tried stay marked unverified

### t11 — Chat-template and base-model check with the real tokenizer

- instruction: The post-trained LiquidAI/LFM2.5-350M exists on the Hub (sha 9e6c6ccf47cd seen 2026-09-22); pin the commit you actually use.
- depends on: t10, t4
- covers: c4, h4, c34, h23
- acceptance:
  - one built example (single-turn propose, escalate, explain) rendered through `apply_chat_template`(messages, tools=tools) with LiquidAI/LFM2.5-350M at a pinned commit is recorded in the guide, with the argument form chosen (object or string) and why
  - loss masking is shown to cover assistant turns only; the base repo id and commit are recorded and are the ones stock is measured as

### t12 — Re-measure stock on the committed split and on the 104 skill evals

- instruction: Run back to back with the tuned runs later under the same background set where possible; check no nvsh-tier2 container is running first.
- depends on: t3, t6, t9, t5
- covers: c2, h2, c34, c35, h24
- acceptance:
  - measure.py runs stock on the validation side and once with --final on the test side, and `measure_skills.py` runs stock on the 104 evals; both write dated files under docs/benchmarks/ with repo id, commit and background load
  - the 2026-09-19 figures are cited as history, not as the baseline

### t13 — Build training data: split, optional x100 augmentation, Jetson skills training set, contamination scan

- instruction: Endpoints for Qwen 3.6 35B-A3B, Qwen 3.8 27B, Gemma4 26B-A4B and associate come from the environment; keys via grant run. Nothing generated is committed without the model ids that produced it.
- depends on: t3, t6, t7, t8, t12, t4
- covers: c42, h29, c43, h30, h27, c8, h7
- acceptance:
  - the nvsh training set comes from the split's train side only; augmentation runs if t12 shows the test fold is too small to separate stock from tuned (the decision and the reason are recorded)
  - the Jetson training set is generated from SKILL.md text only and passes the contamination scan; accepted and rejected counts per stage are recorded

### t14 — Train, iterating on the validation side only

- instruction: Start from docs/needle-finetune.md's lesson: the defaults barely moved loss; 20 epochs, lr 5e-4, r32/alpha64 did. Do not look at the test side while iterating.
- depends on: t11, t13
- covers: c3, h3, c31, h21, c9
- acceptance:
  - one checkpoint per bar (nvsh use case, Jetson skills), each trained only on its train side; every iteration is judged on validation, and each change and result is logged for the guide
  - hyperparameters, final loss, time and GPU use of the run that is taken forward are recorded as measured

### t15 — Merge, template check, private push, cache prefetch and export check

- instruction: Any launcher change is the only nvsh/ runtime edit allowed in this plan and gets its own test and version bump.
- depends on: t14
- covers: c33, h22, c9, h8, c15, h10, c40, h28, c44
- acceptance:
  - the merged checkpoint's `chat_template` is byte-identical to the base's; attribution and licence notes (LFM 1.0, NVIDIA CC-BY/Apache) are written before the push
  - pushed to a private jetson-ai-lab repo with grant run `HF_TOKEN` -- hf upload; fetched into \[tiers.lfm\] `hf_cache_dir` with grant run `HF_TOKEN` -- hf download; no token enters the container
  - served through the real Tier 2 launcher with vLLM + lfm2, at least 10 training requests come back as structured `tool_calls` with the trained answer before any test figure is recorded; if vLLM needs an offline flag, a small \[tiers.lfm\] option is added with a test, and nothing else in `render_launch` changes

### t16 — Final measurement against both bars

- instruction: Report unfavourable numbers as they are. If only method validation passes, say 'the method works, nvsh's data is short'.
- depends on: t15
- covers: c1, c26, c27, c37, h25, h14, h15, h1, h24, h29
- acceptance:
  - measure.py --final on the test side for stock and tuned back to back; `measure_skills.py` on the 104 evals for stock and tuned with the margin stated before scoring
  - results say which bar was met (use-case: >= 70% proposals, >= 60% escalations, 0 wrong mutating, < 500 ms; or method validation by the stated margin), judged per source, and are labelled indicative pending issues 38 and 40

### t17 — Rewrite docs/lfm-finetune.md as the verified guide, plus model and dataset card drafts

- instruction: Follow c45's three-artifact decision. Nothing is made public in this task.
- depends on: t16
- covers: c24, h18, h26, c23, h11, c25, h13, c3, h3, h12, c15
- acceptance:
  - the guide walks setup, split, template check, data build, augmentation, training, merge, export check, push, measurement and reading the result, with the commands that actually ran and a pitfalls section (symptom and fix)
  - states which success bar was met, or records the attempt as unsuccessful with numbers and what changed next; names both readers (operator, maintainer); cites deviation d4
  - model card draft (LFM Open License 1.0, >= 10M USD threshold, modified-files notice, teachers listed as generators/reviewers) and dataset README with the source manifest; docs/lfm-license-notes.md updated; markdownlint passes

### t18 — PR hygiene and boundary checks

- instruction: Use the cicd skill for PRs. Ask the operator before filing the multi-round follow-up issue or commenting on issue 38 (outward actions).
- depends on: t17
- covers: c13, h9, c6, h5, h12
- acceptance:
  - every PR bumps the version and says 'part of #39'; none closes #39
  - git diff of the whole run shows no change to `_spoke` in nvsh/tiers/lfm.py and no training or upload code under nvsh/; switching \[tiers.lfm\] model back to stock is one line

## Risks

- [unknown_nonblocking] The measurement script and an operator's own nvsh daemon share one container name (`nvsh-tier2-<uid>`) and port per user; the script must refuse, never stop the operator's container (task t4)
- [unknown_nonblocking] Whether unsloth (or torch with CUDA) installs and trains on aarch64 GB10 is unverified; fallback is the remote HF route (task t10)
- [unknown_nonblocking] Whether vLLM in the pinned image serves a private repo from the pre-filled cache without a token (offline flag) is unverified (task t15)
- [unknown_nonblocking] The four generator/reviewer API endpoints were not probed during speccing; availability and throughput for a x100 run are unknown (task t13)
