# Qwen3.5-0.8B Tool-Jev fine-tune (issue 46)

> nvsh has an Apache-2.0 Tool-Jev: Qwen3.5-0.8B fine-tuned on spark (generative structured tool caller, Track A) and spark2 (logit-scored candidate actions with measured calibration, Track B), both scored on one shared held-out suite against the stock baseline, with base-model and training-data provenance documented so the result can be shared publicly, including with NVIDIA (issue 46)
> instruction: Read the comparison page's run ids against the `work/<run>.done` files; run `release_bundle.py` and `dataset_bundle.py` on the upload folders and keep their output

## Audience

- The nvsh maintainers deciding what small model backs Tier 1/Tier 2 next, and external readers (NVIDIA's Jetson AI Lab among them) who will download the published model, dataset and report
  - instruction: python3 scripts/scan-secrets.py on the committed docs; card text reviewed for private hosts

## Before → After

- Before: The only tuned Tier 2 model is LFM2.5-350M (issue 39, r8: 28/32 proposals, 13/15 escalations, 1 wrong mutating proposal, not adopted), under the LFM Open License whose USD 10M revenue condition follows every downstream user; no candidate-scoring model, no calibration metric and no clean test side exist
  - instruction: Diff the quoted numbers against docs/benchmarks/2026-09-23-lfm-final-r8.md
- After: Two Apache-2.0 Qwen3.5-0.8B checkpoints (Track A generative tool caller from spark, Track B candidate scorer from spark2) exist with their recipes, scored by one shared harness on one clean test side against stock Qwen3.5-0.8B, with a comparison table (accuracy, abstention, calibration, latency, memory, 2K vs 4K, quantized) and a written recommendation for the next Tool-Jev iteration
  - instruction: Re-run a train stage from the env file on a clean work dir and compare the adapter's config and eval numbers within noise

## Why it matters

- A 0.8B Apache-2.0 decision model small enough for Orin-class devices and cheap K8s services could be shared freely (including with NVIDIA) and could stay resident; issue 46 asks whether a tiny model can give a calibrated tool decision with strong abstention, which issue 39's tooling cannot answer
  - instruction: Every size/memory claim in the report cites a measure run

## Requirements

- The base is Qwen/Qwen3.5-0.8B (the post-trained repo, HF commit 2fc06364715b967f1860aea9cf38778875588b17, licence apache-2.0 with a LICENSE file; a -Base repo also exists at dc7cdfe2), pinned by BASE/`BASE_REV` in the pipeline env file and measured stock at that same commit before any training run
  - instruction: pipeline.sh measure stage with `BASE_REV`=2fc06364715b967f1860aea9cf38778875588b17; the stock run id precedes every tuned run id
  - honesty: Stock Qwen3.5-0.8B is measured at the pinned revision on validation and the clean test side before any tuned run is scored
- train.py's loss mask must be rebuilt for Qwen: it relies on `apply_chat_template`(..., `return_assistant_tokens_mask`=True) (train.py:68-74), and Qwen3.5's chat template has no {% generation %} markers, so the mask would be empty. Either patch generation markers into a local copy of the template or compute labels by span; either way a test asserts the mask is non-empty and covers only the assistant answer
  - instruction: Unit test in tests/`test_lfm_finetune_train.py` over the real Qwen tokenizer (skipped when weights are absent) plus a printed decoded-mask check before the first training run
  - honesty: For one rendered example of each outcome (propose, explain, escalate), the loss mask is non-empty and covers exactly the assistant answer tokens (plus end-of-turn), nothing from system/user/tools
- Thinking is off in both training and serving: examples are rendered with `enable_thinking=False` (the template then emits an empty `<think></think>` block before the answer), and the served request uses the same setting. Nothing in scripts/lfm-finetune handles thinking for the trained model today (only `NVSH_AUG_*_DISABLE_THINKING` for the augmentation roles, pipeline.sh:55)
  - instruction: Assert the rendered string contains `<think>\n\n</think>` and the served `chat_template_kwargs` set `enable_thinking` false; count non-empty think blocks in measure output (must be 0)
  - honesty: Rendered training examples and served requests both use `enable_thinking`=False, and no tuned output contains a non-empty think block on validation
- The tool-call rendering is re-verified for Qwen before training: Qwen3.5 renders calls as XML (<`tool_call`><function=NAME><parameter=ARG>VALUE</parameter></function></`tool_call`>), not LFM2.5's Pythonic calls, so `build_dataset.py`'s LFM workarounds (arguments as an object, operation before arguments; `build_dataset.py`:90-92,120-124; docs/lfm-finetune.md:143-145,369-373) are checked by rendering one example of each outcome with the real Qwen tokenizer
  - instruction: Render with the Qwen tokenizer, feed the assistant span to the parser (or a served stock model echo test), compare JSON
  - honesty: One rendered example of each outcome round-trips: the chosen vLLM tool parser turns the gold assistant text back into the same tool name and arguments object
- Serving a Qwen checkpoint needs an explicit `tool_call_parser` (the vLLM engine default is the literal 'lfm2', `runtime_docker.py`:199; ~/lfm-train/measure-config.toml also sets lfm2) and a vLLM image that supports `qwen3_5` (reported >=0.17) and its XML tool calls (`qwen3_coder` family; Qwen3.5-specific parser fixes are in flux upstream). The pinned measure image vllm/vllm-openai@sha256:8bd082c2… is checked for both before any stock measurement
  - instruction: docker run the pinned image with --help/--version; serve stock model; one tool-call request parses
  - honesty: The measurement image is pinned by digest and its vLLM version supports `qwen3_5` and the chosen tool parser, verified by a stock serve smoke test before measurement
- `release_bundle.py` is generalised so a Qwen bundle carries Apache-2.0 terms: today it hard-codes `LICENSE_FIRST_LINE` 'LFM Open License v1.0', `REQUIRED_CARD_PHRASES` (`license_name`: lfm1.0, the 10,000,000 threshold), Liquid AI NOTICE text and the model-card Licence section; the chat-template equality check (c33 of issue 39) and TEACHERS tuple carry over unchanged
  - instruction: tests/`test_lfm_finetune_release_bundle.py` gains Apache cases; full suite green
  - honesty: `release_bundle.py` refuses a Qwen bundle whose LICENSE is not Apache-2.0 and still refuses an LFM bundle without LFM terms; existing LFM tests keep passing
- Both tracks and the stock baseline share one dataset and one fixed split built by scripts/lfm-finetune/split.py (seeded, 70/15/15, stratified by operation/escalate/explain, refuses held-out.json, variations inherit their source's side; split.py:50-151,180-181), which is model-agnostic and reused as is
  - instruction: sha256 of train/val/test files recorded in the run log for spark and spark2
  - honesty: The split files used by both tracks are byte-identical and produced by split.py with a recorded seed
- Any final (test-side) run of this work is made on a test side that is clean for Qwen: issue 39's test side was partly read during iteration (lapse l2) and four test entries leaked into r8's training by exact wording (lapse l5), so its numbers are an upper bound. The delivery record's remaining work already calls for a fresh or re-seeded test side, with `merge_variations.py` --exclude guarding leakage
  - instruction: `merge_variations.py` --exclude plus `dataset_bundle.py`'s duplicate check pass; lapse filed immediately if the test side is read
  - honesty: No test-side entry or near-duplicate appears in any training file of either track, and nobody reads test-side contents during iteration
- Training-data provenance is documented per record and per teacher so the dataset and model can be published: corpus entries carry id/kind/text/expect/source (no licence field; licence is stated at bundle level), variations record `source_id`/side/models, and `dataset_bundle.py`'s manifest adds split/origin/`source_file`/licence/transformed plus a role->model map (`dataset_bundle.py`:50-55,127-168)
  - instruction: `dataset_bundle.py` manifest.json checked for missing fields (0 missing)
  - honesty: Every published training record carries source, origin and teacher-model fields, and the dataset card lists each teacher with its licence
- Measurement grows the metrics issue 46 asks for that issue 39's tooling lacks: logprobs-based confidence with ECE/Brier, abstention precision/recall, false-positive tool-invocation rate, tokens generated per decision and time to first decision. measure.py today reports right proposals, escalation TP/FN, wrong mutating/argument counts, container memory and warm/cold latency (measure.py:494-538,585-599,853-863) and has no logprob, calibration or token counts
  - instruction: Unit tests on hand-computed ECE/Brier fixtures; comparison table has no empty cell
  - honesty: The shared harness reports every issue-46 metric for stock, Track A and Track B from the same predictions file format, with ECE (10 equal-width bins) and Brier computed from recorded probabilities
- The 2K vs 4K context comparison is run by configuration only: training --max-length (train.py:122) and serving \[tiers.lfm\] ctx (`runtime_docker.py` `DEFAULT_CTX` 4096, validated 256..1,048,576) are both already parameters; issue 39 ran at 4096
  - instruction: Diff the two env files; memory from docker stats, latency warm median/p95
  - honesty: 2K and 4K runs differ only in --max-length / ctx, and the report states peak memory and latency for each
- Hyperparameters are re-tuned for 0.8B on the validation side only (issue 39's 20 epochs / lr 5e-4 / LoRA r32 a64 / batch 8 were tuned for the 350M; 8 epochs beat 20 on skills), and the skills benchmark is scored once with its margin stated before the scored run, or on a validation slice carved first (lapse l4)
  - instruction: Run log lists each val run; skills scoring run id follows the committed margin line
  - honesty: Hyperparameter choices are justified from validation runs only, and the skills benchmark is scored at most once per recipe with the margin written before the run
- The quantized-inference and edge checks account for the hybrid architecture: unsloth/Qwen3.5-0.8B-GGUF (sha 6ab46149) offers `Q8_0`..IQ2 quants plus separate mmproj files for the vision tower, but correct Gated-DeltaNet support needs a recent llama.cpp build, and vLLM Qwen3.5 tool parsing is still being patched upstream; each serving stack's version is recorded with its result
  - instruction: Comparison table has a stack+version column with no blanks
  - honesty: Every quantized or edge result names the serving stack and its exact version/commit
- The best checkpoint(s) are exported as a `Q4_K_M` GGUF (text-only, no vision mmproj) served by nvsh's existing llama-server engine via \[tiers.lfm\] `model_dir`, and as an INT4 AWQ checkpoint served by the vllm engine; both are measured on the clean test side with the same harness, latency and memory as bf16
  - instruction: Two measure runs per quant (spark); the served config for each is committed in the run log
  - honesty: Calibration and healing data come from the train side only; no validation or test entry is used for imatrix, AWQ calibration or healing
    - instruction: grep the calibration file's `source_ids` against the val/test split files: 0 overlap
- Re-seeding the split handles the train-only variations: every stored variation (1,195 accepted, 525 rejected) is side=train, and `merge_variations.py`:61-65 raises when a variation's source is not in the train split. Before the freeze, variations whose source moved to validation or test are dropped (never promoted to those sides), sources newly on the train side are augmented by the same all-Apache pipeline, and the kept/dropped/new counts are recorded
  - instruction: `merge_variations.py` gains a filter-by-current-split mode with a test; run log shows the three counts
  - honesty: No validation or test entry of the new split has any variation in any training file, and no variation appears on the validation or test side
    - instruction: Check every training variation's `source_id` against the new val/test ids: 0 hits
- The evaluation covers issue 46's cases the spec omitted: behaviour when the correct operation is not among the offered candidates (it should escalate), and the rate of invalid or unparseable outputs (malformed tool call, operation not in nvsh/ops/table.py, arguments failing the operation schema), reported for stock, Track A and Track B
  - instruction: A test-side slice with the gold operation removed from the candidate set; harness counts invalid outputs separately from wrong ones
  - honesty: The removed-candidate slice is built from test entries by removing the gold operation only, without editing any entry's text
    - instruction: Slice builder test: text unchanged, candidate list differs by exactly the gold operation
- Every dataset and model bundle passes a secrets and private-host scan (scripts/scan-secrets.py rules, plus nvsh/redact.py over every text field) before any upload, private or public; a bundle with a finding is not uploaded
  - instruction: Scan output saved next to each bundle; upload step refuses without a clean scan file
  - honesty: The scan runs on the exact folder that is uploaded, after the last change to it
    - instruction: Scan file records the folder's content hash, and the upload step checks it
- Training on spark2 is contained so it cannot take down the model-gear stack: the training process has a hard memory cap well under the roughly 30 GB spark2 has free, and a run that nears the cap is stopped rather than left to the kernel OOM killer
  - instruction: Record the cap mechanism (container --memory or cgroup) and spark2 free memory before and during each run
  - honesty: No model-gear container on spark2 restarts or is OOM-killed while a training run is active
    - instruction: docker ps restart counts and journalctl OOM lines checked before and after each spark2 run
- The training environment is pinned and committed (Python, torch, transformers, peft, trl, unsloth versions plus a lock or requirements file) so h3's reproducibility is possible; today the venv on spark is untracked (transformers 5.5.0, peft 0.21.0, trl 0.24.0)
  - instruction: A scripts/lfm-finetune requirements/lock file; spark2's venv built from it; versions printed in each run log
  - honesty: spark and spark2 train from the same pinned versions
    - instruction: Diff the version lines of both hosts' run logs

## Honesty conditions

- Every figure in the final comparison comes from a pipeline.sh stage run on the clean test side exactly once per reported checkpoint, and the published model's licence and data provenance are checked by `release_bundle.py`/`dataset_bundle.py` before any upload
- No file under nvsh/ changes to serve the Qwen checkpoints; only config (and at most the pins.json image list)
- No training file of either track contains a Jetson skills eval or near-duplicate
- The PR says 'part of #46' and does not close issue 39 or issue 46
- Each Spark trains its own track end to end with no cross-machine training traffic
- The report and cards are readable by someone without access to spark or this session: no private endpoints, keys or home paths, redaction applied
- Both tracks' checkpoints are reproducible from the committed recipe plus the pinned base revision and the committed split seed
- The issue-39 numbers quoted are the committed ones, and flagged as an upper bound because of lapse l5
- Size and memory claims about Orin-class or K8s serving are stated only from measurements taken in this work, never from the base model card
- The percentages are computed over the whole clean test side's operation-expected entries, with the denominator shown next to each percentage
- Abstain means the escalate outcome (per the c25 mapping); precision and recall are computed against escalate-expected entries, and false positives counted over explain- and escalate-expected entries
- Probabilities are the model's own normalised scores over the candidate set, recorded per decision before any threshold; the same computation runs on stock
- Latency and memory are measured while spark has no other training or measurement job running, with background load recorded as in issue 39's measure.py
- A heal run is attributable: its trigger (the measured loss vs bf16) and its train-side data are recorded before it runs
  - instruction: Run log entry precedes the heal run id
- No repo from this work is public before the operator's recorded approval
  - instruction: hf repo info shows private until the approval entry's timestamp

## Success signals

- On the clean test side the best tuned variant gets >=80% right proposals and >= stock + 30 percentage points
  - instruction: Table prints n/N and %
- Escalation (abstain) recall >=80% and abstention precision >=80%; false-positive tool calls <=5% on explain/escalate-expected items; 0 wrong mutating proposals
  - instruction: Harness unit test on a fixture with known counts
- Track B's confidence is calibrated: ECE <=0.10 (10 equal-width bins) and Brier score lower than stock's
  - instruction: Predictions file stores the full candidate distribution per entry
- At 2K context on spark the best variant's warm median decision latency is <=250 ms and its container memory is <=6 GB
  - instruction: measure.py background-load field stays within the documented quiet threshold

## Scope / boundaries

- No nvsh runtime code change is needed to serve a Qwen checkpoint as Tier 2: LfmTier/LfmAgent/TierManager speak OpenAI-compatible chat/tool calls with nothing LFM2.5-specific (manager.py `_build`, agent/registry.py:295-304), so it runs under \[tiers.lfm\] by config alone
  - instruction: git diff --stat main -- nvsh/ at PR time shows no runtime changes beyond pins
- NVIDIA's Jetson skills evals (NVIDIA-AI-IOT jetson-device-skills@20137897 and jetson-bsp-skills@fdfafef0; docs CC-BY-4.0, code Apache-2.0) stay test-only and out of any published bundle; any skills-tuned model that is published carries NVIDIA's CC-BY-4.0 attribution
  - instruction: `jetson_skills.py` scan passes on every training file
- Issue 39 stays open and this work does not close it: its delivery record says it stays open until a tuned model is measured on the held-out split through the committed bench; this work is tracked on issue 46
  - instruction: Check PR body keywords
- Both Sparks train independently in parallel; there is no distributed training across spark and spark2 (issue 46: 'do not require distributed training across the two machines')
  - instruction: Run logs on each host; no torch.distributed init
- Uploads start private; making a model or dataset repo public is irreversible in practice, so each repo goes public only after the operator explicitly approves that repo by name (issue 39's t15 precedent: ask first)
  - instruction: Run log records the approval before the visibility change

## Non-goals

- Renaming the 'lfm' flavor (the \[tiers.lfm\] table, @lfm adapter, ADAPTERS\['lfm'\], doctor check ids, nvsh/tiers/lfm.py, docs/tier2.md, CLAUDE.md tiers paragraph) is not part of this work; it is a wide, mechanical naming change that can follow if a Qwen model is adopted

## Assumptions

- Qwen3.5-0.8B is not a plain small transformer: config.json shows `Qwen3_5ForConditionalGeneration`, a 3:1 Gated-DeltaNet linear-attention / full-attention hybrid (24 layers), an MTP head and a 12-layer vision encoder, native context 262,144. The training venv's transformers 5.5.0 already registers `qwen3_5`, so text-only LoRA is expected to load; the vision tower is unused for this task
- Track B's candidate scoring can build on nvsh's existing next-token logprob plumbing: ToolChat.`score_next_token` and `yes_no_probability`/`calibrated_logit` (nvsh/tiers/toolchat.py:273-300,336-350) already serve the router's LogprobVerifier confidence gate (nvsh/tiers/router.py:55,115,192,216-219); nothing in scripts/lfm-finetune calls them, and no classification-head or pairwise-scoring code exists anywhere
- spark2 (hostname spark-e2f0, GB10, DGX OS, driver 580.178.04, 121 GB unified memory) is reachable over passwordless ssh with uv in the user's tool bin and docker, but has no training venv, no Qwen weights and no vLLM measurement image yet, and it currently runs the model-gear serving stack (a Qwen3.8-27B NVFP4 primary plus rerank, embed and hand vLLM engines, about 72 GB of GPU memory, about 30 GB available)
- The fleet reachable from spark has Jetson AGX Thor (ssh thor) and Jetson AGX Orin (ssh orin), not an Orin Nano; the Orin Nano deployment check runs only if one is made available, otherwise AGX Orin stands in and the report says so

## Scope exploration

- `s1` — `HF API: Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-0.8B-Base`: Both repos exist; cardData.license apache-2.0 with LICENSE file; instruct sha 2fc06364…, base sha dc7cdfe2…; train.py --base/--revision and pipeline.env BASE/`BASE_REV` already parameterise the base (train.py:30-31,114-115; pipeline.env.example:14-15)
  - seeds: `c3`
- `s2` — `Qwen3.5-0.8B config.json + ~/lfm-train/.venv transformers`: Hybrid GDN + vision + MTP architecture confirmed from raw config.json; local transformers 5.5.0 lists transformers.models.`qwen3_5` (checked by import listing); unsloth has a Qwen3.5 fine-tune guide covering 0.8B
  - seeds: `c4`
- `s3` — `scripts/lfm-finetune/train.py loss mask vs Qwen3.5 chat_template.jinja`: train.py:57-74 masks via `return_assistant_tokens_mask`; Qwen3.5 `tokenizer_config` `chat_template` read in full has no generation markers
  - seeds: `c5`
- `s4` — `Qwen3.5 chat template thinking branch + pipeline.sh`: Template emits `<think>\n` when `enable_thinking` is true and an empty think block otherwise; pipeline.sh:55 only disables thinking for augmentation LLMs, not the trained model
  - seeds: `c6`
- `s5` — `scripts/lfm-finetune/build_dataset.py + docs/lfm-finetune.md template pitfalls`: Assistant turn is one of propose(operation, arguments)/explain(text)/escalate(reason) with content ''; LFM-specific ordering and object-args workarounds exist; Qwen's XML format differs
  - seeds: `c7`
- `s6` — `nvsh/tiers/runtime_docker.py + measure-config.toml`: Launcher is config-driven (model/engine/image/ctx/`tool_call_parser`/`hf_offline`); only the vLLM default parser 'lfm2' is LFM-specific; no quantization flag in ENGINES templates; pins.json images list is empty
  - seeds: `c8`
- `s7` — `nvsh/tiers/manager.py, nvsh/tiers/lfm.py, nvsh/agent/lfm.py, registry.py`: Tier 2 wiring is model-agnostic; the Tier contract (base.py Tier.select/close) only selects; propose/explain/escalate and read-only inspection are Tier 2 behaviours
  - seeds: `c9`
- `s8` — `user-visible 'lfm' naming surface`: 'lfm' appears in config table, adapter id, doctor checks (`doctor_checks.py`:831-896,1185-1204), module names, docs/tier2.md title and CLAUDE.md; naming only, no architecture coupling
  - seeds: `c10`
- `s9` — `scripts/lfm-finetune/release_bundle.py + docs/lfm-license-notes.md`: LFM Open License v1.0 conditions rights on the user's revenue under USD 10M, which follows every downstream user; the bundle builder enforces that licence's text and card phrases
  - seeds: `c11`
- `s10` — `scripts/lfm-finetune/split.py`: Split reads the corpus JSON directly, never imports nvsh, names no model; `DEFAULT_SEED` 39 and `DEFAULT_FRACTIONS` (0.70,0.15,0.15)
  - seeds: `c12`
- `s11` — `docs/deliveries/2026-09-22-lfm2-5-350m-tier-2-fine-tune-issue-39.md (lapses l2, l5)`: Delivery record: test split compromised by l2 (grep saw a test entry) and l5 (sup-07 + 3 variations leaked into r8 training); remaining work: fresh/re-seeded test side before any further final run
  - seeds: `c13`
- `s12` — `scripts/lfm-finetune/augment.py + dataset_bundle.py + nvsh/tiers/corpus/dev.json`: augment.py hard-codes no model or endpoint (env-configured roles); `dataset_bundle.py` `ROLE_MODELS` names Qwen 3.6 35B-A3B, Qwen 3.8 27B, Gemma 4 26B-A4B (Apache-2.0) and Nemotron 3.5 Lightning (OpenMDW-1.1); card states teacher licences do not carry over to outputs; no separate OpenMDW-1.1 analysis found
  - seeds: `c14`
- `s13` — `scripts/lfm-finetune/jetson_skills.py + skills_dataset.py`: 104 evals are NEVER trained on (`jetson_skills.py`:8-9,383-385), contamination scan fails a build on near-duplicates (586-612); `dataset_bundle.py` excludes skills data from the nvsh bundle (19-22)
  - seeds: `c15`
- `s14` — `scripts/lfm-finetune/measure.py + measure_skills.py`: grep for logprob|calibrat|ece|brier found nothing in either script; `measure_skills.py` only checks the called skill name
  - seeds: `c16`
- `s15` — `nvsh/tiers/toolchat.py + nvsh/tiers/router.py`: Logprob scoring exists only for routing; docs/tiers-improving-accuracy.md:155-162 calls its thresholds guesses and reports AUC 0.76 in an earlier spike
  - seeds: `c17`
- `s16` — `train.py --max-length + runtime_docker.py ctx`: Both default 4096 and both are flags/config; no code change needed for a 2048 run
  - seeds: `c18`
- `s17` — `issue 39 delivery record + CHANGELOG 0.18.0`: r8 not adopted (1 wrong mutating proposal on test); skills method bar met via s3 52/104 (best-of-three, lapse l4); version 0.18.0
  - seeds: `c19`
- `s18` — `docs/lfm-finetune.md run log + lapse l4`: Hyperparameters are flags (train.py:116-121); skills task has no validation side and issue 39's skills pass was best-of-three against the same 104 evals
  - seeds: `c20`
- `s19` — `ssh spark2 (read-only: nvidia-smi, free, docker ps)`: Four VLLM::EngineCore processes at 50.7/6.5/7.1/7.8 GB; containers model-gear-{gateway,vllm-primary,vllm-rerank,vllm-embed,vllm-hand} up 33 h; free -g shows 90 used / 30 available
  - seeds: `c21`
- `s20` — `issue 46 body (Deployment check)`: Issue states both Sparks work independently in parallel for the initial experiment
  - seeds: `c22`
- `s21` — `fleet memory note + issue 46 Deployment check`: Issue lists Orin Nano 'if available'; the recorded fleet (2026-09-13) is thor and AGX orin only
  - seeds: `c23`
- `s22` — `HF API unsloth/Qwen3.5-0.8B-GGUF + vLLM issues on Qwen3.5 tool parsing`: GGUF quant ladder confirmed via API; llama.cpp GDN-support minimum version unverified; vLLM issues report malformed XML tool calls and a Qwen35CoderToolParser fix PR (unverified which image has it)
  - seeds: `c24`
- `s23` — `repo-wide search for Track A output schema`: No occurrence of '"action": "tool"', 'abstain' or 'Track A' anywhere in code or docs; nvsh's labels are propose/explain/escalate (docs/lfm-finetune.md:135-136; `dataset_bundle.py`:278-286)
  - seeds: `q1` (question, resolved)
- `s24` — `nvsh/tiers/needle.py vs docs/tier2.md (tier contracts)`: Tier 1 is one blocking round trip picking one typed operation from nvsh/ops/table.py, no inspection; Tier 2 inspects read-only ops for up to 4 rounds and ends in propose/explain/escalate; no repo text compares a Tool-Jev to either
  - seeds: `q4` (question, resolved)
- `s25` — `challenge pass / failure-mode lens: scripts/lfm-finetune/merge_variations.py + aug/nvsh-*.jsonl`: Probe: all accepted/rejected variations carry side=train; merge() raises 'source ... is not in the split' (lines 61-65) for any variation whose source leaves the train side after a re-seed
  - seeds: `c45`
- `s26` — `challenge pass / missing counter-evidence lens: issue 46 Evaluation section vs exported spec`: Issue 46 asks to 'evaluate behavior when the correct tool is not among the candidates' and for an unsafe/invalid action rate; spec text had 0 matches for 'not among' and 'unsafe'
  - seeds: `c46`
- `s27` — `challenge pass / security lens: scripts/lfm-finetune/dataset_bundle.py + release_bundle.py`: grep for redact|scan|secret in both bundle builders found nothing; scan-secrets.py only covers repo files, not the upload folders
  - seeds: `c47`
- `s28` — `challenge pass / reversibility lens: public release of model + dataset`: Issue 39 pushed privately only after asking (delivery record t15); the spec says 'share publicly' but had no approval step and no named org/repo
  - seeds: `c48`, `q8` (question, resolved)
- `s29` — `challenge pass / operations + containment lens: ssh spark2 (free, docker ps)`: spark2 runs model-gear-{gateway,vllm-primary,rerank,embed,hand} with about 30 GB available of 121 GB unified memory; an unbounded training process competes for the same pool
  - seeds: `c49`
- `s30` — `challenge pass / unstated-assumption lens: scripts/lfm-finetune/ + the training venv`: No requirements/lock file in scripts/lfm-finetune; docs/lfm-finetune.md pins no versions; spark2 has no venv, so it would be built unpinned
  - seeds: `c50`
- `s31` — `challenge pass / overlooked data-flow lens: issue 46 Track B vs c16/c25`: Issue 46 wants argument exact-match for both tracks, but a candidate scorer outputs a choice, not arguments; no claim says where Track B's arguments come from
  - seeds: `q9` (question, resolved)
- `s32` — `challenge pass / overlooked-actors lens: nvsh/tiers/corpus/held-out.json header`: held-out.json has 0 entries; its header requires a separate sitting and names the operator or real opted-in records as authors
  - seeds: `q10` (question, resolved)
- `s33` — `challenge pass / cheap-probe lens: HF cache on spark`: Only models--Qwen--Qwen3.5-4B cached; the 0.8B template and mask probes were not run in this pass (no download during a read-only sweep)
- `s34` — `challenge pass / adjacent-systems lens: augmentation gateway`: `AUG_URL` is localhost:8001 on spark (model-gear-gateway); spark2 is 192.168.1.193 (not the Pi associate endpoint 192.168.1.138); re-review traffic ends before training per c40, so no overlap with Track B training — clean, residual risk only if c40's ordering slips
- `s35` — `challenge pass / concurrency + observability lens: pipeline.sh stages, measure background load`: pipeline.sh writes `work/<log>.log` + `.done` per stage and measure.py records background load (h26); spark permanently runs model-gear-vllm-multimodal (about 33 GB), so "quiet" means no other training/measure job, not an idle GPU — clean pass with that residual

## Decisions

- Both tracks and the stock baseline are trained and scored on nvsh's propose/explain/escalate tools; a fixed, documented mapping to issue 46's {action: tool|abstain} JSON is used only for reporting
- This work is an experiment: it delivers checkpoints, measurements, the comparison table and a recommendation; nvsh's defaults and docs/tier2.md do not change
- The Qwen tooling generalises scripts/lfm-finetune in place (base-aware template, loss mask, thinking and licence handling) rather than forking a new directory
- Track B trains on spark2 alongside the model-gear stack with capped memory; latency and memory are measured when the stack is quiet or on spark
- The clean test side is a re-seeded split plus fresh held-out entries written without looking at issue 39's test side; issue 39's old test entries are excluded from training with `merge_variations.py` --exclude
  - instruction: Record the new seed and the new entries' author/date in the run log
- Nemotron 3.5 Lightning (OpenMDW-1.1) only cast reviewer B's accept/reject vote; the generator (Qwen 3.6 35B-A3B) and corrector (Qwen 3.8 27B) that wrote every training word are Apache-2.0. Reviewer B is re-run with Qwen 3.8 27B over every stored candidate (about 1,195 accepted + 525 rejected nvsh variations, reusing their generator/corrector text), so every model in the shareable dataset's pipeline is Apache-2.0; the card discloses that reviewer B is also the corrector
  - instruction: `dataset_bundle.py` `ROLE_MODELS` lists only Apache-2.0 teachers; manifest has 0 records naming Nemotron
- Both tracks fine-tune from the post-trained Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17, not the -Base repo
  - instruction: BASE/`BASE_REV` in both env files
- Data is frozen before any training on either Spark: re-split with the new seed and fresh held-out entries (c37), reviewer-B re-review (c38), assemble, record split hashes (h13); only then stock measurement and training start
  - instruction: Run log shows the freeze commit/hashes before the first train or measure run id on either host
- Reviewer B's thinking mode is piloted: about 150 candidates are re-reviewed non-thinking and compared with Nemotron's recorded verdicts; the rest run non-thinking if agreement is acceptable, otherwise thinking stays on as pipeline.sh designed it
  - instruction: Pilot agreement rate and the threshold used are recorded in the run log before the full re-review
- Healing is conditional: each quantized build is calibrated from the train side only (llama.cpp imatrix for GGUF, an AWQ calibration set for INT4) and measured against bf16; only a build that loses more than the margin gets a short quantization-aware fine-tune on the train side and is re-exported
  - instruction: Run log shows the bf16 vs quant numbers before any heal run
- A quantized build may lose at most 3 percentage points of right proposals against its bf16 checkpoint and must add no wrong mutating proposal; the absolute success bars (c33-c36) still apply to it
  - instruction: Comparison table shows bf16 and each quant side by side with the delta
- Fresh held-out entries are drafted by an Apache-2.0 model that is not one of the pipeline's teachers, from the operation table only, reviewed by the operator, and sealed: the agent does not read them before the final run
  - instruction: Run log names the drafting model and licence; the sealed file's hash is recorded before training and checked at the final run
- Track B outputs a scored choice only; its arguments come from nvsh's deterministic grounding (as Tier 1 does) and are scored with the same argument metrics as Track A
  - instruction: Comparison table footnotes Track B's argument source
- Published repos use the jetson-ai-lab org with qwen3.5-0.8b-nvsh-\* names (model, -GGUF, -AWQ) and stay private until the operator approves each one
  - instruction: Env files' REPO values match; visibility changes logged per c48

## Hard questions

- Fine-tune from the post-trained Qwen/Qwen3.5-0.8B or from Qwen/Qwen3.5-0.8B-Base (different EOS token: <|`im_end`|> vs <|endoftext|>)? (resolved: User 2026-09-23: post-trained Qwen/Qwen3.5-0.8B.)
- Which single output format do both tracks and the stock baseline get scored on: issue 46's {"action":"tool",...}/{"action":"abstain"} JSON, or nvsh's existing propose/explain/escalate tools (no mapping between them exists anywhere in the repo), or a documented mapping between the two? (resolved: User 2026-09-23: train and score on nvsh's propose/explain/escalate tools (reuses corpus, measure.py, Tier 2 wiring); document a fixed mapping to issue 46's tool/abstain JSON for the report.)
- Which nvsh tier is the Tool-Jev meant for: Tier 1's one-shot pick-one-operation contract (closer to Track B scoring), Tier 2's propose/explain/escalate loop with read-only inspection (closer to Track A), or neither yet (experiment only, adoption decided later)? (resolved: User 2026-09-23: experiment only — deliver checkpoints, measurements, comparison and a recommendation; adoption into Tier 1 or Tier 2 is decided later from results.)
- Does the Qwen work generalise scripts/lfm-finetune/ in place (base-aware `release_bundle`, template handling), or live in a new directory that cites the shared pieces (split, augment, measure)? (resolved: User 2026-09-23: generalise scripts/lfm-finetune in place (base-aware template, loss mask, thinking and licence handling); one pipeline, rename possibly later.)
- Is the clean test side a re-seeded split of the same corpus, or new held-out entries authored for this work? (resolved: User 2026-09-23: re-seed + fresh held-out entries; old test entries excluded from training.)
- Does data generated or reviewed by Nemotron 3.5 Lightning (OpenMDW-1.1) stay in the shareable dataset, or is it filtered to Apache-2.0 teachers only before publication? (resolved: User 2026-09-23: replace Nemotron-generated/reviewed records by re-running that role with an Apache-2.0 teacher (Qwen 3.8 27B suggested).)
- Can Track B train on spark2 alongside the running model-gear stack (about 30 GB free), or does the stack need to be paused/moved during training runs? (resolved: User 2026-09-23: Track B trains on spark2 alongside the running model-gear stack (~30 GB free), memory capped; latency/memory numbers are taken when the stack is quiet or on spark.)
- Track B scores which operation (or explain/escalate) to take; where do its arguments come from and how is Track B scored on argument exact-match: nvsh grounding like Tier 1, arguments generated after the scored choice, or argument metrics reported for Track A only? (resolved: User 2026-09-23: Track B scores the choice only; its arguments come from nvsh's deterministic grounding (as Tier 1 does) and are scored the same way as Track A's; the report says so.)
- Who writes the fresh held-out entries, and where do they live? nvsh/tiers/corpus/held-out.json ships empty by policy: entries must be written in a separate sitting from dev.json, by the operator or sampled from real opted-in records, not by the agent that wrote the training data (resolved: User 2026-09-23: another Apache-2.0 model that is not a teacher drafts the fresh held-out entries from the ops table only; the operator reviews them; the agent never reads them before the final run.)
- Which Hugging Face organisation and repo names do the public model(s) and dataset use (issue 39 used jetson-ai-lab/lfm2.5-350m-nvsh-triage, private)? (resolved: User 2026-09-23: jetson-ai-lab org, qwen3.5-0.8b-nvsh-\* names (e.g. jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev plus -GGUF and -AWQ); private until approved per c48.)

## Open parks

- [unknown_nonblocking] Whether to drop Qwen3.5-0.8B's vision encoder (and MTP head) from the served artifact to cut size and memory, and whether vLLM/llama.cpp can serve it text-only
- [unknown_nonblocking] Minimum llama.cpp build with correct Gated-DeltaNet (`qwen3_5`) support for Jetson/CPU serving, and which build ships in the Jetson containers
- [unknown_nonblocking] What a representative K8s-friendly serving profile is (CPU-only llama.cpp? small-GPU vLLM?) and its resource limits
- [unknown_nonblocking] Whether current AWQ tooling (e.g. llm-compressor) quantizes Qwen3.5's Gated-DeltaNet layers correctly and whether INT4 AWQ kernels run on GB10 (`sm_121`) in the pinned vLLM image; spike before relying on INT4 AWQ
- [unknown_nonblocking] The Qwen3.5-0.8B weights are not in spark's HF cache (only Qwen3.5-4B is), so this pass could not render the real template, check the loss mask, or confirm unsloth loads `Qwen3_5ForConditionalGeneration` text-only; these stay first-task checks (c5, c7)

## Resolved vagueness

- [unknown_blocking] Whether an unmet 'bar' blocks publication: issue 39 set 0 wrong mutating proposals as its adoption bar; issue 46's success criteria are relative (materially beats stock, reliably abstains) — the numeric bars for this work are not set yet — resolved: User 2026-09-23 chose the numeric bar set: proposals >=80% and >=stock+30pts; abstain recall/precision >=80%; FP tool calls <=5%; 0 wrong mutating; Track B ECE <=0.10 and Brier < stock; <=250 ms warm median, <=6 GB at 2K on spark
