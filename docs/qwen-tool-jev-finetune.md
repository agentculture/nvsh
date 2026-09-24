# Fine-tuning Qwen3.5-0.8B as nvsh's Tool-Jev

Tool-Jev is a small model that decides what to do with one operator request:
propose one of nvsh's typed operations, explain in words, or escalate to the
full agent. Issue 46 asks whether a 0.8B model with an Apache-2.0 licence can
make that decision well, with calibrated confidence and strong abstention.
This work fine-tunes `Qwen/Qwen3.5-0.8B` two ways on two DGX Sparks, reusing
issue 39's LFM2.5 pipeline ([`lfm-finetune.md`](lfm-finetune.md)): a
generative tool caller (Track A, on spark) and a candidate scorer (Track B, on
spark2). Both are scored by one harness on one clean test side against stock.
This page is the design, the code map, the split, a ledger of every problem
hit and its fix, and the steps to reproduce the run.

**Status: in progress, 2026-09-24.** The tooling is built and reviewed, the
spikes are done, the split is re-seeded, the stock baseline is measured on
validation, the training data is frozen, and both tracks have trained and
been measured on validation at least once. Steps not yet run are marked
*(not yet run)*, and anything not checked is marked *(unverified)* or listed
under [Not verified yet](#not-verified-yet). The dated
[run log](#run-log-issue-46) at the end records each step as it happens.

**Licences.** The base, `Qwen/Qwen3.5-0.8B`, is Apache-2.0. Every teacher
model in the data pipeline is Apache-2.0 once reviewer B is re-run (see
[the teacher pipeline](#the-teacher-pipeline)). NVIDIA's Jetson skill evals
are CC-BY-4.0 and are used as a test set only; nothing trained on them is
published here. Training happens on development machines. **nvsh itself
never trains and never uploads.**

## Where the run stands (2026-09-24, about 09h45)

This section is a handoff: exactly what is done, what is running, what is
next, and the exact commands, so the run can be picked up cold.

**Done.**

- Tooling complete and live-checked; the split re-seeded (seed 46); the
  sealed held-out set done (69 entries, sha256 `5eb650f9...`).
- t18, the clean-slate re-review, done (1,042 of 1,176 accepted).
- t19, augmenting the 43 train-side sources without stored variations, done;
  the training data is **frozen** (1,463 examples; decision c40).
- t20, spark2 re-sync, done.
- t21, the stock baseline on validation, done (generative and exact
  scorer). t18-t21 numbers are in [Reproduce it, steps
  5-9](#5-re-review-with-reviewer-b).
- t22, Track A, done: `a1`-`a4` trained, merged and measured on validation
  (and `a3` checked at 4K). **Chosen: `a3`.**
- t23, Track B, done: `b1`-`b4` trained, merged and measured. **Chosen:
  `b1`.** See [Reproduce it, steps
  10](#10-train-track-a-on-spark-a1-a4-done-a3-chosen)-[11](#11-train-track-b-on-spark2-b1-b4-done-b1-chosen)
  and [Choosing a configuration on
  validation](#choosing-a-configuration-on-validation-track-a-and-track-b)
  for both recipe searches.
- **t24, the single final run, done.** Stock, `a3` and `scorer-b1`, each
  measured exactly once on the test side, the sealed held-out set and the
  missing-candidate slice, plus Track A's exact calibration and Jetson
  skills for `a3`. **Neither checkpoint clears every success bar yet** —
  `a3` passes c33 and abstention precision but fails abstention recall,
  false positives, 0-wrong-mutating and (at bf16) latency; `scorer-b1`
  passes c33, abstention precision, 0-wrong-mutating and latency but fails
  abstention recall, false positives and ECE. Full numbers, the pass/fail
  table and the findings are in [Final results
  (t24)](#final-results-t24), sourced from the committed
  `docs/benchmarks/2026-09-24-lfm-{final,heldout}-*.md` and
  `2026-09-24-skills-q46-*.md` pages (commit `db55ce4`).
- Unused serving models on the training and measurement machines were
  stopped, with the operator's OK, to free GPU memory for training and
  measurement to run without an out-of-memory failure (ledger P64); they
  are restored once the run finishes.

**Running.** Nothing.

**Next, in order:**

1. **t25:** `Q4_K_M` and AWQ of `a3` and `scorer-b1`, measured on the test
   side against the same bars; heal only if a build loses more than 3
   points of right proposals or adds a new wrong-mutating id (c42, c43).
   This is also where container memory (c36, not yet measured — both t24
   runs were attach-mode against an already-running server) gets measured
   for the first time.
2. **t26:** the edge check on AGX Orin.
3. **t27:** a private upload, only after asking the operator.
4. **t28:** the report and this guide's final pass.
5. **t29:** `/validate-delivery`, `/summarize-delivery`, a version bump, and
   the PR ("part of #46").

**Obstacles hit along the way** are recorded as ledger entries P56-P66 and
lapses l4/l5 under [Pitfalls hit, and the fix for each](#pitfalls-hit-and-the-fix-for-each):
a leakage-check keying bug, Track B training on the wrong (unfrozen,
un-augmented) file, Track A's first merges scoring bit-identical to the
untuned base because the merge loaded the wrong model class (lapse l4), a
GPU-memory-vs-page-cache measurement hazard on unified memory (P64), a
related but distinct Mamba-cache sizing failure at 4K (P65), and a Track B
scorer measured without `--scorer` scoring as a broken generative model
instead of refusing (lapse l5, P66). See also [Choosing a configuration on
validation](#choosing-a-configuration-on-validation-track-a-and-track-b)
for how each recipe was picked, and
[Troubleshooting](#troubleshooting-symptoms-and-causes) for what each
problem actually looked like on screen.

## What "successful" means

The bars were fixed in the spec before any run
(`docs/specs/2026-09-23-qwen3-5-0-8b-tool-jev-fine-tune-issue-46.md`). They
are judged on the clean test side, against stock Qwen3.5-0.8B re-measured
there:

- **Right proposals** (c33): the best tuned variant gets at least 80% right,
  and at least stock + 30 percentage points.
- **Abstention and safety** (c34): escalation (abstain) recall at least 80%
  and abstention precision at least 80%; false-positive tool calls at most 5%
  on explain- and escalate-expected entries; **0 wrong mutating proposals**.
  Abstention precision is the strict figure (deviation d2, below).
- **Calibration** (c35): Track B's expected calibration error (ECE, 10
  equal-width bins) at most 0.10, and a Brier score lower than stock's.
- **Speed and size** (c36): at 2K context on spark, a warm median decision
  latency of at most 250 ms and container memory of at most 6 GB.
- **Quantized builds** (c43): a quantized build may lose at most 3
  percentage points of right proposals against its bf16 checkpoint and must
  add no wrong mutating proposal. The bars above still apply to it.

**This is an experiment** (c26). It delivers checkpoints, measurements, a
comparison table and a recommendation. nvsh's defaults and
[`tier2.md`](tier2.md) do not change.

## Final results (t24)

**t24, the single final run, is done.** Every model — stock, `a3` and
`scorer-b1` — was measured exactly once at 2K on a quiet machine (no
retries), on the clean test side (64 entries: 32 operation, 15 escalate,
17 explain), the sealed held-out set (69 entries: 41 operation, 13
escalate, 15 explain) and the missing-candidate slice (the 32 operation
entries of the test side, each with its gold operation removed from the
candidates, expected answer escalate). Source of truth: the committed
pages `docs/benchmarks/2026-09-24-lfm-final-{stock,a3,scorer-b1}.md` (and
their `-exact`/`-missing-candidate` variants) and
`docs/benchmarks/2026-09-24-lfm-heldout-{stock,a3,scorer-b1}.md` for the
held-out set, `docs/benchmarks/2026-09-24-skills-q46-{stock,a3}.md` for
Jetson skills — all from commit `db55ce4`. Numbers below are the "Issue 46
metrics" tables in those pages (the `metrics.py`/`ISSUE46_MAPPING` figures
the success bars are judged on), not bench's own per-source vote, which can
differ slightly (see the pages themselves for both).

**Stock.** Test: 2 of 32 right proposals, abstention recall 1 of 15,
precision 33.3%, false-positive tool calls 3 of 32, wrong mutating 3, 36 of
64 invalid, warm 803 ms. Held-out: 6 of 41 right proposals, abstention
recall 1 of 13, wrong mutating 4. Exact scorer (used only as c35's
comparison point, d15): test 20 of 32, abstention recall 0 of 15, ECE
0.167, Brier 0.834; held-out ECE 0.169, Brier 0.803.

**`a3` (Track A, chosen).** Test: 32 of 32 right proposals, abstention
recall 11 of 15 (73.3%), precision 100%, false-positive tool calls 4 of 32
(12.5%), wrong mutating 2 (both wrong operation), warm 442 ms / p95 716 ms.
Held-out: 30 of 41 right proposals, abstention recall 11 of 13 (84.6%),
precision 78.6%, false-positive tool calls 3 of 28, wrong mutating 3 (1
wrong operation + 2 wrong arguments). Missing-candidate slice: abstention
recall 5 of 32, false-positive tool calls 24 of 32, wrong mutating 5.
Exact calibration (`track_a_calibration.py`, d6): test ECE 0.080, Brier
0.156; held-out ECE 0.097, Brier 0.191. Jetson skills at 8K: 44 of 104
(against the d12 floor of 37, stock's 42), not-named 34 of 70 (against the
floor of 23, stock's 28) — **d12 PASS**.

**`scorer-b1` (Track B, chosen).** The served run and the exact in-process
run agree on every decision (only the calibration figures differ). Test:
27 of 32 right proposals (84.4%), abstention recall 11 of 15, precision
100%, false-positive tool calls 2 of 32 (6.3%), wrong mutating 0, 3 not
grounded, warm 23 ms. Held-out: 27 of 41 right proposals, abstention recall
11 of 13, precision 68.8%, false-positive tool calls 4 of 28, wrong
mutating 1. Missing-candidate slice: abstention recall 7 of 32,
false-positive tool calls 18 of 32, wrong mutating 0. Exact calibration:
test ECE 0.132, Brier 0.268; held-out ECE 0.145, Brier 0.312; slice ECE
0.547, Brier 1.198.

**Success bars, on the clean test side, pass or fail with n/N:**

| Bar | `a3` | `scorer-b1` |
|---|---|---|
| c33, right proposals (≥80% and ≥ stock+30pp) | **PASS** — 100% (32/32), stock+94pp | **PASS** — 84.4% (27/32), stock+78pp |
| c34, abstention recall (≥80%) | FAIL — 73.3% (11/15) | FAIL — 73.3% (11/15) |
| c34, abstention precision (≥80%, strict) | PASS — 100% | PASS — 100% |
| c34, false-positive tool calls (≤5%) | FAIL — 12.5% (4/32) | FAIL — 6.3% (2/32) |
| c34, 0 wrong mutating proposals | FAIL — 2 | **PASS** — 0 |
| c35, Track B ECE (≤0.10) | — | FAIL — 0.132 |
| c35, Track B Brier (< stock's 0.834) | — | PASS — 0.268 |
| c36, warm median latency (≤250 ms) | FAIL — 442 ms (bf16) | **PASS** — 23 ms |
| c36, container memory (≤6 GB) | not yet measured (attach mode) | not yet measured (attach mode) |

Neither checkpoint clears every bar yet: both fail abstention recall and
false-positive tool calls on the test side; `a3` also fails 0 wrong
mutating and c36's latency at bf16 (quantization, t25, is the plan's answer
to the latency bar); `scorer-b1` also fails c35's ECE bar despite beating
stock's Brier score by a wide margin. Container memory for both is not yet
measured, since both ran in attach mode against an already-running server;
that measurement is t25's job alongside the quantized builds.

**Findings worth stating plainly:**

- **A 66-entry validation set cannot rule out rare errors.** `a3` had 0
  wrong mutating proposals on validation but 2 on the (also small, 32-entry
  operation) test side. This is not a regression introduced between
  validation and test — it is what a small evaluation set can and cannot
  tell you: a rate low enough to show as 0/32 on one 32-entry sample can
  still show up as 2/32 on a different 32-entry sample from the same
  distribution.
- **The missing-candidate slice is the weakest area for both tracks.** With
  the gold operation removed from the candidate list, the model is expected
  to escalate; instead both tracks mostly pick a near-candidate operation
  instead (`a3`: 24 of 32 false-positive tool calls, abstention recall only
  5 of 32; `scorer-b1`: 18 of 32, abstention recall 7 of 32). This is the
  single biggest gap between "successful" and today's checkpoints.
- **`a3`'s exact calibration is better than `scorer-b1`'s**, on both the
  test side (ECE 0.080 vs 0.132, Brier 0.156 vs 0.268) and the held-out set
  (0.097 vs 0.145, 0.191 vs 0.312) — worth noting even though c35 is scored
  for Track B only, since Track A's own exact-calibration tool (d6) makes
  the comparison possible.
- **Jetson skills improved specifically on not-named prompts:** `a3` beats
  stock by 6 points not-named (34/70 vs 28/70) while scoring lower than
  stock on skill-named prompts (10/34 vs 14/34) — the net overall gain
  (44/104 vs 42/104) is smaller than the not-named gain alone, and both
  clear the d12 regression-guard margin either way.

**Process notes.** The pipeline gained `measure-heldout` and the
`-missing-candidate`/`-exact` label suffixes, with an arguments allowlist,
before this run (deviation d16, commits `6f9ab90`, `ff01844`). The skills
step served at `MEASURE_GPU_FRACTION=0.12` (P65, to avoid the 4K-style
Mamba-cache shortfall at the skills prompt's own long context). Stock's
exact-scorer baseline ran after the 13 planned final-measurement steps,
since c35 needs it only as a comparison point (deviation d15) — being last
in the sequence does not mean it is less final; every model here was still
measured exactly once.

## Design

### Two tracks

- **Track A, generative (spark).** A LoRA fine-tune that writes one tool
  call, the same way issue 39's LFM2.5 did. The system brief, tools and user
  message come from `nvsh.tiers.lfm`, so training matches what Tier 2 sends
  at run time.
- **Track B, candidate scorer (spark2).** The model does not generate. Every
  candidate is listed in the prompt under a one-letter label, and the
  model's next-token log-probabilities over those labels are read once.
  Details below.

The two Sparks train independently. There is no distributed training
(c22).

### nvsh's tools, mapped to issue 46's JSON for reporting

Both tracks and stock are trained and scored on nvsh's own three tools,
`propose(operation, arguments)`, `explain(text)` and `escalate(reason)`
(c25). Issue 46 describes an `{action: tool | abstain}` answer. A fixed
mapping, `ISSUE46_MAPPING` in `scripts/lfm-finetune/metrics.py`, is used for
reporting only:

| nvsh outcome | Issue 46 JSON |
|---|---|
| `propose` | `{"action": "tool", "tool": <operation>, "arguments": <arguments>}` |
| `explain` | `{"action": "no_action"}` |
| `escalate` | `{"action": "abstain"}` |
| `invalid` | `{"action": "invalid"}` |

Issue 46's abstain is nvsh's escalate. Explain has no counterpart in issue
46's pair and is never counted as an abstention. An invalid output is not a
decision.

### Track B: the scorer

`scripts/lfm-finetune/scorer.py`:

- **Candidates:** the 16 operations in `nvsh/ops/table.py`, plus `explain`
  and `escalate`: 18 candidates, labelled `A` to `R`.
- **Score:** one forward pass; the next-token log-probabilities over the
  label tokens are normalised over the offered candidates. The distribution
  is what the calibration metrics score, and its argmax is the choice.
- **Arguments:** when the choice is an operation, its arguments come from
  nvsh's deterministic grounding (decision c52). The scorer never generates
  an argument value. Grounding runs against a fixed snapshot of the machine,
  not the live one (deviation d1, below).
- **Training** (`train_scorer.py`): a LoRA with cross-entropy restricted to
  the candidate label columns, at the one position after the prompt. Qwen's
  vocabulary is 248,320 tokens, so the full-vocabulary loss is never
  computed over the sequence. That keeps a run inside the memory spark2 has
  spare (plan risk r1).
- **Served scoring:** a served model returns only its top-k log-probabilities.
  A label below that cutoff is absent. Such a result is marked incomplete and
  carries no distribution; it is never renormalised over the labels that did
  come back (ledger P19).

### Decisions and deviations that shape the measurement

- **d1, fixed grounding snapshot.** `nvsh.ops.ground` checks argument values
  against the live machine's systemd services and docker containers. Track B's
  argument metrics would then depend on machine state and would not compare
  with Track A's. Both tracks are grounded against one committed snapshot:
  spark's real service and container lists plus every service and container
  name the validation, test and held-out sides reference
  (`measure.py snapshot`, then `--ground-snapshot`).
- **d2, strict abstention precision.** Correct escalations divided by *all*
  escalations, including escalations on explain-expected entries. Bench's own
  figure (explain entries left out of the denominator) is reported alongside.
  The c34 bar is judged on the strict figure.
- **d3, temperature 0.** Qwen ships no `generation_config.json`, and nvsh's
  Tier 2 request sets no temperature, so vLLM sampled at 1.0.
  `gen_config.py` writes a `generation_config.json` (temperature 0,
  `do_sample` false) into every served model directory: the stock copy, every
  merged run, the AWQ build and a healed checkpoint. nvsh runtime code is
  unchanged.
- **d4,** re-split before the re-review, re-review only the train side, with
  4 workers (see [Data and split](#data-and-split)).
- **d5,** this guide.
- **d6, exact Track A calibration.** Generation log-probabilities cannot
  give Track A a candidate distribution (P42). Instead, each candidate's
  tool-call prefix (every operation, plus explain and escalate) is
  teacher-forced under the Track A checkpoint with the training-identical
  prompt, in process, and the scores are normalised over the candidates.
  This runs once per checkpoint on the final side, with
  `track_a_calibration.py` (task h1, merged in `6806662`).
- **d7, one serving setup for every measurement.** Every model (the stock
  copy, Track A, Track B and the AWQ build) is measured in attach mode
  against one committed helper that starts the pinned vLLM with identical
  flags: the pinned image digest, `qwen3_coder`, automatic tool choice,
  `--max-logprobs`, `--limit-mm-per-prompt` with image and video 0, and each
  model directory's own `generation_config.json`. Each `measure.py` call
  measures one model. Proposed by the lead and approved by the operator.
  Built as task h2 (`serve_for_measure.sh` and the pipeline's measure
  stages, merged in `1062123`).
- **d8, the re-review is a clean slate.** The fresh, thinking reviewer B
  verdict plus the deterministic guards decide each candidate; stored
  verdicts are kept only as `prior_verdicts` and never bias the fresh call.
  Each reviewer call is one independent two-message request (the rules plus
  one candidate's text and expected answer; no history, no examples, no
  prior verdict). `augment.py --rederive-clean-slate` re-derives what the
  old rule would have said, offline, for comparison only. Operator decision,
  2026-09-23 ~20:40. *Merge:* `ce72f01` (`f13`); Codex found 5 issues in the
  offline re-derive (duplicate ids twice, a wrong agreement count, a
  needless reviewer config, dry-run writing output), fixed in `ccd8a03`.
- **d9, reviewer temperature 0.2.** `NVSH_AUG_<ROLE>_TEMPERATURE`, recorded
  per record and per verdict; the generator keeps 0.7. Timeout 900 s,
  `--workers 2`. Reason: a probe of two real reviewer calls found about 90%
  of wall time was queue wait on the shared gateway model, not thinking, and
  the operator judged temperature 0.7 too hallucination-prone for a judge.
  All 1,176 candidates are redone at 0.2 so every verdict comes from one
  setting; the 0.7 clean-slate outputs are archived (not committed).
  *Merges:* `7a2810a`/`e9d8495` (the temperature knob), `1a35540` (the
  decision record).
- **d10, reviewer prompt fix, conservative parser, calibration.** A wording
  fix ("run this read-only check (its output is shown to the user as the
  answer)") stopped the reviewer misreading a correct read-only proposal as
  incomplete. The verdict parser stayed deliberately conservative — a
  loosening tried during this fix let real rejections through and was
  reverted — and was hardened over three Codex review rounds (markdown- and
  underscore-wrapped "no", a trailing "no", ", no,", "Yes, not ...",
  certainty words in the first four words). Replay of 977 stored accepts: 0
  newly rejected. A new deterministic `handoff_check` guard rejects requests
  that ask for the hand-off in words ("escalate this to a human agent"): 2
  of 1,176 candidates, no false positives against phrases like "privilege
  escalation" or "UEFI handoff". `NVSH_AUG_<ROLE>_REASONING_EFFORT` was
  added and is recorded per call; `calibrate_reviewer.py` builds a
  deterministic known-good/known-bad probe (seed 46) to check a reviewer
  setting before a full run. *Merge:* `56765d1` (`f15`).
- **d11, reviewer B alone decides t19.** For the fresh variations t19 will
  generate, `augment.py --decide-by reviewer_b` lets reviewer B's verdict
  plus the deterministic guards decide on their own; reviewer A is still
  asked and its verdict is still recorded, but does not gate acceptance.
  Every record carries both `decided_by` and both verdicts, so the
  comparison is not lost. Reason: reviewer A (Gemma 4 26B-A4B, no thinking)
  probed at 3 false accepts and 2 false rejects, mostly on escalate versus
  a listed check — it answers "can the assistant handle this" rather than
  "is this response right". Proposed by the lead; Codex review found no
  issues. *Merge:* `ed01957`, recorded in `580675a`.
- **d12, stock's test-side run happens once, in t24.** Stock's only
  test-side and held-out measurement runs in t24, alongside the tuned
  checkpoints, rather than as an earlier separate baseline pass. The Jetson
  skills regression margin is committed before any tuned-checkpoint score
  is seen: a tuned checkpoint passes the skills check if its overall
  accuracy is at least stock's minus 5 points (out of 104) and its
  not-named accuracy is at least stock's minus 5 points (out of 70) — a
  regression guard reported with the raw counts, not a fixed pass bar.
  Committed at `580675a`, before the skills run.
- **d13, skills evals are served at `MEASURE_CTX=8192` for every model.** A
  single skills prompt listing all 38 tools is 3,939 tokens: at 2K every one
  of the 104 evals was a call error (the tier-error gate refuses to write a
  results page at all), and 4K leaves no room for the reply. 8K is the
  smallest context that leaves headroom for both the prompt and the answer,
  and it is used for stock and every tuned checkpoint alike so the skills
  numbers stay comparable.
- **d14, leakage checking before assembly.** Of the 95 train-side sources
  that had no stored variations, 52 are old issue-39 test entries (already
  excluded from training); t19 augments only the remaining 43. A new
  `leakage_check.py` flags a variation as leaked against exact match, a
  5-token shingle Jaccard of at least 0.8, or a word-set Jaccard of at least
  0.8 for texts of 4 or more words (ids only — it never reads or logs the
  matched text, so it is safe to run against the sealed held-out).
  `assemble` now merges with `--filter-to-split`, excludes validation, test
  and `PROTECTED_EXTRA` (the issue-39 test split, the corpus held-out and
  the sealed held-out), and runs `leakage_check.py` before rendering. A dry
  run on the current data found 52 issue-39 test sources and 1 train source
  identical to a test entry inside the 301-entry train split, plus 7
  re-reviewed variations matching protected text (2 against the sealed
  held-out, 2 against validation, 1 against test, 2 against issue-39's old
  test side). *Commit:* `4580c06`, then Codex found `leakage_check.py` keyed
  protected files by basename (so the new split's `test.json` and issue 39's
  `test.json`, both in `PROTECTED_EXTRA`, collapsed into one file), filtered
  by id (which could drop unrelated rows sharing an id) and passed missing or
  null text unchecked; fixed to key by path, filter by row index and exit 2
  on missing text (`6ad2ab3`, ledger P56). *Merge:* `9b6e559` (`f17`).
- **d15, Track B calibration via the exact in-process scorer.** The served
  scorer returned no complete label distribution on any of the 66 validation
  entries: the fine-tuned scorer's other label logprobs fall outside vLLM's
  top 22 returned logprobs (risk r8), and the harness correctly refuses to
  renormalise a partial top-k result rather than report a distorted one.
  Track B's calibration (ECE, Brier) is therefore reported from the
  `--scorer in-process` run, not the served run; the two runs' *decisions*
  and latency still come from the served run, and agreed with the in-process
  run entry for entry on `b1`. *Merge:* `b4eeed6` (`f22`); recorded in
  `673ccb5`.

### Quantization plan

The best checkpoint(s) are exported two ways (c44) and measured with the
same harness as bf16:

- a text-only `Q4_K_M` GGUF (no vision `mmproj`), imatrix-calibrated from
  the train side, served by llama-server;
- an INT4 AWQ (W4A16) checkpoint served by vLLM.

Healing is conditional (c42, c43): `quantize.py`'s `heal_needed()` is true
when right proposals drop by more than 3 points against bf16, or when the
quantized build has a wrong mutating proposal on an entry id the bf16 build
did not. Only then does a short fine-tune on the train side run
(`pipeline.sh heal`). Calibration and healing data come from the train side
only.

## Code map

Everything lives in `scripts/lfm-finetune/`. None of it is imported by the
nvsh package. These files were added or changed for issue 46:

| File | What it does |
|---|---|
| `requirements-train.txt` | Pins the training stack (Python 3.12.12; torch 2.12.1 from the CUDA 13.0 index; transformers 5.5.0, peft 0.21.0, trl 0.24.0, unsloth 2026.9.9, unsloth-zoo 2026.9.7 and the rest). |
| `train.py` | Track A LoRA. For a template without generation markers it locates the answer span in the rendered ids; renders with thinking off where the template supports it. `--max-length` (default 4096), `--merge-only`. |
| `build_dataset.py` | Builds chat-format training lines from a split side. Now renders one example per outcome with the base tokenizer and parses it back (Qwen XML form); a mismatch fails the build. `--base`, `--revision`, `--no-verify-render`. |
| `merge_variations.py` | Folds accepted variations into the train side. New: `--filter-to-split` drops (and counts) variations whose source left the train side; duplicate variation ids are refused. `--exclude` drops leaks into val/test. |
| `split.py` | Seeded 70/15/15 stratified split. Unchanged; run with seed 46. |
| `augment.py` | The generate/correct/review pipeline. New: `--rereview` re-runs reviewer B only over stored candidates; `--workers` applies to it; `--limit`/`--sample` for the pilot; `--dry-run` never calls a reviewer. |
| `dataset_bundle.py` | Builds the data set upload folder. Teachers come from a run file (`--teacher-models`); `--apache-only` refuses a non-Apache teacher; the card lists every teacher per role. |
| `release_bundle.py` | Builds the model upload folder. New: `--licence-kind apache`, `--tool-call-parser` (the card's nvsh snippet). |
| `scan_bundle.py` | New. Scans a bundle with scan-secrets rules plus `nvsh.redact`, including decoded JSON string values and private hosts in prose; writes `scan.json` with the folder hash; `verify` refuses a changed folder. |
| `eval_slices.py` | New. The missing-candidate slice: each operation entry's gold operation removed from the candidates, expected answer escalate, text unchanged. |
| `metrics.py` | New. The predictions JSONL schema and every issue 46 metric: right proposals, escalation recall and precision (bench and strict), false-positive tool calls, wrong mutating, invalid outputs, ECE, Brier, tokens, time to first decision. Holds `ISSUE46_MAPPING`. |
| `scorer.py` | New. Track B scoring over label tokens, grounded arguments, incomplete-result handling. |
| `train_scorer.py` | New. Track B LoRA on label-restricted cross-entropy; seeded; logs each split file's sha256. |
| `quantize.py` | New. Train-side-only calibration set (checks split headers), bf16 GGUF convert + imatrix + `Q4_K_M`, INT4 AWQ through `awq_oneshot.py` in the AWQ venv (`AWQ_PY`), support files and `generation_config.json` for the AWQ output, tool versions, `heal_needed()`. |
| `awq_oneshot.py` | New. Runs inside the separate AWQ venv: `AWQModifier` W4A16 with `lm_head`, vision, linear-attention and MTP ignored, `oneshot` with `processor=`, generation config sanitized before save. |
| `gen_config.py` | New. `write`, `check` and `stock-copy` of a `generation_config.json` pinning greedy decoding (d3). |
| `capped.sh` | New. `run_capped`: a hard RAM and swap cap through `systemd-run`, with a free-memory log. |
| `draft_heldout.py` | New. Drafts the sealed held-out set with Qwen3.5-4B (pinned revision) from the operation table only; prints counts and a hash, never entry text (step 4a). |
| `track_a_calibration.py` | New. Exact Track A candidate distributions by teacher-forcing each candidate in process (d6); fills a predictions file's `candidates` and writes a provenance sidecar. |
| `serve_for_measure.sh` | New. `start`, `wait` and `stop` the pinned vLLM for one model with the fixed measurement flags (d7); refuses an undigested image or a model directory without a valid `generation_config.json`. |
| `pipeline.sh` | New stages (below); training stages run capped. |
| `pipeline-qwen.env.example` | New. The Qwen configuration: `BASE`, `BASE_REV`, `REPO`, seed 46, the four teacher roles, `TRAIN_MEMORY_MAX`. |
| `measure.py` | Writes the shared predictions file and scores it with `metrics.py`. New: `--predictions`, `--scorer served\|in-process`, `--max-logprobs`, the `snapshot` subcommand and `--ground-snapshot` (d1), `--enable-thinking`, `--slice full\|missing-candidate`, `--ctx`. `--final` is still required for the test side. |

`pipeline.sh` stages, in order: `split`, `skills`, `stock-copy`,
`augment-nvsh`, `augment-skills`, `rereview`, `filter-variations`,
`assemble`, `train`, `train-scorer`, `measure-val`, `measure-final`,
`measure-skills`, `scan`, `quantize`, `heal`, `upload`, `status`. The
`upload` stage refuses without `FINAL=1`, a passing `scan_bundle.py verify`
on the exact folder and a passing `gen_config.py check`, and always creates
the repository private.

## Data and split

### The corpus and the split

The corpus is `nvsh/tiers/corpus/dev.json`: 431 entries, of which 212 expect
an operation, 116 an explanation and 103 an escalation. `split.py` makes a
seeded 70/15/15 split stratified by answer kind. Issue 39 used seed 39; this
run uses **seed 46**:

| Side | Entries | sha256 |
|---|---|---|
| train | 301 | `0425a5754f2adfdbdb9d7e20d6d02596f60fc6001ce848bd8cc8c84e7ddb60ab` |
| val | 66 | `6e4c625ccaecfa4f73867542634dc0eb2934bb9bbede7e7406e2f4140baf7dd3` |
| test | 64 | `17add7ba8acb918edb8cfddf8e4f1e6675f294673a1d55f0e963cb146bb29527` |

Why re-seed: issue 39's test side is not clean. It was partly read during
iteration (lapse l2) and four test entries reached r8's training by exact
wording (lapse l5). The clean test side here is the re-seeded split plus a
sealed held-out set (decision c37). Issue 39's old test entries are kept out
of training with `merge_variations.py --exclude`. **Only counts and hashes
of the test side are ever printed; its contents are not read.**

### Variations and the re-split

Every variation keeps its source's side. The 1,720 stored nvsh candidates
from issue 39 (1,195 accepted, 525 rejected) were all train-side under seed
39. After re-splitting with seed 46:

- **1,176 of the 1,720** keep their source on the new train side. They are
  the re-review input.
- The rest have sources now on validation or test. They are dropped, never
  promoted to those sides. `merge_variations.py` used to raise on them;
  `--filter-to-split` now drops and counts them (ledger P24).
- **95 new train-side sources** had no variations yet: 52 are old issue-39
  test entries, already excluded from training; the remaining **43** were
  augmented by the same all-Apache pipeline (t19, done — 258 variations, 229
  accepted, decided by reviewer B alone per d11).

**The frozen training set** (decision c40; any later change is a recorded
deviation): 1,463 rendered examples from 262 sources (248 original plus 14
supplement) and 1,201 variations — propose 725, escalate 357, explain 381,
all 16 operations covered (27-91 each):

| File | sha256 |
|---|---|
| `splits/train.json` | `0425a5754f2adfdbdb9d7e20d6d02596f60fc6001ce848bd8cc8c84e7ddb60ab` |
| `splits/val.json` | `6e4c625ccaecfa4f73867542634dc0eb2934bb9bbede7e7406e2f4140baf7dd3` |
| `splits/test.json` | `17add7ba8acb918edb8cfddf8e4f1e6675f294673a1d55f0e963cb146bb29527` |
| `data/train-augmented.json` | `910a6224585e739eac6bc2ea6faed415b3fed52b2ccf7ed997746b7620bf5fc9` |
| `data/nvsh-train.jsonl` | `47b6e7b031b5cafdffb086f3ed804cfd86d6b56fc3fd39822d98b5c7bcdd580d` |
| `data/leakage.json` | `6fd8e50148a31d354118c6a1f600eb668692bdbf63f9d0aed56908b6e053092a` |
| `ground-snapshot.json` | `0d79c8fe63cef6a9b2e38a1719d173d59c4e793eed5b9ca3fe7b335148dc533b` |
| `aug/nvsh-accepted.jsonl` | `9b2902e2e2849dbdd205d0a70a937731b7c6470468a908334d4453a4fe64f39f` |
| `aug/rereview-accepted.jsonl` | `27b2010834b508acceb549070eca585e9636c2cc03cfe6350bc895c270ef2b38` |
| `aug/rereview-rejected.jsonl` | `7692af9a1f91a8b63984ecec04c09b65791adfbdc9c7530aa5adbe3bbc42b830` |
| `aug/new43-accepted.jsonl` | `8f6be53118f47badd714ba8ffd7179b1f609699c002d86141b985d2a66efea8a` |
| `aug/new43-rejected.jsonl` | `f29ce96ffc28476edd4173fd6b4e9c11c6fcdc9332d960225ff7634187ce6091` |
| `skills/tools.json` | `6948a39e9072ddcad0d2d79c69308b7f3cb4e03e116e3701e7a2b496849a4190` |
| `skills/test.jsonl` | `44d3aa783c0cd332f697606aeb6d83c7b55958f9eb8802f254011a1faedcfca0` |

All 8 files that matter for training (the three splits, `train-augmented.json`,
`nvsh-train.jsonl`, `ground-snapshot.json` and the two skills files) were
copied to spark2 and verified byte-identical.

### The teacher pipeline

`augment.py` has four roles. The generator rewrites a train request without
seeing the answer, the corrector copyedits it, and two reviewers must both
accept it. Deterministic checks reject any rewrite that names an operation
identifier or copies the answer's wording. Teachers for this run, from
`pipeline-qwen.env.example`:

| Role | Model | Licence |
|---|---|---|
| generator | Qwen 3.6 35B-A3B | Apache-2.0 |
| corrector | Qwen 3.8 27B | Apache-2.0 |
| reviewer A | Gemma 4 26B-A4B | Apache-2.0 (as stated in the env example) |
| reviewer B | Qwen 3.8 27B (re-review) | Apache-2.0 |

In issue 39, reviewer B was Nemotron 3.5 Lightning (OpenMDW-1.1). It only
ever cast reviewer B's vote; the generator and corrector wrote every word.
175 stored candidates had been rejected by Nemotron alone. To make every
model in the shareable data set's pipeline Apache-2.0, reviewer B is re-run
with Qwen 3.8 27B over the stored candidates, reusing their generator and
corrector text (decision c38). Qwen 3.8 27B is also the corrector; the data
set card discloses that.

Reviewer B's thinking mode was piloted first, with the rule fixed before the
run (c41): 150 seeded candidates, non-thinking; if agreement with the stored
Nemotron verdicts reached 80%, the full run would go non-thinking. It reached
**56 of 150 (37%)**, so the full re-review runs with thinking on. See the
[run log](#run-log-issue-46) for the pilot and the probe.

### The sealed held-out set

Qwen3.5-4B (Apache-2.0, not a teacher) drafted 72 fresh entries from the
operation table only (seed 46, temperature 0.7, thinking off). The operator
reviewed and edited them. The sealed file has **69 entries: 41 operation, 13
escalate and 15 explain**, covering 15 of the 16 operations (`network_info`
has none; the operator chose to keep it that way).

| File | sha256 |
|---|---|
| draft as generated (v1) | `95c7cd3eb854f1b24a133a3f77df71a162369a91a5299d2696bbcb199ae2107f` |
| sealed, read-only | `5eb650f91c44f54ab112665dc40d71f66fe118efc79d8bcea728ed9ce0a40198` |

Structural checks, run without reading any entry text: unique ids, every
expectation well formed, every operation entry passes
`nvsh.ops.table.validate`, and 0 exact overlaps with `dev.json` or the 1,720
stored variations. The lead has not read it and will not read it before the
single final run, t24 (decision c51). The procedure is
[step 4a](#4a-draft-and-seal-the-held-out-set).

### Leakage guards

- `split.py` refuses the held-out file and duplicate ids.
- `merge_variations.py --exclude val.json test.json` drops any variation that
  repeats a validation or test entry (counted as `leaked`) and refuses such a
  supplement entry; duplicate variation ids are refused.
- `dataset_bundle.py` refuses `held-out.json` and a train record that repeats
  a validation or test entry.
- `quantize.py` refuses calibration data from validation or test, checking
  each split file's header, not only its name.
- `measure.py` needs `--final` for the test side and `--acceptance` for the
  held-out file, and refuses `--details`/`--predictions` with either, so
  failures there are never studied.
- The Jetson skills contamination scan must be clean before training
  *(not yet run for this split)*.

## Reproduce it

Commands run from the nvsh checkout. `$WORK` is the training work directory
and `$TRAIN_PY` the training venv's Python, as in
`scripts/lfm-finetune/pipeline-qwen.env.example`. Copy that file to a
private `qwen.env`, fill it in, and define:

```bash
P=scripts/lfm-finetune/pipeline.sh
```

### 1. Training environment (spark and spark2)

On each DGX Spark (GB10, aarch64, CUDA 13.0), a venv outside the
repository, built from the pinned requirements:

```bash
uv venv --python 3.12 <venv dir>
uv pip install --python "$TRAIN_PY" -r scripts/lfm-finetune/requirements-train.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
```

`--index-strategy unsafe-best-match` lets uv take versions from both
indexes, not only the first index that has a package. spark's venv is the
issue 39 venv the pins were read from. spark2's was built with exactly these
commands and came out identical: torch 2.12.1+cu130, transformers 5.5.0,
peft 0.21.0, trl 0.24.0, unsloth 2026.9.9, CUDA available (h34).

Two things on spark2 (ledger P33):

- `uv` is in the user's tool bin directory, which is not on the `PATH` of a
  non-interactive `ssh spark2 <command>`. Call it by its full path.
- spark2's shared Hugging Face cache is owned by root (the serving
  containers created it). Point `HF_HOME` at a private directory inside the
  spark2 work directory instead. Never change the ownership of a directory
  the serving stack uses.

### 2. Base model

Download `Qwen/Qwen3.5-0.8B` at commit
`2fc06364715b967f1860aea9cf38778875588b17` into the Hugging Face cache
(`HF_CACHE`). Then make a self-contained stock copy with greedy decoding
pinned:

```bash
$P --env qwen.env stock-copy
```

This copies the snapshot to `$WORK/stock` with symlinks resolved (so it can
be bind-mounted into a container) and writes its `generation_config.json`.

### 3. Serving image and parser

The measurement config (`NVSH_CONFIG`) uses the pinned vLLM image
`vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`
(vLLM 0.26.1rc1.dev942) with `tool_call_parser = "qwen3_coder"`. Track B
served scoring needs vLLM started with `--max-logprobs` of at least 22 (18
labels plus 4), or in-process scoring (plan risk r8).

### 4. Split

```bash
python scripts/lfm-finetune/split.py --corpus nvsh/tiers/corpus/dev.json \
  --out-dir "$WORK/splits" --seed 46
sha256sum "$WORK"/splits/{train,val,test}.json
```

Compare the hashes with the table above. Print counts and hashes of the
test side, never its contents. The same files are copied to spark2 byte for
byte *(not yet run)*.

### 4a. Draft and seal the held-out set

The held-out set is written fresh, so no model or person tuning the run has
seen it. The agent running the experiment never reads its text.

1. **Draft** with an Apache-2.0 model that is not one of the teachers, from
   the operation table only, never from the corpus or any split. From the
   repository root, with the training venv's site-packages on the path and
   `Qwen/Qwen3.5-4B` already in the Hugging Face cache:

   ```bash
   PYTHONPATH=<training site-packages>:. python scripts/lfm-finetune/draft_heldout.py <out dir>
   ```

   `draft_heldout.py` loads Qwen3.5-4B pinned at revision
   `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with transformers (bf16, on
   the GPU) and samples with seed 46, temperature 0.7, top_p 0.9 and
   thinking off. It sends 18 prompts (3 requests per operation, 16
   escalate, 16 explain) that contain only the operation table: names,
   descriptions and argument specs. It reads `dev.json` only to drop exact
   repeats. It writes `draft.json` and `raw-generations.jsonl` to the output
   directory and prints counts and a sha256, never entry text. This run
   drafted on spark; the committed script's prompt strings were checked
   byte-identical to the ones that ran.
2. **Keep the draft as generated** as a separate v1 file and record its
   sha256.
3. **The operator reviews and edits** the draft in a separate sitting. The
   agent does not see the edits.
4. **Validate the structure without reading text**: the file parses as JSON
   (one intermediate save in this run did not, and the operator fixed it),
   ids are unique, every expectation is well formed, every operation entry
   passes `nvsh.ops.table.validate`, and no entry exactly repeats a corpus
   entry or a stored variation. Print only counts and per-operation
   coverage.
5. **Seal**: write a read-only copy and record its sha256. It is opened
   only by the final run, with `measure.py --acceptance`.

### 5. Re-review with reviewer B

Filter the stored candidates to those whose `source_id` is on the new train
side (1,176 of 1,720 in this run) *(the filter command is not yet recorded
here)*. Pilot first, then the full run. The gateway key comes only from
`grant run --inject`:

```bash
grant run --inject NVSH_GATEWAY_KEY=<secret> -- $P --env qwen.env rereview
```

The `rereview` stage reads `$WORK/aug/nvsh-accepted.jsonl` and
`nvsh-rejected.jsonl` and writes `*-rereview.jsonl` files to inspect and copy
over by hand. This run instead called `augment.py --rereview` on the
filtered input with `--workers 4` (deviation d4) *(exact invocation
unverified)*. `--dry-run` reports the count and exits; `--sample N` runs the
pilot.

The re-review that actually ran (t18) went through several settings changes
before it reached the fixed configuration; see the [run
log](#run-log-issue-46) for the stall, the timeout fix and the effort
calibration that produced them. The final, fixed settings are: **clean
slate** (d8: fresh reviewer B verdict plus deterministic guards decide;
stored verdicts kept only as `prior_verdicts`), **temperature 0.2** (d9),
**`reasoning_effort` xhigh** (d10, the calibration probe's choice), **900 s
timeout, `--workers 2`** (d9 — matched to the gateway's `--max-num-seqs=2`
so no request queues behind another). Before a full run at a new setting,
run the calibration probe first:

```bash
grant run --inject NVSH_GATEWAY_KEY=<secret> -- \
  python scripts/lfm-finetune/calibrate_reviewer.py --seed 46
```

`calibrate_reviewer.py` builds a deterministic known-good/known-bad probe
(good pairs per class; bad pairs: a different read-only check, a mutating
change for a read request, escalate for an operation request, an operation
for an escalate request, a hand-off request) and reports false accepts and
false rejects per `reasoning_effort`. Only then run the full re-review:

```bash
grant run --inject NVSH_GATEWAY_KEY=<secret> -- \
  $P --env qwen.env rereview --workers 2
```

### 6. Augment the new train sources (t19, done)

Only the 43 train-side sources without a stored variation and without a
matching issue-39 test entry are augmented (d14); the other 52 of the 95 are
old issue-39 test sources and are excluded from training regardless.
Reviewer B alone decides acceptance (d11); reviewer A is still asked and
recorded, but never gates it:

```bash
grant run --inject NVSH_GATEWAY_KEY=<secret> -- \
  $P --env qwen.env augment-nvsh --decide-by reviewer_b
$P --env qwen.env filter-variations
```

`augment-nvsh` is resumable and skips variation ids already written.
`filter-variations` counts accepted variations against the train split
without touching `assemble`'s output. `--decide-by reviewer_b` records both
verdicts and `decided_by` on every new record, so the two reviewers' outputs
stay comparable even though only one decides.

This run used `run-augment-new43.sh` over the 43 sources and produced 258
variations: 229 accepted by reviewer B plus the deterministic guards, 29
rejected; reviewer A alone would have rejected 60 of the 229 (ledger P55, the
reason d11 exists). 0 errors, 0 retries, about 70 minutes.

### 7. Assemble and freeze (done)

```bash
$P --env qwen.env assemble
```

This merges the variations with `--filter-to-split`, excludes validation,
test and `PROTECTED_EXTRA` (issue 39's old test split, the corpus held-out
and the sealed held-out — d14), runs `leakage_check.py` against every
protected side (exact match, or a 5-token shingle or word-set Jaccard of at
least 0.8; ids only), and builds `$WORK/data/nvsh-train.jsonl` from the
filtered file only, round-tripping one rendered example per outcome through
the Qwen template. After this the data is frozen; any later change is a
recorded deviation (decision c40). `leakage_check.py` can also be run by
hand, printing counts and matching ids only (never text), so it is safe to
run against the sealed held-out before the final run:

This run's assemble: `nvsh-accepted.jsonl` (the re-review's accepts plus
t19's accepts) held 1,271 records with 0 duplicate ids. `merge_variations`
folded 315 sources (301 split plus 14 supplement) into 1,206 kept variations,
59 duplicates and 6 exact protected matches excluded, 0 off-split.
`leakage_check` then dropped 58 of the remaining 1,521 candidates (53 exact
matches against issue 39's old test split, 2 against test, 2 against
validation, 1 against the sealed held-out; 53 exact, 5 near-duplicate).
Rendering produced **1,463 training examples**: 262 sources (248 original
plus 14 supplement), 1,201 variations; propose 725, escalate 357, explain
381; all 16 operations covered (27-91 examples each). The frozen file's
hashes are under [Variations and the re-split](#variations-and-the-re-split).
The frozen set was copied to spark2 and verified byte-identical (8 files:
the three splits, the training file, `nvsh-train.jsonl`, the ground snapshot
and the two skills files).

The Jetson skills contamination scan (`jetson_skills` build) took hours on
the first pass over the frozen file, because every rendered training string
repeats the same long system prompt (the tool table); f18 deduplicates
training strings before scanning, giving the identical result by
construction in about 6 seconds (104 evals, clean). *Merge:* `b869c5d`.

```bash
python scripts/lfm-finetune/leakage_check.py --train "$WORK/data/nvsh-train.jsonl" \
  --protected "$WORK/splits/val.json" "$WORK/splits/test.json" <held-out file> \
  [--out-filtered <cleaned file>]
```

With `--out-filtered` it writes the training file without the matched
entries and exits 0; otherwise any match exits 1.

### 8. Grounding snapshot

```bash
python scripts/lfm-finetune/measure.py snapshot --out "$GROUND_SNAPSHOT" \
  --from-split "$WORK/splits/val.json" --from-split "$WORK/splits/test.json"
```

Add the held-out file with another `--from-split`. The command prints
counts only. `GROUND_SNAPSHOT` is a required key in the env file, and the
pipeline's `measure-val` and `measure-final` stages pass it to every
measurement.

This run's snapshot: 263 services (253 read from spark, 10 more only in the
validation/test/held-out split files) and 34 containers (30 from spark, 4
more only in the split files). As always, only counts were printed; entry
text was not read.

### 9. Stock baseline (t21, done on validation)

Every model is measured the same way (deviation d7). The pipeline's measure
stages take one name each, `stock` for the stock copy in `$WORK/stock` or a
run name for `$WORK/runs/<name>/merged`, and for that one model they:

1. start the pinned vLLM with `serve_for_measure.sh` and wait until it
   answers;
2. write an attach-mode nvsh config, `$WORK/measure/<label>.nvsh.toml`;
3. run `measure.py` against it with `--ground-snapshot "$GROUND_SNAPSHOT"`,
   `--enable-thinking` from `ENABLE_THINKING` (default false) and
   `--max-logprobs "$MEASURE_MAX_LOGPROBS"`, forwarding any extra arguments;
4. stop the server on exit.

The helper's settings come from the env file:

```bash
MEASURE_IMAGE=vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695
MEASURE_PORT=18060
MEASURE_CTX=2048                # vLLM --max-model-len
MEASURE_GPU_FRACTION=0.08       # vLLM --gpu-memory-utilization
MEASURE_MAX_LOGPROBS=22         # 18 labels + 4 (plan risk r8)
TOOL_CALL_PARSER=qwen3_coder
```

`serve_for_measure.sh start <model dir> <port> [record.json]` runs container
`q46-measure-<port>`, listening on 127.0.0.1 only, with the model directory
mounted read-only at `/model`, `HF_HUB_OFFLINE=1`, and the flags
`--served-model-name`, `--max-model-len`, `--gpu-memory-utilization`,
`--enable-auto-tool-choice`, `--tool-call-parser`, `--max-logprobs` and
`--limit-mm-per-prompt '{"image": 0, "video": 0}'`. It refuses an image
without a digest and a model directory that fails `gen_config.py check`, so
every model decodes greedily from its own `generation_config.json`.
`serve_for_measure.sh wait <port>` and `stop <port>` complete it.

Measure stock on validation first, then the test side, the held-out set and
the missing-candidate slice once each (step 12), at 2K and 4K, before any
tuned run is scored:

```bash
$P --env qwen.env measure-val stock
```

For 4K, **export** `MEASURE_CTX=4096` before calling the pipeline rather
than only setting it in the env file: an exported `MEASURE_CTX` outranks the
env file's value (lapse l3, below), and `measure-val`/`measure-final`
already always pass `--ctx "$MEASURE_CTX"` themselves — passing your own
`--ctx` on top is refused. A non-2048 `measure-val` run is named
`<name>-val-ctx<N>` (`stock-val-ctx4096` here).

```bash
export MEASURE_CTX=4096
$P --env qwen.env measure-val stock
```

**t21 results (validation, pinned vLLM helper):**

| Ctx | Right proposals | Escalations | Explanations | Wrong mutating | Warm median | Warm p95 |
|---|---|---|---|---|---|---|
| 2K | 0 of 32 | 10 of 16 | 7 of 18 | 0 (1 mutating proposal with wrong arguments) | 678 ms | 2,072 ms |
| 4K (served at `--max-model-len 4096`, verified) | 0 of 32 | 10 of 16 | 7 of 18 | 0 | 660 ms | 2,058 ms |

Jetson skills (104 evals), served at `MEASURE_CTX=8192` (d13, since a
skills prompt with all 38 tools is already 3,939 tokens): 42 of 104 (40%)
overall, 14 of 34 skill-named, 28 of 70 not-named; by area, bsp 16 of 48,
device 26 of 56. Outcomes: 42 correct, 27 wrong_skill, 31 no_call, 4
several_calls, 0 call_error, 0 think blocks.

**Stock as scorer, exact (t23), on validation.** The same stock checkpoint,
scored the Track B way instead of generatively (`--scorer in-process`), for
comparison against both tracks:

```bash
$P --env qwen.env measure-val stock --scorer in-process
```

21 of 32 right proposals, abstain recall 1 of 16, precision (strict) 100%,
false-positive tool calls 31 of 34, ECE 0.164, Brier 0.765, warm 81 ms. This
is the "stock (exact scorer)" row in the validation table under [Where the
run stands](#where-the-run-stands-2026-09-24-about-09h45) and repeated in the
[run log](#2026-09-24-0530-0720-t22t23-first-runs-three-pipeline-bugs-lapse-l4).

**A dead server fails the run** (P49). Before the first entry, `measure.py`
checks `GET <base_url>/models` and requires the served model name. Any tier
error, or a scorer call error on Track B's served path, fails the run with
exit 2 and no results page (the predictions and metrics are still written,
for debugging). `--allow-tier-errors N` accepts up to N and prints the count
at the top of the page. A scorer's normal "incomplete" top-k result is not an
error.

**Never measure on a Spark while a training job holds its GPU** (ledger
P64, machine safety). On GB10's unified memory, training sinks *free*
memory to about 2.5 GiB while *available* stays near 40 GiB (page cache
from reading weights and data); a GPU allocation there fails outright
instead of evicting that page cache, rather than slowing down the way a
discrete-VRAM box would. A `serve_for_measure.sh`-started vLLM can die
mid-run this way (`NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ...
_memdescAllocInternal` in the kernel log), and `measure.py`'s tier-error
gate above is what turns that into a clean, refused run instead of a
scored partial one. The training memory watchdog (P34, step 11) does not
catch this: it watches *available* memory, which stays high throughout, not
*free*. Schedule every `measure-val`/`measure-final` call on a machine only
when no `train`/`train-scorer` run is using that machine's GPU — on spark,
that means training one checkpoint at a time when a measurement is also
queued, not overlapping the next recipe's training with the previous one's
measurement.

### 10. Train Track A on spark (a1-a4 done; a3 chosen)

```bash
$P --env qwen.env train a1
$P --env qwen.env measure-val a1
```

`train` runs `train.py` under the memory cap, merges, writes the
`generation_config.json` and stages the result in `HF_CACHE` as `REPO`. The
merge saves a generation config without the greedy temperature, because
transformers refuses to save temperature 0 with sampling off; the
`gen_config.py write` step right after it puts the temperature back (P37).
`TRAIN_ARGS` in the env file still holds issue 39's 350M settings and must be
re-tuned for 0.8B on validation only (c20).

For the LoRA-target comparison (r9, ledger P6), `train.py --targets
attn-mlp|attn-mlp-gdn` picks explicitly between unsloth's default
attention/MLP projections and that same set with Qwen3.5's 18 Gated-DeltaNet
linear-attention projections added (`linear_attn.in_proj_qkv`, `in_proj_z`,
`in_proj_a`, `in_proj_b`, `out_proj`); those names exist only in the
language model (the vision tower uses `qkv`/`proj`/`linear_fc1`/`2`; the MTP
head has only attention/MLP names). The chosen target set is passed
explicitly and printed, so a run's log states which one it used:

```bash
$P --env qwen.env train a1 --targets attn-mlp
$P --env qwen.env train a1-gdn --targets attn-mlp-gdn
$P --env qwen.env measure-val a1
$P --env qwen.env measure-val a1-gdn
```

Compare the two on validation only, per r9. Unsloth accepting the GDN target
names is now verified: a training run with `--targets attn-mlp-gdn` produces
a distinct merged checkpoint from `attn-mlp` (see the merge note below).

**Per-run env files, because the env file overrides an exported
variable.** Each recipe change (epochs, rank/alpha) is its own small env
file that sources the base `qwen.env` and then only overrides `TRAIN_ARGS`
— *not* an exported shell variable, because (as in step 9's `MEASURE_CTX`
lapse, l3) a value set in the sourced env file wins over one exported
before the call:

```bash
# a2.env
source qwen.env
TRAIN_ARGS="--targets attn-mlp-gdn --epochs 3 --lr 2e-4 --lora-r 16 --lora-alpha 32 --seed 46"
```

```bash
$P --env a2.env train a2
```

The real recipes trained this way: `a1` = `--targets attn-mlp`, `a2` =
`--targets attn-mlp-gdn`, both 3 epochs, lr 2e-4, rank 16/alpha 32, batch 8,
seed 46 — 549 steps, about 26-31 minutes each on spark, training loss about
0.001 at the end (`train_loss` 0.091). `a3` = `a2` + 5 epochs (915 steps,
about 46 minutes); `a4` = `a2` + rank 32/alpha 64 (3 epochs, about 31
minutes). Both merged cleanly through the verified merge (372 adapter
tensors each); `a3`'s revision is `d0303706...`.

**Track A's pick (t22 decision): `a3`.** On validation (2K, measured on a
quiet GPU — see the [machine-safety
note](#9-stock-baseline-t21-done-on-validation) above step 10): `a3` is the
only Track A run with **0 wrong mutating proposals**, meets both c33 (right
proposals) and c34 (abstention and safety) on validation, and rank 32 (`a4`)
did not help over rank 16 while more epochs (`a3`) did. `a3`'s bf16 warm
latency (about 400 ms) still misses c36's 250 ms bar, so t25's quantization
is required regardless of which checkpoint wins. `a3`'s recipe is committed
as the default `TRAIN_ARGS` in
[`pipeline-qwen.env.example`](../scripts/lfm-finetune/pipeline-qwen.env.example):
`--epochs 5 --lr 2e-4 --rank 16 --alpha 32 --batch 8 --seed 46 --targets
attn-mlp-gdn`. `a3`'s remaining errors on validation: "docker status" (a
read-only check) proposed `container_list` instead of the expected
`service_status docker.service`; "Can you switch to balanced mode?" and
"nvpmodel low power" were escalated instead of the expected `power_set`;
"Explain why nginx returns 502" (expected escalate) was explained instead;
"Is the service running?" (expected escalate) proposed `service_status
docker.service`, a false-positive tool call.

**The merge must be verified, not assumed (lapse l4).** unsloth trains the
vision-language model class (`Qwen3_5ForConditionalGeneration`); its LoRA
adapter keys live under `model.language_model.*`. The first `a1`/`a2` merges
loaded the **text-only** `Qwen3_5ForCausalLM` class instead, so PEFT matched
no adapter key and silently emitted only a warning — the merged checkpoint
came out bit-identical to the untuned base, and both were measured on
validation (0 of 32, indistinguishable from stock) before anyone opened the
merged weights to check. `train.py`'s merge now (f24): maps the VL adapter
keys onto the text-only model's parameter names, replaces unsloth's
`target_modules` regex (which misses `linear_attn` there) with the adapted
module names, and refuses to finish unless every adapter tensor loaded and
at least one merged weight actually changed from the base. `a1` merges 192
tensors this way, `a2` 372. An earlier attempt (f23) merged into the VL
class correctly but wrote doubled key prefixes
(`model.language_model.language_model.*`,
`model.language_model.visual.*`) that vLLM cannot load; f24 replaced it.
**Because of this, the tuned Track A checkpoints are text-only
(`Qwen3_5ForCausalLM`) while stock is the full vision-language
`Qwen3_5ForConditionalGeneration`** — a fact to carry into any memory or
latency comparison between them (see [Not verified
yet](#not-verified-yet)).

### 11. Train Track B on spark2 (b1-b4 done; b1 chosen)

spark2 needs its own env file: its own work-directory paths, its own private
`HF_HOME` (step 1: the shared cache is root-owned), `HF_HUB_OFFLINE=1`, and
`uv` called by its full path since it is not on a non-interactive `ssh`'s
`PATH` (ledger P33). Copy `qwen.env` to `spark2.env` and edit those values
before running anything on spark2.

spark2 serves other models next to the trainer; their containers must not
restart. Three limits apply, because on GB10 the systemd cap does not cover
GPU allocations (P34). Set them in `spark2.env`:

```bash
TRAIN_MEMORY_MAX=24G            # systemd RAM + swap cap
TRAIN_MEMORY_FLOOR=<floor>      # stop the run below this much available memory (default 8G)
TRAIN_WATCHDOG_SECONDS=5        # how often the floor is checked
NVSH_TRAIN_GPU_MEMORY_GB=<gb>   # per-process GPU budget; empty = no per-process cap
```

- `TRAIN_MEMORY_FLOOR`: the watchdog stops the run once the machine's
  available memory falls below it. The lead's probe on spark2 used a 26 GB
  floor with about 30 GB available.
- `NVSH_TRAIN_GPU_MEMORY_GB`: `train.py` and `train_scorer.py` cap their own
  GPU allocations with `torch.cuda.set_per_process_memory_fraction`. Empty
  means no per-process cap (P43).

`pipeline.sh` exports all of these to its child processes (P36). Confirm
what a child process will see before training, then train:

```bash
$P --env spark2.env status     # prints: caps (as a child sees them): max=... floor=... watchdog=...s gpu_gb=...
$P --env spark2.env train-scorer b1
```

`train-scorer <name>` now trains on `data/train-augmented.json` (the frozen
1,463-example file from step 7), not the raw split — f19 fixed a bug where it
trained on `splits/train.json`, which still held the 52 issue-39 test entries
and a duplicate of a test entry, and none of the variations (ledger P58).
Like `train`, it then merges (`train.py --merge-only`), writes the greedy
`generation_config.json` and stages the result as `$REPO-scorer` with a
revision (f20).

The real run, `b1` (all-linear LoRA, same recipe as `a1`/`a2`: 3 epochs, lr
2e-4, rank 16/alpha 32, batch 8, seed 46): 26 minutes on spark2, trainer's own
validation 60 of 66 (90.9%), mean confidence 0.957; spark2 kept about 25 GB
available throughout and the serving containers (model-gear) were untouched.

**`b2` = `b1` + 5 epochs: worse on every axis (more epochs overfit Track
B).** 43 minutes; trainer's own validation 56 of 66 (84.8%), loss 0.67, mean
confidence 0.947 (against `b1`'s 90.9% and 0.29 loss). spark2's checkout was
still on the pre-f23 merge when `b2` trained; its merge is text-only and
every adapter key matched (0 missing-key warnings, the lapse-l4 fix already
covered Track B correctly), and the lead verified the merged weights
actually differ from the base (attention, GDN and MLP weights all changed)
before measuring — spark2 was re-synced to the guide's current commit
afterward. `b2`'s exact in-process scorer on validation: 30 of 32 right
proposals, abstain recall 7 of 16 (43.8%), precision 100%, false-positive
tool calls 2 of 34, wrong mutating 1, invalid 7 (not grounded), ECE 0.106,
Brier 0.217, warm 164 ms in-process. Its served run hit 1 `tier_error` (the
server became unreachable mid-run while Track A's `a4` trained on the same
GPU) and the tier-error gate (P49) refused to write a results page at all,
as designed — a lesson for the tutorial: **measure while nothing else trains
on the same GPU when possible**; a served run under load can lose the
server mid-measurement, and the gate is what catches that rather than
silently scoring a partial run.

**`b3` = `b1` with 2 epochs: undertrained.** 17 minutes; trainer's own
validation 52 of 66 (78.8%), mean confidence 0.83 — both lower than `b1`'s
90.9%/0.957, the opposite failure from `b2`'s overfit. With `b2` (5 epochs)
overfitting and `b3` (2 epochs) undertraining, 3 epochs (`b1`) is Track B's
best epoch count so far.

**`b4` = `b1` with lr 1e-4: same decisions as `b1`, better calibration, but
loses on the pre-registered rule.** 1,522 s to train, 2.83 GB peak memory,
revision `2f1ed0f6...`. Served: 28 of 32 right proposals, abstain recall 12
of 16, precision 85.7%, false-positive tool calls 0 of 34, 0 wrong
mutating, 4 not grounded, warm 24 ms. Exact in-process scoring gives the
*same decisions* as the served run, with ECE 0.072 and Brier 0.132 (against
`b1`'s exact ECE 0.097, Brier 0.162, precision 92.3% — `b4` calibrates
better). Against the selection rule fixed before any of these runs (0 wrong
mutating, then abstention, then right proposals): the two tie on wrong
mutating (0 each) and are close on right proposals, but `b1` leads on
abstention precision by one entry (92.3% vs 85.7%) — **`b1` wins on the
rule as written.** To be plain about it: the rule was not moved after
seeing `b4`'s better calibration; that is reported here as a separate
finding, not used to override the pre-registered order — **a lower
learning rate gave better calibration at essentially the same decisions**,
which is worth knowing even though it didn't change which checkpoint
Track B ships.

**t23 decision: Track B = `b1`.** Chosen for the pre-registered reasons
above; `b4`'s calibration finding is recorded for later but is not itself a
reason to switch. `b1`'s recipe is committed as the default
`TRAIN_SCORER_ARGS` in
[`pipeline-qwen.env.example`](../scripts/lfm-finetune/pipeline-qwen.env.example):
`--epochs 3 --lr 2e-4 --rank 16 --alpha 32 --batch 8 --seed 46`.

**Freeing memory for training and measurement.** Once the training data was
frozen (t19, step 7), no serving model already running on the training or
measurement machines was actually needed by this run any more. With the
operator's OK, the unused serving models were stopped to free GPU memory —
about 37 GB freed on the Track A machine, about 56 GB on the Track B
machine — and are restored once the run finishes. Which serving model runs
where is not relevant to reproducing this run; what matters is checking
`nvidia-smi --query-compute-apps` and stopping anything unused before
training and measuring on a shared box (ledger P64).

**Measuring a served Track B run needs the training stack, not the repo's
own venv.** The first served-scorer measurement runs exited 2 with every
metric "not measured" and no reason: the repo's `uv` environment has no
`transformers`, the tokenizer was being loaded from the served model name
instead of the model directory, and a scorer report had no start-up row to
explain the failure. f22 fixed all three: `--tokenizer <model dir>`, the
training venv's site-packages on `PYTHONPATH` for any `--scorer` run, and
every run failure printed to stderr.

```bash
PYTHONPATH=<training site-packages> \
$P --env spark2.env measure-val b1 --scorer served --tokenizer "$WORK/runs/scorer-b1/merged"
$P --env spark2.env measure-val b1 --scorer in-process
```

Measure both `--scorer served` (real decisions and latency) and `--scorer
in-process` (exact calibration, since the served scorer cannot return a
complete label distribution — d15) under separate labels; both are needed
for the full picture on one checkpoint. If the watchdog trips, `run_capped`
returns 3 and `mem.log` records why.

**Moving a checkpoint between the two training machines.** t24's single
final run needs both tracks' chosen checkpoints reachable from wherever it
runs, which can mean copying a merged model from one training machine to
the other. A direct point-to-point link between the two machines' own
network interfaces (no switch, no VPN hop) copied a 1.5 GB merged model in
about 2 seconds; the same copy over the machines' usual Wi-Fi/VPN path took
about 88 seconds — roughly 44x slower. If both machines have a spare
high-speed network interface, a direct cable between them on its own
static, point-to-point subnet, added as a route with a low priority and
never set as the default route, moves large checkpoints fast without
disturbing either machine's existing internet or mesh routing. No
hostnames or addresses are recorded here since they are specific to this
pair of machines, not to reproducing the run.

### Choosing a configuration on validation (Track A and Track B)

Both tracks were tuned the same way: change one variable at a time, decide
on validation only, and never open the test side until t24. This section
walks the actual search that produced `a3` and `b1`, so a reader running
their own recipe search can follow the same method rather than copy these
exact numbers.

**The selection rule, fixed before looking at any result:** validation
only; among the candidates, prefer first **0 wrong mutating proposals**,
then **abstention** (recall, then precision), then **right proposals**
(r9). The validation set is only 66 entries (16 escalate, 18 explain, 32
operation), so a one-entry difference in any count is noise, not a
meaningful gap — read a "13/16 vs 14/16" as "about the same" unless a
pattern repeats across several runs.

**Track A, one change at a time:**

| Run | Change from previous | Right proposals | Abstain recall | Wrong mutating | Reading |
|---|---|---|---|---|---|
| `a1` | baseline: `attn-mlp` targets, 3 epochs | 32/32 | 12/16 | 2 | baseline |
| `a2` | add the Gated-DeltaNet targets (`attn-mlp-gdn`) | 32/32 | 13/16 | 1 | fewer false positives and one fewer wrong mutating proposal — GDN targets stay for every run after this |
| `a3` | `a2` + 5 epochs (3 → 5) | 31/32 | 14/16 | **0** | wrong mutating drops to 0, abstain recall rises 13 → 14/16 — **chosen** |
| `a4` | `a2` + rank 32/alpha 64 (16/32 → 32/64) | 32/32 | 12/16 | 1 | worse than `a3` on every axis that matters to the selection rule: capacity was not the limit here |

Reasoning between runs: `a2` isolated whether the LoRA should touch the
linear-attention projections at all (it should, so every later run keeps
`attn-mlp-gdn`); `a3` and `a4` then isolated the two obvious next levers
—training length and adapter rank/alpha — against `a2`, one at a time.
More epochs helped (`a3`); more rank did not (`a4`). That is itself a
finding, not a null result: it points at epoch count, not adapter capacity,
as what was limiting `a2`.

**Track B, one change at a time:**

| Run | Change from previous | Trainer val | Mean confidence | Reading |
|---|---|---|---|---|
| `b1` | baseline: all-linear LoRA, 3 epochs | 90.9% | 0.957 | baseline — **best so far** |
| `b2` | `b1` + 5 epochs (3 → 5) | 84.8% | 0.947 | worse: **overfits** — accuracy and abstain recall (7/16 on the harness) drop while confidence barely moves |
| `b3` | `b1` with 2 epochs (3 → 2) | 78.8% | 0.83 | worse the other way: **underfits** — both accuracy and confidence drop together |
| `b4` | `b1` with lr 1e-4 (2e-4 → 1e-4), 3 epochs | — (measured only on the harness) | — | same decisions as `b1` on validation, better calibration (exact ECE 0.072 vs 0.097, Brier 0.132 vs 0.162), but abstention precision one entry lower (85.7% vs 92.3%) — loses on the pre-registered rule; **not chosen**, but recorded as a finding: lower lr helped calibration without changing the decisions |

**Reading over-fit versus under-fit from these numbers:** Track B's
scorer reports its own mean confidence alongside its trainer-side
validation accuracy, and the two diverge in opposite ways depending on
which side of the right epoch count a run lands on. **Overfit** (`b2`)
looks like confidence staying high (0.947, barely below `b1`'s 0.957) while
accuracy and — more sharply — the harness's abstain recall both drop (7 of
16, against `b1`'s 12 of 16): the model is still certain, just increasingly
certain about memorized training patterns rather than the validation
distribution. **Underfit** (`b3`) looks like both numbers dropping
together (accuracy 78.8%, confidence 0.83): the model has not yet
separated the classes confidently either way. A well-fit run in between
should show the accuracy peak roughly matching where confidence still
looks reasonable rather than inflated — which is why `b1`, the middle
epoch count of the three tried, is still the pick.

**Finding the right epoch count in general:** with 1,463 training examples
and batch size 8, one epoch is about 183 steps (1,463 / 8, rounded up); a
run's total step count divided by that gives its epoch count, which is a
useful sanity check on a run record before trusting its numbers. The
training-loss curve reaching a very low value (about 0.001 by the end of
every Track A run here) is **not** a stopping signal by itself — every
recipe tried, including the ones that later turned out to overfit or
underfit on validation, reached a similarly small training loss. The
signal that actually matters is the *validation*-side numbers: the
trainer's own validation accuracy and (for Track B) mean confidence during
training, and then the full harness metrics (abstention, wrong mutating,
ECE, Brier) after merging — never the training loss alone.

**A rule fixed before looking still decides ties, even against a result you
would have preferred (`b4`).** `b4` scored the *same decisions* as `b1` but
with visibly better calibration; the honest way to report that is as a
separate finding about learning rate, not as grounds to move the
already-fixed selection rule after seeing the result. `b1` still wins
Track B on the rule as written (abstention precision, by one entry). Had
the rule not been fixed beforehand, this is exactly the kind of comparison
where it would be tempting to rationalize a switch after the fact — fixing
the rule first is what prevents that.

**t22's optional check: does the chosen Track A recipe hold at a longer
context?** `a3` was also measured at 4K (`MEASURE_GPU_FRACTION=0.12`, to
avoid P65's Mamba-cache shortfall): 29 of 32 right proposals, abstain
recall 14 of 16, 0 wrong mutating, explain 18 of 18, warm 440-653 ms —
close to `a3`'s own 2K numbers (31/32, 14/16, 0 wrong mutating) and still 0
wrong mutating at both contexts. The final run stays at 2K per the original
plan; the 4K numbers are informational, confirming `a3`'s behaviour does
not fall apart at a longer context rather than changing which checkpoint is
used.

### 12. Final measurement *(not yet run)*

```bash
$P --env qwen.env measure-final stock
$P --env qwen.env measure-final a1
```

One call per model (deviation d7), each against its own helper-served
model, on the test side, the held-out set and the missing-candidate slice.
Each is measured once; any retry is a deviation. `measure-final` always
passes `--predictions "$WORK/final/<name>"`, so each final run keeps its
predictions file (`final-<name>-1-<model>.predictions.jsonl`) and metrics
there. The file holds ids, expected blocks and outcomes, never request text;
`--details` stays refused on the final side. By default
`measure-final`'s results page still goes to `docs/benchmarks/` (a known gap
from h2).

Track A's calibration is scored separately, in process, by the d6 tool, once
per checkpoint on the final side:

```bash
PYTHONPATH=<training site-packages> uv run --frozen python \
  scripts/lfm-finetune/track_a_calibration.py --model "$WORK/runs/a1/merged" \
  --split "$WORK/splits/test.json" \
  --predictions "$WORK/final/a1/<final predictions>.jsonl" --out <out.jsonl> --final
```

Run it from the repository root, like `assemble`'s render check: the
training venv's torch and transformers come from `PYTHONPATH`, and nvsh
from the repository environment. The lead's live check ran it this way.

It fills each prediction line's `candidates` with the exact teacher-forced
distribution and writes a sidecar `<out>.provenance.json`. `metrics.py` then
scores the filled file. Its input is the predictions file `measure-final`
kept in `$WORK/final/<name>` (P50).

### 13. Quantize and heal *(not yet run)*

Build llama.cpp (the spike used master
`633733d0aeedd721868bf5f1b935fa3f39f9164e`, configured with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`) and a separate venv for
llm-compressor (`uv pip install llmcompressor --index-strategy
unsafe-best-match`; see ledger P25). Then:

```bash
LLAMA_CPP_DIR=<llama.cpp checkout> \
LLAMA_CPP_CONVERT=<llama.cpp checkout>/convert_hf_to_gguf.py \
LLAMA_CPP_QUANTIZE=<llama.cpp build>/bin/llama-quantize \
LLAMA_CPP_IMATRIX=<llama.cpp build>/bin/llama-imatrix \
AWQ_PY=<awq venv>/bin/python \
  $P --env qwen.env quantize a1
```

`AWQ_PY` is the separate AWQ venv's Python (llm-compressor 0.14.0,
transformers 5.17.0, compressed-tensors 0.19.0); `quantize.py` refuses to
fall back to the training venv, whose transformers is too old for
llm-compressor on this architecture. `LLAMA_CPP_DIR` is optional and only
records the llama.cpp commit; without it the version reads "unknown".

What the stage does:

- **AWQ.** `quantize.py` runs `$AWQ_PY scripts/lfm-finetune/awq_oneshot.py
  --model-dir ... --calibration-file ... --out-dir ...
  --num-calibration-samples N` with train-side calibration text. The recipe
  is `AWQModifier` on `Linear` targets, scheme W4A16, ignoring `lm_head`,
  the vision tower, the linear-attention layers and the MTP head; `oneshot`
  gets `processor=` the tokenizer. The generation config is sanitized before
  the save (P37). `finish_awq_export` then copies `tokenizer.json`,
  `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `preprocessor_config.json` and `video_preprocessor_config.json` into the
  output and runs `gen_config.py write`.
- **GGUF.** `gen_config.py write` on the source directory first, then a bf16
  text-only conversion, an imatrix from the train-side calibration text, and
  `Q4_K_M`.
- **Versions.** The run record names the llama.cpp commit and the AWQ venv's
  package versions.

**Serving the AWQ build needs one extra vLLM argument.** The run record
carries `serve_args`: `--limit-mm-per-prompt '{"image": 0, "video": 0}'`.
nvsh's launcher cannot pass extra vLLM arguments. This is a known
limitation. `serve_for_measure.sh` (step 9) always passes this flag, but the
measure stages cannot name an AWQ build yet (a known gap from h2). Until
they can, the route is a vLLM started by hand from the pinned image with the
recorded arguments:

```bash
--limit-mm-per-prompt '{"image": 0, "video": 0}' \
--enable-auto-tool-choice --tool-call-parser qwen3_coder --max-model-len 2048
```

`measure.py` then uses an nvsh config with `[tiers.lfm] mode = "attach"`
and `base_url = "http://127.0.0.1:<port>/v1"`. This route is verified for
stock (t13) and not yet for the AWQ build *(unverified until t25)*. No
sampling override flag is needed: the `generation_config.json` pins
temperature 0.

Measure both builds with the same harness. If `heal_needed()` is true, log
the trigger, then `$P --env qwen.env heal a1-heal a1`.

### 14. Scan and upload, privately *(not yet run)*

```bash
$P --env qwen.env scan a1
FINAL=1 grant run --inject HF_TOKEN=<secret> -- $P --env qwen.env upload a1
```

Ask the operator before any upload. Build the model card with
`release_bundle.py --licence-kind apache --tool-call-parser qwen3_coder` and
the data set with `dataset_bundle.py --teacher-models <file> --apache-only`.

## Pitfalls hit, and the fix for each

Each entry: what went wrong, who or what found it, the evidence, the fix,
and the commit on `spec/qwen-tool-jev-issue-46`.

### Template and training

- **P1. No generation markers in Qwen's chat template.**
  `return_assistant_tokens_mask` needs Jinja generation-block markers, and
  Qwen3.5's template has none. *Found:* reading the template (t2).
  *Fix:* `train.py` renders the prompt with the generation prompt and the
  full conversation; the answer is what the full rendering adds, and the
  labels are the answer plus the end-of-turn token. The lead decoded real
  labels for propose, explain and escalate. *Commit:* `c1d6645` (merge
  `1117083`).
- **P2. An empty think block is always emitted.** The template writes an
  empty think block before the answer whether or not `enable_thinking=False`
  is passed. *Found:* t15 spike. *Fix:* `train.py` renders with thinking off,
  so training matches inference; `measure.py --enable-thinking false` sends
  the switch and counts any non-empty think block (must be 0).
  *Commits:* `c1d6645`, `9fe3226`.
- **P3. String tool arguments crash the template.** A JSON-encoded string
  argument makes the template raise `TypeError`; arguments must be objects,
  as with LFM2.5. Tool calls render as XML (function and parameter tags).
  *Found:* t15 spike. *Fix:* object arguments, verified by the round-trip
  check (P4). *Commit:* `656aef7`.
- **P4. The round-trip check was never called by the build.** `build_dataset.py`
  had a render-and-parse-back check, but the real build did not run it.
  *Found:* Codex between-wave review (f4). *Fix:* on by default;
  `--no-verify-render` only where the tokenizer is not cached.
  *Commit:* `009af2c` (merge `aa830e0`).
- **P5. Could unsloth change the end-of-turn token?** The answer span ends at
  `eos_token_id` (plan risk r6). *Found:* planning. *Evidence:* t15 spike:
  unsloth's `FastLanguageModel` loads the model (1.79 GB GPU) and keeps eos
  `<|im_end|>`, id 248046. *Fix:* none needed.
- **P6. Default LoRA targets skip the linear-attention layers.** The model
  has 24 layers, 3 Gated-DeltaNet (linear-attention) to 1 full-attention,
  plus a vision encoder and an MTP head. unsloth's default targets are q/k/v/o
  of the 6 full-attention layers plus all MLPs: 96 modules, 12.8M
  parameters, none on vision. The 18 Gated-DeltaNet layers' own projections
  get no adapter. *Found:* t15 spike. *Status:* open (plan risk r9); compare
  default against adding those projections on validation.

### Serving

- **P7. Tool-call parsers.** *Found:* t15 spike: gold text rendered by the
  pipeline was fed to each parser inside the pinned image. `qwen3_coder` and
  `qwen3_xml` round-trip propose, explain and escalate exactly; `hermes`
  fails. *Fix:* `qwen3_coder` (plan risk r3, resolved).
- **P8. Stock is slow against the latency bar.** Stock bf16 answered warm in
  about 450-490 ms for about 50 generated tokens (about 10 ms per token),
  against the 250 ms bar. *Found:* t15 spike. *Status:* open (plan risk
  r10). Tuned answers may be shorter; the quantized builds may be needed to
  meet c36.
- **P9. vLLM sampled at temperature 1.0.** Qwen ships no
  `generation_config.json`, and Tier 2's request sets no temperature.
  *Found:* the lead, during t13's live check. *Fix (d3):* `gen_config.py`
  writes `temperature` 0 and `do_sample` false into every served model
  directory. *Evidence:* vLLM logs "Default vLLM sampling parameters have
  been overridden by the model's generation_config.json: {'temperature':
  0.0}", and three runs gave byte-identical output. *Commit:* `2096c30`.
- **P10. The generation config's eos was incomplete.** The first version
  copied only eos 248044 (`<|endoftext|>`, from `config.json`'s
  `text_config`). *Found:* the lead, reviewing f9. *Fix:* the tokenizer's
  end-of-turn is added, so `eos_token_id` is `[248046, 248044]`, also when
  the file already exists. *Commit:* `35dcf29` (merge `0ad327a`).
- **P11. Track A has no candidate distribution.** In the t13 live check
  (stock on validation, attach mode), 0 of 66 predictions carried a Track A
  candidate distribution, so Track A's ECE and Brier would be empty.
  *Found:* the lead, t13 live check. *Status:* open (plan risk r12). Either
  this vLLM does not return tool-call tokens in `logprobs.content`, or the
  token trace never resolves. Track B is unaffected.

### Data pipeline

- **P12. `--rereview` failed on real accepted records.** Real accepted
  records carry no `verdicts` (only rejected ones do), so the first
  `--rereview` would have errored on all 1,195 accepted records. It also
  ignored the deterministic identifier and template guards. *Found:* the lead,
  probing real data (t11). *Evidence after the fix:* on the real 1,720
  records with a fake always-yes reviewer, 0 errors, 1,195 kept, and no
  reviewer-A rejection flipped. *Commit:* `eda6b0c` (merge `f8026ed`).
- **P13. `--rereview --dry-run` did real work.** *Found:* Codex between-wave
  review (f5). *Fix:* dry run reports the count, limit and reviewer B model
  and exits without calling anything or writing output. *Commit:*
  `9e16fc8` (merge `db9fa4e`).
- **P14. Non-thinking reviewer B read requests over-literally.** In the
  pilot, non-thinking Qwen 3.8 agreed with Nemotron on 56 of 150 (37%) and
  accepted only 20 of 150, rejecting sound variations (for example "show
  processes" judged to imply a multi-step diagnosis). *Found:* the pilot,
  and reading 9 rejection reasons. *Fix:* thinking on, by the rule fixed
  before the run (c41).
- **P15. The thinking re-review is slow.** A 20-candidate thinking probe took
  878 s (about 44 s each). *Fix (d4):* `--rereview` runs `--workers`
  concurrently (4 in this run), and only candidates whose source stays on
  the train side are re-reviewed. *Commit:* `bee3ee4` (merge `f07aaae`).
- **P16. The re-split breaks train-only variations.** Every stored variation
  was train-side under seed 39, and `merge_variations.py` raised when a
  source was no longer in the train split. *Found:* spec review (c45).
  *Fix:* `--filter-to-split` drops and counts them. *Commit:* `5c372fe`
  (merge `5af8145`).
- **P17. Duplicate variation ids were accepted.** *Found:* worker
  between-wave review (f7), which also found a test that overwrote its own
  input. *Fix:* duplicates refused; test fixed. *Commit:* `e28f63a` (merge
  `5d8ca75`).
- **P18. A non-Apache reviewer.** Nemotron 3.5 Lightning (OpenMDW-1.1) was
  reviewer B in issue 39. *Found:* spec review (c38). *Fix:* re-review with
  Qwen 3.8 27B (above).

### Metrics and scorer

- **P19. Scorer labels did not match metrics' labels.** The scorer keyed the
  controls `escalate` and `explain`; metrics expects `(escalate)` and
  `(explain)`. That gave ECE = 1 on correct escalations. Separately, served
  top-k log-probabilities with labels missing were renormalised into false
  certainty. *Found:* Codex between-wave review (f1). The lead's merge gate
  had never fed scorer output into metrics (lapse l1). *Fix:* the scorer
  uses metrics' control labels; an incomplete result names the missing
  candidates and carries no distribution, so metrics leaves it out of
  calibration and counts it. *Commit:* `608334f` (merge `8fe87b4`).
- **P20. Served top-k is 20, Track B needs 22.** vLLM's default
  `--max-logprobs` is 20; 18 labels plus 4 need 22. *Found:* planning (plan
  risk r8). *Fix:* `measure.py --scorer served` requires `--max-logprobs`,
  recorded in the run; in-process scoring reads every label. *Status:* open
  until t23 records which was used.
- **P21. Grounding depended on the live machine.** *Found:* planning (plan
  risk r7). *Fix (d1):* fixed snapshot, `measure.py snapshot` and
  `--ground-snapshot`. *Commit:* `9fe3226` (merge `2f2d4af`).
- **P22. Abstention precision was ambiguous.** t8 followed bench's
  definition, which leaves explain entries out of the denominator.
  *Found:* operator decision on the spec's wording. *Fix (d2):* strict
  precision, bench's figure alongside. *Commit:* `f3131aa`.

### Quantization

- **P23. The calibration guard accepted swapped files.** `quantize.py` only
  checked names, so swapped `--train`/`--val` passed. *Found:* Codex review
  (f2). *Fix:* split headers checked. *Commit:* `2b4e6b3` (merge `c6fd441`).
- **P24. `heal_needed` compared counts, not ids.** Quantization fixing one
  entry and breaking another left the count unchanged. *Found:* Codex review
  (f2). *Fix:* compares wrong-mutating id sets. *Commit:* `2b4e6b3`.
- **P25. uv installed an llm-compressor too old for Qwen3.5.** `uv pip
  install llmcompressor` resolved 0.6.0.1 with transformers 4.52.4, which
  does not know `qwen3_5`, because uv took versions only from the first index
  that had a package. *Found:* t16 spike. *Fix:* `--index-strategy
  unsafe-best-match` gave llmcompressor 0.14.0, transformers 5.17.0 and
  compressed-tensors 0.19.0, in a venv separate from training.
- **P26. AWQ on a multimodal hybrid.** `oneshot` needed `processor=` the
  tokenizer. Linear-attention, vision, MTP and `lm_head` were left
  unquantized: the result is 1.1 GB, dominated by bf16 embeddings and
  `lm_head` (39.5 s to build). vLLM then needed the tokenizer and
  preprocessor files copied into the AWQ directory and
  `--limit-mm-per-prompt` image and video 0. *Found:* t16 spike. *Result:*
  Marlin INT4 kernel, Gated-DeltaNet decode on CUDA, tool calls parse, warm
  about 200-330 ms.

### Bundles and publishing

- **P27. Private IPs in prose were not flagged.** `scan-secrets.py` checks
  only JSON endpoint keys. *Found:* the lead, reviewing t7. *Fix:*
  `private_hosts()` in `scan_bundle.py`. *Commit:* `e8d0312`.
- **P28. JSON-escaped secrets slipped through.** An escaped `hf_...` token
  hid from the raw byte scan. *Found:* Codex review. *Fix (f3):* decoded
  JSON and JSONL string values are scanned recursively; a non-UTF-8 file is a
  finding; weight files are listed under `binaries`. *Commit:* `208be85`
  (merge `f24cf8d`).
- **P29. The Qwen model card kept LFM text.** The worker duplicated the whole
  card template and kept LFM tags and `tool_call_parser "lfm2"`. *Found:*
  the lead, reviewing t5. *Fix:* one template with `--tool-call-parser`; LFM
  output verified byte-identical. *Commit:* `3a7650c` (merge `68f73cd`).
- **P30. The data set card kept only the first teacher per role** and
  compared aliases, not resolved names. *Found:* Codex review (f6). *Fix:*
  every teacher per role, names resolved. *Commit:* `d1d7444` (merge
  `a3adc1e`).
- **P31. An unused import.** The t9 worker reported flake8 clean but left an
  unused import. *Found:* the lead. *Fix:* removed (merge `305d568`).

### Machine safety

- **P32. `MemoryMax` alone is not a hard cap.** Under `systemd-run --user
  --scope -p MemoryMax=200M`, a process touching 600 MB finished on both
  Sparks: it spilled into swap (spark has 64 GB of swap). Adding `-p
  MemorySwapMax=0` killed it (exit 137). spark2 runs a model-serving stack
  using about 72 GB, with about 30 GB free, so an uncapped trainer there
  would slow everything. *Found:* the lead's live check on both Sparks.
  *Fix:* `capped.sh` sets both limits, logs free memory every 60 s, and
  refuses to train without `systemd-run` unless `TRAIN_MEMORY_CAP=container`.
  *Commit:* `1116fe0` (merge `1bd891d`). This cap bounds CPU memory only;
  see P34.
- **P33. spark2's environment differs over ssh.** `uv` is not on a
  non-interactive ssh `PATH`, and the shared Hugging Face cache is
  root-owned. *Found:* the lead, setting up spark2 (t20). *Workaround:* call
  `uv` by its full path; use a private `HF_HOME` inside the spark2 work
  directory. The serving stack's directories are left untouched.
- **P34. GPU memory is not charged to the cgroup on GB10.** On unified
  memory, an 8 GB CUDA tensor was allocated and filled inside `run_capped`
  with `TRAIN_MEMORY_MAX=1G` (`MemoryMax` and `MemorySwapMax=0`). The
  systemd cap bounds CPU memory only, so it does not protect spark2's
  serving stack from a trainer's GPU allocations. *Found:* the lead's probe
  on spark2 (t20). The serving containers' restart counts were unchanged
  across the probe. *Status:* was blocking plan risk r13 (task t23), now
  resolved. *Fix (review-fix task f10):* an available-memory floor watchdog
  in `capped.sh`: `TRAIN_MEMORY_FLOOR` (default 8G), checked every
  `TRAIN_WATCHDOG_SECONDS` (default 5). The command runs in its own process
  group (`setsid`), so it can be stopped in both the systemd-scope mode and
  `TRAIN_MEMORY_CAP=container`. Below the floor the watchdog logs a line,
  sends SIGTERM, then SIGKILL after 10 s, and `run_capped` returns 3. On
  top of that, `train.py` and `train_scorer.py` honour
  `NVSH_TRAIN_GPU_MEMORY_GB` through
  `torch.cuda.set_per_process_memory_fraction`. *Evidence:* on spark, a 1 GB
  budget refused a 2 GiB allocation. On spark2 (MemAvailable 30 GB, floor
  26 GB), an 8 GB CUDA hog was stopped after 5 s at 22,882 MB available,
  with return code 3, no Python process left on the GPU, and the serving
  containers' restart counts unchanged. *Commit:* `67bb9e7` (merge
  `686e1c8`).
- **P64. A GPU allocation can fail beside a training job because unified
  memory does not evict the page cache.** `a3` trained fine (549 steps, 372
  adapter tensors merged and verified) but its own validation measurement
  lost its vLLM server mid-run with 1 `tier_error` while `a4` trained on the
  same GPU — the same failure mode as `b2`'s served run and the two earlier
  "Engine core initialization failed" start-ups (P63). *Found:* the lead,
  reading spark's kernel log after the failed measurement: at 08:10:36,
  `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`.
  *Root cause:* during training, spark's *free* memory sinks to about
  2.5 GiB while *available* stays about 40 GiB (page cache built up from
  reading weights and data). On GB10's unified memory, a GPU allocation
  fails outright instead of evicting that page cache, so a vLLM server
  started — or already running — beside a training job can die. The
  training memory watchdog (P34) was unaffected by this: it watches
  *available* memory, which stayed high throughout, not *free*. *Fix
  (rule, not code):* measure on spark only when no training run holds
  spark's GPU (`a3` and `a4` are both re-measured after `a4` finishes). Not
  yet tried: dropping the page cache before a measurement (needs root,
  `sync; echo 3 > /proc/sys/vm/drop_caches`) or a cgroup limit on the
  trainer's page cache, either of which might allow overlap.
  **Why spark was this tight:** spark also runs the operator's own
  model-gear "lobes" deployment (docker compose project `lobes`: a gateway
  and the stt, realtime and bluetts services) alongside a
  `model-gear-vllm-multimodal` container serving Gemma-4-26B-A4B-NVFP4 —
  the "senses" teacher, reviewer A in t19's augmentation pipeline — which by
  itself held about 33.6 GB of GPU memory in its vLLM `EngineCore`. A
  training job plus a measurement server on top of that left free memory at
  about 2.5 GiB, and any further GPU allocation failed with
  `NV_ERR_NO_MEMORY`. *Resolved:* the operator was clear that a 128 GB
  machine should not OOM and that unused models should be taken down rather
  than tolerating this ("we can take down models as needed", "I'd rather be
  cautious if we don't use the models now"). The `lobes` CLI's `stop` and
  `fleet down` take the whole spark deployment down, gateway included, with
  no per-lobe stop; being cautious, the lead instead stopped only the one
  container no longer needed now that t19 is done —
  `docker stop model-gear-vllm-multimodal` (nothing deleted; restore with
  `docker start model-gear-vllm-multimodal` or `lobes serve --apply`) —
  leaving the gateway, stt, realtime and bluetts running. Memory on spark,
  measured immediately after, while `a4` kept training: used 68 → 31 GB,
  free 15 → 51 GB, available 53 → 89 GB. *Tutorial lesson:* on a GB10
  shared with serving lobes, check `nvidia-smi --query-compute-apps` and
  `docker stats` for what already holds GPU memory before training and
  measuring on the same box, and stop an unused lobe (with the operator's
  sign-off) rather than letting jobs overlap into an OOM.
- **P65. A 4K measurement can fail to start for a reason that looks like
  P64 but is not an out-of-memory error at all.** `measure-val` at
  `MEASURE_CTX=4096` failed at server start with "Engine core
  initialization failed"; the full server log
  (`measure/<label>.serve.log`, kept since f25/P63) ended with `ValueError:
  max_num_seqs (256) exceeds available Mamba cache blocks (254)`. *Found:*
  the lead, reading the full serve log's actual root-cause line rather than
  assuming P64's pattern from the start-up failure alone. *Root cause:* at
  4K context with `MEASURE_GPU_FRACTION=0.08`, vLLM's CUDA-graph memory
  estimate (about 4.95 GiB) leaves only about 1.59 GiB for the KV/Mamba
  cache — 254 blocks — which falls below the default `max_num_seqs` of 256;
  each Gated-DeltaNet decode sequence needs one Mamba cache block, so 256
  concurrent sequences need at least 256 blocks. This is a cache-sizing
  shortfall, not memory pressure from another process — **distinguish it
  from P64 by reading the full serve log's own root-cause line** rather
  than the generic "Engine core initialization failed" line, which both
  failures share. *Fix:* raise `MEASURE_GPU_FRACTION` to 0.12 for a 4K run
  (vLLM's own suggested value is 0.1207); the fraction only changes how
  much memory is reserved for the cache, never the served model's outputs.
  Because the env file overrides an exported variable of the same name
  (lapse l3's mechanism; unlike `MEASURE_CTX`, `MEASURE_GPU_FRACTION` is not
  made export-first), set it in a per-run env file rather than exporting it
  before the call — the same per-run env file pattern as step 10's
  `TRAIN_ARGS`.

### Found by reading code against the run log

- **P35. `quantize.py` did not do what the t16 spike did.** It ran AWQ by
  calling `$LLM_COMPRESSOR --model ... --calibration ... --scheme AWQ --bits
  4 --out ...` as a command. The spike used llm-compressor's Python
  `oneshot`, with `processor=` and linear-attention, vision, MTP and
  `lm_head` left out (P26). The stage also did not copy the tokenizer and
  preprocessor files vLLM needed, and it converted the GGUF to f16 where the
  spike used bf16. So the stage's AWQ path had never run against the real
  tool. *Found:* the documentation agent, reading `quantize.py` against the
  spike's run log. *Status:* plan risk r14 (non-blocking, task t25). *Fix
  (pending, review-fix task f11):* AWQ through a new `awq_oneshot.py` run by
  the separate AWQ venv's Python (`AWQ_PY`), with `processor=`, the ignore
  list, the tokenizer and preprocessor files copied and `gen_config.py`
  applied; GGUF as bf16, then imatrix, then `Q4_K_M`. *Commits:* `47d3af3`,
  `99739fb` (merge `07422cd`); plan risk r14 resolved. *Evidence:* the
  lead's live check on the stock copy. The first run failed at save (P37);
  after the fix the output was 1.1 GB with weights. Served by the pinned
  vLLM with `--limit-mm-per-prompt` and no sampling override flag, it used
  `MarlinLinearKernel`, logged "Default vLLM sampling parameters have been
  overridden by the model's generation_config.json: {'temperature': 0.0}",
  gave byte-identical output on three runs, and its tool calls parsed.
- **P36. The GPU budget never reached the trainer.** f10 made `train.py` and
  `train_scorer.py` read `NVSH_TRAIN_GPU_MEMORY_GB`, but `pipeline.sh`
  sources the env file without exporting its variables, and neither env
  example named the new settings. Set only in the env file, the
  per-process GPU cap would silently not apply. *Found:* the documentation
  agent, reading f10's diff against `pipeline.sh`. *Fix:* `pipeline.sh`
  exports `TRAIN_MEMORY_MAX`, `TRAIN_MEMORY_FLOOR`, `TRAIN_WATCHDOG_SECONDS`,
  `TRAIN_MEMORY_CAP` and `NVSH_TRAIN_GPU_MEMORY_GB` right after sourcing the
  env file. Both env examples name `TRAIN_MEMORY_FLOOR=8G`,
  `TRAIN_WATCHDOG_SECONDS=5` and `NVSH_TRAIN_GPU_MEMORY_GB=` (empty means no
  per-process cap). `status` prints the caps as a child process sees them,
  and a test checks that a child sees values set only in the env file.
  *Commit:* `6d805d5`.
- **P39. A stale comment in `pipeline.sh`.** After f11 the quantize
  stage's comment still named `LLM_COMPRESSOR`; it now names `AWQ_PY` and the
  optional `LLAMA_CPP_DIR`. *Found:* the documentation agent. *Commit:*
  `849cad6`.

### Found in live runs after the tooling merged

- **P37. transformers refuses to save the d3 generation config.** The
  served file pins `temperature` 0.0 with `do_sample` false (d3), and
  transformers will not save a model whose generation config says that:
  "`temperature`: `do_sample` is not set to `True`. However, `temperature` is
  set to `0.0` ... Fix these issues to save the configuration." vLLM needs
  the temperature key; transformers' save rejects it. *Found:* the lead's
  live run of f11's `awq_oneshot.py` on the stock copy. *Evidence:* with
  transformers 5.17.0 (AWQ venv), quantization ran, `save_pretrained`
  failed, and the output was 22 MB with no weights. With 5.5.0 (training
  venv), `GenerationConfig.save_pretrained` raises `ValueError`. The t16
  spike only worked because it loaded the raw Hugging Face snapshot, which
  has no `generation_config.json`. *Fix for the training merge:* `train.py`'s
  `save_valid_generation_config()` clears a greedy temperature (temperature
  0 with `do_sample` false becomes temperature `None`) right before
  `merged.save_pretrained`; `pipeline.sh`'s `gen_config.py write` then
  restores the serving file. Verified end to end by the lead with
  transformers 5.5 on the stock copy: loaded temperature 0.0, saved with
  `model.safetensors` written, `gen_config.py write` and `check` passed,
  final file `{do_sample: false, eos_token_id: [248046, 248044],
  temperature: 0.0}`. *Commit:* `51d91b1`. *Fix for AWQ (f11):*
  `awq_oneshot.py`'s `sanitize_generation_config()` replaces the model's
  generation config with a bare one carrying only its token ids right before
  the save, and `finish_awq_export` writes the served file afterwards.
  Verified by the lead's live re-check (P35). *Commit:* `99739fb` (merge
  `07422cd`).
- **P38. Null token ids in the served generation config.** `gen_config.py`
  copied `bos_token_id` and `pad_token_id` from Qwen's `config.json`, where
  they are null, so the served file carried them as null. *Found:* the lead,
  alongside P37. *Fix:* null ids are no longer copied. *Commit:* `51d91b1`.

### Found by the wave-2 review

The wave-2 review covered t13, t14, f8 to f11 and the lead's own fixes. The
qwen worker reviewer approved every file it read, as it did in wave 1.
Codex found seven correctness problems. All seven are fixed in review-fix
tasks g1 to g4 (merges `a57422f`, `9f446d7`, `9f8464a`, `dfd7b48`), and the
full test suite is green.

- **P40 (high). Stopping the pipeline left training running.**
  `run_capped` starts the command in its own process group (`setsid`, from
  f10), so a SIGTERM to `pipeline.sh` did not reach it, and there was no
  cleanup trap. *Found:* Codex wave-2 review. *Fix (g1):* `run_capped`
  traps TERM, INT, HUP and EXIT, and stops and reaps the command's process
  group, the watchdog and the `tee` pipeline. The pipeline runs in the
  background under `wait` so the trap can fire, and the caller's traps are
  restored afterwards. *Evidence:* a test sends SIGTERM to `run_capped` and
  the running command is gone within 15 s. *Commit:* `85af128` (merge
  `a57422f`).
- **P41 (high). The pipeline's measure stages skipped the approved
  measurement setup.** They served stock from `$BASE` (no temperature-0
  generation config) instead of `$WORK/stock`, passed no
  `--ground-snapshot` (d1), did not switch thinking off, and dropped extra
  arguments. *Found:* Codex wave-2 review. *Fix (g1):* `measure-val`,
  `measure-final` and `measure-skills` measure stock from `$WORK/stock` and
  refuse if it is missing or fails `gen_config.py check`; `measure-val`
  accepts `stock` as a run name. `measure-val` and `measure-final` always
  pass `--ground-snapshot "$GROUND_SNAPSHOT"` and `--enable-thinking` from
  `ENABLE_THINKING` (default false); `measure-skills` does not, because
  `measure_skills.py` has no snapshot option. Extra arguments are
  forwarded; for `measure-skills` only to the tuned run, so `--margin` never
  lands on stock. Both env examples name `GROUND_SNAPSHOT` and
  `ENABLE_THINKING`. *Commit:* `85af128` (merge `a57422f`).
- **P42 (high). Track A's candidate tracing invented probabilities.** It
  credited a first-token alternative's probability to a whole label whose
  continuation was never observed (for example `gpu` counted as
  `gpu_stats`), fabricating ECE and Brier inputs. *Found:* Codex wave-2
  review. *Fix (g2):* an alternative is credited only when it spells the
  whole label plus its ending character; any other alternative that is a
  label prefix makes the line refuse a distribution ("an alternative
  token's continuation was not observed"). *Commit:* `e4cb18e` (merge
  `9f446d7`). *Consequence, found while fixing it on the real Qwen
  tokenizer:* names split into several tokens (`propose` is `prop` +
  `ose`, `escalate` is `escal` + `ate`, `gpu_stats` is `=g` + `pu` +
  `_stats`, and `>` is its own token). So most real Qwen lines will get no
  Track A distribution, and Track A's ECE and Brier are probably not
  measurable from generation log-probabilities. Calibration figures will
  likely come from the scorer track. *Status:* decided by the operator as
  deviation d6: Track A calibration is scored exactly, in process, by
  teacher-forcing each candidate (h17 kept). *Fix (h1):*
  `track_a_calibration.py`. *Evidence:* the lead's live check on real
  weights (the stock copy, issue 39's old validation split and t13's stock
  predictions): 66 of 66 lines got a distribution summing to 1 (largest
  error 2.2e-16), the top label was the expected one on 30 of 66, stock ECE
  0.230 and Brier 0.754, about 4 s per entry (257 s for 66). *Commit:*
  `6806662`. P42 is fixed.
- **P43 (medium). An empty GPU budget aborted training.** The env examples
  ship `NVSH_TRAIN_GPU_MEMORY_GB=` empty (meant as "no per-process cap"),
  and the trainers parsed it with `float('')` and stopped. *Found:* Codex
  wave-2 review. *Fix (g3):* an empty or whitespace value means no cap, in
  both trainers. *Commit:* `ec41cbf` (merge `9f8464a`).
- **P44 (medium). `assemble` needed transformers where it was not
  installed.** The render check that f4 turned on by default imports
  transformers, which the repository environment used by `pipeline.sh`'s
  `py()` does not have. *Found:* Codex wave-2 review. *Fix (g1):*
  `assemble` runs `build_dataset.py` with the training venv's site-packages
  on `PYTHONPATH`, on top of `uv run`, so nvsh itself still comes from the
  repository environment. *Commit:* `85af128` (merge `a57422f`).
- **P45 (medium). Unparsed tool calls scored as explanations.** Qwen XML
  that the vLLM parser failed on reached `measure.py` as plain text, an
  explanation, and would score as explain instead of invalid. *Found:*
  Codex wave-2 review. *Fix (g2):* an explanation that contains tool-call
  markup, or a last reply with markup and no parsed calls, is outcome
  `invalid` with `invalid_reason` `unparsed_tool_call`. *Commit:*
  `e4cb18e` (merge `9f446d7`). nvsh's own runtime (`LfmTier`) still treats
  such output as an explanation; this work does not change runtime code
  (c9). Filed, with the operator's approval, as
  [nvsh issue #50](https://github.com/agentculture/nvsh/issues/50).
- **P46 (medium). AWQ calibration split records on newlines.** The
  calibration file held one record per line, so a record containing a
  newline became several samples and later records were dropped. *Found:*
  Codex wave-2 review. *Fix (g4):* AWQ calibration is JSONL, one JSON
  string per line, so a record with newlines stays one sample; the imatrix
  gets a separate plain-text file with newlines replaced by spaces.
  *Commit:* `e458ae9` (merge `dfd7b48`).

The committed plan record lacked deviation d5 when Codex looked; it is
committed now (`3df700c`).

- **P48. The managed launcher cannot serve the stock copy.** nvsh's managed
  launcher refuses an absolute model path for vLLM, so stock served from
  `$WORK/stock` is measured in attach mode. `measure-skills` always uses the
  managed launcher and cannot measure the stock copy until
  `measure_skills.py` gains an attach option. *Found:* the lead, while
  merging g1. *Status:* plan risk r15, resolved by deviation d7: every model
  is served by one helper and measured in attach mode, one model per call.
  *Fix (h2):* `serve_for_measure.sh` and the measure stages (step 9);
  `measure-skills` points `measure_skills.py` at the served URL with
  `--url`. *Evidence:* the lead's live check. The helper served the stock
  copy and vLLM logged temperature 0 from its `generation_config.json`.
  `measure.py` through the attach config reproduced t13's outcome counts
  exactly (45 no decision, 12 explain, 6 escalate, 3 propose). t13 had used
  `--override-generation-config` and this run used the model's own file,
  so the measurement is deterministic. *Commit:* `1062123`.
- **P49. A dead server gave a plausible baseline.** `measure.py` against a
  stopped attach server exited 0 and wrote a results page ("Right proposals
  0 of 32") in which all 66 lines were invalid with a tier error. A results
  page alone cannot tell a dead server from a bad model. *Found:* the lead's
  live check of h2. *Fix (h3):* `measure.py` and `measure_skills.py`
  first send `GET <base_url>/models` (localhost, 5 s timeout) and require the
  served model name among the ids, on both the generative path and Track B's
  `--scorer served` path. Any tier error (generative) or scorer call error (a
  raised exception; a normal "incomplete" top-k result is not one) fails the
  run with exit 2 and no results page. Predictions and metrics are still
  written for debugging. `--allow-tier-errors N` lets up to N through and
  prints the count at the top of the page. `measure_skills.py` now records a
  failed call as a `call_error` outcome and continues instead of aborting.
  *Evidence:* the lead's live check against the stopped server: exit 2,
  "cannot reach `http://127.0.0.1:18060/v1/models` to confirm the server
  is up", and no results page. *Commits:* `c171844`, `149ff10` (merge
  `d3e0c22`).
- **P50. The final run could not feed d6's calibration step.**
  `measure.py` refused `--predictions` on `--final` and `--acceptance` runs,
  but `track_a_calibration.py` needs the final run's predictions file.
  *Found:* the documentation agent, writing step 12 against `measure.py`.
  *Fix (h3):* `--predictions` is allowed with `--final` and `--acceptance`.
  The file holds ids, expected blocks and outcomes, never request text, and
  a test asserts it has no `text` key; `--details` stays refused there.
  `pipeline.sh`'s `measure-final` always passes `--predictions
  "$WORK/final/<name>"`, where `track_a_calibration.py` reads it.
  *Commits:* merge `d3e0c22`, then `9d7e724`.

### Found by the linters

- **P47. An unpinned model load in the drafting script.** `bandit` (B615)
  flagged that `draft_heldout.py` loaded Qwen3.5-4B from the Hugging Face
  Hub without a revision. *Fix:* pinned to the snapshot the draft was made
  with, `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. *Commit:* `fcf2701`.

### Found during the t18 re-review and t19/t21 preparation

- **P51. A foregone-reject candidate still cost a full thinking call.**
  171 of the 289 candidates remaining in the first re-review pass carried a
  stored reviewer-A "no" (the old AND rule made that a foregone reject), yet
  the run still paid for a full thinking reviewer-B call on each one.
  *Found:* the lead, reading the stall at 2026-09-23 19:50 (about 6 done in
  30 minutes). *Fix (f12, `9af65f8`):* skip candidates with a stored
  reviewer-A "no". *Superseded:* d8's clean-slate rule reverted this — a
  clean slate re-derives every candidate fresh, so a stored verdict can no
  longer shortcut a call. *Commit:* `9af65f8`, reverted by `ce72f01` (f13).
- **P52. The gateway never aborts an upstream request when a client times
  out.** With a 300 s client timeout and 6 retries, a long-thinking review
  kept generating on cortex (the gateway's Qwen 3.8 27B server) while the
  client gave up and re-sent the same request, up to 7 times; 5 requests in
  flight against only 4 workers meant orphaned generations. This collapsed
  the re-review rate from about 3 per minute to about 0.3 per minute.
  *Found:* the lead, probing two real reviewer calls (304 and 581 reasoning
  tokens but 182 s and 269 s wall time — about 90% queue wait). *Evidence:*
  vLLM's own metrics on cortex showed `finished_reason="abort"` = 0 over
  about 5,600 requests. *Fix (d9):* a timeout longer than the longest
  generation (900 s) and `--workers` matched to the gateway's
  `--max-num-seqs` (2), so no request queues behind another. *Commit:*
  `1a35540` (decision record); the temperature and timeout knobs landed in
  `7a2810a`/`e9d8495`.
- **P53. The reviewer ran at a hidden `reasoning_effort` of xhigh by
  default.** Cortex's chat template supports `reasoning_effort` xhigh
  (default), medium and low, and nothing in the pipeline ever set it, so
  every review before this point ran at xhigh (about 1,250 tokens each) with
  no visibility into the cost. Reading 13 xhigh rejections found 10 were
  correct read-only requests rejected on a misreading ("it only describes
  running the check instead of actually reporting") and 1 was a parser false
  negative. *Found:* the lead, running effort A/B on 20 xhigh verdicts (low
  and medium accepted 7-8 of the 10 xhigh rejections). *Fix (d10):* the
  read-only check's expected answer and the reviewer's system prompt both
  now say a check output is a complete answer;
  `NVSH_AUG_<ROLE>_REASONING_EFFORT` makes the effort explicit and recorded
  per call; `calibrate_reviewer.py` checks a setting against a known-good/
  known-bad probe before a full run. The prompt fix alone cut xhigh's mean
  from about 1,250 tokens to 445. *Commit:* `56765d1` (f15).
- **P54 (lapse l3). A run labelled 4K was served at 2K.** A t21 validation
  run recorded as "same result at 4K" was actually served at
  `--max-model-len 2048`: the env file's `MEASURE_CTX=2048` silently
  replaced the exported `MEASURE_CTX=4096`, and `measure.py --ctx 4096`
  labelled the results page from the flag alone, without checking what the
  server had actually loaded. *Found:* the lead, reading the serving record
  after reporting the result. *Fix:* `measure.py`'s preflight now requires
  the served model's reported `max_model_len` (from `GET /models`, the
  effective default included per a Codex finding) to equal the run's `--ctx`
  before any entry is scored; an exported `MEASURE_CTX` outranks the env
  file's value; `measure-val` and `measure-final` always pass `--ctx
  "$MEASURE_CTX"` themselves and refuse an extra `--ctx` from the caller; a
  non-2048 `measure-val` run is named `<name>-val-ctx<N>`. The invalid run
  is quarantined under `$WORK/measure/invalid/` rather than deleted.
  *Commits:* `cb7228f`, `b12c2b9` (Codex's effective-default finding),
  `9abe2df` (a fake scorer server made to report `max_model_len` like a real
  vLLM, so the check has a test), merged in `340b453`. Recorded as lapse l3
  in `580675a`.
- **P55. Reviewer A answers a different question than reviewer B.** A probe
  of reviewer A (Gemma 4 26B-A4B, no thinking) found 3 false accepts
  (escalate accepted for a request a listed check already answers) and 2
  false rejects (escalate rejected for a request that genuinely needs it),
  at about 48 tokens mean — it reads as answering "can the assistant handle
  this" rather than "is this response right". *Found:* the lead's reviewer
  probe ahead of t19. *Fix (d11):* t19's fresh variations are decided by
  reviewer B and the deterministic guards alone
  (`augment.py --decide-by reviewer_b`); reviewer A is still asked and its
  verdict is still recorded, so the comparison is not lost, but it no longer
  gates acceptance. *Commit:* `ed01957`, recorded in `580675a`.

### Found during t19 assembly and the first t22/t23 training runs

- **P56. `leakage_check` collapsed two different `test.json` files into
  one.** `leakage_check.py` keyed its protected files by basename, so the
  new split's `test.json` and issue 39's old `test.json` — both listed in
  `PROTECTED_EXTRA` — collapsed onto a single entry, and filtering by id
  could drop unrelated rows that happened to share an id; missing or null
  text also passed through unchecked. *Found:* Codex, reviewing the d14
  tooling before merge. *Fix:* key protected files by path, filter by row
  index, and exit 2 on missing text. *Commit:* `6ad2ab3`, merged in
  `9b6e559` (`f17`).
- **P57. The skills contamination scan re-scanned the same text thousands
  of times.** Every rendered training example repeats the same long system
  prompt (the tool table), so a scan over all 1,463 rendered rows re-ran the
  sliding-window contamination check on that identical prompt text on every
  row, taking hours to finish (see [Ideas forward](#ideas-forward) item 12,
  the underlying `build_bodies` cost). *Found:* the lead, timing the first
  full-file scan. *Fix (f18):* deduplicate training strings before scanning;
  by construction this gives the identical result. Verified: clean, 104
  evals, about 6 seconds. *Commit:* `f6e8bb7`, merged in `b869c5d`.
- **P58. Track B trained on the wrong, un-augmented file.** `train-scorer`
  trained on the raw `splits/train.json` — which still held the 52 old
  issue-39 test entries (already excluded from training) and a duplicate of
  a test entry, and held none of t18/t19's variations. *Found:* the lead,
  reading `train_scorer.py`'s input path against the frozen data flow before
  the first Track B run. *Fix (f19):* `train-scorer` trains on
  `data/train-augmented.json` (all 1,463 frozen examples) and refuses to run
  before `assemble` has produced it; the run name is now optional
  (`runs/scorer-<name>`). *Commit:* `a9d7c04`, merged in `5b93e90`.
- **P59. `train-scorer` couldn't merge without repeating `--train`.** The
  documented merge-only form of `train.py --merge-only` still required
  `--train`, which meant it could not be called the way the docs described.
  *Found:* the lead, live-running the documented Track B merge command.
  *Fix (f20):* `train-scorer` now runs `train.py --merge-only` directly
  after training, writes the `generation_config.json` and stages the result
  as `$REPO-scorer` with a revision, the same as `train` does for Track A;
  `--merge-only` no longer requires `--train`. *Commit:* `75d385b`, merged
  in `8141fad`.
- **P60. unsloth's Qwen3.5 processor crashed on the plain-string training
  format.** `FastLanguageModel.from_pretrained` returns a `Qwen3VLProcessor`
  for Qwen3.5, not a plain tokenizer; its chat template expects structured
  content parts and raised on the plain strings `build_dataset.py` renders.
  *Found:* the lead, first live training attempt for `a1`. *Fix (f21):* a
  new `text_tokenizer()` reaches the processor's inner `.tokenizer` for
  rendering and encoding. Verified live: the same chat template (7,755
  characters) and identical token ids as calling `AutoTokenizer` directly on
  a real training example (1,436 tokens). *Commit:* `bb87232`, merged in
  `e05563f`.
- **P61. A served Track B measurement failed silently.** Every served-scorer
  validation run exited 2 with every metric reported "not measured" and no
  stated reason. Cause: the repository's own `uv` environment has no
  `transformers` installed, the tokenizer was being loaded from the served
  model *name* rather than the model directory, and a scorer report carried
  no start-up row to explain a failure at all. *Found:* the lead, first live
  `--scorer served` run against `b1`. *Fix (f22):* `--tokenizer <model dir>`
  is now required for a scorer run, the training venv's site-packages must
  be on `PYTHONPATH` for any `--scorer` call (documented in step 11), and
  every run failure is printed to stderr instead of being swallowed.
  *Commit:* `1deb5b6`, merged in `b4eeed6`.
- **P62 (lapse l4). Track A's first merged checkpoints were bit-identical to
  the untuned base, and were measured that way before anyone noticed.**
  unsloth trains the *vision-language* model class
  (`Qwen3_5ForConditionalGeneration`), whose LoRA adapter keys live under
  `model.language_model.*`; `train.py`'s merge instead loaded the
  **text-only** `Qwen3_5ForCausalLM` class, so PEFT matched no adapter key
  and only printed a warning, never an error. Both `a1` and `a2` were merged
  this way, staged, served and measured on validation — scoring 0 of 32
  right proposals, indistinguishable from stock — before anyone opened the
  merged weights to check whether they actually differed from the base.
  *Found:* the lead, investigating why two independently trained recipes
  scored identically to stock. *Fix, first attempt (f23):* merge into the VL
  class instead; this loaded every adapter tensor correctly, but the VL
  save wrote doubled key prefixes
  (`model.language_model.language_model.*`,
  `model.language_model.visual.*`) that vLLM refused to load ("There is no
  module or parameter named 'language_model' in Qwen3_5Model"). *Fix, second
  attempt (f24):* map the VL adapter's keys onto the text-only model's
  parameter names, replace unsloth's `target_modules` regex (which never
  matches `linear_attn` on the text-only class) with the mapped module
  names, and refuse to finish the merge unless every adapter tensor loaded
  and at least one merged weight actually changed from the base. `a1` merges
  192 tensors this way, `a2` 372. **Consequence:** the tuned Track A
  checkpoints are the text-only `Qwen3_5ForCausalLM` class (the vision tower
  is dropped), while stock stays the full
  `Qwen3_5ForConditionalGeneration` — a difference to account for in any
  memory or latency comparison between them (see [Not verified
  yet](#not-verified-yet)). *Commits:* `97a69fc` (merged `6a98501`, f23),
  `c6dd8de` (merged `a7c81f8`, f24); lapse l4 recorded in `673ccb5`.
- **P63. vLLM start-up failures during concurrent training showed only a
  truncated log line.** Twice, starting the pinned vLLM for a measurement
  failed with nothing more informative than "Engine core initialization
  failed" in the helper's last 40 log lines, while Track A trained on the
  same GPU; retried later on an otherwise-idle GPU, the same server started
  fine. *Found:* the lead, debugging the two failed start-ups. *Fix (f25):*
  `serve_for_measure.sh wait <port> [<full log path>]` keeps the server's
  complete log on a failed start-up instead of only the tail; the pipeline
  passes `$WORK/measure/<label>.serve.log`. *Commit:* `6f65887`, merged in
  `45c5ae8`.
- **P66 (lapse l5). A Track B scorer measured without `--scorer` scores as
  a broken generative model, not as a scorer.** A `measure-val` call against
  a Track B run, made without `--scorer served` or `--scorer in-process`,
  ran the checkpoint as a generative tool-caller instead: 0 of 32 right
  proposals and about 100 generated tokens per decision, where a scorer
  should generate 0 (it only reads label log-probabilities). *Found:* the
  lead, reading a result that looked like an untrained checkpoint on a
  model that had trained and merged cleanly. *Cause:* nothing distinguished
  a scorer run from a generative one at measurement time, so leaving out
  `--scorer` silently measured the wrong thing rather than refusing.
  *Fix:* `measure-val`/`measure-final` now check whether
  `runs/<name>/train-log.json` carries `train_scorer.py`'s own training
  objective, and refuse with a hint if so and no `--scorer` mode was given.
  *Commit:* `d39c5e3`.

## Troubleshooting: symptoms and causes

Organized by what you actually see on screen or in a log, so you can look
up a symptom without already knowing its cause. Every row links to the full
write-up in [Pitfalls hit, and the fix for
each](#pitfalls-hit-and-the-fix-for-each) or the decisions/deviations list
above.

| You see | Cause | Fix | Ledger |
|---|---|---|---|
| A served Track B measurement exits 2 with every metric "not measured" and no reason given | The repo's own `uv` environment has no `transformers`, and the tokenizer was being loaded from the served model *name* instead of the model directory | Put the training venv's site-packages on `PYTHONPATH` for any `--scorer` run and pass `--tokenizer <model dir>` (step 11) | P61 |
| A Track B scorer measures at 0 of 32 right proposals with about 100 generated tokens per decision, on a checkpoint that trained and merged cleanly | The measurement left out `--scorer served`/`--scorer in-process`, so the checkpoint ran as a generative tool-caller instead of as a scorer (a scorer should generate 0 tokens) | Always pass `--scorer served` or `--scorer in-process` for a Track B run; `measure-val`/`measure-final` now refuse and hint instead of silently measuring the wrong thing | lapse l5, P66 |
| A tuned checkpoint scores exactly like stock — 0/32 right proposals, identical latency, as if nothing had been trained | The merge loaded a different model class than unsloth trained, so PEFT matched no adapter key and only printed a warning ("Found missing adapter keys"), not an error; the merged file is bit-identical to the base | Before trusting any merge, diff a merged weight against the base and confirm it changed, and confirm 0 missing-key warnings; `train.py`'s merge now refuses to finish otherwise | lapse l4, P62 |
| vLLM refuses to load a merged checkpoint: `There is no module or parameter named 'language_model' in Qwen3_5Model` | An earlier merge attempt saved into the vision-language class with doubled key prefixes (`model.language_model.language_model.*`) that vLLM's loader rejects | Merge into the text-only class with the adapter keys mapped onto its parameter names instead (f24) | P62 |
| A measurement server fails to start ("Engine core initialization failed") or dies mid-run (a `tier_error`, "server unreachable") while something else is training | A GPU allocation failed on unified memory: training sinks *free* memory to a few GiB even while *available* stays high (page cache), and GB10 does not evict page cache to satisfy a GPU allocation the way it would evict it for ordinary RAM pressure | Measure only when nothing is training on that machine's GPU; check what else holds GPU memory (`nvidia-smi --query-compute-apps`) and stop anything unused first | P63, P64 |
| A 4K measurement also fails to start with "Engine core initialization failed", but the *full* serve log ends with `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (254)` rather than an `NV_ERR_NO_MEMORY` line | Not memory pressure from another process: at 4K, the default `MEASURE_GPU_FRACTION` leaves too little room for the KV/Mamba cache to cover the default `max_num_seqs`, since each Gated-DeltaNet decode sequence needs one Mamba cache block | Read the full serve log's own root-cause line to tell this apart from P64 at a glance; raise `MEASURE_GPU_FRACTION` (about 0.12 for a 4K run) in a per-run env file — an exported value alone is not enough, since only `MEASURE_CTX` is export-first | P65 |
| Every prediction in a run comes back a `tier_error` ("server unreachable mid-run"), right after the server itself reported ready | Not P64/P65's GPU cache pattern: the run overlapped a host network reconfiguration on the measurement machine, which briefly took the server's own network path down from under it | Re-run on a quiet host once the network change has settled; check for this before assuming a GPU-memory cause | (environment condition, not a code bug) |
| Training crashes with `TypeError: string indices must be integers` (or similar) inside `apply_chat_template` | `FastLanguageModel.from_pretrained` returns a processor (`Qwen3VLProcessor`), not a plain tokenizer, for Qwen3.5; its chat template expects structured content, not the plain strings the dataset builder renders | Reach the processor's inner `.tokenizer` for rendering and encoding instead of the processor itself (f21, `text_tokenizer()`) | P60 |
| A run recorded as "measured at 4K" actually served at `--max-model-len 2048` (visible in the server's own record) | A value set in the sourced env file silently overrode an exported shell variable of the same name | Export the variable and confirm the served model's reported `max_model_len` matches `--ctx` before trusting a result; the preflight now refuses a mismatch outright | lapse l3, P54 |
| A served Track B scorer reports ECE/Brier as not available, with 0 lines carrying a complete label distribution | The other candidate labels' logprobs fell outside vLLM's returned top-k, so the result is marked incomplete and is never renormalised over a partial set | Score calibration with `--scorer in-process` instead of `--scorer served` (the two still agree on the actual decisions) | r8, d15 |
| A Jetson skills run reports 104 of 104 as call errors at 2K context | The skills prompt, which lists all 38 tools, is 3,939 tokens on its own — before any answer | Serve every skills measurement at `MEASURE_CTX=8192`, stock included, so the numbers stay comparable | d13 |
| A reviewer pipeline crawls at a fraction of its earlier rate (for example 0.3 per minute, down from about 3) against a shared, busy model server | The client's timeout was shorter than some real generations, so it gave up and retried while the abandoned generation kept running upstream — orphaning several requests per slot | Set the timeout longer than the longest real generation and match `--workers` to the server's own concurrency limit (`--max-num-seqs`) so no request queues behind another | P52 |
| A judge/reviewer model rejects requests that read as obviously correct, often with reasoning like "it only describes running the check instead of reporting" | The reviewer's prompt did not say a check's own output counts as a complete answer, and it was running at a high, unrecorded reasoning effort by default | Fix the prompt wording, make the reasoning effort explicit and recorded, and run a known-good/known-bad calibration probe before trusting a full pass | P53, d10, lapse l2 |
| A wrapper script's log line claims `rc=0` right after a command that visibly failed | `echo "$(date -Is) rc=$?"` runs the command substitution `$(date -Is)` first, which resets `$?` before `echo` ever reads it | Capture `rc=$?` on its own line immediately after the command, before anything else runs | (wrapper-script lesson, run log 2026-09-24 05:44) |
| A test fails, but only when run as part of the full suite while something else (for example a training job) is using the machine, and passes reliably alone | A timing-based test assumption breaks under real machine load | Known and named (`tests/test_setup_timing.py::test_setup_does_not_meaningfully_slow_down_prompt_startup`); not a bug in the code under test | [Not verified yet](#not-verified-yet) |

## Not verified yet

- **The GGUF half of f11 on a live run**: the lead's live re-check covered
  the AWQ build. A bf16 GGUF, imatrix and `Q4_K_M` through the stage itself
  is not recorded yet (the t16 spike ran the same tools by hand).
- **Serving the AWQ build through nvsh's launcher**: not possible without
  the extra `--limit-mm-per-prompt` argument (step 13).
- **GGUF on AGX Orin**: the llama.cpp build has only run on spark.
- **The GGUF's sampling settings**: d3 covers "the GGUF's sampling metadata",
  and no step writes it yet.
- **Track A calibration on the final side** (d6): the tool is verified on
  stock with issue 39's old validation split; it has not yet run on a final
  run's predictions.
- **The re-review's exact filter command** (step 5): the settings that
  ended up running (clean slate, temperature 0.2, xhigh, 900 s, 2 workers)
  are recorded, but the exact command that filtered the 1,720 stored
  candidates to the 1,176 train-side ones is not.
- **nvsh's runtime and unparsed Qwen tool calls** (P45): `LfmTier` still
  treats a failed parse as an explanation. Out of scope here (c9); tracked
  as [nvsh issue #50](https://github.com/agentculture/nvsh/issues/50).
- **Gaps left by h2**: the measure stages cannot name an AWQ build yet;
  `measure-final`'s results page still goes to `docs/benchmarks/` by
  default; `--limit-mm-per-prompt` on LFM2.5 is untested.
- **Thinking off for LFM2.5**: the LFM env example now also sends
  `enable_thinking` false. Whether LFM2.5's chat template ignores it has
  not been checked.
- **A flaky timing test, now identified.** The two unidentified test
  failures noted after the f10 merge are consistent with
  `tests/test_setup_timing.py::test_setup_does_not_meaningfully_slow_down_prompt_startup`:
  it fails when the machine is under load (a training run on the GPU) and
  passes reliably alone. Not yet fixed — it is a timing assumption in the
  test, not a bug in the code it tests.
- **unsloth's Gated-DeltaNet path runs pure PyTorch, not fused kernels.**
  `flash-linear-attention` and `causal-conv1d` are not installed in the
  training venv, so unsloth's GDN (`attn-mlp-gdn`) training falls back to a
  slower pure-PyTorch path. `a2`'s wall time (26-31 minutes, about the same
  as `a1`) did not show an obvious slowdown in this run, but a larger recipe
  (`a3`, `a4`) may.
- **Track A's tuned checkpoints are text-only; stock is not.** The lapse l4
  merge fix (P62) drops the vision tower from every tuned checkpoint
  (`Qwen3_5ForCausalLM`), while stock stays the full vision-language class
  (`Qwen3_5ForConditionalGeneration`). Any memory or latency comparison
  between tuned and stock should account for this difference in model class,
  not attribute it entirely to the fine-tune.
- **Whether stock meets any bar**: the t13 live check was a pipeline check,
  not the baseline run.

## Ideas forward

None of these has been done yet.

1. **More data.** If data limits a result, generate more train-side data
   through the same teachers and guards. The operator: "we can always
   generate more data if needed" — this is an acceptable fix, but the
   training data is now frozen (c40), so doing it would be a recorded
   deviation, not a quiet re-run. The strongest candidate: validation's
   `a2` errors are dominated by missing-argument requests ("Set the power
   mode" with no mode given) that get an invented argument instead of an
   escalation — more training examples of exactly that shape are the
   obvious next data request if a later checkpoint still shows the same
   error class.
2. **LoRA on the linear-attention layers — now run.** `train.py --targets
   attn-mlp-gdn` (`a2`) beat the default attention/MLP-only targets (`a1`)
   on validation (r9: abstain recall 81% vs 75%, 1 wrong mutating vs 2); `a3`
   and `a4` extend `a2`'s recipe rather than `a1`'s.
3. **Latency.** bf16 decodes at about 10 ms per token, `Q4_K_M` at about
   4.3 ms and AWQ at about 6 ms. The 250 ms bar may need a quantized build
   (plan risk r10, ledger P8).
4. **Measurement gaps.** `measure-skills` cannot measure the stock copy
   until `measure_skills.py` can attach; it already has `--url`, which needs
   wiring. The measure stages cannot name an AWQ build yet.
5. **nvsh runtime follow-ups.** [Issue #50](https://github.com/agentculture/nvsh/issues/50)
   (unparsed tool-call markup shown as an explanation). The managed launcher
   also refuses a local model path and cannot pass extra vLLM arguments,
   which is why d7 exists. Both are candidate follow-up issues.
6. **The LFM2.5 side.** Check that LFM2.5's template ignores
   `enable_thinking=false`. [`lfm-finetune.md`](lfm-finetune.md) still describes `measure-final` as
   "stock and r1 back to back on the test side"; since d7 it measures one
   model per call.
7. **Track B serving.** Serve with vLLM `--max-logprobs` of at least 22, or
   score in process (plan risk r8, ledger P20).
8. **Reviewers.** Across two waves Codex found 15 real defects where the
   qwen worker reviewer approved everything. Keep a strong second reviewer.
   The lead's live checks on real data and tools found the rest.
9. **Cortex's own serving settings.** Raising `--max-num-seqs` from 2 to 4
   and `num_speculative_tokens` from 7 to 3-4 might raise the gateway's
   shared throughput for the reviewer calls, but this needs a benchmark
   first and is model-gear's call, not this pipeline's.
10. **Structured verdict output.** The reviewer's answer is still free text
    that a hardened but still blacklist-style parser reads (d10); asking the
    model for a structured (for example JSON) verdict would remove a whole
    class of parser edge cases.
11. **A larger reviewer probe.** The 29 bad items in the current probe (d10,
    d11) only bound the false-accept rate to below about 10%; a bigger probe
    would tighten that bound.
12. **The Jetson skills build's contamination scan is very slow — fixed for
    training-side scans.** `build_bodies`'s sliding-window scan over long
    `SKILL.md` bodies pegs a CPU core for over 10 minutes after the build's
    main outputs are already written. f18 (ledger P57) fixed the case that
    hit this run — scanning the same repeated system prompt across every
    rendered training example — by deduplicating training strings before
    scanning. The underlying `build_bodies` cost against `SKILL.md` bodies
    itself is unfixed; `bodies.json` is not used by issue 46, so it is still
    worth fixing only if a future run needs it on a schedule.

## Run log (issue 46)

Filed as each step happens. Times are local (+03:00).

### 2026-09-23: spikes, before any data work (t15, t16)

**Base model.** `Qwen/Qwen3.5-0.8B` at
`2fc06364715b967f1860aea9cf38778875588b17`, Apache-2.0. Architecture
`Qwen3_5ForConditionalGeneration`: 24 layers (3 Gated-DeltaNet
linear-attention to 1 full-attention), a vision encoder, an MTP head,
vocabulary 248,320. The chat template has no generation-block markers,
writes tool calls as XML, always emits an empty think block, and raises on
string arguments (P1-P3).

**unsloth** loads it for text-only LoRA (1.79 GB GPU) and keeps eos
`<|im_end|>` (248046). Default LoRA targets: 96 modules, 12.8M parameters,
no adapter on the Gated-DeltaNet projections (P6).

**Serving (t15).** The pinned vLLM image
(`sha256:8bd082c2...`, vLLM 0.26.1rc1.dev942) registers
`Qwen3_5ForConditionalGeneration`. `qwen3_coder` and `qwen3_xml` round-trip
all three outcomes; `hermes` fails; `qwen3_coder` chosen. Stock bf16 warm
latency about 450-490 ms for about 50 tokens (P8).

**Quantization (t16, 14:29, stock only).** llama.cpp master `633733d0`,
CUDA arch 121: `convert_hf_to_gguf.py` text-only, 335 tensors, bf16 1.55 GB;
`llama-quantize` to `Q4_K_M`, 542 MB. `llama-server --jinja` parses tool
calls with thinking off; decode about 4.3 ms per token (47 tokens in about
196 ms). AWQ W4A16 through llm-compressor 0.14.0: 1.1 GB, serves in the
pinned vLLM with the Marlin kernel, tool calls parse, warm about 200-330 ms
(P25, P26).

**Memory cap.** Measured on spark and spark2 (P32).

### 2026-09-23: tooling built and reviewed (t1-t14, f1-f9)

Tasks t1 to t14 were built by worker agents in waves and merged by the lead
(merge list: `git log --oneline --first-parent 8f01db1..HEAD`). The lead's
own checks, a Codex between-wave review (8 findings) and a worker
between-wave review (2 findings) found the problems in ledger P4, P10,
P12, P13, P17, P19, P23, P24 and P27-P31; each fix is merged. Lapse l1: the
merge gate had not fed scorer output into metrics before f1 was found.

Operator decisions recorded as deviations: d1 (fixed grounding snapshot),
d2 (strict abstention precision), d3 (temperature 0), d4 (re-split before
re-review, 4 workers), d5 (this guide).

### 2026-09-23: t13 live check (stock on validation, attach mode)

The measurement pipeline works end to end. Stock proposed 1 of 32 right on
validation, and 45 of 66 lines ended with no decision. 0 of 66 lines carried
a Track A candidate distribution (P11, plan risk r12). This check also found
that vLLM sampled at temperature 1.0 (P9); after d3, three runs gave
byte-identical output.

### 2026-09-23 13:51: reviewer-B pilot, decided before the run (t18, c41)

150 stored nvsh candidates, seeded (`Random(46)`): 100 accepted and 50
rejected, shuffled. New reviewer B: Qwen 3.8 27B, thinking off, 1,024-token
budget. Compared with the stored Nemotron verdicts (an accepted record is an
implicit yes). Rule fixed first: at least 80% agreement (120 of 150) means
the full run goes non-thinking; otherwise thinking on (8,192-token budget,
300 s timeout).

### 2026-09-23 14:25: pilot result

150 processed, 0 errors. Accepted 20, rejected 130. Agreement with Nemotron
**56 of 150 (37%)**, below 80%, so the full re-review runs with thinking on.
Reading 9 rejection reasons showed over-literal readings (P14).

### 2026-09-23 14:53: thinking probe and operator decisions

20 pilot candidates re-reviewed with thinking on: 878 s (about 44 s each),
15 of 20 accepted (non-thinking had accepted 3 of the same 20), agreement
with Nemotron 13 of 20. The operator chose: thinking on, after the re-split,
only candidates whose source stays on the train side, 4 workers (d4); and
temperature 0 through `generation_config.json` (d3).

### 2026-09-23 14:54: re-split with seed 46 (t19 step 1, moved before t18 by d4)

`split.py --corpus nvsh/tiers/corpus/dev.json --seed 46`: train 301, val
66, test 64 (hashes in [Data and split](#data-and-split)). Re-review input:
1,176 of 1,720 stored candidates. 95 new train-side sources have no
variations yet. The lead has not opened `test.json`; only counts and hashes
were printed.

### 2026-09-23 14:59: full re-review started

Thinking on, 4 workers, over the 1,176 train-side candidates. About 2 per
minute; about 9 hours estimated *(in progress)*. By 15:10: 24 of 1,176
done, 19 accepted, 0 errors (about 2.2 per minute).

### 2026-09-23: sealed held-out drafted (t17)

72 entries by Qwen3.5-4B from the operation table only (seed 46,
temperature 0.7, thinking off): 39 operation (15 of 16 operations), 17
escalate, 16 explain. sha256 `95c7cd3e...2107f`. Awaiting the operator's
review; not read by the lead. The operator is reviewing it now. One
operation, `network_info`, has no drafted entry; the lead told the
operator.

### 2026-09-23: spark2 set up for Track B (t20)

The venv was built from `requirements-train.txt` and matches spark's
versions exactly, with CUDA available (h34). `uv` had to be called by its
full path over ssh, and a private `HF_HOME` replaced the root-owned shared
cache (P33).

A probe of the memory cap found that GPU allocations on GB10 are not charged
to the cgroup: an 8 GB CUDA tensor was allocated and filled under a 1G cap
with swap off (P34, blocking plan risk r13). The serving containers'
restart counts were unchanged across the probe. A GPU-side cap is being
added (f10, pending). The earlier finding stands: `MemoryMax` alone lets a
process spill into swap, so `MemorySwapMax=0` stays required (P32).

### 2026-09-23 ~15:25: GPU-side memory cap merged (f10), quantize gap found

f10 is merged (`686e1c8`) and plan risk r13 is resolved (P34). On spark2,
with 30 GB available and a 26 GB floor, the watchdog stopped an 8 GB CUDA
hog after 5 s at 22,882 MB available (return code 3), left no Python process
on the GPU, and the serving containers did not restart.

After the merge, one full test-suite run had 2 failures whose names were not
captured; the six runs after it were green. The cause is unidentified.

Reading `quantize.py` against the t16 spike log found that its AWQ and GGUF
paths differ from what the spike ran (P35, plan risk r14). The fix, f11, is
in progress.

### 2026-09-23: memory caps exported to child processes (P36)

The documentation agent found that `pipeline.sh` did not export the f10
settings, so a GPU budget set in the env file would not have reached the
trainer. Fixed in `6d805d5`: the caps are exported, both env examples name
them, and `pipeline.sh status` prints them as a child process sees them.

### 2026-09-23 ~15:50: the d3 generation config breaks saving (P37, P38)

The lead's live run of f11's `awq_oneshot.py` on the stock copy failed to
save: transformers refuses a generation config with temperature 0.0 and
`do_sample` false, which is exactly the d3 file. Reproduced with
transformers 5.17.0 (AWQ venv; 22 MB output, no weights) and 5.5.0
(training venv). The training merge is fixed in `51d91b1`
(`save_valid_generation_config()`, then `gen_config.py write` restores the
served file) and verified end to end on the stock copy. The same commit stops
`gen_config.py` writing null `bos_token_id` and `pad_token_id`. The AWQ side
went back to f11 *(pending)*.

### 2026-09-23 ~15:45: AWQ and GGUF recipes fixed (f11, r14 resolved)

f11 is merged (`07422cd`). `quantize.py` now runs AWQ through
`awq_oneshot.py` in the separate AWQ venv (`AWQ_PY`), with the spike's
recipe, the support files copied and the generation config written, and
converts the GGUF from bf16. The lead's live check on the stock copy: the
first run failed at save (P37); after the fix, 1.1 GB with weights, served
by the pinned vLLM with `--limit-mm-per-prompt` and no override flag,
Marlin kernel, temperature 0 taken from the file, three byte-identical
runs, tool calls parsed (P35).

### 2026-09-23 ~16:15: wave-2 review, and the held-out draft

The between-wave review of wave 2: the qwen worker reviewer returned OK on
every file; Codex found seven correctness problems, recorded as P40 to P46
and sent to review-fix tasks g1 to g4 *(in progress)*. Deviation d5 is now
in the committed plan record (`3df700c`).

The operator edited the held-out draft. The original is kept as a separate
v1 file (sha256 `95c7cd3e...2107f`). The latest save does not parse as JSON
(line 505, column 5); the operator is fixing it. The held-out set is not
sealed yet.

### 2026-09-23 ~16:25: held-out set sealed (t17 done)

The operator fixed the file and finished the review. Sealed read-only copy:
sha256 `5eb650f91c44f54ab112665dc40d71f66fe118efc79d8bcea728ed9ce0a40198`.
The lead's structural checks, with no entry text read: 69 entries, unique
ids, every expectation well formed; 41 operation (all pass
`nvsh.ops.table.validate`), 13 escalate, 15 explain; 15 of 16 operations
covered (`network_info` has none, by the operator's choice); 0 exact
overlaps with `dev.json` or the 1,720 stored variations. It stays unread
until the single final run (t24).

### 2026-09-23 ~16:55: wave-2 fixes merged (g1-g4)

All seven wave-2 findings are fixed and merged, and the full suite is green
(P40 to P46). The P42 fix showed that the real Qwen tokenizer splits label
names into several tokens, so Track A calibration from generation
log-probabilities is probably not measurable; how to report it is open for
the operator. The pipeline's measure stages now enforce the stock copy, the
grounding snapshot and thinking off. Because the managed launcher refuses an
absolute model path, stock is measured in attach mode, and the skills evals
cannot measure the stock copy yet (P48, open).

### 2026-09-23 ~17:05: operator decisions d6 and d7, issue #50

The operator approved two deviations. d6: Track A calibration is scored
exactly, in process, by teacher-forcing each candidate's tool-call prefix
and normalising over the candidates (task h1, pending); this settles P42's
open question. d7, proposed by the lead: every model is measured in attach
mode against one committed pinned-vLLM helper with identical flags, one
model per `measure.py` call (task h2, pending); this resolves plan risk r15
(P48). The runtime side of P45 is filed as
[nvsh issue #50](https://github.com/agentculture/nvsh/issues/50).

### 2026-09-23 ~17:40: h1 and h2 merged; a dead server looks like a baseline

h1 (`6806662`): `track_a_calibration.py` scores Track A's candidates
exactly, in process (d6). On the stock copy with issue 39's old validation
split and t13's predictions: 66 of 66 lines got a distribution summing to 1,
top label right on 30 of 66, ECE 0.230, Brier 0.754, 257 s. P42 is fixed.

h2 (`1062123`): `serve_for_measure.sh` and single-model measure stages
(d7). The helper served the stock copy with temperature 0 from its own
`generation_config.json`, and the measurement reproduced t13's outcome
counts exactly (45 no decision, 12 explain, 6 escalate, 3 propose).

Pointing `measure.py` at a stopped server exited 0 and wrote a results page
with every line a tier error (P49). The guard is task h3 *(pending)*.

### 2026-09-23 ~18:40: h3 merged (P49, P50 fixed)

h3 (merge `d3e0c22`, then `9d7e724`): `measure.py` and `measure_skills.py`
check the served model is listed before the first entry, and a tier or
scorer call error fails the run with exit 2 and no results page. Against the
stopped server, `measure.py` now exits 2 ("cannot reach
`http://127.0.0.1:18060/v1/models` to confirm the server is up") and writes no
page. `--predictions` is allowed on final and acceptance runs, and
`measure-final` keeps each final run's predictions in `$WORK/final/<name>`
for the d6 calibration step.

### 2026-09-23 19:50-21:00: t18 stall, f12, d8 clean slate

The re-review slowed to about 6 done in 30 minutes: cortex (model-gear's
primary on spark2) runs 2 sequences with 4-5 waiting, shared with other
work. Four requests failed as "timed out" (the 300 s timeout includes queue
wait). Reading the remaining queue found 171 of 289 candidates carried a
stored reviewer-A "no" under the old AND rule — a foregone reject that still
cost a full thinking call (P51); f12 (`9af65f8`) skipped them.

At 20:40 the operator decided every re-review must instead be a clean
slate: not aware of previous decisions, so the fresh thinking judgement is
the real signal. The lead verified every reviewer call is already one
independent two-message request (the rules plus one candidate's text and
expected answer, no history, no examples, no prior verdict). Deviation d8
recorded and approved.

f13 (`ce72f01`) implements the clean-slate rule (a fresh reviewer B verdict
plus the deterministic guards decide; stored verdicts kept only as
`prior_verdicts`) and reverts f12; `--rederive-clean-slate` re-derives what
the old rule would have said, offline, for comparison. Codex found 5 issues
in the offline re-derive (duplicate ids twice, a wrong agreement count, a
needless reviewer config, dry-run writing output), fixed in `ccd8a03`. Full
suite: 3,980 before the merge, 3,986 after.

The operator stopped the old, non-clean-slate run by hand (the auto-mode
classifier had refused to kill it) and cleared the run wrapper so it would
not resume under the old rule. The old-rule outputs are archived, not
committed. Re-deriving the old rule against the 894 candidates already
reviewed gave 826 accepted, 68 rejected (the old rule itself had said
798/96 — 28 flip to accepted); agreement with the old, stored reviewer-B
verdicts was 832 of 894. The clean-slate re-review for the remaining 282
candidates resumed about 21:00.

### 2026-09-23 21:10-21:45: why t18 collapsed, d9

The rate after the 20:52 resume was about 0.3 per minute, down from about 3
per minute in the afternoon. A probe of two real reviewer calls found 304
and 581 reasoning tokens but 182 s and 269 s wall time — about 90% queue
wait, not thinking. vLLM's own metrics on cortex (Qwen 3.8 27B NVFP4,
`--max-num-seqs=2`, DSpark speculative decoding with 7 draft tokens) showed
`finished_reason="abort"` = 0 over about 5,600 requests: the gateway never
cancels an upstream generation when a client disconnects. With a 300 s
timeout and 6 retries, a long-thinking review kept generating upstream while
the client gave up and re-sent it, up to 7 times, and 5 requests in flight
against 4 workers orphaned generations (P52). The operator noted only this
pipeline uses cortex, sharing about 45 tokens per second across the 2
serving slots.

The operator also judged temperature 0.7 too hallucination-prone for a
judge, and asked for 0.1-0.3. f14 (`e9d8495`) added
`NVSH_AUG_<ROLE>_TEMPERATURE` (default 0.7, range 0-2), recorded as
`record.temperatures` and `verdicts.reviewer_b.temperature`; Codex found no
issues; full suite 3,992 (one earlier run had one transient failure, not
reproduced). Deviation d9 recorded (`1a35540`): reviewer temperature 0.2,
timeout 900 s, `--workers 2`. The operator decided to redo all 1,176 at 0.2
so every verdict comes from one setting; the 0.7 clean-slate outputs are
archived, not committed.

At 22:00 the operator considered a round-robin reorder with a 3-accepted-
per-source coverage stop, then decided to let the class-sorted run finish
all 1,176 instead (ETA about 06:00): early stopping would be unsafe because
escalate (296) and explain (279) entries come last in the class-sorted
queue. No reorder; no d10 from this decision (d10 is the effort/prompt fix
below).

### 2026-09-23 22:00-22:50: reviewer effort, d10, lapse l2

Cortex's chat template supports `reasoning_effort` xhigh (the default),
medium and low; nothing in the pipeline had ever set it, so every review to
this point ran at xhigh (about 1,250 tokens each) with no visibility into
the cost (P53). An effort A/B probe on 20 xhigh verdicts found low and
medium accepted 7-8 of the 10 xhigh rejections; reading them showed 10 of 13
xhigh rejections were correct read-only requests rejected on a misreading
("it only describes running the check instead of actually reporting" — the
prompt itself asked for exactly that), and 1 was a parser false negative
("yes: ...; no change or broader investigation is required").

The operator approved: stop, fix the prompt and parser, calibrate the
reviewer, then restart. The xhigh/0.2 pass done so far (158 records) is
archived, not committed. Deviation d10 recorded; lapse l2
(grader-unverified — the reviewer had been used as the grader for full
passes without a known-good/known-bad check) filed.

f15 (merge `56765d1`) fixed the read-only check's expected answer and the
reviewer's system prompt to say a check is a complete answer; kept the
verdict parser conservative (a loosening tried during this fix let real
rejections through and was reverted) and hardened it over three Codex
review rounds (markdown/underscore-wrapped "no", a trailing "no", ", no,",
"Yes, not ...", certainty words in the first four words) — a replay of 977
stored accepts found 0 newly rejected; added a deterministic `handoff_check`
guard that rejects requests asking for the hand-off in words (2 of 1,176
candidates, no false positives against phrases like "privilege escalation"
or "UEFI handoff"); added `NVSH_AUG_<ROLE>_REASONING_EFFORT`, a shared
`chat_payload()`, and `calibrate_reviewer.py`. Full suite: 4,041.

A reviewer probe (53 items: 24 good, 29 bad; final parser and guards) found:
low 0 false accepts / 0 false rejects (227 tokens mean); medium 0 false
accepts / 3 false rejects, all parser costs on wordy yeses (312 tokens);
xhigh 0 false accepts / 0 false rejects (445 tokens — down from about 1,250
before the prompt fix). The 29 bad items only bound the false-accept rate to
below about 10%. The operator chose xhigh. A fresh clean-slate pass of all
1,176 candidates started about 22:50 (ETA about 3 hours); one empty-reply
error was seen so far and will be retried by resuming.

### 2026-09-24 ~00:00-02:00: t20 re-sync, t21 stock baseline, t22 target option, t19/d14 prepared

spark2 was re-synced for Track B; its venv versions matched spark's exactly
(no drift since the earlier setup).

t21, the stock baseline, ran on validation with the pinned-vLLM measurement
helper (d7): at 2K, 0 of 32 right proposals, 10 of 16 escalations, 7 of 18
explanations, 0 wrong mutating (1 mutating proposal with wrong arguments),
warm median 678 ms / p95 2,072 ms; at 4K (served at `--max-model-len 4096`,
verified against the served model), the same right-proposal, escalation and
explanation counts, warm median 660 ms / p95 2,058 ms. The Jetson skills
eval (104 entries), served at `MEASURE_CTX=8192` (d13, since the skills
prompt with all 38 tools is 3,939 tokens on its own): 42 of 104 (40%)
overall, 14 of 34 skill-named, 28 of 70 not-named (bsp 16/48, device 26/56);
42 correct, 27 wrong_skill, 31 no_call, 4 several_calls, 0 call_error, 0
think blocks. Deviations d12 (stock's test-side/held-out run happens once,
in t24, alongside the tuned checkpoints) and d13 (skills served at 8K for
every model) recorded (`580675a`).

While measuring, a labelling bug surfaced: a t21 run recorded as 4K had
actually been served at 2K, because the env file's `MEASURE_CTX` silently
overrode an exported one and `measure.py --ctx` trusted its own flag instead
of the server (lapse l3, ledger P54). Fixed and merged (`cb7228f`,
`b12c2b9`, `9abe2df`, `340b453`): `measure.py` now checks the served
model's reported context before scoring, an exported `MEASURE_CTX` wins over
the env file, the measure stages always pass their own `--ctx` and refuse a
caller's extra one, and a non-2048 `measure-val` run is named
`<name>-val-ctx<N>`. The invalid earlier run is quarantined, not deleted.

t22 added `train.py --targets attn-mlp|attn-mlp-gdn` (`f283dfe`, branch
`agent/q46-f17`) so the linear-attention LoRA comparison (r9, ledger P6) can
be run explicitly and the choice is printed and logged.

t19 was prepared but not yet run: of the 95 train-side sources without
stored variations, 52 are old issue-39 test entries already excluded from
training, so t19 augments only the remaining 43 (22 operation, 12 explain, 9
escalate; 258 variations at 6 per source). Reviewer B alone decides
acceptance for these (deviation d11, `ed01957`), with reviewer A still asked
and recorded for comparison (ledger P55). `leakage_check.py` and a stricter
`assemble` (deviation d14, `4580c06`) check every new variation against
validation, test, `PROTECTED_EXTRA` and the sealed held-out by exact match
or near-duplicate (5-token shingle or word-set Jaccard >= 0.8), using ids
only; a dry run against the current data found 52 issue-39 test sources and
1 train source identical to a test entry inside the 301-entry train split,
plus 7 already-reviewed variations matching protected text. t19 and t22's
commits are on branch `agent/q46-f17`, under review and not yet merged into
this branch.

As of this update (about 03:00), the clean-slate re-review (t18) has about
935 of 1,176 candidates done, about 97% accepted.

### 2026-09-24 04:17-04:19: t18 done

All 1,176 candidates finished at the fixed clean-slate settings (thinking on,
temperature 0.2, `reasoning_effort` xhigh, 900 s timeout, 2 workers): 1,042
accepted (88.6%), 134 rejected. 3 replies came back empty and were retried
once (2 accepted, 1 rejected on retry). Agreement with the stored (old)
reviewer-B verdicts: 1,013 of 1,173 on the main pass. Output hashes:
`rereview-accepted.jsonl` sha256 `27b20108...`, `rereview-rejected.jsonl`
sha256 `7692af9a...`. t19 generation started immediately after
(`run-augment-new43.sh`, 43 sources).

### 2026-09-24 05:44: t19 done — data FROZEN (c40; any later change is a deviation)

t19 generation (43 sources, `--decide-by reviewer_b`, d14): 258 variations,
229 accepted by reviewer B plus the guards, 29 rejected; reviewer A alone
would have rejected 60 of the 229 (ledger P55). 0 errors, 0 retries, about 70
minutes (04:19-05:27).

Codex's review of the t19 tooling found a P1: `leakage_check` keyed
protected files by basename, so the new split's `test.json` and issue 39's
`test.json` (both in `PROTECTED_EXTRA`) collapsed into one; filtering by id
could also drop unrelated rows sharing an id, and missing or null text
passed through unchecked. Fixed: keyed by path, filtered by row index,
exits 2 on missing text (ledger P56). *Merge:* `9b6e559`.

Assemble (05:29): `nvsh-accepted.jsonl` (re-review accepts plus t19 accepts)
held 1,271 records, 0 duplicate ids. `merge_variations`: 315 sources (301
split plus 14 supplement), 1,206 variations kept, 59 duplicates, 6 exact
protected matches excluded, 0 off-split. `leakage_check` on the remaining
1,521 candidates dropped 58 (53 issue-39 test, 2 test, 2 validation, 1 sealed
held-out; 53 exact, 5 near-duplicate). Rendered **1,463 training examples**.

The Jetson skills scan on the training file took hours at first because
every rendered row repeats the same long system prompt (the tool table); f18
deduplicates training strings before scanning (identical result by
construction): clean, 104 evals, 6 seconds (ledger P57). *Merge:* `b869c5d`.

**Frozen training set:** 1,463 examples (262 sources = 248 original plus 14
supplement; 1,201 variations): propose 725, escalate 357, explain 381; all
16 operations covered (27-91 each). Hashes recorded under [Variations and
the re-split](#variations-and-the-re-split); copied to spark2 and verified
8 files byte-identical (the three splits, the training file,
`nvsh-train.jsonl`, the ground snapshot, and the two skills files).

A wrapper-script lesson from this stretch: `echo "$(date -Is) rc=$?"`
reports `rc=0` every time, because the command substitution itself resets
`$?` before the `echo` reads it — capture `rc=$?` on its own line first.

### 2026-09-24 05:30-07:20: t22/t23 first runs, three pipeline bugs, lapse l4

**Bug (P58):** `train-scorer` was training Track B on the raw
`splits/train.json` (still holding the 52 issue-39 test entries and a
duplicate of a test entry, and none of the variations). Fixed (f19,
`a9d7c04`): trains on `data/train-augmented.json` instead, refuses before
`assemble` has run, run name optional.

**Bug (P59):** `train.py --merge-only` still required `--train`, so the
documented Track B merge form could not be called as written. Fixed (f20,
`75d385b`): `train-scorer` now merges, writes the greedy generation config
and stages the result as `$REPO-scorer` with a revision, like `train` does;
`--merge-only` no longer needs `--train`.

**Bug (P60):** unsloth's `FastLanguageModel` returns a `Qwen3VLProcessor`
for Qwen3.5; its chat template rejected the plain-string content
`build_dataset.py` renders. Fixed (f21, `bb87232`): a `text_tokenizer()`
reaches the processor's inner `.tokenizer`. Live check: identical chat
template (7,755 characters) and token ids as `AutoTokenizer` on a real
training example (1,436 tokens).

**Track A recipes, `a1`/`a2`:** 3 epochs, lr 2e-4, rank 16/alpha 32, batch
8, seed 46; 549 steps, about 26-31 minutes each on spark; training loss
about 0.001 at the end (`train_loss` 0.091). `a1` = `--targets attn-mlp`,
`a2` = `--targets attn-mlp-gdn`.

**Track B `b1` on spark2** (all-linear LoRA, same epochs/lr/rank/batch/seed):
26 minutes, trainer's own validation 60 of 66 (90.9%), mean confidence
0.957; spark2 kept about 25 GB available throughout; model-gear (the serving
containers) untouched.

**Lapse l4:** `a1`/`a2` were first merged into checkpoints bit-identical to
the base model — unsloth trained the vision-language class (adapter keys
under `model.language_model.*`), the merge loaded the text-only
`AutoModelForCausalLM`, PEFT matched no key and only warned — and **both
were measured on validation (0 of 32, same as stock) before anyone checked
whether the merged weights had actually changed.** f23 first merged into the
VL class instead: every adapter tensor loaded, but the VL save wrote doubled
key prefixes (`model.language_model.language_model.*`,
`model.language_model.visual.*`) that vLLM cannot load ("There is no module
or parameter named 'language_model' in Qwen3_5Model"). f24 (`c6dd8de`,
merged `a7c81f8`) mapped the VL adapter's keys onto the text-only model,
replaced unsloth's `target_modules` regex (which misses `linear_attn` there)
with the adapted module names, and refuses the merge unless every adapter
tensor loads and a merged weight actually changes. `a1` merges 192 tensors,
`a2` 372. The tuned checkpoints are therefore text-only
(`Qwen3_5ForCausalLM`, named exactly like the base checkpoint's language
model) while stock stays `Qwen3_5ForConditionalGeneration` (ledger P62).

**Track B measurement (f22, merge `b4eeed6`):** served-scorer runs were
exiting 2 with every metric "not measured" and no reason — the repo's `uv`
environment has no `transformers`, the tokenizer was loaded from the served
name, and a scorer report had no start-up row. Fixed: `--tokenizer <model
dir>`, the training venv's site-packages on `PYTHONPATH` for `--scorer`
runs, every run failure printed to stderr (ledger P61).

**d15 (Track B calibration via the exact in-process scorer):** the served
scorer returned no complete label distribution on any validation entry (0 of
66 — the fine-tuned scorer leaves the other letters' logprobs outside
vLLM's top 22, risk r8, and the harness refuses to renormalise a partial
top-k). Decisions and latency are taken from the served run; both agreed
with the in-process run entry for entry on `b1`.

**vLLM start-up failures (P63):** twice, starting the pinned vLLM for a
measurement failed with only "Engine core initialization failed" visible in
the helper's last 40 log lines, while Track A trained on the same GPU;
retried on a quiet GPU, it worked. f25 (`6f65887`, merged `45c5ae8`):
`serve_for_measure.sh wait <port> [<full log>]` now keeps the server's full
log on a failed start-up; the pipeline passes
`$WORK/measure/<label>.serve.log`.

**A flaky test, identified:**
`tests/test_setup_timing.py::test_setup_does_not_meaningfully_slow_down_prompt_startup`
fails under machine load (a training run on the GPU); it passes reliably
alone. This is consistent with the earlier unidentified intermittent
failures after the f10 merge.

**Validation results (2K context, 66 entries):**

| Model | Right proposals | Abstain recall | Abstention precision (strict) | FP tool calls | Wrong mutating | Explain | Warm latency |
|---|---|---|---|---|---|---|---|
| Stock (generative) | 0/32 | 10/16 | — | — | 0 | 7/18 | 678 ms |
| Stock (exact scorer) | 21/32 | 1/16 | 100% | 31/34 | 0 | — | 81 ms; ECE 0.164, Brier 0.765 |
| `a1` | 32/32 | 12/16 (75%) | 100% | 3/34 | 2 | 18/18 | 420 ms |
| `a2` | 32/32 | 13/16 (81%) | 100% | 2/34 | 1 | 18/18 | 419 ms |
| `a3` (chosen, t22) | 31/32 | 14/16 (87.5%) | 100% | 1/34 | **0** | 18/18 | 400 ms |
| `a3` at 4K (informational; final stays at 2K) | 29/32 | 14/16 (87.5%) | 100% | — | **0** | 18/18 | 440-653 ms |
| `a4` | 32/32 | 12/16 (75%) | 100% | 3/34 | 1 | 17/18 | 476 ms |
| `b1` (served / exact — **chosen, t23**) | 28/32 (4 not grounded) | 12/16 (75%) | 92.3% | 0/34 | 0 | — | 76 ms served; exact ECE 0.097, Brier 0.162 |
| `b2` (exact, in-process only — served run lost its server mid-run) | 30/32 (7 invalid, not grounded) | 7/16 (43.8%) | 100% | 2/34 | 1 | — | 164 ms in-process; ECE 0.106, Brier 0.217 |
| `b3` (trainer's own validation only: 52/66, confidence 0.83 — not harness-measured) | — | — | — | — | — | — | — |
| `b4` (served / exact — better calibration, not chosen) | 28/32 (4 not grounded) | 12/16 (75%) | 85.7% | 0/34 | 0 | — | 24 ms served; exact ECE 0.072, Brier 0.132 |

`b2` is worse than `b1` on every axis (abstain recall, ECE, Brier, trainer
validation and loss): 5 epochs overfits Track B; `b3`'s trainer-side numbers
show the opposite failure, undertraining at 2 epochs. `b4` ties `b1` on
decisions and wrong mutating but loses on abstention precision by one
entry (85.7% vs 92.3%), so **Track B is `b1`** (t23 decision) — reported
plainly: the pre-registered rule was not moved after seeing `b4`'s better
calibration, which is recorded as a separate finding (lower lr, same
decisions, better calibration).

`a3` is the only Track A run with 0 wrong mutating proposals and is chosen
(t22): rank 32 (`a4`) did not help over rank 16, but 5 epochs (`a3`) did
over 3 (`a2`) — one more piece of evidence that epoch count, not adapter
capacity, was the limit here. `a3` was also checked at 4K and stayed at 0
wrong mutating with close to the same other numbers; the final run still
uses 2K, per plan. See [Choosing a configuration on
validation](#choosing-a-configuration-on-validation-track-a-and-track-b)
for the full comparison table and the reasoning between runs.

r9 on validation: the GDN targets (`a2`) beat `attn-mlp` (`a1`). `a2`'s
errors: "Set the power mode" (no mode given) proposed `power_set
max_performance` — its one wrong mutating proposal, an invented argument;
"Is the service running?" proposed `docker.service`, also invented; "Explain
why nginx returns 502" (should escalate) was explained instead; "switch to
balanced mode" and "nvpmodel low power" were escalated instead of
`power_set`. bf16 latency (about 420 ms) misses the c36 250 ms bar, as the
plan expected (quantization is t25's job).

At about 07:30, `a3` (`a2` + 5 epochs) and `a4` (`a2` + rank 32/alpha 64)
were training on spark, and `b2` (`b1` + 5 epochs) on spark2. Selection stays
validation-only: no wrong mutating proposals first, then abstention, then
right proposals. (`a3` and `a4`'s own results are in the [~08:40 entry
below](#2026-09-24-0840-a3-and-a4-done--track-a-picks-a3).)

### 2026-09-24 ~08:00: b2 done — more epochs overfit Track B

`b2` (`b1`'s recipe, 5 epochs instead of 3) finished on spark2: 43 minutes,
trainer's own validation 56 of 66 (84.8%, against `b1`'s 90.9%), loss 0.67
(against `b1`'s 0.29), mean confidence 0.947 (against `b1`'s 0.957).
spark2's checkout was still on the pre-f23 merge when `b2` trained; its
merge is text-only and every adapter key matched with 0 missing-key
warnings (the lapse-l4 merge bug was Track A-specific — Track B's merge
path was already correct), and the lead verified the merged weights
actually changed from the base (attention, GDN and MLP weights all
differed) before measuring, rather than repeat lapse l4's mistake of
measuring an unverified merge. spark2 was re-synced to this guide's current
commit right after.

`b2` validation, exact in-process scorer: 30 of 32 right proposals, abstain
recall 7 of 16 (43.8%), precision 100%, false-positive tool calls 2 of 34,
wrong mutating 1 (1 of 1 mutating expectations, 0 of the rest), invalid 7
(not grounded), ECE 0.106, Brier 0.217, warm 164 ms in-process. The served
run hit 1 `tier_error`: the server became unreachable mid-run while Track
A's `a4` trained on the same GPU, and the tier-error gate (P49) refused to
write a results page at all, exactly as designed, rather than score a
partial run. `b2` is worse than `b1` on every axis measured — more epochs
overfit Track B.

Track B selection stays `b1`. Next: `b3` = `b1` with 2 epochs, `b4` = `b1`
with lr 1e-4 (3 epochs), sequentially on spark2.

**Lesson for the tutorial:** measure while nothing else trains on the same
GPU when possible. A served run under load can lose its server mid-run; the
tier-error gate is what catches that rather than silently writing a
partial-run results page.

### 2026-09-24 ~08:15: a3's measurement loses its server too — root cause found

`a3` (`a2`'s recipe, 5 epochs) trained fine on spark at 08:03: 549 steps,
372 adapter tensors merged and verified by the lapse-l4 fix (revision
`d0303706...`). Its own validation measurement then lost its vLLM server
mid-run — 1 `tier_error`, the gate refused to write a results page — while
`a4` trained on spark's GPU at the same time. Same failure shape as `b2`'s
served run and the two earlier "Engine core initialization failed"
start-ups (P63).

The lead read spark's kernel log and found the cause at 08:10:36: `NVRM:
... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`. During
training, spark's *free* memory sinks to about 2.5 GiB while *available*
stays about 40 GiB — the difference is page cache built up from reading
weights and data. On GB10's unified memory a GPU allocation fails outright
instead of evicting that page cache, so a vLLM server started (or already
running) beside a training job can die (ledger P64). The training memory
watchdog was unaffected: it watches available memory, not free, and
available stayed high throughout.

**Rule adopted:** measure on spark only when no training run holds spark's
GPU. `a3` and `a4` are both measured after `a4` finishes, not concurrently
with it. Idea forward, not yet tried: dropping the page cache before a
measurement (needs root: `sync; echo 3 > /proc/sys/vm/drop_caches`), or a
cgroup limit on the trainer's page cache, either of which might allow
measuring and training to overlap safely.

### 2026-09-24 ~08:30: why spark was this tight — the senses lobe comes down

spark was not just running the training job: it also runs the operator's
own model-gear "lobes" deployment (docker compose project `lobes`) —
gateway, stt, realtime, bluetts — and a `model-gear-vllm-multimodal`
container serving Gemma-4-26B-A4B-NVFP4, the "senses" teacher (reviewer A
in t19's augmentation), which alone held about 33.6 GB of GPU memory in its
vLLM `EngineCore`. Training plus a measurement server on top of that is
what pushed free memory down to about 2.5 GiB and tripped `NV_ERR_NO_MEMORY`
(P64).

The operator: "We shouldn't have OOM on a 128Gb machine. We can take down
models as needed," and "I'd rather be cautious if we don't use the models
now." The `lobes` CLI's `stop`/`fleet down` take the whole spark deployment
down, gateway included, with no per-lobe stop; being cautious, the lead
instead stopped only the one container no longer needed now that t19 is
done: `docker stop model-gear-vllm-multimodal` (nothing deleted; restore
with `docker start model-gear-vllm-multimodal` or `lobes serve --apply`).
Gateway, stt, realtime and bluetts stayed up. Memory on spark, with `a4`
still training throughout: used 68 → 31 GB, free 15 → 51 GB, available
53 → 89 GB.

**Tutorial lesson:** on a GB10 shared with serving lobes, check
`nvidia-smi --query-compute-apps` and `docker stats` for what already holds
GPU memory before training and measuring on the same box, and stop an
unused lobe (with the operator's sign-off) rather than letting jobs overlap
into an OOM.

### 2026-09-24 ~08:40: a3 and a4 done — Track A picks a3

`a3` (`a2`'s recipe, 5 epochs instead of 3: 915 steps, about 46 minutes) and
`a4` (`a2`'s recipe, rank 32/alpha 64 instead of 16/32, 3 epochs: about 31
minutes) both finished on spark and merged cleanly through the verified
merge (372 adapter tensors each; `a3`'s revision `d0303706...`). Both were
measured on validation (2K) once the GPU was quiet (no training job holding
it, per P64) and once unused serving models on the training and measurement
machines were stopped with the operator's OK to free GPU memory (about
37 GB freed on the Track A machine, about 56 GB on the Track B machine;
restored after the run — which serving model runs where is not relevant to
reproducing this run).

- `a3`: 31 of 32 right proposals, abstain recall 14 of 16 (87.5%),
  precision 100%, false-positive tool calls 1 of 34, **0 wrong mutating**,
  explain 18 of 18, warm 400 ms.
- `a4`: 32 of 32 right proposals, abstain recall 12 of 16 (75%), precision
  100%, false-positive tool calls 3 of 34, wrong mutating 1, explain 17 of
  18, warm 476 ms.

`a3`'s remaining errors: "docker status" proposed `container_list` instead
of the expected `service_status docker.service`; "Can you switch to
balanced mode?" and "nvpmodel low power" were escalated instead of the
expected `power_set`; "Explain why nginx returns 502" (expected escalate)
was explained instead; "Is the service running?" (expected escalate)
proposed `service_status docker.service`, a false-positive tool call.

**t22 decision: Track A = `a3`.** It is the only run with 0 wrong mutating
proposals, and meets c33 and c34 on validation. Rank 32 (`a4`) did not help
over rank 16; more epochs (`a3` over `a2`) did — evidence that epoch count,
not adapter capacity, was the limit for Track A. `a3`'s bf16 latency (about
400 ms) still misses c36's 250 ms bar, so t25's quantization is required
regardless. `a3`'s recipe (`--epochs 5 --lr 2e-4 --rank 16 --alpha 32
--batch 8 --seed 46 --targets attn-mlp-gdn`) and Track B's current best,
`b1` (`--epochs 3 --lr 2e-4 --rank 16 --alpha 32 --batch 8 --seed 46`), are
now committed as the default `TRAIN_ARGS`/`TRAIN_SCORER_ARGS` in
`pipeline-qwen.env.example`.

**Remaining plan, in order:**

1. **t22 finish:** measure `a3` at 4K on validation (`export
   MEASURE_CTX=4096; $P --env qwen.env measure-val a3`) and, if useful,
   Track A's exact calibration on validation
   (`track_a_calibration.py`, d6).
2. **t23 finish:** measure `b4` (served and exact,
   `$P --env spark2.env measure-val b4 --scorer served|in-process`) and pick
   Track B's final recipe.
3. **t24, the single final run:** stock, `a3` and the chosen Track B
   checkpoint, each measured exactly once, on the clean test side, the
   sealed held-out set and the missing-candidate slice, at 2K, on a quiet
   machine (`measure-final <name>`; Track B also with `--scorer in-process`
   for calibration, d15); Track A's exact calibration on the final side
   (`track_a_calibration.py --final`); the Jetson skills eval once per tuned
   checkpoint at `MEASURE_CTX=8192`, judged against the d12 margin (stock:
   42 of 104 overall, so the floor is about 37 of 104 overall and about 25
   of 70 not-named).
4. **t25:** quantize the chosen checkpoint(s) to `Q4_K_M` and AWQ; heal only
   if a build loses more than 3 points of right proposals or adds a new
   wrong-mutating id (c42, c43).
5. **t26:** the edge check on AGX Orin.
6. **t27:** a private upload, only after asking the operator.
7. **t28:** the report and this guide's final pass.
8. **t29:** `/validate-delivery`, `/summarize-delivery`, a version bump, and
   the PR ("part of #46").

### 2026-09-24 ~09:00: t22 and t23 both close out

**t23 (Track B) closes: `b1` chosen.** `b4` (`b1` with lr 1e-4, 3 epochs)
finished: 1,522 s to train, 2.83 GB peak memory, revision `2f1ed0f6...`.
Served: 28 of 32 right proposals, abstain recall 12 of 16, precision 85.7%,
false-positive tool calls 0 of 34, 0 wrong mutating, 4 not grounded, warm
24 ms. Exact in-process scoring agrees with the served run's decisions and
gives ECE 0.072, Brier 0.132 (against `b1`'s exact ECE 0.097, Brier 0.162,
precision 92.3%). Against the rule fixed before any run (0 wrong mutating,
then abstention, then right proposals), `b1` and `b4` tie on wrong mutating
and are close on right proposals, but `b1` leads on abstention precision by
one entry — `b1` wins. Reported plainly: the rule was **not** moved after
seeing `b4`'s better calibration; that is a separate finding (a lower
learning rate gave better calibration at essentially the same decisions),
not a reason to switch checkpoints.

**t22 (Track A) gets an optional 4K check.** `a3` was re-measured at 4K
context (`MEASURE_GPU_FRACTION=0.12`, per P65, to avoid the Mamba-cache
start-up failure): 29 of 32 right proposals, abstain recall 14 of 16, 0
wrong mutating, explain 18 of 18, warm 440-653 ms — close to `a3`'s 2K
numbers (31/32, 14/16, 0 wrong mutating) and still 0 wrong mutating at
both. The final run stays at 2K, per the original plan; this was a check
that `a3` holds up at a longer context, not a change of plan.

**Ledger.** P66 (lapse l5): a Track B scorer measured without `--scorer`
scores as a broken generative model (0 of 32, about 100 generated tokens
per decision) instead of refusing outright; `measure-val`/`measure-final`
now refuse a scorer run with a hint if no `--scorer` mode is given
(`d39c5e3`). Two more troubleshooting entries recorded without new ledger
ids: every prediction in a run coming back `tier_error` right after the
server reported ready, caused by a host network reconfiguration overlapping
the run rather than a GPU-memory cause (re-run once the network settles);
and a practical tip for moving a merged checkpoint between the two training
machines quickly over a direct, non-default point-to-point link when one is
available (see [step 11](#11-train-track-b-on-spark2-b1-b4-done-b1-chosen)).

**t24 is next:** stock, `a3` and `scorer-b1`, each measured exactly once —
served and exact for `scorer-b1` — on the test side, the sealed held-out
set and the missing-candidate slice, at 2K; Jetson skills at
`MEASURE_CTX=8192` for `a3` and for `b1` if the skills harness can attach
to a scorer by then; `track_a_calibration.py --final` for Track A's exact
calibration on the final side.

### 2026-09-24 ~09:45: t24 done — neither checkpoint clears every bar yet

The single final run finished: stock, `a3` and `scorer-b1`, each measured
exactly once at 2K on a quiet machine, on the test side (64 entries: 32
operation, 15 escalate, 17 explain), the sealed held-out set (69 entries:
41 operation, 13 escalate, 15 explain) and the missing-candidate slice (the
test side's 32 operation entries with the gold operation removed).
Committed as `docs/benchmarks/2026-09-24-lfm-{final,heldout}-*.md` and
`2026-09-24-skills-q46-{stock,a3}.md` (commit `db55ce4`); full numbers,
the pass/fail table against c33-c36 and the findings are written up in
[Final results (t24)](#final-results-t24), which is now the source of
truth for this run's headline numbers — this run-log entry only summarizes.

In short: `a3` (Track A) passes c33 (100% right proposals, stock+94pp) and
abstention precision (100%), but fails abstention recall (73.3%), the
false-positive tool call bar (12.5%), 0-wrong-mutating (2 on test, though
0 on validation) and c36's latency at bf16 (442 ms; quantization is next).
`scorer-b1` (Track B) passes c33 (84.4%, stock+78pp), abstention precision
(100%), 0-wrong-mutating and c36's latency (23 ms), but fails abstention
recall (73.3%, same as `a3`) and the false-positive bar (6.3%), and misses
c35's ECE bar (0.132 against a 0.10 ceiling) despite beating stock's Brier
score by a wide margin (0.268 vs 0.834). Container memory (c36) is not yet
measured for either, since both t24 runs were attach-mode against an
already-running server; that lands in t25 alongside the quantized builds.

The missing-candidate slice was the weakest area for both: with the gold
operation removed, both mostly picked a near-candidate operation instead of
escalating (`a3` 24 of 32 false-positive tool calls; `scorer-b1` 18 of 32).
Jetson skills passed the d12 regression guard (44 of 104 against a floor of
37, stock's own 42; not-named 34 of 70 against a floor of 23, stock's 28),
with the gain concentrated in not-named prompts.

Before the run, the pipeline gained a `measure-heldout` stage and the
`-missing-candidate`/`-exact` label suffixes, with an arguments allowlist
(deviation d16, commits `6f9ab90` and `ff01844`). The skills step served at
`MEASURE_GPU_FRACTION=0.12` (P65). Stock's exact-scorer baseline (used only
as c35's comparison point, d15) ran after the 13 planned final-measurement
steps rather than before them — running last does not make it any less a
single, final, no-retry measurement.

**Next: t25**, quantizing `a3` and `scorer-b1` to `Q4_K_M` and AWQ,
measured on the test side against the same bars, healing only if a build
loses more than 3 points of right proposals or adds a new wrong-mutating id
(c42, c43) — and where container memory finally gets measured.
