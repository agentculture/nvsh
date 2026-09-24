# Tool-Jev calibration cycle (issue 53)

This is an improvement cycle on Track B, the Jev-style candidate scorer from
issue 46 (`docs/qwen-tool-jev-finetune.md`, PR #52). It measures how much
scorer-b1's decision depends on the order the candidates are listed in and
the letters they are assigned, adds a separate uncertainty-abstention gate
next to semantic escalate, retrains with per-example randomized candidate
labels on a grown data set, and reports calibration for the artifact nvsh
would actually deploy (the served `Q4_K_M` build), not only bf16. **This is
an experiment, like issue 46: no file under `nvsh/` changes in this cycle's
PRs, and nvsh's defaults do not change.** Runtime wiring of any gate this
cycle produces is issue #54, not this cycle.

This page is a long-lived guide and ledger, owned only by the documentation
subagent for plan task t10. It is updated in the same step as every run,
obstacle or fix, in the style of `docs/qwen-tool-jev-finetune.md`.

**Status: in progress, 2026-09-25.** Wave 1 of the plan has begun. t9, t10
and t11 are merged; t1, t3, t4 and t5 are in progress. See [Where the run
stands](#where-the-run-stands) for the live picture, and the [pre-registered
decision rule](#the-pre-registered-checkpoint-decision-rule-t9) for how the
chosen checkpoint will be picked.

## What this cycle is

Issue #53 follows issue #46 / PR #52. PR #52 shipped `scorer-b1`, a Track B
candidate scorer, but left four gaps this cycle closes:

1. **Permutation dependence is unmeasured.** `scorer-b1` picks a candidate by
   bare argmax over labels fixed to the operations table's own order.
   Nobody has measured whether it evaluates the candidates it is given or
   has partly memorised which letter tends to be right.
2. **No calibration exists for the deployed artifact.** `scorer-b1`'s exact
   (in-process, bf16) ECE is 0.132 against the issue 46 bar of 0.10, and the
   quantized `Q4_K_M` build nvsh would actually serve has no calibration
   number at all, because served scoring returns 0 of 64 complete label
   distributions on that build.
3. **Escalation is one undifferentiated label.** There is no separate
   "I don't know" outcome distinct from a semantic decision to hand off.
4. **Track A and Track B get described as equally Jev-like**, though only
   Track B is a genuine candidate-scoring model; Track A is a generative
   tool router whose distribution is reconstructed offline by
   `track_a_calibration.py`.

Track B is the sole subject of this cycle (Track A gets only the
documentation relabel above; Track A improvements go to issue #56). The
cycle is experiment-only, exactly like issue 46: it delivers checkpoints,
measurements, a comparison report and a recommendation, in
`scripts/lfm-finetune/`, never a change under `nvsh/`.

## Where this starts (scorer-b1, PR #52)

Every number below is sourced from `docs/benchmarks/2026-09-24-*scorer-b1*`
or the issue-46 delivery record, per the spec's honesty condition.

- **Test side** (32 operation / 15 escalate / 17 explain, 64 total): 27 of
  32 right proposals (84.4%), abstention recall 11 of 15, precision 100%,
  false-positive tool calls 2 of 32 (6.3%), 0 wrong mutating.
- **Calibration (exact, in-process, bf16):** ECE 0.132 (bar 0.10, fails),
  Brier 0.268 (passes against stock's 0.834).
- **Missing-candidate slice** (32 operation entries, gold operation removed
  from the candidates, expected answer escalate): abstention recall only 7
  of 32, false-positive tool calls 18 of 32, 0 wrong mutating.
- **Held-out set** (41 operation / 13 escalate / 15 explain, 69 total): 27
  of 41 right proposals, abstention recall 11 of 13, precision 68.8%.
- **Served readout is broken.** Every served run measured in issue 46 (bf16
  vLLM, AWQ vLLM, `Q4_K_M` llama-server on the training machine and on an
  AGX Orin) returned 0 of 64 complete label distributions: label tokens for
  the model's other candidates fall outside the top-k logprobs the server
  returns. No ECE/Brier number exists today beyond the bf16 in-process
  route.
- **Labels are fixed by table order**, and candidate descriptions are
  hard-wired to the operations table; there is no seam today to shuffle
  order, permute letters, offer a subset, or paraphrase a description.
- Picking `scorer-b1` over the better-calibrated `b4` (lower learning rate:
  exact ECE 0.072 vs 0.097, Brier 0.132 vs 0.162) rested on a rule decided
  during that run, not one pre-registered before training; this cycle
  writes the rule first (t9), before any training command runs.

## The bars

Fixed in the spec before any run this cycle
(`docs/specs/2026-09-24-tool-jev-calibration-cycle-issue-53.md`):

- **Permutation robustness:** shuffling candidate order and permuting the
  operation-to-letter labels changes the semantic (operation-level) choice
  in at most 2.3% of cases, point estimate, with the 95% CI reported —
  OpenJev's own published figure for its model (OpenJev's *base* model was
  18.5%). Measured with at least 10 permutations per entry on a roughly
  150-entry fresh test side (about 1,500 trials). Canonicalizing the
  runtime order to dodge the bar is an explicit non-solution: it does not
  count toward this bar, and using it needs its own operator decision.
- **Calibration:** ECE at most 0.10, with a bootstrap confidence interval,
  on the deployed `Q4_K_M` build, measured through the served path (not
  bf16 alone), after a temperature fitted on validation only.
- **Zero wrong mutating proposals** at the chosen gate thresholds, on the
  fresh test side and the sealed held-out.
- **Missing-candidate escalation** (semantic escalate plus the new
  uncertainty-abstain outcome combined) at least 80% on the missing-
  candidate slice (7 of 32, 21.9%, today).

OpenJev publishes no calibration number (it ships a fixed readout
temperature, `READOUT_T` = 0.85) and no abstention mechanism, so the
calibration and abstention bars are nvsh's own, carried over from issue 46's
c34/c35.

## Code map

Everything lives in `scripts/lfm-finetune/`, exactly as in issue 46; nothing
here is imported by the `nvsh` package, and no file under `nvsh/` changes in
this cycle's PRs. Marked *(planned)* until the task's PR merges.

| File | What it does | Task |
|---|---|---|
| `scorer.py` | **Merged** (`0b48c79`). One shared label-probability definition, `distribution()`: a candidate's mass is the sum of next-token probabilities over every vocabulary token whose whitespace-stripped text equals its label (3 variants per letter on Qwen3.5-0.8B: `'A'`, `' A'`, `'\tA'`), normalised over offered candidates. New `label_variant_ids()` scans the vocabulary for those variants (~0.4 s over ~248k tokens); `TransformersScorer` now reads all variant ids and goes through `distribution()`, the same function a served model uses; new `label_logits_from_vocab()` is a differentiable log-sum-exp over variant ids, for training. `READOUT_TOP = 5000` is what `score()` requests from a server. Incomplete results are still never renormalised. The permutation seam (order, letter map, subsets, description overrides) is *(still planned)*, t2. | t1, t2 |
| `serve_for_measure.sh`, `pipeline.env.example`, `pipeline-qwen.env.example` | **Merged.** `MEASURE_MAX_LOGPROBS` now defaults to 5000 (was 22) in the script and both pipeline env examples. `llama-server` has no max-logprobs flag, so nothing there caps the readout. | t3 |
| `calibration_fit.py` | *(planned, new)* Seeded validation fit/selection folds; fits and applies temperature scaling (then vector scaling) by minimising NLL on the fit fold only; refuses test/held-out as fit input. | t4 |
| `metrics.py` | *(planned)* Per-slice ECE/Brier/reliability bins (read-only vs mutating, candidate count, missing-candidate, confidence bucket); every rate and ECE carries n and a bootstrap 95% CI; `abstain_uncertain` counted as its own outcome, separate from semantic escalate, both rolling into the escalation bar. | t5 |
| `gate.py` | *(planned, new)* The uncertainty-abstention gate: decides propose / explain / escalate / `abstain_uncertain` from `p_escalate`, a `p_top1` floor, the top1-top2 margin and normalized entropy; thresholds keyed only by `Operation.read_only`, never an operation name. | t6 |
| `sweep_gate.py` | *(planned, new)* Offline threshold sweep over stored `predictions.jsonl` files, no GPU needed; refuses test/held-out without `--final`. | t6 |
| `permutation_probe.py` | *(planned, new)* Runs the permutation seam at least 10 times per entry per perturbation kind (order, letters, subset, paraphrase) and reports the operation-level answer-change rate with n and a bootstrap 95% CI; never canonicalises runtime order. | t7 |
| `measure.py` | *(planned)* Wires in the new readout, applies a fitted calibration file, renders per-slice reliability tables into the committed benchmark markdown, records complete/incomplete served counts, and fixes issue #57 (a failed startup no longer blocks a clean re-run under the same label). | t8 |
| `docs/tool-jev-calibration-rule.md` | **Merged** (commit `b1c6cb6`). The pre-registered checkpoint-selection rule — selection fold, ordered criteria, tie-breaks, and the calibration-aware-stage trigger — confirmed by the operator before any training. See [below](#the-pre-registered-checkpoint-decision-rule-t9). | t9 |
| `docs/qwen-tool-jev-calibration.md` | This file. | t10 |
| `docs/qwen-tool-jev-finetune.md`, `docs/benchmarks/2026-09-24-qwen-tool-jev-comparison.md`, `release_bundle.py` (model-card text) | **Merged.** Track A is now described as a specialized generative tool router and Track B as the Jev-style candidate scorer; no text calls them equally Jev-like. | t11 |
| `split.py` | *(planned)* Can write a versioned corpus v2 (header records version, seed, source hashes, and the calibration fit/selection fold assignment) to a path outside `nvsh/`; validation and test sizes become parameters (validation >= 150, test ~150); never writes `nvsh/tiers/corpus`. | t12 |
| `draft_heldout.py` | Unchanged tool from issue 46, reused to draft the new sealed held-out set for this cycle (lead reads counts and hashes only). | t13 |
| `build_dataset.py`, `merge_variations.py` | *(planned)* Store each rendered example's offered candidates, order and letter map; generate deterministic missing-candidate / no-valid-option train examples from train entries only (`eval_slices.py`'s shape); an optional mode renders the corpus's 8 decline classes as distinct escalate-reason candidates, described from a scripts-side JSON, rolling up to escalate in gold labels. | t14 |
| `data/reasons.json`, `data/paraphrases.json` | *(planned, new)* Scripts-side descriptions for escalation-reason candidates and for the paraphrase permutation probe — never read from `nvsh/`. | t14 |
| `train_scorer.py` | *(planned)* Per-row label ids and cross-entropy target come from that row's own stored letter map and offered set (today one fixed `label_ids` list serves the whole run); an optional `--calibration-loss` (label smoothing and/or a Brier term beside CE), off by default. | t16 |
| `quantize.py` | Reused unchanged tooling from issue 46 to build the chosen checkpoint's `Q4_K_M` (and optionally AWQ). | t19 |

## Corpus v2 design

- **Lives scripts-side and in the private data repo only.**
  `nvsh/tiers/corpus/dev.json` and `held-out.json` are not modified by this
  cycle — the corpus these files ship in the wheel and feed nvsh's own
  tiers bench, and the Needle, LFM and Laya work, so touching them would be
  a runtime change and would move every other experiment's baseline.
  Runtime promotion of a v2 corpus, if it happens, is issue #54's job, not
  this cycle's.
- **A versioned re-split**, not an edit of the v1 split: `split.py` gains
  the ability to write a header naming its version, seed and source
  hashes, with validation and test sizes as parameters — validation grown
  to at least 150 entries (up from 66) and test to roughly 150, both
  stratified by answer kind, so a single flipped entry no longer swings a
  bar by several points (the v1 validation side flips 6.7 percentage
  points of abstention recall per entry). The fit/selection fold split for
  calibration is recorded in the same header.
- **New sources are teacher-drafted**, through the same all-Apache pipeline
  as issue 46 (`augment.py`'s generator, corrector and reviewer B) and
  checked with `leakage_check.py` against every v2 evaluation side. Roughly
  250+ new reviewed entries are expected beyond today's 431-entry corpus.
- **New per-example content the v1 corpus never carried:** offered
  candidate sets and letter maps stored at dataset-build time (not invented
  at collate time), deterministic missing-candidate / no-valid-option
  training examples (the frozen v1 training set has zero of these), and
  optionally the corpus's 8 existing decline classes
  (`outside_table`, repair, diagnosis, `missing_argument`, `not_a_request`,
  `multi_step`, injection, `over_time`) as distinguishable escalate-reason
  targets — kept only if an ablation shows they help validation.
- **The comparison doc's other five data recommendations** are folded in
  before the freeze: missing-argument escalation, diagnosis-vs-explanation
  pairs, explicit-mode `power_set` positives, disambiguation pairs and hard
  negatives.
- **A fresh sealed held-out set**, teacher-drafted with `draft_heldout.py`
  exactly as in issue 46: the lead reads counts and hashes only, never the
  text, and it is stored only in the private data repo, never committed
  publicly (the issue-38/issue-46 held-out was committed publicly; this
  one is not).
- **The issue-46 test side is spent** and is not used for any claim in this
  cycle.

## Pre-challenge probe (before any code or training)

Ledger entry **P1**, run 2026-09-25, ahead of wave 1: a scratch CPU probe
against the served `scorer-b1` build (no repository or GPU changes), which
shaped several of this cycle's requirements.

- **Symptom:** every served scoring run in issue 46 — bf16 vLLM, AWQ vLLM,
  `Q4_K_M` llama-server, on the training machine and on an AGX Orin —
  returned 0 of 64 complete label distributions, so no calibration number
  exists for any served or quantized build.
- **Cause (found by the probe):** the server was asked for far too few
  logprobs. With the served top capped at 22 (18 labels + a small margin),
  10-15 of 18 labels were missing on the probed prompts. Raising the
  requested top to 400 still missed one label on 3 of 6 prompts; only top
  5000 was complete on 6 of 6, in about 0.4 s per request. A `logit_bias`
  approach was also tried and rejected: it left the server's *returned*
  probabilities unchanged, so it does not solve the completeness problem.
- **Fix (adopted into the spec/plan):** raise both the scorer's requested
  top-k (`READOUT_TOP`, replacing `TOP_MARGIN`) and the server's own cap
  (`serve_for_measure.sh`'s logprob/`n_probs` setting) to at least 5000 —
  t1 and t3.
- **A second, distinct gap the same probe found:** in-process scoring reads
  one token id per label, while served scoring sums multiple token variants
  for the same label (for example `"A"` and `" A"`). The bf16 GGUF's served
  readout matched the in-process exact distribution within 1e-4 on 5 of 6
  probed prompts, but on one prompt (`dev-f02`) it differed by 0.137 —
  **a definition gap, not a quantization artifact**, since both runs used
  the same bf16 weights. **Fix:** pick one shared label-probability
  definition (the served variant-sum definition) used by training,
  in-process scoring and served scoring alike — t1.
- **A third finding, kept as a design constraint rather than a bug:** the
  quantized `Q4_K_M` build's distribution legitimately differs from bf16 by
  up to 0.096 (on `dev-e08`, where the bf16 GGUF itself matched the
  in-process exact distribution to 1e-4). **Consequence:** temperature and
  gate thresholds for the deployed build are fitted on `Q4_K_M`'s own
  validation distributions, never inherited from a bf16 fit — carried into
  t17 and t19.

## The pre-registered checkpoint decision rule (t9)

Committed as `docs/tool-jev-calibration-rule.md` (commit `b1c6cb6`),
confirmed by the operator before any training command of this cycle runs.
Any run outside it, or any rerun, is a recorded `/deviate`, not a silent
choice.

- **Judged on the selection fold** of corpus v2 validation only; the fit
  fold is used only to fit the temperature and gate thresholds for that
  candidate. The fresh test side and the sealed held-out are never
  consulted before the choice — they are measured once, after it (t20).
- **Four pre-registered candidates**, each 3 epochs (issue 46 found 3 best
  for Track B): `r1` (per-example randomized order/letters/subsets, corpus
  v2, one escalate label), `r2` (`r1` plus the 8 escalation reasons as
  distinct candidates — the c28 ablation), `r3` (`r1` at learning rate 1e-4,
  following `b4`'s calibration lead), and a conditional `r4` (below).
  `scorer-b1` is measured the same way as a reference point, but is not
  itself a candidate.
- **The rule, in order:** (1) hard filter — 0 wrong mutating actions on the
  selection fold and its missing-candidate slice; (2) hard filter — right
  proposals at least `scorer-b1`'s rate on the same fold minus 5 points;
  (3) lowest permutation answer-change rate, pooled across perturbation
  kinds (candidates within 1 point advance together); (4) lowest ECE after
  temperature (candidates within 0.01 advance together); (5) highest
  missing-candidate escalation rate; (6) ties broken by higher right-
  proposal rate, then the simpler recipe (`r1` before `r3` before `r2`).
  `r2`'s escalation reasons are kept only if `r2` wins under this rule.
- **`r4`, the conditional calibration-aware stage**, runs only if the
  chosen candidate's selection-fold ECE after temperature is still above
  0.10: the same recipe plus label smoothing 0.1 and a Brier term (weight
  0.5) beside cross-entropy. `r4` replaces the choice only if it wins under
  the same rule; otherwise the choice stands and the miss is recorded. When
  `r4` is not triggered, the selection-fold ECE that kept it off is
  recorded as evidence.

## Where the run stands

**Wave 1 in progress**, 2026-09-25. **Merged:** t1 (readout core, `0b48c79`),
t3 (served readout cap), t9 (pre-registered decision rule, `b1c6cb6`), t10
(this guide) and t11 (Track A/B documentation relabel). **In progress:** t4
(calibration-fit module) and t5 (per-slice metrics). No training run has
happened yet — t9's rule is committed and operator-confirmed ahead of the
cycle's first training command, as required. This section will be updated
at each step as the lead forwards findings.

Cumulative issue **#61** records every deviation, lapse and status update
for this cycle as it happens; this guide's ledger below is the narrative
version of the same events.

**Run condition.** For the duration of this cycle, the operator has allowed
raising or dropping teacher models on the shared model-serving lobes, and
switching a lobe to a different supported Qwen or Gemma model, as this run
needs (no machine names recorded here, consistent with this guide's
redaction rule).

## Ledger (symptom -> cause -> fix)

- **P1** — see [Pre-challenge probe](#pre-challenge-probe-before-any-code-or-training)
  above: served scoring returned 0/64 complete distributions (missing
  top-k); `logit_bias` did not fix it; in-process and served scoring used
  different label-probability definitions (1e-4 agreement on 5/6 probed
  prompts, 0.137 gap on `dev-f02`); `Q4_K_M` legitimately drifts from bf16
  by up to 0.096. Fixes: raise the requested/served top-k to >= 5000 (t1,
  t3); unify the label-probability definition (t1); fit `Q4_K_M`'s
  temperature and thresholds on its own validation distributions, not
  bf16's (t17, t19).
- **P2, worktree/branch naming.** **Symptom:** setting up this cycle's
  per-task worktrees under the plan's default naming (`agent/t1`, ...)
  collided. **Cause:** branches and worktrees named `agent/t1..` already
  existed on disk from earlier assign-to-workforce runs (issue 46's cycle
  used the same default pattern). **Fix:** this cycle's task branches and
  worktrees are namespaced by issue: `agent/i53-<task>` (for example this
  guide's own `agent/i53-t10`) and worktree directories `i53-<task>` (for
  example `i53-t10`), instead of the plan's bare `agent/<task>` default.
- **P3, plan housekeeping — t22's date placeholder.** **Symptom:** the
  exported plan failed markdownlint (MD033, bare/invalid inline HTML-like
  token) on task t22's first acceptance criterion. **Cause:** the
  acceptance line used a literal, unbackticked `<date>` placeholder for the
  benchmark report's filename pattern
  (`docs/benchmarks/<date>-tool-jev-calibration-cycle.md`), which
  markdownlint parses as an HTML tag. **Fix:** the placeholder was
  backticked (`` `<date>` ``); because the plan's content changed, t22
  reverted to proposed and the operator re-confirmed it before it counted
  as part of the approved split.
- **d1, t1's scope trim at merge (operator-approved deviation).** t1's
  first acceptance criterion originally asked `train_scorer.py` to adopt
  the shared `distribution()` definition too; that adoption moves to t16,
  which owns `train_scorer.py`. At the same merge, the lead updated one
  t8-owned assertion in `tests/test_lfm_finetune_measure.py` to
  `READOUT_TOP` so the merge stayed green; `measure.py`'s own
  `--max-logprobs` preflight check still belongs to t8. Reason: keep task
  ownership boundaries clean without blocking a passing merge.
- **l1, an assumption not yet re-measured (operator-approved lapse).** The
  pre-challenge probe's `dev-f02` 0.137 definition gap (P1 above) is
  claimed closed on the strength of t1's code change alone; it has not yet
  been re-measured on `scorer-b1` itself. t17 (the baseline re-probe on
  corpus v2 validation) re-measures it.
- **l2, no CI control for the training-interpreter path (operator-approved
  lapse).** The torch/tokenizer-backed scorer tests (`label_variant_ids`,
  `TransformersScorer`, `label_logits_from_vocab`) run only under the
  training interpreter (44/44 passing there); the CI/uv environment skips
  them (35 pass, 9 skip) because torch is not installed there. There is no
  CI job that exercises this path; it depends on a human or the lead
  running it under the training interpreter before trusting it.

## Reproduce steps

*(Placeholder — filled in as each step of the plan lands and is confirmed
by the lead. Commands will not include any home-directory path,
`/home/...` path, hostname, IP address, or serving-model location.)*

1. Training environment setup — *(not yet run this cycle; see
   `docs/qwen-tool-jev-finetune.md`'s "Reproduce it" step 1 for the base
   pattern, unchanged).*
2. Base model and serving image — *(unchanged from issue 46; no new step
   recorded yet)*.
3. Corpus v2 split (t12) — *(not yet run)*.
4. Draft and seal the new held-out set (t13) — *(not yet run)*.
5. Baseline probe of `scorer-b1` on v2 validation (t17) — *(not yet run;
   the pre-challenge probe above was a scratch, out-of-repository probe,
   not this step)*.
6. Dataset build with per-example candidate sets (t14/t15) — *(not yet
   run)*.
7. Training runs and selection (t18) — *(not yet run)*.
8. Quantize and calibrate the deployed build (t19) — *(not yet run)*.
9. Final measurement, Spark and Orin (t20) — *(not yet run)*.
10. Private uploads (t21) — *(not yet run)*.

## Follow-up issues

Out-of-scope items from this cycle each get their own issue rather than a
silent non-goal:

- **#54** — runtime wiring of the gate and scorer into nvsh (the cycle
  stays experiment-only in `scripts/lfm-finetune`).
- **#55** — a generic Choice/Noul/Score composable-decision API.
- **#56** — Track A follow-up improvements (this cycle gives Track A only
  the documentation relabel, t11).
- **#57** — a `measure.py` bug where a failed startup leaves a results page
  that blocks a clean re-run under the same label; closed by t8.
- **#58**, **#59**, **#60** — linked from the cycle report (t22) per the
  plan; not yet detailed in anything forwarded to this guide.
