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

**Status: in progress, 2026-09-25.** Code waves 1-3 are merged; the t13
data is sealed (fresh evaluation pool, private held-out) and corpus v2 is
built; the t17 baseline on `scorer-b1` is running and t15's targeted
augmentation is generating. No training run has started. See the
[decision path](#decision-path) for every choice made so far and why,
[Where the run stands](#where-the-run-stands) for the live picture, and the
[pre-registered decision rule](#the-pre-registered-checkpoint-decision-rule-t9)
for how the chosen checkpoint will be picked.

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
| `scorer.py` | **Merged** (`0b48c79`, permutation seam `6ea1105`). One shared label-probability definition, `distribution()`: a candidate's mass is the sum of next-token probabilities over every vocabulary token whose whitespace-stripped text equals its label (3 variants per letter on Qwen3.5-0.8B: `'A'`, `' A'`, `'\tA'`), normalised over offered candidates. New `label_variant_ids()` scans the vocabulary for those variants (~0.4 s over ~248k tokens); `TransformersScorer` now reads all variant ids and goes through `distribution()`, the same function a served model uses; new `label_logits_from_vocab()` is a differentiable log-sum-exp over variant ids, for training. `READOUT_TOP` is what `score()` requests from a server — raised from 5000 to **20000** in the P13 rework below (still >= the c38 floor of 5000). Incomplete results are still never renormalised. **Permutation seam (t2):** a frozen `Permutation` dataclass (`order`, `labels`, with `to_json`/`from_json` round-trip); `permute(seed, pool, subset=, keep=)` draws a seeded order and letter permutation without replacement from `LABEL_ALPHABET`, with an optional subset that always keeps the gold candidate; `same_choice()` compares by operation name, never by letter; description overrides let a name outside the operations table (for example `escalate:repair`) supply its own text; `prompt_messages`/`score` take explicit `labels`/`order`/`descriptions`. The default (no permutation given) is pinned byte-identical to before, so `scorer-b1`'s own decisions stay reproducible. | t1, t2 |
| `serve_for_measure.sh`, `pipeline.env.example`, `pipeline-qwen.env.example`, `pipeline.sh`, `release_bundle.py` | **Merged.** `MEASURE_MAX_LOGPROBS` now defaults to **20000** (raised from an initial 5000, itself up from 22) in the script, both pipeline env examples, `pipeline.sh`'s own default and `release_bundle.py`'s serving instructions — all derived from `scorer.READOUT_TOP` rather than a separately hard-coded number (P5 fixed the derivation; P13 below raised the number itself). `llama-server` has no max-logprobs flag, so nothing there caps the readout. | t3 |
| `calibration_fit.py` | **Merged**, then reworked in the review-fix pass on `#61`. New, stdlib only. Three subcommands: `folds` (writes `{seed, source, fit_ids, selection_ids}`, disjoint and sorted, default 70/30 split); `fit` (fits a temperature `T` by golden-section search on log T minimising NLL on the fit fold, bounds `T` in `[1/20, 20]`, then a per-label vector by coordinate descent on the temperature-scaled rows); `apply` (applies temperature then vector, renormalising over each line's offered labels; null-candidate lines pass through and are counted). Refuses test/held-out inputs by file name and split header — predictions files carry no header, so the fit refusal is name-based only (a documented limit, not a full guarantee). Review fix: NLL is now computed in log space without clipping the gold probability, `escalate:<reason>` mass rolls up to `escalate` before NLL, and rows whose gold has no candidate are skipped and counted (`skipped_gold_absent`) instead of corrupting the fit. **P14 fix (merged):** `make_folds` gained an optional `group_of` so a whole source's stored variations fold together, never splitting across fit and selection. **t17 (merged):** `evaluate --predictions --params --folds [--fold selection]` reports ECE and Brier with bootstrap CIs for raw, temperature-only and temperature+vector on one fold, so the selection fold decides whether vector scaling is kept. | t4 |
| `metrics.py` | **Merged** (`6ea1105`). Per-slice results: read-only vs mutating (by the gold operation's `Operation.read_only`), `escalate_or_explain`, each carrying calibration, candidate-count and missing-candidate rate. Every rate and ECE/Brier carries n and a seeded percentile bootstrap 95% CI (1000 resamples, seed 0 default; precision uses a stratified bootstrap). New outcome `abstain_uncertain` is counted separately from semantic escalate but still counts as "escalated" for escalation bars. Escalation-reason labels use the form `escalate:<reason>` and roll up to plain `escalate` everywhere. `reliability_markdown()` renders the bin tables per slice. | t5 |
| `gate.py` | **Merged, new.** `decide(distribution, offered, thresholds)` -> `propose` / `explain` / `escalate` / `abstain_uncertain`: escalate if the argmax or the rolled-up escalate mass clears its threshold; else explain if top1 is explain; else a per-`Operation.read_only` threshold set (a `p_top1` floor, the top1-top2 margin, and normalized entropy) decides `abstain_uncertain`; any threshold set to `None` disables that check. Thresholds are keyed only by `Operation.read_only`, never an operation name. | t6 |
| `sweep_gate.py` | **Merged, new.** Re-decides stored `predictions.jsonl` rows offline, over a threshold grid, per fold; refuses test/held-out/final-labelled paths unless `--final`. Sanity-checked against `scorer-b1`'s own exact test-side predictions with every threshold disabled: 61 of 64 decisions reproduced exactly; the other 3 were recorded `invalid` there because `scorer-b1`'s argument grounding failed on them — a step the gate (distribution only) never sees, so this is expected, not a bug. **P14 fix (merged):** an argument-grounding failure it finds itself is now kept as `invalid` (same operation) or reported as "not grounded by the sweep" (a different operation), instead of being turned into a proposal with invented `{}` arguments. | t6 |
| `permutation_probe.py` | **Merged, new.** Kinds: `order`, `letters`, `subset` (keeps the gold candidate; also reports how often the *baseline's own choice* would have been the one removed), `paraphrase`, and `all` (order + letters + subset drawn together in one `scorer.permute` call, OpenJev-style). Per-trial seeding is `"<seed>:<entry id>:<kind>:<i>"`; an answer change is the op-level choice differing from the baseline (default map) choice. The bootstrap CI resamples *entries*, not trials, since trials of one entry are correlated. Also splits change rates by lowercase vs uppercase letters (park v3's residual risk). Never canonicalises runtime order; refuses test/held-out without `--final`. The real scorer is built through `measure.py`'s `Seams().build_scorer`. **P14 fixes (merged):** the in-process scorer it builds now covers the whole 52-letter alphabet (`measure.build_scorer` gained a `labels` override — the tool previously only ever exercised letters `A` through `R`), and incomplete trials are reported per kind separately rather than being scored as if complete. | t7 |
| `measure.py` | **Merged, closes #57.** `--calibration PARAMS` applies the fitted temperature then vector to a line's candidates before metrics/predictions are computed (refuses a params file fitted on test/held-out by name, or a malformed one). The report gains 95% bootstrap CI rows, a "Scorer readouts, complete / incomplete (never renormalised)" row, an "ECE / Brier before `--calibration`" row for comparison, and a "Per-slice calibration" section with reliability tables. Preflight now requires `--max-logprobs >= READOUT_TOP` (d1). **Issue #57 fixed:** a run where every server start-up failed now writes a page explicitly marked "nothing measured", which a re-run under the same label freely replaces; a page that measured anything, even partially, still refuses to be silently overwritten; a "nothing measured" page is never counted as a final run. **Caveat recorded for later tasks:** vector scaling can change which candidate is top-1 even though the recorded *decision* stays whatever the model actually chose before scaling — so any thresholded decision (the gate, t6) must be re-made on the calibrated distribution, never read off the pre-calibration argmax. | t8 |
| `docs/tool-jev-calibration-rule.md` | **Merged** (commit `b1c6cb6`). The pre-registered checkpoint-selection rule — selection fold, ordered criteria, tie-breaks, and the calibration-aware-stage trigger — confirmed by the operator before any training. See [below](#the-pre-registered-checkpoint-decision-rule-t9). | t9 |
| `docs/qwen-tool-jev-calibration.md` | This file. | t10 |
| `docs/qwen-tool-jev-finetune.md`, `docs/benchmarks/2026-09-24-qwen-tool-jev-comparison.md`, `release_bundle.py` (model-card text) | **Merged.** Track A is now described as a specialized generative tool router and Track B as the Jev-style candidate scorer; no text calls them equally Jev-like. | t11 |
| `split.py` | **Merged.** New v2 mode: repeatable `--corpus`, `--version`, `--val-size`/`--test-size` as counts, `--fold-seed`. The header is a string opening with the v1 side note (`Split '<side>' of corpus-<version> (seed=N).`) so every existing reader and the test/held-out guards still parse it; the structured metadata `{version, seed, sources: [{path, sha256}], sizes, side}` sits under a top-level `"split"` key, which on the validation side also carries `fold_seed`/`fit_ids`/`selection_ids`, and a `folds.json` is written for `calibration_fit`. **l5 fix (merged):** an earlier revision put the metadata in a dict `header`, which broke `train_scorer`, `build_dataset`, `merge_variations`, `measure` and the refusal guards; the agy review found it before any v2 split was built. Refuses any output path under `nvsh/` and refuses `held-out.json` as input. **P14 fix (merged):** assembly now allocates per class across sides with source groups kept intact, instead of the earlier contiguous round-robin slicing that could starve validation/test of rare decline classes; a non-vacuity assertion guards the fold test. Requested totals can still land 1-2 off target per answer kind, from largest-remainder rounding. **d3 (merged):** `--train-only CORPUS` (repeatable) appends a corpus to the train side only; val and test are split from the `--corpus` inputs alone, each such entry carries `train_only: true` and its file is listed in `sources` with `"train_only": true`. | t12 |
| `draft_heldout.py` | Reused from issue 46, extended with a `--seed` option (default 46) so this cycle can draft its own fresh held-out set independent of issue 46's; gained a new `as_item()` fix (see [ledger P9](#ledger-symptom---cause---fix)) that turns a bare-string model reply into a text-only item instead of crashing. See [ledger P7](#ledger-symptom---cause---fix) for the seed-53 draft's original parse gap. | t13 |
| `draft_sources.py` | **Merged, new** (a second t13 tool, alongside `draft_heldout.py`). `draft OUT --pool eval\|heldout --seed N --per-op K --per-reason K --explain K`: a table-only generator producing per-operation requests with validated arguments, per-decline-reason escalate prompts for all 8 classes, and explain Q/A pairs. `review IN OUT`: two independent reviewers give a strict yes/no; only what both accept is kept; exact and near-duplicate (Jaccard >= 0.8) dedupe against `dev.json` and within the draft, reusing `leakage_check.py`. Prints counts and hashes only; `review.jsonl` omits entry text entirely for the `heldout` pool. Reviewer roles are configured through `NVSH_DRAFT_<ROLE>_*` environment variables — no literal URL, key or model name is hard-coded. **Reworked (P12 below):** a reviewer's empty reply is now retried up to twice before being counted as a genuine reject (reason `"empty reply"`); the per-reviewer reply budget was raised to 8192 tokens; the escalate-reason definitions (`missing_argument`, `not_a_request`, `multi_step`, `injection`) were tightened to match the corpus's own decline classes exactly. **P14 fix (merged):** `--seed` now actually controls generation via a deterministic per-call request seed (a sha256 of the run seed, role, prompt key and attempt number), recorded as `"sampling": {"per_call_seed": true}` with a note that reproducibility only holds on endpoints that honour the request seed. **l6/l7 (merged):** `draft --only-reasons a,b` drafts a top-up for named decline classes; the reviewer system prompt states the corpus policy (only technical knowledge questions are answered in words; small talk is handed off); `missing_argument` requires an action the table offers; `ambiguous`/`unclear` may justify an escalation verdict. | t13 |
| `build_dataset.py` | **Merged**, then reworked for the d2 integration fix (below). Every rendered example is enriched with its `permutation` (`{order, labels}`), `gold`, `perm_seed` (a sha256 of `"<perm-seed>:<example id>"`), and `descriptions` (only when reason candidates are offered). Flags: `--randomize-labels`, `--perm-seed`, `--min-subset` (default 6), `--full-set-probability` (default 0.3), `--reasons` (16 operations + `explain` + 8 `escalate:<reason>` = 25 candidates; the reason is read from the entry's `class` field, `decline:<reason>`, falling back to `outside_table` when unrecognised; descriptions come from `scripts/lfm-finetune/data/reasons.json`). **`--missing-candidate-rate` no longer touches the rendered Track A file** (`nvsh-train.jsonl`): its `<id>-nocand` examples (gold operation removed, gold retargeted to `escalate` / `escalate:outside_table`) are now written **only** into a new corpus-format scorer training file, and only when `--scorer-out PATH` is also given — see the d2 entry below for why. Default behaviour (no new flags passed) is unchanged. | t14 |
| `merge_variations.py` | Existing issue-46 tool; test coverage extended alongside t14's `build_dataset.py` changes. | t14 |
| `targeted_augment.py` | **New** (t15 tooling). `--train TRAIN.json --out SUPPLEMENT.json --recipes ... --per-recipe N --seed N [--exclude SIDES...] [--decide-by reviewer_b\|both] [--roles-from augment\|draft] [--review-out R.jsonl] [--dry-run]`. Five recipes for the comparison doc's data-set recommendations 2-6: `missing-argument` (rule-based: a train-side request's argument value replaced by a vague reference, deterministic under the seed, expect escalate `decline:missing_argument`, linked by `pair_of` and the original's `source_id`; reviewer asked "is the argument left unspecified?" and "is it a natural request?"), `diagnosis-explain` (teacher-drafted contrastive pairs, explain vs `decline:diagnosis`, kept or dropped whole, one shared `source_id`), `power-set` (explicit positives for every choice of every table operation with a choice argument), `disambiguation` (validated gold plus the `confusable` operation; reviewer must agree the gold is the most natural reading) and `hard-negative` (explain questions that name an operation's subject in passing). Reuses `augment.py`'s roles, retry, verdict parser and guards and `draft_sources.py`'s seeded client, JSON parsing and reviewer prompt. Drops exact repeats of train texts and exact or near-duplicate repeats of `--exclude` texts before any review; `leakage_check.py` stays the gate. The output's header names the train side, so `merge_variations.py --supplement` accepts it; prints counts and a sha256 only. | t15 |
| `merge_variations.py` (t15) | `--supplement` now keeps an entry's own `source_id` when it has one, so a t15 pair stays one group. | t15 |
| `data/reasons.json` | **Merged, new.** Scripts-side descriptions for the 8 `escalate:<reason>` candidates — never read from `nvsh/`. | t14 |
| `data/paraphrases.json` | **Merged, new.** At least 2 alternative descriptions per candidate, for `permutation_probe.py`'s `paraphrase` kind. | t14 |
| `augment.py` (verdict parser) | **l7 (merged).** `parse_verdict` no longer reads a hyphenated `no-` compound (`no-argument`) as a no, and takes `allowed_hedges`: `draft_sources.py` and `targeted_augment.py` pass `ambiguous`/`unclear` for escalation verdicts only. Every other hedge and any standalone no still reject. | t13 |
| `pipeline.sh` | **Reworked for d2 (below).** The `assemble` stage takes `SCORER_BUILD_ARGS`, which adds `--scorer-out $WORK/data/scorer-train.json` to the `build_dataset.py` call, producing the corpus-format scorer training file alongside the rendered Track A file. **l5/t17 (merged):** `MEASURE_REASONS=1` adds `--reasons` to every scorer measure stage; `<run>.bf16_gguf` measures quantize's unquantized GGUF through llama-server, for the served-vs-in-process readout check. | t14/t16 (d2) |
| `train_scorer.py` | **Merged.** Reads each row's own permutation, gold, descriptions and `perm_seed`, and renders that row's own prompt; per-row label columns are padded with `-inf` masks. `--label-readout` chooses `variants` (default, the shared `distribution()` definition) or `single` (reproduces `scorer-b1` bit-identically — reproducing `b1` now requires passing `--label-readout single` explicitly). `--label-smoothing` / `--brier-weight` (default 0, off) add the calibration-aware loss terms beside cross-entropy. `train-log.json` gains `label_variant_ids`, `letter_ids`, `calibration_loss` and `permutations`; each row's own map is written to `<out>/row-maps.json` (path + sha256), so a run can be replayed exactly. **Fixed by d2 (below):** now trains on the new corpus-format scorer file (`scorer-train.json`) when it exists and is newer than `train-augmented.json`, and prints which file it used — closing an integration gap where randomized-label runs would otherwise have trained on the fixed label map with no missing-candidate rows at all. | t16 |
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

## Decision path

Every choice that shaped this cycle, in the order it was made: the
evidence that forced it, what was chosen, who decided, and where it is
recorded. Operator decisions come from the scope/think/challenge passes
(`.devague/` frame, exported spec); `dN` are approved deviations and `lN`
lapses, all posted on issue #61 when they happened.

### Scope (before any code)

- **D1.** **Runtime wiring is out of scope.** Evidence: the issue's own order puts
   measurement and retraining first. Choice: the gate this cycle produces is
   wired into nvsh by issue #54, not here; no file under `nvsh/` changes.
   Operator, q1.
- **D2.** **The bar is OpenJev's level.** OpenJev reports about 2.3% answer change
   under shuffled options. Choice: judge the permutation answer-change rate
   on the point estimate (<= 2.3%) with its 95% CI reported, at least 10
   permutations per entry on a ~150-entry fresh test side. "Missing data is
   not an explanation - it's a target to fill." Operator, q2 and q12.
- **D3.** **Robustness by training, not by a trick.** Forcing one strict candidate
   order at runtime would hide order sensitivity instead of removing it.
   Choice: avoid it; order is never canonicalised. Operator note, c33.
- **D4.** **No `logit_bias` readout.** The pre-challenge probe showed it leaves
   llama-server's returned probabilities unchanged. Choice: complete
   readout comes from a large top-k instead. Operator, q8.
- **D5.** **The 8 escalation reasons only if they help.** Choice: an ablation
   (candidate r2) decides. Operator, q3.
- **D6.** **Re-split, and add new sources where data is short.** Operator, q4;
   refined by d3 below.
- **D7.** **Track B is the subject.** Track A gets only the documentation relabel;
   its improvements are issue #56. Operator, q5.
- **D8.** **Corpus v2 lives scripts-side and in the private data repo;**
   `nvsh/tiers/corpus` stays untouched. Operator, q10.
- **D9.** **A new, teacher-drafted, private held-out;** the lead sees counts and
   hashes only. Operator, q11.
- **D10.** **No RLCD.** TypeSafe did not publish the method; a conditional
  calibration-aware stage of this cycle's own (r4, label smoothing plus a
  Brier term) runs only if the chosen checkpoint's ECE is above 0.10.
- **D11.** **Checkpoint choice is pre-registered** (t9, `tool-jev-calibration-rule.md`)
  before any training: r1 randomized labels, r2 plus the 8 reasons, r3 at
  lr 1e-4; judged on the validation selection fold in a fixed order (0
  wrong mutating, accuracy floor, lowest permutation change, lowest ECE
  after temperature, highest missing-candidate escalation). Operator:
  "Rule ok".

### During the run

- **D12.** **d1** (t1): `train_scorer`'s adoption of the shared label-probability
  definition moved to t16. Approved.
- **D13.** **`READOUT_TOP` 5000 -> 20000.** Evidence: at 5000 the deployed Q4
  GGUF missed labels on 5 of 64 prompts; 20000 missed none at 0.25 s per
  request on CPU. Ledger P13.
- **D14.** **d2** (t14): the per-example randomized data reaches Track B training
  through a separate corpus-format scorer file (`--scorer-out`); it had
  only reached the Track A file. Approved.
- **D15.** **Reviewer reply budget 1024 -> 8192 tokens, empty reply retried.**
  Evidence: reviewer B (a thinking model) often spent the whole budget
  reasoning and replied empty, counted as a reject. Lapse l4; ledger P12.
- **D16.** **Use Codex, agy and kiro as reviewers.** Operator permission. agy
  headless denies every tool, so it runs with skip-permissions inside its
  sandbox under a read-only prompt (operator-approved); the checkout is
  checked unchanged afterwards.
- **D17.** **l5: split v2's header back to a string.** Evidence: the agy review
  found v2 wrote a dict header that every reader, including the
  test/held-out refusal guards, parses as a string, plus two reasons-mode
  gaps in `measure.py`. Choice: keep the v1 side note as the header and
  move the metadata under `"split"`; add `--reasons` to measure and the
  probe. Fixed before any v2 split existed. Ledger P15.
- **D18.** **d3: fresh-only evaluation sides.** Evidence: a plain re-split would
  put `scorer-b1`'s own training entries, and issue-46 validation entries
  the lead read one by one, on the new val/test sides, inflating the
  baseline and making "fresh test" untrue. Choice: `dev.json` goes to
  train only (`split.py --train-only`); val and test come only from this
  cycle's teacher drafts, unseen by `scorer-b1` and by the lead. Operator
  chose this option.
- **D19.** **More data over less.** Operator: "Overall I prefer more entries and
  data than less." Choice: val and test sized at 200 each (the plan said
  ~150), top-ups for thin classes, generous t15 counts, and the reviewed
  issue-46 variations reused where leakage-clean.
- **D20.** **l6: the reviewers were told the corpus's small-talk policy.**
  Evidence: counting seed 60's verdicts per class showed every
  `not_a_request` candidate rejected: reviewer B's prompt allowed
  "answer in words" for any non-machine text, against `dev.json`'s
  escalate label; and the `missing_argument` definition let the generator
  draft unsupported actions. Choice: state the policy in the reviewer
  prompt; require an offered action for `missing_argument`;
  `--only-reasons` for targeted top-ups. `not_a_request` went from 0 to
  42 pool entries. Ledger P16.
- **D21.** **l7: two over-strict verdict-parser rules narrowed.** Evidence: stored
  yes-replies turned into rejects by `no-argument` (read as "no") and by
  "ambiguous"/"unclear" used as the reason for an escalation. Choice: a
  hyphenated `no-` compound is not a no; the two words are allowed in
  escalation verdicts only; every other hedge still rejects. The 159
  previously rejected escalate candidates were re-reviewed: 51 kept.
  Ledger P17.
- **D22.** **Cortex concurrency was not raised.** Reviewer B throughput limits
  every data step, but the lobe's own configuration documents
  `--max-num-seqs 2` as an out-of-memory guard. Choice: run two streams
  in parallel instead, one per reviewer slot.
- **D23.** **Held-out sealed; corpus v2 built** (counts and hashes in
  [Where the run stands](#where-the-run-stands)).
- **D24.** **t17 measures the served bf16 GGUF through the pipeline**
  (`<run>.bf16_gguf`), and uses `--predictions` rather than `--details`,
  which is empty for scorer runs. Ledger P18.
- **D25.** **Vector scaling earns its parameters on `scorer-b1`.** Evidence: on the
  validation selection fold (61 entries) temperature alone barely moves
  ECE (Q4 0.131 -> 0.120; exact 0.111 -> 0.120) while temperature plus
  vector reaches 0.074 (Q4) and 0.060 (exact). Choice: t19 fits both and
  keeps the variant that wins on the selection fold for the new
  checkpoint; with n = 61 the CI is wide, so the fresh test side decides
  the claim.

## Where the run stands

**Latest, 2026-09-25 afternoon.**

- **Eval pool (t13):** 480 entries (200 operation, 194 escalate, 86
  explain), deduplicated across seeds (44 near-duplicates removed) and with
  no exact or near match in `dev.json`. Escalate classes: `outside_table`
  29, repair 26, diagnosis 16, injection 18, `over_time` 30, `multi_step`
  25, `missing_argument` 8, `not_a_request` 42.
- **Held-out (sealed, private, read-only):** 149 entries (60 operation, 54
  explain, 35 escalate): the 159 kept by the fixed re-review, less 10
  removed by `leakage_check.py` against `dev.json` and the eval pool.
  Header opens "Held-out split" so every guard refuses it; sha256
  `62dca6ae05d79db7…`. Never read by the lead.
- **Corpus v2 (d3):** train 509 (245 operation, 134 escalate, 130 explain,
  including all 431 `dev.json` entries), val 204 (84 / 84 / 36; fit fold
  143, selection fold 61), test 198 (83 / 79 / 36). Val and test are fresh
  drafts only.
- **t17 baseline, `scorer-b1` on v2 validation (204):**

  | Build | Right proposals | Abstention recall | False-positive calls | Wrong mutating | Invalid | ECE | Brier |
  |---|---|---|---|---|---|---|---|
  | in-process (exact) | 54 / 84 | 70 / 84 | 16 / 120 | 0 | 14 | 0.081 | 0.240 |
  | served bf16 GGUF | — | 70 / 84 | — | 0 | 14 | 0.078 | 0.241 |
  | served `Q4_K_M` | — | 70 / 84 | 16 / 120 | 1 | 13 | 0.097 | 0.258 |

  Every readout was complete (204 / 204). Readout fidelity: served bf16
  GGUF vs in-process differs by more than 0.01 on 22 of 204 entries (max
  0.12, median 0.0001), so the 0.01 target is **not** met everywhere and
  the gap is reported; `Q4_K_M` vs in-process differs by more than 0.05 on
  26 entries (max 0.78) and changes 7 decisions, including the one wrong
  mutating proposal: a check-then-change request ("check swap usage, and if
  it's over 50%, switch to low power") that the exact build answers with
  the read-only `swap_status` and `Q4_K_M` with `power_set`. Calibration
  fits and the selection-fold result are decision D25 above; the gate sweep
  and the permutation probe are running.
- **t15 generating:** `targeted_augment.py` in two parallel streams
  (missing-argument and power-set, then diagnosis-explain; disambiguation
  and hard-negative), every protected side excluded.

**Wave 1 done**, 2026-09-25. **Merged:** t1 (readout core, `0b48c79`), t3
(served readout cap), t4 (calibration-fit module, `86f1504`), t9
(pre-registered decision rule, `b1c6cb6`), t10 (this guide) and t11 (Track
A/B documentation relabel). No training run has happened yet — t9's rule is
committed and operator-confirmed ahead of the cycle's first training
command, as required.

**Wave 2 merged in full**, 2026-09-25: t2 (permutation seam), t5 (per-slice
metrics), t6 (`gate.py` and `sweep_gate.py`), t8 (`measure.py` wiring,
closes issue #57) and t12 (corpus v2 split tooling) are all merged. A Codex
review of every task merged so far surfaced review-fix work landed inside
t3/t4 (see [ledger below](#ledger-symptom---cause---fix)) and lapse l3.
Full suite: 4459 passed.

**Wave 3 in progress**, 2026-09-25. **Merged (code):** t7 (`permutation_probe.py`),
t14 (`build_dataset.py` enrichment, `data/reasons.json`, `data/paraphrases.json`)
and the `draft_sources.py` tool for t13. Wave 3's code (t7, t14, t16) is now
fully merged; full suite 4540 passed.

**t13 running (data, not yet merged).** Held-out side: the combined
134-entry draft (seeds 53+54) went through two-reviewer review — **kept
86** (65 operation, 13 explain, 8 escalate), rejected by reviewer B (38),
reviewer A (16) and exact duplicates (4), hash `1cd89cdb5caaf1be…` (partial,
as forwarded). 8 escalates is too few to measure escalation on the held-out
side, so a top-up followed: seeds 53 and 55 each produced 0 escalates again
(the same single-reply parse gap, see [ledger P7](#ledger-symptom---cause---fix)),
and seeds 56/58 crashed outright until the `as_item()` fix (ledger P9,
below); reruns after the fix gave seed 56 16 escalate/16 explain, seed 57
16/16, seed 58 17/16. A top-up draft from seeds 55-58 (113 entries: 64
explain, 49 escalate) is now under the same two-reviewer review; once that
review lands, a cross-draft near-duplicate filter runs over the union
(counts only).

Eval-pool side: the seed-53 draft (10 per operation, 8 per reason, 60
explain) went through review — **kept 164** (142 operation, 22 escalate, 0
explain); by decline reason: `outside_table` 6, repair 7, diagnosis 3,
injection 1, `over_time` 5, and none for `missing_argument`,
`not_a_request` or `multi_step`. Rejects: reviewer B 50, reviewer A 31,
parse failures 3, exact duplicates 3, near duplicates 3. The pass took
about 2.5 hours at roughly 3 teacher calls a minute. Vote pattern:
operations were accepted by both reviewers 142 of 154 times; declines only
22 of 64 (23 rejected by both reviewers). The whole 60-Q/A explain request
never parsed in one reply (likely the generator's own token limit), and
some reason prompts failed too. A top-up is running now: seeds 54-56, 6
per reason and 20 explain per seed, generator max tokens raised to 12000,
no operation requests this round.

All teachers used across both drafts are Apache-2.0 (generator
Qwen3.6-35B-A3B, reviewer A Gemma-4-26B-A4B, reviewer B Qwen3.8-27B).

**Root cause found for the low escalate/explain yields, and fixed** (see
[ledger P12](#ledger-symptom---cause---fix)): reviewer B is a thinking
model that was running with a 1024-token reply budget and often spent it
all reasoning, replying empty — which every review pass so far had counted
as an ordinary reject. Fixed and re-running now, one runner at a time: a
full re-review of all 247 held-out candidates (seeds 53-58) under the fixed
budgets, then a re-review of the 42 previously rejected eval-pool declines,
then three fresh eval top-ups (seeds 57-59; 6 per reason, 20 explain each).
The first eval top-up under the *old* budget (seed 54) had already
finished and kept 25 (17 escalate, 8 explain) before the fix landed.

**Readout fidelity re-checked on `scorer-b1`** (spent test side, diagnosis
only, not a cycle claim): the shared `distribution()` definition closed the
`dev-f02` gap the [pre-challenge probe](#pre-challenge-probe-before-any-code-or-training)
found — in-process vs served bf16 GGUF top-choice agreement is now 100% on
all 54 complete entries, and `dev-f02` itself differs by 0.005 (was 0.137).
This is the evidence lapse l1 was waiting on. Remaining small gaps: max
absolute difference 0.029, with 3 of 54 entries above 0.01 (worst:
`dev-g278`, `dev-g190`, `dev-g269`). Completeness at the old `READOUT_TOP`
of 5000 was still short: 10 of 64 entries incomplete on the bf16 GGUF, and
on `Q4_K_M` 5 of 64 incomplete at top 5000 (0.17 s/request on CPU), 0 of 64
at top 20000 (0.25 s), 0 at top 60000 (0.44 s), 0 at the full 250000-token
vocabulary (1.46 s). **Fix (ledger P13):** `READOUT_TOP` raised from 5000
to **20000**, with every server cap, env example, release instruction and
test updated to match (rework inside t1/t3's files; c38's floor is >= 5000,
so this stays compliant). Full suite: 4543 passed.

**Integration gap found and fixed (deviation d2, proposed, awaiting
operator confirmation — see [ledger](#ledger-symptom---cause---fix)
below):** t14 enriched the rendered Track A file, `nvsh-train.jsonl`, but
`train_scorer.py` reads the corpus-format `train-augmented.json`, which
never got the enrichment — so a randomized-label training run would
silently have trained on the fixed label map with zero missing-candidate
rows. Fixed: `build_dataset.py --scorer-out` now writes a separate
corpus-format scorer training file; `pipeline.sh`'s `assemble` stage wires
it in via `SCORER_BUILD_ARGS`; `train_scorer.py` trains on it when present
and newer, and reports which file it used. Full suite: 4566 passed.

**A second Codex review of wave 2-3's code found six real bugs, all now
fixed and merged** (ledger P14, logged on issue #61): `permutation_probe.py`
only ever read labels `A` through `R` and did not separate incomplete
trials from complete ones in its counts — fixed with a `labels` override on
`measure.build_scorer` and separate incomplete-trial reporting;
`calibration_fit.py`'s fold split assigned individual example ids to
fit/selection, so one source's variations could straddle both folds —
fixed by folding on `source_id` as a group; `split.py`'s class-balanced
round-robin used contiguous slicing, which starved validation and test of
rare decline classes — fixed with per-class allocation across sides,
keeping source groups intact; `sweep_gate.py` turned a scorer's
argument-grounding failure into an ordinary proposal instead of preserving
it — fixed, it now keeps such rows `invalid` or reports them as "not
grounded by the sweep"; `draft_sources.py`'s `--seed` did not actually
control generation — fixed with a deterministic per-call request seed and
a recorded reproducibility caveat. Full suite: 4589 passed.

**Throughput note:** reviews run at roughly 2-4 teacher calls a minute
under the fixed 8192-token reviewer budgets. The serial runner's remaining
queue was stopped so the eval re-review, the eval top-ups and the held-out
re-review can all run in parallel. Reviewer B (Qwen3.8-27B) thinks at about
42 tokens/s with up to 8192 tokens per verdict and serves 2 requests at
once, which is why the held-out and eval reviews continue to share that
one reviewer rather than running strictly faster in parallel.

This section will be updated at each step as the lead forwards findings.

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
- **l1, an assumption not re-measured at the time (operator-approved
  lapse), now resolved.** The pre-challenge probe's `dev-f02` 0.137
  definition gap (P1 above) was claimed closed on the strength of t1's code
  change alone, before it had been re-measured on `scorer-b1` itself.
  **Resolved:** the diagnosis-only readout fidelity check recorded in
  [ledger P13](#ledger-symptom---cause---fix) re-measured it directly:
  `dev-f02` now differs by 0.005 (was 0.137), and top-choice agreement
  between in-process and served bf16 scoring is 100% on all 54 complete
  entries. The shared `distribution()` definition closed the gap as
  claimed.
- **l2, no CI control for the training-interpreter path (operator-approved
  lapse).** The torch/tokenizer-backed scorer tests (`label_variant_ids`,
  `TransformersScorer`, `label_logits_from_vocab`) run only under the
  training interpreter (44/44 passing there); the CI/uv environment skips
  them (35 pass, 9 skip) because torch is not installed there. There is no
  CI job that exercises this path; it depends on a human or the lead
  running it under the training interpreter before trusting it.
- **P4, calibration_fit's vector fit on the wrong rows.** **Symptom:** a
  synthetic-recovery test failed. **Cause:** the per-label vector fit ran
  on the unscaled rows, while `apply()` scales by temperature first, then
  by the vector — so the vector was fitted against a distribution `apply()`
  would never actually produce, silently corrupting the calibrated output.
  **Fix:** fit the per-label vector on the temperature-scaled rows, so the
  fit and `apply()` agree on what "the rows the vector corrects" means.
  Caught before merge by the failing synthetic-recovery test; t4 merged
  with the fix in place (28 tests for `calibration_fit.py`, full suite
  4325 passed).
- **P5, review-fix pass across t3/t4's scope (logged on issue #61).** A
  Codex review of every task merged so far found three fixes, landed as
  rework inside t3 and t4 rather than as new tasks: (1) `pipeline.sh`'s own
  default for `MEASURE_MAX_LOGPROBS` was still 22 (t3 had only changed
  `serve_for_measure.sh` and the env examples); raised to 5000, and
  `release_bundle.py`'s serving instructions now derive the cap from
  `scorer.READOUT_TOP` instead of a second hard-coded number — this also
  fixed an unrelated bug where `release_bundle.py`'s `_sibling()` helper
  was not registering the modules it imported in `sys.modules`. (2)
  `calibration_fit.py`'s NLL was computed by clipping the gold probability
  before taking its log; on a 99-right/1-wrong synthetic repro this drove
  the fitted temperature to about 0.06 instead of recovering the true
  T≈1. **Fixed:** compute NLL in log space without clipping. (3) The same
  fit did not roll `escalate:<reason>` mass up to plain `escalate` before
  computing NLL, which could drive a fit to the bounds (`T=20`) on a single
  adversarial row; **fixed**, and rows whose gold has no matching candidate
  at all are now skipped and counted (`skipped_gold_absent`) instead of
  corrupting the fit.
- **l3, a narrow merge-gate check (proposed, not yet operator-adjudicated).**
  The lead's merge gate for t3 checked only the files t3's own acceptance
  criteria listed (`serve_for_measure.sh`, the two pipeline env examples),
  which is why `pipeline.sh`'s separate hard-coded default (P5, above) was
  missed at t3's own merge and only caught later by the cross-task Codex
  review.
- **P6, `train_scorer.py`'s reproducibility trade-off.** Not a bug: adopting
  the shared `variants` label-probability definition as `train_scorer.py`'s
  default readout means the tool no longer reproduces `scorer-b1` bit-
  identically by default. **Fix (by design):** a `--label-readout single`
  flag reproduces `scorer-b1` exactly; reproducing it now requires passing
  that flag explicitly rather than relying on the default.
- **P7, `draft_heldout.py`'s seed-53 escalate gap (in progress).**
  **Symptom:** the seed-53 draft of this cycle's own fresh held-out set
  produced 59 entries (42 operation, 17 explain) and **0 escalate**
  entries. **Cause:** the single drafted escalate prompt's model reply
  failed to parse (1 parse reject), and separately 3 entries had invalid
  arguments and 3 were exact duplicates, all dropped by the existing
  structural checks — none of the losses were escalate-specific, but
  escalate started from the smallest pool and lost its only entry. **Fix
  in progress:** a second pass with `--seed 54` drafts escalate entries
  specifically, merged into the seed-53 draft programmatically (the lead
  never reads either draft's text, only counts and hashes, per c51/q11).
  Held-out draft hash (seed 53, pre-merge): `385092216b1e9b74…` (partial,
  as forwarded). **Update:** the seed-54 pass produced 75 entries (43
  operation, 16 escalate, 16 explain; 3 invalid args, 2 duplicates
  dropped); combined with the seed-53 draft into one 134-entry held-out
  draft (85 operation, 33 explain, 16 escalate), ids prefixed by seed,
  hash `767c5639b2ceef4f…` (partial, as forwarded). **Still not fully
  resolved:** review of that 134-entry draft kept only 8 escalates (see
  [P10](#ledger-symptom---cause---fix) below), too few to measure
  escalation on the held-out side, so a further top-up round is running.
- **P9, `draft_heldout.py` crashing outright on a bare-string reply.**
  **Symptom:** while chasing P7's escalate shortfall, seeds 53 and 55 again
  produced 0 escalates (the same single-reply parse gap), and seeds 56 and
  58 crashed with `AttributeError` instead of merely producing zero
  entries. **Cause:** `Qwen/Qwen3.5-4B` sometimes answers with a list of
  bare strings instead of a list of item objects, and the parser assumed
  every reply item was an object. **Fix:** a new `as_item()` helper turns a
  bare string into a text-only item; a bare-string reply for an operation
  request then correctly fails argument validation (as it should, since a
  bare string carries no arguments) rather than crashing. A test was added.
  Reruns after the fix: seed 56 gave 16 escalate/16 explain, seed 57 16/16,
  seed 58 17/16.
- **P10, the held-out review's escalate shortfall (in progress).**
  **Symptom:** two-reviewer review of the combined 134-entry held-out draft
  (seeds 53+54) kept only 86 entries (65 operation, 13 explain, **8
  escalate**) — rejected by reviewer B (38), reviewer A (16) and exact
  duplicates (4); hash `1cd89cdb5caaf1be…` (partial, as forwarded). **Cause:**
  8 escalates is too few to measure escalation reliably on the held-out
  side. **Fix in progress:** a top-up draft from seeds 55-58 (113 entries:
  64 explain, 49 escalate, drafted after the P9 fix) is under the same
  two-reviewer review; once that lands, a cross-draft near-duplicate filter
  runs over the union of everything kept so far (counts only, as always).
- **P11, the eval-pool draft's explain gap (in progress).**
  **Symptom:** the seed-53 eval-pool draft (10 per operation, 8 per decline
  reason, 60 explain) kept 164 of 284 on review (142 operation, 22
  escalate, **0 explain**); by reason: `outside_table` 6, repair 7,
  diagnosis 3, injection 1, `over_time` 5, and none at all for
  `missing_argument`, `not_a_request` or `multi_step`. Rejects: reviewer B
  50, reviewer A 31, parse failures 3, exact duplicates 3, near duplicates
  3 (vote pattern: operations agreed-accepted 142 of 154 times; declines
  agreed-accepted only 22 of 64, with 23 rejected by both reviewers). The
  pass took about 2.5 hours at roughly 3 teacher calls a minute. **Cause:**
  the entire 60-Q/A explain request never parsed in one reply, most likely
  because the reply hit the generator's own token limit; some reason
  prompts failed for the same reason. **Fix in progress:** top-up drafts
  from seeds 54-56, 6 per reason and 20 explain per seed (smaller requests),
  generator max tokens raised to 12000, no operation requests this round
  (the operation slice is already well covered).
- **P12, the actual root cause of the low escalate/explain yields: reviewer
  B's reply budget (merged, tested).** **Symptom:** across P7, P10 and P11,
  escalate and explain entries kept coming back far short of what was
  drafted, with many eval top-up rejects reasoned as "empty reply."
  **Cause:** reviewer B (Qwen3.8-27B, a thinking model) ran with the
  default 1024-token reply budget; it frequently spent the whole budget
  reasoning and returned an empty reply, and `draft_sources.py` counted an
  empty reply as an ordinary reject rather than a review failure — silently
  discarding entries the reviewer never actually judged. **A second,
  distinct issue found in the same pass:** the generator's own definitions
  for `missing_argument`, `not_a_request`, `multi_step` and `injection` did
  not match the corpus's own decline classes — for example the generator
  wrote "what's the status of the ssh service" as `missing_argument` (it is
  a normal, fully-specified request), which both reviewers correctly
  rejected. **Fix:** `draft_sources.py` now re-asks a reviewer up to twice
  on an empty reply, counting it as a genuine reject (reason `"empty
  reply"`) only if it stays empty after retrying; every reviewer's reply
  budget was raised to 8192 tokens; the four reason definitions were
  tightened to match the corpus's decline classes exactly (e.g.
  `missing_argument` = an action needing a target given with none at all,
  like "restart it"; `not_a_request` = small talk or questions about the
  assistant itself; `multi_step` = several operations, a condition between
  them, or acting on every service; `injection` = smuggled instructions or
  chained shell commands). Tested and merged.
- **l4, the reviewer-budget fix shipped without its own pilot check
  (proposed, grader-unverified — posted on issue #61).** The fix in P12 ran
  without first checking, on a small pilot batch, what the empty-reply rate
  actually is under the new 8192-token budget before committing to the
  larger re-review runs described above.
- **P13, `READOUT_TOP` was still too low for the deployed `Q4_K_M` build.**
  **Symptom:** a diagnosis-only readout fidelity check (spent test side,
  not a cycle claim) found completeness gaps even after t1/t3's earlier
  fixes: at the-then `READOUT_TOP` of 5000, 10 of 64 entries were
  incomplete on the bf16 GGUF, and 5 of 64 were still incomplete on the
  deployed `Q4_K_M` GGUF (0.17 s/request on CPU). **Cause:** 5000 was
  enough to close most of the gap (per the earlier P1 probe) but not all of
  it on this build. Sweeping further: `Q4_K_M` was complete (0 of 64
  incomplete) at top 20000 (0.25 s/request), top 60000 (0.44 s) and the
  full 250000-token vocabulary (1.46 s). **Fix:** `READOUT_TOP` raised from
  5000 to 20000 (still above the c38 floor of >= 5000), with every server
  cap, env example, release instruction and test updated to match, as
  rework inside t1's and t3's own files. The same check re-measured the
  `dev-f02` label-definition gap directly and found it closed (see the
  updated [l1](#ledger-symptom---cause---fix) above): top-choice agreement
  100% on 54 complete entries, `dev-f02` itself now 0.005 off (was 0.137),
  worst remaining gap 0.029 on 3 of 54 entries (`dev-g278`, `dev-g190`,
  `dev-g269`). Full suite: 4543 passed.
- **P8, a load-sensitive test flake (unrelated to this cycle's files).**
  **Symptom:** a full-suite run under heavy load failed
  `tests/test_readline_bash.py::test_bash_at_target_grammar_accepts_every_python_positive_row`
  once. **Cause:** the failure did not reproduce in isolation (36 of 36
  passed there), so it is a timing flake under load, not a real
  regression. **Fix:** none needed in this cycle's own files; noted here
  only so a future run does not mistake this specific test for evidence
  of a regression this cycle introduced.
- **d2, the scorer-training file integration gap (proposed deviation,
  awaiting operator confirmation on issue #61, under the operator's
  standing rule to `/deviate` as fitting and record cumulatively).**
  **Symptom:** t14 enriched every example in the rendered Track A file,
  `nvsh-train.jsonl`, but `train_scorer.py` (t16) reads the corpus-format
  `train-augmented.json` instead — which t14 never touched. **Cause:** the
  enrichment and the file the scorer actually trains on live in two
  different formats, and nothing wired the new fields from one to the
  other; had this gone unnoticed, every randomized-label training run of
  this cycle would have silently trained on the fixed label map with no
  missing-candidate rows at all, defeating the point of t2/t14/t16.
  **Fix (merged):** `build_dataset.py --scorer-out PATH` writes a new
  corpus-format scorer training file — the original entries plus the
  `<id>-nocand` missing-candidate examples, each carrying the permutation,
  gold, `perm_seed` and descriptions from the same enrichment pass, plus a
  provenance block; `pipeline.sh`'s `assemble` stage gained
  `SCORER_BUILD_ARGS` to add `--scorer-out $WORK/data/scorer-train.json`;
  `train_scorer.py` now trains on that file when it exists and is newer
  than `train-augmented.json`, and prints which file it used. **A related
  fix, found by Codex during this same pass:** `<id>-nocand` rows are now
  written *only* into the scorer file, never into the rendered Track A
  output — in the Track A file they would have carried the same prompt and
  tool list as their source row but the opposite target, which is wrong
  for a generative tool router. Default behaviour (no `--scorer-out`) is
  unchanged. Full suite: 4566 passed.
- **P14, a second Codex review of wave 2-3's code found six real bugs, all
  now merged (logged on issue #61).**
  (1) `permutation_probe.py`'s letter permutation only ever drew from `A`
  through `R`, silently never reaching later letters of the alphabet —
  **fixed:** `measure.build_scorer` gained a `labels` override so the
  in-process scorer it builds covers the whole 52-letter alphabet. (2) The
  same tool did not separate incomplete trials from complete ones in its
  counts — **fixed:** incomplete trials are reported per kind separately,
  never scored as if complete. (3) `calibration_fit.py`'s fold split
  assigned individual example ids to the fit or selection fold, so one
  source's stored variations could land on both sides of the split it is
  meant to keep separate — **fixed:** `make_folds` gained an optional
  `group_of` so a whole source folds together. (4) `split.py`'s
  class-balanced round-robin used contiguous slicing per class, which
  could starve validation and test of the rarer decline classes —
  **fixed:** allocation is now per class across sides with source groups
  kept intact, guarded by a non-vacuity assertion in the fold test. (5)
  `sweep_gate.py` turned a scorer's own argument-grounding failure into an
  ordinary `propose` decision instead of preserving it — **fixed:** a
  grounding failure is now kept as `invalid` (same operation) or reported
  as "not grounded by the sweep" (a different operation), never given
  invented `{}` arguments. (6) `draft_sources.py`'s `--seed` flag did not
  actually control generation (each teacher call ran effectively
  unseeded) — **fixed:** a deterministic per-call request seed (sha256 of
  the run seed, role, prompt key and attempt) is now sent, recorded as
  `"sampling": {"per_call_seed": true}`, with a note that reproducibility
  only holds on endpoints that honour the request seed. Full suite: 4589
  passed.
- **P15, split v2's dict header broke every reader (lapse l5).**
  **Symptom:** an agy review found `train_scorer.read_split`,
  `build_dataset`, `merge_variations`, `measure.py`'s side/seed parsing,
  `quantize.py`, `augment.py`, `dataset_bundle.py` and the test/held-out
  guards in `calibration_fit`/`permutation_probe` all expect a string header.
  **Cause:** t12 changed the header's type without an end-to-end test
  through its readers. **Fix:** string header with the v1 side note,
  metadata under `"split"`, `check_version()` refusing side-like versions,
  and an end-to-end test through every reader. Same review: `measure.py`
  now scores an `escalate:<reason>` choice as an escalation and has a
  `--reasons` mode (probe too), so r2 is measured on the prompt it trained
  on.
- **P16, a reviewer policy mismatch starved two decline classes (lapse
  l6).** **Symptom:** 0 `not_a_request` and 1 `missing_argument` in the
  whole eval pool. **Cause:** reviewer B's system prompt offered "answer in
  words" for any non-machine text, so it rejected small talk that
  `dev.json` labels escalate; the `missing_argument` definition did not
  require an action the table offers. **Fix:** the prompt states the
  corpus policy; the definition requires an offered action;
  `draft_sources.py draft --only-reasons` for targeted top-ups.
- **P17, the verdict parser discarded clear yes votes (lapse l7).**
  **Symptom:** yes-replies stored as rejects. **Cause:**
  `augment.parse_verdict` read ", no-argument" as a standalone "no", and
  counted "ambiguous"/"unclear" as hedges even when they were the reason an
  escalation is right. **Fix:** hyphenated `no-` compounds are not a no;
  `ambiguous`/`unclear` are allowed in escalation verdicts only;
  `targeted_augment.py` uses the same rule. Recovery: 51 of 159 previously
  rejected escalate candidates kept on re-review.
- **P18, `--details` is empty for scorer runs.** **Symptom:** t17's
  per-entry details files were 0 bytes. **Cause:** `detail_rows` reads the
  generative result object, which a scorer run does not produce. **Fix
  (workaround):** t17 keeps `--predictions`, which carry every entry's
  distribution; the details gap itself is left as it is.
- **Throughput note (not a bug).** Reviewer B (Qwen3.8-27B) thinks at
  about 42 tokens/s with up to 8192 tokens per verdict and serves 2
  requests at once, so reviews run at about 2-3 teacher calls a minute; the
  held-out and eval reviews continue to share that one reviewer.

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
