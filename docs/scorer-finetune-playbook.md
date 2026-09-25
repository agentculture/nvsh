# Calibrated scorer playbook

How to build, from nothing, a small local model that reads a request and
picks one action from a fixed table, answers in words, or hands off, and
whose probabilities can be trusted. Use this page when you need such a
model for **your own domain** (your own table of actions). It is written
from nvsh's two runs, issue 46 (`scorer-b1`) and issue 53 (the calibration
cycle). Those runs' guides are the case study behind every step:
[`qwen-tool-jev-finetune.md`](qwen-tool-jev-finetune.md) and
[`qwen-tool-jev-calibration.md`](qwen-tool-jev-calibration.md).

Every step below says what to run, what a healthy result looks like, and
what went wrong for us at that step. Steps nvsh has not finished yet in
issue 53 are marked **(first run in progress)**. Their tooling exists and is
tested, but no nvsh result backs them yet.

## What you get

A **candidate scorer**: a fine-tuned Qwen3.5-0.8B that reads one prompt and
generates exactly one token. The prompt lists the offered actions as
lettered lines (`A) disk_usage: ...`). The model's next-token probabilities
over those letters are the decision:

```text
request ──► prompt with lettered candidates ──► one forward pass
        ──► P(letter) summed over its token variants, normalised over offered letters
        ──► calibration (temperature, then per-label vector)
        ──► gate: propose / explain / escalate / abstain_uncertain
        ──► argument grounding for a proposal (deterministic, from the table)
```

The candidates are your table's operations plus two controls: `explain`
(answer in words) and `escalate` (hand off to a bigger model or a human).
You can also split `escalate` into named reasons such as
`escalate:missing_argument`. The deliverables are:

- a merged checkpoint and a `Q4_K_M` GGUF that `llama-server` can serve;
- calibration parameters and gate thresholds fitted for the build you
  actually deploy;
- a results page with bootstrap confidence intervals for every figure;
- a data set card and a model card, uploaded privately first.

### Is this the right tool?

Use it when:

- the decision is **one choice** from a fixed list;
- a wrong *mutating* action is expensive, so you need a calibrated way to
  say "not sure";
- the model must run on small hardware (0.8B, one token, about 80 ms warm
  on a DGX Spark in process).

Do not use it for multi-step plans or free-form argument extraction. The
scorer picks the action; arguments come from deterministic grounding
against your table.

**Hard limit:** the letter alphabet is `A-Z` then `a-z`, so operations plus
controls must fit in **52** candidates. In reasons mode (8 escalate
reasons in place of one `escalate`) that leaves room for at most 43
operations.

## What you need before starting

| Need | What nvsh used | Why it matters |
|---|---|---|
| A training GPU | DGX Spark (GB10, 128 GB unified memory). A Track B run of about 1,500 examples, 3 epochs, took about 26 minutes | Unified memory changes how out-of-memory failures look (see [Lessons](#lessons-that-transfer)) |
| A separate measuring GPU, or strict turn-taking | A second Spark | A served measurement on a GPU that is also training can lose its server mid-run |
| Three teacher models from **two model families**, with licences that allow redistribution of what they generate | Generator Qwen3.6-35B-A3B, reviewer A Gemma-4-26B-A4B, reviewer B Qwen3.8-27B (all Apache-2.0), behind an OpenAI-compatible endpoint | Teacher output becomes your published data set. Two families catch each other's blind spots |
| A drafter for the sealed held-out set that is not one of the teachers | Qwen3.5-4B, in process | Keeps the held-out set independent of the models that wrote the training data |
| An edge device for the last check (optional) | AGX Orin | The deployed build's latency and readout on the real target |
| Helper models for **code** review only (optional) | Codex, agy, kiro | They found most of the real bugs in our tooling. They never write training data |

**Time budget.** Teacher review is the slow part. A 27B thinking reviewer
serving 2 requests at once managed about 2-3 verdicts a minute. Reviewing
284 drafted entries took about 2.5 hours. Plan for a day or more of review
time across all the pools.

## Step 0: port the pipeline to your domain

Everything lives in `scripts/lfm-finetune/` and runs from an nvsh checkout.
**There is no configuration switch for the domain today.** The scripts
import nvsh's table, grounding, prompts and corpus loader directly. The
logic almost never names an operation (the only literal names are one
model-card example and a comment). So a port is a fork that replaces the
modules below. The operation names do not have to change throughout the
scripts.

Work through this checklist in a fork of nvsh:

1. **The operation table** (`nvsh/ops/table.py`, types in
   `nvsh/ops/_model.py`). Provide `OPERATIONS`, a tuple of
   `Operation(name, description, read_only, args)`, where each argument is
   an `ArgSpec(name, kind="str"|"choice", choices)`. Keep the API the
   scripts call: `get`, `names`, `validate` (never raises, returns codes
   such as `missing_argument` and `bad_choice`). Mark `read_only`
   carefully: the safety bar, the gate thresholds and the per-slice
   reports all key on it. Write descriptions that tell look-alike
   operations apart, because the description is the only thing the model
   sees about an operation.
2. **Argument grounding** (`nvsh/ops/ground.py`). nvsh grounds two
   argument kinds, `service` (from `systemctl`) and `container` (from
   `docker`). Add your own lookup kinds, or use only `choice` arguments,
   which ground without any lookup. Replace the matching world-snapshot
   pieces: `nvsh/tiers/bench.py` (`world_runner`, `load_world`) and
   `measure.py`'s `SNAPSHOT_KEYS` and `snapshot` subcommand.
3. **Prompts.** These strings carry nvsh wording. Rewrite each for your
   domain, keeping the structure:
   - `scorer.py` `_INSTRUCTION` ("pick the one action ... on this
     machine"). The candidate line format `"<L>) <name>: <description>"`
     stays.
   - The `explain`/`escalate` control descriptions, from
     `nvsh/tiers/lfm.py` `tools_for()`, and its system brief (the brief is
     needed only for a generative Track A model).
   - `augment.py`: `PHRASING_STYLES`, `DETAILED_STYLES`,
     `REVIEWER_SYSTEM`, the English guard regexes.
   - `draft_sources.py`: `GENERATOR_SYSTEM`, `REVIEWER_SYSTEM`,
     `_EXPLAIN_ASK` topics, and the `dev.json` de-duplication path.
   - `draft_heldout.py`: `SYSTEM` and its `dev.json` path.
   - `targeted_augment.py`: `_SPAN_NOUNS` and `_DX_ASK`.
4. **Escalation reasons** (only if you want named hand-off reasons).
   Three places must agree with each other and with the corpus's
   `class: decline:<reason>` values:
   - `scorer.REASON_CANDIDATES`;
   - `scripts/lfm-finetune/data/reasons.json`;
   - `draft_sources.REASON_DEFINITIONS`.
5. **Probe paraphrases**: `scripts/lfm-finetune/data/paraphrases.json`,
   at least two alternative descriptions per candidate.
6. **Packaging names**: `pipeline.sh`'s `BUNDLE_REPO_PREFIX` and
   `hub_upload.ALLOWED_PREFIX` (both hard-coded to nvsh's namespace), and
   the model-card text in `release_bundle.py`.
7. **Tests.** Several `tests/test_lfm_finetune_*.py` files use nvsh's real
   operation names as fixtures (gate, metrics, measure, augment,
   draft_sources, targeted_augment, dataset_bundle, calibrate_reviewer).
   Re-fixture them, then run the full suite green before you touch any
   data: `uv run pytest -n auto`.

Already domain-generic, no change needed: `split.py` (pass `--corpus`),
`merge_variations.py`, `leakage_check.py`, `calibration_fit.py`,
`gate.py`, `sweep_gate.py`, `metrics.py` (apart from the table lookup),
`train.py`, `train_scorer.py`, `quantize.py`, `gen_config.py`,
`stage_cache.py`, `scan_bundle.py`, `serve_for_measure.sh`, `capped.sh`.

`pipeline.sh split` passes no `--corpus`, so it always splits nvsh's
`dev.json`. Run `split.py` by hand (step 5) instead.

## Step 1: write the bars and the decision rule before any data

Write these down and commit them before you draft data or train anything.
A bar or rule chosen after seeing results is not a test. nvsh learned this
when `scorer-b1` was picked over a better-calibrated sibling by a rule
decided during the run.

**Bars** (nvsh's, adapt the numbers):

- **Zero wrong mutating proposals** at the chosen gate thresholds, on the
  test side and the sealed held-out set.
- **Calibration:** ECE at most 0.10 (10 equal-width bins, with a
  bootstrap CI), on the **deployed** build (for example `Q4_K_M` through
  `llama-server`), after calibration fitted on validation only.
- **Permutation robustness:** reordering the candidates, re-lettering
  them, offering a subset, or paraphrasing their descriptions changes the
  chosen *operation* in at most X% of trials. nvsh's X was 2.3%, OpenJev's
  published figure. Use at least 10 permutations per entry and report
  the 95% CI.
- **Missing-candidate escalation:** when the right operation is not
  offered, escalate or abstain at least 80% of the time.
- **Accuracy floor:** right proposals no worse than your reference minus
  5 points.

**Decision rule.** A template is
[`tool-jev-calibration-rule.md`](tool-jev-calibration-rule.md):

- the candidate runs, named in advance;
- the fold they are judged on (the validation selection fold, never test);
- ordered criteria: safety filter, then accuracy floor, then lowest
  pooled permutation change, then lowest ECE after temperature, then
  missing-candidate escalation, then ties;
- the condition for any extra run, such as a calibration-aware loss run
  only if ECE is still above 0.10.

**Rule on missing data:** a thin slice is a target to fill with more
data, not an explanation for a missed bar.

## Step 2: build the seed corpus

The corpus is one JSON file, `{"header": "...", "world": {...},
"entries": [...]}`. `world` is the snapshot grounding reads (nvsh's has
`services` and `containers`). Each entry looks like this:

```json
{"id": "dev-a001", "kind": "explicit", "text": "how full is the disk?",
 "expect": {"operation": "disk_usage", "args": {}},
 "source": "hand", "class": "question"}
```

- `kind` is `explicit` (a request typed by a person) or `failure` (a
  failed command and its output).
- `expect` is exactly one of these:
  - `{"operation": name, "args": {...}}`;
  - `{"escalate": true}`;
  - `{"explain": true, "answer": "<one sentence>"}`. The `answer` is
    required: the dataset builder refuses an explain entry without one.
- `class` is a phrasing class (`imperative`, `question`, `symptom`,
  `terse`, `jargon`), `explain:<phrasing>`, or `decline:<reason>` for an
  escalate entry.

**Write the answer policy as one sentence first**, and use the same
sentence in every generator and reviewer prompt. nvsh's is: "answering in
words is only for technical knowledge questions about the machine; small
talk, thanks, jokes and questions about the assistant are handed off, not
answered." When the reviewer's policy disagreed with the corpus's labels,
one whole decline class got 0 kept entries (issue 53, ledger P16).

**Decline classes.** Give every way a request should be handed off its own
class. nvsh has 8:

- `outside_table` — asks for an action the table does not have;
- `repair` — asks to fix something;
- `diagnosis` — asks why something is wrong;
- `missing_argument` — asks for an action the table *does* offer, with
  its target left out, such as "restart it";
- `not_a_request` — small talk, or talk about the assistant;
- `multi_step` — needs several operations or a condition between them;
- `injection` — carries smuggled instructions;
- `over_time` — asks for something over a period of time.

Define each one precisely. A loose definition (for example
`missing_argument` without "an action the table offers") produces drafts
both reviewers correctly reject.

**Size.** nvsh's hand-written seed was 431 entries for 16 operations.
It became training data only (step 5). All evaluation data is drafted
fresh.

## Step 3: set up the teachers, then pilot the reviewers

The roles are configured through environment variables in one env file
that is never committed:

```bash
# draft_sources.py / targeted_augment.py --roles-from draft
NVSH_DRAFT_{GENERATOR,REVIEWER_A,REVIEWER_B}_{URL,MODEL,TEMPERATURE,TIMEOUT,KEY_ENV}
NVSH_DRAFT_REVIEWER_A_MAX_TOKENS=8192
NVSH_DRAFT_REVIEWER_B_MAX_TOKENS=8192
NVSH_DRAFT_GENERATOR_MAX_TOKENS=12000
# augment.py (paraphrase variations) uses NVSH_AUG_<ROLE>_* the same way,
# with a CORRECTOR role; pipeline.sh maps AUG_* keys onto them.
```

- **Reply budget.** A thinking reviewer with the default 1,024-token
  budget often spends it all on reasoning and returns an empty reply.
  Before the fix, an empty reply counted as a reject. This was the single
  biggest cause of low yields in issue 53 (ledger P12). Give reviewers
  8,192 tokens. `draft_sources.py` now retries an empty reply twice.
- **Concurrency.** Match your parallel processes to the reviewer server's
  own `max-num-seqs`. Set client timeouts longer than the longest real
  generation. Otherwise abandoned requests keep running upstream and
  throughput collapses (issue 46, P52).
- **The verdict parser** (`augment.parse_verdict`) is strict: the first
  word must be "yes", with no standalone "no" and no hedge word. Two
  false rejects we hit:
  - "no-argument" was read as a no;
  - "ambiguous" was treated as a hedge, but it is the very reason an
    escalation is right.

  Escalation verdicts may now contain `ambiguous`/`unclear`, and the
  naturalness question may contain `might`/`could`/`may`/`though` (P17,
  P19).
- **Pilot before every large pass.** Review about 20 known-good and 20
  known-bad items, and read the verdict *reasons* for every reject. Each
  of issue 53's four yield problems (P12, P16, P17, P19) would have shown
  up in a 40-item pilot. Two of them cost hours of re-review because no
  pilot ran.

## Step 4: draft and seal the held-out set

The held-out set is written fresh, from the table only, by a model that is
not a teacher. **Nobody on the development side reads its text. Only
counts and hashes are ever printed.**

```bash
for s in 53 54 55 56 57 58; do                  # several seeds; merge the drafts
  python scripts/lfm-finetune/draft_heldout.py "$DRAFTS/heldout-draft-s$s" --seed "$s"
done
python scripts/lfm-finetune/draft_sources.py review "$DRAFTS/heldout-all.json" "$DRAFTS/heldout-rev"
python scripts/lfm-finetune/leakage_check.py --train "$DRAFTS/heldout-rev/draft.json" \
  --protected <seed corpus> "$DRAFTS/eval-pool.json" --out-filtered "$SEALED.tmp"
```

To seal it, write `$SEALED` with a header that starts with `Held-out
split`, so every measuring tool refuses it without `--acceptance`. Make it
read-only, store it privately (never in a public repository), and record
its sha256.

**Healthy:** at least about 30 entries per answer kind (operation, explain,
escalate). nvsh's first held-out draft had 0 escalates, because one
unparsed reply lost the only escalate batch. It took six seeds and a
review fix to reach 149 entries (60 operation / 54 explain / 35 escalate).
Check counts per kind after each seed, not at the end.

## Step 5: draft the evaluation pool and split

The evaluation pool becomes the validation and test sides. Draft it fresh
and review it with both reviewers:

```bash
python scripts/lfm-finetune/draft_sources.py draft "$DRAFTS/eval-draft" --pool eval \
  --seed 53 --per-op 10 --per-reason 8 --explain 60
python scripts/lfm-finetune/draft_sources.py review "$DRAFTS/eval-draft/draft.json" "$DRAFTS/eval-rev"
# top up any thin class with more, smaller requests:
python scripts/lfm-finetune/draft_sources.py draft "$DRAFTS/eval-topup-s62" --pool eval \
  --seed 62 --per-op 0 --per-reason 12 --only-reasons missing_argument,not_a_request --explain 0
```

`review` keeps only what both reviewers accept. It removes exact and
near-duplicates (Jaccard at least 0.8) against the seed corpus and within
the draft. Merge every kept file into one pool, deduplicating again across
files.

**Watch the yield per class** after each review (`review.jsonl` has the
reasons). Operation requests kept 142 of 154. Decline classes kept only
22 of 64 until the reviewer budget, policy and parser fixes landed. If a
class is starved, fix the prompt or the grader first, then top it up with
`--only-reasons`. Rejected entries can be re-reviewed after a grader fix
instead of redrafted. Ask for smaller batches: a 60-item explain request
never parsed in one reply.

**Split.** Evaluation sides come from the fresh pool only. The
hand-written seed corpus goes to train only:

```bash
python scripts/lfm-finetune/split.py --version v2 --corpus "$DRAFTS/eval-pool.json" \
  --train-only <seed corpus> --val-size 200 --test-size 200 \
  --seed 53 --fold-seed 53 --out-dir "$WORK/splits"
sha256sum "$WORK"/splits/*.json
```

This writes `train.json`, `val.json`, `test.json` and `folds.json`.
Validation is split into a **fit fold** (70%, for fitting calibration and
gate thresholds) and a **selection fold** (30%, for choosing a
checkpoint). Folds group by `source_id`, so variations of one source
never land on both sides.

- Aim for at least about 150 validation and 150 test entries. With 66,
  one flipped entry moved a rate by 6.7 points.
- Never read test entries one by one. Validation per-entry reading is
  fine.
- The requested totals can land 1-2 off per answer kind (rounding).

## Step 6: measure a baseline and check the readout

Before training, measure your starting model (stock, or a previous
checkpoint) on validation. Check that the numbers you will rely on are
real.

1. **Readout completeness.** A served model returns only the top-k
   next-token log-probabilities. If a candidate's letter is missing from
   them, the distribution is incomplete and is never renormalised.
   - Top 22 missed 10-15 of 18 labels on every prompt.
   - Top 5,000 still missed some on the quantized build.
   - `READOUT_TOP = 20000` was complete on all 64 prompts (0.25 s a
     request on CPU).

   Keep `MEASURE_MAX_LOGPROBS` equal to `scorer.READOUT_TOP`, and check
   the results page row "Scorer readouts, complete / incomplete". It
   must read N / 0.
2. **One label definition everywhere.** A letter is several tokens (`"A"`,
   `" A"`, `"\tA"`). Training, in-process scoring and served scoring must all
   sum the same variants. When they did not, one prompt differed by 0.137
   between two runs of the same weights.
3. **Served vs in-process fidelity.** Measure the same checkpoint in
   process, as an unquantized GGUF (`<run>.bf16_gguf`) and as the deployed
   quant (`<run>.q4_k_m`). Differences between in-process and bf16 GGUF
   are runtime differences. Differences between bf16 GGUF and Q4 are
   quantization.
4. **Baseline permutation probe**, so you know how much the label
   shortcut matters before you fix it.

```bash
P=scripts/lfm-finetune
for spec in "base in-process" "base.bf16_gguf served" "base.q4_k_m served"; do
  set -- $spec
  $P/pipeline.sh --env run.env measure-val "$1" --scorer "$2" --predictions "$WORK/pred"
done
python $P/permutation_probe.py --split "$WORK/splits/val.json" \
  --paraphrases $P/data/paraphrases.json --per-entry 10 --seed 53 \
  --model "$WORK/runs/base/merged" --tokenizer "$WORK/runs/base/merged" \
  --scorer-kind in-process --out "$WORK/probe/base-val.json" --markdown "$WORK/probe/base-val.md"
```

Scorer runs must pass `--scorer served` or `--scorer in-process`. Without
it, a scorer checkpoint is measured as a generative tool-caller and scores
about 0. Use `--predictions`, because `--details` is empty for scorer runs.

nvsh's baseline, `scorer-b1` on validation: ECE 0.081 in process and 0.097
on `Q4_K_M`, with 1 wrong mutating proposal on Q4 only. The pooled
permutation change was 18.8%: letters 40.6%, order 5.1%, and lowercase
letters 37.8% against uppercase 13.5%. **A model trained with fixed
letters learns the letters.** That is why training randomizes them (step
8).

## Step 7: grow the training data, then freeze it

Train-side data comes from three places, all reviewed by both teachers
and all checked for leakage against every protected side (validation,
test, held-out, and any older evaluation set):

1. **The seed corpus** (train-only, from step 5).
2. **Paraphrase variations** of each train entry: generator, then
   corrector, then two reviewers. Run the `augment-nvsh` (for nvsh's
   corpus), `rereview` and `filter-variations` stages of `pipeline.sh`.
   nvsh kept 1,206 variations from 315 sources.
3. **Targeted recipes** for the error classes your baseline shows, with
   `targeted_augment.py`:
   - `missing-argument` — rule-based: remove a train request's argument
     value, expect escalate;
   - `diagnosis-explain` — contrastive pairs, "why is X" against "what is
     X";
   - `power-set` — explicit positives for every choice of every choice
     argument (the name is nvsh's; the recipe is generic);
   - `disambiguation` — the request against a look-alike operation;
   - `hard-negative` — explain questions that mention an operation's
     subject in passing.

   ```bash
   T="python scripts/lfm-finetune/targeted_augment.py --train $WORK/splits/train.json \
     --seed 53 --roles-from draft --exclude <val> <test> <held-out> <eval pool>"
   $T --recipes missing-argument,power-set --per-recipe 20 --out "$WORK/aug/sup-a.json"
   $T --recipes diagnosis-explain --per-recipe 40 --out "$WORK/aug/sup-b.json"
   $T --recipes disambiguation,hard-negative --per-recipe 8 --out "$WORK/aug/sup-c.json"
   ```

   Check each recipe's reject reasons in its log. For example, 104 of 129
   hard-negatives were dropped because the generator wrote an operation
   identifier into the question. The fix was a prompt rule: "name the
   subject the way a user would, never the identifier."

Combine the supplements into one file. Give ids a unique prefix per file,
keep each entry's `source_id`, and use a header containing
`Split 'train' of`. Then assemble:

```bash
# run.env: SUPPLEMENT=<combined>, PROTECTED_EXTRA=<every other protected file>,
# SCORER_BUILD_ARGS="--randomize-labels --perm-seed 53 --missing-candidate-rate 0.3"
$P/pipeline.sh --env run.env assemble
sha256sum "$WORK"/data/*          # this is the freeze
```

`assemble` does five things:

1. Merges the variations and the supplement.
2. Runs `leakage_check.py` (exact text, or a 5-token shingle or word-set
   Jaccard of at least 0.8, printing ids only).
3. Builds the scorer training file, `data/scorer-train.json`.
4. Gives each row its own random order, letters and candidate subset.
5. Adds `-nocand` rows, with the gold operation removed and the target
   set to escalate, for 30% of operation entries.

For reasons mode, build a second file with `--reasons` added. After the
freeze, any data change is a recorded deviation.

## Step 8: train the candidates

```bash
$P/pipeline.sh --env run.env status                   # memory caps as a child process sees them
$P/pipeline.sh --env run.env train-scorer r1          # TRAIN_SCORER_ARGS for lr, etc.
```

`train_scorer.py` trains LoRA (all-linear, rank 16 / alpha 32) on
cross-entropy restricted to the offered letters, reading each row's own
letter map. It then merges, pins greedy generation settings and stages
the checkpoint. The defaults that worked for nvsh are 3 epochs and lr
2e-4. 5 epochs overfit and 2 underfit. lr 1e-4 calibrated better.

- **Check the merge.** Diff one merged weight against the base and
  confirm it changed, with 0 "missing adapter keys" warnings. A silent
  merge failure gave a "tuned" model identical to stock (issue 46,
  lapse l4).
- **Memory on unified-memory machines.** Set `TRAIN_MEMORY_MAX`,
  `TRAIN_MEMORY_FLOOR` and `NVSH_TRAIN_GPU_MEMORY_GB`. The systemd cap
  does not cover GPU allocations there. Do not measure on a GPU that is
  training.
- `train-log.json` and `row-maps.json` record everything needed to replay
  the run exactly.

## Step 9: choose a checkpoint on the selection fold

**(First run in progress.)** For each candidate, in process:

```bash
$P/pipeline.sh --env run.env measure-val r1 --scorer in-process --predictions "$WORK/pred"
# add MEASURE_REASONS=1 for a reasons-mode candidate
python $P/calibration_fit.py fit --predictions <r1 val predictions> \
  --folds "$WORK/splits/folds.json" --out "$WORK/calib/r1.params.json"
python $P/calibration_fit.py evaluate --predictions <r1 val predictions> \
  --params "$WORK/calib/r1.params.json" --folds "$WORK/splits/folds.json" --fold selection
python $P/permutation_probe.py --split "$WORK/splits/val.json" ... --model "$WORK/runs/r1/merged"
```

Then add the missing-candidate slice (`--slice missing-candidate`, or
`eval_slices.py`). Apply your pre-registered rule mechanically and write
down every candidate's numbers, including the losers'. Run a conditional
extra candidate only under the condition you wrote in step 1, for example
`--label-smoothing 0.1 --brier-weight 0.5` if ECE is still above 0.10.

## Step 10: quantize, then calibrate the deployed build

```bash
$P/pipeline.sh --env run.env quantize r1              # bf16 GGUF, imatrix, Q4_K_M (+ AWQ)
$P/pipeline.sh --env run.env measure-val r1.q4_k_m --scorer served --predictions "$WORK/pred"
python $P/calibration_fit.py fit ...                  # on Q4's own fit-fold predictions
python $P/calibration_fit.py evaluate ... --fold selection
python $P/calibration_fit.py apply ... --out "$WORK/calib/r1-q4-val.calibrated.predictions.jsonl"
python $P/sweep_gate.py --predictions "$WORK/calib/r1-q4-val.calibrated.predictions.jsonl" \
  --folds "$WORK/splits/folds.json" --fold fit --escalate none,0.3,0.4,0.5 \
  --floor none,0.3,0.4,0.5 --margin none,0.1,0.2 --max-entropy none,0.6,0.8 \
  --mutating-floor none,0.5,0.6,0.7 --mutating-margin none,0.2,0.3 --out "$WORK/calib/sweep-fit.json"
```

- **Fit on the deployed build's own validation predictions.** `Q4_K_M`
  moved probabilities by up to 0.1 against bf16, so parameters fitted on
  bf16 do not transfer.
- **Keep the vector step only if it helps.** Fit temperature first, then
  the per-label vector, and keep the vector only if it helps on the
  *selection* fold. For nvsh's baseline, temperature alone did not help
  (Q4 ECE 0.131 raw, 0.120 with temperature). The vector did (0.074),
  because the escalate label's own scale was off.
- **Re-make decisions after calibration.** The vector can change the
  top-1, so gate decisions are always re-made on the calibrated
  distribution.
- **The gate** (`gate.py`) has one set of thresholds for read-only
  operations and one for mutating operations. Each set has a top-1
  floor, a top-1/top-2 margin and a maximum normalised entropy, and
  returns `abstain_uncertain` when a check fails. Pick thresholds on the
  fit fold and confirm them on the selection fold. For nvsh's baseline
  the gate added nothing once calibration had removed the one wrong
  mutating proposal. Say so if that is your result too.

## Step 11: measure once on test and the held-out set

**(First run in progress.)** Run each final measurement once, after the
choice, with the calibration and thresholds frozen:

```bash
$P/pipeline.sh --env run.env measure-final r1.q4_k_m --scorer served
$P/pipeline.sh --env run.env measure-final r1.q4_k_m --slice missing-candidate --scorer served
$P/pipeline.sh --env run.env measure-heldout r1.q4_k_m --scorer served
python $P/permutation_probe.py --split "$WORK/splits/test.json" --final --per-entry 10 ...
```

Repeat the deployed build on the edge device. Report every bar with its CI
and its n, including the bars you missed. A rerun of a final measurement
is a recorded deviation.

## Step 12: package and publish

```bash
$P/pipeline.sh --env run.env bundle gguf r1.q4_k_m <suffix> <final report>.md ...
$P/pipeline.sh --env run.env bundle-dataset <suffix>-dataset
FINAL=1 $P/pipeline.sh --env run.env upload-bundle <suffix>   # token injected, never printed
```

`upload-bundle` refuses in any of these cases:

- `FINAL=1` is not set;
- the repository id is outside the allowed prefix;
- the bundle fails the secrets scan;
- the bundle holds a symlink (it could point at the sealed held-out set).

It uploads **private**, fetches the upload back and compares every file's
hash, then confirms the private flag. Making a repository public is a
separate, explicit decision by the owner. The data set card lists the
teachers and their licences. The held-out set never ships.

## Health checks at a glance

| After | Look at | Trouble looks like |
|---|---|---|
| Porting (step 0) | Full test suite | Any red. Do not start data work on a red suite |
| Each draft/review pass | Kept count per class, reject reasons | A class under about 30% yield, or "empty reply" rejects |
| Split | Per-kind counts per side, fold sizes, hashes | A decline class missing from validation or test |
| Baseline | "Scorer readouts complete / incomplete" | Any incomplete readout |
| Baseline | Letters vs order change in the probe | A large letters rate: a label shortcut to train out |
| Assemble | Leakage drops, rows per operation, `-nocand` count | Operations with very few rows. Leakage drops you cannot explain |
| Training | Merged weight differs from base, missing-key warnings | Identical weights, or any missing-key warning |
| Selection | Wrong mutating on the selection fold and missing-candidate slice | Any above 0 |
| Deployed build | ECE raw / temperature / temperature+vector on selection | A calibration step that makes the selection fold worse |

## Lessons that transfer

- **Measure the artifact you ship.** In-process bf16 numbers said nothing
  about the served `Q4_K_M` build, which at first returned 0 of 64
  complete distributions.
- **Fixed letters teach letters.** Randomize order, letters and subsets
  per example at build time, store the map with the example, and
  measure all four perturbation kinds.
- **Grader bugs look like data scarcity.** Every "not enough data in
  class X" in issue 53 was a reviewer budget, policy or parser problem
  first. Read the reject reasons before drafting more.
- **More data is an acceptable fix, before the freeze.** After the
  freeze it is a deviation.
- **Pre-register, then record every departure** (in nvsh, the devague
  `/deviate` records and a cumulative tracking issue). A choice made
  after seeing results is flagged as such.
- **Keep evaluation fresh and sealed.** Draft evaluation sides new, keep
  the hand-written seed on train, and never read the held-out set's text.
- **Use two strong reviewers for code as well as data.** Independent
  code reviews (Codex, agy) found more than 20 real defects that tests
  and a same-family reviewer had passed.
- **One GPU, one job.** A server measured next to a training run can
  fail to start or die mid-run on unified memory.
- **Keep a guide and ledger live.** Every step, obstacle, fix and
  decision goes in the moment it happens. This page was written from
  those ledgers.

## Where nvsh stands

Issue 53's run is at step 7 (the final targeted supplements are being
reviewed). Steps 8-12 have tested tooling but no issue-53 result yet.
This page will be updated with that run's numbers. The decision path,
ledger and live status are in
[`qwen-tool-jev-calibration.md`](qwen-tool-jev-calibration.md).
