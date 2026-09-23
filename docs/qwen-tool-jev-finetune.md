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

**Status: in progress, 2026-09-23.** The tooling is built and reviewed, the
spikes are done, the split is re-seeded, and the reviewer-B re-review is
running. Nothing has been trained yet. Steps not yet run are marked *(not yet
run)*, and anything not checked is marked *(unverified)* or listed under
[Not verified yet](#not-verified-yet). The dated [run log](#run-log-issue-46)
at the end records each step as it happens.

**Licences.** The base, `Qwen/Qwen3.5-0.8B`, is Apache-2.0. Every teacher
model in the data pipeline is Apache-2.0 once reviewer B is re-run (see
[the teacher pipeline](#the-teacher-pipeline)). NVIDIA's Jetson skill evals
are CC-BY-4.0 and are used as a test set only; nothing trained on them is
published here. Training happens on development machines. **nvsh itself
never trains and never uploads.**

## Where the run stands (2026-09-23, about 18:50)

- **Done:** the tooling is complete and live-checked. The sealed held-out
  set is done: 69 entries, sha256 `5eb650f9...`.
- **Running:** the reviewer-B re-review with thinking on, over 1,176
  candidates. About 92% are accepted so far. One empty-reply error will be
  retried by resuming.
- **Next, data:** augment the 95 new train-side sources through the
  all-Apache pipeline. Then filter and exclude (the new validation and test
  sides, issue 39's old test side, the held-out texts), assemble, scan,
  freeze with hashes, and copy to spark2.
- **Then, models:** the stock baseline at 2K and 4K; Track A on spark and
  Track B on spark2; one final run per checkpoint plus Track A's exact
  calibration; quantization; the edge check on AGX Orin; a private upload
  (with the operator's approval); the report; and a PR ("part of #46").

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
- **95 new train-side sources** have no variations yet. They are augmented by
  the same all-Apache pipeline *(not yet run)*.

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

### 6. Augment the new train sources *(not yet run)*

```bash
grant run --inject NVSH_GATEWAY_KEY=<secret> -- $P --env qwen.env augment-nvsh
$P --env qwen.env filter-variations
```

`augment-nvsh` is resumable and skips variation ids already written.
`filter-variations` counts accepted variations against the train split
without touching `assemble`'s output.

### 7. Assemble and freeze *(not yet run)*

```bash
$P --env qwen.env assemble
```

This merges the variations (with `--exclude` over val and test) and builds
`$WORK/data/nvsh-train.jsonl`, round-tripping one rendered example per
outcome through the Qwen template. After this the data is frozen; any later
change is a recorded deviation (decision c40).

### 8. Grounding snapshot *(not yet run)*

```bash
python scripts/lfm-finetune/measure.py snapshot --out "$GROUND_SNAPSHOT" \
  --from-split "$WORK/splits/val.json" --from-split "$WORK/splits/test.json"
```

Add the held-out file with another `--from-split`. The command prints
counts only. `GROUND_SNAPSHOT` is a required key in the env file, and the
pipeline's `measure-val` and `measure-final` stages pass it to every
measurement.

### 9. Stock baseline *(not yet run as a final baseline)*

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

For 4K, set `MEASURE_CTX=4096` and pass `--ctx 4096`
*(the 4K run is not yet recorded)*.

**A dead server fails the run** (P49). Before the first entry, `measure.py`
checks `GET <base_url>/models` and requires the served model name. Any tier
error, or a scorer call error on Track B's served path, fails the run with
exit 2 and no results page (the predictions and metrics are still written,
for debugging). `--allow-tier-errors N` accepts up to N and prints the count
at the top of the page. A scorer's normal "incomplete" top-k result is not an
error.

### 10. Train Track A on spark *(not yet run)*

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

### 11. Train Track B on spark2 *(not yet run)*

spark2 serves other models next to the trainer; their containers must not
restart. Three limits apply, because on GB10 the systemd cap does not cover
GPU allocations (P34). Set them in `qwen.env`:

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
$P --env qwen.env status     # prints: caps (as a child sees them): max=... floor=... watchdog=...s gpu_gb=...
$P --env qwen.env train-scorer
```

The floor and budget values for the real Track B run are not chosen yet.
If the watchdog trips, `run_capped` returns 3 and `mem.log` records why.

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
- **The re-review's exact command and filter** (step 5).
- **nvsh's runtime and unparsed Qwen tool calls** (P45): `LfmTier` still
  treats a failed parse as an explanation. Out of scope here (c9); tracked
  as [nvsh issue #50](https://github.com/agentculture/nvsh/issues/50).
- **Gaps left by h2**: the measure stages cannot name an AWQ build yet;
  `measure-final`'s results page still goes to `docs/benchmarks/` by
  default; `--limit-mm-per-prompt` on LFM2.5 is untested.
- **Thinking off for LFM2.5**: the LFM env example now also sends
  `enable_thinking` false. Whether LFM2.5's chat template ignores it has
  not been checked.
- **Two unidentified test failures.** After the f10 merge, one full-suite
  run had 2 failures whose names were not captured. The six runs after it
  were green. The cause is unknown.
- **Whether stock meets any bar**: the t13 live check was a pipeline check,
  not the baseline run.

## Ideas forward

None of these has been done yet.

1. **More data.** If data limits a result, generate more train-side data
   through the same teachers and guards. The operator: "we can always
   generate more data if needed".
2. **LoRA on the linear-attention layers.** Compare adapters on the
   Gated-DeltaNet projections with unsloth's default targets, on validation
   (plan risk r9, ledger P6).
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
