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
| `quantize.py` | New. Train-side-only calibration set (checks split headers), GGUF convert + imatrix + `Q4_K_M`, INT4 AWQ export, tool versions, `heal_needed()`. |
| `gen_config.py` | New. `write`, `check` and `stock-copy` of a `generation_config.json` pinning greedy decoding (d3). |
| `capped.sh` | New. `run_capped`: a hard RAM and swap cap through `systemd-run`, with a free-memory log. |
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

72 fresh entries were drafted by Qwen3.5-4B (Apache-2.0, not a teacher) from
the operation table only (seed 46, temperature 0.7, thinking off): 39
operation entries covering 15 of the 16 operations, 17 escalations and 16
explain asks. sha256
`95c7cd3eb854f1b24a133a3f77df71a162369a91a5299d2696bbcb199ae2107f`. It awaits
the operator's review. The lead has not read it, and it is unsealed only for
the final run (decision c51).

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
python scripts/lfm-finetune/measure.py snapshot --out <snapshot.json> \
  --from-split "$WORK/splits/val.json" --from-split "$WORK/splits/test.json"
```

Add the held-out file with another `--from-split` when it is unsealed. The
command prints counts only. Every later measurement passes
`--ground-snapshot <snapshot.json>`.

### 9. Stock baseline *(not yet run as a final baseline)*

Measure stock on validation, then once on the test side, the held-out set
and the missing-candidate slice, at 2K and 4K, before any tuned run is
scored:

```bash
python scripts/lfm-finetune/measure.py --split "$WORK/splits/val.json" \
  --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --label stock-val --config "$NVSH_CONFIG" --enable-thinking false \
  --ground-snapshot <snapshot.json> --ctx 2048 --predictions "$WORK/measure/stock-val"
```

The t13 live check ran stock on validation through attach mode; its exact
command is not recorded here.

### 10. Train Track A on spark *(not yet run)*

```bash
$P --env qwen.env train a1
$P --env qwen.env measure-val a1
```

`train` runs `train.py` under the memory cap, merges, writes the
`generation_config.json` and stages the result in `HF_CACHE` as `REPO`.
`TRAIN_ARGS` in the env file still holds issue 39's 350M settings and must be
re-tuned for 0.8B on validation only (c20).

### 11. Train Track B on spark2 *(not yet run)*

```bash
$P --env qwen.env train-scorer
```

Under the memory cap (`TRAIN_MEMORY_MAX`, 24G in the example). spark2 serves
other models next to it; their containers must not restart. On GB10 that cap
does not cover GPU allocations (P34); the GPU cap from f10 is required
before this step *(pending)*.

### 12. Final measurement *(not yet run)*

```bash
$P --env qwen.env measure-final a1
```

Once per checkpoint, on the test side, the held-out set and the
missing-candidate slice. Any retry is a deviation.

### 13. Quantize and heal *(not yet run)*

Build llama.cpp (the spike used master
`633733d0aeedd721868bf5f1b935fa3f39f9164e`, configured with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`) and a separate venv for
llm-compressor (`uv pip install llmcompressor --index-strategy
unsafe-best-match`; see ledger P28). Then:

```bash
LLAMA_CPP_CONVERT=<llama.cpp>/convert_hf_to_gguf.py \
LLAMA_CPP_QUANTIZE=<llama.cpp build>/bin/llama-quantize \
LLAMA_CPP_IMATRIX=<llama.cpp build>/bin/llama-imatrix \
LLM_COMPRESSOR=<awq venv>/bin/llmcompressor \
  $P --env qwen.env quantize a1
```

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
  across the probe. *Status:* blocking plan risk r13 (task t23). *Fix
  (pending, review-fix task f10):* an available-memory floor watchdog in
  `capped.sh` (`TRAIN_MEMORY_FLOOR`), plus
  `torch.cuda.set_per_process_memory_fraction` driven by
  `NVSH_TRAIN_GPU_MEMORY_GB` in `train.py` and `train_scorer.py` *(not yet
  merged)*.

## Not verified yet

- **`quantize.py`'s AWQ path.** It calls `$LLM_COMPRESSOR --model ...
  --calibration ... --scheme AWQ --bits 4 --out ...` as a command. The t16
  spike drove llm-compressor through its Python `oneshot` (with `processor=`,
  and linear-attention, vision, MTP and `lm_head` left out). The stage's AWQ
  invocation has not been run against the real tool, and it does not copy
  the tokenizer and preprocessor files vLLM needed (P26). Its GGUF
  conversion uses `--outtype f16`; the spike converted to bf16.
- **GGUF on AGX Orin**: the llama.cpp build has only run on spark.
- **The GGUF's sampling settings**: d3 covers "the GGUF's sampling metadata",
  and no step writes it yet.
- **Track A calibration** (P11, plan risk r12).
- **The re-review's exact command and filter** (step 5).
- **A GPU memory cap on GB10** (P34): the f10 fix is not merged or
  measured yet. Until it is, `TRAIN_MEMORY_MAX` does not bound a trainer's
  GPU allocations.
- **Whether stock meets any bar**: the t13 live check was a pipeline check,
  not the baseline run.

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
review; not read by the lead.

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
