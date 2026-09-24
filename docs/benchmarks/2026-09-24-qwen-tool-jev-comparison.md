# Tool-Jev (issue 46) comparison report, 2026-09-24

This page collects every measurement from the Qwen3.5-0.8B Tool-Jev
fine-tune (issue 46) into one comparison table, states the c33-c36 success
bars pass/fail per shipped build, and closes with a recommendation for the
next iteration. It is built entirely from the committed pages listed below
by file name, each of which is that number's run id; nothing here is
measured fresh. See [`docs/qwen-tool-jev-finetune.md`](../qwen-tool-jev-finetune.md)
for the full design, ledger and reproduce steps this report summarizes.

**Shipped:** Track A `a3-heal.q4_k_m`, Track B `scorer-b1` (both its
`.awq` and `.q4_k_m` builds pass c43 unhealed; `scorer-b1.q4_k_m` is the
recommended pick — see [Recommendation](#recommendation)). **Neither
checkpoint clears every success bar.**

## Accuracy and abstention, test side (64 entries: 32 operation, 15 escalate, 17 explain)

| Model | Run id (file) | Right proposals | Abstain recall | Precision (strict) | FP tool calls | Wrong mutating |
|---|---|---|---|---|---|---|
| Stock (generative) | `2026-09-24-lfm-final-stock.md` | 2/32 | 1/15 (6.7%) | 33.3% | 3/32 | 3 |
| Stock (exact scorer) | `2026-09-24-lfm-final-stock-exact.md` | 20/32 | 0/15 | 0.0% | 28/32 | 0 |
| `a3` (bf16) | `2026-09-24-lfm-final-a3.md` | 32/32 | 11/15 (73.3%) | 100% | 4/32 | 2 |
| `a3.awq` | `2026-09-24-lfm-final-a3.awq.md` | 31/32 | 11/15 (73.3%) | 91.7% | 3/32 | 2 (same ids as bf16) |
| `a3.q4_k_m` | `2026-09-24-lfm-final-a3.q4_k_m.md` | 31/32 | 11/15 (73.3%) | 100% | 4/32 | 2 (same ids as bf16) |
| `a3-heal` (bf16) | `2026-09-24-lfm-final-a3-heal.md` | 32/32 | 11/15 (73.3%) | 100% | 4/32 | 2 (same ids as `a3`) |
| `a3-heal.awq` | `2026-09-24-lfm-final-a3-heal.awq.md` | 30/32 (93.75%) | 11/15 (73.3%) | 91.7% | 4/32 | 2 (same ids) |
| **`a3-heal.q4_k_m` (shipped, Track A)** | `2026-09-24-lfm-final-a3-heal.q4_k_m.md` | **32/32** | 11/15 (73.3%) | 91.7% | 4/32 | 2 (same ids) |
| `scorer-b1` (served / exact, same decisions) | `2026-09-24-lfm-final-scorer-b1.md`, `-exact.md` | 27/32 (84.4%) | 11/15 (73.3%) | 100% | 2/32 | 0 |
| `scorer-b1.awq` | `2026-09-24-lfm-final-scorer-b1.awq.md` | 28/32 | 11/15 (73.3%) | 91.7% | 2/32 | 0 |
| **`scorer-b1.q4_k_m` (shipped, Track B)** | `2026-09-24-lfm-final-scorer-b1.q4_k_m.md` | **27/32** | 11/15 (73.3%) | 91.7% | 2/32 | **0** |

Every model's abstain recall lands at exactly 11/15 (73.3%) or 1/15 (6.7%,
stock) on the test side — the test side has few escalate entries (15), so
this is a coarse figure; see the held-out set (13 escalate) and the
missing-candidate slice below for more signal.

## Held-out set (69 entries: 41 operation, 13 escalate, 15 explain)

| Model | Run id (file) | Right proposals | Abstain recall | Precision (strict) | FP tool calls | Wrong mutating |
|---|---|---|---|---|---|---|
| Stock (generative) | `2026-09-24-lfm-heldout-stock.md` | 6/41 | 1/13 (7.7%) | 25.0% | 5/28 | 4 |
| Stock (exact scorer) | `2026-09-24-lfm-heldout-stock-exact.md` | 22/41 | 1/13 (7.7%) | 100.0% | 26/28 | 0 |
| `a3` | `2026-09-24-lfm-heldout-a3.md` | 30/41 | 11/13 (84.6%) | 78.6% | 3/28 | 3 (1 wrong op + 2 wrong args) |
| `scorer-b1` (served / exact) | `2026-09-24-lfm-heldout-scorer-b1.md`, `-exact.md` | 27/41 | 11/13 (84.6%) | 68.8% | 4/28 | 1 |

The healed and quantized builds were not re-run on the held-out set; the
held-out figures above are `a3` and `scorer-b1` before healing and
quantization, and are still the best evidence available for those tracks'
held-out behaviour.

## Missing-candidate slice (the test side's 32 operation entries, gold operation removed, expected answer escalate)

| Model | Run id (file) | Abstain recall | FP tool calls | Wrong mutating |
|---|---|---|---|---|
| Stock | `2026-09-24-lfm-final-stock-missing-candidate.md` | 1/32 (3.1%) | 3/32 | 2 |
| `a3` | `2026-09-24-lfm-final-a3-missing-candidate.md` | 5/32 (15.6%) | 24/32 | 5 |
| `scorer-b1` (served / exact) | `2026-09-24-lfm-final-scorer-b1-missing-candidate.md`, `-exact.md` | 7/32 (21.9%) | 18/32 | 0 |

This is the weakest area for both tracks: with the gold operation removed,
both mostly pick a near-candidate operation instead of escalating. See
[Recommendation, item 1](#recommendation).

## Calibration (Track B and stock's exact scorer; Track A via `track_a_calibration.py`, d6)

| Model | Run id (file) | Side | ECE | Brier |
|---|---|---|---|---|
| Stock (exact) | `2026-09-24-lfm-final-stock-exact.md` | test | 0.167 | 0.834 |
| Stock (exact) | `2026-09-24-lfm-heldout-stock-exact.md` | held-out | 0.169 | 0.803 |
| `a3` (exact calibration, d6) | (`track_a_calibration.py --final`, recorded in the run log) | test | 0.080 | 0.156 |
| `a3` (exact calibration, d6) | (`track_a_calibration.py`, recorded in the run log) | held-out | 0.097 | 0.191 |
| `scorer-b1` (exact) | `2026-09-24-lfm-final-scorer-b1-exact.md` | test | 0.132 | 0.268 |
| `scorer-b1` (exact) | `2026-09-24-lfm-heldout-scorer-b1-exact.md` | held-out | 0.145 | 0.312 |
| `scorer-b1` (exact) | `2026-09-24-lfm-final-scorer-b1-missing-candidate-exact.md` | slice | 0.547 | 1.198 |

**Quantized-scorer calibration is not measurable at all (d17, corrected by
lapse l6):** exact in-process scoring of a quantized scorer fails outright
(a `compressed-tensors` dependency gap for AWQ, and a shape mismatch in
`_dequantize` even with the right stack loaded), and the served route
(`llama-server`) also returns 0 of 64 complete label distributions for the
same structural reason vLLM's served scorer does (r8: it asks for only
`len(labels) + TOP_MARGIN` top log-probabilities). There is currently no
calibration figure for `scorer-b1.awq` or `scorer-b1.q4_k_m`.

## Latency and memory

Every latency figure below is each page's own warm median / p95, on its
own device (test side, 2K, unless marked). Every memory figure states
which page or sampler it came from — several are simply not measured, and
are shown as such rather than guessed.

| Build | Run id (file) | Warm latency | Memory |
|---|---|---|---|
| Stock (generative) | `2026-09-24-lfm-final-stock.md` | 803 ms / p95 2,458 ms | not measured (`docker stats`, attach mode) |
| Stock (exact scorer) | `2026-09-24-lfm-final-stock-exact.md` | 82 ms / p95 86 ms | not measured |
| `a3` (bf16) | `2026-09-24-lfm-final-a3.md` | 442 ms / p95 716 ms | not measured |
| `a3.awq` | `2026-09-24-lfm-final-a3.awq.md` | 295 ms / p95 518 ms | not measured |
| `a3.q4_k_m` | `2026-09-24-lfm-final-a3.q4_k_m.md` | 244 ms / p95 398 ms, cold 353 ms | about 0.65-0.80 GB resident + about 0.83 GB GPU (829 MiB in this page's own `nvidia-smi` snapshot); process-level, not the `docker stats` field, since a native `llama-server` has no container |
| `a3-heal` (bf16) | `2026-09-24-lfm-final-a3-heal.md` | 430 ms / p95 687 ms | not measured |
| `a3-heal.awq` | `2026-09-24-lfm-final-a3-heal.awq.md` | 296 ms / p95 528 ms | not measured |
| **`a3-heal.q4_k_m` (shipped, Track A)** | `2026-09-24-lfm-final-a3-heal.q4_k_m.md` | **245 ms / p95 405 ms** | about 0.83 GB GPU (829 MiB in this page's own `nvidia-smi` snapshot, matching `a3.q4_k_m`'s); process-level, not the `docker stats` field |
| `scorer-b1` (served) | `2026-09-24-lfm-final-scorer-b1.md` | 23 ms / p95 26 ms | not measured |
| `scorer-b1.awq` | `2026-09-24-lfm-final-scorer-b1.awq.md` | 22 ms / p95 26 ms | not measured |
| **`scorer-b1.q4_k_m` (shipped, Track B)** | `2026-09-24-lfm-final-scorer-b1.q4_k_m.md` | **33 ms / p95 36 ms** | not measured |

`a3.q4_k_m`'s and `a3-heal.q4_k_m`'s memory readings are the only
process-level memory samples taken on the training-side runs; every vLLM
attach-mode run above (stock, `a3` bf16/AWQ, `scorer-b1` bf16/AWQ) has no
`docker stats` reading, since attach mode measures against an
already-running server rather than one the harness itself started and
could sample.

## 2K vs 4K (validation side only — not test)

| Context | Run | Right proposals | Wrong mutating | Warm latency |
|---|---|---|---|---|
| 2K | `a3`, validation | 31/32 | 0 | 400 ms |
| 4K | `a3`, validation | 29/32 | 0 | 440 ms / p95 653 ms |

This check ran on the **validation** side only, as an optional t22 check
that `a3` holds up at a longer context (`MEASURE_GPU_FRACTION=0.12`, ledger
P65) — it is not a test-side or final-run result, and the shipped final
run stays at 2K throughout, per the original plan.

## Edge check (an AGX Orin, JetPack R39, validation side)

| Build | Mode | Run id (file) | Right proposals | Abstain recall | Wrong mutating | Warm latency | Peak memory |
|---|---|---|---|---|---|---|---|
| `a3-heal.q4_k_m` | GPU | `2026-09-24-lfm-edge-orin-a3-heal.q4_k_m-gpu.md` | 32/32 | 13/16 (81.2%) | 1 | 539 ms / p95 956 ms, cold 902 ms | about 1.75 GiB (sampled separately; the page's own `docker stats` field reads "not measured") |
| `scorer-b1.q4_k_m` (served) | GPU | `2026-09-24-lfm-edge-orin-scorer-b1.q4_k_m-gpu.md` | 28/32 | 12/16 (75.0%) | 0 (4 not grounded) | 109 ms / p95 111 ms | about 1.58 GiB (sampled separately) |
| `a3-heal.q4_k_m` | CPU-only (4 CPUs, 4 GB, `-ngl 0`) | `2026-09-24-lfm-edge-orin-a3-heal.q4_k_m-cpu4-4g.md` | 32/32 | 13/16 (81.2%) | 1 | 2,852 ms / p95 4,177 ms | about 1.12 GiB (sampled separately) |

This runs on the **validation** side, not test — the test side is never
re-exposed after t24. Decisions are identical between the GPU and
CPU-only runs, and match the same checkpoints' own validation-side
behaviour measured on the training machine. The device is about 2x slower
than the training machine for Track A (539 ms here against about
245-296 ms there), while Track B's scorer stays fast everywhere (109 ms
here). Every build measured here fits well under 6 GB, GPU and CPU-only
alike.

## Jetson skills (104 entries, `MEASURE_CTX=8192`, CC-BY-4.0 test set — see [Licensing and the skills-eval boundary](#licensing-and-the-skills-eval-boundary))

| Model | Run id (file) | Overall | Skill named | Not named |
|---|---|---|---|---|
| Stock | `2026-09-24-skills-q46-stock.md` | 42/104 (40%) | 14/34 (41%) | 28/70 (40%) |
| `a3` | `2026-09-24-skills-q46-a3.md` | 44/104 (42%) | 10/34 (29%) | 34/70 (49%) |

d12's regression guard (a floor of stock's score minus 5 points, both
overall and not-named) is met: `a3` clears the overall floor (37/104) and
the not-named floor (23/70), with the gain concentrated in not-named
prompts. `scorer-b1` was not run through the skills eval (the skills
harness measures a generative tool-caller, not a scorer).

## Issue 39, for context (LFM2.5-350M, quoted as an upper bound)

| Model | Run id (file) | Right proposals | Abstain recall | Wrong mutating | Warm latency | Container memory |
|---|---|---|---|---|---|---|
| Stock LFM2.5-350M | `2026-09-23-lfm-final-r8.md` | 0/32 | 0/15 | 0 | 290 ms / p95 797 ms | 3.5 GiB |
| Tuned LFM2.5-350M (r8) | `2026-09-23-lfm-final-r8.md` | 28/32 | 13/15 | 1 | 119 ms / p95 533 ms | 3.6 GiB |

**These tuned-model figures are an upper bound (lapse l5),** quoted exactly
as the page's own note states: 4 of the 64 test entries had reached r8's
training by exact wording (one supplement entry, three generated
variations) before this was caught. They are shown here only for scale —
a different base model (LFM2.5-350M vs Qwen3.5-0.8B), a different split
seed (39 vs this run's 46) and a different measurement harness generation
— not as a like-for-like comparison with the numbers above.

## Success bars (c33-c36), per shipped build

Judged on the test side, against stock re-measured there (c33: at least
80% right and at least stock+30pp; c34: abstain recall/precision at least
80%, false-positive tool calls at most 5%, 0 wrong mutating; c35, Track B
only: ECE at most 0.10, Brier below stock's; c36: warm median latency at
most 250 ms and container memory at most 6 GB, both measured on the
training machine at 2K).

| Bar | `a3-heal.q4_k_m` (Track A, shipped) | `scorer-b1.q4_k_m` (Track B, shipped) |
|---|---|---|
| c33, right proposals | **PASS** — 100% (32/32), stock+94pp | **PASS** — 84.4% (27/32), stock+78pp |
| c34, abstain recall (≥80%) | FAIL — 73.3% (11/15) | FAIL — 73.3% (11/15) |
| c34, abstain precision (≥80%, strict) | PASS — 91.7% | PASS — 91.7% |
| c34, FP tool calls (≤5%) | FAIL — 12.5% (4/32) | FAIL — 6.3% (2/32) |
| c34, 0 wrong mutating | FAIL — 2 (same ids as `a3` throughout, including after healing) | **PASS** — 0 |
| c35, Track B ECE (≤0.10) | — | FAIL — 0.132 (exact) |
| c35, Track B Brier (< stock's 0.834) | — | PASS — 0.268 (exact) |
| c36, warm median latency (≤250 ms) | **PASS** — 245 ms (`a3-heal.q4_k_m`) | **PASS** — 33 ms |
| c36, container memory (≤6 GB) | **PASS** — about 0.83 GB GPU, `a3-heal.q4_k_m`'s own `nvidia-smi` snapshot (process-level, not `docker stats`) | not measured (attach mode; no process-level sample was taken for a served vLLM scorer) |
| c43, quantized build vs its own bf16 | **PASS** (after heal) — 0 points lost, no new wrong-mutating id | PASS (unhealed) — `+3.125` points on `.awq`, `0.00` on `.q4_k_m` |

**Neither shipped build clears every bar.** Both fail abstention recall and
false-positive tool calls on the test side; `a3-heal.q4_k_m` also fails
0-wrong-mutating (unchanged by healing, since the heal only recovered the
quantization-induced loss, not `a3`'s own pre-existing errors);
`scorer-b1.q4_k_m` also fails c35's ECE bar despite a Brier score far below
stock's.

## c43 quantization deltas, all builds (right proposals against each model's own bf16 checkpoint)

| Build | bf16 right proposals | Quantized right proposals | Delta | `heal_needed()` |
|---|---|---|---|---|
| `a3.awq` | 32/32 | 31/32 | -3.125 pts | TRUE (no new wrong-mutating id) |
| `a3.q4_k_m` | 32/32 | 31/32 | -3.125 pts | TRUE (no new wrong-mutating id) |
| `a3-heal.awq` | 32/32 (`a3-heal`) | 30/32 | -6.25 pts | still fails c43; not shipped (c42 allows one heal round) |
| `a3-heal.q4_k_m` | 32/32 (`a3-heal`) | 32/32 | 0.00 pts | FALSE — cleared |
| `scorer-b1.awq` | 27/32 | 28/32 | +3.125 pts | FALSE |
| `scorer-b1.q4_k_m` | 27/32 | 27/32 | 0.00 pts | FALSE |

## Licensing and the skills-eval boundary

The Jetson skill-routing evaluation set is CC-BY-4.0 and is used here as a
**test-only** measurement: nothing in this run trains on it, and it never
enters the training or augmentation pipeline. If a skills-tuned model
(one that *did* train on this data) is ever published, it would need
CC-BY-4.0 attribution to the evaluation set's source; no such model exists
from this run.

## Recommendation

### Data-set recommendations for the next Tool-Jev iteration

Grounded per the rule this run followed throughout: validation supports
per-entry evidence (individual request texts and their errors); test and
held-out support only aggregate counts, since their individual entries are
not read or quoted once they are the measurement side.

1. **Missing-candidate training examples.** The missing-candidate slice is
   the weakest area for both tracks (`a3` 5/32 escalations, `scorer-b1`
   7/32), and the frozen training set has 0 entries with a reduced
   candidate list — nothing in training resembles this slice's shape.
   Build train-side slices deterministically with `eval_slices.py` (the
   gold operation removed, expected answer escalate); no teacher call is
   needed, since the transformation is mechanical. Track B is likely to
   benefit most, since its candidate list is explicit in its own prompt.
2. **Missing-argument requests should escalate.** Validation: "Is the
   service running?" (no service named) got `a3` proposing
   `service_status` with an invented `docker.service` argument. Strip a
   required argument from existing train-side operation requests
   (rule-based, no teacher needed) and pair the stripped version with its
   complete original. The same error class dominated issue 39's known
   gaps.
3. **Diagnosis vs. explanation boundary.** Validation: "Explain why nginx
   returns 502" (expected escalate) was explained instead. Add contrastive
   pairs distinguishing "what does X mean" (explain — a definitional
   question) from "why is X happening to my system" (escalate — a
   diagnosis request), since the surface wording is close but the correct
   outcome differs.
4. **Over-caution on clear mutating requests.** Validation: "Can you
   switch to balanced mode?" and "nvpmodel low power" were both escalated
   instead of proposing `power_set`. Add explicit-mode `power_set`
   positive examples per mode synonym, and check the power-related
   escalation balance in the current frozen set (91 `power_set` training
   examples today) against how often power requests should escalate versus
   propose.
5. **Ambiguous operation names.** Validation: "docker status" proposed
   `container_list` where `service_status docker.service` was expected.
   Add disambiguation pairs for names that plausibly match more than one
   operation, and re-check whether the gold label itself is the more
   natural reading.
6. **Hard negatives for false positives.** Test-side false positives
   (`a3` 4/32, `scorer-b1` 2/32, aggregate only, per the rule above): add
   requests that name an operation's own keyword in passing but want an
   explanation, not the operation itself, as explicit non-mutating
   negatives.

**Process for generating this data:** the same pipeline as this run — the
same Apache-2.0 teachers, reviewer B's clean-slate review, and
`leakage_check.py` before assembly. Two side notes for the next run: this
run's **test side is now spent** for any future final-run claim (it has
been read in aggregate and is committed to benchmark pages), so the next
iteration needs a **fresh sealed held-out set and test side**; and
**validation should grow to about 150-200 entries** — this run's 66-entry
validation set never surfaced the rare wrong-mutating proposals that only
showed up on the (also small) 32-entry test side, and a larger validation
set would catch more of that before the single final run.

### Overall recommendation

**Adopt `scorer-b1.q4_k_m` (Track B) as the safer Tier 2 decider**: 0 wrong
mutating proposals on the test side, 33 ms warm on the training machine,
109 ms on an AGX Orin — the only shipped build that clears c34's
0-wrong-mutating bar, and the fastest by a wide margin everywhere it was
measured. **Keep `a3-heal.q4_k_m` (Track A) as the generative option**,
pending a fix for its wrong-mutating proposals (recommendations 2 and 4
above target exactly its two recurring error ids) — it offers the highest
raw accuracy (100% right proposals on the test side) and clears every
other bar it was checked against, but should not replace `scorer-b1` as
the default decider until its mutating-proposal errors are addressed.
**Neither checkpoint is a full replacement for a human-reviewed decision
yet**; both still fail abstention recall and false-positive tool calls on
the test side, and the missing-candidate slice remains the clearest sign
that more training data — not a different architecture or a longer
context — is this run's highest-leverage next step.
