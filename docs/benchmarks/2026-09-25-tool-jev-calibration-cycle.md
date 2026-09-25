# Tool-Jev calibration cycle report (issue 53)

This report closes the measurement side of
[issue #53](https://github.com/agentculture/nvsh/issues/53). It is written for
someone who has not followed issue 46: it names the artifact, how to read its
output, the thresholds it ships with, the slices it was measured on, and every
figure with its n and 95% bootstrap confidence interval. The full decision
path, ledger and reproduce steps are in
[`docs/qwen-tool-jev-calibration.md`](../qwen-tool-jev-calibration.md); the
domain-general method is
[`docs/scorer-finetune-playbook.md`](../scorer-finetune-playbook.md).

## The artifact

- **Model:** `scorer-r3b`, a LoRA fine-tune of `Qwen/Qwen3.5-0.8B` (revision
  `2fc06364715b967f1860aea9cf38778875588b17`), merged, served as a **`Q4_K_M`
  GGUF** through `llama-server`. It is a Jev-style **candidate scorer**
  (Track B): the prompt lists each offered candidate under a one-letter label
  (nvsh's operations, `explain`, `escalate`), and the next-token probability
  mass over those letters (summed over each letter's token variants,
  normalised over the offered letters) is the decision. It generates no text;
  arguments come from nvsh's deterministic grounding.
- **Calibration (frozen, fitted on the validation fit fold of this build):**
  temperature **1.5366**, no per-label vector (the vector made the selection
  fold worse).
- **Gate (frozen):** on the calibrated distribution, escalate or explain when
  that is the top candidate; for a read-only proposal, abstain
  (`abstain_uncertain`) when the top-1/top-2 margin is below **0.2**; no
  mutating-specific threshold (none was needed for 0 wrong mutating on the
  fit fold). Thresholds key on `Operation.read_only`, never an operation name.
- **Where:** private Hugging Face repositories
  `jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev-scorer-v2-gguf` (the build plus
  `calibration.json` and `gate.json`), `...-tool-jev-scorer-v2` (bf16) and
  `...-tool-jev-v2-dataset` (the data). nvsh's runtime does not load it yet;
  runtime wiring is [#54](https://github.com/agentculture/nvsh/issues/54).

## The sides and slices

- **Validation (204):** fresh teacher-drafted requests, split into a seeded
  fit fold (143, for calibration and gate fits) and selection fold (61, for
  choosing the checkpoint).
- **Test (198):** fresh teacher-drafted requests, measured once.
- **Sealed held-out (149):** drafted separately and sealed before training;
  nobody on the development side read its text; measured once.
- **Missing-candidate slices (test 83, held-out 60):** each operation entry
  with its correct operation removed from the candidates; the right answer is
  escalate or abstain.
- **Per-slice calibration:** by the gold answer's kind: `read_only` and
  `mutating` operations, and `escalate_or_explain`.
- The issue-46 test side was spent and is used for no claim here.

## Every announcement clause, before and after

Before is `scorer-b1` (PR #52; sources: `docs/benchmarks/2026-09-24-*scorer-b1*`,
the issue-46 delivery record, and this cycle's t17 baseline on v2 validation).
After is `scorer-r3b.q4_k_m` with the frozen calibration and gate, measured
once on the fresh sides on a DGX Spark and on an AGX Orin.

| Clause | Bar | Before (`scorer-b1`) | After, test (n=198) | After, held-out (n=149) | Met |
|---|---|---|---|---|---|
| Label-permutation robust | pooled answer change <= 2.3% | 18.8% (v2 validation, t17) | **2.07%** (205 / 9900 trials, 10 per entry and kind) | not measured | yes |
| Calibrated, deployed `Q4_K_M` | ECE <= 0.10 | none served (0 / 64 complete readouts); exact bf16 0.132 | **0.016** [0.011, 0.042], Orin 0.009 [0.009, 0.036] | **0.044** [0.024, 0.092], Orin 0.046 [0.023, 0.090] | yes |
| Complete served readouts | >= 95% | 0 / 64 | **198 / 198** | **149 / 149** | yes |
| No wrong mutating action | 0 | 0 on test, 1 on its v2-validation `Q4_K_M` | **0** (and 0 on the missing-candidate slice) | **0** (and 0 on the missing-candidate slice) | yes |
| Missing-candidate escalation | >= 80% | 7 / 32 (21.9%) | **85.5%** (71 / 83), Orin 84.3% | **76.7%** (46 / 60), Orin 76.7% | **no (held-out)** |
| Uncertainty gate separate from escalate | separate outcome | bare argmax | `abstain_uncertain` counted apart (test 0, held-out 1) | | yes |

Permutation kinds on test (rate, 95% CI over entries): order 2.5% [0.8,
4.3], letters 1.0% [0.1, 2.1], subset 1.6% [0.4, 3.1], paraphrase 2.5%
[0.8, 4.6], all together 2.8% [1.3, 4.6]. The probe ran in process on the
merged weights, since the runtime order is never canonicalised; the pooled
figure has no single CI because trials of one entry are correlated across
kinds.

**The missed bar.** On the held-out missing-candidate slice, 14 of 60
requests were not escalated; the model proposed a related read-only
operation instead (for example a GPU check when the memory check was
removed). None of them was a mutating action. The slice was also
understated on validation (73.8% gate off), and the gate fitted on the fit
fold reached exactly 80% there; a stricter escalate threshold was considered
after the selection fold had been seen and not taken (guide D47).

## Other figures on the fresh sides

| Figure | Test (Spark) | Held-out (Spark) |
|---|---|---|
| Right proposals | 79 / 83 (95.2%) [90.4, 98.8] | 49 / 60 (81.7%) [71.7, 90.0] |
| Escalation recall / precision (gated) | 94.9% / 98.7% | 97.1% / 85.0% |
| False-positive tool calls (raw) | 1 / 115 | 0 / 89 |
| Invalid (not grounded) | 3 / 198 | 2 / 149 |
| Brier (calibrated) | 0.049 | 0.135 |
| Warm decision latency | 124 ms (Spark), 355 ms (Orin) | 120 ms (Spark), 355 ms (Orin) |

Readout fidelity on validation (the same weights, served vs in process): the
bf16 GGUF matched within 0.01 on 196 of 204 entries (max 0.072, top-1
agreement 203 / 204), so the 0.01 target is missed on 8; `Q4_K_M` differs by
more than 0.05 on 10 entries (top-1 agreement 201 / 204), which is why its
temperature and gate are fitted on its own predictions.

## Per-slice reliability (calibrated `Q4_K_M`)

The held-out's `read_only` slice is **not** calibrated to the bar (ECE
0.147 [0.079, 0.271], n=49): its reliability bins show confidence above
accuracy. The side-level ECE (0.044) hides it, because the escalate/explain
slice is very well calibrated (0.022). nvsh should threshold read-only
proposals with that in mind (#54).

### Test (198), calibrated `Q4_K_M`, Spark

- `read_only`: n=71, ECE 0.030 [0.009, 0.074], Brier 0.065
- `mutating`: n=12, ECE 0.033 [0.014, 0.057], Brier 0.004
- `escalate_or_explain`: n=115, ECE 0.032 [0.015, 0.058], Brier 0.045

#### Slice `read_only`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 1 | 0.365 | 0.000 |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 1 | 0.755 | 1.000 |
| [0.8, 0.9) | 2 | 0.842 | 1.000 |
| [0.9, 1.0) | 67 | 0.988 | 0.970 |

#### Slice `mutating`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 0 | - | - |
| [0.8, 0.9) | 1 | 0.866 | 1.000 |
| [0.9, 1.0) | 11 | 0.976 | 1.000 |

#### Slice `escalate_or_explain`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 2 | 0.573 | 0.000 |
| [0.6, 0.7) | 1 | 0.693 | 1.000 |
| [0.7, 0.8) | 3 | 0.796 | 1.000 |
| [0.8, 0.9) | 1 | 0.823 | 0.000 |
| [0.9, 1.0) | 108 | 0.983 | 0.991 |

### Sealed held-out (149), calibrated `Q4_K_M`, Spark

- `read_only`: n=49, ECE 0.147 [0.079, 0.271], Brier 0.362
- `mutating`: n=11, ECE 0.097 [0.036, 0.172], Brier 0.048
- `escalate_or_explain`: n=89, ECE 0.022 [0.010, 0.046], Brier 0.021

#### Slice `read_only`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 1 | 0.384 | 1.000 |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 2 | 0.573 | 0.500 |
| [0.6, 0.7) | 2 | 0.676 | 0.500 |
| [0.7, 0.8) | 2 | 0.716 | 0.500 |
| [0.8, 0.9) | 5 | 0.867 | 0.600 |
| [0.9, 1.0) | 37 | 0.981 | 0.865 |

#### Slice `mutating`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 1 | 0.544 | 1.000 |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 1 | 0.797 | 1.000 |
| [0.8, 0.9) | 2 | 0.888 | 1.000 |
| [0.9, 1.0) | 7 | 0.974 | 1.000 |

#### Slice `escalate_or_explain`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 2 | 0.622 | 1.000 |
| [0.7, 0.8) | 2 | 0.736 | 0.500 |
| [0.8, 0.9) | 0 | - | - |
| [0.9, 1.0) | 85 | 0.991 | 1.000 |

### Test missing-candidate slice (83), calibrated `Q4_K_M`, Spark

- `escalate_or_explain`: n=83, ECE 0.090 [0.049, 0.176], Brier 0.275

#### Slice `read_only`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 0 | - | - |
| [0.8, 0.9) | 0 | - | - |
| [0.9, 1.0) | 0 | - | - |

#### Slice `mutating`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 0 | - | - |
| [0.8, 0.9) | 0 | - | - |
| [0.9, 1.0) | 0 | - | - |

#### Slice `escalate_or_explain`

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 5 | 0.460 | 0.600 |
| [0.5, 0.6) | 3 | 0.541 | 0.667 |
| [0.6, 0.7) | 5 | 0.637 | 0.600 |
| [0.7, 0.8) | 2 | 0.756 | 0.000 |
| [0.8, 0.9) | 9 | 0.870 | 0.667 |
| [0.9, 1.0) | 59 | 0.964 | 0.915 |

## How the checkpoint was chosen

Four candidates were pre-registered (`docs/tool-jev-calibration-rule.md`,
before any training): r1 (randomized labels), r2 (+8 escalation reasons), r3
(r1 at lr 1e-4), and a conditional r4 (never triggered). Measured on the
validation selection fold:

| Candidate | Right (selection, fitted gate) | Wrong mutating, gate off (validation) | Pooled permutation | ECE after temperature | Outcome |
|---|---|---|---|---|---|
| r1 | 20 / 24 | 0 | 5.0% | 0.082 | passed; lost at step 3 to r3b |
| r2 | 18 / 24 | 4 | 9.2% | 0.028 | out: below the 78.3% floor; reasons dropped |
| r3 | 21 / 24 | 3 | 4.2% | 0.020 | out under d6: confident `power_set` on check-then-change requests |
| **r3b** | 21 / 24 | **0** | **3.1%** | 0.032 | **chosen** |

Recorded departures from the plan, all operator-approved: d6 added a
safety-first filter (0 wrong mutating on all of validation, gate off) before
the calibration step; d7 drafted 76 check-then-change training pairs and
retrained r3's recipe as r3b. d4 (train on every `dev.json` entry) and d5
(compare a service argument as grounding matches it) changed the data and the
metric; every figure above uses them.

## OpenJev comparison

Limited to what OpenJev publishes
([openjev/openjev](https://huggingface.co/openjev/openjev)): answer change
under candidate order shuffle of about 2.3% for its model and 18.5% for its
base model. `scorer-r3b` changes its answer in 2.5% of order shuffles and
2.07% pooled over order, letters, subsets and paraphrases; `scorer-b1` was
at 18.8% pooled. OpenJev publishes no calibration figure (it ships a fixed
readout temperature) and no abstention mechanism, so the calibration and
escalation bars are nvsh's own.

## Follow-ups

- [#54](https://github.com/agentculture/nvsh/issues/54): wire the scorer and
  gate into nvsh's runtime; use this report's thresholds and slices.
- [#55](https://github.com/agentculture/nvsh/issues/55): Choice/Noul/Score
  primitives.
- [#56](https://github.com/agentculture/nvsh/issues/56): Track A follow-ups.
- [#57](https://github.com/agentculture/nvsh/issues/57): a failed start-up
  blocking a re-run (fixed in this cycle).
- [#58](https://github.com/agentculture/nvsh/issues/58): a concurrent
  stop/start on one measuring port can kill the new server.
- [#59](https://github.com/agentculture/nvsh/issues/59): promote corpus v2
  into `nvsh/tiers/corpus`, re-baseline the tiers, retire the public held-out.
- [#60](https://github.com/agentculture/nvsh/issues/60): close out issue
  46's records and decide per-repository Hub visibility.
- [#62](https://github.com/agentculture/nvsh/issues/62): a domain-module
  setting for the pipeline.
- [#64](https://github.com/agentculture/nvsh/issues/64): DeepEval evaluation
  of the model and the harness, starting from the private repositories above.
- The held-out missing-candidate miss is the next data target: more
  train-side requests whose right operation is absent from the candidates.
