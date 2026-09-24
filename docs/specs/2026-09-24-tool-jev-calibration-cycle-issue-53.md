# Tool-Jev calibration cycle (issue 53)

> The Tool-Jev scorer now makes calibrated, label-permutation-robust decisions on its deployed quantized build: an improvement cycle over code and data measured the permutation dependence of scorer-b1, added an uncertainty gate separate from semantic escalate, trained with per-example randomized candidate labels on a grown dataset, and reports reliability per slice for the actual shipped artifact (issue #53)

## Audience

- The nvsh operator deciding whether a local Tool-Jev scorer can make bounded routing decisions it can threshold, and the next implementer of #54 who needs a measured artifact and measured thresholds.

## Before → After

- Before: scorer-b1 (PR #52) picks by bare argmax over labels fixed to table order; nobody knows whether it evaluates the supplied candidates or memorised letters; its exact ECE is 0.132 (bar 0.10), it escalates on 7/32 missing-candidate entries, and the shipped `Q4_K_M` has no calibration measurement at all because the served readout returns 0/64 complete distributions.
- After: A retrained Track B scorer whose choice survives label/order permutation, whose served `Q4_K_M` returns complete label distributions that are calibrated after a fitted temperature, with separate semantic-escalate and uncertainty-abstain outcomes, thresholds measured per read-only/mutating, and every result reported per slice with confidence intervals on fresh evaluation sides.

## Why it matters

- nvsh only acts safely on a small model's decision if a probability means what it says and the decision tracks the offered alternatives: that is the difference between a tiny tool classifier and a local decision primitive nvsh can threshold (issue #53 design note).

## Requirements

- Permutation probe of the existing scorer-b1 before any retraining: score the val side with (a) shuffled candidate order, (b) a random op->letter permutation, (c) candidate subsets, (d) paraphrased candidate descriptions, and report how often the semantic choice changes (OpenJev reports ~2.3% on order shuffle). Needs a new seam: labels are fixed by `LABEL_ALPHABET`\[full.index(name)\] (scripts/lfm-finetune/scorer.py:139-153) and descriptions are hard-wired to `ops_table` (scorer.py:174-181); no shuffle/permute hook exists today.
  - instruction: scorer.py gains a seeded permutation/paraphrase option; a test asserts letters remap and the argmax op is compared at the op level, not the letter.
  - honesty: The probe runs on the existing scorer-b1 checkpoint before any new training and reports answer-change rate per perturbation (order, letters, subset, paraphrase) on val.
- Uncertainty-abstention gate kept separate from semantic escalate: escalate stays an ordinary candidate label (scorer.py:64), and a policy over the distribution (`p_escalate`, `p_top1` floor, top1-top2 margin, normalized entropy) decides `abstain_uncertain`; today the decision is a bare argmax (scorer.py:363). Thresholds are chosen separately for read-only vs mutating ops using Operation.`read_only` (nvsh/ops/table.py), never an operation name.
  - instruction: grep the gate for operation names: none; metrics count escalate and `abstain_uncertain` separately.
  - honesty: Escalate stays a candidate label and `abstain_uncertain` is a separate outcome in predictions and metrics; the gate code references Operation.`read_only` only.
- Threshold sweeps and calibration fits run offline from stored per-example distributions: the in-process ('exact') predictions.jsonl files already hold the full label->probability dict for every entry (work/q46/final/scorer-b1/\*-exact-\*.predictions.jsonl: test 64, missing-candidate 32, held-out 69), so no model re-run is needed to sweep gates on existing checkpoints; fits use val.json only (measure.py `check_split_allowed` refuses test without --final, held-out without --acceptance).
  - honesty: Gate sweeps on scorer-b1 are reproducible from stored predictions.jsonl without a GPU; thresholds are chosen on val only.
- Post-hoc calibration (temperature scaling first, then vector scaling) fitted on val and applied to each line's candidates before `top_candidate`/`brier_one` (scripts/lfm-finetune/metrics.py:334-402); no fit/apply step exists today.
  - honesty: The fitted temperature (and vector) is fitted on val only, stored with the run, and applied identically to test/held-out and the served build.
- Per-slice calibration reporting: ECE/Brier and reliability tables split by read-only vs mutating, candidate count, missing-candidate/OOD slice and confidence bucket. metrics.py already computes 10-bin {n, confidence, accuracy} (`ece_bins`, metrics.py:347-363) but only over a whole file, and the committed docs/benchmarks/\*.md render only the two scalars.
  - honesty: Per-slice reports render the reliability bins in the committed benchmark markdown, not only the JSON.
- A complete-label readout for served builds: today every served scorer run (bf16 vLLM, AWQ vLLM, `Q4_K_M` llama-server on Spark and Orin) returns 0/64 complete distributions because label tokens fall outside the top len(labels)+`TOP_MARGIN` logprobs (scorer.py:245-256, nvsh/tiers/toolchat.py:336-350, docs/benchmarks/2026-09-24-lfm-final-scorer-b1.md). The gap is the readout path, not quantization.
  - honesty: A served run of the deployed build yields complete distributions for >= 95% of entries (0/64 today), and incomplete ones are still counted, never renormalised.
- Calibration is measured on the deployed artifact (scorer `Q4_K_M` GGUF via llama-server on DGX Spark, and on AGX Orin), not only on bf16: no ECE/Brier exists today for AWQ or `Q4_K_M` (deviation d17, lapse l6 in docs/deliveries/2026-09-23-qwen3-5-0-8b-tool-jev-fine-tune-issue-46.md).
  - honesty: Calibration numbers exist for the `Q4_K_M` build measured through the served path on DGX Spark and on AGX Orin.
- Training variant with per-example randomized candidate order, op->letter mapping and candidate subsets. `train_scorer.py` today uses one fixed `label_ids` list for the whole run (`train_scorer.py`:297) and encode() always passes offered=None (`train_scorer.py`:130-143), so label ids and the CE target index must become per-row.
  - instruction: unit test: two rows with different letter maps produce different target indices for the same gold op.
  - honesty: Training randomises order, letters and subsets per example with a recorded seed; the label ids and CE target are derived per row from the rendered prompt.
- Dataset cycle: train-side missing-candidate / no-valid-option examples (the frozen 1,463-example set has 0 entries with a reduced candidate list, docs/benchmarks/2026-09-24-qwen-tool-jev-comparison.md:224-229), built deterministically from `eval_slices.py`'s shape, plus the comparison doc's other five recommendations (missing-argument escalates, diagnosis-vs-explanation pairs, explicit-mode `power_set` positives, disambiguation pairs, hard negatives), through the same all-Apache augment.py teachers, reviewer B and `leakage_check.py`.
  - honesty: New train-side data goes through the all-Apache teachers, reviewer B and `leakage_check.py` against every eval side, and is frozen before the final run (post-freeze changes are deviations).
- Candidate sets and label assignments are stored per training example at dataset build time (`build_dataset.py` / `merge_variations.py`), not invented at collate time: corpus entries carry no candidates field today, and label token ids and the CE target are keyed off whatever list encode() renders.
  - honesty: Every training example records its offered candidates and letter map, so a run can be replayed exactly.
- Fresh evaluation sides: the issue-46 test side is spent (comparison doc next-iteration notes), so this cycle needs a new sealed test side and held-out, and validation grown from 66 to ~150-200 entries so threshold and temperature fits are not driven by single entries (1/15 escalate = 6.7 pts).
  - honesty: The new test side and held-out are sealed before any training of this cycle; validation has >= 150 entries; the issue-46 test side is not used for any claim.
- Results carry uncertainty: with n=64 test and n=32 missing-candidate, ECE/recall move 3-7 pts per entry, so reports include bootstrap confidence intervals and reliability diagrams, not just point ECE/Brier.
  - honesty: Every reported rate and ECE carries a bootstrap CI and n.
- Documentation distinguishes Track A (specialized generative tool router; its distribution is reconstructed offline by `track_a_calibration.py` teacher forcing) from Track B (Jev-style candidate decision/scoring model) so the two are not described as equally Jev-like (issue #53 gap 4).
  - honesty: Guide, comparison doc and model cards call Track A a generative tool router and Track B the Jev-style scorer.
- Out-of-scope items get their own issue rather than a silent non-goal: runtime wiring -> #54, Choice/Noul/Score -> #55, Track A follow-ups -> #56 (operator 2026-09-25).
  - honesty: Issues #54, #55, #56 exist and are linked from #53 and the spec.
- Conditional calibration-aware training stage (our own objective, not RLCD, whose method TypeSafe has not published): if post-hoc temperature/vector scaling plus the new data does not reach the calibration bar, train with a calibration-aware objective — label smoothing / a proper-scoring-rule term (Brier) beside CE, and/or a cost-sensitive decision reward on the gate's act/abstain outcome — and compare against the post-hoc result.
  - honesty: The calibration-aware stage runs only if the post-hoc + data result misses the ECE bar, and its result is compared head-to-head with post-hoc on val.
- Served readout completeness comes from a large top-k, not `logit_bias`: raise the scorer's requested top (`TOP_MARGIN` in scripts/lfm-finetune/scorer.py) and the server cap (`MEASURE_MAX_LOGPROBS` / vLLM --max-logprobs in `serve_for_measure.sh`) to >= 5000. Probe 2026-09-25 (llama-server, scorer-b1 `Q4_K_M` and bf16 GGUF, CPU, 6 spent-test prompts): top-22 missed 10-15 labels, top-400 missed 1 on 3/6, top-5000 was complete on 6/6 in ~0.4 s per request. ToolChat.`score_next_token` already takes top as a parameter, so no nvsh/ change is needed.
  - honesty: A served run with top >= 5000 returns complete distributions for every entry of validation on llama-server (Spark and Orin) and vLLM, or the incomplete count is reported.
- One label-probability definition for every path: in-process scoring reads one token id per label (`label_token_ids`, scorer.py:161-171) while served scoring sums token variants such as 'A' and ' A' (distribution, scorer.py:222-242). Probe: bf16 GGUF matched the in-process exact distribution within 1e-4 on 5/6 prompts, but dev-f02 differed by 0.137 in bf16 — a readout definition gap, not quantization. Pick one definition (train, in-process and served) before any calibration number is compared across paths.
  - honesty: The chosen definition is one function shared by `train_scorer`, the in-process scorer and the served scorer, with a test that both paths return the same distribution on a fixture.
- Calibration and readout fidelity are judged against the same artifact: the `Q4_K_M` distribution legitimately differs from bf16 (probe: max abs diff up to 0.096 on dev-e08 where the bf16 GGUF matched to 1e-4), so the temperature and thresholds are fitted on the deployed `Q4_K_M`'s own validation distributions, and h12's 'match bf16 exact within 0.01' check applies to the bf16 GGUF (readout fidelity), not to Q4.
  - honesty: The bf16 GGUF served readout matches the in-process exact distribution within 0.01 on validation; Q4 temperature/thresholds are fitted on Q4's own validation predictions.
- Validation is not double-dipped: the temperature fit, the gate thresholds and the checkpoint choice all come from validation, so validation is split into a fit fold and a selection fold (or cross-validated), fixed with a seed before the first run.
  - honesty: The fold assignment is seeded, recorded in the run, and no entry informs both the fit and the selection.
- A pre-registered checkpoint decision rule is written before any training run of this cycle, including how clean-test accuracy trades against permutation robustness and calibration (q7 risk; in #46, b1 beat the better-calibrated b4 only on the pre-registered rule — docs/qwen-tool-jev-finetune.md Track B runs).
  - honesty: The rule is committed (spec or plan) before the first training command of the cycle runs.
- The run keeps a live guide and ledger (code map, split, design, every issue with symptom -> cause -> fix, results, reproduce steps), maintained by a documentation subagent after every step, in the style of docs/qwen-tool-jev-finetune.md (operator standing rule for fine-tune runs since issue 46).
  - honesty: The guide shows an entry for every run, obstacle and fix within the same step it happened.
- New models and data never overwrite the PR #52 artifacts: scorer-b1 stays reproducible as the baseline, new builds go to new private repos or tagged revisions, and every bundle passes scripts/lfm-finetune/`scan_bundle.py` before upload (issue 46 c47/c48 lineage).
  - honesty: PR #52's six repos are byte-identical before and after the cycle; each new upload has a clean `scan_bundle` report.
- Escalation-reason candidates (if the c28 ablation keeps them) get descriptions from scripts-side data, not nvsh/tiers/lfm.py tools (scorer.`_description` reads lfm.`tools_for` for controls, scorer.py:174-181), and metrics roll every reason up to escalate so the missing-candidate and escalation bars stay comparable; candidate count stays within the single-token label check (`label_token_ids` refuses multi-token labels).
  - honesty: No reason description is read from nvsh/; metrics tests show reasons rolling up to escalate.

## Honesty conditions

- Every clause of the announcement is backed by a measured report on the fresh sides; any bar missed is reported as missed, not rounded up.
- No gate, report or data rule mentions a specific operation name.
- The lead reads held-out counts/hashes only; each final checkpoint is exposed to test/held-out once.
- Every Hub repo created in this cycle is private; none made public without a per-repo operator approval.
- The Track B path emits `max_tokens`=1 / one forward position; grounding and validation stay in deterministic code.
- The rate is measured on the fresh test side with at least 3 random permutations per entry, at the op level, with the runtime order not canonicalised.
- ECE and CI are computed from complete served distributions of the deployed `Q4_K_M` after the val-fitted temperature.
- Wrong-mutating is counted at the chosen thresholds on test and held-out; missing-candidate escalation counts semantic escalate plus uncertainty abstain.
- No file under nvsh/ changes in this cycle's PRs.
- The audience can read the final report without knowing issue 46's internals: it names the artifact, thresholds, slices and CIs.
- Every number in the before state is traceable to docs/benchmarks/2026-09-24-\*scorer-b1\* or the #46 delivery record.
- Each after-state property has a report line or chart on the fresh sides; none is claimed from bf16 alone when it concerns the deployed build.
- The gate and calibration are evaluated on decisions nvsh would act on (proposals and mutating ops), not only on aggregate accuracy.

## Success signals

- Permutation robustness at OpenJev level: shuffling candidate order and permuting op->letter labels changes the semantic choice in <= 2.3% of cases (OpenJev's published figure; its own base model was 18.5%, <https://huggingface.co/openjev/openjev>). Closed or explained with a cause other than missing data.
  - instruction: Report answer-change % with CI; compare with OpenJev 2.3% and scorer-b1's pre-retrain rate.
- Calibration on the deployed `Q4_K_M` scorer: ECE <= 0.10 with a bootstrap CI and a published reliability diagram, on the fresh test side. OpenJev publishes no ECE/Brier (it ships a fixed readout temperature, `READOUT_T`=0.85), so there is no OpenJev calibration number to match; the #46 bar c35 stands.
- 0 wrong mutating actions at the chosen gate thresholds, and escalation (semantic or uncertainty) on >= 80% of the missing-candidate slice (7/32 today); OpenJev reports no abstention mechanism, so these bars are nvsh's own (c34 lineage).

## Scope / boundaries

- No code switches on a specific operation name: gates and per-slice reports key on Operation.`read_only` and candidate count only (docs/tiers-improving-accuracy.md rule; nvsh/tiers/router.py:17-18).
- The sealed held-out is never read by the lead (counts and hashes only) and is exposed once per final checkpoint; test/held-out are never used for fitting thresholds or temperature.
- All data and model repos stay private on the Hub; making any repo public needs per-repo operator approval (issue 46 c48).
- No free-form generation on the Track B path; deterministic nvsh code keeps grounding, validation and execution (issue #53 success criteria).

## Non-goals

- A generic Choice/Noul/Score API is not built in this cycle; it moved to issue #55 (issue #53 gap 5, proposed order step 5).
- No change to the nvsh runtime's third-party dependency set (pyproject dependencies = \[\]); any runtime use of the scorer goes through the existing HTTP ToolChat.`score_next_token` pattern (nvsh/tiers/toolchat.py).

## Assumptions

- Lower learning rate is a calibration lever: b4 (lr 1e-4) made the same decisions as b1 with better validation calibration (exact ECE 0.072 vs 0.097, Brier 0.132 vs 0.162; docs/qwen-tool-jev-finetune.md Track B runs) and was only rejected on abstention precision.

## Scope exploration

- `s1` — `scripts/lfm-finetune/scorer.py`: Labels fixed by table order (`labels_for`, :139-153; offered subsets keep letters), descriptions hard-wired to `ops_table` (:174-181), decision is bare argmax (:363), served results missing labels are marked incomplete not renormalised (:245-256, :364-375). No permutation, paraphrase or gate seam exists.
  - seeds: `c2`, `c3`
- `s2` — `scripts/lfm-finetune/train_scorer.py`: One fixed `label_ids` list for the run (:297); encode() always offered=None (:130-143); CE target names.index(gold) (:142); only row order is shuffled (:224); no temperature parameter; plain softmax in evaluate (:251).
  - seeds: `c10`, `c12`
- `s3` — `scripts/lfm-finetune/metrics.py + measure.py`: ECE (10 bins) + multi-class Brier over a whole file (metrics.py:340-402), bins emitted only in metrics JSON; no per-slice split, no post-hoc fit. val.json is the only split free for fitting (measure.py:414-461). Wrong-mutating comes from Operation.`read_only` (metrics.py:297-301).
  - seeds: `c4`, `c5`, `c6`
- `s4` — `work/q46/final/scorer-b1 prediction files (outside the repo)`: Exact predictions.jsonl store the full 18-label distribution per entry (test 64, missing-candidate 32, held-out 69); the served and `q4_k_m` files carry candidates=null. Offline sweeps on scorer-b1 are possible without re-running; the files are not committed.
  - seeds: `c4`
- `s5` — `served readout: nvsh/tiers/toolchat.py:336-350, serve_for_measure.sh, docs/benchmarks scorer-b1 reports`: Served scoring POSTs /completions with logprobs=len(labels)+4 (22); every served run (bf16 vLLM, AWQ, `Q4_K_M` llama-server Spark + Orin) returned 0/64 complete distributions, so no ECE/Brier exists beyond bf16 in-process; AWQ in-process fails on compressed-tensors (d17, l6). No `logit_bias`/allowed-token path is used today.
  - seeds: `c7`, `c8` (rejected), `c9`, `q6` (question)
- `s6` — `dataset pipeline: build_dataset.py, augment.py, eval_slices.py, split.py, leakage_check.py`: Corpus entries have no candidates field; frozen 1,463 examples (propose 725/escalate 357/explain 381), 0 reduced-candidate entries; missing-candidate exists only as an eval slice (`eval_slices.py`:32-66); escalate reasons collapse to one `ESCALATE_REASON` (`build_dataset.py`:92-93) though corpus classes distinguish 8 decline kinds; post-freeze additions are deviations (train-supplement.json).
  - seeds: `c11`, `c12`, `q3` (question, resolved)
- `s7` — `docs/benchmarks/2026-09-24-qwen-tool-jev-comparison.md + delivery record`: scorer-b1: test 27/32, abstain 11/15, wm 0, exact ECE 0.132 Brier 0.268 (bar 0.10 failed); missing-candidate abstain 7/32, ECE 0.547; held-out 27/41. Test side declared spent; validation should grow to ~150-200; six data recommendations recorded (:218-267).
  - seeds: `c11`, `c13`, `c14`
- `s8` — `docs/qwen-tool-jev-finetune.md Track B runs`: b1 (3 ep, lr 2e-4) chosen; b2 5 ep overfits; b3 2 ep underfits; b4 lr 1e-4 same decisions, better calibration (ECE 0.072 vs 0.097) but lower abstain precision.
  - seeds: `c15`
- `s9` — `nvsh runtime: nvsh/tiers/{router,manager,base}.py, nvsh/config.py, nvsh/ops/table.py`: Tool-Jev not loaded anywhere under nvsh/; TierDecision.confidence and an unwired LogprobVerifier (ask/escalate thresholds, router.py:161-199) already exist; \[tiers\] config has the `min_confidence` validation pattern; ops carry `read_only`; runtime deps are \[\]; no op-name switching.
  - seeds: `c16`, `c22`, `q1` (question, resolved)
- `s10` — `issue #53 + OriNachum design note`: Order: permutation test first (no retrain), gate, randomized-label training, complete-logit readout + calibrate deployed Q4, then decide on data/objective/API. Semantic escalate vs epistemic abstention kept distinct; Track A vs B relabel; RLCD not required.
  - seeds: `c2`, `c3`, `c20`, `c21`
- `s11` — `external: OpenJev model card + TypeSafe Jev blog`: OpenJev: 27B, CC BY-NC 4.0 (reference only), order-shuffle answer change 2.3% vs 18.5% for its base, trained for order consistency; calibration is a fixed readout temperature (`READOUT_T`=0.85, noul T=1.829) with no published ECE/Brier and no abstention mechanism. TypeSafe RLCD: no reward, scoring rule, data, paper or numbers disclosed.
  - seeds: `c25`, `c26`, `c27`, `c31`
- `s12` — `challenge pass / cheap-probe lens: llama-server CPU probe, scorer-b1 Q4_K_M + bf16 GGUF`: 6 spent-test prompts: `logit_bias` leaves returned probs unchanged; top-5000 complete 6/6; bf16 GGUF = in-process exact within 1e-4 on 5/6, dev-f02 off by 0.137 (variant-summing gap); Q4 drifts up to 0.096. Scratch port, no repo/GPU use.
  - seeds: `c38`, `c39`, `c40`, `q9` (question)
- `s13` — `challenge pass / adjacent-systems lens: nvsh/tiers/toolchat.py + nvsh/tiers/corpus/`: `score_next_token` takes top, so readout needs no nvsh change; but dev.json and held-out.json ship in the wheel and feed nvsh tiers bench, Needle, LFM and Laya #51 — re-splitting there is a runtime change.
  - seeds: `c38`, `q10` (question, resolved), `q11` (question, resolved)
- `s14` — `challenge pass / unstated-assumptions lens: validation reuse + 2.3% bar`: Temperature, thresholds and checkpoint choice all read validation; the 2.3% bar has no stated estimator and n makes its CI wider than the bar.
  - seeds: `c41`, `c42`, `q12` (question, resolved)
- `s15` — `challenge pass / lifecycle + operations lens: guide ledger, Hub artifacts, bundle scan`: Operator standing rule for a live guide/ledger was not on the frame; baseline artifacts could be overwritten by new uploads; `scan_bundle` existed in #46 but not in this frame.
  - seeds: `c43`, `c44`
- `s16` — `challenge pass / hidden-dependency lens: scorer._description + metrics CALIBRATION_LABELS`: Reason candidates would otherwise pull descriptions from nvsh/tiers/lfm.py and break escalate roll-up.
  - seeds: `c45`
- `s17` — `challenge pass / security lens: redaction, secrets, private hosts`: Examined: `scan_bundle.py` + scripts/scan-secrets.py exist and cover uploads and commits; teachers are local lobes. No new finding beyond c44.
- `s18` — `challenge pass / concurrency + hardware lens: serve_for_measure.sh, shared GPU, Orin`: Known limits: P68 same-port stop/start race (single serial runner), `NV_ERR_NO_MEMORY` when training and serving overlap on a Spark, lobes share GPUs. Plan-side risks, not spec changes.
- `s19` — `challenge pass / reversibility + migration lens`: No schema or runtime migration: experiment-only (c32). Reversibility rests on c44 (baseline kept). Unexamined: vLLM and Orin behaviour at top-5000.

## Decisions

- Missing data is never a reason to stop short of a bar: where a gap is data-limited, generating train-side data through the all-Apache pipeline is the fix and becomes a task (operator 2026-09-25: 'Missing data is not an explanation - it's a target to fill').
- Escalation reasons: train an ablation with the 8 corpus decline classes (`outside_table`, repair, diagnosis, `missing_argument`, `not_a_request`, `multi_step`, injection, `over_time`) as distinguishable targets vs one escalate label; keep the reasons only if they improve validation results (operator 2026-09-25: 'if it helps, we add the 8 reasons').
- Fresh evaluation sides come from a re-split of the corpus; new teacher-drafted sources are added when a slice or validation size needs them (operator 2026-09-25).
- Track B is the subject of the cycle, as issue #53 frames it (every gap targets the candidate scorer); Track A only gets the documentation relabel (gap 4), its improvements go to #56 (operator 2026-09-25).
- The cycle is experiment-only: no nvsh/\*\* runtime change; the gate, thresholds and scorer are delivered as scripts/lfm-finetune code, measured artifacts and reports, and runtime wiring is issue #54 (operator accepted the recommendation 2026-09-25).
- Permutation robustness must come from the model, not from the runtime forcing one strict candidate order: canonicalizing order is a known workaround the operator prefers to avoid; it does not count toward the <= 2.3% bar, and falling back to it needs an explicit operator decision (operator 2026-09-25).
- Corpus v2 lives scripts-side and in the private data repo; nvsh/tiers/corpus/dev.json and held-out.json are not modified by this cycle (operator 2026-09-25, q10).
- The new sealed held-out is teacher-drafted (`draft_heldout.py`), read by the lead as counts/hashes only, and stored only in the private data repo (operator 2026-09-25, q11).
- The permutation bar is judged on the point estimate (<= 2.3%) with the 95% CI reported, from >= 10 permutations per entry on a ~150-entry test side (operator 2026-09-25, q12).

## Hard questions

- Is wiring the gate into the nvsh runtime in scope for this cycle — implementing a Verifier for the existing but unwired LogprobVerifier slot (nvsh/tiers/router.py:112-207; `verifier_factory` defaults to None, nvsh/tiers/manager.py:384-385) with threshold keys under \[tiers\] — or does the cycle stay experiment-only in scripts/lfm-finetune like PR #52? (resolved: Operator accepted (2026-09-25): the cycle stays experiment-only in scripts/lfm-finetune; runtime wiring of the gate and scorer is issue #54.)
- Which bars define done for this cycle? Candidates: ECE <= 0.10 on the deployed `Q4_K_M` with a CI; 0 wrong mutating at the chosen thresholds; answer-change rate under permutation <= some X% (OpenJev ~2.3%); missing-candidate abstain >= some Y% (7/32 today). (resolved: OpenJev level (operator 2026-09-25): permutation answer-change <= 2.3% (c25); calibration and abstention bars are nvsh's own since OpenJev publishes neither (c26, c27); missing data is a target, not an excuse (c24).)
- risk: Per-example label randomization may cost clean-test accuracy relative to scorer-b1 (27/32) while improving robustness; the trade-off needs its own decision rule before the run.
- Should semantic escalate reasons stay one label, or become distinguishable? The corpus already tags them (class decline:`outside_table` 28, repair 14, diagnosis 13, `missing_argument` 12, `not_a_request` 10, `multi_step` 8, injection 7, `over_time` 6 in nvsh/tiers/corpus/dev.json) but `build_dataset.py`:92-93 collapses all to one `ESCALATE_REASON`. (resolved: Ablation: 8 decline classes as targets vs one label; keep only if validation improves (c28).)
- Where does the fresh sealed test side and held-out come from: new sources drafted by the teachers (`draft_heldout.py` pattern) or a re-split of the existing 431-entry corpus plus new entries? (resolved: Re-split the corpus; add new sources when needed (c29).)
- The corpus lives inside the runtime package (nvsh/tiers/corpus/dev.json and held-out.json, shipped in the wheel per tests/`test_wheel_packaging.py`, read by nvsh tiers bench and by the Needle, LFM and Laya #51 work). Re-splitting it or adding sources there is an nvsh/ change (conflicts c32, publishes to PyPI) and moves every other experiment's baseline. Does the new corpus version live scripts-side (e.g. a versioned corpus under scripts/lfm-finetune or the private data repo), or do we update nvsh/tiers/corpus and accept the runtime change? (resolved: Operator accepted 2026-09-25: a versioned corpus lives scripts-side and in the private data repo; nvsh/tiers/corpus stays untouched (runtime promotion belongs to #54).)
- The existing held-out is committed publicly (nvsh/tiers/corpus/held-out.json) and its authoring is the operator task in issue #38. Who authors the new sealed held-out (teacher-drafted by `draft_heldout.py` with the lead reading counts only, as in #46, or the operator), and is it committed publicly or kept only in the private data repo? (resolved: Operator accepted 2026-09-25: the new sealed held-out is teacher-drafted with `draft_heldout.py`, the lead sees counts and hashes only, and it is kept only in the private data repo, never committed publicly.)
- Does the calibration cycle cover Track A (a3-heal, ECE 0.080 via teacher forcing) as well, or Track B only, with Track A limited to the documentation relabel? (resolved: Track B only; Track A gets the doc relabel; Track A work -> #56 (c30).)
- Is the <= 2.3% bar met on the point estimate or on the CI upper bound? With ~100 test entries x 3 permutations, a 2.3% rate is ~7 flips and the 95% CI upper bound sits near 4-5%; proving the upper bound <= 2.3% needs roughly 1,000+ independent permutation trials (more permutations per entry help only partly, since they are correlated). (resolved: Operator accepted 2026-09-25: point estimate <= 2.3% with the 95% CI reported, measured with >= 10 permutations per entry on a ~150-entry test side (~1,500 trials).)

## Open parks

- [unknown_nonblocking] vLLM /completions with --max-logprobs >= 5000 (memory, latency, whether returned logprobs are raw or processed) and Orin's llama.cpp build 10373 at `n_probs` 5000 were not probed; only llama-server on the Spark CPU was.
- [unknown_nonblocking] Residual surprise risk after this pass: per-example label randomization may interact with Qwen3.5 tokenization of lowercase letter labels (a-z) differently from uppercase; nothing examined yet shows whether the model treats them as equivalent labels.
- [follow_up] Choice/Noul/Score primitives (e.g. 'is retrying this command safe?') as composable narrow decisions, per issue #53 gap 5 — revisit after this cycle's calibration results.
