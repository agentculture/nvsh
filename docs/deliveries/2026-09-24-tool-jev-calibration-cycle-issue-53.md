# Delivery Summary — Tool-Jev calibration cycle (issue 53)

plan: `tool-jev-calibration-cycle-issue-53` · run: `partial` · date: `2026-09-24`
baseline: `devague summary skeleton`

## Intent

> The Tool-Jev scorer now makes calibrated, label-permutation-robust decisions on its deployed quantized build: an improvement cycle over code and data measured the permutation dependence of scorer-b1, added an uncertainty gate separate from semantic escalate, trained with per-example randomized candidate labels on a grown dataset, and reports reliability per slice for the actual shipped artifact (issue #53)

After: A retrained Track B scorer whose choice survives label/order permutation, whose served `Q4_K_M` returns complete label distributions that are calibrated after a fitted temperature, with separate semantic-escalate and uncertainty-abstain outcomes, thresholds measured per read-only/mutating, and every result reported per slice with confidence intervals on fresh evaluation sides.

## Planned Work

- `t1` — Readout core in scorer.py: one label-probability definition and a complete top-k
- `t2` — Permutation seam in scorer.py: seeded order/letter permutation, subsets and description overrides
- `t3` — Served readout cap: servers started for measurement return >= 5000 logprobs
- `t4` — Post-hoc calibration module: seeded validation folds, temperature and vector scaling fit/apply
- `t5` — metrics.py: per-slice ECE/Brier, bootstrap CIs, `abstain_uncertain` outcome, reliability bins in markdown
- `t6` — Uncertainty gate module and offline threshold sweep
- `t7` — Permutation probe runner with answer-change rate and CI
- `t8` — measure.py: wire the new readout, calibration apply and per-slice report into measure runs; fix P71
- `t9` — Pre-registered decision rule for checkpoint choice, committed before any training
- `t10` — Live guide and ledger with a documentation subagent
- `t11` — Documentation relabel: Track A is a generative tool router, Track B the Jev-style scorer
- `t12` — Corpus v2 tooling: versioned re-split kept outside nvsh/, with validation folds
- `t13` — Draft new sources for the fresh evaluation sides and the sealed held-out
- `t14` — Dataset build: per-example candidate sets and letter maps, missing-candidate and no-valid-option examples, reason candidates
- `t15` — Augment for the comparison doc's five other recommendations, then freeze the v2 training set
- `t16` — `train_scorer.py`: per-row label ids and targets from the rendered prompt; optional calibration-aware loss
- `t17` — Baseline on scorer-b1 before any retraining: permutation probe, readout fidelity, offline gate and temperature
- `t18` — Training runs and selection: randomized-label variant, reasons ablation, lr variants, conditional calibration-aware stage
- `t19` — Quantize the chosen checkpoint and calibrate the deployed build
- `t20` — Final measurement on the fresh test side and held-out, Spark and Orin
- `t21` — Private uploads without touching PR #52's artifacts
- `t22` — Cycle report: comparison against scorer-b1 and OpenJev, before/after, follow-up links
- `t23` — Close the loop: validate-delivery, summarize-delivery, version bump, PR

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `scorer.distribution()` sums every token variant of a label; `READOUT_TOP` 20000 (raised from 5000 after P13); `label_variant_ids`; the `train_scorer` adoption moved to t16 (d1) |
| `t2` | delivered | `Permutation`/`permute()` seam, description overrides, op-level `same_choice()`; default map byte-identical |
| `t3` | delivered | `MEASURE_MAX_LOGPROBS` 20000 in `serve_for_measure.sh`, `pipeline.sh`, env examples and release instructions (P5 caught the pipeline default) |
| `t4` | delivered | `calibration_fit.py` folds/fit/apply/evaluate; source-grouped folds (P14); log-space NLL (P5) |
| `t5` | delivered | per-slice ECE/Brier, bootstrap CIs, `abstain_uncertain`, reliability markdown; later d5 canonical argument compare |
| `t6` | delivered | `gate.py` keyed by `Operation.read_only`; `sweep_gate.py` offline sweep with `--final` |
| `t7` | delivered | `permutation_probe.py` five kinds, CIs over entries, 52-letter alphabet (P14) |
| `t8` | delivered | `measure.py --calibration`, CI rows, readout completeness row, per-slice section, `--reasons`; closes #57 |
| `t9` | delivered | `docs/tool-jev-calibration-rule.md` committed (b1c6cb6) before the first training command; amended in use by d6 |
| `t10` | delivered | `docs/qwen-tool-jev-calibration.md`: decision path D1-D49, ledger P1-P26, reproduce steps 1-13; kept by the lead after the doc agent went idle; plus `docs/scorer-finetune-playbook.md` |
| `t11` | delivered | Track A called a generative tool router, Track B the Jev-style scorer, in the guide, comparison doc and model-card text |
| `t12` | delivered | `split.py` v2 with string header, `--train-only` (d3), folds; l5 fixed after an agy review |
| `t13` | delivered | eval pool 480 (two reviewers), sealed held-out 149 (read-only, private); reviewer fixes P12, P16, P17 (l4, l6, l7) |
| `t14` | delivered | `build_dataset.py` per-row maps, `--scorer-out`, `-nocand` rows, `--reasons`; integration fix d2 |
| `t15` | delivered | `targeted_augment.py` six recipes (incl. check-then-change, d7); frozen twice: 2333 rows (D37) and 2503 rows for r3b (D45); d4 |
| `t16` | delivered | `train_scorer.py` per-row label ids and targets, variant readout, optional smoothing/Brier loss |
| `t17` | delivered | scorer-b1 baseline on v2 validation, re-measured after l9 and d5: right 73/84, permutation 18.8%, Q4 ECE 0.097 raw |
| `t18` | delivered | r1, r2, r3 trained and judged; r3b added by d7 and chosen under the rule with d6; r4 not triggered |
| `t19` | delivered | `scorer-r3b.q4_k_m`: temperature 1.54, read-only margin 0.2 gate, frozen (D47) |
| `t20` | delivered | one final run on test (198) and the sealed held-out (149), Spark and AGX Orin (D48) |
| `t21` | delivered | three new private repos, fetched back identical, private=True; PR #52's six repos untouched (D49) |
| `t22` | delivered | `docs/benchmarks/2026-09-25-tool-jev-calibration-cycle.md` |
| `t23` | partial | `/validate-delivery` filed (o1-o14, e1-e15, b1-b7) and this summary; version bump and PR still to do |

## Mid-work Decisions

- `d1` — t1 criterion 1's `train_scorer` adoption of the shared label-probability definition moves to t16 (`label_variant_ids` + `label_logits_from_vocab`, variant ids recorded in run metadata); the lead updates one t8-owned assertion in tests/`test_lfm_finetune_measure.py` (served top == `READOUT_TOP`) at t1's merge; measure.py's --max-logprobs preflight stays with t8 — `train_scorer.py` is owned by t16 and t1 was told not to touch it; t1's required `READOUT_TOP` change breaks one hard-coded top-22 assertion in a t8 test file, and merges must stay green
- `d2` — Wire t14's per-example data into Track B training: `build_dataset.py` gains --scorer-out writing a corpus-format scorer training file (original entries + derived -nocand entries, each with permutation/gold/`perm_seed`/descriptions); pipeline.sh assemble passes --randomize-labels/--reasons/--missing-candidate-rate/--perm-seed through (env vars) and train-scorer trains on the scorer file when present; an end-to-end test proves rows reaching `train_scorer` carry randomized maps and -nocand entries — integration gap: t14 enriched the rendered nvsh-train.jsonl but train-scorer reads the corpus-format train-augmented.json, so r1-r3 would silently train on the fixed map without missing-candidate data; no task owned the hand-off
- `d3` — corpus v2 split: nvsh/tiers/corpus/dev.json (all 431, incl. the issue-46 val/test sides) goes to the v2 train side only; the new validation (~150) and test (~140) sides come only from this cycle's teacher drafts (eval pool), unseen by scorer-b1 and by the lead; split.py gains a --train-only CORPUS flag for this — a plain re-split (c29) would place scorer-b1 training entries and issue-46 validation entries the lead read per entry into the new val/test sides, inflating the t17 baseline and making the fresh test side not fresh; operator chose fresh-only eval sides 2026-09-25
- `d4` — assemble no longer protects issue 39's and issue 46's old (spent) test sides, so all 431 dev.json entries and their variations train, as d3 intended; issue 46's sealed held-out, the v2 val/test sides and the q53 sealed held-out stay protected — `leakage_check` dropped 123 of 431 dev.json entries as exact matches of the old test sides carried over from issue 46's `PROTECTED_EXTRA`; this cycle makes no claim on those spent sides and the operator prefers more data
- `d5` — metrics.py compares a service argument the way grounding matches it (case-insensitive, '.service' unit suffix optional; containers case-insensitive) instead of by exact string; every figure so far is recomputed or remeasured — the fresh v2 eval sides (and very likely the sealed held-out) write bare service names while grounding returns the systemd unit, so exact comparison scored every right service proposal as wrong arguments (b1 val 59 exact vs 73 canonical; r1 52 vs 71, and 4 of r1's 5 'wrong mutating' were the same unit spelled two ways)
- `d6` — checkpoint choice adds a safety-first filter before the rule's calibration step: a candidate must have 0 wrong mutating on the whole validation side (fit + selection) and its missing-candidate slice with the gate off; only r1 passes, so r1 goes to t19 instead of the rule's r3 — r3 (the rule's pick by ECE after temperature) proposes `power_set` on three check-then-change validation requests at 0.97/0.83/0.78 and is safe only behind a 0.95 mutating floor fitted on the same fold; r1 escalates them with no gate, and the final bar is 0 wrong mutating on test and held-out
- `d7` — after the freeze, draft and review a train-only check-then-change supplement (multi-step requests that check then change, expect escalate, each paired with its plain read-only check), re-freeze, and retrain the r3 recipe (r3b) for comparison with r1 under the d6 filter before t20; t19 proceeds on r1 meanwhile; t20 waits for that comparison — operator: prepare more data in parallel to retrain r3; r3's failures are all this one request shape and the training set has no recipe that targets it

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t1` (`d1`) | `train_scorer.py` is owned by t16 and t1 was told not to touch it; t1's required `READOUT_TOP` change breaks one hard-coded top-22 assertion in a t8 test file, and merges must stay green | `acceptable` |
| `t14` (`d2`) | integration gap: t14 enriched the rendered nvsh-train.jsonl but train-scorer reads the corpus-format train-augmented.json, so r1-r3 would silently train on the fixed map without missing-candidate data; no task owned the hand-off | `acceptable` |
| `t12` (`d3`) | a plain re-split (c29) would place scorer-b1 training entries and issue-46 validation entries the lead read per entry into the new val/test sides, inflating the t17 baseline and making the fresh test side not fresh; operator chose fresh-only eval sides 2026-09-25 | `acceptable` |
| `t15` (`d4`) | `leakage_check` dropped 123 of 431 dev.json entries as exact matches of the old test sides carried over from issue 46's `PROTECTED_EXTRA`; this cycle makes no claim on those spent sides and the operator prefers more data | `acceptable` |
| `t18` (`d5`) | the fresh v2 eval sides (and very likely the sealed held-out) write bare service names while grounding returns the systemd unit, so exact comparison scored every right service proposal as wrong arguments (b1 val 59 exact vs 73 canonical; r1 52 vs 71, and 4 of r1's 5 'wrong mutating' were the same unit spelled two ways) | `acceptable` |
| `t18` (`d6`) | r3 (the rule's pick by ECE after temperature) proposes `power_set` on three check-then-change validation requests at 0.97/0.83/0.78 and is safe only behind a 0.95 mutating floor fitted on the same fold; r1 escalates them with no gate, and the final bar is 0 wrong mutating on test and held-out | `risky` |
| `t15` (`d7`) | operator: prepare more data in parallel to retrain r3; r3's failures are all this one request shape and the training set has no recipe that targets it | `needs-follow-up` |

## Evidence

- tests: full suite `uv run pytest -n auto` at `3b66166` — 4700 passed, 55 skipped (torch-path scorer tests: 121 passed under the training interpreter on the training machine)
- behavioral evidence (devague): `e1`-`e15` for obligations `o1`-`o14`; **`e5` fail** (held-out missing-candidate escalation 76.7%), **`e11` fail** (bf16 GGUF readout within 0.01 on 196/204); `o1`-`o14`, `e1`-`e15` (including both fails) and deltas `b1`-`b7` approved by the operator 2026-09-26
- measurements: `docs/benchmarks/2026-09-25-lfm-final-scorer-r3b*.md`, `...-heldout-scorer-r3b*.md`, `...-edge-orin-scorer-r3b*.md`; report `docs/benchmarks/2026-09-25-tool-jev-calibration-cycle.md`
- lint: `black --check`, `flake8` on changed files, `markdownlint-cli2` on every changed doc, `scripts/scan-secrets.py` (570 files clean), `harness-smoke --stage config` (6 passed)
- commits: `main..3b66166` (148 commits on `spec/tool-jev-calibration-issue-53`, not yet pushed)
- PRs / issues: #53, #61 (ledger), follow-ups #54-#60, #62, #64; PR not opened yet

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| scorer-r3b's deployed `Q4_K_M` makes 0 wrong mutating proposals at the frozen gate on the fresh test side and the sealed held-out (both slices, Spark and Orin) | high | evidence `e3` · `docs/benchmarks/2026-09-25-tool-jev-calibration-cycle.md` |
| calibrated ECE on the deployed build is at or below 0.10: test 0.016 [0.011, 0.042], held-out 0.044 [0.024, 0.092] | high | evidence `e2` · final Q4 pages |
| the held-out `read_only` slice is calibrated to the bar | unverified | ECE 0.147 [0.079, 0.271] — not claimed |
| pooled permutation answer change on test is at or below 2.3% (2.07%) | high | evidence `e1` · `probe-scorer-r3b-test` in the report |
| missing-candidate escalation >= 80% | medium | evidence `e4` pass on test (85.5%), **`e5` fail on held-out (76.7%)** — met on one fresh side only |
| served readouts are complete (198/198, 149/149) | high | evidence `e6` |
| one label-probability definition across training, in-process and served scoring | medium | torch tests passed; evidence **`e11` fail**: bf16 GGUF within 0.01 on 196/204; lapses `l1`, `l2` cap this |
| the fresh evaluation sides are sealed and never informed a fit or choice | medium | evidence `e12`, `e14`; lapses `l4`, `l6`, `l7`, `l10` show the graders and metric needed fixes along the way |
| scorer-b1 before-figures (t17) are comparable with the after-figures | medium | re-measured after lapse `l9` and deviation `d5`; lapse `l9` caps this |
| new models and data are private and PR #52's artifacts untouched | high | evidence `e9`, `e10` |
| no gate, report or metric switches on an operation name | high | evidence `e7`, `e8` |
| the model is ready to publish | unverified | the held-out missing-candidate bar is missed; DeepEval evaluation (#64) is the operator's final gate |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `assumption-for-measurement` | t1: the 0.137 dev-f02 readout gap is claimed closed from code (both paths now sum the same variant token ids), not re-measured against scorer-b1 |
| `l2` | `control-absent` | t1: the torch/tokenizer tests (variant ids, one forward position, in-process vs served) only run under the lfm-train interpreter; CI's uv env skips them |
| `l3` | `assumption-for-measurement` | Lead's t3 merge gate checked only t3's listed files; pipeline.sh (default 22) and release instructions (22) still capped the served readout below `READOUT_TOP` — found by codex review, not by the gate |
| `l4` | `grader-unverified` | t13: the two-reviewer check ran with the default 1024-token reply budget; reviewer B (a thinking model) often spent it all and replied empty, which counted as a reject — the empty-reply rate was not checked before trusting the eval (164 kept) and held-out (86 kept) reviews |
| `l5` | `control-absent` | t12 changed split.py v2's header from the string note to a dict without checking the ~10 downstream header readers (`train_scorer`.`read_split`, `build_dataset`, `merge_variations`, measure seed/side parsing, quantize, augment, `dataset_bundle`, and the test/held-out refusal guards in `calibration_fit`/`permutation_probe`); t14's reasons mode (r2) was never wired into measure.py (bare-escalate prompt, `escalate:<reason>` choices scored invalid). Found by the agy review, not by the merge gates |
| `l6` | `grader-unverified` | t13: reviewer B's system prompt offered 'answer in words' for any non-machine text, so it rejected every decline:`not_a_request` candidate (dev.json labels small talk escalate), and the `missing_argument` definition let the generator draft unsupported actions; eval drafts kept 0 `not_a_request` and 1 `missing_argument` across all seeds before s60's per-class verdict count exposed it |
| `l7` | `grader-unverified` | t13: augment.`parse_verdict` (shared by `draft_sources`/`targeted_augment`) turned clear yes verdicts into rejects: ', no-argument' matched the standalone-no rule and 'ambiguous'/'unclear' counted as hedges even when they were the reason for an escalation; in seeds 60-63 16 yes votes were lost this way, most of the `missing_argument` class |
| `l8` | `grader-unverified` | t15 missing-argument rerun ran without a reviewer pilot; augment.`parse_verdict` treats any 'but' as a hedge, so 20 of 37 reviewer-B rejects in the rerun (38 of 56 in the first pass) were clear yes verdicts ('says "that service" but does not name it') |
| `l9` | `assumption-for-measurement` | t17 measured scorer-b1 on v2 validation with issue 46's grounding snapshot (`GROUND_SNAPSHOT` left at q46), which lacks 23 services and 11 containers named only in the v2 splits and the q53 held-out; right-proposal and invalid (`not_grounded` 14/204) figures may be understated. Distributions, ECE and Brier are unaffected |
| `l10` | `grader-unverified` | t13/t17: the fresh v2 eval pool's gold service arguments were never checked against the spelling grounding returns (bare 'rsyslog' vs 'rsyslog.service'), and metrics.py compares arguments by exact string; t17's right-proposal figures and floor were understated, found only when r1's mutating proposals came back 'wrong arguments' |

## Remaining Work / Follow-up

- `t23` — version bump, CHANGELOG, push the branch, open the PR via cicd ("part of #53", closes #57), address review, CI and Sonar green — lead, this session
- held-out missing-candidate escalation 76.7% (bar 80%) — draft more train-side requests whose right operation is absent from the candidates; re-measure on a new sealed side — next data cycle
- held-out `read_only` slice ECE 0.147 — threshold read-only proposals with it in mind when wiring the runtime — #54
- bf16 GGUF readout differs from in-process by more than 0.01 on 8/204 — investigate the runtime difference — follow-up
- evaluate the uploaded model and the harness separately with DeepEval — #64, next session
- decide per-repository Hub visibility for the three new repos — operator (they stay private until then)
- a domain-module setting for the pipeline — #62
