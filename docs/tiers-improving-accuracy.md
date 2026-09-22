# Improving tier accuracy: a working brief

This page is for whoever picks up the job of making the local tiers *good*,
not just wired. The plumbing is done and measured; the model is the weak
part. Everything here is a measurement, a tool that exists, or a plainly
labelled guess.

## Where it stands

Tier 1 is Needle3 (121M) picking one operation from `nvsh/ops/table.py`.
Measured on a DGX Spark, real engine, the shipped code path:

| What | Result | Target (spec c20) |
|---|---|---|
| Correct operation on explicit asks | 4 of 9 (`tiers bench`), 5 of 9 (pick only) | at least 90% |
| Should-escalate asks declined | 2 to 3 of 5 | at least 80% |
| Wrong mutating picks that reach the operator | 0 | 0 |
| Warm latency p95 / cold start | 78 ms / 312 ms | under 150 ms |
| Whole path, warm, through the daemon's tier manager | 46 to 110 ms | — |

Speed and safety pass. Accuracy is about half of what is needed.

Typical misses: "How hot is this machine?" and "Show GPU usage" both come
back as `disk_stats`; "Show running containers" as `process_list`; "Run nvsh
doctor" as a service operation; "Why did vLLM crash?" gets a confident
read-only pick instead of a decline.

## Update, 2026-09-19 (later the same day): the grid corpus and the first fine-tune

The dev corpus is now a grid of **318 entries** (every operation across five
phrasing classes, a third should-decline), each with a `class` field, and
`nvsh tiers bench` reports accuracy per expected operation and per class.

Stock Needle3 on it, shipped path, DGX Spark:

| What | Result |
|---|---|
| Operation and arguments right, explicit asks | 72 of 208 (35%) |
| Should-decline asks declined | 78 of 106 (74%) |
| Wrong mutating picks (each shown with its interpretation) | 7 |
| Operations at zero | `thermal_stats`, `power_get`, `nvsh_doctor`, `container_restart` |
| Worst phrasing class | operator jargon, 3 of 33 |
| Warm latency p95 | 70 ms |

A LoRA was trained on a 75% fold of the grid (233 examples, 20 epochs,
`--lr 5e-4 --lora-rank 32 --lora-alpha 64`, about 35 minutes on the GB10 with
`jax[cuda13]`; the default 3 to 5 epochs at `1e-4` barely moves the loss) and
scored on the other 80 entries. Same author for both folds, so this is
**indicative only**:

| Pick only, 80-entry test fold | Stock | Tuned |
|---|---|---|
| Run in JAX: asks expecting an operation | 11 of 53 | **42 of 53** |
| Run in JAX: should-decline asks declined | 0 of 27 | 16 of 27 |
| Through nvsh, using the exported `.cact` | 18 of 52 | 18 of 52 |

Two things follow.

1. **Fine-tuning works, and the data is the lever**: 21% to 79% on picks.
2. **The export is broken, upstream.** The `.cact` that `needle build --lora`
   writes does not behave like the weights it was built from: through the
   engine it gets 2 of 30 of its own training examples right. Matching the
   prompt (compact tool JSON, `auto_date=False`) changes nothing, and nothing
   is truncated. Reported as
   [cactus-compute/needle#134](https://github.com/cactus-compute/needle/issues/134).
   **Until that is resolved a tuned Needle3 cannot ship through nvsh's engine
   path**, whatever its accuracy in JAX. Do not read the "through nvsh" row as
   a verdict on the model.

Also learned: in JAX the stock model never declines (the engine adds that),
and the tuned model still answers many should-decline asks with a confident
pick ("Is grafana running?" becomes `service_status grafana.service`, which
grounding then refuses). Decline examples need more weight, and nvsh's
grounding already catches the unknown-name half of them.

The scripts for all of this (`foldbench.py`, `jaxfold.py`, `jaxcheck.py`,
`promptmatch.py`) are development tools kept outside the repository; the
recipe in `needle-finetune.md` has the commands.

## Update, 2026-09-22: explain entries in the dev corpus

The dev corpus now has **431 entries**: 212 expect an operation, 103 expect
escalate, and 116 expect **explain** (113 new ones, ids `dev-w001` to `dev-w113`,
source `explain-2026-09-22`, class `explain:<phrasing>`). An explain entry is
a read-only question a person answers in words — what `jetson_clocks` does,
what `tegrastats` fields mean, why a CUDA out-of-memory error means something
different on unified memory — that no table operation answers by inspecting
the machine and that does not need the full agent. Each carries an authored
1 to 3 sentence answer in `expect.answer`, used to train Tier 2 (issue 39).
The seeded split (`scripts/lfm-finetune/split.py`, seed 39) puts them
81 / 18 / 17 across train / val / test, next to 72 / 16 / 15 escalate and
148 / 32 / 32 operation entries. The numbers in the sections above were
measured on the 318-entry grid, before these were added; for Tier 1 an
explain entry is scored as a should-decline, and the per-operation accuracy
leaves it out.

Three older entries were the same kind of question but expected escalate:
`dev-g284` ("What is a Jetson?"), `dev-g285` ("How does nvpmodel work?") and
`dev-g286` ("Explain unified memory"). With the operator's approval
(deviation `d1`) they now expect explain with a written answer, so the corpus
no longer teaches two answers to one question; Tier 1 scores both labels as a
should-decline, so its figures are unaffected. `dev-g287` (an opinion) and
`dev-g288` (off topic) still expect escalate.

## What was tried, and did not work

- **Rewording operation descriptions.** 5 of 9 became 6 of 9, and a request
  that had been correctly declined became a wrong *mutating* pick
  (`power_set low_power` for "Fix this linker error").
- **Tool order.** Reversed and shuffled orders changed *which* requests
  failed, not how many (7 to 8 of 14 either way).
- **Showing fewer tools** (read-only operations only): worse, 3 of 14.

Conclusion: on this table stock Needle3 is unstable — small prompt changes
flip individual answers without moving the total. More prompt work is not
the lever. Do not spend another session on wording.

## The lever: data, organised

The dev corpus had 21 entries when this was written; it is now the 318-entry
grid described in the update above. What follows is how it was built and how
to extend it.

Build it as a grid, not a pile, so gaps are visible:

1. **One row per operation** in the table. For each, at least 12 phrasings
   across these classes: the plain imperative ("show gpu usage"), a question
   ("is the gpu busy?"), a symptom ("everything is slow, is it the gpu?"),
   terse/typo'd ("gpu?"), and operator jargon (`tegrastats`, `nvidia-smi`,
   `jtop`). About 200 positive examples in all.
2. **Arguments**: for every operation with an argument, phrasings with the
   unit or container named exactly, abbreviated ("vllm" for `vllm.service`),
   and missing (expected: decline, not a guess).
3. **Should-decline**, at least a third of the corpus: diagnosis ("why did X
   crash"), repair ("fix this linker error"), comparison over time,
   multi-step asks, anything needing a command outside the table, and
   injection strings. These teach the empty answer.
4. **Confusable pairs**, deliberately: gpu/memory/disk/thermal stats;
   container list/process list; service status/logs/restart; `nvsh doctor`
   versus a service called nvsh.
5. **Held-out**: authored by the operator, in a different sitting from the
   operation descriptions and the dev phrasings. It ships empty; until it has
   entries, no tuned model can be adopted (see `docs/needle-finetune.md`).

Record for every entry its `source` and its class, so the benchmark can
report accuracy per class and per operation — that is what says what to
write next. Adding those two breakdowns to `nvsh/tiers/bench.py` is a small
change next to `accuracy_by_kind()`.

Then, in this order:

1. Re-measure stock Needle3 on the larger dev split (the baseline to beat).
2. Fine-tune per `docs/needle-finetune.md`; measure dev and held-out.
3. Turn on the second-opinion check (`LogprobVerifier` in
   `nvsh/tiers/router.py`) with a local LFM2.5 and measure what it does to
   the should-decline rate. Its three thresholds (`ask_below=0.0`,
   `escalate_below=-2.0`, `min_mass=0.05`) are **guesses**; `tiers bench`
   prints a threshold suggestion from the dev split once verifier scores
   exist. An earlier spike measured AUC 0.76 for this check after
   per-operation calibration — useful, not decisive.
4. Only then consider changing the table (fewer, broader operations is a real
   option: Needle did better in a spike with 10 tools than with 16).

## Tools that exist

| Task | Command |
|---|---|
| Fetch and verify the pinned engine and weights | `nvsh tiers prefetch --yes` |
| Full benchmark, real engine, fixture machine | `nvsh tiers bench --tier needle --out result.json` |
| Same, against this host's real units and containers | add `--live` |
| Per-request rows (expected, picked, decline reasons) | the `items` list in the result file |
| Try rewordings without editing the table (seconds) | `python scripts/needle-finetune/try_table.py --overrides new.json` |
| Build `needle finetune` JSONL from the dev split | `python scripts/needle-finetune/build_dataset.py --out train.jsonl` |
| What the tiers did in real use | `nvsh tiers stats`, `nvsh tiers export FILE` |
| Ask Tier 1 directly at the prompt | `@needle show gpu usage` |

All of these need `cactus-needle` importable: `pip install 'nvsh[needle]'`,
or a throwaway venv with `cactus-needle` and `PYTHONPATH` set to the checkout.
The benchmark grounds against the fixture machine declared in the corpus
file's `world` object, so a score measures the model, not the host it ran on.

## Rules that keep the numbers honest

- Never train, tune thresholds, or reword against the held-out split. The
  tools refuse it by file name; do not work around that.
- A target that was not measured prints "not measured". It is never a pass.
- Report per-request rows with every claim of improvement. A total that moved
  by one request on a nine-request corpus is noise.
- Accuracy is not the safety mechanism and must not become one: a wrong pick
  is caught because the proposal states the operation and arguments it
  understood and nothing runs without approval. Do not weaken that to gain
  points.
- Keep the table the only place operations are named. If an improvement needs
  code that mentions a specific operation, it is the wrong improvement.

## Not built yet

- Loading a tuned archive: the tuned pin, the operation-table hash check and
  the config override (`docs/needle-finetune.md`, "Pin it").
- Tier 2 is built (`docs/tier2.md`) but its stock models do not yet use the
  control tools; the LFM2.5 fine-tune recipe (`docs/lfm-finetune.md`) is
  written and has not been run. `nvsh tiers bench` has no Tier 2 option yet.
- Memory measurements on an Orin-class device, and llama.cpp as the engine
  for the 8 GB budget.
- The larger edge-agent goal (LFM2.5-1.2B/2.6B answering from a maintained
  knowledge store, with sources) is tracked in issue 33.
