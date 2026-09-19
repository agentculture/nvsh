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

The dev corpus has **21 entries, 9 of them explicit picks** across 16
operations. That is too small to train on and too small to measure a 90%
target (one request is 11 points). The first job is the corpus.

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
- Tier 2 (the LFM2.5 loop, its container launcher, the `@lfm` adapter) and
  the LFM2.5 fine-tune recipes — the next pull request.
- Memory measurements on an Orin-class device, and llama.cpp as the engine
  for the 8 GB budget.
- The larger edge-agent goal (LFM2.5-1.2B/2.6B answering from a maintained
  knowledge store, with sources) is tracked in issue 33.
