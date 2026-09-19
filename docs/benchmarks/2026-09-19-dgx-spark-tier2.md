# DGX Spark, 2026-09-19: Tier 1 and Tier 2 baselines

Device: NVIDIA DGX Spark (GB10, unified memory, 121 GiB), driver 580.126.09,
Docker with `--gpus all`. The GPU was shared with the operator's own vLLM
server (about 42 GiB) during every run, so latencies are under load.
nvsh at branch `feat/tiers-tier2`. Grounding: the corpus's fixture machine.

## Tier 1, stock Needle3 (cactus-needle 3.0.1 engine, CPU)

Command: `nvsh tiers bench --tier needle`. Corpus: the 318-entry dev grid.

| Metric | Result | Target |
|---|---|---|
| Operation and arguments right, explicit asks | 72 of 208 (35%) | at least 90%: **missed** |
| Should-escalate asks declined | 78 of 106 (74%) | at least 80%: **missed** |
| Wrong mutating picks shown without their interpretation | 0 (7 wrong picks, all shown) | 0: met |
| Warm latency p95 / cold | 70 ms / 345 ms | under 150 ms: met |
| Added memory | not measured | under 1 GiB |
| Image size | not applicable (no container) | |

## Tier 1, LoRA-tuned Needle3

Not measurable through nvsh: the exported archive is faulty upstream
([cactus-compute/needle#134](https://github.com/cactus-compute/needle/issues/134)).
In JAX, pick only, on an 80-entry fold by the same author as the training
fold (indicative only): 42 of 53 against stock's 11 of 53; should-decline 16
of 27 against 0 of 27.

## Tier 2, stock LFM2.5, engine vLLM (GPU)

Image `vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`,
**20.6 GB on disk**. `--gpu-memory-utilization 0.08`, `--max-model-len 4096`,
`--tool-call-parser lfm2`. Tier 2 alone (no Tier 1), 80-entry fold, run with a
scratch script around `nvsh.tiers.bench.bench` because `nvsh tiers bench` has
no Tier 2 option yet.

| Metric | LFM2.5-350M | LFM2.5-1.2B-Instruct |
|---|---|---|
| Right operation and arguments proposed | 0 of 53 | 2 of 53 |
| Should-escalate asks escalated | 1 of 27 | 5 of 27 |
| Wrong mutating proposals | 0 | 0 |
| Warm latency, median / p95 | 218 ms / 579 ms | 743 ms / 1306 ms |
| First request after start | 22.6 s | 22.7 s |
| Container memory (`docker stats`) | not measured | 4.5 GiB |
| Start-up, including first download | 254 s | 185 s |

## Not measured

- A second engine (`llama-server`, SGLang) on any device.
- AGX Thor and AGX Orin, and the Orin Nano estimate: they need the operator's
  go-ahead before anything is installed there.
- The same prompts through the default full agent.
- A run with networking blocked for Tier 2 (Tier 1's offline run was done for
  the previous pull request).
- Added resident memory for Tier 1, and GPU memory for Tier 2.
