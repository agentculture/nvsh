# Tier 2: a small resident LFM2.5

Tier 2 sits between Needle3 (Tier 1) and the configured full agent. It is
**off until you configure it**, and its measured judgement with stock models
is poor (see [Measured](#measured)): read that section before turning it on.

## What it does

`nvsh/tiers/lfm.py` runs a bounded loop against one OpenAI-compatible
endpoint on localhost:

1. The model is offered every operation in `nvsh/ops/table.py` as a tool
   (the schemas are generated from the table), plus three control tools:
   `propose`, `explain` and `escalate`.
2. It may **inspect**: call a read-only operation. nvsh validates the call,
   grounds its arguments against real units and containers, renders the
   command from the table, runs it with a timeout, and feeds back a redacted,
   bounded result. At most 4 rounds.
3. It ends in exactly one of three ways: **propose** one operation (shown as
   an ordinary nvsh proposal naming what was understood, under the usual
   approval, `sudo` and destructive-command rules), **explain** in plain
   words, or **escalate** to the full agent, which receives what Tier 2
   already inspected.

A mutating operation is never run inside the loop: a bare call to one is
refused and fed back as an error, and `propose` only ever produces a proposal.
A command failure starts at Tier 2 (Tier 1 never sees failures). `@lfm` asks
Tier 2 alone; `@target` for any other harness bypasses the tiers.

Unlike Tier 1, **Tier 2 runs read-only commands on the machine without asking**
(`free`, `df`, `systemctl status`, `journalctl` for a grounded unit, and the
like: whatever the table marks read-only). That is the point of the tier, and
it is why it is opt-in.

## Configure it

```toml
[tiers]
enabled = true            # automatic routing; @lfm works without it

[tiers.lfm]
model  = "LiquidAI/LFM2.5-350M"
engine = "vllm"           # llama-server | vllm | sglang
image  = "vllm/vllm-openai@sha256:<digest>"
```

`model` is what switches Tier 2 on. Other keys, all optional:

| Key | Meaning |
|---|---|
| `mode` | `managed` (nvsh starts a container) or `attach` (use `base_url`, start nothing) |
| `base_url` | For `attach`: a localhost URL ending in `/v1` |
| `image` | The container image, **by `@sha256:` digest**; a tag is refused |
| `model_dir` | For `llama-server`: the host directory holding the GGUF named by `model` |
| `gpu` | `auto` (default) or `off` for a CPU-only container |
| `gpu_memory_fraction` | Up-front GPU share for vLLM and SGLang, default `0.08` |
| `tool_call_parser` | Server-side tool-call parser; vLLM defaults to `lfm2` |
| `hf_cache_dir` | Host download cache; defaults to nvsh's own tier cache |
| `hf_offline` | `true` serves only from `hf_cache_dir` (`HF_HUB_OFFLINE=1`), for a private model fetched on the host; nvsh never passes a token into the container |
| `port`, `ctx`, `startup_timeout_seconds` | Host port (1024 to 65535), context length, start-up wait |

Every value is validated before a launch line is built; a bad one costs one
status line and the request goes to the full agent.

## The container

`nvsh/tiers/runtime_docker.py` holds the only `docker run` in nvsh, and its
arguments come from config and platform detection only: never from a request
and never from model output. It publishes on `127.0.0.1` only, names the
container and the port per OS user (`nvsh-tier2-<uid>`, `18400 + uid % 40000`),
passes `--gpus all` on a DGX Spark and `--runtime nvidia` on a Jetson (see
`docs/platforms.md`, "Docker GPU path"), and changes no Docker configuration.
An engine that downloads its model runs as you, not root, with its cache on
the host, so a restart needs no network and `nvsh uninstall` can delete the
files. The container is stopped on idle unload, on daemon exit and by
`nvsh uninstall`, which leaves images alone and prints how to remove them.

No image is pinned in `nvsh/tiers/pins.json` yet, so `image` is required.
Read [`lfm-license-notes.md`](lfm-license-notes.md): LFM2.5 is under the LFM
Open License, whose commercial-use threshold applies to each user.

## Measured

DGX Spark, 2026-09-19, real launcher, `vllm/vllm-openai` nightly, Tier 2
alone over an 80-entry fold of the development corpus, fixture machine.
Results file: [`benchmarks/2026-09-19-dgx-spark-tier2.md`](benchmarks/2026-09-19-dgx-spark-tier2.md).

| Stock model | Right proposal (of 53) | Escalated when it should (of 27) | Warm median / p95 |
|---|---|---|---|
| LFM2.5-350M | 0 | 1 | 218 ms / 579 ms |
| LFM2.5-1.2B-Instruct | 2 | 5 | 743 ms / 1306 ms |

The plumbing works: real tool calls, inspections, grounding, sub-second
answers, and **no wrong mutating proposal from either model**. The judgement
does not: both models nearly always answer in plain words and almost never
use `propose` or `escalate`. The spec's accuracy targets are **not met**.
The benchmark credits proposals and escalations only, so a correct plain-words
answer to a read-only question is not counted; the mutating and
should-escalate rows are a fair reading.

The follow-up is a fine-tune of the 350M that teaches the control tools:
[`lfm-finetune.md`](lfm-finetune.md). Cold start was 185 to 254 s including
the first model download; start-up from a warm cache was not timed.
