# Tier 2 measurement, 2026-09-25: edge-orin-scorer-r3b.q4_k_m-heldout

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-edge/q53/work/splits/held-out-q53.sealed.json --model scorer-r3b.q4_k_m --revision sha256:3568f660ab186887 --label edge-orin-scorer-r3b.q4_k_m-heldout --config $HOME/lfm-edge/q53/out/scorer-r3b.q4_k_m.toml --ctx 2048 --ground-snapshot $HOME/lfm-edge/q53/work/ground-snapshot.json --enable-thinking false --max-logprobs 20000 --scorer served --tokenizer $HOME/lfm-edge/q53/tok/scorer-r3b --slice full --acceptance --out $HOME/lfm-edge/q53/out/edge-orin-scorer-r3b.q4_k_m-heldout.md --predictions $HOME/lfm-edge/q53/out/pred`
- Split: `$HOME/lfm-edge/q53/work/splits/held-out-q53.sealed.json` (149 entries, 149 sources)
- Seed: not recorded
- nvsh: 0.19.1, commit `unknown`
- Models (repo id @ revision): `scorer-r3b.q4_k_m` @ `sha256:3568f660ab186887` (operator-supplied, not verified: attached endpoint)
- Tier 2 settings, identical for every run except the model: engine=llama-server, mode=attach, image=ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191 (llama.cpp 10373 38406d597, Jetson AGX Orin, JetPack R39), ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-edge/q53/work/ground-snapshot.json` (platform: fixture world from `nvsh/tiers/corpus/dev.json`)
- Serving: engine=llama-server, mode=attach, ctx=2048, image=`ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191 (llama.cpp 10373 38406d597, Jetson AGX Orin, JetPack R39)`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false (the scorer renders its own prompts, thinking off)
- Decision mode: candidate scorer (served) through scorer.py, max-logprobs 20000 (operator-supplied); asks for 20000 per request
- Slice: full split
- Ground snapshot: `$HOME/lfm-edge/q53/work/ground-snapshot.json` sha256 `a6e21fee60571a2c9431f5145b2b8be5829a721d2661725733cbf5d32f63aa5a` (277 services, 41 containers; created 2026-09-25)
- Calibration: none (the model's own candidate distributions)
- Acceptance run: yes
- Final run: no

A candidate-scorer run: the bench table does not apply; the figures are
metrics.py's, over the predictions file the scorer's results were written to.

## Issue 46 metrics

metrics.py over each model's predictions file (one line per entry). The candidate
distribution of a generative run comes from its deciding reply's log-probabilities.

| Metric | `scorer-r3b.q4_k_m` |
|---|---|
| Right proposals (metrics.py) | 49 of 60 |
| Right proposals, 95% bootstrap CI | [71.7%, 90.0%] (n=60) |
| Abstention recall (escalate entries escalated) | 97.1% (34 of 35) |
| Abstention recall, 95% bootstrap CI | [91.4%, 100.0%] (n=35) |
| Abstention precision, strict (deviation d2) | 87.2% |
| Abstention precision, strict, 95% bootstrap CI | [78.0%, 94.6%] (n=149) |
| False-positive tool calls (proposals on explain/escalate entries) | 0 of 89 |
| False-positive tool calls, 95% bootstrap CI | [0.0%, 0.0%] (n=89) |
| Wrong mutating, total (wrong operation + wrong arguments) | 0 (0 + 0) |
| Invalid outputs | 2 of 149 (not_grounded: 2) |
| Invalid outputs, 95% bootstrap CI | [0.0%, 3.4%] (n=149) |
| Lines with a candidate distribution | 149 of 149 |
| Scorer readouts, complete / incomplete (never renormalised) | 149 / 0 |
| ECE (10 equal-width bins) | 0.065 |
| ECE, 95% bootstrap CI | [0.034, 0.110] (n=149) |
| Brier (multi-class) | 0.140 |
| Brier, 95% bootstrap CI | [0.075, 0.218] (n=149) |
| ECE / Brier before --calibration | n/a |
| Tokens generated per decision, mean / median | 0.0 / 0.0 |
| Time to first decision, cold / warm median / warm p95 | 346 ms / 355 ms / 360 ms |
| Decision latency, cold / warm median / warm p95 | 346 ms / 355 ms / 360 ms |
| Non-empty think blocks (must be 0) | 0 |

| nvsh outcome | issue 46 JSON |
|---|---|
| propose | `{"action": "tool", "tool": <operation>, "arguments": <arguments>}` |
| explain | `{"action": "no_action"}` |
| escalate | `{"action": "abstain"}` |
| abstain_uncertain | `{"action": "abstain"}` |
| invalid | `{"action": "invalid"}` |

Reporting only: every model is trained and scored on nvsh's propose/explain/escalate tools. Issue 46's abstain is nvsh's escalate, so abstention recall is the escalation recall and abstention precision the strict escalation precision (an escalation on an explain entry counts against it). Explain (answer in words, no tool) has no counterpart in issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for Track B and is never counted as an abstention. abstain_uncertain (issue 53: a confidence gate, not a semantic escalate decision) maps to the same issue-46 abstain action as escalate, and counts the same way in every escalation bar, but is tallied separately in nvsh's own escalation/outcome_counts. An invalid output is not a decision.

## Per-slice calibration

metrics.py's slices by the gold label: read-only and mutating operations (the
operation table's `read_only` flag) and escalate-or-explain entries. Confidence
intervals are seeded percentile bootstraps over the slice's lines.

### `scorer-r3b.q4_k_m`

| Slice | Lines | With a distribution | ECE [95% CI] | Brier [95% CI] | Missing-candidate rate [95% CI] | Offered candidates, mean / median |
|---|---|---|---|---|---|---|
| read_only | 49 | 49 | 0.186 [0.092, 0.302] (n=49) | 0.383 [0.187, 0.601] (n=49) | 0 of 49 [0.0%, 0.0%] (n=49) | 18.0 / 18.0 |
| mutating | 11 | 11 | 0.059 [0.008, 0.139] (n=11) | 0.039 [0.000, 0.114] (n=11) | 0 of 11 [0.0%, 0.0%] (n=11) | 18.0 / 18.0 |
| escalate_or_explain | 89 | 89 | 0.011 [0.001, 0.035] (n=89) | 0.018 [0.000, 0.053] (n=89) | 0 of 89 [0.0%, 0.0%] (n=89) | 18.0 / 18.0 |

#### read_only

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 1 | 0.645 | 1.000 |
| [0.7, 0.8) | 2 | 0.731 | 0.500 |
| [0.8, 0.9) | 2 | 0.882 | 0.500 |
| [0.9, 1.0) | 44 | 0.989 | 0.818 |

#### mutating

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 1 | 0.545 | 1.000 |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 0 | - | - |
| [0.8, 0.9) | 0 | - | - |
| [0.9, 1.0) | 10 | 0.981 | 1.000 |

#### escalate_or_explain

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 1 | 0.732 | 1.000 |
| [0.8, 0.9) | 3 | 0.877 | 0.667 |
| [0.9, 1.0) | 85 | 0.999 | 1.000 |

## Background before each run

### `scorer-r3b.q4_k_m`

Other running containers: model-gear-gateway, model-gear-vllm-embed, model-gear-vllm-rerank, prod-worker-1, q53-edge

`docker ps`:

```text
CONTAINER ID   IMAGE                COMMAND                  CREATED              STATUS                 PORTS                                         NAMES
37e017a19623   f7c67c102b08         "llama-server --mode…"   About a minute ago   Up About a minute      127.0.0.1:18090->8080/tcp                     q53-edge
dfbc45dc003e   lobes-gateway        "python -m lobes.gat…"   6 days ago           Up 2 days (healthy)    0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   model-gear-gateway
1c44dd522337   7c5a10e9a8b3         "bash /usr/local/bin…"   12 days ago          Up 12 days (healthy)   8000/tcp                                      model-gear-vllm-rerank
3bfdffc535f5   7c5a10e9a8b3         "bash /usr/local/bin…"   12 days ago          Up 12 days (healthy)   8000/tcp                                      model-gear-vllm-embed
255b819d6360   culture-nodes:prod   "/nodes worker"          2 weeks ago          Up 2 weeks                                                           prod-worker-1
```

`nvidia-smi`:

```text
Fri Sep 25 22:04:53 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.78                 Driver Version: 595.78         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Orin (nvgpu)                  N/A  |   N/A              N/A |                  N/A |
| N/A   N/A  N/A             N/A  /  N/A  | Not Supported          |     N/A          N/A |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
