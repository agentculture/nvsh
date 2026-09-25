# Tier 2 measurement, 2026-09-25: edge-orin-scorer-r3b.q4_k_m-heldout-mc

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-edge/q53/work/splits/held-out-q53.sealed.json --model scorer-r3b.q4_k_m --revision sha256:3568f660ab186887 --label edge-orin-scorer-r3b.q4_k_m-heldout-mc --config $HOME/lfm-edge/q53/out/scorer-r3b.q4_k_m.toml --ctx 2048 --ground-snapshot $HOME/lfm-edge/q53/work/ground-snapshot.json --enable-thinking false --max-logprobs 20000 --scorer served --tokenizer $HOME/lfm-edge/q53/tok/scorer-r3b --slice missing-candidate --acceptance --out $HOME/lfm-edge/q53/out/edge-orin-scorer-r3b.q4_k_m-heldout-mc.md --predictions $HOME/lfm-edge/q53/out/pred`
- Split: `$HOME/lfm-edge/q53/work/splits/held-out-q53.sealed.json` (60 entries, 60 sources)
- Seed: not recorded
- nvsh: 0.19.1, commit `unknown`
- Models (repo id @ revision): `scorer-r3b.q4_k_m` @ `sha256:3568f660ab186887` (operator-supplied, not verified: attached endpoint)
- Tier 2 settings, identical for every run except the model: engine=llama-server, mode=attach, image=ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191 (llama.cpp 10373 38406d597, Jetson AGX Orin, JetPack R39), ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-edge/q53/work/ground-snapshot.json` (platform: fixture world from `nvsh/tiers/corpus/dev.json`)
- Serving: engine=llama-server, mode=attach, ctx=2048, image=`ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191 (llama.cpp 10373 38406d597, Jetson AGX Orin, JetPack R39)`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false (the scorer renders its own prompts, thinking off)
- Decision mode: candidate scorer (served) through scorer.py, max-logprobs 20000 (operator-supplied); asks for 20000 per request
- Slice: missing-candidate (eval_slices.py: each operation entry with its gold operation left out of the offered candidates, expected to escalate)
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
| Right proposals (metrics.py) | 0 of 0 |
| Right proposals, 95% bootstrap CI | n/a (n=0) |
| Abstention recall (escalate entries escalated) | 76.7% (46 of 60) |
| Abstention recall, 95% bootstrap CI | [65.0%, 86.7%] (n=60) |
| Abstention precision, strict (deviation d2) | 100.0% |
| Abstention precision, strict, 95% bootstrap CI | [100.0%, 100.0%] (n=60) |
| False-positive tool calls (proposals on explain/escalate entries) | 7 of 60 |
| False-positive tool calls, 95% bootstrap CI | [5.0%, 20.0%] (n=60) |
| Wrong mutating, total (wrong operation + wrong arguments) | 0 (0 + 0) |
| Invalid outputs | 2 of 60 (not_grounded: 2) |
| Invalid outputs, 95% bootstrap CI | [0.0%, 8.3%] (n=60) |
| Lines with a candidate distribution | 60 of 60 |
| Scorer readouts, complete / incomplete (never renormalised) | 60 / 0 |
| ECE (10 equal-width bins) | 0.205 |
| ECE, 95% bootstrap CI | [0.128, 0.314] (n=60) |
| Brier (multi-class) | 0.399 |
| Brier, 95% bootstrap CI | [0.234, 0.596] (n=60) |
| ECE / Brier before --calibration | n/a |
| Tokens generated per decision, mean / median | 0.0 / 0.0 |
| Time to first decision, cold / warm median / warm p95 | 359 ms / 349 ms / 362 ms |
| Decision latency, cold / warm median / warm p95 | 359 ms / 349 ms / 362 ms |
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
| read_only | 0 | 0 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a / n/a |
| mutating | 0 | 0 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a / n/a |
| escalate_or_explain | 60 | 60 | 0.205 [0.128, 0.314] (n=60) | 0.399 [0.234, 0.596] (n=60) | 60 of 60 [100.0%, 100.0%] (n=60) | 17.0 / 17.0 |

#### read_only

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

#### mutating

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

#### escalate_or_explain

| bin | n | confidence | accuracy |
| --- | --- | --- | --- |
| [0.0, 0.1) | 0 | - | - |
| [0.1, 0.2) | 0 | - | - |
| [0.2, 0.3) | 0 | - | - |
| [0.3, 0.4) | 1 | 0.356 | 0.000 |
| [0.4, 0.5) | 1 | 0.491 | 1.000 |
| [0.5, 0.6) | 1 | 0.563 | 0.000 |
| [0.6, 0.7) | 1 | 0.682 | 0.000 |
| [0.7, 0.8) | 1 | 0.789 | 1.000 |
| [0.8, 0.9) | 4 | 0.840 | 0.750 |
| [0.9, 1.0) | 51 | 0.992 | 0.804 |

## Background before each run

### `scorer-r3b.q4_k_m`

Other running containers: model-gear-gateway, model-gear-vllm-embed, model-gear-vllm-rerank, prod-worker-1, q53-edge

`docker ps`:

```text
CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS                 PORTS                                         NAMES
37e017a19623   f7c67c102b08         "llama-server --mode…"   2 minutes ago   Up 2 minutes           127.0.0.1:18090->8080/tcp                     q53-edge
dfbc45dc003e   lobes-gateway        "python -m lobes.gat…"   6 days ago      Up 2 days (healthy)    0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   model-gear-gateway
1c44dd522337   7c5a10e9a8b3         "bash /usr/local/bin…"   12 days ago     Up 12 days (healthy)   8000/tcp                                      model-gear-vllm-rerank
3bfdffc535f5   7c5a10e9a8b3         "bash /usr/local/bin…"   12 days ago     Up 12 days (healthy)   8000/tcp                                      model-gear-vllm-embed
255b819d6360   culture-nodes:prod   "/nodes worker"          2 weeks ago     Up 2 weeks                                                           prod-worker-1
```

`nvidia-smi`:

```text
Fri Sep 25 22:05:52 2026       
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
