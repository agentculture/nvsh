# Tier 2 measurement, 2026-09-24: edge-orin-a3-heal.q4_k_m-gpu

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-edge/q46/work/splits/val.json --model a3-heal.q4_k_m --revision sha256:8bd6ebe4009ac90d --label edge-orin-a3-heal.q4_k_m-gpu --config $HOME/lfm-edge/q46/out/a3-heal.q4_k_m.toml --ctx 2048 --ground-snapshot $HOME/lfm-edge/q46/work/ground-snapshot.json --enable-thinking false --max-logprobs 22 --out $HOME/lfm-edge/q46/out/edge-orin-a3-heal.q4_k_m-gpu.md --predictions $HOME/lfm-edge/q46/out/pred`
- Split: `$HOME/lfm-edge/q46/work/splits/val.json` (66 entries, 66 sources)
- Seed: 46 (from the split header)
- nvsh: 0.18.0, commit `unknown`
- Models (repo id @ revision): `a3-heal.q4_k_m` @ `sha256:8bd6ebe4009ac90d` (operator-supplied, not verified: attached endpoint)
- Tier 2 settings, identical for every run except the model: engine=llama-server, mode=attach, image=ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191 (llama.cpp 10373 38406d597, Jetson AGX Orin, JetPack R39), ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-edge/q46/work/ground-snapshot.json` (platform: fixture world from the split file)
- Serving: engine=llama-server, mode=attach, ctx=2048, image=`ghcr.io/nvidia-ai-iot/llama_cpp@sha256:f7c67c102b08252e963f9e5f92c3a36554c8f69305eb7ea257c6cd12e24c3191 (llama.cpp 10373 38406d597, Jetson AGX Orin, JetPack R39)`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false; log-probabilities: top 20 per generated token
- Decision mode: generative (LfmTier through nvsh.tiers.bench), max-logprobs 22 (operator-supplied)
- Slice: full split
- Ground snapshot: `$HOME/lfm-edge/q46/work/ground-snapshot.json` sha256 `0d79c8fe63cef6a9b2e38a1719d173d59c4e793eed5b9ca3fe7b335148dc533b` (263 services, 34 containers; created 2026-09-24)
- Acceptance run: no
- Final run: no

Per-source figures (one vote per `source_id`, majority over its variations, a
tie counts against the model) are the ones claims are judged on; per-variation
figures show paraphrase robustness.

The use-case bar's "0 wrong mutating proposals" is judged on the sum of both rows:
"Wrong mutating proposals, per source / per variation" (bench's own count: a mutating operation other than the
expected one) plus "Mutating proposals with wrong arguments, per source / per variation" (the expected mutating
operation with arguments bench does not accept, e.g. the wrong container).

| Metric | `a3-heal.q4_k_m` |
|---|---|
| Model revision | `sha256:8bd6ebe4009ac90d` (operator-supplied, not verified: attached endpoint) |
| Right operation and arguments proposed, per source | 30 of 32 |
| Right operation and arguments proposed, per variation | 30 of 32 |
| Should-escalate asks escalated, per source | 13 of 16 |
| Should-escalate asks escalated, per variation | 13 of 16 |
| Wrong mutating proposals, per source / per variation | 1 / 1 |
| Mutating proposals with wrong arguments, per source / per variation | 0 / 0 |
| Explain asks explained, per source | 18 of 18 |
| Explain asks: explained / proposed / escalated, per variation | 18 / 0 / 0 of 18 |
| Mutating proposals on explain asks | 0 |
| Warm latency, median / p95 | 539 ms / 956 ms |
| First request after start | 0.9 s |
| Container memory (`docker stats`) | not measured |
| Start-up, including first download | 0.0 s |

## Issue 46 metrics

metrics.py over each model's predictions file (one line per entry). The candidate
distribution of a generative run comes from its deciding reply's log-probabilities.

| Metric | `a3-heal.q4_k_m` |
|---|---|
| Right proposals (metrics.py) | 32 of 32 |
| Abstention recall (escalate entries escalated) | 81.2% (13 of 16) |
| Abstention precision, strict (deviation d2) | 100.0% |
| False-positive tool calls (proposals on explain/escalate entries) | 2 of 34 |
| Wrong mutating, total (wrong operation + wrong arguments) | 1 (1 + 0) |
| Invalid outputs | 0 of 66 |
| Lines with a candidate distribution | 0 of 66 |
| ECE (10 equal-width bins) | n/a |
| Brier (multi-class) | n/a |
| Tokens generated per decision, mean / median | 46.6 / 40.0 |
| Time to first decision, cold / warm median / warm p95 | 901 ms / 537 ms / 954 ms |
| Decision latency, cold / warm median / warm p95 | 902 ms / 538 ms / 956 ms |
| Non-empty think blocks (must be 0) | 0 |

- `a3-heal.q4_k_m`: lines without a candidate distribution: an alternative token's continuation was not observed: 66

| nvsh outcome | issue 46 JSON |
|---|---|
| propose | `{"action": "tool", "tool": <operation>, "arguments": <arguments>}` |
| explain | `{"action": "no_action"}` |
| escalate | `{"action": "abstain"}` |
| invalid | `{"action": "invalid"}` |

Reporting only: every model is trained and scored on nvsh's propose/explain/escalate tools. Issue 46's abstain is nvsh's escalate, so abstention recall is the escalation recall and abstention precision the strict escalation precision (an escalation on an explain entry counts against it). Explain (answer in words, no tool) has no counterpart in issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for Track B and is never counted as an abstention. An invalid output is not a decision.

## Background before each run

### `a3-heal.q4_k_m`

Other running containers: model-gear-gateway, model-gear-vllm-embed, model-gear-vllm-rerank, prod-worker-1, q46-edge

`docker ps`:

```text
CONTAINER ID   IMAGE                COMMAND                  CREATED          STATUS                  PORTS                                         NAMES
cd47aa6df971   f7c67c102b08         "llama-server --mode…"   20 seconds ago   Up 20 seconds           127.0.0.1:18090->8080/tcp                     q46-edge
dfbc45dc003e   lobes-gateway        "python -m lobes.gat…"   5 days ago       Up 21 hours (healthy)   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   model-gear-gateway
1c44dd522337   7c5a10e9a8b3         "bash /usr/local/bin…"   11 days ago      Up 11 days (healthy)    8000/tcp                                      model-gear-vllm-rerank
3bfdffc535f5   7c5a10e9a8b3         "bash /usr/local/bin…"   11 days ago      Up 11 days (healthy)    8000/tcp                                      model-gear-vllm-embed
255b819d6360   culture-nodes:prod   "/nodes worker"          2 weeks ago      Up 2 weeks                                                            prod-worker-1
```

`nvidia-smi`:

```text
Thu Sep 24 12:02:05 2026       
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
