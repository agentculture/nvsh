# Tier 2 measurement, 2026-09-24: final-stock

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-train/work/q46/splits/test.json --final --model stock --revision 2fc06364715b967f1860aea9cf38778875588b17 --label final-stock --config $HOME/lfm-train/work/q46/measure/final-stock.nvsh.toml --ctx 2048 --ground-snapshot $HOME/lfm-train/work/q46/ground-snapshot.json --enable-thinking false --max-logprobs 22 --predictions $HOME/lfm-train/work/q46/final/stock`
- Split: `$HOME/lfm-train/work/q46/splits/test.json` (64 entries, 64 sources)
- Seed: 46 (from the split header)
- nvsh: 0.18.0, commit `ff0184462b53ab3614cc09be519b89df9f3180c1`
- Models (repo id @ revision): `stock` @ `2fc06364715b967f1860aea9cf38778875588b17` (operator-supplied, not verified: attached endpoint)
- Tier 2 settings, identical for every run except the model: engine=vllm, mode=attach, image=vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695, ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-train/work/q46/ground-snapshot.json` (platform: fixture world from the split file)
- Serving: engine=vllm, mode=attach, ctx=2048, image=`vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false; log-probabilities: top 20 per generated token
- Decision mode: generative (LfmTier through nvsh.tiers.bench), max-logprobs 22 (operator-supplied)
- Slice: full split
- Ground snapshot: `$HOME/lfm-train/work/q46/ground-snapshot.json` sha256 `0d79c8fe63cef6a9b2e38a1719d173d59c4e793eed5b9ca3fe7b335148dc533b` (263 services, 34 containers; created 2026-09-24)
- Acceptance run: no
- Final run: yes
- Final runs on the test side, including this one: 3

Per-source figures (one vote per `source_id`, majority over its variations, a
tie counts against the model) are the ones claims are judged on; per-variation
figures show paraphrase robustness.

The use-case bar's "0 wrong mutating proposals" is judged on the sum of both rows:
"Wrong mutating proposals, per source / per variation" (bench's own count: a mutating operation other than the
expected one) plus "Mutating proposals with wrong arguments, per source / per variation" (the expected mutating
operation with arguments bench does not accept, e.g. the wrong container).

| Metric | `stock` |
|---|---|
| Model revision | `2fc06364715b967f1860aea9cf38778875588b17` (operator-supplied, not verified: attached endpoint) |
| Right operation and arguments proposed, per source | 1 of 32 |
| Right operation and arguments proposed, per variation | 1 of 32 |
| Should-escalate asks escalated, per source | 15 of 15 |
| Should-escalate asks escalated, per variation | 15 of 15 |
| Wrong mutating proposals, per source / per variation | 0 / 0 |
| Mutating proposals with wrong arguments, per source / per variation | 0 / 0 |
| Explain asks explained, per source | 7 of 17 |
| Explain asks: explained / proposed / escalated, per variation | 7 / 0 / 10 of 17 |
| Mutating proposals on explain asks | 0 |
| Warm latency, median / p95 | 803 ms / 2459 ms |
| First request after start | 18.9 s |
| Container memory (`docker stats`) | not measured |
| Start-up, including first download | 0.0 s |

## Issue 46 metrics

metrics.py over each model's predictions file (one line per entry). The candidate
distribution of a generative run comes from its deciding reply's log-probabilities.

| Metric | `stock` |
|---|---|
| Right proposals (metrics.py) | 2 of 32 |
| Abstention recall (escalate entries escalated) | 6.7% (1 of 15) |
| Abstention precision, strict (deviation d2) | 33.3% |
| False-positive tool calls (proposals on explain/escalate entries) | 3 of 32 |
| Wrong mutating, total (wrong operation + wrong arguments) | 3 (3 + 0) |
| Invalid outputs | 36 of 64 (no_decision: 36) |
| Lines with a candidate distribution | 0 of 64 |
| ECE (10 equal-width bins) | n/a |
| Brier (multi-class) | n/a |
| Tokens generated per decision, mean / median | 105.1 / 72.5 |
| Time to first decision, cold / warm median / warm p95 | 18345 ms / 196 ms / 1899 ms |
| Decision latency, cold / warm median / warm p95 | 18886 ms / 803 ms / 2458 ms |
| Non-empty think blocks (must be 0) | 0 |

- `stock`: lines without a candidate distribution: an alternative token's continuation was not observed: 19; invalid output: 36; no tool call in the deciding reply: 9

| nvsh outcome | issue 46 JSON |
|---|---|
| propose | `{"action": "tool", "tool": <operation>, "arguments": <arguments>}` |
| explain | `{"action": "no_action"}` |
| escalate | `{"action": "abstain"}` |
| invalid | `{"action": "invalid"}` |

Reporting only: every model is trained and scored on nvsh's propose/explain/escalate tools. Issue 46's abstain is nvsh's escalate, so abstention recall is the escalation recall and abstention precision the strict escalation precision (an escalation on an explain entry counts against it). Explain (answer in words, no tool) has no counterpart in issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for Track B and is never counted as an abstention. An invalid output is not a decision.

## Background before each run

### `stock`

Other running containers: eidetic-mongo, eidetic-neo4j, events-mosquitto, model-gear-bluetts, model-gear-gateway, model-gear-realtime, model-gear-stt, q46-measure-18060, qq-mongodb, weather-mongodb, weather-tracker, weather-web

`docker ps`:

```text
CONTAINER ID   IMAGE                            COMMAND                  CREATED          STATUS                PORTS                                                                                                NAMES
e04ef423a1d4   vllm/vllm-openai                 "vllm serve --model …"   2 minutes ago    Up 2 minutes          127.0.0.1:18060->8000/tcp                                                                            q46-measure-18060
0205ea8dabbf   climate-weather-web              "python -m climate.w…"   30 minutes ago   Up 30 minutes         127.0.0.1:8095->8095/tcp                                                                             weather-web
93b52a3f60bb   climate-weather-tracker          "python -m climate.w…"   30 minutes ago   Up 30 minutes                                                                                                              weather-tracker
7df2604debd0   lobes-gateway                    "python -m lobes.gat…"   5 days ago       Up 3 days (healthy)   0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp                                                          model-gear-gateway
57d878edb8b0   lobes-realtime                   "python -m lobes.rea…"   5 days ago       Up 3 days (healthy)   8080/tcp                                                                                             model-gear-realtime
d7919efdf434   lobes-bluetts:local              "python -m lobes.rea…"   5 days ago       Up 3 days (healthy)   9000/tcp                                                                                             model-gear-bluetts
f927adc7c08c   lobes-stt                        "/opt/nvidia/nvidia_…"   5 days ago       Up 3 days (healthy)   127.0.0.1:9002->9002/tcp                                                                             model-gear-stt
0e0f1bec5b41   mongo:8.0                        "docker-entrypoint.s…"   6 days ago       Up 3 days (healthy)   27017/tcp                                                                                            weather-mongodb
e66b11938c79   eclipse-mosquitto:2.1.2-alpine   "/docker-entrypoint.…"   2 months ago     Up 3 days (healthy)   127.0.0.1:1883->1883/tcp                                                                             events-mosquitto
e0810336da84   mongo:8.0                        "docker-entrypoint.s…"   3 months ago     Up 3 days (healthy)   0.0.0.0:27018->27017/tcp, [::]:27018->27017/tcp                                                      eidetic-mongo
a4183a674c55   neo4j:5-community                "tini -g -- /startup…"   3 months ago     Up 3 days (healthy)   0.0.0.0:7474->7474/tcp, [::]:7474->7474/tcp, 7473/tcp, 0.0.0.0:7687->7687/tcp, [::]:7687->7687/tcp   eidetic-neo4j
8f346da46891   mongo:8.0                        "docker-entrypoint.s…"   7 months ago     Up 3 days (healthy)   0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp                                                      qq-mongodb
```

`nvidia-smi`:

```text
Thu Sep 24 09:32:10 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0  On |                  N/A |
| N/A   50C    P0             12W /  N/A  | Not Supported          |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5459      C   python3                                2079MiB |
|    0   N/A  N/A            8099      G   /usr/lib/xorg/Xorg                      315MiB |
|    0   N/A  N/A            8320      G   /usr/bin/gnome-shell                    203MiB |
|    0   N/A  N/A            8856      G   ...exec/xdg-desktop-portal-gnome         57MiB |
|    0   N/A  N/A           10315      G   /usr/bin/ghostty                        808MiB |
|    0   N/A  N/A          147875      G   ...rack-uuid=3190708988185955192        236MiB |
|    0   N/A  N/A         2632314      C   VLLM::EngineCore                       8204MiB |
|    0   N/A  N/A         3732597      G   /usr/bin/nautilus                        39MiB |
|    0   N/A  N/A         3733171      G   .../8862/usr/lib/firefox/firefox        186MiB |
|    0   N/A  N/A         3734121      G   /usr/bin/gnome-text-editor               43MiB |
+-----------------------------------------------------------------------------------------+
```
