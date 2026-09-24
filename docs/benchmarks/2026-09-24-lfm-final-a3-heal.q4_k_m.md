# Tier 2 measurement, 2026-09-24: final-a3-heal.q4_k_m

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-train/work/q46/splits/test.json --final --model a3-heal.q4_k_m --revision sha256:8dfd6ddc788c4d70 --label final-a3-heal.q4_k_m --config $HOME/lfm-train/work/q46/measure/final-a3-heal.q4_k_m.nvsh.toml --ctx 2048 --ground-snapshot $HOME/lfm-train/work/q46/ground-snapshot.json --enable-thinking false --max-logprobs 22 --predictions $HOME/lfm-train/work/q46/final/a3-heal.q4_k_m`
- Split: `$HOME/lfm-train/work/q46/splits/test.json` (64 entries, 64 sources)
- Seed: 46 (from the split header)
- nvsh: 0.18.0, commit `a368ab1cf4554700e67668cebc8c4e45d8eb037b`
- Models (repo id @ revision): `a3-heal.q4_k_m` @ `sha256:8dfd6ddc788c4d70` (operator-supplied, not verified: attached endpoint)
- Tier 2 settings, identical for every run except the model: engine=llama-server, mode=attach, image=native llama-server /home/spark/lfm-train/llama.cpp/build/bin/llama-server (0.00.000.404 I srv llama_server: initializing ... version: 0.4.1-dev (build 11132, commit 633733d0a) built with GNU 13.3.0 for Linux aarch64), ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-train/work/q46/ground-snapshot.json` (platform: fixture world from the split file)
- Serving: engine=llama-server, mode=attach, ctx=2048, image=`native llama-server /home/spark/lfm-train/llama.cpp/build/bin/llama-server (0.00.000.404 I srv llama_server: initializing ... version: 0.4.1-dev (build 11132, commit 633733d0a) built with GNU 13.3.0 for Linux aarch64)`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false; log-probabilities: top 20 per generated token
- Decision mode: generative (LfmTier through nvsh.tiers.bench), max-logprobs 22 (operator-supplied)
- Slice: full split
- Ground snapshot: `$HOME/lfm-train/work/q46/ground-snapshot.json` sha256 `0d79c8fe63cef6a9b2e38a1719d173d59c4e793eed5b9ca3fe7b335148dc533b` (263 services, 34 containers; created 2026-09-24)
- Acceptance run: no
- Final run: yes
- Final runs on the test side, including this one: 18

Per-source figures (one vote per `source_id`, majority over its variations, a
tie counts against the model) are the ones claims are judged on; per-variation
figures show paraphrase robustness.

The use-case bar's "0 wrong mutating proposals" is judged on the sum of both rows:
"Wrong mutating proposals, per source / per variation" (bench's own count: a mutating operation other than the
expected one) plus "Mutating proposals with wrong arguments, per source / per variation" (the expected mutating
operation with arguments bench does not accept, e.g. the wrong container).

| Metric | `a3-heal.q4_k_m` |
|---|---|
| Model revision | `sha256:8dfd6ddc788c4d70` (operator-supplied, not verified: attached endpoint) |
| Right operation and arguments proposed, per source | 31 of 32 |
| Right operation and arguments proposed, per variation | 31 of 32 |
| Should-escalate asks escalated, per source | 13 of 15 |
| Should-escalate asks escalated, per variation | 13 of 15 |
| Wrong mutating proposals, per source / per variation | 1 / 1 |
| Mutating proposals with wrong arguments, per source / per variation | 0 / 0 |
| Explain asks explained, per source | 15 of 17 |
| Explain asks: explained / proposed / escalated, per variation | 15 / 1 / 1 of 17 |
| Mutating proposals on explain asks | 0 |
| Warm latency, median / p95 | 245 ms / 405 ms |
| First request after start | 0.3 s |
| Container memory (`docker stats`) | not measured |
| Start-up, including first download | 0.0 s |

## Issue 46 metrics

metrics.py over each model's predictions file (one line per entry). The candidate
distribution of a generative run comes from its deciding reply's log-probabilities.

| Metric | `a3-heal.q4_k_m` |
|---|---|
| Right proposals (metrics.py) | 32 of 32 |
| Abstention recall (escalate entries escalated) | 73.3% (11 of 15) |
| Abstention precision, strict (deviation d2) | 91.7% |
| False-positive tool calls (proposals on explain/escalate entries) | 4 of 32 |
| Wrong mutating, total (wrong operation + wrong arguments) | 2 (2 + 0) |
| Invalid outputs | 0 of 64 |
| Lines with a candidate distribution | 0 of 64 |
| ECE (10 equal-width bins) | n/a |
| Brier (multi-class) | n/a |
| Tokens generated per decision, mean / median | 45.4 / 40.0 |
| Time to first decision, cold / warm median / warm p95 | 347 ms / 245 ms / 404 ms |
| Decision latency, cold / warm median / warm p95 | 348 ms / 245 ms / 405 ms |
| Non-empty think blocks (must be 0) | 0 |

- `a3-heal.q4_k_m`: lines without a candidate distribution: an alternative token's continuation was not observed: 64

| nvsh outcome | issue 46 JSON |
|---|---|
| propose | `{"action": "tool", "tool": <operation>, "arguments": <arguments>}` |
| explain | `{"action": "no_action"}` |
| escalate | `{"action": "abstain"}` |
| invalid | `{"action": "invalid"}` |

Reporting only: every model is trained and scored on nvsh's propose/explain/escalate tools. Issue 46's abstain is nvsh's escalate, so abstention recall is the escalation recall and abstention precision the strict escalation precision (an escalation on an explain entry counts against it). Explain (answer in words, no tool) has no counterpart in issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for Track B and is never counted as an abstention. An invalid output is not a decision.

## Background before each run

### `a3-heal.q4_k_m`

Other running containers: eidetic-mongo, eidetic-neo4j, events-mosquitto, model-gear-bluetts, model-gear-gateway, model-gear-realtime, model-gear-stt, qq-mongodb, weather-mongodb, weather-tracker, weather-web

`docker ps`:

```text
CONTAINER ID   IMAGE                            COMMAND                  CREATED          STATUS                PORTS                                                                                                NAMES
93050d5259c1   climate-weather-tracker          "python -m climate.w…"   44 minutes ago   Up 44 minutes                                                                                                              weather-tracker
21571f98493d   climate-weather-web              "python -m climate.w…"   44 minutes ago   Up 44 minutes         127.0.0.1:8095->8095/tcp                                                                             weather-web
7df2604debd0   lobes-gateway                    "python -m lobes.gat…"   5 days ago       Up 4 days (healthy)   0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp                                                          model-gear-gateway
57d878edb8b0   lobes-realtime                   "python -m lobes.rea…"   5 days ago       Up 4 days (healthy)   8080/tcp                                                                                             model-gear-realtime
d7919efdf434   lobes-bluetts:local              "python -m lobes.rea…"   5 days ago       Up 4 days (healthy)   9000/tcp                                                                                             model-gear-bluetts
f927adc7c08c   lobes-stt                        "/opt/nvidia/nvidia_…"   5 days ago       Up 4 days (healthy)   127.0.0.1:9002->9002/tcp                                                                             model-gear-stt
0e0f1bec5b41   mongo:8.0                        "docker-entrypoint.s…"   6 days ago       Up 4 days (healthy)   27017/tcp                                                                                            weather-mongodb
e66b11938c79   eclipse-mosquitto:2.1.2-alpine   "/docker-entrypoint.…"   2 months ago     Up 4 days (healthy)   127.0.0.1:1883->1883/tcp                                                                             events-mosquitto
e0810336da84   mongo:8.0                        "docker-entrypoint.s…"   3 months ago     Up 4 days (healthy)   0.0.0.0:27018->27017/tcp, [::]:27018->27017/tcp                                                      eidetic-mongo
a4183a674c55   neo4j:5-community                "tini -g -- /startup…"   3 months ago     Up 4 days (healthy)   0.0.0.0:7474->7474/tcp, [::]:7474->7474/tcp, 7473/tcp, 0.0.0.0:7687->7687/tcp, [::]:7687->7687/tcp   eidetic-neo4j
8f346da46891   mongo:8.0                        "docker-entrypoint.s…"   7 months ago     Up 4 days (healthy)   0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp                                                      qq-mongodb
```

`nvidia-smi`:

```text
Thu Sep 24 11:59:55 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   53C    P0             12W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5459      C   python3                                2079MiB |
|    0   N/A  N/A            8099      G   /usr/lib/xorg/Xorg                      338MiB |
|    0   N/A  N/A            8320      G   /usr/bin/gnome-shell                    206MiB |
|    0   N/A  N/A            8856      G   ...exec/xdg-desktop-portal-gnome         57MiB |
|    0   N/A  N/A           10315      G   /usr/bin/ghostty                        839MiB |
|    0   N/A  N/A          147875      G   ...rack-uuid=3190708988185955192        184MiB |
|    0   N/A  N/A         3294715      C   ...ma.cpp/build/bin/llama-server        829MiB |
|    0   N/A  N/A         3732597      G   /usr/bin/nautilus                        41MiB |
|    0   N/A  N/A         3733171      G   .../8862/usr/lib/firefox/firefox        186MiB |
|    0   N/A  N/A         3734121      G   /usr/bin/gnome-text-editor               42MiB |
+-----------------------------------------------------------------------------------------+
```
