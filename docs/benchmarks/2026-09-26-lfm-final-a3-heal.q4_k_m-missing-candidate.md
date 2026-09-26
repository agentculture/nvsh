# Tier 2 measurement, 2026-09-26: final-a3-heal.q4_k_m-missing-candidate

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-train/work/i64-a3heal/splits/test.json --final --model a3-heal.q4_k_m --revision sha256:8dfd6ddc788c4d70 --label final-a3-heal.q4_k_m-missing-candidate --config $HOME/lfm-train/work/i64-a3heal/measure/final-a3-heal.q4_k_m-missing-candidate.nvsh.toml --ctx 2048 --ground-snapshot $HOME/lfm-train/work/q53-run/ground-snapshot.json --enable-thinking false --max-logprobs 22 --predictions $HOME/lfm-train/work/i64-a3heal/final/a3-heal.q4_k_m --slice missing-candidate`
- Split: `$HOME/lfm-train/work/i64-a3heal/splits/test.json` (83 entries, 83 sources)
- Seed: 53 (from the split header)
- nvsh: 0.20.0, commit `95952419060356ae2a561e30c690ea6607ae9ec2`
- Models (repo id @ revision): `a3-heal.q4_k_m` @ `sha256:8dfd6ddc788c4d70` (operator-supplied, not verified: attached endpoint)
- Tier 2 settings, identical for every run except the model: engine=llama-server, mode=attach, image=native llama-server $HOME/lfm-train/llama.cpp/build/bin/llama-server (0.00.000.388 I srv llama_server: initializing ... version: 0.4.1-dev (build 11132, commit 633733d0a) built with GNU 13.3.0 for Linux aarch64), ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-train/work/q53-run/ground-snapshot.json` (platform: fixture world from `nvsh/tiers/corpus/dev.json`)
- Serving: engine=llama-server, mode=attach, ctx=2048, image=`native llama-server $HOME/lfm-train/llama.cpp/build/bin/llama-server (0.00.000.388 I srv llama_server: initializing ... version: 0.4.1-dev (build 11132, commit 633733d0a) built with GNU 13.3.0 for Linux aarch64)`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false; log-probabilities: top 20 per generated token
- Decision mode: generative (LfmTier through nvsh.tiers.bench), max-logprobs 22 (operator-supplied)
- Slice: missing-candidate (eval_slices.py: each operation entry with its gold operation left out of the offered candidates, expected to escalate)
- Ground snapshot: `$HOME/lfm-train/work/q53-run/ground-snapshot.json` sha256 `a6e21fee60571a2c9431f5145b2b8be5829a721d2661725733cbf5d32f63aa5a` (277 services, 41 containers; created 2026-09-25)
- Calibration: none (the model's own candidate distributions)
- Acceptance run: no
- Final run: yes
- Final runs on the test side, including this one: 26

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
| Right operation and arguments proposed, per source | 0 of 0 |
| Right operation and arguments proposed, per variation | 0 of 0 |
| Should-escalate asks escalated, per source | 30 of 83 |
| Should-escalate asks escalated, per variation | 30 of 83 |
| Wrong mutating proposals, per source / per variation | 4 / 4 |
| Mutating proposals with wrong arguments, per source / per variation | 0 / 0 |
| Explain asks explained, per source | 0 of 0 |
| Explain asks: explained / proposed / escalated, per variation | 0 / 0 / 0 of 0 |
| Mutating proposals on explain asks | 0 |
| Warm latency, median / p95 | 274 ms / 947 ms |
| First request after start | 0.4 s |
| Container memory (`docker stats`) | not measured |
| Start-up, including first download | 0.0 s |

## Issue 46 metrics

metrics.py over each model's predictions file (one line per entry). The candidate
distribution of a generative run comes from its deciding reply's log-probabilities.

| Metric | `a3-heal.q4_k_m` |
|---|---|
| Right proposals (metrics.py) | 0 of 0 |
| Right proposals, 95% bootstrap CI | n/a (n=0) |
| Abstention recall (escalate entries escalated) | 12.0% (10 of 83) |
| Abstention recall, 95% bootstrap CI | [4.8%, 19.3%] (n=83) |
| Abstention precision, strict (deviation d2) | 100.0% |
| Abstention precision, strict, 95% bootstrap CI | [100.0%, 100.0%] (n=83) |
| False-positive tool calls (proposals on explain/escalate entries) | 60 of 83 |
| False-positive tool calls, 95% bootstrap CI | [62.7%, 81.9%] (n=83) |
| Wrong mutating, total (wrong operation + wrong arguments) | 11 (11 + 0) |
| Invalid outputs | 11 of 83 (no_decision: 11) |
| Invalid outputs, 95% bootstrap CI | [6.0%, 20.5%] (n=83) |
| Lines with a candidate distribution | 0 of 83 |
| Scorer readouts, complete / incomplete (never renormalised) | n/a |
| ECE (10 equal-width bins) | n/a |
| ECE, 95% bootstrap CI | n/a (n=0) |
| Brier (multi-class) | n/a |
| Brier, 95% bootstrap CI | n/a (n=0) |
| ECE / Brier before --calibration | n/a |
| Tokens generated per decision, mean / median | 61.9 / 40.0 |
| Time to first decision, cold / warm median / warm p95 | 352 ms / 243 ms / 362 ms |
| Decision latency, cold / warm median / warm p95 | 352 ms / 274 ms / 947 ms |
| Non-empty think blocks (must be 0) | 0 |

- `a3-heal.q4_k_m`: lines without a candidate distribution: an alternative token's continuation was not observed: 72; invalid output: 11

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

### `a3-heal.q4_k_m`

| Slice | Lines | With a distribution | ECE [95% CI] | Brier [95% CI] | Missing-candidate rate [95% CI] | Offered candidates, mean / median |
|---|---|---|---|---|---|---|
| read_only | 0 | 0 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a / n/a |
| mutating | 0 | 0 | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a / n/a |
| escalate_or_explain | 83 | 0 | n/a (n=0) | n/a (n=0) | 83 of 83 [100.0%, 100.0%] (n=83) | n/a / n/a |

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
| [0.3, 0.4) | 0 | - | - |
| [0.4, 0.5) | 0 | - | - |
| [0.5, 0.6) | 0 | - | - |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 0 | - | - |
| [0.8, 0.9) | 0 | - | - |
| [0.9, 1.0) | 0 | - | - |

## Background before each run

### `a3-heal.q4_k_m`

Other running containers: eidetic-mongo, eidetic-neo4j, events-mosquitto, model-gear-bluetts, model-gear-gateway, model-gear-realtime, model-gear-stt, model-gear-vllm-multimodal, qq-mongodb, weather-mongodb, weather-tracker, weather-web

`docker ps`:

```text
CONTAINER ID   IMAGE                            COMMAND                  CREATED        STATUS                  PORTS                                                                                                NAMES
5b2e75694123   climate-weather-tracker          "python -m climate.w…"   35 hours ago   Up 35 hours                                                                                                                  weather-tracker
2b2b3ee7a088   climate-weather-web              "python -m climate.w…"   35 hours ago   Up 35 hours             127.0.0.1:8095->8095/tcp                                                                             weather-web
a7c04673c948   lobes/vllm-gemma4:local          "bash /usr/local/bin…"   7 days ago     Up 43 hours (healthy)   8000/tcp                                                                                             model-gear-vllm-multimodal
7df2604debd0   lobes-gateway                    "python -m lobes.gat…"   7 days ago     Up 5 days (healthy)     0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp                                                          model-gear-gateway
57d878edb8b0   lobes-realtime                   "python -m lobes.rea…"   7 days ago     Up 5 days (healthy)     8080/tcp                                                                                             model-gear-realtime
d7919efdf434   lobes-bluetts:local              "python -m lobes.rea…"   7 days ago     Up 5 days (healthy)     9000/tcp                                                                                             model-gear-bluetts
f927adc7c08c   lobes-stt                        "/opt/nvidia/nvidia_…"   7 days ago     Up 5 days (healthy)     127.0.0.1:9002->9002/tcp                                                                             model-gear-stt
0e0f1bec5b41   mongo:8.0                        "docker-entrypoint.s…"   8 days ago     Up 5 days (healthy)     27017/tcp                                                                                            weather-mongodb
e66b11938c79   eclipse-mosquitto:2.1.2-alpine   "/docker-entrypoint.…"   2 months ago   Up 5 days (healthy)     127.0.0.1:1883->1883/tcp                                                                             events-mosquitto
e0810336da84   mongo:8.0                        "docker-entrypoint.s…"   3 months ago   Up 5 days (healthy)     0.0.0.0:27018->27017/tcp, [::]:27018->27017/tcp                                                      eidetic-mongo
a4183a674c55   neo4j:5-community                "tini -g -- /startup…"   3 months ago   Up 5 days (healthy)     0.0.0.0:7474->7474/tcp, [::]:7474->7474/tcp, 7473/tcp, 0.0.0.0:7687->7687/tcp, [::]:7687->7687/tcp   eidetic-neo4j
8f346da46891   mongo:8.0                        "docker-entrypoint.s…"   7 months ago   Up 5 days (healthy)     0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp                                                      qq-mongodb
```

`nvidia-smi`:

```text
Sat Sep 26 08:53:07 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   54C    P0             12W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5459      C   python3                                2079MiB |
|    0   N/A  N/A            8099      G   /usr/lib/xorg/Xorg                      456MiB |
|    0   N/A  N/A            8320      G   /usr/bin/gnome-shell                    206MiB |
|    0   N/A  N/A            8856      G   ...exec/xdg-desktop-portal-gnome         61MiB |
|    0   N/A  N/A           10315      G   /usr/bin/ghostty                        873MiB |
|    0   N/A  N/A          147875      G   ...rack-uuid=3190708988185955192        356MiB |
|    0   N/A  N/A         3250722      C   ...ma.cpp/build/bin/llama-server        829MiB |
|    0   N/A  N/A         3699287      C   VLLM::EngineCore                      33790MiB |
|    0   N/A  N/A         3732597      G   /usr/bin/nautilus                        41MiB |
|    0   N/A  N/A         3733171      G   .../8862/usr/lib/firefox/firefox        186MiB |
|    0   N/A  N/A         3734121      G   /usr/bin/gnome-text-editor               43MiB |
+-----------------------------------------------------------------------------------------+
```
