# Tier 2 measurement, 2026-09-25: heldout-scorer-r3b.q4_k_m-missing-candidate

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-train/work/heldout-q53/held-out-q53.sealed.json --acceptance --model scorer-r3b.q4_k_m --revision sha256:52061cec322242c4 --label heldout-scorer-r3b.q4_k_m-missing-candidate --config $HOME/lfm-train/work/q53-run/measure/heldout-scorer-r3b.q4_k_m-missing-candidate.nvsh.toml --ctx 2048 --ground-snapshot $HOME/lfm-train/work/q53-run/ground-snapshot.json --enable-thinking false --max-logprobs 20000 --predictions $HOME/lfm-train/work/q53-run/final/scorer-r3b.q4_k_m --tokenizer $HOME/lfm-train/work/q53-run/runs/scorer-r3b/merged --scorer served --slice missing-candidate`
- Split: `$HOME/lfm-train/work/heldout-q53/held-out-q53.sealed.json` (60 entries, 60 sources)
- Seed: not recorded
- nvsh: 0.19.1, commit `8976a6a38cfdbab0b8f79210f72f189df5c61bb1`
- Models (repo id @ revision): `scorer-r3b.q4_k_m` @ `sha256:52061cec322242c4` (operator-supplied, not verified: attached endpoint)
- Tier 2 settings, identical for every run except the model: engine=llama-server, mode=attach, image=native llama-server $HOME/lfm-train/llama.cpp/build/bin/llama-server (0.00.000.451 I srv llama_server: initializing ... version: 0.4.1-dev (build 11132, commit 633733d0a) built with GNU 13.3.0 for Linux aarch64), ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-train/work/q53-run/ground-snapshot.json` (platform: fixture world from `nvsh/tiers/corpus/dev.json`)
- Serving: engine=llama-server, mode=attach, ctx=2048, image=`native llama-server $HOME/lfm-train/llama.cpp/build/bin/llama-server (0.00.000.451 I srv llama_server: initializing ... version: 0.4.1-dev (build 11132, commit 633733d0a) built with GNU 13.3.0 for Linux aarch64)`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false (the scorer renders its own prompts, thinking off)
- Decision mode: candidate scorer (served) through scorer.py, max-logprobs 20000 (operator-supplied); asks for 20000 per request
- Slice: missing-candidate (eval_slices.py: each operation entry with its gold operation left out of the offered candidates, expected to escalate)
- Ground snapshot: `$HOME/lfm-train/work/q53-run/ground-snapshot.json` sha256 `a6e21fee60571a2c9431f5145b2b8be5829a721d2661725733cbf5d32f63aa5a` (277 services, 41 containers; created 2026-09-25)
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
| Abstention recall (escalate entries escalated) | 75.0% (45 of 60) |
| Abstention recall, 95% bootstrap CI | [63.3%, 85.0%] (n=60) |
| Abstention precision, strict (deviation d2) | 100.0% |
| Abstention precision, strict, 95% bootstrap CI | [100.0%, 100.0%] (n=60) |
| False-positive tool calls (proposals on explain/escalate entries) | 8 of 60 |
| False-positive tool calls, 95% bootstrap CI | [5.0%, 23.3%] (n=60) |
| Wrong mutating, total (wrong operation + wrong arguments) | 0 (0 + 0) |
| Invalid outputs | 2 of 60 (not_grounded: 2) |
| Invalid outputs, 95% bootstrap CI | [0.0%, 8.3%] (n=60) |
| Lines with a candidate distribution | 60 of 60 |
| Scorer readouts, complete / incomplete (never renormalised) | 60 / 0 |
| ECE (10 equal-width bins) | 0.204 |
| ECE, 95% bootstrap CI | [0.126, 0.314] (n=60) |
| Brier (multi-class) | 0.406 |
| Brier, 95% bootstrap CI | [0.240, 0.601] (n=60) |
| ECE / Brier before --calibration | n/a |
| Tokens generated per decision, mean / median | 0.0 / 0.0 |
| Time to first decision, cold / warm median / warm p95 | 144 ms / 119 ms / 167 ms |
| Decision latency, cold / warm median / warm p95 | 144 ms / 119 ms / 167 ms |
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
| escalate_or_explain | 60 | 60 | 0.204 [0.126, 0.314] (n=60) | 0.406 [0.240, 0.601] (n=60) | 60 of 60 [100.0%, 100.0%] (n=60) | 17.0 / 17.0 |

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
| [0.4, 0.5) | 1 | 0.426 | 0.000 |
| [0.5, 0.6) | 2 | 0.586 | 0.000 |
| [0.6, 0.7) | 0 | - | - |
| [0.7, 0.8) | 2 | 0.747 | 0.500 |
| [0.8, 0.9) | 3 | 0.862 | 0.667 |
| [0.9, 1.0) | 52 | 0.991 | 0.808 |

## Background before each run

### `scorer-r3b.q4_k_m`

Other running containers: eidetic-mongo, eidetic-neo4j, events-mosquitto, model-gear-bluetts, model-gear-gateway, model-gear-realtime, model-gear-stt, model-gear-vllm-multimodal, qq-mongodb, weather-mongodb, weather-tracker, weather-web

`docker ps`:

```text
CONTAINER ID   IMAGE                            COMMAND                  CREATED        STATUS                  PORTS                                                                                                NAMES
5b2e75694123   climate-weather-tracker          "python -m climate.w…"   24 hours ago   Up 24 hours                                                                                                                  weather-tracker
2b2b3ee7a088   climate-weather-web              "python -m climate.w…"   24 hours ago   Up 24 hours             127.0.0.1:8095->8095/tcp                                                                             weather-web
a7c04673c948   lobes/vllm-gemma4:local          "bash /usr/local/bin…"   6 days ago     Up 33 hours (healthy)   8000/tcp                                                                                             model-gear-vllm-multimodal
7df2604debd0   lobes-gateway                    "python -m lobes.gat…"   6 days ago     Up 5 days (healthy)     0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp                                                          model-gear-gateway
57d878edb8b0   lobes-realtime                   "python -m lobes.rea…"   7 days ago     Up 5 days (healthy)     8080/tcp                                                                                             model-gear-realtime
d7919efdf434   lobes-bluetts:local              "python -m lobes.rea…"   7 days ago     Up 5 days (healthy)     9000/tcp                                                                                             model-gear-bluetts
f927adc7c08c   lobes-stt                        "/opt/nvidia/nvidia_…"   7 days ago     Up 5 days (healthy)     127.0.0.1:9002->9002/tcp                                                                             model-gear-stt
0e0f1bec5b41   mongo:8.0                        "docker-entrypoint.s…"   7 days ago     Up 5 days (healthy)     27017/tcp                                                                                            weather-mongodb
e66b11938c79   eclipse-mosquitto:2.1.2-alpine   "/docker-entrypoint.…"   2 months ago   Up 5 days (healthy)     127.0.0.1:1883->1883/tcp                                                                             events-mosquitto
e0810336da84   mongo:8.0                        "docker-entrypoint.s…"   3 months ago   Up 5 days (healthy)     0.0.0.0:27018->27017/tcp, [::]:27018->27017/tcp                                                      eidetic-mongo
a4183a674c55   neo4j:5-community                "tini -g -- /startup…"   3 months ago   Up 5 days (healthy)     0.0.0.0:7474->7474/tcp, [::]:7474->7474/tcp, 7473/tcp, 0.0.0.0:7687->7687/tcp, [::]:7687->7687/tcp   eidetic-neo4j
8f346da46891   mongo:8.0                        "docker-entrypoint.s…"   7 months ago   Up 5 days (healthy)     0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp                                                      qq-mongodb
```

`nvidia-smi`:

```text
Fri Sep 25 22:03:00 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   50C    P0             11W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5459      C   python3                                2079MiB |
|    0   N/A  N/A            8099      G   /usr/lib/xorg/Xorg                      338MiB |
|    0   N/A  N/A            8320      G   /usr/bin/gnome-shell                    204MiB |
|    0   N/A  N/A            8856      G   ...exec/xdg-desktop-portal-gnome         57MiB |
|    0   N/A  N/A           10315      G   /usr/bin/ghostty                        823MiB |
|    0   N/A  N/A          147875      G   ...rack-uuid=3190708988185955192        382MiB |
|    0   N/A  N/A         1261077      C   ...ma.cpp/build/bin/llama-server        829MiB |
|    0   N/A  N/A         3699287      C   VLLM::EngineCore                      33790MiB |
|    0   N/A  N/A         3732597      G   /usr/bin/nautilus                        41MiB |
|    0   N/A  N/A         3733171      G   .../8862/usr/lib/firefox/firefox        186MiB |
|    0   N/A  N/A         3734121      G   /usr/bin/gnome-text-editor               42MiB |
+-----------------------------------------------------------------------------------------+
```
