# Tier 2 measurement, 2026-09-24: final-scorer-b1-missing-candidate-exact

- Command: `scripts/lfm-finetune/measure.py --split $HOME/lfm-train/work/q46/splits/test.json --final --model scorer-b1 --revision 541e1a9127b9682be883a517bb1c148aa6c79eb4 --label final-scorer-b1-missing-candidate-exact --config $HOME/lfm-train/work/q46/measure/final-scorer-b1-missing-candidate-exact.nvsh.toml --ctx 2048 --ground-snapshot $HOME/lfm-train/work/q46/ground-snapshot.json --enable-thinking false --max-logprobs 22 --predictions $HOME/lfm-train/work/q46/final/scorer-b1 --tokenizer $HOME/lfm-train/work/q46/runs/scorer-b1/merged --slice missing-candidate --scorer in-process`
- Split: `$HOME/lfm-train/work/q46/splits/test.json` (32 entries, 32 sources)
- Seed: 46 (from the split header)
- nvsh: 0.18.0, commit `ff0184462b53ab3614cc09be519b89df9f3180c1`
- Models (repo id @ revision): `scorer-b1` @ `541e1a9127b9682be883a517bb1c148aa6c79eb4` (passed to from_pretrained in-process)
- Tier 2 settings, identical for every run except the model: engine=vllm, mode=attach, image=vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695, ctx=2048, tool_call_parser=qwen3_coder
- Grounding: fixed snapshot `$HOME/lfm-train/work/q46/ground-snapshot.json` (platform: fixture world from the split file)
- Serving: engine=vllm, mode=attach, ctx=2048, image=`vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`, tool_call_parser=qwen3_coder
- Requests: chat_template_kwargs enable_thinking=false (the scorer renders its own prompts, thinking off)
- Decision mode: candidate scorer (in-process) through scorer.py, max-logprobs 22 (operator-supplied); asks for 22 per request
- Slice: missing-candidate (eval_slices.py: each operation entry with its gold operation left out of the offered candidates, expected to escalate)
- Ground snapshot: `$HOME/lfm-train/work/q46/ground-snapshot.json` sha256 `0d79c8fe63cef6a9b2e38a1719d173d59c4e793eed5b9ca3fe7b335148dc533b` (263 services, 34 containers; created 2026-09-24)
- Acceptance run: no
- Final run: yes
- Final runs on the test side, including this one: 10

A candidate-scorer run: the bench table does not apply; the figures are
metrics.py's, over the predictions file the scorer's results were written to.

## Issue 46 metrics

metrics.py over each model's predictions file (one line per entry). The candidate
distribution of a generative run comes from its deciding reply's log-probabilities.

| Metric | `scorer-b1` |
|---|---|
| Right proposals (metrics.py) | 0 of 0 |
| Abstention recall (escalate entries escalated) | 21.9% (7 of 32) |
| Abstention precision, strict (deviation d2) | 100.0% |
| False-positive tool calls (proposals on explain/escalate entries) | 18 of 32 |
| Wrong mutating, total (wrong operation + wrong arguments) | 0 (0 + 0) |
| Invalid outputs | 3 of 32 (not_grounded: 3) |
| Lines with a candidate distribution | 32 of 32 |
| ECE (10 equal-width bins) | 0.547 |
| Brier (multi-class) | 1.198 |
| Tokens generated per decision, mean / median | 0.0 / 0.0 |
| Time to first decision, cold / warm median / warm p95 | 601 ms / 82 ms / 88 ms |
| Decision latency, cold / warm median / warm p95 | 601 ms / 82 ms / 88 ms |
| Non-empty think blocks (must be 0) | 0 |

| nvsh outcome | issue 46 JSON |
|---|---|
| propose | `{"action": "tool", "tool": <operation>, "arguments": <arguments>}` |
| explain | `{"action": "no_action"}` |
| escalate | `{"action": "abstain"}` |
| invalid | `{"action": "invalid"}` |

Reporting only: every model is trained and scored on nvsh's propose/explain/escalate tools. Issue 46's abstain is nvsh's escalate, so abstention recall is the escalation recall and abstention precision the strict escalation precision (an escalation on an explain entry counts against it). Explain (answer in words, no tool) has no counterpart in issue 46's tool|abstain pair; it is shown as the no_action label issue 46 uses for Track B and is never counted as an abstention. An invalid output is not a decision.

## Background before each run

### `scorer-b1`

Other running containers: eidetic-mongo, eidetic-neo4j, events-mosquitto, model-gear-bluetts, model-gear-gateway, model-gear-realtime, model-gear-stt, q46-measure-18060, qq-mongodb, weather-mongodb, weather-tracker, weather-web

`docker ps`:

```text
CONTAINER ID   IMAGE                            COMMAND                  CREATED             STATUS                PORTS                                                                                                NAMES
5b88ebd2f04d   vllm/vllm-openai                 "vllm serve --model …"   2 minutes ago       Up 2 minutes          127.0.0.1:18060->8000/tcp                                                                            q46-measure-18060
0205ea8dabbf   climate-weather-web              "python -m climate.w…"   About an hour ago   Up 59 minutes         127.0.0.1:8095->8095/tcp                                                                             weather-web
93b52a3f60bb   climate-weather-tracker          "python -m climate.w…"   About an hour ago   Up 59 minutes                                                                                                              weather-tracker
7df2604debd0   lobes-gateway                    "python -m lobes.gat…"   5 days ago          Up 3 days (healthy)   0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp                                                          model-gear-gateway
57d878edb8b0   lobes-realtime                   "python -m lobes.rea…"   5 days ago          Up 3 days (healthy)   8080/tcp                                                                                             model-gear-realtime
d7919efdf434   lobes-bluetts:local              "python -m lobes.rea…"   5 days ago          Up 3 days (healthy)   9000/tcp                                                                                             model-gear-bluetts
f927adc7c08c   lobes-stt                        "/opt/nvidia/nvidia_…"   5 days ago          Up 3 days (healthy)   127.0.0.1:9002->9002/tcp                                                                             model-gear-stt
0e0f1bec5b41   mongo:8.0                        "docker-entrypoint.s…"   6 days ago          Up 3 days (healthy)   27017/tcp                                                                                            weather-mongodb
e66b11938c79   eclipse-mosquitto:2.1.2-alpine   "/docker-entrypoint.…"   2 months ago        Up 3 days (healthy)   127.0.0.1:1883->1883/tcp                                                                             events-mosquitto
e0810336da84   mongo:8.0                        "docker-entrypoint.s…"   3 months ago        Up 3 days (healthy)   0.0.0.0:27018->27017/tcp, [::]:27018->27017/tcp                                                      eidetic-mongo
a4183a674c55   neo4j:5-community                "tini -g -- /startup…"   3 months ago        Up 3 days (healthy)   0.0.0.0:7474->7474/tcp, [::]:7474->7474/tcp, 7473/tcp, 0.0.0.0:7687->7687/tcp, [::]:7687->7687/tcp   eidetic-neo4j
8f346da46891   mongo:8.0                        "docker-entrypoint.s…"   7 months ago        Up 3 days (healthy)   0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp                                                      qq-mongodb
```

`nvidia-smi`:

```text
Thu Sep 24 10:01:38 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   51C    P0             12W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5459      C   python3                                2079MiB |
|    0   N/A  N/A            8099      G   /usr/lib/xorg/Xorg                      431MiB |
|    0   N/A  N/A            8320      G   /usr/bin/gnome-shell                    206MiB |
|    0   N/A  N/A            8856      G   ...exec/xdg-desktop-portal-gnome         57MiB |
|    0   N/A  N/A           10315      G   /usr/bin/ghostty                        837MiB |
|    0   N/A  N/A          147875      G   ...rack-uuid=3190708988185955192        184MiB |
|    0   N/A  N/A         2722107      C   VLLM::EngineCore                       8566MiB |
|    0   N/A  N/A         3732597      G   /usr/bin/nautilus                        41MiB |
|    0   N/A  N/A         3733171      G   .../8862/usr/lib/firefox/firefox        186MiB |
|    0   N/A  N/A         3734121      G   /usr/bin/gnome-text-editor               43MiB |
+-----------------------------------------------------------------------------------------+
```
