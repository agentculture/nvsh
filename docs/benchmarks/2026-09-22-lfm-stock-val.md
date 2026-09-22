# Tier 2 measurement, 2026-09-22: stock-val

- Command: `scripts/lfm-finetune/measure.py --split /home/spark/lfm-train/splits/val.json --model LiquidAI/LFM2.5-350M --revision 9e6c6ccf47cd318696e137d381a7ded8fe4df09f --label stock-val --config /home/spark/lfm-train/measure-config.toml`
- Split: `/home/spark/lfm-train/splits/val.json` (66 entries, 66 sources)
- Seed: 39 (from the split header)
- nvsh: 0.17.0, commit `ec9aa833d0e04a4a24a5aafcc7dc79136644a790`
- Models (repo id @ revision): `LiquidAI/LFM2.5-350M` @ `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` (revision verified from the cache)
- Tier 2 settings, identical for every run except the model: engine=vllm, mode=managed, image=vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695, ctx=4096, gpu_memory_fraction=0.08, tool_call_parser=lfm2
- Grounding: fixture world from the split file
- Acceptance run: no
- Final run: no

Per-source figures (one vote per `source_id`, majority over its variations, a
tie counts against the model) are the ones claims are judged on; per-variation
figures show paraphrase robustness.

The use-case bar's "0 wrong mutating proposals" is judged on the sum of both rows:
"Wrong mutating proposals, per source / per variation" (bench's own count: a mutating operation other than the
expected one) plus "Mutating proposals with wrong arguments, per source / per variation" (the expected mutating
operation with arguments bench does not accept, e.g. the wrong container).

| Metric | `LiquidAI/LFM2.5-350M` |
|---|---|
| Model revision | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` (revision verified from the cache) |
| Right operation and arguments proposed, per source | 1 of 32 |
| Right operation and arguments proposed, per variation | 1 of 32 |
| Should-escalate asks escalated, per source | 0 of 16 |
| Should-escalate asks escalated, per variation | 0 of 16 |
| Wrong mutating proposals, per source / per variation | 0 / 0 |
| Mutating proposals with wrong arguments, per source / per variation | 0 / 0 |
| Explain asks explained, per source | 18 of 18 |
| Explain asks: explained / proposed / escalated, per variation | 18 / 0 / 0 of 18 |
| Mutating proposals on explain asks | 0 |
| Warm latency, median / p95 | 257 ms / 1088 ms |
| First request after start | 23.5 s |
| Container memory (`docker stats`) | 3.5 GiB |
| Start-up, including first download | 119.5 s |

## Background before each run

### `LiquidAI/LFM2.5-350M`

Other running containers: eidetic-mongo, eidetic-neo4j, events-mosquitto, model-gear-bluetts, model-gear-gateway, model-gear-realtime, model-gear-stt, model-gear-vllm-multimodal, qq-mongodb, weather-mongodb, weather-tracker, weather-web

`docker ps`:

```text
CONTAINER ID   IMAGE                            COMMAND                  CREATED        STATUS                PORTS                                                                                                NAMES
a7c04673c948   lobes/vllm-gemma4:local          "bash /usr/local/bin…"   3 days ago     Up 2 days (healthy)   8000/tcp                                                                                             model-gear-vllm-multimodal
7df2604debd0   lobes-gateway                    "python -m lobes.gat…"   3 days ago     Up 2 days (healthy)   0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp                                                          model-gear-gateway
57d878edb8b0   lobes-realtime                   "python -m lobes.rea…"   4 days ago     Up 2 days (healthy)   8080/tcp                                                                                             model-gear-realtime
d7919efdf434   lobes-bluetts:local              "python -m lobes.rea…"   4 days ago     Up 2 days (healthy)   9000/tcp                                                                                             model-gear-bluetts
f927adc7c08c   lobes-stt                        "/opt/nvidia/nvidia_…"   4 days ago     Up 2 days (healthy)   127.0.0.1:9002->9002/tcp                                                                             model-gear-stt
440cd5be79c1   climate-weather-web              "python -m climate.w…"   4 days ago     Up 2 days             127.0.0.1:8095->8095/tcp                                                                             weather-web
ee4053f9b93f   climate-weather-tracker          "python -m climate.w…"   4 days ago     Up 2 days                                                                                                                  weather-tracker
0e0f1bec5b41   mongo:8.0                        "docker-entrypoint.s…"   5 days ago     Up 2 days (healthy)   27017/tcp                                                                                            weather-mongodb
e66b11938c79   eclipse-mosquitto:2.1.2-alpine   "/docker-entrypoint.…"   2 months ago   Up 2 days (healthy)   127.0.0.1:1883->1883/tcp                                                                             events-mosquitto
e0810336da84   mongo:8.0                        "docker-entrypoint.s…"   3 months ago   Up 2 days (healthy)   0.0.0.0:27018->27017/tcp, [::]:27018->27017/tcp                                                      eidetic-mongo
a4183a674c55   neo4j:5-community                "tini -g -- /startup…"   3 months ago   Up 2 days (healthy)   0.0.0.0:7474->7474/tcp, [::]:7474->7474/tcp, 7473/tcp, 0.0.0.0:7687->7687/tcp, [::]:7687->7687/tcp   eidetic-neo4j
8f346da46891   mongo:8.0                        "docker-entrypoint.s…"   7 months ago   Up 2 days (healthy)   0.0.0.0:27017->27017/tcp, [::]:27017->27017/tcp                                                      qq-mongodb
```

`nvidia-smi`:

```text
Tue Sep 22 22:54:17 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0  On |                  N/A |
| N/A   48C    P0             12W /  N/A  | Not Supported          |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5459      C   python3                                2079MiB |
|    0   N/A  N/A            8099      G   /usr/lib/xorg/Xorg                      177MiB |
|    0   N/A  N/A            8320      G   /usr/bin/gnome-shell                    168MiB |
|    0   N/A  N/A            8394      C   VLLM::EngineCore                      34172MiB |
|    0   N/A  N/A            8856      G   ...exec/xdg-desktop-portal-gnome         57MiB |
|    0   N/A  N/A           10315      G   /usr/bin/ghostty                        646MiB |
|    0   N/A  N/A          147875      G   ...rack-uuid=3190708988185955192        209MiB |
+-----------------------------------------------------------------------------------------+
```
