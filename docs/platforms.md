# Platform detection

`nvsh.platform.detect()` (`nvsh/platform/`) answers "what machine is this?"
for the failure panel and `nvsh doctor`. It is **file-first**: every fact is
read from a file when a file exposes it, and a subprocess is used only when
no file does. Nothing is guessed and nothing is silently omitted — a value
that could not be found comes back as `present: false` with the exact
source that was checked, so `docs/platforms.md` (this file) and
`Platform.render_block()` always show what was looked at, not just what was
found.

This module has not been wired into a CLI verb yet (see the plan's t17
task, which wires `nvsh doctor`). It is a library today:

```python
from nvsh.platform import detect

platform = detect()          # real machine
platform = detect(root="/some/fixture/tree")   # tests
print(platform.render_block())
print(platform.to_dict())    # JSON-serializable
```

`detect(root="/", run=default_run, which=default_which)` takes the
filesystem root and injectable subprocess/PATH lookups so tests can replay
captured output without a real subprocess or real hardware. `Platform.kind`
is one of `dgx-spark`, `jetson`, `rtx`, `generic` — `dgx-spark` when
`/etc/dgx-release` exists, `jetson` when `/etc/nv_tegra_release` exists,
`rtx` when the DMI product name mentions "RTX", `generic` otherwise.

## Values, sources and methods

`method` is `file` (read under `root`), `subprocess` (one of the four
allow-listed commands below, each called with a 2-second timeout), or
`path` (a `shutil.which` lookup, no execution). Every value in this table is
always present in `Platform.values`, whether found or not.

| Value | Method | Source | Notes |
| --- | --- | --- | --- |
| `dgx_name` | file | `/etc/dgx-release` (`DGX_NAME=`) | DGX Spark only |
| `dgx_swbuild_version` | file | `/etc/dgx-release` (`DGX_SWBUILD_VERSION=`) | DGX OS build, e.g. `7.2.3` |
| `l4t_release` | file | `/etc/nv_tegra_release` | first line, Jetson only |
| `device_tree_model` | file | `/proc/device-tree/model` | NUL-terminated string |
| `device_tree_compatible` | file | `/proc/device-tree/compatible` | NUL-terminated list, joined with `,` |
| `dmi_product_name` | file | `/sys/class/dmi/id/product_name` | e.g. `NVIDIA_DGX_Spark` |
| `cuda_version` | file | `/usr/local/cuda/version.json` (`.cuda.version`) | CUDA toolkit, absent when the toolkit isn't installed |
| `nvidia_driver_version` | file | `/proc/driver/nvidia/version` (`NVRM version:` line) | absent when the line doesn't carry a parseable `N.N[.N]` |
| `mem_total` | file | `/proc/meminfo` (`MemTotal:`) | raw `N kB` |
| `mem_available` | file | `/proc/meminfo` (`MemAvailable:`) | the CUDA-OOM pressure signal on unified-memory boxes — see below |
| `cudnn_version` | file | `/usr/include/aarch64-linux-gnu/cudnn_version.h` | `CUDNN_MAJOR.CUDNN_MINOR.CUDNN_PATCHLEVEL` |
| `docker_default_runtime` | file | `/etc/docker/daemon.json` (`default-runtime`) | absent both when the file is missing and when the key is missing from an existing file |
| `nvidia_smi_gpu_name` | subprocess | `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv` | |
| `nvidia_smi_driver_version` | subprocess | same `nvidia-smi` call | |
| `unified_memory` | subprocess | same `nvidia-smi` call | `true` when `memory.total`/`memory.used` come back `[N/A]` (GB10, Jetson) — see below |
| `nvpmodel_power_mode` | subprocess | `nvpmodel -q` | Jetson only; absent on DGX Spark (no `nvpmodel` binary) |
| `tensorrt_version` | subprocess | `dpkg-query -W` (`libnvinfer10`, falling back to `tensorrt`) | |
| `tmux` | path | `tmux` on `PATH` | |
| `pi` | path | `pi` on `PATH` | the Pi/associate agent harness |
| `spark_cli` | path | `spark` on `PATH` | `dgx-spark-cli` |
| `spark_status_available` | subprocess | `spark status --json` (`.available`) | only called when `spark_cli` is present; merged in, never required |
| `thor_cli` | path | `thor` on `PATH` | `jetson-thor-cli`; reported the same way as `spark_cli` — a plain PATH check, no subprocess call, no `--help` probe |
| `orin_cli` | path | `orin` on `PATH` | `jetson-orin-cli`; reported the same way as `spark_cli` — a plain PATH check, no subprocess call, no `--help` probe |

Both `thor` and `orin` install to each board's per-user `.local/bin`
directory (under the operator's home), same as `spark` does on the DGX
Spark box: a non-interactive, non-login shell (e.g. `ssh host 'which
thor'` in `BatchMode`) does not source the rc-file line that puts that
directory on `PATH`, so `which` can miss it there even though the binary
is installed — the same interactive-shell guard nvsh's own hook installer
respects (see `CLAUDE.md`'s "Hook constraints"). `nvsh` runs `detect()`
from the operator's already-interactive hooked shell, whose `PATH` does
include that directory, so this does not affect real usage; it only means
a bare non-interactive probe of the same command can look absent when the
CLI is in fact installed.

## Unified memory and the CUDA-OOM signal

On DGX Spark (GB10) and both Jetson boards, `nvidia-smi`'s
`memory.total`/`memory.used` columns report `[N/A]` — there is no discrete
VRAM pool to query, because GPU and CPU share one memory pool
(unified/coherent memory). `detect()` treats that `[N/A]` as the signal
that sets `unified_memory` to `true`. On a unified-memory box, a CUDA
out-of-memory failure is a system memory-pressure event, not a discrete-GPU
one, so the pressure signal nvsh reports is `mem_available`
(`/proc/meminfo`'s `MemAvailable:`), not any `nvidia-smi` memory column.

## Verified per platform

Each platform below was verified against real hardware (`ssh thor`,
`ssh orin`, and the local DGX Spark this repo was developed on) on
2026-09-13. The captured file/subprocess output backing these facts is
committed as fixtures under `tests/fixtures/platform/{spark,thor,orin}/`,
laid out under each host's own root (e.g. `tests/fixtures/platform/thor/etc/nv_tegra_release`
stands in for `/etc/nv_tegra_release` on thor); `tests/test_platform.py`
replays it through `detect(root=..., run=..., which=...)` with no real
subprocess calls. `tests/fixtures/platform/spark/subprocess/spark-status.json`
and the fixture `which()` maps are hand-sanitized (hostname/IPs stripped)
copies of real output; every other fixture file is byte-identical to what
the verifying command below produced.

### DGX Spark (this repo's dev machine)

`kind` = `dgx-spark`.

| Value | Verified value | Verifying command |
| --- | --- | --- |
| `dgx_name` | `DGX Spark` | `cat /etc/dgx-release` |
| `dgx_swbuild_version` | `7.2.3` (DGX OS 7.x) | `cat /etc/dgx-release` |
| `dmi_product_name` | `NVIDIA_DGX_Spark` | `cat /sys/class/dmi/id/product_name` |
| `cuda_version` | `13.0.2` | `cat /usr/local/cuda/version.json` |
| `nvidia_driver_version` | `580.126.09` | `cat /proc/driver/nvidia/version` |
| `mem_available` | present, e.g. `25684948 kB` | `cat /proc/meminfo` |
| `unified_memory` | `true` (`memory.total`/`memory.used` = `[N/A]`) | `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv` |
| `l4t_release`, `device_tree_model`, `device_tree_compatible` | absent (not a Jetson) | `cat /etc/nv_tegra_release`; `cat /proc/device-tree/model` |
| `cudnn_version` | absent (not installed on this Spark) | `cat /usr/include/aarch64-linux-gnu/cudnn_version.h` |
| `tensorrt_version` | absent (not installed on this Spark) | `dpkg-query -W \| grep -E 'libnvinfer10\|tensorrt'` |
| `nvpmodel_power_mode` | absent (`nvpmodel` not on PATH) | `which nvpmodel` |
| `tmux` | present | `which tmux` |
| `pi` | present | `which pi` |
| `spark_cli` | present | `which spark` |
| `spark_status_available` | `true` | `spark status --json` |
| `thor_cli`, `orin_cli` | absent (neither installed) | `which thor`; `which orin` |

### Jetson AGX Thor (`ssh thor`, L4T R38.2)

`kind` = `jetson`.

| Value | Verified value | Verifying command |
| --- | --- | --- |
| `l4t_release` | `R38 (release), REVISION: 2.2, ...` | `cat /etc/nv_tegra_release` |
| `device_tree_model` | `NVIDIA Jetson AGX Thor Developer Kit` | `cat /proc/device-tree/model` |
| `device_tree_compatible` | includes `nvidia,tegra264` | `cat /proc/device-tree/compatible` |
| `cuda_version` | `13.0.0` | `cat /usr/local/cuda/version.json` |
| `cudnn_version` | `9.12.0` | `cat /usr/include/aarch64-linux-gnu/cudnn_version.h` |
| `tensorrt_version` | `10.13.3.9-1+cuda13.0` (`libnvinfer10`) | `dpkg-query -W \| grep -E 'libnvinfer10\|tensorrt'` |
| `docker_default_runtime` | `nvidia` | `cat /etc/docker/daemon.json` |
| `nvpmodel_power_mode` | `MAXN` | `nvpmodel -q` |
| `unified_memory` | `true` (`memory.total`/`memory.used` = `[N/A]`) | `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv` |
| `dgx_name` | absent (not a DGX) | `cat /etc/dgx-release` |
| `tmux` | present | `which tmux` |
| `pi`, `spark_cli` | absent (neither installed) | `which pi`; `which spark` |
| `thor_cli` | present (`thor 0.5.0`, verbs `status/memory/gpu/disk/thermal/containers/network/processes/power/monitor/swap`, each accepting `--json`; `swap status --json` is the read-only swap subcommand) | `which thor && thor --version && thor --help` |
| `orin_cli` | absent | `which orin` |

### Jetson AGX Orin (`ssh orin`, L4T R39.2)

`kind` = `jetson`.

| Value | Verified value | Verifying command |
| --- | --- | --- |
| `l4t_release` | `R39 (release), REVISION: 2.0, ...` | `cat /etc/nv_tegra_release` |
| `device_tree_model` | `NVIDIA Jetson AGX Orin Developer Kit` | `cat /proc/device-tree/model` |
| `device_tree_compatible` | includes `nvidia,tegra234` | `cat /proc/device-tree/compatible` |
| `cuda_version` | **absent** — no `/usr/local/cuda/version.json` on this board | `cat /usr/local/cuda/version.json` |
| `cudnn_version` | **absent** — no cuDNN header on this board | `cat /usr/include/aarch64-linux-gnu/cudnn_version.h` |
| `tensorrt_version` | **absent** — no `libnvinfer10`/`tensorrt` package (only `nvidia-l4t-core 39.2.0`) | `dpkg-query -W \| grep -E 'libnvinfer10\|tensorrt\|nvidia-l4t-core'` |
| `docker_default_runtime` | **absent** — `/etc/docker/daemon.json` exists (defines the `nvidia` runtime) but sets no `default-runtime` key | `cat /etc/docker/daemon.json` |
| `nvpmodel_power_mode` | `MAXN` | `nvpmodel -q` |
| `unified_memory` | `true` (`memory.total`/`memory.used` = `[N/A]`) | `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv` |
| `tmux`, `pi`, `spark_cli` | absent (none installed) | `which tmux`; `which pi`; `which spark` |
| `orin_cli` | present (`orin 0.5.0`; `{whoami,learn,explain,overview,doctor,cli}` only — no machine verbs yet, confirming `nvsh/ops/render.py`'s empty `DEVICE_CLI_VERBS["orin"]`) | `which orin && orin --version && orin --help` |
| `thor_cli` | absent | `which thor` |

## Docker GPU path

Tier 2 runs its model server in a container nvsh starts and stops
(`nvsh/tiers/runtime_docker.py`). How a machine exposes its GPU to that
container is **not** uniform across the fleet, so nvsh detects it rather
than assuming one launch form. The mapping lives in one table,
`runtime_docker.GPU_FLAGS`, keyed by the same `Platform.kind` values the
rest of this document uses; a kind that is not in the table gets no GPU
flag at all and the container runs on the CPU. `[tiers.lfm] gpu = "off"`
forces that CPU path on any machine.

| `Platform.kind` | Rendered flag | Detection source | Checked |
|---|---|---|---|
| `dgx-spark` | `--gpus all` | `docker info` on the Spark lists **no** `nvidia` runtime — only CDI devices (`nvidia.com/gpu=all`) — and the operator's own running vLLM container uses the default `runc` runtime with a `--gpus all` device request | 2026-09-19, DGX Spark, Docker 29.1.3 (spec s17/s21) |
| `jetson` | `--runtime nvidia` | `--gpus all` is refused on Jetson: *"invoking the NVIDIA Container Runtime Hook directly (e.g. specifying the docker --gpus flag) is not supported. Please use the NVIDIA Container Runtime"* (container toolkit in csv mode). `--runtime nvidia` works and exposes `/dev/nvgpu` and `/dev/nvhost-gpu`. `docker info` shows the `nvidia` runtime registered on both Jetsons — default on Thor, `runc`-by-default on Orin | 2026-09-19, AGX Orin (L4T R39.2) and AGX Thor (L4T R38.2) (spec s17/s21) |
| anything else / `unknown` | *(none — CPU)* | no probe: nvsh does not guess a GPU wiring it has not measured | — |

Two things this deliberately does **not** do:

- **nvsh changes no Docker configuration.** It does not write
  `/etc/docker/daemon.json`, register a runtime, or set a default runtime.
  It only picks which flag to put on its own launch line; the `daemon_json`
  value in the tables above is read for *reporting* only.
- **nvsh runs no container but its own.** The launch line is built from
  `[tiers.lfm]` config plus the detection above and nothing else — never
  from request text or model output — and it names the image by `@sha256:`
  digest, publishes on `127.0.0.1` only, and names the container
  `nvsh-tier2-<uid>` on a port derived from the same uid so two operators on
  one machine do not collide. A test asserts the launcher is the only place
  under `nvsh/` that starts a container.

Host CUDA is not a prerequisite: the AGX Orin has a working GPU driver and
the `nvidia` container runtime but **no** host CUDA toolkit (no
`/usr/local/cuda`, no `nvcc`) and no host `llama-server`/`vllm`/`sglang`
binary, which is exactly why the runtime is delivered as a container.
Container images for these engines are large (llama.cpp server-cuda 6.7 GB,
jetson `llama_cpp` 23 GB, vllm-openai 30–33 GB as measured on the fleet) —
that is disk, not memory; resident memory is the engine process, the
weights, the KV cache and whatever the engine reserves up front.

## Redaction

Platform detection reads only version/capability facts (release files,
device-tree strings, package versions, PATH presence). None of the sources
above carry tokens or credentials, so this module does not redact anything
itself. The context collector that wraps a failure report (bounded command
line, exit code, cwd, captured output, and this platform block) is a
separate, later piece of work and owns its own redaction pass and tests —
see the spec's context-collector requirement.

## Playbooks built on this table

`nvsh/agent/playbooks.py` turns the facts above into a short per-kind
markdown block that rides in every backend's system prompt (see
`nvsh/agent/prompt.py`). The playbooks are keyed by the same
`Platform.kind` values this module produces, and every command they name is
either verified in the table above (`free -h` and `/proc/meminfo`,
`nvidia-smi` and its `[N/A]` memory columns, `nvpmodel -q`,
`/etc/nv_tegra_release`, `/etc/dgx-release`, `/usr/local/cuda/version.json`,
`/etc/docker/daemon.json`, `spark status --json`) or explicitly marked *if
installed* because it could not be verified here — `tegrastats`, `jtop` and
`jetson_release` are in that second group, and `jtop` is additionally
marked never-propose because it is interactive.

Why they exist: without them the model has no background and investigates
its own harness instead of the machine (deviation d19 — a failing
`whats memory levels are now?` on the Spark drew `which pi && pi --help`).
