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

`detect(root="/", run=default_run, which=default_which, own_bin=None)`
takes the filesystem root, injectable subprocess/PATH lookups and nvsh's own
env bin directory (`None` means `Path(sys.executable).parent`) so tests can
replay captured output without a real subprocess or real hardware. `Platform.kind`
is one of `dgx-spark`, `jetson`, `rtx`, `generic` — `dgx-spark` when
`/etc/dgx-release` exists, `jetson` when `/etc/nv_tegra_release` exists,
`rtx` when the DMI product name mentions "RTX", `generic` otherwise. `kind`
does not split the two Jetson boards; the separate `jetson_board` value does
(see [Jetson board and device CLIs](#jetson-board-and-device-clis)).

## Values, sources and methods

`method` is `file` (read under `root`), `subprocess` (one of the
allow-listed commands below — `nvidia-smi`, `nvpmodel -q`, `dpkg-query -W`,
`spark status --json` and one `<cli> --version` per device CLI found — each
called with a 2-second timeout), or `path` (a lookup, no execution: an
executable file in nvsh's own env bin directory for the device CLIs, a
`shutil.which` otherwise). Every value in this table is
always present in `Platform.values`, whether found or not.

| Value | Method | Source | Notes |
| --- | --- | --- | --- |
| `dgx_name` | file | `/etc/dgx-release` (`DGX_NAME=`) | DGX Spark only |
| `dgx_swbuild_version` | file | `/etc/dgx-release` (`DGX_SWBUILD_VERSION=`) | DGX OS build, e.g. `7.2.3` |
| `l4t_release` | file | `/etc/nv_tegra_release` | first line, Jetson only |
| `device_tree_model` | file | `/proc/device-tree/model` | NUL-terminated string |
| `device_tree_compatible` | file | `/proc/device-tree/compatible` | NUL-terminated list, joined with `,` |
| `jetson_board` | file | `/proc/device-tree/model` | `thor` or `orin` when the model string contains the whole word `Thor` or `Orin` (exactly one of them); absent otherwise, including on the DGX Spark, which has no `/proc/device-tree/model` |
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
| `spark_cli` | path | `spark (nvsh env)`, `spark (PATH)`, or `spark (nvsh env, PATH)` when absent | `dgx-spark-cli`; value is the resolved path; nvsh's own env bin directory is checked before `PATH` — see below |
| `thor_cli` | path | same form as `spark_cli`, for `thor` | `jetson-thor-cli`; no `--help` probe |
| `orin_cli` | path | same form as `spark_cli`, for `orin` | `jetson-orin-cli`; no `--help` probe |
| `spark_cli_version` | subprocess | `spark --version` | last whitespace token of the banner (`dgx-spark-cli 0.7.0` → `0.7.0`); only called when `spark_cli` is present; absent on a non-zero exit, timeout or unparseable output |
| `thor_cli_version` | subprocess | `thor --version` | same, for `thor` (`thor 0.5.0` → `0.5.0`) |
| `orin_cli_version` | subprocess | `orin --version` | same, for `orin` (`orin 0.5.0` → `0.5.0`) |
| `spark_status_available` | subprocess | `spark status --json` (`.available`) | only called when `spark_cli` is present; merged in, never required |

## Jetson board and device CLIs

`Platform.kind` is `jetson` on both Thor and Orin. The board comes from
`/proc/device-tree/model` (NUL-terminated; read the same way as
`device_tree_model`), measured on 2026-09-29 with
`tr -d '\0' </proc/device-tree/model`:

| Host | Model string | `jetson_board` |
| --- | --- | --- |
| `ssh thor` | `NVIDIA Jetson AGX Thor Developer Kit` | `thor` |
| `ssh orin` | `NVIDIA Jetson AGX Orin Developer Kit` | `orin` |
| DGX Spark | *(no `/proc/device-tree/model`)* | absent |

A model string that names neither board (or both) leaves `jetson_board`
absent, and callers fall back to trying `thor` then `orin`, as before the
split. `Platform.board()` returns the value's text or `None`.

Each device CLI (`spark`, `thor`, `orin`) is looked up in **nvsh's own env
bin directory first** — `Path(sys.executable).parent`, an existing
executable file there — **then `PATH`** (plan decision c27). nvsh is
installed as a uv tool on every fleet box, and `uv tool install
'nvsh[orin]'` installs `orin` inside nvsh's tool env without exposing it on
`PATH`; the own-env check finds it there. For a uv tool, `sys.executable`
is the tool env's own `bin/python` (not resolved through its symlink), so
its parent is the tool env's `bin/`. The value's `source` records which
place won (`<cli> (nvsh env)` or `<cli> (PATH)`), and its text is the
resolved path.

Install a device CLI with nvsh's per-platform extra, which puts it in
nvsh's own env:

| Extra | Package | Command | Minimum version |
| --- | --- | --- | --- |
| `uv tool install 'nvsh[spark]'` | `dgx-spark-cli` | `spark` | 0.8.0 |
| `uv tool install 'nvsh[thor]'` | `jetson-thor-cli` | `thor` | 0.5.0 (the extra pins 0.5.1) |
| `uv tool install 'nvsh[orin]'` | `jetson-orin-cli` | `orin` | 0.6.0 |

Adding `--with-executables-from <package>` also puts the CLI on your
`PATH`; nvsh does not need that. A CLI below its minimum version, or whose
version cannot be read, is treated as absent (nvsh uses system commands),
and `nvsh doctor`'s `device_cli` check says so.

For each CLI found, `detect()` runs `<cli> --version` once and records the
last whitespace token of the output as `<cli>_cli_version` (a leading `v`
is dropped; anything that does not start with `N.N` is absent). Detection
never runs on the hook's success path — only once a failure qualifies, and
from `doctor`, the tiers and slash routing — so these calls add no latency
to a successful command. `Platform.device_cli(<cli>)` returns a `DeviceCli`
(`name`, `path`, `origin` = `nvsh-env` or `path`, `version`) or `None`.

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
| `spark_cli_version` | `0.7.0` (`dgx-spark-cli 0.7.0`, 2026-09-29) | `spark --version` |
| `spark_status_available` | `true` | `spark status --json` |
| `thor_cli`, `orin_cli` | absent (neither installed) | `which thor`; `which orin` |
| `jetson_board` | absent (no `/proc/device-tree/model`, 2026-09-29) | `cat /proc/device-tree/model` |

### Jetson AGX Thor (`ssh thor`, L4T R38.2)

`kind` = `jetson`.

| Value | Verified value | Verifying command |
| --- | --- | --- |
| `l4t_release` | `R38 (release), REVISION: 2.2, ...` | `cat /etc/nv_tegra_release` |
| `device_tree_model` | `NVIDIA Jetson AGX Thor Developer Kit` | `cat /proc/device-tree/model` |
| `jetson_board` | `thor` (model re-measured 2026-09-29) | `tr -d '\0' </proc/device-tree/model` |
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
| `thor_cli_version` | `0.5.0` (`thor 0.5.0`, 2026-09-29) | `thor --version` |
| `orin_cli` | absent | `which orin` |

### Jetson AGX Orin (`ssh orin`, L4T R39.2)

`kind` = `jetson`.

| Value | Verified value | Verifying command |
| --- | --- | --- |
| `l4t_release` | `R39 (release), REVISION: 2.0, ...` | `cat /etc/nv_tegra_release` |
| `device_tree_model` | `NVIDIA Jetson AGX Orin Developer Kit` | `cat /proc/device-tree/model` |
| `jetson_board` | `orin` (model re-measured 2026-09-29) | `tr -d '\0' </proc/device-tree/model` |
| `device_tree_compatible` | includes `nvidia,tegra234` | `cat /proc/device-tree/compatible` |
| `cuda_version` | **absent** — no `/usr/local/cuda/version.json` on this board | `cat /usr/local/cuda/version.json` |
| `cudnn_version` | **absent** — no cuDNN header on this board | `cat /usr/include/aarch64-linux-gnu/cudnn_version.h` |
| `tensorrt_version` | **absent** — no `libnvinfer10`/`tensorrt` package (only `nvidia-l4t-core 39.2.0`) | `dpkg-query -W \| grep -E 'libnvinfer10\|tensorrt\|nvidia-l4t-core'` |
| `docker_default_runtime` | **absent** — `/etc/docker/daemon.json` exists (defines the `nvidia` runtime) but sets no `default-runtime` key | `cat /etc/docker/daemon.json` |
| `nvpmodel_power_mode` | `MAXN` | `nvpmodel -q` |
| `unified_memory` | `true` (`memory.total`/`memory.used` = `[N/A]`) | `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv` |
| `tmux`, `pi`, `spark_cli` | absent (none installed) | `which tmux`; `which pi`; `which spark` |
| `orin_cli` | present (measured 2026-09-29 at `orin 0.5.0`, which had only `{whoami,learn,explain,overview,doctor,cli}`; nvsh's `DEVICE_CLI_VERBS["orin"]` now maps the same ten read-only verbs as spark/thor, gated by the `DEVICE_CLI_MIN_VERSIONS` floors spark 0.8.0, thor 0.5.0, orin 0.6.0, so a 0.5.0 orin CLI is treated as absent) | `which orin && orin --version && orin --help` |
| `orin_cli_version` | `0.5.0` (`orin 0.5.0`, 2026-09-29) | `orin --version` |
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
