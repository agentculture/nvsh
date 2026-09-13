"""Per-platform playbooks: what to look at on *this* kind of machine.

Data in code, not configuration, so an operator who installs the wheel on an
air-gapped Jetson gets them out of the box. Each playbook is a short
markdown block keyed by ``Platform.kind`` (``dgx-spark``, ``jetson``,
``rtx``, ``generic``) and is pasted into the system brief by
:func:`nvsh.agent.prompt.build_system_prompt`.

Provenance: every command named below is either verified on real hardware in
``docs/platforms.md`` (the DGX Spark this repo was developed on, ``ssh thor``
and ``ssh orin``) and replayed from the fixture trees under
``tests/fixtures/platform/``, or marked "if installed" because it could not
be verified there. Nothing is claimed that was not checked.

Stdlib only; imports nothing from :mod:`nvsh`.
"""

from __future__ import annotations

#: Kinds spelled with an underscore (a caller's own normalization) map onto
#: the hyphenated kinds ``nvsh.platform.detect`` produces.
_ALIASES = {"dgx_spark": "dgx-spark"}

DGX_SPARK = """\
## This machine: NVIDIA DGX Spark (GB10, DGX OS)

- Memory is **unified**: CPU and GPU share one pool. `nvidia-smi` reports
  `memory.total`/`memory.used` as `[N/A]` here — it is not the memory
  truth and never will be. (verified: docs/platforms.md, DGX Spark)
- The memory truth is `free -h`, or `/proc/meminfo` (`MemTotal:`,
  `MemAvailable:`). `MemAvailable` is the pressure signal.
- A CUDA out-of-memory here means **system** memory pressure, not a full
  discrete VRAM pool. Check `free -h` first, then what is holding the
  memory (`ps -eo pid,rss,comm --sort=-rss | head`).
- GPU/driver detail: `nvidia-smi` (present), `nvidia-smi -q` for the long
  form. Driver version also in `/proc/driver/nvidia/version`; CUDA toolkit
  in `/usr/local/cuda/version.json`.
- DGX OS build: `/etc/dgx-release` (`DGX_NAME`, `DGX_SWBUILD_VERSION`).
- `spark status --json` (dgx-spark-cli) when `spark` is on PATH — on this
  box it is; treat its absence as "not installed", not as an error.
- Container runtime: `/etc/docker/daemon.json` (`default-runtime`).
- Disk: `df -h`. Kernel/driver errors: `dmesg | tail` (may need sudo),
  `journalctl -p err -n 50`.
"""

JETSON = """\
## This machine: NVIDIA Jetson (L4T / JetPack)

- Memory is **unified**: CPU and GPU share one pool, and `nvidia-smi`
  reports `memory.total`/`memory.used` as `[N/A]`. (verified:
  docs/platforms.md, AGX Thor and AGX Orin)
- The memory truth is `free -h` or `/proc/meminfo` (`MemAvailable:`). A
  CUDA out-of-memory is system memory pressure here.
- `tegrastats` (one sample: `timeout 2 tegrastats`) for live RAM/GPU/EMC
  load, if installed. `jtop` is interactive — never run it from a
  proposal; if installed, tell the operator to run it themselves.
- Power mode: `nvpmodel -q` (verified present on Thor and Orin). Changing
  it needs sudo — propose the command, never run it.
- L4T / JetPack: `cat /etc/nv_tegra_release`; board: `cat
  /proc/device-tree/model`. `jetson_release` if installed.
- CUDA `/usr/local/cuda/version.json`, cuDNN
  `/usr/include/aarch64-linux-gnu/cudnn_version.h`, TensorRT
  `dpkg-query -W libnvinfer10` — all three are genuinely absent on some
  boards (Orin, verified), so "absent" is an answer, not a failure.
- Container runtime: `/etc/docker/daemon.json` (`default-runtime`, often
  `nvidia`). Disk: `df -h`. Errors: `dmesg | tail`,
  `journalctl -p err -n 50`.
"""

RTX = """\
## This machine: x86/RTX workstation with a discrete NVIDIA GPU

- Memory is **not** unified: `nvidia-smi` reports real VRAM totals, and a
  CUDA out-of-memory is a VRAM event. `nvidia-smi
  --query-gpu=memory.total,memory.used --format=csv` for the numbers,
  `nvidia-smi -q` for the long form.
- System memory is separate: `free -h`, `/proc/meminfo`.
- Driver `/proc/driver/nvidia/version`; CUDA `/usr/local/cuda/version.json`.
- Disk: `df -h`. Errors: `dmesg | tail`, `journalctl -p err -n 50`.
"""

GENERIC = """\
## This machine: Linux (no NVIDIA platform file matched)

- Memory: `free -h`, or `/proc/meminfo` (`MemTotal:`, `MemAvailable:`).
- Disk: `df -h`. Largest consumers: `du -xh --max-depth=1 . | sort -h`.
- Recent errors: `journalctl -p err -n 50`, `dmesg | tail`.
- Processes: `ps -eo pid,rss,pcpu,comm --sort=-rss | head`.
- GPU facts may simply not exist here — say so rather than guessing.
  `nvidia-smi` only if installed.
"""

PLAYBOOKS = {
    "dgx-spark": DGX_SPARK,
    "jetson": JETSON,
    "rtx": RTX,
    "generic": GENERIC,
}


def playbook_for(kind: str) -> str:
    """The playbook for ``kind``, falling back to :data:`GENERIC`.

    An unknown kind is not an error: a newer detector (or a hand-written
    context) must still get a usable brief.
    """
    normalized = (kind or "").strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    return PLAYBOOKS.get(normalized, GENERIC)
