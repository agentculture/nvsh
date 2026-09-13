"""Subprocess runner + parsers for the small set of commands platform
detection is allowed to shell out to: ``nvidia-smi``, ``nvpmodel -q``,
``dpkg-query -W``, and ``spark status --json``.

File reads are always preferred (see ``_files.py``); these are used only for
facts no file exposes. Every call goes through the injectable ``run``
signature so tests can replay real captured output without a subprocess.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # nosec B404 - fixed argv lists below, no shell=True
from typing import Callable

#: (returncode, stdout, stderr)
RunResult = tuple[int, str, str]
Runner = Callable[[list[str], float], RunResult]
Which = Callable[[str], "str | None"]

DEFAULT_TIMEOUT = 2.0


def default_run(argv: list[str], timeout: float = DEFAULT_TIMEOUT) -> RunResult:
    """Real subprocess runner: fixed argv, no shell, bounded by timeout."""
    try:
        proc = subprocess.run(  # nosec B603 - argv is a fixed list, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired):
        return 1, "", ""


def default_which(name: str) -> str | None:
    return shutil.which(name)


# --- nvidia-smi --------------------------------------------------------------


def parse_nvidia_smi_csv(stdout: str) -> dict[str, str] | None:
    """Parse the second line of
    'nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version
    --format=csv' into {"name", "memory.total", "memory.used",
    "driver_version"}.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    values = [part.strip() for part in lines[1].split(",")]
    if len(values) < 4:
        return None
    return {
        "name": values[0],
        "memory.total": values[1],
        "memory.used": values[2],
        "driver_version": values[3],
    }


def is_unified_memory(gpu: dict[str, str]) -> bool:
    """nvidia-smi reports memory.total/used as '[N/A]' on unified-memory
    boxes (Jetson, GB10) since there is no discrete VRAM pool to query."""
    return gpu.get("memory.total") == "[N/A]" or gpu.get("memory.used") == "[N/A]"


# --- nvpmodel -q ---------------------------------------------------------------

_NVPMODEL_RE = re.compile(r"NV Power Mode:\s*(\S+)")


def parse_nvpmodel(stdout: str) -> str | None:
    match = _NVPMODEL_RE.search(stdout)
    return match.group(1) if match else None


# --- dpkg-query -W -------------------------------------------------------------


def parse_dpkg_query(stdout: str, package: str) -> str | None:
    """Find `package`'s version in 'dpkg-query -W' two-column output."""
    for line in stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == package:
            return parts[1].strip()
    return None


# --- spark status --json --------------------------------------------------------


def parse_spark_status(stdout: str) -> bool | None:
    """Return the top-level 'available' flag, or None if unparseable."""
    try:
        # ValueError covers json.JSONDecodeError, which derives from it.
        data = json.loads(stdout)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    available = data.get("available")
    return available if isinstance(available, bool) else None
