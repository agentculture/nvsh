"""File-read helpers and parsers for platform detection.

Every reader takes ``root`` (the filesystem root to read under — ``/`` in
production, a fixture tree in tests) and an absolute-looking path such as
``/etc/dgx-release``; it returns the raw text or ``None`` if the path is
missing, unreadable or not a regular file. Parsers are separate pure
functions so a fixture's real byte layout (including embedded NUL bytes in
``/proc/device-tree/*``) can be tested independently of I/O.
"""

from __future__ import annotations

import json
import os
import re


def read_text(root: str, path: str) -> str | None:
    """Read ``path`` under ``root``; ``None`` if missing/unreadable."""
    full = os.path.join(root, path.lstrip("/"))
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


# --- /etc/dgx-release ------------------------------------------------------


def parse_dgx_release(text: str) -> dict[str, str]:
    """Parse KEY="value" lines; a repeated key keeps its last occurrence."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r'^([A-Z_]+)="?([^"\n]*)"?$', line.strip())
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


# --- /etc/nv_tegra_release --------------------------------------------------


def parse_nv_tegra_release(text: str) -> str | None:
    """First line, e.g. '# R38 (release), REVISION: 2.2, ...' -> stripped."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return None


# --- /proc/device-tree/model and /proc/device-tree/compatible --------------


def parse_nul_terminated(text: str) -> list[str]:
    """Split a NUL-terminated /proc/device-tree/* value into its entries."""
    return [entry for entry in text.split("\x00") if entry]


def read_device_tree_string(root: str, path: str) -> str | None:
    """Read a single NUL-terminated device-tree string, e.g. model."""
    full = os.path.join(root, path.lstrip("/"))
    try:
        with open(full, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    text = raw.decode("utf-8", errors="replace")
    entries = parse_nul_terminated(text)
    return entries[0] if entries else None


def read_device_tree_list(root: str, path: str) -> list[str] | None:
    """Read a NUL-terminated multi-value device-tree property, e.g. compatible."""
    full = os.path.join(root, path.lstrip("/"))
    try:
        with open(full, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    entries = parse_nul_terminated(raw.decode("utf-8", errors="replace"))
    return entries or None


# --- /usr/local/cuda/version.json ------------------------------------------


def parse_cuda_version(text: str) -> str | None:
    # Valid JSON of the wrong *shape* (null, a list, a string, a `cuda` that
    # is not an object) is an absent fact, not an exception: raising here
    # aborted detection of the whole platform block. ValueError covers
    # json.JSONDecodeError, which derives from it.
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    cuda = data.get("cuda")
    if not isinstance(cuda, dict):
        return None
    version = cuda.get("version")
    return version if isinstance(version, str) else None


# --- /proc/driver/nvidia/version --------------------------------------------

_NVRM_VERSION_RE = re.compile(r"NVRM version:.*?\s(\d+\.\d+(?:\.\d+)?)\s+Release Build")


def parse_nvidia_driver_version(text: str) -> str | None:
    match = _NVRM_VERSION_RE.search(text)
    return match.group(1) if match else None


# --- /proc/meminfo -----------------------------------------------------------


def parse_meminfo(text: str) -> dict[str, str]:
    """Map MemTotal/MemAvailable/... -> raw 'N kB' string, verbatim."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^(\w+):\s+(\d+(?:\s+kB)?)", line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


# --- cudnn_version.h ---------------------------------------------------------

_CUDNN_MAJOR_RE = re.compile(r"#define\s+CUDNN_MAJOR\s+(\d+)")
_CUDNN_MINOR_RE = re.compile(r"#define\s+CUDNN_MINOR\s+(\d+)")
_CUDNN_PATCH_RE = re.compile(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)")


def parse_cudnn_version(text: str) -> str | None:
    major = _CUDNN_MAJOR_RE.search(text)
    minor = _CUDNN_MINOR_RE.search(text)
    patch = _CUDNN_PATCH_RE.search(text)
    if not (major and minor and patch):
        return None
    return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"


# --- /etc/docker/daemon.json --------------------------------------------------


def parse_docker_default_runtime(text: str) -> str | None:
    try:
        # ValueError covers json.JSONDecodeError, which derives from it.
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    runtime = data.get("default-runtime")
    return runtime if isinstance(runtime, str) else None


# --- /sys/class/dmi/id/product_name -----------------------------------------


def parse_dmi_product_name(text: str) -> str | None:
    stripped = text.strip()
    return stripped or None
