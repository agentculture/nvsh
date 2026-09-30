"""``detect()``: build a :class:`Platform` for the current machine.

File-first, as required by the spec's platform-detection claim: every fact
is read from a file when a file exposes it, and subprocesses are used only
for ``nvidia-smi``, ``nvpmodel -q`` and ``dpkg-query -W`` (each with a
timeout), plus an optional ``spark status --json`` merge when the
``dgx-spark-cli`` ``spark`` binary is found, and one ``<cli> --version``
per device CLI (``spark``/``thor``/``orin``) found. Device CLIs are looked up
in nvsh's own env bin directory first, then PATH (plan decision c27). The
Jetson board (thor/orin) comes from ``/proc/device-tree/model``.

Detection never runs on the hook's success path: it is called only once a
failure qualifies, and by ``doctor``, the tiers and slash routing.

Nothing is ever required to
be present: a missing file, absent binary or unparseable value comes back as
an absent :class:`Value` carrying the source that was checked, not a
silently dropped fact.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from . import _files as files
from . import _subprocess as subp
from ._model import (
    FILE,
    NVSH_ENV,
    ON_PATH,
    PATH,
    SUBPROCESS,
    Platform,
    Value,
    device_cli_source,
)

_DGX_RELEASE = "/etc/dgx-release"
_NV_TEGRA_RELEASE = "/etc/nv_tegra_release"
_DEVICE_TREE_MODEL = "/proc/device-tree/model"
_DEVICE_TREE_COMPATIBLE = "/proc/device-tree/compatible"
_DMI_PRODUCT_NAME = "/sys/class/dmi/id/product_name"
_CUDA_VERSION_JSON = "/usr/local/cuda/version.json"
_NVIDIA_DRIVER_VERSION = "/proc/driver/nvidia/version"
_MEMINFO = "/proc/meminfo"
_CUDNN_HEADER = "/usr/include/aarch64-linux-gnu/cudnn_version.h"
_DOCKER_DAEMON_JSON = "/etc/docker/daemon.json"

#: Device-tree model word -> Jetson board. Measured 2026-09-29:
#: "NVIDIA Jetson AGX Thor Developer Kit", "NVIDIA Jetson AGX Orin Developer
#: Kit". Matched as whole words so e.g. "Thorough" is not a Thor.
_BOARD_WORDS = (("thor", re.compile(r"\bThor\b")), ("orin", re.compile(r"\bOrin\b")))

#: The device CLIs nvsh knows, in report order: value name prefix == binary.
_DEVICE_CLIS = ("spark", "thor", "orin")


def _absent(name: str, source: str, method: str) -> Value:
    return Value(name=name, text=None, source=source, method=method, present=False)


def _present(name: str, text: str, source: str, method: str) -> Value:
    return Value(name=name, text=text, source=source, method=method, present=True)


def _value_from_file(name: str, root: str, path: str, parse) -> Value:
    """Read `path` under root and hand its text to `parse`; absent on any miss."""
    text = files.read_text(root, path)
    if text is None:
        return _absent(name, path, FILE)
    parsed = parse(text)
    if parsed is None:
        return _absent(name, path, FILE)
    return _present(name, parsed, path, FILE)


def _value_from_which(name: str, which, tool: str) -> Value:
    found = which(tool)
    source = f"{tool} (PATH)"
    if found:
        return _present(name, found, source, PATH)
    return _absent(name, source, PATH)


def _maybe(name: str, text: str | None, source: str, method: str) -> Value:
    """``_present`` when *text* is truthy, ``_absent`` otherwise."""
    if text:
        return _present(name, text, source, method)
    return _absent(name, source, method)


def _run_and_parse(run, which, tool: str, argv_tail: list[str], parse):
    """Run ``<tool> <argv_tail>`` if *tool* is on PATH; parse stdout on exit 0.

    ``None`` when the tool is absent, exits non-zero, or its output does not
    parse -- the three ways a subprocess-only fact comes back absent.
    """
    found = which(tool)
    if not found:
        return None
    code, out, _err = run([found, *argv_tail], subp.DEFAULT_TIMEOUT)
    if code != 0:
        return None
    return parse(out)


def _file_values(root: str) -> list[Value]:
    """Every fact a file exposes, in report order."""
    values: list[Value] = []

    dgx = files.read_text(root, _DGX_RELEASE)
    dgx_fields = files.parse_dgx_release(dgx) if dgx is not None else {}
    values.append(_maybe("dgx_name", dgx_fields.get("DGX_NAME"), _DGX_RELEASE, FILE))
    values.append(
        _maybe(
            "dgx_swbuild_version",
            dgx_fields.get("DGX_SWBUILD_VERSION"),
            _DGX_RELEASE,
            FILE,
        )
    )

    nv_tegra = files.read_text(root, _NV_TEGRA_RELEASE)
    l4t_release = files.parse_nv_tegra_release(nv_tegra) if nv_tegra is not None else None
    values.append(_maybe("l4t_release", l4t_release, _NV_TEGRA_RELEASE, FILE))

    dt_model = files.read_device_tree_string(root, _DEVICE_TREE_MODEL)
    values.extend(
        [
            _maybe("device_tree_model", dt_model, _DEVICE_TREE_MODEL, FILE),
            _maybe("jetson_board", parse_board(dt_model), _DEVICE_TREE_MODEL, FILE),
        ]
    )

    dt_compatible = files.read_device_tree_list(root, _DEVICE_TREE_COMPATIBLE)
    values.append(
        _maybe(
            "device_tree_compatible",
            ",".join(dt_compatible) if dt_compatible else None,
            _DEVICE_TREE_COMPATIBLE,
            FILE,
        )
    )

    values.append(
        _value_from_file("dmi_product_name", root, _DMI_PRODUCT_NAME, files.parse_dmi_product_name)
    )
    values.append(
        _value_from_file("cuda_version", root, _CUDA_VERSION_JSON, files.parse_cuda_version)
    )
    values.append(
        _value_from_file(
            "nvidia_driver_version",
            root,
            _NVIDIA_DRIVER_VERSION,
            files.parse_nvidia_driver_version,
        )
    )

    meminfo_text = files.read_text(root, _MEMINFO)
    meminfo = files.parse_meminfo(meminfo_text) if meminfo_text is not None else {}
    values.extend(
        [
            _maybe("mem_total", meminfo.get("MemTotal"), _MEMINFO, FILE),
            _maybe("mem_available", meminfo.get("MemAvailable"), _MEMINFO, FILE),
        ]
    )

    values.append(_value_from_file("cudnn_version", root, _CUDNN_HEADER, files.parse_cudnn_version))
    values.append(
        _value_from_file(
            "docker_default_runtime",
            root,
            _DOCKER_DAEMON_JSON,
            files.parse_docker_default_runtime,
        )
    )
    return values


def _nvidia_smi_values(run: subp.Runner, which: subp.Which) -> list[Value]:
    """GPU name, driver version and the unified-memory flag, from nvidia-smi."""
    cmd = "nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv"
    gpu = _run_and_parse(
        run,
        which,
        "nvidia-smi",
        ["--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv"],
        subp.parse_nvidia_smi_csv,
    )
    if not gpu:
        return [
            _absent("nvidia_smi_gpu_name", cmd, SUBPROCESS),
            _absent("nvidia_smi_driver_version", cmd, SUBPROCESS),
            _absent("unified_memory", cmd, SUBPROCESS),
        ]
    return [
        _present("nvidia_smi_gpu_name", gpu["name"], cmd, SUBPROCESS),
        _present("nvidia_smi_driver_version", gpu["driver_version"], cmd, SUBPROCESS),
        _present(
            "unified_memory",
            "true" if subp.is_unified_memory(gpu) else "false",
            cmd,
            SUBPROCESS,
        ),
    ]


def _subprocess_values(run: subp.Runner, which: subp.Which) -> list[Value]:
    """Facts no file exposes: nvidia-smi, nvpmodel -q and dpkg-query -W."""
    values = _nvidia_smi_values(run, which)

    power_mode = _run_and_parse(run, which, "nvpmodel", ["-q"], subp.parse_nvpmodel)
    values.append(_maybe("nvpmodel_power_mode", power_mode, "nvpmodel -q", SUBPROCESS))

    def _tensorrt(out: str) -> str | None:
        return subp.parse_dpkg_query(out, "libnvinfer10") or subp.parse_dpkg_query(out, "tensorrt")

    tensorrt_version = _run_and_parse(run, which, "dpkg-query", ["-W"], _tensorrt)
    values.append(_maybe("tensorrt_version", tensorrt_version, "dpkg-query -W", SUBPROCESS))
    return values


def parse_board(model: str | None) -> str | None:
    """``thor``/``orin`` when the device-tree model names exactly one of them."""
    if not model:
        return None
    boards = [board for board, word in _BOARD_WORDS if word.search(model)]
    return boards[0] if len(boards) == 1 else None


def _own_env_executable(own_bin: str, tool: str) -> str | None:
    candidate = os.path.join(own_bin, tool)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _device_cli_values(
    run: subp.Runner, which: subp.Which, own_bin: str, cli: str
) -> tuple[Value, Value]:
    """``<cli>_cli`` (own env first, then PATH) and ``<cli>_cli_version``.

    ``<cli> --version`` runs once, only when the CLI was found; a failed or
    unparseable call is an absent version, never an exception.
    """
    name = f"{cli}_cli"
    version_name = f"{cli}_cli_version"
    version_source = f"{cli} --version"
    found, origin = _own_env_executable(own_bin, cli), NVSH_ENV
    if not found:
        found, origin = which(cli), ON_PATH
    if not found:
        return (
            _absent(name, device_cli_source(cli, None), PATH),
            _absent(version_name, version_source, SUBPROCESS),
        )
    cli_value = _present(name, found, device_cli_source(cli, origin), PATH)
    code, out, _err = run([found, "--version"], subp.DEFAULT_TIMEOUT)
    version = subp.parse_cli_version(out) if code == 0 else None
    return cli_value, _maybe(version_name, version, version_source, SUBPROCESS)


def _path_values(run: subp.Runner, which: subp.Which, own_bin: str) -> list[Value]:
    """PATH presence checks, the device CLIs, and the ``spark status --json``
    merge the spark CLI gates."""
    values = [
        _value_from_which("tmux", which, "tmux"),
        _value_from_which("pi", which, "pi"),
    ]
    # nvsh/ops/render.py's static DEVICE_CLI_VERBS table decides what each
    # CLI supports; detection here answers "where is it, which version" --
    # no --help probe, and nothing at request time.
    device = {cli: _device_cli_values(run, which, own_bin, cli) for cli in _DEVICE_CLIS}
    spark_value = device["spark"][0]
    values.extend([spark_value, device["thor"][0], device["orin"][0]])
    values.extend(version for _cli, version in device.values())

    spark_status_cmd = "spark status --json"
    spark_available = None
    if spark_value.present:
        code, out, _err = run([spark_value.text, "status", "--json"], subp.DEFAULT_TIMEOUT)
        if code == 0:
            spark_available = subp.parse_spark_status(out)
    if spark_available is None:
        values.append(_absent("spark_status_available", spark_status_cmd, SUBPROCESS))
    else:
        values.append(
            _present(
                "spark_status_available",
                "true" if spark_available else "false",
                spark_status_cmd,
                SUBPROCESS,
            )
        )
    return values


def detect(
    root: str = "/",
    run: subp.Runner = subp.default_run,
    which: subp.Which = subp.default_which,
    own_bin: str | None = None,
) -> Platform:
    """Detect the machine nvsh is running on.

    Args:
        root: filesystem root to read files under (``/`` in production; a
            fixture tree in tests).
        run: ``(argv, timeout) -> (returncode, stdout, stderr)``, injectable
            so tests replay captured subprocess output.
        which: ``name -> path or None``, injectable for the same reason.
        own_bin: nvsh's own env bin directory, searched for device CLIs
            before PATH; ``None`` means ``Path(sys.executable).parent`` (the
            uv tool env an extra such as ``nvsh[orin]`` installs into).
    """
    if own_bin is None:
        own_bin = str(Path(sys.executable).parent)
    values = _file_values(root)
    values.extend(_subprocess_values(run, which))
    values.extend(_path_values(run, which, own_bin))
    return Platform(kind=_classify(values), values=tuple(values))


def _classify(values: list[Value]) -> str:
    by_name = {value.name: value for value in values}
    if by_name["dgx_name"].present:
        return "dgx-spark"
    if by_name["l4t_release"].present:
        return "jetson"
    dmi = by_name["dmi_product_name"]
    if dmi.present and dmi.text and "rtx" in dmi.text.lower():
        return "rtx"
    return "generic"
