"""``detect()``: build a :class:`Platform` for the current machine.

File-first, as required by the spec's platform-detection claim: every fact
is read from a file when a file exposes it, and subprocesses are used only
for ``nvidia-smi``, ``nvpmodel -q`` and ``dpkg-query -W`` (each with a
timeout), plus an optional ``spark status --json`` merge when the
``dgx-spark-cli`` ``spark`` binary is on PATH. Nothing is ever required to
be present: a missing file, absent binary or unparseable value comes back as
an absent :class:`Value` carrying the source that was checked, not a
silently dropped fact.
"""

from __future__ import annotations

from . import _files as files
from . import _subprocess as subp
from ._model import FILE, PATH, SUBPROCESS, Platform, Value

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


def _absent(name: str, source: str, method: str) -> Value:
    return Value(name=name, value=None, source=source, method=method, present=False)


def _present(name: str, value: str, source: str, method: str) -> Value:
    return Value(name=name, value=value, source=source, method=method, present=True)


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


def detect(
    root: str = "/",
    run: subp.Runner = subp.default_run,
    which: subp.Which = subp.default_which,
) -> Platform:
    """Detect the machine nvsh is running on.

    Args:
        root: filesystem root to read files under (``/`` in production; a
            fixture tree in tests).
        run: ``(argv, timeout) -> (returncode, stdout, stderr)``, injectable
            so tests replay captured subprocess output.
        which: ``name -> path or None``, injectable for the same reason.
    """
    values: list[Value] = []

    dgx = files.read_text(root, _DGX_RELEASE)
    dgx_fields = files.parse_dgx_release(dgx) if dgx is not None else {}
    if dgx is not None and dgx_fields.get("DGX_NAME"):
        values.append(_present("dgx_name", dgx_fields["DGX_NAME"], _DGX_RELEASE, FILE))
    else:
        values.append(_absent("dgx_name", _DGX_RELEASE, FILE))
    if dgx is not None and dgx_fields.get("DGX_SWBUILD_VERSION"):
        values.append(
            _present("dgx_swbuild_version", dgx_fields["DGX_SWBUILD_VERSION"], _DGX_RELEASE, FILE)
        )
    else:
        values.append(_absent("dgx_swbuild_version", _DGX_RELEASE, FILE))

    nv_tegra = files.read_text(root, _NV_TEGRA_RELEASE)
    l4t_release = files.parse_nv_tegra_release(nv_tegra) if nv_tegra is not None else None
    if l4t_release:
        values.append(_present("l4t_release", l4t_release, _NV_TEGRA_RELEASE, FILE))
    else:
        values.append(_absent("l4t_release", _NV_TEGRA_RELEASE, FILE))

    dt_model = files.read_device_tree_string(root, _DEVICE_TREE_MODEL)
    if dt_model:
        values.append(_present("device_tree_model", dt_model, _DEVICE_TREE_MODEL, FILE))
    else:
        values.append(_absent("device_tree_model", _DEVICE_TREE_MODEL, FILE))

    dt_compatible = files.read_device_tree_list(root, _DEVICE_TREE_COMPATIBLE)
    if dt_compatible:
        values.append(
            _present(
                "device_tree_compatible", ",".join(dt_compatible), _DEVICE_TREE_COMPATIBLE, FILE
            )
        )
    else:
        values.append(_absent("device_tree_compatible", _DEVICE_TREE_COMPATIBLE, FILE))

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
    if meminfo.get("MemTotal"):
        values.append(_present("mem_total", meminfo["MemTotal"], _MEMINFO, FILE))
    else:
        values.append(_absent("mem_total", _MEMINFO, FILE))
    if meminfo.get("MemAvailable"):
        values.append(_present("mem_available", meminfo["MemAvailable"], _MEMINFO, FILE))
    else:
        values.append(_absent("mem_available", _MEMINFO, FILE))

    values.append(_value_from_file("cudnn_version", root, _CUDNN_HEADER, files.parse_cudnn_version))
    values.append(
        _value_from_file(
            "docker_default_runtime",
            root,
            _DOCKER_DAEMON_JSON,
            files.parse_docker_default_runtime,
        )
    )

    # --- subprocess-only facts --------------------------------------------

    nvidia_smi_path = which("nvidia-smi")
    nvidia_smi_cmd = (
        "nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv"
    )
    gpu = None
    if nvidia_smi_path:
        code, out, _err = run(
            [
                nvidia_smi_path,
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv",
            ],
            subp.DEFAULT_TIMEOUT,
        )
        if code == 0:
            gpu = subp.parse_nvidia_smi_csv(out)
    if gpu:
        values.append(_present("nvidia_smi_gpu_name", gpu["name"], nvidia_smi_cmd, SUBPROCESS))
        values.append(
            _present("nvidia_smi_driver_version", gpu["driver_version"], nvidia_smi_cmd, SUBPROCESS)
        )
        values.append(
            _present(
                "unified_memory",
                "true" if subp.is_unified_memory(gpu) else "false",
                nvidia_smi_cmd,
                SUBPROCESS,
            )
        )
    else:
        values.append(_absent("nvidia_smi_gpu_name", nvidia_smi_cmd, SUBPROCESS))
        values.append(_absent("nvidia_smi_driver_version", nvidia_smi_cmd, SUBPROCESS))
        values.append(_absent("unified_memory", nvidia_smi_cmd, SUBPROCESS))

    nvpmodel_path = which("nvpmodel")
    nvpmodel_cmd = "nvpmodel -q"
    power_mode = None
    if nvpmodel_path:
        code, out, _err = run([nvpmodel_path, "-q"], subp.DEFAULT_TIMEOUT)
        if code == 0:
            power_mode = subp.parse_nvpmodel(out)
    if power_mode:
        values.append(_present("nvpmodel_power_mode", power_mode, nvpmodel_cmd, SUBPROCESS))
    else:
        values.append(_absent("nvpmodel_power_mode", nvpmodel_cmd, SUBPROCESS))

    dpkg_path = which("dpkg-query")
    dpkg_cmd = "dpkg-query -W"
    tensorrt_version = None
    if dpkg_path:
        code, out, _err = run([dpkg_path, "-W"], subp.DEFAULT_TIMEOUT)
        if code == 0:
            tensorrt_version = subp.parse_dpkg_query(out, "libnvinfer10") or subp.parse_dpkg_query(
                out, "tensorrt"
            )
    if tensorrt_version:
        values.append(_present("tensorrt_version", tensorrt_version, dpkg_cmd, SUBPROCESS))
    else:
        values.append(_absent("tensorrt_version", dpkg_cmd, SUBPROCESS))

    # --- PATH-only presence checks ----------------------------------------

    values.append(_value_from_which("tmux", which, "tmux"))
    values.append(_value_from_which("pi", which, "pi"))
    spark_value = _value_from_which("spark_cli", which, "spark")
    values.append(spark_value)

    spark_status_cmd = "spark status --json"
    spark_available = None
    if spark_value.present:
        code, out, _err = run([spark_value.value, "status", "--json"], subp.DEFAULT_TIMEOUT)
        if code == 0:
            spark_available = subp.parse_spark_status(out)
    if spark_available is not None:
        values.append(
            _present(
                "spark_status_available",
                "true" if spark_available else "false",
                spark_status_cmd,
                SUBPROCESS,
            )
        )
    else:
        values.append(_absent("spark_status_available", spark_status_cmd, SUBPROCESS))

    kind = _classify(values)
    return Platform(kind=kind, values=tuple(values))


def _classify(values: list[Value]) -> str:
    by_name = {value.name: value for value in values}
    if by_name["dgx_name"].present:
        return "dgx-spark"
    if by_name["l4t_release"].present:
        return "jetson"
    dmi = by_name["dmi_product_name"]
    if dmi.present and dmi.value and "rtx" in dmi.value.lower():
        return "rtx"
    return "generic"
