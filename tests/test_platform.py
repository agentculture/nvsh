"""Tests for nvsh.platform.detect() against fixture trees copied from real
DGX Spark, Jetson AGX Thor and Jetson AGX Orin hardware.

Fixtures live under tests/fixtures/platform/{spark,thor,orin}/, mirroring
the absolute path layout under each root (e.g. .../thor/etc/nv_tegra_release
stands in for /etc/nv_tegra_release on thor), plus a subprocess/ directory
holding real captured output for nvidia-smi, nvpmodel and dpkg-query so the
detector is fully testable without shelling out.
"""

from __future__ import annotations

import os

from nvsh.platform import Platform, Value, detect
from nvsh.platform._subprocess import DEFAULT_TIMEOUT

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "platform")


def _read_fixture(host: str, *parts: str) -> str | None:
    path = os.path.join(FIXTURES, host, *parts)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def make_run(host: str, which_map: dict[str, str | None]):
    """Fake subprocess runner backed by tests/fixtures/platform/<host>/subprocess/."""

    def run(argv, timeout=DEFAULT_TIMEOUT):
        assert timeout == DEFAULT_TIMEOUT
        exe = argv[0]
        if exe == which_map.get("nvidia-smi"):
            text = _read_fixture(host, "subprocess", "nvidia-smi.csv")
        elif exe == which_map.get("nvpmodel"):
            text = _read_fixture(host, "subprocess", "nvpmodel.txt")
        elif exe == which_map.get("dpkg-query"):
            text = _read_fixture(host, "subprocess", "dpkg-query.txt")
        elif exe == which_map.get("spark"):
            text = _read_fixture(host, "subprocess", "spark-status.json")
        else:
            raise AssertionError(f"unexpected subprocess call: {argv}")
        if text is None:
            return 1, "", "not found"
        return 0, text, ""

    return run


def make_which(present: dict[str, str]):
    def which(name: str) -> str | None:
        return present.get(name)

    return which


# --- SPARK -------------------------------------------------------------


SPARK_WHICH = {
    "nvidia-smi": "/usr/bin/nvidia-smi",
    "dpkg-query": "/usr/bin/dpkg-query",
    "tmux": "/usr/bin/tmux",
    "pi": "/home/spark/.nvm/versions/node/v24.13.1/bin/pi",
    "spark": "/home/spark/.local/bin/spark",
    # nvpmodel deliberately absent on Spark
}


def _spark_platform() -> Platform:
    root = os.path.join(FIXTURES, "spark")
    return detect(root=root, run=make_run("spark", SPARK_WHICH), which=make_which(SPARK_WHICH))


def test_spark_kind_is_dgx_spark():
    assert _spark_platform().kind == "dgx-spark"


def test_spark_known_facts():
    platform = _spark_platform()
    assert platform.get("dgx_name").value == "DGX Spark"
    assert platform.get("dgx_swbuild_version").value.startswith("7.")
    assert platform.get("cuda_version").value == "13.0.2"
    assert platform.get("nvidia_driver_version").value == "580.126.09"
    assert platform.get("dmi_product_name").value == "NVIDIA_DGX_Spark"


def test_spark_unified_memory_from_nvidia_smi_na():
    platform = _spark_platform()
    unified = platform.get("unified_memory")
    assert unified.present is True
    assert unified.value == "true"
    assert unified.method == "subprocess"


def test_spark_mem_available_is_pressure_signal():
    platform = _spark_platform()
    mem_available = platform.get("mem_available")
    assert mem_available.present is True
    assert mem_available.source == "/proc/meminfo"
    assert mem_available.value.endswith("kB")


def test_spark_absent_facts_reported_not_omitted():
    platform = _spark_platform()
    for name in ("l4t_release", "device_tree_model", "cudnn_version", "nvpmodel_power_mode"):
        value = platform.get(name)
        assert value is not None, f"{name} must never be omitted"
        assert value.present is False
        assert value.source  # source recorded even when absent


def test_spark_tmux_pi_spark_cli_present():
    platform = _spark_platform()
    assert platform.get("tmux").present is True
    assert platform.get("pi").present is True
    assert platform.get("spark_cli").present is True
    assert platform.get("spark_status_available").present is True
    assert platform.get("spark_status_available").value == "true"


# --- THOR ----------------------------------------------------------------


THOR_WHICH = {
    "nvidia-smi": "/usr/sbin/nvidia-smi",
    "nvpmodel": "/usr/sbin/nvpmodel",
    "dpkg-query": "/usr/bin/dpkg-query",
    "tmux": "/usr/bin/tmux",
    # pi and spark absent on thor
}


def _thor_platform() -> Platform:
    root = os.path.join(FIXTURES, "thor")
    return detect(root=root, run=make_run("thor", THOR_WHICH), which=make_which(THOR_WHICH))


def test_thor_kind_is_jetson():
    assert _thor_platform().kind == "jetson"


def test_thor_known_facts():
    platform = _thor_platform()
    assert platform.get("dgx_name").present is False
    l4t = platform.get("l4t_release")
    assert l4t.present is True
    assert "R38" in l4t.value and "REVISION: 2.2" in l4t.value
    assert platform.get("device_tree_model").value == "NVIDIA Jetson AGX Thor Developer Kit"
    assert "tegra264" in platform.get("device_tree_compatible").value
    assert platform.get("cuda_version").value == "13.0.0"
    assert platform.get("cudnn_version").value == "9.12.0"
    tensorrt = platform.get("tensorrt_version")
    assert tensorrt.present is True
    assert tensorrt.value.startswith("10.13")
    assert platform.get("docker_default_runtime").value == "nvidia"


def test_thor_nvpmodel_and_presence_flags():
    platform = _thor_platform()
    assert platform.get("nvpmodel_power_mode").value == "MAXN"
    assert platform.get("tmux").present is True
    assert platform.get("pi").present is False
    assert platform.get("spark_cli").present is False
    # spark_status_available still reported, just absent since spark isn't on PATH
    assert platform.get("spark_status_available") is not None
    assert platform.get("spark_status_available").present is False


def test_thor_unified_memory():
    platform = _thor_platform()
    assert platform.get("unified_memory").value == "true"


# --- ORIN ------------------------------------------------------------------


ORIN_WHICH = {
    "nvidia-smi": "/usr/sbin/nvidia-smi",
    "nvpmodel": "/usr/sbin/nvpmodel",
    "dpkg-query": "/usr/bin/dpkg-query",
    # tmux, pi, spark all absent on orin
}


def _orin_platform() -> Platform:
    root = os.path.join(FIXTURES, "orin")
    return detect(root=root, run=make_run("orin", ORIN_WHICH), which=make_which(ORIN_WHICH))


def test_orin_kind_is_jetson():
    assert _orin_platform().kind == "jetson"


def test_orin_known_facts():
    platform = _orin_platform()
    l4t = platform.get("l4t_release")
    assert l4t.present is True
    assert "R39" in l4t.value and "REVISION: 2.0" in l4t.value
    assert platform.get("device_tree_model").value == "NVIDIA Jetson AGX Orin Developer Kit"
    assert "tegra234" in platform.get("device_tree_compatible").value
    assert platform.get("nvpmodel_power_mode").value == "MAXN"


def test_orin_cuda_toolkit_and_tensorrt_absent():
    platform = _orin_platform()
    cuda = platform.get("cuda_version")
    assert cuda.present is False
    assert cuda.source == "/usr/local/cuda/version.json"
    tensorrt = platform.get("tensorrt_version")
    assert tensorrt.present is False
    cudnn = platform.get("cudnn_version")
    assert cudnn.present is False


def test_orin_docker_daemon_present_but_no_default_runtime():
    # orin's daemon.json exists but has no "default-runtime" key: file is
    # present, but the specific fact is absent - not the same as a missing file.
    platform = _orin_platform()
    runtime = platform.get("docker_default_runtime")
    assert runtime.present is False
    assert runtime.source == "/etc/docker/daemon.json"


def test_orin_tmux_pi_spark_absent():
    platform = _orin_platform()
    assert platform.get("tmux").present is False
    assert platform.get("pi").present is False
    assert platform.get("spark_cli").present is False


# --- GENERIC / EMPTY ROOT ----------------------------------------------------


def test_generic_root_reports_every_value_as_absent_not_omitted(tmp_path):
    platform = detect(
        root=str(tmp_path),
        run=lambda argv, timeout=DEFAULT_TIMEOUT: (1, "", "not found"),
        which=lambda name: None,
    )
    assert platform.kind == "generic"
    names = {v.name for v in platform.values}
    expected = {
        "dgx_name",
        "dgx_swbuild_version",
        "l4t_release",
        "device_tree_model",
        "device_tree_compatible",
        "dmi_product_name",
        "cuda_version",
        "nvidia_driver_version",
        "mem_total",
        "mem_available",
        "unified_memory",
        "cudnn_version",
        "tensorrt_version",
        "nvpmodel_power_mode",
        "docker_default_runtime",
        "tmux",
        "pi",
        "spark_cli",
        "spark_status_available",
    }
    assert expected <= names
    for value in platform.values:
        assert value.present is False
        assert value.value is None
        assert value.source


def test_rtx_kind_from_dmi_product_name(tmp_path):
    dmi_dir = tmp_path / "sys" / "class" / "dmi" / "id"
    dmi_dir.mkdir(parents=True)
    (dmi_dir / "product_name").write_text("NVIDIA RTX Spark\n")
    platform = detect(
        root=str(tmp_path),
        run=lambda argv, timeout=DEFAULT_TIMEOUT: (1, "", "not found"),
        which=lambda name: None,
    )
    assert platform.kind == "rtx"


# --- Value / Platform model ---------------------------------------------


def test_value_invariants():
    Value(name="x", value="1", source="s", method="file", present=True)
    Value(name="x", value=None, source="s", method="file", present=False)
    try:
        Value(name="x", value="1", source="s", method="file", present=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        Value(name="x", value=None, source="s", method="file", present=True)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        Value(name="x", value=None, source="s", method="bogus", present=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_platform_to_dict_json_roundtrip():
    import json

    platform = _thor_platform()
    encoded = json.dumps(platform.to_dict())
    decoded = json.loads(encoded)
    assert decoded["kind"] == "jetson"
    assert any(v["name"] == "l4t_release" for v in decoded["values"])


def test_platform_render_block_lists_every_value_with_source():
    platform = _thor_platform()
    block = platform.render_block()
    assert block.startswith("platform: jetson")
    for value in platform.values:
        assert value.name in block
        assert value.source in block
