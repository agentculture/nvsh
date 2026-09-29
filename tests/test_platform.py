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

import pytest

from nvsh.platform import DeviceCli, Platform, Value, detect
from nvsh.platform._files import parse_cuda_version, parse_docker_default_runtime
from nvsh.platform._subprocess import DEFAULT_TIMEOUT, parse_cli_version, parse_spark_status

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "platform")


def _read_fixture(host: str, *parts: str) -> str | None:
    path = os.path.join(FIXTURES, host, *parts)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


#: What each device CLI prints for ``--version`` (measured 2026-09-29 on the
#: fleet: ``dgx-spark-cli 0.7.1`` on the Spark, ``thor 0.5.0``, ``orin 0.5.0``).
CLI_VERSION_OUTPUT = {
    "spark": "dgx-spark-cli 0.7.1\n",
    "thor": "thor 0.5.0\n",
    "orin": "orin 0.5.0\n",
}

#: A bin directory that never exists, so fixture tests never see the real
#: nvsh env's bin directory (which could hold a real device CLI).
NO_OWN_BIN = os.path.join(FIXTURES, "_no_own_bin")


def make_run(host: str, which_map: dict[str, str | None]):
    """Fake subprocess runner backed by tests/fixtures/platform/<host>/subprocess/."""

    def run(argv, timeout=DEFAULT_TIMEOUT):
        assert timeout == DEFAULT_TIMEOUT
        exe = argv[0]
        if argv[1:] == ["--version"] and os.path.basename(exe) in CLI_VERSION_OUTPUT:
            return 0, CLI_VERSION_OUTPUT[os.path.basename(exe)], ""
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
    "pi": "/usr/local/lib/node/bin/pi",
    "spark": "/usr/local/bin/spark",
    # nvpmodel deliberately absent on Spark
}


def _spark_platform() -> Platform:
    root = os.path.join(FIXTURES, "spark")
    return detect(
        root=root,
        run=make_run("spark", SPARK_WHICH),
        which=make_which(SPARK_WHICH),
        own_bin=NO_OWN_BIN,
    )


def test_spark_kind_is_dgx_spark():
    assert _spark_platform().kind == "dgx-spark"


def test_spark_known_facts():
    platform = _spark_platform()
    assert platform.get("dgx_name").text == "DGX Spark"
    assert platform.get("dgx_swbuild_version").text.startswith("7.")
    assert platform.get("cuda_version").text == "13.0.2"
    assert platform.get("nvidia_driver_version").text == "580.126.09"
    assert platform.get("dmi_product_name").text == "NVIDIA_DGX_Spark"


def test_spark_unified_memory_from_nvidia_smi_na():
    platform = _spark_platform()
    unified = platform.get("unified_memory")
    assert unified.present is True
    assert unified.text == "true"
    assert unified.method == "subprocess"


def test_spark_mem_available_is_pressure_signal():
    platform = _spark_platform()
    mem_available = platform.get("mem_available")
    assert mem_available.present is True
    assert mem_available.source == "/proc/meminfo"
    assert mem_available.text.endswith("kB")


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
    assert platform.get("spark_status_available").text == "true"


# --- THOR ----------------------------------------------------------------


THOR_WHICH = {
    "nvidia-smi": "/usr/sbin/nvidia-smi",
    "nvpmodel": "/usr/sbin/nvpmodel",
    "dpkg-query": "/usr/bin/dpkg-query",
    "tmux": "/usr/bin/tmux",
    "thor": "/usr/local/bin/thor",
    # pi, spark and orin absent on thor
}


def _thor_platform() -> Platform:
    root = os.path.join(FIXTURES, "thor")
    return detect(
        root=root,
        run=make_run("thor", THOR_WHICH),
        which=make_which(THOR_WHICH),
        own_bin=NO_OWN_BIN,
    )


def test_thor_kind_is_jetson():
    assert _thor_platform().kind == "jetson"


def test_thor_known_facts():
    platform = _thor_platform()
    assert platform.get("dgx_name").present is False
    l4t = platform.get("l4t_release")
    assert l4t.present is True
    assert "R38" in l4t.text
    assert "REVISION: 2.2" in l4t.text
    assert platform.get("device_tree_model").text == "NVIDIA Jetson AGX Thor Developer Kit"
    assert "tegra264" in platform.get("device_tree_compatible").text
    assert platform.get("cuda_version").text == "13.0.0"
    assert platform.get("cudnn_version").text == "9.12.0"
    tensorrt = platform.get("tensorrt_version")
    assert tensorrt.present is True
    assert tensorrt.text.startswith("10.13")
    assert platform.get("docker_default_runtime").text == "nvidia"


def test_thor_nvpmodel_and_presence_flags():
    platform = _thor_platform()
    assert platform.get("nvpmodel_power_mode").text == "MAXN"
    assert platform.get("tmux").present is True
    assert platform.get("pi").present is False
    assert platform.get("spark_cli").present is False
    # spark_status_available still reported, just absent since spark isn't on PATH
    assert platform.get("spark_status_available") is not None
    assert platform.get("spark_status_available").present is False


def test_thor_cli_present_orin_cli_absent():
    platform = _thor_platform()
    thor_cli = platform.get("thor_cli")
    assert thor_cli is not None
    assert thor_cli.present is True
    assert thor_cli.text == "/usr/local/bin/thor"
    orin_cli = platform.get("orin_cli")
    assert orin_cli is not None
    assert orin_cli.present is False


def test_thor_unified_memory():
    platform = _thor_platform()
    assert platform.get("unified_memory").text == "true"


# --- ORIN ------------------------------------------------------------------


ORIN_WHICH = {
    "nvidia-smi": "/usr/sbin/nvidia-smi",
    "nvpmodel": "/usr/sbin/nvpmodel",
    "dpkg-query": "/usr/bin/dpkg-query",
    "orin": "/usr/local/bin/orin",
    # tmux, pi, spark and thor all absent on orin
}


def _orin_platform() -> Platform:
    root = os.path.join(FIXTURES, "orin")
    return detect(
        root=root,
        run=make_run("orin", ORIN_WHICH),
        which=make_which(ORIN_WHICH),
        own_bin=NO_OWN_BIN,
    )


def test_orin_kind_is_jetson():
    assert _orin_platform().kind == "jetson"


def test_orin_known_facts():
    platform = _orin_platform()
    l4t = platform.get("l4t_release")
    assert l4t.present is True
    assert "R39" in l4t.text
    assert "REVISION: 2.0" in l4t.text
    assert platform.get("device_tree_model").text == "NVIDIA Jetson AGX Orin Developer Kit"
    assert "tegra234" in platform.get("device_tree_compatible").text
    assert platform.get("nvpmodel_power_mode").text == "MAXN"


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


def test_orin_cli_present_thor_cli_absent():
    platform = _orin_platform()
    orin_cli = platform.get("orin_cli")
    assert orin_cli is not None
    assert orin_cli.present is True
    assert orin_cli.text == "/usr/local/bin/orin"
    thor_cli = platform.get("thor_cli")
    assert thor_cli is not None
    assert thor_cli.present is False


# --- GENERIC / EMPTY ROOT ----------------------------------------------------


def test_generic_root_reports_every_value_as_absent_not_omitted(tmp_path):
    platform = detect(
        root=str(tmp_path),
        run=lambda argv, timeout=DEFAULT_TIMEOUT: (1, "", "not found"),
        which=lambda name: None,
        own_bin=NO_OWN_BIN,
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
        "thor_cli",
        "orin_cli",
        "jetson_board",
        "spark_cli_version",
        "thor_cli_version",
        "orin_cli_version",
    }
    assert expected <= names
    for value in platform.values:
        assert value.present is False
        assert value.text is None
        assert value.source


def test_rtx_kind_from_dmi_product_name(tmp_path):
    dmi_dir = tmp_path / "sys" / "class" / "dmi" / "id"
    dmi_dir.mkdir(parents=True)
    (dmi_dir / "product_name").write_text("NVIDIA RTX Spark\n")
    platform = detect(
        root=str(tmp_path),
        run=lambda argv, timeout=DEFAULT_TIMEOUT: (1, "", "not found"),
        which=lambda name: None,
        own_bin=NO_OWN_BIN,
    )
    assert platform.kind == "rtx"


# --- Value / Platform model ---------------------------------------------


def test_value_invariants():
    Value(name="x", text="1", source="s", method="file", present=True)
    Value(name="x", text=None, source="s", method="file", present=False)
    try:
        Value(name="x", text="1", source="s", method="file", present=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        Value(name="x", text=None, source="s", method="file", present=True)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        Value(name="x", text=None, source="s", method="bogus", present=False)
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


# --- PR #8 review: wrong-shaped JSON is an absent fact, not an exception ----


@pytest.mark.parametrize("text", ["null", "[1, 2]", '"a string"', "3", '{"cuda": "13.0"}'])
def test_parse_cuda_version_survives_wrong_shaped_json(text):
    assert parse_cuda_version(text) is None


@pytest.mark.parametrize("text", ["null", "[1, 2]", '"a string"', "3"])
def test_parse_docker_default_runtime_survives_wrong_shaped_json(text):
    assert parse_docker_default_runtime(text) is None


@pytest.mark.parametrize("text", ["null", "[1, 2]", '"a string"', "3"])
def test_parse_spark_status_survives_wrong_shaped_json(text):
    assert parse_spark_status(text) is None


def test_parse_cuda_version_still_reads_a_well_shaped_file():
    assert parse_cuda_version('{"cuda": {"version": "13.0.2"}}') == "13.0.2"


# --- Device CLIs: Thor/Orin board split, own-env lookup, --version -------
#
# Plan device-cli-alignment-spark-thor-orin, task t7 (claims c6/h6, c21/h16,
# c22/h17, decision c27). The board comes from /proc/device-tree/model; a
# device CLI is looked up in nvsh's own env bin directory first
# (Path(sys.executable).parent -- where `uv tool install 'nvsh[orin]'` puts
# `orin` without exposing it on PATH), then PATH; each CLI found has its
# `--version` read exactly once.


def _both_on_path(which_map: dict[str, str]) -> dict[str, str]:
    return {**which_map, "thor": "/usr/local/bin/thor", "orin": "/usr/local/bin/orin"}


def _counting(run):
    calls: list[list[str]] = []

    def wrapped(argv, timeout=DEFAULT_TIMEOUT):
        calls.append(list(argv))
        return run(argv, timeout)

    return wrapped, calls


@pytest.mark.parametrize(
    "host, base_which, board",
    [("thor", THOR_WHICH, "thor"), ("orin", ORIN_WHICH, "orin")],
)
def test_board_from_device_tree_model_with_both_clis_on_path(host, base_which, board):
    which_map = _both_on_path(base_which)
    platform = detect(
        root=os.path.join(FIXTURES, host),
        run=make_run(host, which_map),
        which=make_which(which_map),
        own_bin=NO_OWN_BIN,
    )
    assert platform.kind == "jetson"  # kind is unchanged; the board is a new value
    jetson_board = platform.get("jetson_board")
    assert jetson_board.present is True
    assert jetson_board.text == board
    assert jetson_board.method == "file"
    assert jetson_board.source == "/proc/device-tree/model"
    for cli in ("thor", "orin"):
        value = platform.get(f"{cli}_cli")
        assert value.present is True
        assert value.text == f"/usr/local/bin/{cli}"
        assert value.source == f"{cli} (PATH)"
        version = platform.get(f"{cli}_cli_version")
        assert version.present is True
        assert version.text == "0.5.0"
        assert version.method == "subprocess"
        assert version.source == f"{cli} --version"


def _jetson_root(tmp_path, model: bytes | None):
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "nv_tegra_release").write_text("# R38 (release), REVISION: 2.2\n")
    if model is not None:
        dt = tmp_path / "proc" / "device-tree"
        dt.mkdir(parents=True)
        (dt / "model").write_bytes(model)
    return str(tmp_path)


def _no_run(argv, timeout=DEFAULT_TIMEOUT):
    return 1, "", "not found"


def test_model_unreadable_reports_no_board_and_keeps_both_clis(tmp_path):
    root = _jetson_root(tmp_path, model=None)
    which_map = {"thor": "/usr/local/bin/thor", "orin": "/usr/local/bin/orin"}
    platform = detect(
        root=root, run=make_run("thor", which_map), which=make_which(which_map), own_bin=NO_OWN_BIN
    )
    assert platform.kind == "jetson"
    jetson_board = platform.get("jetson_board")
    assert jetson_board.present is False
    assert jetson_board.source == "/proc/device-tree/model"
    # No board: both CLIs are still reported, so the caller falls back to
    # its PATH order (thor, then orin) exactly as before.
    assert platform.get("thor_cli").present is True
    assert platform.get("orin_cli").present is True


@pytest.mark.parametrize(
    "model, board",
    [
        (b"NVIDIA Jetson AGX Thor Developer Kit\x00", "thor"),
        (b"NVIDIA Jetson AGX Orin Developer Kit\x00", "orin"),
        (b"NVIDIA Jetson Orin Nano Developer Kit\x00", "orin"),
        (b"NVIDIA Jetson Xavier NX Developer Kit\x00", None),
        (b"Thorough Test Board\x00", None),
        (b"\x00", None),
    ],
)
def test_board_matches_the_word_thor_or_orin_only(tmp_path, model, board):
    platform = detect(
        root=_jetson_root(tmp_path, model), run=_no_run, which=lambda n: None, own_bin=NO_OWN_BIN
    )
    jetson_board = platform.get("jetson_board")
    assert jetson_board.text == board
    assert jetson_board.present is (board is not None)


def test_spark_has_no_board():
    assert _spark_platform().get("jetson_board").present is False


def _own_bin_with(tmp_path, *names: str, executable: bool = True):
    own_bin = tmp_path / "tool-env" / "bin"
    own_bin.mkdir(parents=True)
    for name in names:
        exe = own_bin / name
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755 if executable else 0o644)
    return own_bin


def test_cli_only_in_own_env_is_found_with_version(tmp_path):
    own_bin = _own_bin_with(tmp_path, "orin")
    run, calls = _counting(make_run("orin", {}))
    platform = detect(
        root=os.path.join(FIXTURES, "orin"),
        run=run,
        which=lambda name: None,  # nothing on PATH at all
        own_bin=str(own_bin),
    )
    orin_cli = platform.get("orin_cli")
    assert orin_cli.present is True
    assert orin_cli.text == str(own_bin / "orin")
    assert orin_cli.method == "path"
    assert orin_cli.source == "orin (nvsh env)"
    assert platform.get("orin_cli_version").text == "0.5.0"
    assert platform.get("thor_cli").present is False
    assert platform.get("jetson_board").text == "orin"
    assert calls == [[str(own_bin / "orin"), "--version"]]


def test_own_env_wins_over_path(tmp_path):
    own_bin = _own_bin_with(tmp_path, "thor")
    which_map = {"thor": "/usr/local/bin/thor"}
    platform = detect(
        root=os.path.join(FIXTURES, "thor"),
        run=make_run("thor", which_map),
        which=make_which(which_map),
        own_bin=str(own_bin),
    )
    thor_cli = platform.get("thor_cli")
    assert thor_cli.text == str(own_bin / "thor")
    assert thor_cli.source == "thor (nvsh env)"


def test_non_executable_file_in_own_env_is_ignored(tmp_path):
    own_bin = _own_bin_with(tmp_path, "thor", executable=False)
    which_map = {"thor": "/usr/local/bin/thor"}
    platform = detect(
        root=os.path.join(FIXTURES, "thor"),
        run=make_run("thor", which_map),
        which=make_which(which_map),
        own_bin=str(own_bin),
    )
    assert platform.get("thor_cli").text == "/usr/local/bin/thor"
    assert platform.get("thor_cli").source == "thor (PATH)"


def test_absent_cli_records_both_places_checked_and_no_version_call(tmp_path):
    run, calls = _counting(_no_run)
    platform = detect(root=str(tmp_path), run=run, which=lambda n: None, own_bin=NO_OWN_BIN)
    for cli in ("spark", "thor", "orin"):
        value = platform.get(f"{cli}_cli")
        assert value.present is False
        assert value.source == f"{cli} (nvsh env, PATH)"
        version = platform.get(f"{cli}_cli_version")
        assert version.present is False
        assert version.source == f"{cli} --version"
    assert not any(argv[1:] == ["--version"] for argv in calls)


def test_default_own_bin_is_the_python_executables_directory(tmp_path, monkeypatch):
    own_bin = _own_bin_with(tmp_path, "orin")
    monkeypatch.setattr("sys.executable", str(own_bin / "python"))
    platform = detect(
        root=os.path.join(FIXTURES, "orin"), run=make_run("orin", {}), which=lambda n: None
    )
    assert platform.get("orin_cli").text == str(own_bin / "orin")


def test_version_read_once_per_present_cli():
    which_map = _both_on_path(SPARK_WHICH)
    run, calls = _counting(make_run("spark", which_map))
    detect(
        root=os.path.join(FIXTURES, "spark"),
        run=run,
        which=make_which(which_map),
        own_bin=NO_OWN_BIN,
    )
    version_calls = sorted(argv[0] for argv in calls if argv[1:] == ["--version"])
    assert version_calls == ["/usr/local/bin/orin", "/usr/local/bin/spark", "/usr/local/bin/thor"]


def test_spark_version_parsed_from_dgx_spark_cli_banner():
    version = _spark_platform().get("spark_cli_version")
    assert version.present is True
    assert version.text == "0.7.1"


@pytest.mark.parametrize(
    "result",
    [
        (1, "thor 0.5.0\n", ""),  # non-zero exit
        (0, "", ""),  # no output
        (0, "thor version unknown\n", ""),  # last token is not a version
        (1, "", ""),  # timeout / OSError, as default_run reports it
    ],
)
def test_unusable_version_output_is_absent_not_raised(result):
    which_map = {"thor": "/usr/local/bin/thor"}

    def run(argv, timeout=DEFAULT_TIMEOUT):
        if argv == ["/usr/local/bin/thor", "--version"]:
            return result
        return 1, "", ""

    platform = detect(
        root=os.path.join(FIXTURES, "thor"), run=run, which=make_which(which_map), own_bin=NO_OWN_BIN
    )
    assert platform.get("thor_cli").present is True
    version = platform.get("thor_cli_version")
    assert version.present is False
    assert version.text is None


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ("dgx-spark-cli 0.7.1\n", "0.7.1"),
        ("thor 0.5.0\n", "0.5.0"),
        ("orin 0.5.0", "0.5.0"),
        ("orin v0.6.0rc1\n", "0.6.0rc1"),
        ("orin 1.2.3.dev4\n", "1.2.3.dev4"),
        ("orin\n", None),
        ("", None),
        ("orin unknown", None),
    ],
)
def test_parse_cli_version(stdout, expected):
    assert parse_cli_version(stdout) == expected


def test_device_cli_accessor(tmp_path):
    own_bin = _own_bin_with(tmp_path, "orin")
    which_map = {"thor": "/usr/local/bin/thor"}
    platform = detect(
        root=os.path.join(FIXTURES, "orin"),
        run=make_run("orin", which_map),
        which=make_which(which_map),
        own_bin=str(own_bin),
    )
    orin = platform.device_cli("orin")
    assert orin == DeviceCli(
        name="orin", path=str(own_bin / "orin"), origin="nvsh-env", version="0.5.0"
    )
    thor = platform.device_cli("thor")
    assert thor == DeviceCli(name="thor", path="/usr/local/bin/thor", origin="path", version="0.5.0")
    assert platform.device_cli("spark") is None
    assert platform.board() == "orin"
    assert Platform(kind="jetson").device_cli("thor") is None
    assert Platform(kind="jetson").board() is None
