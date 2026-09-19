"""Tier 2's Docker launcher: what it renders, and what it refuses to run.

Nothing here executes ``docker``. Every test drives
:class:`~nvsh.tiers.runtime_docker.DockerRuntime` through an injected
runner that records the argv it was handed, an injected probe, and an
injected clock/sleep pair, so the suite never waits and never touches the
machine's container runtime.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from nvsh.config import _TIERS_LFM_ENGINES
from nvsh.platform._model import Platform
from nvsh.tiers import runtime_docker as rd
from nvsh.tiers.memfloor import FloorResult
from nvsh.tiers.runtime import AttachedRuntime, RuntimeUnavailable

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "nvsh" / "tiers" / "runtime_docker.py"

DIGEST = "sha256:" + "ab" * 32
IMAGE = "example.invalid/tier2@" + DIGEST
SPARK = Platform(kind="dgx-spark")
JETSON = Platform(kind="jetson")
UNKNOWN = Platform(kind="unknown")


def settings(**overrides: object) -> dict[str, object]:
    """A managed llama-server config with everything the template needs."""
    base: dict[str, object] = {
        "engine": "llama-server",
        "mode": "managed",
        "image": IMAGE,
        "model": "lfm2.gguf",
        "model_dir": "/var/lib/nvsh/models",
    }
    base.update(overrides)
    return base


class FakeDocker:
    """Records every argv it is handed; answers from a prefix script."""

    def __init__(self, script: list[tuple[tuple[str, ...], tuple[int, str]]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._script = script or []

    def __call__(self, argv: list[str], timeout: float) -> tuple[int, str]:
        self.calls.append(list(argv))
        for prefix, result in self._script:
            if tuple(argv[: len(prefix)]) == prefix:
                return result
        return (0, "")

    @property
    def verbs(self) -> list[str]:
        """The second word of each call -- ``run``, ``rm``, ``stop``, ..."""
        return [call[1] for call in self.calls if len(call) > 1]


class ExplodingDocker:
    """A runner that fails the test if anything tries to call docker."""

    def __call__(self, argv: list[str], timeout: float) -> tuple[int, str]:
        raise AssertionError(f"docker must not be called, got {argv!r}")


class FakeClock:
    """A monotonic clock that only moves when ``sleep`` is called."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def runtime(runner: object, *, probe: object = None, **kwargs: object) -> rd.DockerRuntime:
    """A DockerRuntime wired to fakes; uid is fixed so names are predictable."""
    clock = kwargs.pop("clock", None) or FakeClock()
    return rd.DockerRuntime(
        kwargs.pop("settings", None) or settings(),
        kwargs.pop("platform", None) or SPARK,
        runner=runner,
        uid=lambda: 1000,
        probe=probe if probe is not None else (lambda url, timeout: True),
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


# -- render_launch: the GPU flag table ------------------------------------


def test_spark_launch_line_uses_gpus_all():
    assert "--gpus all" in " ".join(rd.render_launch(settings(), SPARK, uid=1000))


def test_jetson_launch_line_uses_runtime_nvidia():
    assert "--runtime nvidia" in " ".join(rd.render_launch(settings(), JETSON, uid=1000))


def test_jetson_launch_line_never_uses_gpus_all():
    assert "--gpus" not in rd.render_launch(settings(), JETSON, uid=1000)


def test_unknown_platform_falls_back_to_cpu():
    argv = rd.render_launch(settings(), UNKNOWN, uid=1000)
    assert [part for part in argv if part in ("--gpus", "--runtime")] == []


def test_gpu_off_forces_cpu_on_a_gpu_platform():
    argv = rd.render_launch(settings(gpu="off"), SPARK, uid=1000)
    assert "--gpus" not in argv


def test_gpu_auto_is_the_default_and_matches_the_table():
    assert rd.gpu_flags(SPARK, "auto") == list(rd.GPU_FLAGS["dgx-spark"])


# -- render_launch: binding, naming and the image digest ------------------


def test_port_is_published_on_loopback_only():
    argv = rd.render_launch(settings(), SPARK, uid=1000)
    assert argv[argv.index("-p") + 1].startswith("127.0.0.1:")


def test_published_port_names_the_container_port_too():
    argv = rd.render_launch(settings(), SPARK, uid=1000)
    expected = f"127.0.0.1:{rd.host_port(settings(), 1000)}:{rd.ENGINES['llama-server'].port}"
    assert argv[argv.index("-p") + 1] == expected


def test_no_argument_ever_binds_all_interfaces_to_the_host():
    argv = rd.render_launch(settings(), SPARK, uid=1000)
    assert [part for part in argv if part.startswith("0.0.0.0:")] == []


def test_image_is_named_by_digest():
    assert IMAGE in rd.render_launch(settings(), SPARK, uid=1000)


def test_a_latest_tag_is_refused():
    arg0 = settings(image="example.invalid/tier2:latest")
    with pytest.raises(RuntimeUnavailable, match="digest"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_ref_without_a_digest_is_refused():
    arg0 = settings(image="example.invalid/tier2")
    with pytest.raises(RuntimeUnavailable, match="digest"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_truncated_digest_is_refused():
    arg0 = settings(image="example.invalid/tier2@sha256:abcd")
    with pytest.raises(RuntimeUnavailable, match="digest"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_the_container_is_not_started_with_rm_so_docker_rm_has_work():
    assert "--rm" not in rd.render_launch(settings(), SPARK, uid=1000)


def test_the_container_is_detached():
    assert "-d" in rd.render_launch(settings(), SPARK, uid=1000)


# -- render_launch: engines ------------------------------------------------


def test_engine_table_matches_the_engines_config_accepts():
    assert sorted(rd.ENGINES) == sorted(_TIERS_LFM_ENGINES)


def test_unknown_engine_names_the_accepted_ones():
    arg0 = settings(engine="ollama")
    with pytest.raises(RuntimeUnavailable, match="llama-server, sglang, vllm"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_llama_server_mounts_the_model_directory_read_only():
    argv = rd.render_launch(settings(), SPARK, uid=1000)
    assert argv[argv.index("-v") + 1] == "/var/lib/nvsh/models:/models:ro"


def test_a_mounted_engine_refers_to_the_model_inside_the_container():
    argv = rd.render_launch(settings(), SPARK, uid=1000)
    assert "/models/lfm2.gguf" in argv


def test_llama_server_without_a_model_dir_is_refused():
    arg0 = settings(model_dir=None)
    with pytest.raises(RuntimeUnavailable, match="model_dir"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_an_engine_without_a_model_is_refused():
    arg0 = settings(engine="vllm", model=None)
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_vllm_needs_no_model_mount():
    assert "-v" not in rd.render_launch(settings(engine="vllm"), SPARK, uid=1000)


def test_vllm_passes_the_model_name_through():
    assert "lfm2.gguf" in rd.render_launch(settings(engine="vllm"), SPARK, uid=1000)


def test_sglang_renders_its_own_container_port():
    argv = rd.render_launch(settings(engine="sglang"), SPARK, uid=1000)
    assert str(rd.ENGINES["sglang"].port) in argv[argv.index("-p") + 1]


def test_switching_engine_changes_nothing_but_config():
    first = rd.render_launch(settings(engine="vllm"), SPARK, uid=1000)
    second = rd.render_launch(settings(engine="sglang"), SPARK, uid=1000)
    assert first != second


# -- per-user name and port -----------------------------------------------


def test_container_name_carries_the_uid():
    assert rd.container_name(1000) == "nvsh-tier2-1000"


def test_two_users_do_not_share_a_container_name():
    assert rd.container_name(1000) != rd.container_name(1001)


def test_two_users_do_not_share_a_port():
    assert rd.host_port({}, 1000) != rd.host_port({}, 1001)


def test_the_derived_port_sits_in_the_reserved_span():
    port = rd.host_port({}, 1000)
    assert rd.PORT_BASE <= port < rd.PORT_BASE + rd.PORT_SPAN


def test_an_explicit_port_overrides_the_derived_one():
    assert rd.host_port({"port": 9123}, 1000) == 9123


def test_uids_a_thousand_apart_do_not_share_a_port():
    """finding 4054701431: PORT_BASE + uid % PORT_SPAN with the old
    PORT_SPAN=1000 gave uid 1000 and uid 2000 the same port."""
    assert rd.host_port({}, 1000) != rd.host_port({}, 2000)


def test_the_launch_line_names_this_users_container():
    argv = rd.render_launch(settings(), SPARK, uid=4321)
    assert argv[argv.index("--name") + 1] == "nvsh-tier2-4321"


# -- provenance of every rendered argument (criterion 3) -------------------

#: Fixed words the template itself contributes -- docker's own flags, the
#: engine's flag names, and the in-container mount point.
TEMPLATE_TOKENS = frozenset(
    {
        "docker",
        "run",
        "-d",
        "--name",
        "-p",
        "-v",
        "--gpus",
        "all",
        "--runtime",
        "nvidia",
        "--model",
        "--model-path",
        "--host",
        "0.0.0.0",
        "--port",
        "--ctx-size",
        "--max-model-len",
        "--context-length",
        "/models",
    }
)


def test_render_launch_takes_no_request_text_and_no_model_output():
    assert list(inspect.signature(rd.render_launch).parameters) == ["settings", "platform", "uid"]


def test_every_rendered_argument_traces_to_settings_detection_or_template():
    sentinels = ("SENTINELIMAGE", "SENTINELMODEL", "SENTINELDIR", "4242", "19999", "7777")
    rendered = rd.render_launch(
        settings(
            image="SENTINELIMAGE@" + DIGEST,
            model="SENTINELMODEL",
            model_dir="/SENTINELDIR",
            ctx=4242,
            port=19999,
        ),
        SPARK,
        uid=7777,
    )
    stray = [
        part
        for part in rendered
        if part not in TEMPLATE_TOKENS
        and not any(mark in part for mark in sentinels)
        and part != str(rd.ENGINES["llama-server"].port)
    ]
    assert stray == []


# -- hostile settings: the model name --------------------------------------


def test_a_model_escaping_the_mount_is_refused():
    arg0 = settings(model="../../etc/passwd")
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_mounted_model_with_a_slash_is_refused():
    arg0 = settings(model="sub/lfm2.gguf")
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_mounted_model_with_a_backslash_is_refused():
    arg0 = settings(model="sub\\lfm2.gguf")
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_dot_model_is_refused():
    arg0 = settings(model=".")
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_model_starting_with_a_dash_is_refused():
    arg0 = settings(model="-v")
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_repo_id_is_accepted_by_an_unmounted_engine():
    argv = rd.render_launch(settings(engine="vllm", model="LiquidAI/LFM2.5-350M"), SPARK, uid=1000)
    assert "LiquidAI/LFM2.5-350M" in argv


def test_a_repo_id_with_two_slashes_is_refused():
    arg0 = settings(engine="vllm", model="a/b/c")
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_model_with_whitespace_is_refused():
    arg0 = settings(engine="vllm", model="lfm2 --privileged")
    with pytest.raises(RuntimeUnavailable, match="model"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_refused_model_says_so_on_one_line():
    arg0 = settings(model="../../etc/passwd")
    with pytest.raises(RuntimeUnavailable) as caught:
        rd.render_launch(arg0, SPARK, uid=1000)
    assert "\n" not in str(caught.value)


# -- hostile settings: the model directory ---------------------------------


def test_a_model_dir_smuggling_a_second_volume_is_refused():
    arg0 = settings(model_dir="rel/dir:/x")
    with pytest.raises(RuntimeUnavailable, match="model_dir"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_relative_model_dir_is_refused():
    arg0 = settings(model_dir="models")
    with pytest.raises(RuntimeUnavailable, match="model_dir"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_model_dir_with_a_comma_is_refused():
    arg0 = settings(model_dir="/m,rw")
    with pytest.raises(RuntimeUnavailable, match="model_dir"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_model_dir_starting_with_a_dash_is_refused():
    arg0 = settings(model_dir="-v")
    with pytest.raises(RuntimeUnavailable, match="model_dir"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_an_absolute_model_dir_is_accepted():
    argv = rd.render_launch(settings(model_dir="/srv/models"), SPARK, uid=1000)
    assert "/srv/models:/models:ro" in argv


def test_a_model_dir_is_checked_even_for_an_unmounted_engine():
    arg0 = settings(engine="vllm", model_dir="rel/dir:/x")
    with pytest.raises(RuntimeUnavailable, match="model_dir"):
        rd.render_launch(arg0, SPARK, uid=1000)


# -- hostile settings: the numbers -----------------------------------------


def test_a_non_integer_port_is_refused_not_ignored():
    arg0 = settings(port="80; rm")
    with pytest.raises(RuntimeUnavailable, match="port"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_privileged_port_is_refused():
    arg0 = settings(port=22)
    with pytest.raises(RuntimeUnavailable, match="port"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_port_above_the_range_is_refused():
    arg0 = settings(port=70000)
    with pytest.raises(RuntimeUnavailable, match="port"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_boolean_port_is_refused():
    arg0 = settings(port=True)
    with pytest.raises(RuntimeUnavailable, match="port"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_non_integer_ctx_is_refused_not_ignored():
    arg0 = settings(ctx="4096; rm")
    with pytest.raises(RuntimeUnavailable, match="ctx"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_tiny_ctx_is_refused():
    arg0 = settings(ctx=8)
    with pytest.raises(RuntimeUnavailable, match="ctx"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_huge_ctx_is_refused():
    arg0 = settings(ctx=99999999)
    with pytest.raises(RuntimeUnavailable, match="ctx"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_non_numeric_startup_timeout_is_refused():
    arg0 = settings(startup_timeout_seconds="soon")
    with pytest.raises(RuntimeUnavailable, match="startup_timeout_seconds"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_zero_startup_timeout_is_refused():
    arg0 = settings(startup_timeout_seconds=0)
    with pytest.raises(RuntimeUnavailable, match="startup_timeout_seconds"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_an_overlong_startup_timeout_is_refused():
    arg0 = settings(startup_timeout_seconds=99999)
    with pytest.raises(RuntimeUnavailable, match="startup_timeout_seconds"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_a_float_startup_timeout_is_accepted():
    assert rd.render_launch(settings(startup_timeout_seconds=2.5), SPARK, uid=1000)


def test_a_bad_startup_timeout_stops_ensure_before_docker_runs():
    subject = runtime(FakeDocker(), settings=settings(startup_timeout_seconds=-1))
    with pytest.raises(RuntimeUnavailable, match="startup_timeout_seconds"):
        subject.ensure()


def test_an_image_ref_starting_with_a_dash_is_refused():
    arg0 = settings(image="-v@" + DIGEST)
    with pytest.raises(RuntimeUnavailable, match="digest"):
        rd.render_launch(arg0, SPARK, uid=1000)


# -- hostile settings: no smuggled flag, whatever the key ------------------

#: Every ``-``-leading element the launch line is allowed to contain: docker's
#: own flags, the detection flags, and the engines' flag names.
ALLOWED_FLAGS = frozenset(
    {
        "-d",
        "--name",
        "-p",
        "-v",
        "--gpus",
        "--runtime",
        "--model",
        "--model-path",
        "--host",
        "--port",
        "--ctx-size",
        "--max-model-len",
        "--context-length",
    }
)

HOSTILE_SETTINGS = [
    ("engine", "--privileged"),
    ("image", "-v@" + DIGEST),
    ("model", "-v /:/host"),
    ("model_dir", "-v"),
    ("gpu", "--privileged"),
    ("port", "--privileged"),
    ("ctx", "--privileged"),
    ("startup_timeout_seconds", "--privileged"),
    ("mode", "--privileged"),
]


def _smuggled_flags(hostile: dict[str, object]) -> list[str]:
    """Flags a hostile config got into the argv; ``[]`` when it was refused."""
    try:
        argv = rd.render_launch(hostile, SPARK, uid=1000)
    except RuntimeUnavailable:
        return []
    return [part for part in argv if part.startswith("-") and part not in ALLOWED_FLAGS]


@pytest.mark.parametrize("key,value", HOSTILE_SETTINGS)
def test_no_setting_can_smuggle_a_docker_flag(key, value):
    assert _smuggled_flags(settings(**{key: value})) == []


# -- the grep-style test (criterion 3) ------------------------------------


def _argv_has_docker_run(tree: ast.AST) -> bool:
    """True when a list/tuple literal holds the adjacent items 'docker', 'run'."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        items = [item.value if isinstance(item, ast.Constant) else None for item in node.elts]
        if any(left == "docker" and right == "run" for left, right in zip(items, items[1:])):
            return True
    return False


def _launches_a_container(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "docker run" in text or _argv_has_docker_run(ast.parse(text))


def _launcher_files() -> list[Path]:
    return sorted(p for p in (REPO_ROOT / "nvsh").rglob("*.py") if _launches_a_container(p))


def test_the_tier2_launcher_is_the_only_docker_run_in_nvsh():
    assert _launcher_files() == [LAUNCHER]


# -- image_refs / leftover_note -------------------------------------------


def test_image_refs_reports_the_configured_image():
    assert rd.image_refs(settings(), SPARK) == [IMAGE]


def test_image_refs_is_empty_when_nothing_resolves():
    assert rd.image_refs({"engine": "llama-server"}, SPARK) == []


def test_image_refs_is_empty_in_attach_mode():
    """finding 4054701435: attach mode never started a container, so it
    never pulled an image either -- reporting one would send the operator
    to remove an image nvsh does not own."""
    assert rd.image_refs(settings(mode="attach"), SPARK) == []


def test_leftover_note_prints_a_removal_command():
    assert f"docker image rm {IMAGE}" in rd.leftover_note([IMAGE])


def test_leftover_note_is_silent_with_nothing_left():
    assert rd.leftover_note([]) == ""


# -- pinned images ---------------------------------------------------------


def test_a_pinned_image_is_used_when_config_names_none(monkeypatch):
    monkeypatch.setattr(rd, "machine_arch", lambda: "aarch64")
    monkeypatch.setattr(
        rd,
        "pinned_images",
        lambda: [{"engine": "llama-server", "arch": "aarch64", "ref": IMAGE, "size_bytes": 1}],
    )
    assert rd.resolve_image({"engine": "llama-server"}, "llama-server") == IMAGE


def test_a_pin_for_another_arch_is_not_used(monkeypatch):
    monkeypatch.setattr(rd, "machine_arch", lambda: "x86_64")
    monkeypatch.setattr(
        rd,
        "pinned_images",
        lambda: [{"engine": "llama-server", "arch": "aarch64", "ref": IMAGE, "size_bytes": 1}],
    )
    with pytest.raises(RuntimeUnavailable, match="no pinned image"):
        rd.resolve_image({}, "llama-server")


def test_no_pinned_image_says_which_setting_to_write(monkeypatch):
    monkeypatch.setattr(rd, "pinned_images", list)
    with pytest.raises(RuntimeUnavailable, match=r"\[tiers.lfm\] image"):
        rd.resolve_image({}, "vllm")


def test_the_shipped_pin_table_has_an_images_list():
    assert isinstance(rd.pinned_images(), list)


# -- ensure(): Docker missing or unreachable ------------------------------


def test_docker_not_on_path_declines_with_one_line():
    def missing(argv, timeout):
        raise FileNotFoundError(2, "No such file or directory", "docker")

    subject = runtime(missing)
    with pytest.raises(RuntimeUnavailable) as caught:
        subject.ensure()
    assert "\n" not in str(caught.value)


def test_docker_not_on_path_says_docker_is_missing():
    def missing(argv, timeout):
        raise FileNotFoundError(2, "No such file or directory", "docker")

    subject = runtime(missing)
    with pytest.raises(RuntimeUnavailable, match="Docker"):
        subject.ensure()


def test_a_shell_reporting_127_is_treated_as_docker_missing():
    docker = FakeDocker([(("docker", "info"), (127, "command not found"))])
    subject = runtime(docker)
    with pytest.raises(RuntimeUnavailable, match="not on PATH"):
        subject.ensure()


def test_an_unreachable_daemon_declines_with_one_line():
    docker = FakeDocker([(("docker", "info"), (1, "Cannot connect to the Docker daemon"))])
    subject = runtime(docker)
    with pytest.raises(RuntimeUnavailable) as caught:
        subject.ensure()
    assert "\n" not in str(caught.value)


def test_an_unreachable_daemon_starts_nothing():
    docker = FakeDocker([(("docker", "info"), (1, "Cannot connect to the Docker daemon"))])
    unavailable = pytest.raises(RuntimeUnavailable)
    with unavailable:
        runtime(docker).ensure()
    assert docker.verbs == ["info"]


# -- ensure(): the memory floor -------------------------------------------


def test_below_the_memory_floor_the_tier_declines():
    low = FloorResult(ok=False, available_mb=100, status="local tier skipped: 100 MB available")
    subject = runtime(ExplodingDocker(), floor_check=lambda: low)
    with pytest.raises(RuntimeUnavailable, match="100 MB"):
        subject.ensure()


def test_above_the_memory_floor_the_tier_starts():
    fine = FloorResult(ok=True, available_mb=9000, status="")
    assert runtime(FakeDocker(), floor_check=lambda: fine).ensure().startswith("http://127.0.0.1:")


# -- ensure(): the happy path and idempotence -----------------------------


def test_ensure_returns_a_localhost_base_url():
    assert runtime(FakeDocker()).ensure() == f"http://127.0.0.1:{rd.host_port({}, 1000)}/v1"


def test_ensure_starts_the_container_when_none_exists():
    docker = FakeDocker([(("docker", "inspect"), (1, "No such object"))])
    runtime(docker).ensure()
    assert "run" in docker.verbs


def test_a_running_healthy_container_of_our_name_is_reused():
    docker = FakeDocker([(("docker", "inspect"), (0, "true"))])
    runtime(docker).ensure()
    assert "run" not in docker.verbs


def test_a_stale_stopped_container_is_removed_before_relaunch():
    docker = FakeDocker([(("docker", "inspect"), (0, "false"))])
    runtime(docker).ensure()
    assert docker.verbs.index("rm") < docker.verbs.index("run")


def test_ensure_is_idempotent_across_calls():
    docker = FakeDocker([(("docker", "inspect"), (0, "true"))])
    managed = runtime(docker)
    first = managed.ensure()
    assert managed.ensure() == first


def test_a_failed_docker_run_declines_with_the_output_tail():
    docker = FakeDocker(
        [
            (("docker", "inspect"), (1, "")),
            (("docker", "run"), (125, "unknown flag: --gpus")),
        ]
    )
    subject = runtime(docker)
    with pytest.raises(RuntimeUnavailable, match="unknown flag"):
        subject.ensure()


def test_a_port_conflict_tells_the_operator_to_set_the_port():
    """finding 4054701431: when docker run fails because the host port is
    already taken, the decline line should point at [tiers.lfm] port."""
    docker = FakeDocker(
        [
            (("docker", "inspect"), (1, "")),
            (
                ("docker", "run"),
                (125, "Bind for 127.0.0.1:18400 failed: port is already allocated"),
            ),
        ]
    )
    subject = runtime(docker)
    with pytest.raises(RuntimeUnavailable, match=r"\[tiers\.lfm\] port"):
        subject.ensure()


# -- ensure(): readiness ---------------------------------------------------


def test_a_slow_start_is_waited_out_without_real_sleeping():
    answers = iter([False, False, True])
    assert runtime(FakeDocker(), probe=lambda url, timeout: next(answers)).ensure()


def test_a_runtime_that_never_answers_declines():
    subject = runtime(FakeDocker(), probe=lambda url, timeout: False)
    with pytest.raises(RuntimeUnavailable, match="did not answer"):
        subject.ensure()


def test_a_runtime_that_never_answers_is_stopped():
    docker = FakeDocker()
    unavailable = pytest.raises(RuntimeUnavailable)
    with unavailable:
        runtime(docker, probe=lambda url, timeout: False).ensure()
    assert "stop" in docker.verbs


def test_the_timeout_detail_carries_the_container_log_tail():
    docker = FakeDocker([(("docker", "logs"), (0, "CUDA error: out of memory"))])
    subject = runtime(docker, probe=lambda url, timeout: False)
    with pytest.raises(RuntimeUnavailable, match="out of memory"):
        subject.ensure()


def test_the_log_tail_is_bounded():
    docker = FakeDocker([(("docker", "logs"), (0, "x" * 5000))])
    subject = runtime(docker, probe=lambda url, timeout: False)
    with pytest.raises(RuntimeUnavailable) as caught:
        subject.ensure()
    assert str(caught.value).count("x") <= rd.LOG_TAIL_CHARS


def test_the_log_tail_is_redacted():
    leak = "Authorization: Bearer " + "".join(("s", "k", "-", "x" * 20))
    docker = FakeDocker([(("docker", "logs"), (0, leak))])
    subject = runtime(docker, probe=lambda url, timeout: False)
    with pytest.raises(RuntimeUnavailable) as caught:
        subject.ensure()
    assert "x" * 20 not in str(caught.value)


def test_the_startup_timeout_is_configurable():
    clock = FakeClock()
    managed = runtime(
        FakeDocker(),
        probe=lambda url, timeout: False,
        clock=clock,
        settings=settings(startup_timeout_seconds=5),
    )
    unavailable = pytest.raises(RuntimeUnavailable)
    with unavailable:
        managed.ensure()
    assert clock.now - 1000.0 <= 5 + rd.POLL_SECONDS


# -- stop() and status() ---------------------------------------------------


def test_stop_stops_and_removes_our_container():
    docker = FakeDocker()
    runtime(docker).stop()
    assert docker.verbs == ["stop", "rm"]


def test_stop_names_only_our_own_container():
    docker = FakeDocker()
    runtime(docker).stop()
    assert {call[-1] for call in docker.calls} == {"nvsh-tier2-1000"}


def test_stop_never_raises():
    def broken(argv, timeout):
        raise OSError("docker went away")

    assert runtime(broken).stop() is None


def test_status_starts_nothing():
    assert runtime(ExplodingDocker()).status()


# -- stop_container() (``nvsh uninstall``'s safety net) --------------------


def test_stop_container_stops_and_removes_by_name():
    docker = FakeDocker()
    status = rd.stop_container(1000, docker)
    assert docker.verbs == ["stop", "rm"]
    assert {call[-1] for call in docker.calls} == {"nvsh-tier2-1000"}
    assert "nvsh-tier2-1000" in status


def test_stop_container_never_raises_when_docker_is_missing():
    def missing(argv, timeout):
        raise OSError("docker: command not found")

    status = rd.stop_container(1000, missing)
    assert "docker not available" in status


def test_stop_container_reports_no_such_container():
    docker = FakeDocker(
        [
            (("docker", "stop"), (1, "Error: No such container")),
            (("docker", "rm"), (1, "Error: No such container")),
        ]
    )
    status = rd.stop_container(1000, docker)
    assert "no container named nvsh-tier2-1000" in status


def test_stop_container_touches_only_stop_and_rm():
    docker = FakeDocker()
    rd.stop_container(1000, docker)
    assert docker.verbs == ["stop", "rm"]
    assert "run" not in docker.verbs
    assert "rmi" not in docker.verbs


def test_stop_container_still_calls_rm_when_stop_raises():
    """finding 4054701423: docker stop and docker rm used to share one try,
    so a stop that raises (a client-side timeout) skipped rm entirely."""
    rm_calls: list[list[str]] = []

    def flaky_stop(argv: list[str], timeout: float) -> tuple[int, str]:
        if argv[1] == "stop":
            raise TimeoutError("client-side timeout")
        rm_calls.append(list(argv))
        return (0, "")

    rd.stop_container(1000, flaky_stop)

    assert rm_calls == [["docker", "rm", "nvsh-tier2-1000"]]


def test_status_names_the_container_and_url():
    assert "nvsh-tier2-1000" in runtime(ExplodingDocker()).status()


# -- build_runtime() and attach mode --------------------------------------


def test_attach_mode_returns_an_attached_runtime():
    built = rd.build_runtime(
        settings(mode="attach", base_url="http://127.0.0.1:8080/v1"),
        SPARK,
        runner=ExplodingDocker(),
    )
    assert isinstance(built, AttachedRuntime)


def test_attach_mode_runs_no_docker_commands():
    built = rd.build_runtime(
        settings(mode="attach", base_url="http://127.0.0.1:8080/v1"),
        SPARK,
        runner=ExplodingDocker(),
    )
    assert built.ensure() == "http://127.0.0.1:8080/v1"


def test_attach_mode_needs_a_base_url():
    arg0 = settings(mode="attach")
    runner = ExplodingDocker()
    with pytest.raises(RuntimeUnavailable, match="base_url"):
        rd.build_runtime(arg0, SPARK, runner=runner)


def test_attach_mode_refuses_a_remote_base_url():
    built = rd.build_runtime(
        settings(mode="attach", base_url="http://example.invalid/v1"),
        SPARK,
        runner=ExplodingDocker(),
    )
    with pytest.raises(RuntimeUnavailable, match="127.0.0.1"):
        built.ensure()


def test_managed_mode_returns_the_docker_runtime():
    built = rd.build_runtime(settings(), SPARK, runner=FakeDocker())
    assert isinstance(built, rd.DockerRuntime)


def test_an_unknown_mode_is_refused():
    arg0 = settings(mode="podman")
    runner = ExplodingDocker()
    with pytest.raises(RuntimeUnavailable, match="attach, managed"):
        rd.build_runtime(arg0, SPARK, runner=runner)


def test_constructing_the_runtime_starts_nothing():
    assert rd.DockerRuntime(settings(), SPARK, runner=ExplodingDocker()) is not None


# -- docs/platforms.md -----------------------------------------------------


def _platforms_doc() -> str:
    return (REPO_ROOT / "docs" / "platforms.md").read_text(encoding="utf-8")


def test_platforms_doc_has_the_docker_gpu_section():
    assert "## Docker GPU path" in _platforms_doc()


def test_platforms_doc_records_the_spark_launch_form():
    assert "`--gpus all`" in _platforms_doc()


def test_platforms_doc_records_the_jetson_launch_form():
    assert "`--runtime nvidia`" in _platforms_doc()


# -- up-front GPU memory share (vLLM, SGLang) ------------------------------


def _after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_vllm_reserves_a_small_gpu_share_by_default():
    argv = rd.render_launch(settings(engine="vllm"), SPARK, uid=1000)
    assert _after(argv, "--gpu-memory-utilization") == str(rd.DEFAULT_GPU_FRACTION)


def test_sglang_takes_the_configured_gpu_share():
    argv = rd.render_launch(settings(engine="sglang", gpu_memory_fraction=0.2), SPARK, uid=1000)
    assert _after(argv, "--mem-fraction-static") == "0.2"


def test_llama_server_has_no_gpu_share_flag():
    argv = rd.render_launch(settings(), SPARK, uid=1000)
    assert "--gpu-memory-utilization" not in argv


@pytest.mark.parametrize("value", [0, 1, 0.99, -0.5, "0.5", True, None.__class__])
def test_a_bad_gpu_share_is_refused(value):
    with pytest.raises(rd.RuntimeUnavailable, match="gpu_memory_fraction"):
        rd.check_gpu_memory_fraction(value)


# -- server-side tool-call parsing -----------------------------------------


def test_vllm_turns_on_tool_call_parsing_with_its_default_parser():
    argv = rd.render_launch(settings(engine="vllm"), SPARK, uid=1000)
    assert argv[argv.index("--enable-auto-tool-choice") + 1 :][:2] == ["--tool-call-parser", "lfm2"]


def test_the_tool_call_parser_is_a_config_choice():
    argv = rd.render_launch(settings(engine="vllm", tool_call_parser="hermes"), SPARK, uid=1000)
    assert _after(argv, "--tool-call-parser") == "hermes"


def test_sglang_names_no_parser_unless_configured():
    argv = rd.render_launch(settings(engine="sglang"), SPARK, uid=1000)
    assert "--tool-call-parser" not in argv


def test_llama_server_takes_no_parser_flag_even_when_one_is_configured():
    argv = rd.render_launch(settings(tool_call_parser="lfm2"), SPARK, uid=1000)
    assert "--tool-call-parser" not in argv


@pytest.mark.parametrize("value", ["--privileged", "a b", "LFM2", "", 7, "x" * 41])
def test_a_bad_tool_call_parser_is_refused(value):
    with pytest.raises(rd.RuntimeUnavailable, match="tool_call_parser"):
        rd.check_tool_call_parser(value)


# -- the download cache of an engine that fetches its model by id ----------


def test_a_downloading_engine_mounts_the_host_cache(tmp_path):
    argv = rd.render_launch(settings(engine="vllm", hf_cache_dir="/var/cache/x"), SPARK, uid=1000)
    assert f"/var/cache/x:{rd.CACHE_MOUNT}" in argv


def test_a_downloading_engine_with_a_cache_runs_as_the_operator():
    argv = rd.render_launch(settings(engine="vllm", hf_cache_dir="/var/cache/x"), SPARK, uid=1234)
    assert _after(argv, "--user") == "1234:1234"


def test_llama_server_ignores_the_download_cache():
    argv = rd.render_launch(settings(hf_cache_dir="/var/cache/x"), SPARK, uid=1000)
    assert "--user" not in argv


def test_a_cache_dir_that_could_smuggle_a_volume_is_refused():
    arg0 = settings(engine="vllm", hf_cache_dir="/a:/b")
    with pytest.raises(rd.RuntimeUnavailable, match="hf_cache_dir"):
        rd.render_launch(arg0, SPARK, uid=1000)


def test_build_runtime_defaults_the_cache_under_nvsh_own_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rd.build_runtime(settings(engine="vllm"), SPARK)
    assert (tmp_path / "nvsh" / "tiers" / rd.HF_CACHE_NAME).is_dir()
