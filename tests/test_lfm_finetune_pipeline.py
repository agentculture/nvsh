"""scripts/lfm-finetune/pipeline.sh (issue 46, task t14): stage wiring for the
Apache-2.0 Qwen3.5-0.8B fine-tune, reusing issue 39's pipeline.

Every test here is a dry run: no augmentation gateway, no training stack, no
Hugging Face Hub call and no ``FINAL=1`` ever runs, exactly like the rest of
the lfm-finetune test suite (``pytest.importorskip`` is not needed -- this
module only shells out to ``pipeline.sh`` and ``shellcheck``, both of which
fail closed with a plain skip when missing).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from nvsh.config import load as load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _REPO_ROOT / "scripts" / "lfm-finetune" / "pipeline.sh"
_LFM_ENV = _REPO_ROOT / "scripts" / "lfm-finetune" / "pipeline.env.example"
_QWEN_ENV = _REPO_ROOT / "scripts" / "lfm-finetune" / "pipeline-qwen.env.example"
_SERVE = _REPO_ROOT / "scripts" / "lfm-finetune" / "serve_for_measure.sh"

#: The stages task t14 adds, on top of the ones issue 39 already wired up.
_NEW_STAGES = (
    "rereview",
    "filter-variations",
    "train-scorer",
    "scan",
    "quantize",
    "heal",
    "upload",
)

#: The stage task f9 adds (deviation d3: every served model needs a
#: generation_config.json pinning greedy decoding).
_GEN_CONFIG_STAGES = ("stock-copy",)


def _run(env_file: Path, tmp_path: Path, *stage_args: str) -> subprocess.CompletedProcess:
    """Invoke pipeline.sh from an isolated cwd, so $PWD in the env file never
    touches the real checkout (WORK/HF_CACHE/etc. land under *tmp_path*)."""
    return subprocess.run(
        ["bash", str(_PIPELINE), "--env", str(env_file), *stage_args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_shellcheck_is_clean_on_pipeline_sh() -> None:
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    capped = _PIPELINE.parent / "capped.sh"
    result = subprocess.run(
        ["shellcheck", "-x", str(_PIPELINE), str(capped), str(_SERVE)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_an_unknown_stage_dry_run_lists_every_stage(env_file: Path, tmp_path) -> None:
    result = _run(env_file, tmp_path, "__not_a_real_stage__")
    assert result.returncode == 1
    for stage in (
        "split",
        "skills",
        "augment-nvsh",
        "augment-skills",
        "assemble",
        "train",
        "measure-val",
        "measure-final",
        "measure-skills",
        "status",
        *_NEW_STAGES,
        *_GEN_CONFIG_STAGES,
    ):
        assert stage in result.stderr, f"{stage!r} missing from: {result.stderr!r}"


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_status_is_a_real_no_op_dry_run(env_file: Path, tmp_path) -> None:
    """`status` never calls the gateway, a training stack or the Hub -- a
    genuine dry run that just reports what exists (nothing, on a fresh WORK)."""
    result = _run(env_file, tmp_path, "status")
    assert result.returncode == 0, result.stderr
    assert "splits/train.json" in result.stdout


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_upload_refuses_without_final_set(env_file: Path, tmp_path) -> None:
    result = _run(env_file, tmp_path, "upload", "somerun")
    assert result.returncode == 1
    assert "FINAL=1" in result.stderr


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_upload_refuses_a_missing_bundle_even_with_final_set(env_file: Path, tmp_path) -> None:
    env = {"FINAL": "1"}
    import os

    result = subprocess.run(
        ["bash", str(_PIPELINE), "--env", str(env_file), "upload", "somerun"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **env},
    )
    assert result.returncode == 1
    assert "merged" in result.stderr


def _load_scan_bundle():
    spec_path = _REPO_ROOT / "scripts" / "lfm-finetune" / "scan_bundle.py"
    spec = importlib.util.spec_from_file_location("scan_bundle", spec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_upload_refuses_a_bundle_without_a_valid_generation_config(
    env_file: Path, tmp_path
) -> None:
    """Deviation d3: a merged checkpoint scan_bundle.py would happily verify
    still can't upload without a generation_config.json pinning greedy decoding."""
    text = env_file.read_text(encoding="utf-8")
    work_rel = next(
        line.split("=", 1)[1].replace("$PWD/", "")
        for line in text.splitlines()
        if line.startswith("WORK=")
    )
    bundle = tmp_path / work_rel / "runs" / "somerun" / "merged"
    bundle.mkdir(parents=True)
    (bundle / "weights.safetensors").write_text("x", encoding="utf-8")

    # Make scan_bundle.py verify pass on its own, so the refusal below isolates
    # to gen_config.py's check rather than scan_bundle's.
    scan_bundle = _load_scan_bundle()
    scan_bundle.write_scan(bundle, scan_bundle._get_scan_secrets())  # noqa: SLF001

    result = subprocess.run(
        ["bash", str(_PIPELINE), "--env", str(env_file), "upload", "somerun"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "FINAL": "1"},
    )
    assert result.returncode == 1
    assert "generation_config.json" in result.stderr
    assert not (bundle / "generation_config.json").exists()


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_heal_refuses_without_a_base_run(env_file: Path, tmp_path) -> None:
    # heal checks its training data before its base-run checkpoint; on a
    # fresh WORK neither exists, but either refusal proves it never trains
    # blind -- this never reaches run_capped/TRAIN_PY.
    result = _run(env_file, tmp_path, "heal", "healed", "no-such-run")
    assert result.returncode == 1
    assert "run assemble first" in result.stderr or "no-such-run" in result.stderr


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_heal_refuses_a_missing_base_run_once_data_exists(env_file: Path, tmp_path) -> None:
    text = env_file.read_text(encoding="utf-8")
    work_rel = next(
        line.split("=", 1)[1].replace("$PWD/", "")
        for line in text.splitlines()
        if line.startswith("WORK=")
    )
    data_dir = tmp_path / work_rel / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "nvsh-train.jsonl").write_text('{"messages": []}\n', encoding="utf-8")

    result = _run(env_file, tmp_path, "heal", "healed", "no-such-run")
    assert result.returncode == 1
    assert "no-such-run" in result.stderr
    assert "merged" in result.stderr


@pytest.mark.parametrize("env_file", [_LFM_ENV, _QWEN_ENV], ids=["lfm", "qwen"])
def test_quantize_refuses_without_a_trained_run(env_file: Path, tmp_path) -> None:
    result = _run(env_file, tmp_path, "quantize", "no-such-run")
    assert result.returncode == 1
    assert "merged" in result.stderr


def test_pipeline_env_example_keeps_the_lfm_defaults() -> None:
    text = _LFM_ENV.read_text(encoding="utf-8")
    assert "BASE=LiquidAI/LFM2.5-350M" in text
    assert "BASE_REV=9e6c6ccf47cd318696e137d381a7ded8fe4df09f" in text
    assert "REPO=jetson-ai-lab/lfm2.5-350m-nvsh-triage" in text


def test_pipeline_qwen_env_example_pins_the_qwen_base_and_repo() -> None:
    text = _QWEN_ENV.read_text(encoding="utf-8")
    assert "BASE=Qwen/Qwen3.5-0.8B" in text
    assert "BASE_REV=2fc06364715b967f1860aea9cf38778875588b17" in text
    for line in text.splitlines():
        if line.startswith("REPO="):
            assert line.startswith("REPO=jetson-ai-lab/qwen3.5-0.8b-nvsh-")
            break
    else:  # pragma: no cover - defensive
        pytest.fail("no REPO= line in pipeline-qwen.env.example")


def test_neither_env_example_names_a_host_or_a_secret() -> None:
    for env_file in (_LFM_ENV, _QWEN_ENV):
        text = env_file.read_text(encoding="utf-8")
        assert "http://localhost" in text or "AUG_URL=" in text  # a placeholder gateway URL only
        for line in text.splitlines():
            if line.strip().startswith("#") or not line.strip():
                continue
            key = line.split("=", 1)[0]
            assert key == "HF_TOKEN_ENV" or not key.endswith("_TOKEN"), (
                f"{env_file.name}: {line!r} looks like it names a token directly, "
                "not the variable that holds one"
            )


_CAPPED = _REPO_ROOT / "scripts" / "lfm-finetune" / "capped.sh"
_HOG = "b = bytearray(600 * 1024 * 1024)\nfor i in range(0, len(b), 4096):\n    b[i] = 1\n"


def _user_scope_works() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "-q", "true"], capture_output=True, timeout=30
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _user_scope_works(), reason="needs a systemd user session")
def test_run_capped_kills_a_process_over_the_cap_even_with_swap(tmp_path: Path) -> None:
    """The cap is hard: RAM plus swap (c49). MemoryMax alone let 600M spill to swap."""
    hog = tmp_path / "hog.py"
    hog.write_text(_HOG, encoding="utf-8")
    script = f'source "{_CAPPED}"; TRAIN_MEMORY_MAX=200M run_capped "{tmp_path}" python3 "{hog}"'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert (tmp_path / "mem.log").read_text(encoding="utf-8").strip()


@pytest.mark.skipif(not _user_scope_works(), reason="needs a systemd user session")
def test_run_capped_lets_a_process_under_the_cap_finish(tmp_path: Path) -> None:
    script = f'source "{_CAPPED}"; TRAIN_MEMORY_MAX=200M run_capped "{tmp_path}" true'
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0


def test_run_capped_refuses_to_run_uncapped_without_systemd_run(tmp_path: Path) -> None:
    """No systemd-run and no declared container cap: refuse, never train uncapped."""
    script = (
        f'source "{_CAPPED}";'
        ' command() { [ "$2" = systemd-run ] && return 1; builtin command "$@"; };'
        f' TRAIN_MEMORY_MAX=200M run_capped "{tmp_path}" true'
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "TRAIN_MEMORY_CAP=container" in result.stderr


def _mem_available_kb() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise AssertionError("no MemAvailable in /proc/meminfo")


def _run_watched(tmp_path: Path, floor: str, command: str, *, prefix: str = "") -> tuple:
    script = (
        f'source "{_CAPPED}";{prefix} TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR={floor}'
        f' TRAIN_WATCHDOG_SECONDS=1 run_capped "{tmp_path}" {command}'
    )
    started = time.monotonic()
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    return result, time.monotonic() - started


_HIDE_SYSTEMD_RUN = ' command() { [ "$2" = systemd-run ] && return 1; builtin command "$@"; };'


@pytest.mark.skipif(not _user_scope_works(), reason="needs a systemd user session")
def test_the_watchdog_stops_a_run_when_available_memory_falls_below_the_floor(
    tmp_path: Path,
) -> None:
    """GPU allocations escape the cgroup cap on unified memory (c49/h33): a floor
    on MemAvailable is the backstop. A floor above what is free trips at once."""
    floor = f"{_mem_available_kb() * 2}K"
    result, seconds = _run_watched(tmp_path, floor, "sleep 60")
    assert result.returncode == 3, result.stderr
    assert seconds < 20
    mem_log = (tmp_path / "mem.log").read_text(encoding="utf-8")
    assert "watchdog: MemAvailable" in mem_log
    assert f"below floor {floor}, stopping" in mem_log
    assert "watchdog: MemAvailable" in result.stderr


@pytest.mark.skipif(not _user_scope_works(), reason="needs a systemd user session")
def test_the_watchdog_stops_the_commands_children_too(tmp_path: Path) -> None:
    """A child holding the output pipe open dies with the command, so the run ends."""
    floor = f"{_mem_available_kb() * 2}K"
    result, seconds = _run_watched(tmp_path, floor, "bash -c 'sleep 60 & wait'")
    assert result.returncode == 3, result.stderr
    assert seconds < 20


@pytest.mark.skipif(not _user_scope_works(), reason="needs a systemd user session")
def test_a_run_above_the_floor_finishes_normally(tmp_path: Path) -> None:
    result, _ = _run_watched(tmp_path, "1K", "true")
    assert result.returncode == 0, result.stderr
    assert "watchdog" not in (tmp_path / "mem.log").read_text(encoding="utf-8")


def test_the_watchdog_also_guards_a_container_capped_run(tmp_path: Path) -> None:
    """With TRAIN_MEMORY_CAP=container there is no scope; the floor still applies."""
    floor = f"{_mem_available_kb() * 2}K"
    prefix = _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container"
    result, seconds = _run_watched(tmp_path, floor, "sleep 60", prefix=prefix)
    assert result.returncode == 3, result.stderr
    assert seconds < 20
    assert "watchdog: MemAvailable" in (tmp_path / "mem.log").read_text(encoding="utf-8")


def test_a_container_capped_run_above_the_floor_finishes_normally(tmp_path: Path) -> None:
    prefix = _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container"
    result, _ = _run_watched(tmp_path, "1K", "true", prefix=prefix)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("floor", ["lots", "8X", "-1G", "0", "1.5G"])
def test_an_unreadable_memory_floor_is_refused(tmp_path: Path, floor: str) -> None:
    prefix = _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container"
    result, _ = _run_watched(tmp_path, floor, "true", prefix=prefix)
    assert result.returncode == 2
    assert "TRAIN_MEMORY_FLOOR" in result.stderr


def test_the_memory_settings_in_the_env_file_reach_a_child_process(tmp_path: Path) -> None:
    """Caps set only in the env file must reach train.py/train_scorer.py (child processes)."""
    env = tmp_path / "caps.env"
    env.write_text(
        _QWEN_ENV.read_text(encoding="utf-8")
        + "\nTRAIN_MEMORY_FLOOR=9G\nTRAIN_WATCHDOG_SECONDS=4\nNVSH_TRAIN_GPU_MEMORY_GB=18\n",
        encoding="utf-8",
    )
    result = _run(env, tmp_path, "status")
    assert result.returncode == 0, result.stderr
    assert "caps (as a child sees them): max=24G floor=9G watchdog=4s gpu_gb=18" in result.stdout


def test_both_env_examples_name_the_memory_floor_and_gpu_budget() -> None:
    for example in (_LFM_ENV, _QWEN_ENV):
        text = example.read_text(encoding="utf-8")
        assert "TRAIN_MEMORY_FLOOR=" in text and "NVSH_TRAIN_GPU_MEMORY_GB=" in text


# ---------------------------------------------------------------------------
# Review fixes (issue 46, Codex findings #1, #2 and #5)
# ---------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    """*pid* exists and is not a zombie waiting to be reaped."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _start_capped_sleep(tmp_path: Path, prefix: str) -> tuple[subprocess.Popen, int]:
    """run_capped with a 60 s sleep in the background; returns (shell, sleep pid)."""
    pid_file = tmp_path / "sleep.pid"
    script = (
        f'source "{_CAPPED}";{prefix} TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR=1K'
        f' TRAIN_WATCHDOG_SECONDS=1 run_capped "{tmp_path}/run"'
        f" bash -c 'echo $$ > \"{pid_file}\"; exec sleep 60'"
    )
    shell = subprocess.Popen(
        ["bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        text = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else ""
        if text:
            return shell, int(text)
        time.sleep(0.1)
    shell.kill()
    raise AssertionError("the capped command never started")


def _assert_sigterm_stops_the_command(tmp_path: Path, prefix: str) -> None:
    shell, sleep_pid = _start_capped_sleep(tmp_path, prefix)
    try:
        time.sleep(0.5)  # let `wait` start, so the signal lands mid-run
        shell.terminate()
        deadline = time.monotonic() + 15
        while _alive(sleep_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _alive(sleep_pid), "the capped command survived SIGTERM to run_capped"
        assert shell.wait(timeout=15) != 0
    finally:
        if _alive(sleep_pid):
            os.kill(sleep_pid, 9)
        if shell.poll() is None:
            shell.kill()


def test_sigterm_to_run_capped_stops_a_container_capped_command(tmp_path: Path) -> None:
    """Finding #1: the command runs in its own session, so stopping the caller
    must stop it explicitly -- training never outlives a stopped pipeline."""
    _assert_sigterm_stops_the_command(tmp_path, _HIDE_SYSTEMD_RUN + " TRAIN_MEMORY_CAP=container")


@pytest.mark.skipif(not _user_scope_works(), reason="needs a systemd user session")
def test_sigterm_to_run_capped_stops_a_scope_capped_command(tmp_path: Path) -> None:
    _assert_sigterm_stops_the_command(tmp_path, "")


def test_run_capped_restores_the_callers_traps(tmp_path: Path) -> None:
    script = (
        f'source "{_CAPPED}";{_HIDE_SYSTEMD_RUN} TRAIN_MEMORY_CAP=container;'
        " trap 'echo caller-term' TERM; trap 'echo caller-exit' EXIT;"
        f' TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR=1K run_capped "{tmp_path}" true;'
        " trap -p TERM INT HUP"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "trap -- 'echo caller-term' SIGTERM" in result.stdout
    assert "SIGINT" not in result.stdout and "SIGHUP" not in result.stdout
    assert result.stdout.rstrip().endswith("caller-exit")


def test_run_capped_still_returns_the_commands_status(tmp_path: Path) -> None:
    script = (
        f'source "{_CAPPED}";{_HIDE_SYSTEMD_RUN} TRAIN_MEMORY_CAP=container;'
        f' TRAIN_MEMORY_MAX=200M TRAIN_MEMORY_FLOOR=1K run_capped "{tmp_path}"'
        " bash -c 'echo out; exit 7'"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 7, result.stderr
    assert "out" in result.stdout
    assert (tmp_path / "train.log").read_text(encoding="utf-8").strip() == "out"


_FAKE_UV = """#!/usr/bin/env bash
# Records every `uv run --frozen python ...` call; runs gen_config.py for real.
printf '%s\\t%s\\n' "PYTHONPATH=${PYTHONPATH:-}" "$*" >> "$UV_LOG"
case "$*" in
  *gen_config.py*) shift 3; exec "$REAL_PY" "$@" ;;
  *measure.py*|*measure_skills.py*)
    printf 'uv %s\\n' "${4##*/}" >> "$EVENT_LOG"
    exit "${FAKE_MEASURE_STATUS:-0}" ;;
esac
exit 0
"""

#: Records each docker argv as one JSON line, and the subcommand in EVENT_LOG;
#: `logs` prints a recognisable line, everything else prints nothing.
_FAKE_DOCKER = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["DOCKER_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
with open(os.environ["EVENT_LOG"], "a", encoding="utf-8") as log:
    log.write("docker " + (sys.argv[1] if len(sys.argv) > 1 else "") + "\\n")
if sys.argv[1:2] == ["logs"]:
    print("fake-vllm: the last log line")
"""

#: `curl` as the readiness probe sees it: ready unless FAKE_CURL_STATUS says not.
_FAKE_CURL = """#!/usr/bin/env bash
printf 'curl %s\\n' "${*: -1}" >> "$EVENT_LOG"
exit "${FAKE_CURL_STATUS:-0}"
"""

_IMAGE = "vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695"


def _fake_bin(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A bin dir with fake uv/docker/curl (made once), and the env that points
    them at logs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in (("uv", _FAKE_UV), ("docker", _FAKE_DOCKER), ("curl", _FAKE_CURL)):
        path = bin_dir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    env = {
        **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "UV_LOG": str(tmp_path / "uv.log"),
        "DOCKER_LOG": str(tmp_path / "docker.log"),
        "EVENT_LOG": str(tmp_path / "events.log"),
        "REAL_PY": sys.executable,
    }
    return bin_dir, env


def _docker_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "docker.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _events(tmp_path: Path) -> list[str]:
    log = tmp_path / "events.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def _greedy_dir(path: Path) -> Path:
    """A model dir that passes `gen_config.py check`."""
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    gen_config = _REPO_ROOT / "scripts" / "lfm-finetune" / "gen_config.py"
    subprocess.run(
        [sys.executable, str(gen_config), "write", str(path)], check=True, capture_output=True
    )
    return path


class _Pipeline:
    """pipeline.sh against the Qwen example env, with `uv`, `docker` and `curl`
    replaced by recorders."""

    def __init__(self, tmp_path: Path, extra_env: str = "") -> None:
        self.tmp = tmp_path
        self.work = tmp_path / "qwen-work"
        self.snapshot = tmp_path / "ground.json"
        _, self.env = _fake_bin(tmp_path)
        self.log = tmp_path / "uv.log"
        self.env_file = tmp_path / "test.env"
        self.env_file.write_text(
            _QWEN_ENV.read_text(encoding="utf-8")
            + f"\nGROUND_SNAPSHOT={self.snapshot}\n"
            + extra_env,
            encoding="utf-8",
        )

    def run(self, *stage_args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(_PIPELINE), "--env", str(self.env_file), *stage_args],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            timeout=60,
            env={**self.env, **env},
        )

    def calls(self, script: str) -> list[tuple[str, list[str]]]:
        """(PYTHONPATH, argv) of every recorded call to *script*."""
        if not self.log.exists():
            return []
        found = []
        for line in self.log.read_text(encoding="utf-8").splitlines():
            pythonpath, args = line.split("\t", 1)
            argv = args.split(" ")
            if any(arg.endswith(script) for arg in argv):
                found.append((pythonpath.removeprefix("PYTHONPATH="), argv))
        return found

    def ready(self, *, stock: bool = True, snapshot: bool = True, run: str = "a1") -> None:
        if snapshot:
            self.snapshot.write_text("{}", encoding="utf-8")
        if stock:
            _greedy_dir(self.work / "stock")
        if run:
            _greedy_dir(self.work / "runs" / run / "merged")
            (self.work / "runs" / run / "revision").write_text("abc123\n", encoding="utf-8")


def _option(argv: list[str], name: str) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == name]


_QWEN_BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"


def test_measure_final_measures_one_model_from_its_own_dir(tmp_path: Path) -> None:
    """Finding #2a / deviation d7: stock is served from the stock copy (greedy
    generation_config.json), and each final call measures exactly one model."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-final", "stock")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == ["stock"]
    assert _option(argv, "--revision") == [_QWEN_BASE_REV]
    assert "--final" in argv
    assert _option(argv, "--split") == [str(pipe.work / "splits" / "test.json")]
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "-v") == [f"{pipe.work / 'stock'}:/model:ro"]
    assert "Qwen/Qwen3.5-0.8B" not in argv


def test_measure_final_of_a_run_serves_its_merged_dir(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    result = pipe.run("measure-final", "a1")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == ["a1"]
    assert _option(argv, "--revision") == ["abc123"]
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "-v") == [f"{pipe.work / 'runs' / 'a1' / 'merged'}:/model:ro"]


@pytest.mark.parametrize("stage", ["measure-val", "measure-final", "measure-skills"])
def test_a_measure_stage_refuses_without_the_stock_copy(stage: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    result = pipe.run(stage, "stock")
    assert result.returncode == 1
    assert "stock-copy" in result.stderr
    assert not pipe.calls("measure.py") and not pipe.calls("measure_skills.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


@pytest.mark.parametrize("stage", ["measure-val", "measure-final", "measure-skills"])
@pytest.mark.parametrize("name", ["stock", "a1"])
def test_a_measure_stage_refuses_a_model_without_greedy_decoding(
    stage: str, name: str, tmp_path: Path
) -> None:
    """Deviation d3: every measured model samples at temperature 0 from its own file."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    model_dir = pipe.work / "stock" if name == "stock" else pipe.work / "runs" / name / "merged"
    (model_dir / "generation_config.json").unlink()
    result = pipe.run(stage, name, *(["--margin", "+15"] if stage == "measure-skills" else []))
    assert result.returncode != 0
    assert "generation_config.json" in result.stderr
    assert not pipe.calls("measure.py") and not pipe.calls("measure_skills.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


def test_measure_val_measures_stock_from_the_stock_copy(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-val", "stock")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == ["stock"]
    assert _option(argv, "--revision") == [_QWEN_BASE_REV]
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "-v") == [f"{pipe.work / 'stock'}:/model:ro"]


def test_measure_val_of_a_tuned_run_needs_no_stock_copy(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == ["a1"]
    assert _option(argv, "--revision") == ["abc123"]


def test_a_measure_stage_refuses_a_run_that_was_never_trained(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(run="")
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 1
    assert "merged" in result.stderr
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


@pytest.mark.parametrize(
    ("stage", "name"), [("measure-val", "a1"), ("measure-val", "stock"), ("measure-final", "a1")]
)
def test_measure_stages_pass_the_snapshot_thinking_and_extra_args(
    stage: str, name: str, tmp_path: Path
) -> None:
    """Finding #2b/#2c: fixed grounding, thinking off, and the operator's own flags."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run(stage, name, "--slice", "missing-candidate")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--ground-snapshot") == [str(pipe.snapshot)]
    assert _option(argv, "--enable-thinking") == ["false"]
    assert _option(argv, "--ctx") == ["2048"]  # from MEASURE_CTX, never an extra arg
    assert _option(argv, "--slice") == ["missing-candidate"]
    assert _option(argv, "--max-logprobs") == ["22"]


def test_enable_thinking_comes_from_the_env_file(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, "ENABLE_THINKING=true\n")
    pipe.ready()
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--enable-thinking") == ["true"]


@pytest.mark.parametrize("stage", ["measure-val", "measure-final"])
def test_a_measure_stage_refuses_a_missing_ground_snapshot(stage: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(snapshot=False)
    result = pipe.run(stage, "a1")
    assert result.returncode == 1
    assert "GROUND_SNAPSHOT" in result.stderr
    assert not pipe.calls("measure.py")
    assert not _docker_calls(tmp_path)


def test_a_measure_stage_refuses_an_unset_ground_snapshot(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, "GROUND_SNAPSHOT=\n")
    pipe.ready()
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 1
    assert "GROUND_SNAPSHOT" in result.stderr


@pytest.mark.parametrize("name", ["stock", "a1"])
def test_measure_skills_measures_one_served_model_with_thinking_off(
    name: str, tmp_path: Path
) -> None:
    """measure_skills.py has no --ground-snapshot (it grounds nothing) and cannot
    read an attach config: it gets the helper's served URL and model name with
    --url/--model, never --launch; the margin reaches only a tuned run."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-skills", name, "--margin", "+15")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure_skills.py")
    assert _option(argv, "--url") == ["http://127.0.0.1:18060/v1"]
    assert _option(argv, "--model") == [name]
    assert _option(argv, "--enable-thinking") == ["false"]
    assert "--launch" not in argv and "--config" not in argv
    assert "--ground-snapshot" not in argv
    assert _option(argv, "--label") == [name]
    if name == "stock":
        assert _option(argv, "--model-revision") == [_QWEN_BASE_REV]
        assert "--margin" not in argv and "--tuned" not in argv
    else:
        assert _option(argv, "--model-revision") == ["abc123"]
        assert _option(argv, "--margin") == ["+15"] and "--tuned" in argv


@pytest.mark.parametrize(
    ("stage", "script"),
    [
        ("measure-val", "measure.py"),
        ("measure-final", "measure.py"),
        ("measure-skills", "measure_skills.py"),
    ],
)
@pytest.mark.parametrize("name", ["stock", "a1"])
def test_a_measure_stage_starts_waits_runs_and_stops_the_helper(
    stage: str, script: str, name: str, tmp_path: Path
) -> None:
    """Deviation d7: one model per call, served by the committed helper and
    measured in attach mode, the container removed afterwards."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run(stage, name, *(["--margin", "+15"] if stage == "measure-skills" else []))
    assert result.returncode == 0, result.stderr
    events = [e for e in _events(tmp_path) if not e.startswith("docker ps")]
    run_at = events.index("docker run")
    ready_at = events.index("curl http://127.0.0.1:18060/v1/models")
    measure_at = events.index(f"uv {script}")
    stop_at = events.index("docker rm")
    assert run_at < ready_at < measure_at < stop_at
    [stop] = [c for c in _docker_calls(tmp_path) if c[0] == "rm"]
    assert stop == ["rm", "-f", "q46-measure-18060"]
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "--served-model-name") == [name]
    assert _option(run, "--tool-call-parser") == ["qwen3_coder"]
    records = list((pipe.work / "measure").glob("*.serve.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["argv"] == ["docker", *run]


@pytest.mark.parametrize("stage", ["measure-val", "measure-final"])
def test_a_measure_stage_writes_an_attach_config_for_the_served_model(
    stage: str, tmp_path: Path
) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run(stage, "a1")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    [config_path] = _option(argv, "--config")
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    assert config["tiers"]["enabled"] is True
    lfm = config["tiers"]["lfm"]
    assert lfm["mode"] == "attach"
    assert lfm["base_url"] == "http://127.0.0.1:18060/v1"
    assert lfm["tool_call_parser"] == "qwen3_coder"
    assert lfm["model"] == "a1"
    assert lfm["ctx"] == 2048
    assert lfm["image"] == _IMAGE
    assert load_config(Path(config_path)).tiers["lfm"]["mode"] == "attach"


@pytest.mark.parametrize(
    ("stage", "script"),
    [
        ("measure-val", "measure.py"),
        ("measure-final", "measure.py"),
        ("measure-skills", "measure_skills.py"),
    ],
)
def test_a_failing_measure_still_stops_the_helper(stage: str, script: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    args = ["--margin", "+15"] if stage == "measure-skills" else []
    result = pipe.run(stage, "a1", *args, FAKE_MEASURE_STATUS="5")
    assert result.returncode != 0
    events = _events(tmp_path)
    assert f"uv {script}" in events
    assert events.index("docker rm") > events.index(f"uv {script}")
    assert ["rm", "-f", "q46-measure-18060"] in _docker_calls(tmp_path)


def test_a_server_that_never_becomes_ready_is_stopped_and_not_measured(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, "MEASURE_WAIT_SECONDS=1\n")
    pipe.ready()
    result = pipe.run("measure-val", "a1", FAKE_CURL_STATUS="7", MEASURE_POLL_SECONDS="0.2")
    assert result.returncode != 0
    assert "fake-vllm: the last log line" in result.stderr
    assert not pipe.calls("measure.py")
    assert ["rm", "-f", "q46-measure-18060"] in _docker_calls(tmp_path)


def test_measure_port_comes_from_the_env_file(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, "MEASURE_PORT=18077\n")
    pipe.ready()
    result = pipe.run("measure-skills", "stock")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure_skills.py")
    assert _option(argv, "--url") == ["http://127.0.0.1:18077/v1"]
    assert ["rm", "-f", "q46-measure-18077"] in _docker_calls(tmp_path)


# ---------------------------------------------------------------------------
# serve_for_measure.sh (deviation d7): one pinned vLLM for every measured model
# ---------------------------------------------------------------------------


def _serve(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    _, fake_env = _fake_bin(tmp_path)
    base = {
        "MEASURE_IMAGE": _IMAGE,
        "TOOL_CALL_PARSER": "qwen3_coder",
        "MEASURE_CTX": "2048",
        "MEASURE_GPU_FRACTION": "0.08",
        "MEASURE_MAX_LOGPROBS": "22",
    }
    return subprocess.run(
        ["bash", str(_SERVE), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env={**fake_env, **base, **env},
    )


def test_serve_start_builds_exactly_the_pinned_flags(tmp_path: Path) -> None:
    model = _greedy_dir(tmp_path / "models" / "a1")
    record = tmp_path / "out" / "a1.serve.json"
    result = _serve(tmp_path, "start", str(model), "18060", str(record))
    assert result.returncode == 0, result.stderr
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert run == [
        "run",
        "-d",
        "--name",
        "q46-measure-18060",
        "-p",
        "127.0.0.1:18060:8000",
        "--gpus",
        "all",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-v",
        f"{model}:/model:ro",
        _IMAGE,
        "--model",
        "/model",
        "--served-model-name",
        "a1",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.08",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--max-logprobs",
        "22",
        "--limit-mm-per-prompt",
        '{"image": 0, "video": 0}',
    ]
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["argv"] == ["docker", *run]
    assert saved["container"] == "q46-measure-18060"
    assert saved["model_dir"] == str(model)
    assert saved["image"] == _IMAGE


def test_serve_start_takes_the_served_name_from_measure_model_name(tmp_path: Path) -> None:
    model = _greedy_dir(tmp_path / "runs" / "a1" / "merged")
    result = _serve(tmp_path, "start", str(model), "18060", MEASURE_MODEL_NAME="a1")
    assert result.returncode == 0, result.stderr
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "--served-model-name") == ["a1"]


def test_serve_start_takes_every_value_from_the_environment(tmp_path: Path) -> None:
    model = _greedy_dir(tmp_path / "m")
    result = _serve(
        tmp_path,
        "start",
        str(model),
        "18099",
        MEASURE_CTX="4096",
        MEASURE_GPU_FRACTION="0.12",
        MEASURE_MAX_LOGPROBS="30",
        TOOL_CALL_PARSER="lfm2",
    )
    assert result.returncode == 0, result.stderr
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "--max-model-len") == ["4096"]
    assert _option(run, "--gpu-memory-utilization") == ["0.12"]
    assert _option(run, "--max-logprobs") == ["30"]
    assert _option(run, "--tool-call-parser") == ["lfm2"]
    assert _option(run, "-p") == ["127.0.0.1:18099:8000"]
    assert _option(run, "--name") == ["q46-measure-18099"]


@pytest.mark.parametrize(
    "image",
    [
        "vllm/vllm-openai:latest",
        "vllm/vllm-openai",
        "vllm/vllm-openai@sha256:abc",
        "",
    ],
)
def test_serve_start_refuses_an_image_without_a_digest(image: str, tmp_path: Path) -> None:
    model = _greedy_dir(tmp_path / "m")
    result = _serve(tmp_path, "start", str(model), "18060", MEASURE_IMAGE=image)
    assert result.returncode != 0
    assert "sha256" in result.stderr
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


def test_serve_start_refuses_a_model_dir_failing_gen_config_check(tmp_path: Path) -> None:
    model = _greedy_dir(tmp_path / "m")
    (model / "generation_config.json").write_text('{"temperature": 0.7}', encoding="utf-8")
    result = _serve(tmp_path, "start", str(model), "18060")
    assert result.returncode != 0
    assert "gen_config.py check" in result.stderr
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


def test_serve_start_refuses_a_missing_model_dir(tmp_path: Path) -> None:
    result = _serve(tmp_path, "start", str(tmp_path / "nope"), "18060")
    assert result.returncode != 0
    assert not _docker_calls(tmp_path)


@pytest.mark.parametrize("port", ["80", "abc", "70000", ""])
def test_serve_refuses_an_unusable_port(port: str, tmp_path: Path) -> None:
    model = _greedy_dir(tmp_path / "m")
    for args in (("start", str(model), port), ("stop", port), ("wait", port)):
        result = _serve(tmp_path, *args)
        assert result.returncode != 0, args
    assert not _docker_calls(tmp_path)


def test_serve_start_refuses_a_tool_call_parser_left_unset(tmp_path: Path) -> None:
    model = _greedy_dir(tmp_path / "m")
    result = _serve(tmp_path, "start", str(model), "18060", TOOL_CALL_PARSER="")
    assert result.returncode != 0
    assert "TOOL_CALL_PARSER" in result.stderr
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


def test_serve_stop_removes_only_its_own_container(tmp_path: Path) -> None:
    result = _serve(tmp_path, "stop", "18060")
    assert result.returncode == 0, result.stderr
    assert _docker_calls(tmp_path) == [["rm", "-f", "q46-measure-18060"]]


def test_serve_wait_returns_once_the_models_endpoint_answers(tmp_path: Path) -> None:
    result = _serve(tmp_path, "wait", "18060")
    assert result.returncode == 0, result.stderr
    assert "curl http://127.0.0.1:18060/v1/models" in _events(tmp_path)


def test_serve_wait_prints_the_last_log_lines_on_a_timeout(tmp_path: Path) -> None:
    result = _serve(
        tmp_path,
        "wait",
        "18060",
        FAKE_CURL_STATUS="7",
        MEASURE_WAIT_SECONDS="1",
        MEASURE_POLL_SECONDS="0.2",
    )
    assert result.returncode != 0
    assert "fake-vllm: the last log line" in result.stderr
    logs = [c for c in _docker_calls(tmp_path) if c[0] == "logs"]
    assert logs and logs[0][-1] == "q46-measure-18060"
    assert "--tail" in logs[0]


def test_serve_rejects_an_unknown_command(tmp_path: Path) -> None:
    result = _serve(tmp_path, "restart", "18060")
    assert result.returncode != 0
    assert "start" in result.stderr and "stop" in result.stderr and "wait" in result.stderr
    assert not _docker_calls(tmp_path)


def test_both_env_examples_name_the_measure_server_settings() -> None:
    parsers = {_LFM_ENV: "lfm2", _QWEN_ENV: "qwen3_coder"}
    for example, parser in parsers.items():
        text = example.read_text(encoding="utf-8")
        assert f"\nMEASURE_IMAGE={_IMAGE}\n" in text
        assert "\nMEASURE_PORT=18060\n" in text
        assert "\nMEASURE_CTX=2048\n" in text
        assert "\nMEASURE_GPU_FRACTION=0.08\n" in text
        assert "\nMEASURE_MAX_LOGPROBS=22\n" in text
        assert f"\nTOOL_CALL_PARSER={parser}\n" in text


def test_assemble_renders_with_the_training_environment_on_the_path(tmp_path: Path) -> None:
    """Finding #5: build_dataset.py's render check needs transformers, which only
    the training environment has; nvsh stays importable from the repo's own env."""
    site = tmp_path / "train-site-packages"
    site.mkdir()
    train_py = tmp_path / "train-python"
    train_py.write_text(f'#!/usr/bin/env bash\necho "{site}"\n', encoding="utf-8")
    train_py.chmod(0o755)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\n")
    result = pipe.run("assemble")
    assert result.returncode == 0, result.stderr
    [(pythonpath, argv)] = pipe.calls("build_dataset.py")
    assert pythonpath == str(site)
    assert argv[:3] == ["run", "--frozen", "python"]
    [(merge_path, _)] = pipe.calls("merge_variations.py")
    assert merge_path == ""


def test_assemble_refuses_a_training_python_that_does_not_run(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={tmp_path / 'no-such-python'}\n")
    result = pipe.run("assemble")
    assert result.returncode == 1
    assert "TRAIN_PY" in result.stderr
    assert not pipe.calls("build_dataset.py")


def test_both_env_examples_name_the_ground_snapshot_and_thinking() -> None:
    for example in (_LFM_ENV, _QWEN_ENV):
        text = example.read_text(encoding="utf-8")
        assert "\nGROUND_SNAPSHOT=" in text
        assert "\nENABLE_THINKING=false" in text


@pytest.mark.parametrize("name", ["stock", "a1"])
def test_measure_final_keeps_the_single_final_runs_predictions(name: str, tmp_path: Path) -> None:
    """The final run happens once; d6's calibration step reads its predictions (P50)."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-final", name)
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    [out] = _option(argv, "--predictions")
    assert out.endswith(f"/final/{name}")


def test_measure_val_at_another_context_takes_it_from_the_environment(tmp_path: Path) -> None:
    # Issue 46, lapse l3: the env file's MEASURE_CTX=2048 silently overrode an
    # exported MEASURE_CTX=4096, so a run labelled 4K was served at 2048.
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-val", "stock", MEASURE_CTX="4096")
    assert result.returncode == 0, result.stderr
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "--max-model-len") == ["4096"]
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--ctx") == ["4096"]
    assert _option(argv, "--label") == ["stock-val-ctx4096"]
    assert _option(argv, "--out") == [str(pipe.work / "measure" / "stock-val-ctx4096.md")]


def test_measure_val_at_the_default_context_keeps_its_names(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-val", "stock")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--ctx") == ["2048"]
    assert _option(argv, "--label") == ["stock-val"]


@pytest.mark.parametrize("stage", ["measure-val", "measure-final"])
def test_measure_stages_refuse_an_extra_ctx(stage: str, tmp_path: Path) -> None:
    # Lapse l3: an extra --ctx relabels the report without changing the server.
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run(stage, "a1", "--ctx", "4096")
    assert result.returncode != 0
    assert "MEASURE_CTX" in result.stderr
    assert not pipe.calls("measure.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


def test_assemble_filters_to_the_split_and_checks_leakage(tmp_path: Path) -> None:
    """Issue 46 t19: variations of sources that left the train side are dropped
    (--filter-to-split), extra sides are excluded, and the merged set passes
    leakage_check.py against every protected file before it is rendered."""
    protected = tmp_path / "held-out.json"
    site = tmp_path / "train-site-packages"
    site.mkdir()
    train_py = tmp_path / "train-python"
    train_py.write_text(f'#!/usr/bin/env bash\necho "{site}"\n', encoding="utf-8")
    train_py.chmod(0o755)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\nPROTECTED_EXTRA={protected}\n")
    result = pipe.run("assemble")
    assert result.returncode == 0, result.stderr
    [(_, merge)] = pipe.calls("merge_variations.py")
    assert "--filter-to-split" in merge
    excluded = merge[merge.index("--exclude") + 1 :]
    assert str(protected) in excluded
    [(_, leak)] = pipe.calls("leakage_check.py")
    assert _option(leak, "--train") == [str(pipe.work / "data" / "train-augmented.merged.json")]
    assert _option(leak, "--out-filtered") == [str(pipe.work / "data" / "train-augmented.json")]
    protected_args = leak[leak.index("--protected") + 1 :]
    for side in ("val.json", "test.json"):
        assert str(pipe.work / "splits" / side) in protected_args
    assert str(protected) in protected_args
    [(_, build)] = pipe.calls("build_dataset.py")
    assert _option(build, "--split") == [str(pipe.work / "data" / "train-augmented.json")]
