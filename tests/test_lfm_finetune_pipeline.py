"""scripts/lfm-finetune/pipeline.sh (issue 46, task t14): stage wiring for the
Apache-2.0 Qwen3.5-0.8B fine-tune, reusing issue 39's pipeline.

Every test here is a dry run: no augmentation gateway, no training stack, no
Hugging Face Hub call and no ``FINAL=1`` ever runs, exactly like the rest of
the lfm-finetune test suite (``pytest.importorskip`` is not needed -- this
module only shells out to ``pipeline.sh`` and ``shellcheck``, both of which
fail closed with a plain skip when missing).
"""

from __future__ import annotations

import hashlib
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
        "measure-heldout",
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


def _held_out(tmp_path: Path) -> Path:
    path = tmp_path / "held-out-q46.sealed.json"
    path.write_text('{"header": "held-out", "entries": []}\n', encoding="utf-8")
    return path


def test_measure_final_labels_the_missing_candidate_slice_apart(tmp_path: Path) -> None:
    """Issue 46 t24: the slice is its own run, so it never overwrites or is
    refused as a re-run of the full test-side run."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-final", "a1", "--slice", "missing-candidate")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert "--final" in argv
    assert _option(argv, "--split") == [str(pipe.work / "splits" / "test.json")]
    assert _option(argv, "--label") == ["final-a1-missing-candidate"]
    assert _option(argv, "--slice") == ["missing-candidate"]
    [out] = _option(argv, "--predictions")
    assert out.endswith("/final/a1")


def test_measure_heldout_scores_the_sealed_file_with_acceptance(tmp_path: Path) -> None:
    held_out = _held_out(tmp_path)
    pipe = _Pipeline(tmp_path, f"HELDOUT_SPLIT={held_out}\n")
    pipe.ready()
    result = pipe.run("measure-heldout", "a1")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--split") == [str(held_out)]
    assert "--acceptance" in argv and "--final" not in argv
    assert "--details" not in argv
    assert _option(argv, "--label") == ["heldout-a1"]
    [out] = _option(argv, "--predictions")
    assert out.endswith("/final/a1")


def test_measure_heldout_labels_the_missing_candidate_slice_apart(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, f"HELDOUT_SPLIT={_held_out(tmp_path)}\n")
    pipe.ready()
    result = pipe.run("measure-heldout", "a1", "--slice=missing-candidate")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--label") == ["heldout-a1-missing-candidate"]


@pytest.mark.parametrize("setting", ["", "HELDOUT_SPLIT=/no/such/held-out.json\n"])
def test_measure_heldout_needs_the_sealed_file(setting: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, setting)
    pipe.ready()
    result = pipe.run("measure-heldout", "a1")
    assert result.returncode == 1
    assert "HELDOUT_SPLIT" in result.stderr
    assert not pipe.calls("measure.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


@pytest.mark.parametrize("stage", ["measure-final", "measure-heldout"])
@pytest.mark.parametrize(
    "args",
    [
        ["--ctx", "4096"],
        ["--details", "x.jsonl"],
        ["--label", "other"],
        ["--lab", "final-a1"],
        ["--split", "other-held-out.json"],
        ["--sli=missing-candidate"],
        ["--slice", "missing-candidate", "--slice=full"],
        ["--scorer", "served", "--scorer", "in-process"],
        ["--slice", "partial"],
        ["--slice"],
        ["--out", "x.md"],
    ],
)
def test_final_stages_take_only_slice_and_scorer(stage: str, args: list, tmp_path: Path) -> None:
    """Codex on d16: measure.py's argparse keeps the last value and accepts
    abbreviations, so any other arg could relabel the run, swap the split or
    overwrite another run's predictions."""
    pipe = _Pipeline(tmp_path, f"HELDOUT_SPLIT={_held_out(tmp_path)}\n")
    pipe.ready()
    result = pipe.run(stage, "a1", *args)
    assert result.returncode == 1
    assert not pipe.calls("measure.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


@pytest.mark.parametrize(
    ("args", "label"),
    [
        (["--scorer", "served"], "final-a1"),
        (["--scorer=in-process"], "final-a1-exact"),
        (
            ["--slice", "missing-candidate", "--scorer", "in-process"],
            "final-a1-missing-candidate-exact",
        ),
    ],
)
def test_the_exact_scorer_run_is_labelled_apart(args: list, label: str, tmp_path: Path) -> None:
    """d15: Track B's exact in-process run is a second run of the same set."""
    site = tmp_path / "train-site-packages"
    site.mkdir()
    train_py = tmp_path / "train-python"
    train_py.write_text(f'#!/usr/bin/env bash\necho "{site}"\n', encoding="utf-8")
    train_py.chmod(0o755)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\n")
    pipe.ready()
    _mark_scorer_run(pipe)
    result = pipe.run("measure-final", "a1", *args)
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--label") == [label]


@pytest.mark.parametrize("args", [["--slice", "full"], ["--slice=full"]])
def test_an_explicit_full_slice_keeps_the_plain_label(args: list, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-final", "a1", *args)
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--label") == ["final-a1"]


def test_measure_heldout_refuses_a_scorer_run_without_a_scorer_mode(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, f"HELDOUT_SPLIT={_held_out(tmp_path)}\n")
    pipe.ready()
    _mark_scorer_run(pipe)
    result = pipe.run("measure-heldout", "a1")
    assert result.returncode == 1
    assert "--scorer" in result.stderr


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


def _mark_scorer_run(pipe: "_Pipeline", run: str = "a1") -> None:
    (pipe.work / "runs" / run / "train-log.json").write_text(
        '{"objective": "cross-entropy over the candidate label tokens"}\n', encoding="utf-8"
    )


@pytest.mark.parametrize("stage", ["measure-val", "measure-final"])
def test_a_measure_stage_refuses_a_scorer_run_without_a_scorer_mode(
    stage: str, tmp_path: Path
) -> None:
    """Issue 46, P66: a Track B scorer measured without --scorer is scored as a
    generative tool-caller (0 of 32 on validation) instead of as a scorer."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    _mark_scorer_run(pipe)
    result = pipe.run(stage, "a1")
    assert result.returncode == 1
    assert "--scorer" in result.stderr
    assert not pipe.calls("measure.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


@pytest.mark.parametrize("stage", ["measure-val", "measure-final"])
@pytest.mark.parametrize("mode", [["--scorer", "served"], ["--scorer=in-process"]])
def test_a_scorer_run_measures_with_a_scorer_mode(stage: str, mode: list, tmp_path: Path) -> None:
    site = tmp_path / "train-site-packages"
    site.mkdir()
    train_py = tmp_path / "train-python"
    train_py.write_text(f'#!/usr/bin/env bash\necho "{site}"\n', encoding="utf-8")
    train_py.chmod(0o755)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\n")
    pipe.ready()
    _mark_scorer_run(pipe)
    result = pipe.run(stage, "a1", *mode)
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert mode[-1] in argv


def test_a_generative_run_needs_no_scorer_mode(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    (pipe.work / "runs" / "a1" / "train-log.json").write_text('{"epochs": 5}\n', encoding="utf-8")
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 0, result.stderr


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


def test_train_scorer_trains_on_the_assembled_shared_dataset() -> None:
    """Issue 46: Track B must train on the same frozen, leakage-filtered set as
    Track A (data/train-augmented.json), never the raw split, which still holds
    the issue-39 test entries and a duplicate of a test entry."""
    text = _PIPELINE.read_text(encoding="utf-8")
    block = text[text.index("  train-scorer)") : text.index(";;", text.index("  train-scorer)"))]
    assert 'data="$WORK/data/train-augmented.json"' in block
    assert '--train "$data"' in block
    assert "splits/train.json" not in block
    assert "run assemble first" in block


def test_train_scorer_merges_and_stages_like_train() -> None:
    """Issue 46 t23: a Track B run must be measurable by measure-val like Track A
    -- a merged model dir with a greedy generation config and a staged revision."""
    text = _PIPELINE.read_text(encoding="utf-8")
    start = text.index("  train-scorer)")
    block = text[start : text.index(";;", start)]
    assert '--merge-only "$run/adapter"' in block
    assert 'gen_config.py write "$run/merged"' in block
    assert '--repo "$REPO-scorer"' in block
    assert '> "$run/revision"' in block


@pytest.mark.parametrize("stage", ["measure-val", "measure-final"])
def test_scorer_measure_stages_get_the_training_stack_and_a_tokenizer_path(
    stage: str, tmp_path: Path
) -> None:
    """Issue 46 t23: the served scorer needs transformers (the training
    environment's) and a loadable tokenizer path, not the served name."""
    site = tmp_path / "train-site-packages"
    site.mkdir()
    train_py = tmp_path / "train-python"
    train_py.write_text(f'#!/usr/bin/env bash\necho "{site}"\n', encoding="utf-8")
    train_py.chmod(0o755)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\n")
    pipe.ready()
    result = pipe.run(stage, "a1", "--scorer", "served")
    assert result.returncode == 0, result.stderr
    [(pythonpath, argv)] = pipe.calls("measure.py")
    assert pythonpath == str(site)
    assert _option(argv, "--tokenizer") == [str(pipe.work / "runs" / "a1" / "merged")]


def test_generative_measure_stages_keep_the_repo_environment(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 0, result.stderr
    [(pythonpath, argv)] = pipe.calls("measure.py")
    assert pythonpath == ""
    assert "--tokenizer" not in argv


def test_serve_wait_saves_the_full_log_when_the_server_fails(tmp_path: Path) -> None:
    # Issue 46: the last 40 lines never reached vLLM's root cause.
    log = tmp_path / "run.serve.log"
    result = _serve(
        tmp_path,
        "wait",
        "18060",
        str(log),
        FAKE_CURL_STATUS="7",
        MEASURE_WAIT_SECONDS="1",
        MEASURE_POLL_SECONDS="0.2",
    )
    assert result.returncode != 0
    assert "fake-vllm" in log.read_text(encoding="utf-8")
    assert str(log) in result.stderr
    full = [c for c in _docker_calls(tmp_path) if c[0] == "logs" and "--tail" not in c]
    assert full and full[0][-1] == "q46-measure-18060"


# ---------------------------------------------------------------------------
# Issue 46, t25 (known gap h2): measuring a quantized build of a run --
# <run>.awq (vLLM, like any model dir) and <run>.q4_k_m (native llama-server)
# ---------------------------------------------------------------------------

#: A native llama-server as serve_for_measure.sh sees it: `--version` prints to
#: stderr like the real binary; otherwise it records its argv, prints a log
#: line and stays up (bounded) until it is sent SIGTERM.
_FAKE_LLAMA_SERVER = """#!/usr/bin/env bash
if [ "${1:-}" = --version ]; then
  echo "version: 9999 (deadbeef)" >&2
  echo "built with fake-cc for test" >&2
  exit 0
fi
python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@" >> "$LLAMA_LOG"
echo "llama-server start" >> "$EVENT_LOG"
echo "fake-llama: the last log line"
[ -z "${FAKE_LLAMA_DIE:-}" ] || exit 1
trap 'echo "llama-server stop" >> "$EVENT_LOG"; exit 0' TERM
for _ in $(seq 300); do sleep 0.1; done
"""


def _fake_llama_server(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "bin" / "llama-server"
    path.parent.mkdir(exist_ok=True)
    path.write_text(_FAKE_LLAMA_SERVER, encoding="utf-8")
    path.chmod(0o755)
    return {"LLAMA_SERVER": str(path), "LLAMA_LOG": str(tmp_path / "llama.log")}


def _llama_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "llama.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _quantized(pipe: "_Pipeline", run: str = "a1") -> Path:
    """What `pipeline.sh quantize <run>` leaves: the GGUF, the greedy AWQ dir
    and quantize-run.json."""
    quant = pipe.work / "quant" / run
    _greedy_dir(quant / "awq")
    (quant / "model-q4_k_m.gguf").write_bytes(b"GGUF fake")
    (quant / "quantize-run.json").write_text('{"model_dir": "merged"}\n', encoding="utf-8")
    return quant


def _build_revision(quant: Path) -> str:
    digest = hashlib.sha256((quant / "quantize-run.json").read_bytes()).hexdigest()
    return f"sha256:{digest[:16]}"


def _train_py(tmp_path: Path) -> tuple[Path, Path]:
    site = tmp_path / "train-site-packages"
    site.mkdir()
    train_py = tmp_path / "train-python"
    train_py.write_text(f'#!/usr/bin/env bash\necho "{site}"\n', encoding="utf-8")
    train_py.chmod(0o755)
    return train_py, site


def test_measure_final_of_an_awq_build_serves_its_awq_dir_with_vllm(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    quant = _quantized(pipe)
    result = pipe.run("measure-final", "a1.awq")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == ["a1.awq"]
    assert _option(argv, "--label") == ["final-a1.awq"]
    assert _option(argv, "--revision") == [_build_revision(quant)]
    [out] = _option(argv, "--predictions")
    assert out.endswith("/final/a1.awq")
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "-v") == [f"{quant / 'awq'}:/model:ro"]
    assert _option(run, "--served-model-name") == ["a1.awq"]
    assert _option(run, "--limit-mm-per-prompt") == ['{"image": 0, "video": 0}']
    [config_path] = _option(argv, "--config")
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    assert config["tiers"]["lfm"]["engine"] == "vllm"
    record = json.loads((pipe.work / "measure" / "final-a1.awq.serve.json").read_text())
    assert record["backend"] == "vllm"


def test_measure_final_of_a_gguf_build_serves_it_with_llama_server(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    quant = _quantized(pipe)
    llama = _fake_llama_server(tmp_path)
    result = pipe.run("measure-final", "a1.q4_k_m", **llama)
    assert result.returncode == 0, result.stderr
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    [served] = _llama_calls(tmp_path)
    assert served == [
        "--model",
        str(quant / "model-q4_k_m.gguf"),
        "--host",
        "127.0.0.1",
        "--port",
        "18060",
        "--ctx-size",
        "2048",
        "--jinja",
        "--n-gpu-layers",
        "999",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--alias",
        "a1.q4_k_m",
    ]
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == ["a1.q4_k_m"]
    assert _option(argv, "--label") == ["final-a1.q4_k_m"]
    assert _option(argv, "--revision") == [_build_revision(quant)]
    [config_path] = _option(argv, "--config")
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    lfm = config["tiers"]["lfm"]
    assert lfm["engine"] == "llama-server"
    assert lfm["mode"] == "attach"
    assert lfm["base_url"] == "http://127.0.0.1:18060/v1"
    assert lfm["model"] == "a1.q4_k_m"
    # The run record's image field names the native serving stack and version.
    assert "llama-server" in lfm["image"] and "9999 (deadbeef)" in lfm["image"]
    assert _IMAGE not in lfm["image"]
    assert load_config(Path(config_path)).tiers["lfm"]["engine"] == "llama-server"
    record = json.loads((pipe.work / "measure" / "final-a1.q4_k_m.serve.json").read_text())
    assert record["backend"] == "llama-server"
    assert record["binary"] == llama["LLAMA_SERVER"]
    assert "version: 9999 (deadbeef)" in record["version"]
    assert record["argv"] == [llama["LLAMA_SERVER"], *served]
    events = _events(tmp_path)
    start_at = events.index("llama-server start")
    ready_at = events.index("curl http://127.0.0.1:18060/v1/models")
    measure_at = events.index("uv measure.py")
    stop_at = events.index("llama-server stop")
    assert start_at < ready_at < measure_at < stop_at


@pytest.mark.parametrize("stage", ["measure-val", "measure-final", "measure-skills"])
def test_a_failing_gguf_measure_still_stops_llama_server(stage: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    _quantized(pipe)
    args = ["--margin", "+15"] if stage == "measure-skills" else []
    result = pipe.run(
        stage, "a1.q4_k_m", *args, FAKE_MEASURE_STATUS="5", **_fake_llama_server(tmp_path)
    )
    assert result.returncode != 0
    assert "llama-server stop" in _events(tmp_path)
    assert not list((pipe.work / "measure").glob("q46-measure-*.pid"))


def test_measure_skills_of_a_gguf_build_uses_the_llama_server_url(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    quant = _quantized(pipe)
    result = pipe.run("measure-skills", "a1.q4_k_m", "--margin", "+15", **_fake_llama_server(tmp_path))
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure_skills.py")
    assert _option(argv, "--url") == ["http://127.0.0.1:18060/v1"]
    assert _option(argv, "--model") == ["a1.q4_k_m"]
    assert _option(argv, "--model-revision") == [_build_revision(quant)]
    assert _option(argv, "--margin") == ["+15"]


@pytest.mark.parametrize("build", ["a1.awq", "a1.q4_k_m"])
@pytest.mark.parametrize("stage", ["measure-val", "measure-final", "measure-skills"])
def test_a_build_that_was_never_quantized_is_refused_with_a_hint(
    stage: str, build: str, tmp_path: Path
) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    args = ["--margin", "+15"] if stage == "measure-skills" else []
    result = pipe.run(stage, build, *args, **_fake_llama_server(tmp_path))
    assert result.returncode == 1
    assert "quantize a1" in result.stderr
    assert not pipe.calls("measure.py") and not pipe.calls("measure_skills.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert not _llama_calls(tmp_path)


@pytest.mark.parametrize("build", ["a1.awq", "a1.q4_k_m"])
def test_a_build_without_its_quantize_record_is_refused(build: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    quant = _quantized(pipe)
    (quant / "quantize-run.json").unlink()
    result = pipe.run("measure-final", build, **_fake_llama_server(tmp_path))
    assert result.returncode == 1
    assert "quantize-run.json" in result.stderr and "quantize a1" in result.stderr
    assert not pipe.calls("measure.py")
    assert not _llama_calls(tmp_path)


def test_an_awq_build_without_greedy_decoding_is_refused(tmp_path: Path) -> None:
    """Deviation d3 holds for the AWQ dir like any model dir."""
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    quant = _quantized(pipe)
    (quant / "awq" / "generation_config.json").unlink()
    result = pipe.run("measure-final", "a1.awq")
    assert result.returncode != 0
    assert "generation_config.json" in result.stderr
    assert not pipe.calls("measure.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]


def test_a_gguf_build_needs_llama_server(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    _quantized(pipe)
    env = {k: v for k, v in pipe.env.items() if k != "LLAMA_SERVER"}
    result = subprocess.run(
        ["bash", str(_PIPELINE), "--env", str(pipe.env_file), "measure-final", "a1.q4_k_m"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode != 0
    assert "LLAMA_SERVER" in result.stderr
    assert not pipe.calls("measure.py")


@pytest.mark.parametrize(
    ("stage", "args", "label"),
    [
        ("measure-final", ["--slice", "missing-candidate"], "final-a1.q4_k_m-missing-candidate"),
        ("measure-heldout", [], "heldout-a1.q4_k_m"),
        ("measure-heldout", ["--slice=missing-candidate"], "heldout-a1.awq-missing-candidate"),
    ],
)
def test_build_names_keep_the_d16_labels(
    stage: str, args: list, label: str, tmp_path: Path
) -> None:
    pipe = _Pipeline(tmp_path, f"HELDOUT_SPLIT={_held_out(tmp_path)}\n")
    pipe.ready(stock=False)
    _quantized(pipe)
    build = label.split("-", 1)[1].removesuffix("-missing-candidate")
    result = pipe.run(stage, build, *args, **_fake_llama_server(tmp_path))
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--label") == [label]


@pytest.mark.parametrize("build", ["a1.awq", "a1.q4_k_m"])
@pytest.mark.parametrize("stage", ["measure-final", "measure-heldout"])
def test_build_names_keep_the_final_args_allowlist(stage: str, build: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, f"HELDOUT_SPLIT={_held_out(tmp_path)}\n")
    pipe.ready(stock=False)
    _quantized(pipe)
    result = pipe.run(stage, build, "--label", "other", **_fake_llama_server(tmp_path))
    assert result.returncode == 1
    assert not pipe.calls("measure.py")
    assert not _llama_calls(tmp_path)


@pytest.mark.parametrize("build", ["a1.awq", "a1.q4_k_m"])
def test_build_names_refuse_an_extra_ctx(build: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    _quantized(pipe)
    result = pipe.run("measure-val", build, "--ctx", "4096", **_fake_llama_server(tmp_path))
    assert result.returncode != 0
    assert "MEASURE_CTX" in result.stderr
    assert not _llama_calls(tmp_path)


@pytest.mark.parametrize("build", ["a1.awq", "a1.q4_k_m"])
@pytest.mark.parametrize("stage", ["measure-val", "measure-final"])
def test_a_scorer_build_still_needs_a_scorer_mode(stage: str, build: str, tmp_path: Path) -> None:
    """P66 holds for a build: the base run's train-log.json says it is a scorer."""
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    _quantized(pipe)
    _mark_scorer_run(pipe)
    result = pipe.run(stage, build, **_fake_llama_server(tmp_path))
    assert result.returncode == 1
    assert "--scorer" in result.stderr
    assert not pipe.calls("measure.py")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert not _llama_calls(tmp_path)


@pytest.mark.parametrize(
    ("build", "tokenizer"), [("a1.awq", "quant/a1/awq"), ("a1.q4_k_m", "runs/a1/merged")]
)
def test_a_served_scorer_build_gets_a_tokenizer_dir(
    build: str, tokenizer: str, tmp_path: Path
) -> None:
    train_py, site = _train_py(tmp_path)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\n")
    pipe.ready(stock=False)
    _quantized(pipe)
    _mark_scorer_run(pipe)
    result = pipe.run("measure-final", build, "--scorer", "served", **_fake_llama_server(tmp_path))
    assert result.returncode == 0, result.stderr
    [(pythonpath, argv)] = pipe.calls("measure.py")
    assert pythonpath == str(site)
    assert _option(argv, "--tokenizer") == [str(pipe.work / tokenizer)]
    assert _option(argv, "--label") == [f"final-{build}"]


@pytest.mark.parametrize(
    ("stage", "mode"),
    [
        ("measure-val", ["--scorer", "in-process"]),
        ("measure-val", ["--scorer=in-process"]),
        ("measure-final", ["--scorer", "in-process"]),
        ("measure-heldout", ["--scorer=in-process"]),
    ],
)
def test_a_gguf_build_refuses_the_in_process_scorer(
    stage: str, mode: list, tmp_path: Path
) -> None:
    """transformers never loads the GGUF: an in-process run would measure the
    bf16 base run under the quant's name."""
    train_py, _ = _train_py(tmp_path)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\nHELDOUT_SPLIT={_held_out(tmp_path)}\n")
    pipe.ready(stock=False)
    _quantized(pipe)
    _mark_scorer_run(pipe)
    result = pipe.run(stage, "a1.q4_k_m", *mode, **_fake_llama_server(tmp_path))
    assert result.returncode == 1
    assert "in-process" in result.stderr and "GGUF" in result.stderr
    assert not pipe.calls("measure.py")
    assert not _llama_calls(tmp_path)


def test_a_plain_run_name_with_a_dot_is_still_a_run(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False, run="a1.5")
    result = pipe.run("measure-final", "a1.5")
    assert result.returncode == 0, result.stderr
    [run] = [c for c in _docker_calls(tmp_path) if c[0] == "run"]
    assert _option(run, "-v") == [f"{pipe.work / 'runs' / 'a1.5' / 'merged'}:/model:ro"]


# serve_for_measure.sh's llama-server backend (a MODEL that is a .gguf file)


def _gguf(tmp_path: Path) -> Path:
    path = tmp_path / "quant" / "a1" / "model-q4_k_m.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF fake")
    return path


def _alive_pid(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").split()[0])


def test_serve_start_runs_llama_server_for_a_gguf_file(tmp_path: Path) -> None:
    model = _gguf(tmp_path)
    record = tmp_path / "out" / "a1.serve.json"
    run_dir = tmp_path / "run"
    llama = _fake_llama_server(tmp_path)
    env = {**llama, "MEASURE_RUN_DIR": str(run_dir), "MEASURE_MODEL_NAME": "a1.q4_k_m"}
    result = _serve(tmp_path, "start", str(model), "18061", str(record), MEASURE_IMAGE="", **env)
    try:
        assert result.returncode == 0, result.stderr
        assert not _docker_calls(tmp_path)
        pid_file = run_dir / "q46-measure-18061.pid"
        pid = _alive_pid(pid_file)
        deadline = time.time() + 10
        while not _llama_calls(tmp_path) and time.time() < deadline:
            time.sleep(0.1)
        [served] = _llama_calls(tmp_path)
        assert _option(served, "--model") == [str(model)]
        assert _option(served, "--port") == ["18061"]
        assert _option(served, "--host") == ["127.0.0.1"]
        assert _option(served, "--alias") == ["a1.q4_k_m"]
        saved = json.loads(record.read_text(encoding="utf-8"))
        assert saved["backend"] == "llama-server"
        assert saved["model_file"] == str(model)
        assert saved["served_model_name"] == "a1.q4_k_m"
        assert saved["port"] == 18061
        assert "temperature 0" in saved["decoding"]
        assert "built with fake-cc" in saved["version"]
        assert saved["log"] == str(run_dir / "q46-measure-18061.log")
        assert _alive(pid)
    finally:
        stopped = _serve(tmp_path, "stop", "18061", MEASURE_RUN_DIR=str(run_dir))
    assert stopped.returncode == 0, stopped.stderr
    deadline = time.time() + 10
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.1)
    assert not _alive(pid)
    assert not pid_file.exists()
    assert "llama-server stop" in _events(tmp_path)


@pytest.mark.parametrize("setting", ["", "/no/such/llama-server"])
def test_serve_start_refuses_a_gguf_without_a_runnable_llama_server(
    setting: str, tmp_path: Path
) -> None:
    model = _gguf(tmp_path)
    result = _serve(
        tmp_path, "start", str(model), "18061", LLAMA_SERVER=setting, MEASURE_RUN_DIR=str(tmp_path)
    )
    assert result.returncode != 0
    assert "LLAMA_SERVER" in result.stderr
    assert not list(tmp_path.glob("q46-measure-*.pid"))


def test_serve_start_refuses_a_second_llama_server_on_one_port(tmp_path: Path) -> None:
    model = _gguf(tmp_path)
    env = {**_fake_llama_server(tmp_path), "MEASURE_RUN_DIR": str(tmp_path / "run")}
    first = _serve(tmp_path, "start", str(model), "18061", **env)
    try:
        assert first.returncode == 0, first.stderr
        second = _serve(tmp_path, "start", str(model), "18061", **env)
        assert second.returncode != 0
        assert "stop 18061" in second.stderr
    finally:
        _serve(tmp_path, "stop", "18061", **env)


def test_serve_stop_never_kills_a_process_it_did_not_start(tmp_path: Path) -> None:
    other = subprocess.Popen(["sleep", "30"])
    try:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "q46-measure-18061.pid").write_text(
            f"{other.pid}\n{tmp_path / 'bin' / 'llama-server'}\n", encoding="utf-8"
        )
        result = _serve(tmp_path, "stop", "18061", MEASURE_RUN_DIR=str(run_dir))
        assert result.returncode == 0, result.stderr
        assert other.poll() is None
        assert not (run_dir / "q46-measure-18061.pid").exists()
    finally:
        other.kill()
        other.wait()


def test_serve_wait_prints_llama_servers_log_when_it_dies(tmp_path: Path) -> None:
    model = _gguf(tmp_path)
    run_dir = tmp_path / "run"
    full_log = tmp_path / "a1.serve.log"
    env = {**_fake_llama_server(tmp_path), "MEASURE_RUN_DIR": str(run_dir), "FAKE_LLAMA_DIE": "1"}
    started = _serve(tmp_path, "start", str(model), "18061", **env)
    assert started.returncode == 0, started.stderr
    result = _serve(
        tmp_path,
        "wait",
        "18061",
        str(full_log),
        FAKE_CURL_STATUS="7",
        MEASURE_WAIT_SECONDS="10",
        MEASURE_POLL_SECONDS="0.2",
        **env,
    )
    assert result.returncode == 2
    assert "fake-llama: the last log line" in result.stderr
    assert "stopped before it was ready" in result.stderr
    assert "fake-llama" in full_log.read_text(encoding="utf-8")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "logs"]
    _serve(tmp_path, "stop", "18061", **env)


def test_the_gguf_temperature_rule_is_documented() -> None:
    """Deviation d3 is applied by flags for a GGUF (no generation_config.json)."""
    text = _SERVE.read_text(encoding="utf-8")
    header = text[: text.index("set -euo pipefail")]
    assert "llama-server" in header and "--temp 0 --top-k 1" in header
    assert "LLAMA_SERVER" in header
    usage = _PIPELINE.read_text(encoding="utf-8")
    usage = usage[: usage.index("set -euo pipefail")]
    assert "<run>.awq" in usage and "<run>.q4_k_m" in usage
