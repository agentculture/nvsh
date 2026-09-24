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
import socket
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

#: The stages task t27 adds (issue 46: build, scan and privately upload bundles).
_BUNDLE_STAGES = ("bundle", "bundle-dataset", "upload-bundle")

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
        *_BUNDLE_STAGES,
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
# FAKE_UV_REAL: space-separated script names that run for real (t27).
for real in ${FAKE_UV_REAL:-}; do
  case "$*" in *"$real"*) shift 3; exec "$REAL_PY" "$@" ;; esac
done
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
[ "${FAKE_CURL_STATUS:-0}" = 0 ] || exit "$FAKE_CURL_STATUS"
# FAKE_CURL_ONCE: a file; once a model list has been served, every later
# call fails (a server that goes away right after answering).
if [ -n "${FAKE_CURL_ONCE:-}" ] && [ -f "$FAKE_CURL_ONCE" ]; then exit 7; fi
# What the fake llama-server (or a stale server, in a test) lists.
if [ -n "${LLAMA_MODELS:-}" ] && [ -f "$LLAMA_MODELS" ]; then
  cat "$LLAMA_MODELS"
  if [ -n "${FAKE_CURL_ONCE:-}" ]; then touch "$FAKE_CURL_ONCE"; fi
fi
exit 0
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
if [ -n "${FAKE_LLAMA_DIE:-}" ]; then sleep 0.5; exit 1; fi
alias=''; port=''; previous=''
for arg in "$@"; do
  [ "$previous" = --alias ] && alias=$arg
  [ "$previous" = --port ] && port=$arg
  previous=$arg
done
# Listen on the port like the real server (a child of this process, bounded).
listener=''
if [ -z "${FAKE_LLAMA_NO_LISTEN:-}" ]; then
  python3 -c 'import socket, sys, time
s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(); time.sleep(30)' \
    "$port" &
  listener=$!
fi
printf '{"object": "list", "data": [{"id": "%s"}]}\n' "$alias" > "$LLAMA_MODELS"
trap '[ -z "$listener" ] || kill "$listener"; echo "llama-server stop" >> "$EVENT_LOG"; exit 0' TERM
for _ in $(seq 300); do sleep 0.1; done
"""


def _fake_llama_server(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "bin" / "llama-server"
    path.parent.mkdir(exist_ok=True)
    path.write_text(_FAKE_LLAMA_SERVER, encoding="utf-8")
    path.chmod(0o755)
    return {
        "LLAMA_SERVER": str(path),
        "LLAMA_LOG": str(tmp_path / "llama.log"),
        "LLAMA_MODELS": str(tmp_path / "llama-models.json"),
    }


def _free_port() -> int:
    """A port nothing listens on now: serve_for_measure.sh refuses a busy one,
    and a real measure server may hold 18060 while these tests run."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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
    port = _free_port()
    pipe = _Pipeline(tmp_path, f"MEASURE_PORT={port}\n")
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
        str(port),
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
    assert lfm["base_url"] == f"http://127.0.0.1:{port}/v1"
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
    # The fake logs its own start asynchronously, so only the readiness poll
    # is ordered against it by the helper, not by this log.
    ready_at = events.index(f"curl http://127.0.0.1:{port}/v1/models")
    measure_at = events.index("uv measure.py")
    stop_at = events.index("llama-server stop")
    assert events.index("llama-server start") < measure_at
    assert ready_at < measure_at < stop_at


@pytest.mark.parametrize("stage", ["measure-val", "measure-final", "measure-skills"])
def test_a_failing_gguf_measure_still_stops_llama_server(stage: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, f"MEASURE_PORT={_free_port()}\n")
    pipe.ready(stock=False)
    _quantized(pipe)
    args = ["--margin", "+15"] if stage == "measure-skills" else []
    result = pipe.run(
        stage, "a1.q4_k_m", *args, FAKE_MEASURE_STATUS="5", **_fake_llama_server(tmp_path)
    )
    assert result.returncode != 0
    assert "llama-server stop" in _events(tmp_path)
    assert not list((pipe.work / "measure").glob("q46-measure-*.pid"))
    assert not list((pipe.work / "measure").glob("q46-measure-*.argv"))


def test_measure_skills_of_a_gguf_build_uses_the_llama_server_url(tmp_path: Path) -> None:
    port = _free_port()
    pipe = _Pipeline(tmp_path, f"MEASURE_PORT={port}\n")
    pipe.ready(stock=False)
    quant = _quantized(pipe)
    result = pipe.run(
        "measure-skills", "a1.q4_k_m", "--margin", "+15", **_fake_llama_server(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure_skills.py")
    assert _option(argv, "--url") == [f"http://127.0.0.1:{port}/v1"]
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
    pipe = _Pipeline(
        tmp_path, f"HELDOUT_SPLIT={_held_out(tmp_path)}\nMEASURE_PORT={_free_port()}\n"
    )
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
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\nMEASURE_PORT={_free_port()}\n")
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
def test_a_gguf_build_refuses_the_in_process_scorer(stage: str, mode: list, tmp_path: Path) -> None:
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


def _starttime(pid: int) -> str:
    """/proc/<pid>/stat field 22: when the process started, in clock ticks."""
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return stat.rsplit(")", 1)[1].split()[19]


def _llama_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        **_fake_llama_server(tmp_path),
        "MEASURE_RUN_DIR": str(tmp_path / "run"),
        "MEASURE_MODEL_NAME": "a1.q4_k_m",
        **extra,
    }


def _expected_argv(tmp_path: Path, model: Path, port: int) -> list[str]:
    return [
        str(tmp_path / "bin" / "llama-server"),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
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


def _running_with(text: str) -> list[int]:
    """Pids of processes (other than pgrep) whose command line contains *text*."""
    found = subprocess.run(["pgrep", "-f", text], capture_output=True, text=True)
    return [int(pid) for pid in found.stdout.split()]


def _wait_gone(pid: int) -> bool:
    deadline = time.time() + 10
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.1)
    return not _alive(pid)


def _impostor(tmp_path: Path, argv: list[str]) -> subprocess.Popen:
    """The fake llama-server started outside the helper, e.g. by someone else."""
    env = {**os.environ, **_fake_llama_server(tmp_path), "EVENT_LOG": str(tmp_path / "x.log")}
    env["LLAMA_MODELS"] = str(tmp_path / "impostor-models.json")
    env["FAKE_LLAMA_NO_LISTEN"] = "1"
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    while str(argv[0]) not in Path(f"/proc/{proc.pid}/cmdline").read_text(errors="replace"):
        assert time.time() < deadline
        time.sleep(0.05)
    return proc


def _claim_for(run_dir: Path, port: int, pid: int, starttime: str, argv: list[str]) -> None:
    run_dir.mkdir(exist_ok=True)
    (run_dir / f"q46-measure-{port}.pid").write_text(f"{pid}\n{starttime}\n", encoding="utf-8")
    (run_dir / f"q46-measure-{port}.argv").write_bytes(
        b"".join(arg.encode() + b"\0" for arg in argv)
    )


def test_serve_start_runs_llama_server_for_a_gguf_file(tmp_path: Path) -> None:
    model = _gguf(tmp_path)
    port = _free_port()
    record = tmp_path / "out" / "a1.serve.json"
    run_dir = tmp_path / "run"
    env = _llama_env(tmp_path)
    result = _serve(tmp_path, "start", str(model), str(port), str(record), MEASURE_IMAGE="", **env)
    pid_file = run_dir / f"q46-measure-{port}.pid"
    try:
        assert result.returncode == 0, result.stderr
        assert not _docker_calls(tmp_path)
        pid_text, starttime = pid_file.read_text(encoding="utf-8").split()
        pid = int(pid_text)
        assert starttime == _starttime(pid)
        argv = _expected_argv(tmp_path, model, port)
        recorded = (run_dir / f"q46-measure-{port}.argv").read_bytes()
        assert recorded == b"".join(arg.encode() + b"\0" for arg in argv)
        deadline = time.time() + 10
        while not _llama_calls(tmp_path) and time.time() < deadline:
            time.sleep(0.1)
        [served] = _llama_calls(tmp_path)
        assert served == argv[1:]
        saved = json.loads(record.read_text(encoding="utf-8"))
        assert saved["backend"] == "llama-server"
        assert saved["model_file"] == str(model)
        assert saved["served_model_name"] == "a1.q4_k_m"
        assert saved["port"] == port
        assert "temperature 0" in saved["decoding"]
        assert "built with fake-cc" in saved["version"]
        assert saved["log"] == str(run_dir / f"q46-measure-{port}.log")
        assert saved["argv"] == argv
        assert _alive(pid)
        waited = _serve(tmp_path, "wait", str(port), **env)
        assert waited.returncode == 0, waited.stderr
        assert "is ready" in waited.stderr
    finally:
        stopped = _serve(tmp_path, "stop", str(port), **env)
    assert stopped.returncode == 0, stopped.stderr
    assert _wait_gone(pid)
    assert not pid_file.exists()
    assert not (run_dir / f"q46-measure-{port}.argv").exists()
    assert "llama-server stop" in _events(tmp_path)


@pytest.mark.parametrize("setting", ["", "/no/such/llama-server"])
def test_serve_start_refuses_a_gguf_without_a_runnable_llama_server(
    setting: str, tmp_path: Path
) -> None:
    model = _gguf(tmp_path)
    result = _serve(
        tmp_path,
        "start",
        str(model),
        str(_free_port()),
        LLAMA_SERVER=setting,
        MEASURE_RUN_DIR=str(tmp_path),
    )
    assert result.returncode != 0
    assert "LLAMA_SERVER" in result.stderr
    assert not list(tmp_path.glob("q46-measure-*.pid"))


def test_serve_start_refuses_a_second_llama_server_on_one_port(tmp_path: Path) -> None:
    model = _gguf(tmp_path)
    port = _free_port()
    env = _llama_env(tmp_path)
    first = _serve(tmp_path, "start", str(model), str(port), **env)
    try:
        assert first.returncode == 0, first.stderr
        second = _serve(tmp_path, "start", str(model), str(port), **env)
        assert second.returncode != 0
        assert f"stop {port}" in second.stderr
    finally:
        _serve(tmp_path, "stop", str(port), **env)
    assert len(_llama_calls(tmp_path)) == 1


def test_serve_start_refuses_a_port_something_already_listens_on(tmp_path: Path) -> None:
    """Codex P1: a server already on the port would answer /v1/models and be
    measured as the model this helper was asked to serve."""
    model = _gguf(tmp_path)
    env = _llama_env(tmp_path)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = _serve(tmp_path, "start", str(model), str(port), **env)
    assert result.returncode != 0
    assert f"127.0.0.1:{port}" in result.stderr
    assert not _running_with(str(model))
    assert not list((tmp_path / "run").glob("q46-measure-*"))


def test_serve_start_refuses_while_another_start_holds_the_claim(tmp_path: Path) -> None:
    """Codex P2: the pid file is claimed atomically; an empty one is a start in
    progress, never overwritten."""
    model = _gguf(tmp_path)
    port = _free_port()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / f"q46-measure-{port}.pid").write_text("", encoding="utf-8")
    result = _serve(tmp_path, "start", str(model), str(port), **_llama_env(tmp_path))
    assert result.returncode != 0
    assert f"stop {port}" in result.stderr
    assert not _running_with(str(model))
    assert (run_dir / f"q46-measure-{port}.pid").read_text(encoding="utf-8") == ""


def test_serve_start_replaces_a_stale_pid_file(tmp_path: Path) -> None:
    model = _gguf(tmp_path)
    port = _free_port()
    gone = subprocess.Popen(["true"])
    gone.wait()
    argv = _expected_argv(tmp_path, model, port)
    _claim_for(tmp_path / "run", port, gone.pid, "1", argv)
    env = _llama_env(tmp_path)
    result = _serve(tmp_path, "start", str(model), str(port), **env)
    try:
        assert result.returncode == 0, result.stderr
    finally:
        _serve(tmp_path, "stop", str(port), **env)


def test_serve_start_stops_what_it_launched_when_recording_fails(tmp_path: Path) -> None:
    """Codex P2: nothing may keep running when the pid file cannot be written."""
    model = _gguf(tmp_path)
    port = _free_port()
    run_dir = tmp_path / "run"
    (run_dir / f"q46-measure-{port}.pid.tmp").mkdir(parents=True)  # the write fails
    result = _serve(tmp_path, "start", str(model), str(port), **_llama_env(tmp_path))
    assert result.returncode != 0
    deadline = time.time() + 10
    while _running_with(str(model)) and time.time() < deadline:
        time.sleep(0.1)
    assert not _running_with(str(model))
    assert not (run_dir / f"q46-measure-{port}.pid").exists()


@pytest.mark.parametrize("impostor", ["other-port", "same-argv-other-start"])
def test_serve_stop_never_kills_a_process_it_did_not_start(impostor: str, tmp_path: Path) -> None:
    """Codex P1: a reused pid running the same binary is not the server this
    helper started: identity is the exact recorded argv and start time."""
    model = _gguf(tmp_path)
    port = _free_port()
    argv = _expected_argv(tmp_path, model, port)
    other_argv = list(argv)
    if impostor == "other-port":
        other_argv[other_argv.index("--port") + 1] = str(_free_port())
    other = _impostor(tmp_path, other_argv)
    try:
        starttime = _starttime(other.pid)
        if impostor == "same-argv-other-start":
            starttime = str(int(starttime) + 1)
        _claim_for(tmp_path / "run", port, other.pid, starttime, argv)
        result = _serve(tmp_path, "stop", str(port), MEASURE_RUN_DIR=str(tmp_path / "run"))
        assert result.returncode == 0, result.stderr
        time.sleep(0.3)
        assert other.poll() is None
        assert not (tmp_path / "run" / f"q46-measure-{port}.pid").exists()
    finally:
        other.kill()
        other.wait()


def test_serve_stop_never_kills_an_unrelated_pid(tmp_path: Path) -> None:
    other = subprocess.Popen(["sleep", "30"])
    try:
        port = _free_port()
        argv = _expected_argv(tmp_path, tmp_path / "m.gguf", port)
        _claim_for(tmp_path / "run", port, other.pid, _starttime(other.pid), argv)
        result = _serve(tmp_path, "stop", str(port), MEASURE_RUN_DIR=str(tmp_path / "run"))
        assert result.returncode == 0, result.stderr
        assert other.poll() is None
    finally:
        other.kill()
        other.wait()


def test_serve_wait_fails_when_llama_server_dies_even_if_the_port_answers(
    tmp_path: Path,
) -> None:
    """Codex P1: a /v1/models answer listing the alias is not enough when the
    child this helper started has died -- something else is answering."""
    model = _gguf(tmp_path)
    port = _free_port()
    full_log = tmp_path / "a1.serve.log"
    env = _llama_env(tmp_path, FAKE_LLAMA_DIE="1")
    Path(env["LLAMA_MODELS"]).write_text('{"data": [{"id": "a1.q4_k_m"}]}\n', encoding="utf-8")
    started = _serve(tmp_path, "start", str(model), str(port), **env)
    assert started.returncode == 0, started.stderr
    [pid] = [int((tmp_path / "run" / f"q46-measure-{port}.pid").read_text().split()[0])]
    assert _wait_gone(pid)
    result = _serve(
        tmp_path,
        "wait",
        str(port),
        str(full_log),
        MEASURE_WAIT_SECONDS="10",
        MEASURE_POLL_SECONDS="0.2",
        **env,
    )
    assert result.returncode == 2
    assert "is ready" not in result.stderr
    assert "fake-llama: the last log line" in result.stderr
    assert "stopped before it was ready" in result.stderr
    assert "fake-llama" in full_log.read_text(encoding="utf-8")
    assert not [c for c in _docker_calls(tmp_path) if c[0] == "logs"]
    _serve(tmp_path, "stop", str(port), **env)


def test_serve_wait_needs_its_own_alias_and_stops_the_server_on_a_timeout(
    tmp_path: Path,
) -> None:
    """A /v1/models answer that does not list MEASURE_MODEL_NAME is not ready;
    a standalone wait that times out stops the server it was waiting on."""
    model = _gguf(tmp_path)
    port = _free_port()
    env = _llama_env(tmp_path)
    started = _serve(tmp_path, "start", str(model), str(port), **env)
    assert started.returncode == 0, started.stderr
    pid = int((tmp_path / "run" / f"q46-measure-{port}.pid").read_text().split()[0])
    other_models = tmp_path / "other-models.json"
    other_models.write_text('{"data": [{"id": "some-other-model"}]}\n', encoding="utf-8")
    try:
        result = _serve(
            tmp_path,
            "wait",
            str(port),
            MEASURE_WAIT_SECONDS="1",
            MEASURE_POLL_SECONDS="0.2",
            **{**env, "LLAMA_MODELS": str(other_models)},
        )
        assert result.returncode == 2
        assert "not ready after" in result.stderr
        assert _wait_gone(pid)
        assert not (tmp_path / "run" / f"q46-measure-{port}.pid").exists()
    finally:
        _serve(tmp_path, "stop", str(port), **env)


def test_the_gguf_temperature_rule_is_documented() -> None:
    """Deviation d3 is applied by flags for a GGUF (no generation_config.json)."""
    text = _SERVE.read_text(encoding="utf-8")
    header = text[: text.index("set -euo pipefail")]
    assert "llama-server" in header and "--temp 0 --top-k 1" in header
    assert "LLAMA_SERVER" in header
    usage = _PIPELINE.read_text(encoding="utf-8")
    usage = usage[: usage.index("set -euo pipefail")]
    assert "<run>.awq" in usage and "<run>.q4_k_m" in usage


def test_a_served_scorer_gguf_build_needs_its_base_runs_tokenizer(tmp_path: Path) -> None:
    """The GGUF's tokenizer comes from the base run's merged dir; without it
    the run must stop, not reach measure.py without --tokenizer."""
    train_py, _ = _train_py(tmp_path)
    pipe = _Pipeline(tmp_path, f"TRAIN_PY={train_py}\n")
    pipe.ready(stock=False)
    _quantized(pipe)
    _mark_scorer_run(pipe)
    shutil.rmtree(pipe.work / "runs" / "a1" / "merged")
    result = pipe.run(
        "measure-final", "a1.q4_k_m", "--scorer", "served", **_fake_llama_server(tmp_path)
    )
    assert result.returncode == 1
    assert "runs/a1/merged" in result.stderr
    assert not pipe.calls("measure.py")
    assert not _llama_calls(tmp_path)


def test_serve_wait_returns_as_soon_as_llama_server_is_ready(tmp_path: Path) -> None:
    """Codex P2: a ready native server must not fall through into the vLLM
    readiness loop, where a later failed probe turned it into exit 2."""
    model = _gguf(tmp_path)
    port = _free_port()
    env = _llama_env(tmp_path, FAKE_CURL_ONCE=str(tmp_path / "curl-once"))
    started = _serve(tmp_path, "start", str(model), str(port), **env)
    try:
        assert started.returncode == 0, started.stderr
        waited = _serve(
            tmp_path, "wait", str(port), MEASURE_WAIT_SECONDS="3", MEASURE_POLL_SECONDS="0.1", **env
        )
        assert waited.returncode == 0, waited.stderr
        assert "is ready" in waited.stderr
        assert not [c for c in _docker_calls(tmp_path) if c[0] == "inspect"]
    finally:
        _serve(tmp_path, "stop", str(port), **env)


def test_serve_wait_needs_the_listener_to_belong_to_its_llama_server(tmp_path: Path) -> None:
    """Codex P2: an answer listing the alias from a living child is not enough
    when another process holds the port -- the listening socket must be the
    child's (or its descendants')."""
    model = _gguf(tmp_path)
    port = _free_port()
    env = _llama_env(tmp_path, FAKE_LLAMA_NO_LISTEN="1")
    started = _serve(tmp_path, "start", str(model), str(port), **env)
    assert started.returncode == 0, started.stderr
    pid = int((tmp_path / "run" / f"q46-measure-{port}.pid").read_text().split()[0])
    try:
        with socket.socket() as squatter:
            squatter.bind(("127.0.0.1", port))
            squatter.listen()
            result = _serve(
                tmp_path,
                "wait",
                str(port),
                MEASURE_WAIT_SECONDS="1",
                MEASURE_POLL_SECONDS="0.2",
                **env,
            )
        assert result.returncode == 2
        assert "is ready" not in result.stderr
        assert _wait_gone(pid)
    finally:
        _serve(tmp_path, "stop", str(port), **env)


def test_serve_start_refuses_while_another_start_holds_the_lock(tmp_path: Path) -> None:
    """Codex P2: the stale-file check, removal and claim run under one lock, so
    a second start never removes a claim the first has just made."""
    model = _gguf(tmp_path)
    port = _free_port()
    run_dir = tmp_path / "run"
    gone = subprocess.Popen(["true"])
    gone.wait()
    _claim_for(run_dir, port, gone.pid, "1", _expected_argv(tmp_path, model, port))
    (run_dir / f"q46-measure-{port}.lock").mkdir()
    result = _serve(tmp_path, "start", str(model), str(port), **_llama_env(tmp_path))
    assert result.returncode != 0
    assert "lock" in result.stderr and f"stop {port}" in result.stderr
    assert not _running_with(str(model))
    assert (run_dir / f"q46-measure-{port}.pid").read_text(encoding="utf-8") == f"{gone.pid}\n1\n"
    stopped = _serve(tmp_path, "stop", str(port), MEASURE_RUN_DIR=str(run_dir))
    assert stopped.returncode == 0, stopped.stderr
    assert not (run_dir / f"q46-measure-{port}.lock").exists()
    assert not (run_dir / f"q46-measure-{port}.pid").exists()


# ---------------------------------------------------------------------------
# Issue 46, t27: bundle, bundle-dataset and upload-bundle. No Hub call is
# real: upload-bundle runs hub_upload.py against a fake huggingface_hub put on
# PYTHONPATH by a fake TRAIN_PY, and nothing ever sees a real token.
# ---------------------------------------------------------------------------

_PREFIX = "jetson-ai-lab/qwen3.5-0.8b-nvsh-"
_FAKE_TOKEN = "hf_fake_pipeline_token"

#: A stand-in huggingface_hub that keeps "remote" repos under FAKE_HUB_DIR and
#: logs every call (never the token) to FAKE_HUB_DIR/calls.jsonl.
_FAKE_HUB = """
import json, os, shutil
from pathlib import Path
from types import SimpleNamespace

REMOTE = Path(os.environ["FAKE_HUB_DIR"])


def _log(*call):
    with open(REMOTE / "calls.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(call) + "\\n")


class HfApi:
    def __init__(self, token=None):
        _log("HfApi", token == os.environ.get("EXPECTED_TOKEN"))

    def create_repo(self, repo_id, **kwargs):
        _log("create_repo", repo_id, kwargs)
        (REMOTE / repo_id).mkdir(parents=True, exist_ok=True)

    def update_repo_visibility(self, repo_id, **kwargs):
        _log("update_repo_visibility", repo_id, kwargs)

    def upload_folder(self, *, folder_path, repo_id, **kwargs):
        _log("upload_folder", repo_id, kwargs)
        shutil.copytree(folder_path, REMOTE / repo_id, dirs_exist_ok=True)
        return SimpleNamespace(oid="abc123")

    def repo_info(self, repo_id, **kwargs):
        _log("repo_info", repo_id, kwargs)
        return SimpleNamespace(private=True)

    def list_repo_files(self, repo_id, **kwargs):
        _log("list_repo_files", repo_id, kwargs)
        root = REMOTE / repo_id
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def snapshot_download(repo_id, *, local_dir, **kwargs):
    _log("snapshot_download", repo_id, {k: v for k, v in kwargs.items() if k != "token"})
    shutil.copytree(REMOTE / repo_id, local_dir, dirs_exist_ok=True)
    tamper = os.environ.get("FAKE_HUB_TAMPER")
    if tamper:
        path = Path(local_dir) / tamper
        path.write_bytes(path.read_bytes() + b"x")
    return local_dir
"""


def _bundle_pipeline(tmp_path: Path, extra_env: str = "") -> "_Pipeline":
    teachers = tmp_path / "teacher-models.json"
    teachers.write_text('{"worker": {"name": "Qwen", "licence": "Apache-2.0"}}\n')
    site = tmp_path / "fake-site"
    (site / "huggingface_hub").mkdir(parents=True)
    (site / "huggingface_hub" / "__init__.py").write_text(_FAKE_HUB, encoding="utf-8")
    train_py = tmp_path / "fake-train-python"
    train_py.write_text(f"#!/usr/bin/env bash\necho {site}\n", encoding="utf-8")
    train_py.chmod(0o755)
    return _Pipeline(
        tmp_path,
        extra_env=f"TEACHER_MODELS={teachers}\nBUNDLE_DATA_SUMMARY=n-records\n"
        f"TRAIN_PY={train_py}\n" + extra_env,
    )


def _report(tmp_path: Path, name: str = "final-a3-heal.md") -> Path:
    path = tmp_path / name
    path.write_text("# report\n", encoding="utf-8")
    return path


def _quant(pipe: "_Pipeline", run: str) -> None:
    quant = pipe.work / "quant" / run
    _greedy_dir(quant / "awq")
    (quant / "model-q4_k_m.gguf").write_bytes(b"GGUF")
    (quant / "quantize-run.json").write_text('{"awq_serve_args": []}', encoding="utf-8")


def test_bundle_bf16_builds_scans_and_records_the_bundle(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    pipe.ready(stock=False, run="a3-heal")
    report, edge = _report(tmp_path), _report(tmp_path, "edge.md")
    result = pipe.run("bundle", "bf16", "a3-heal", "tool-jev", str(report), str(edge))
    assert result.returncode == 0, result.stderr
    ((_, argv),) = pipe.calls("release_bundle.py")
    out = pipe.work / "bundles" / "tool-jev"
    assert _option(argv, "--kind") == ["bf16"]
    assert _option(argv, "--merged") == [str(pipe.work / "runs" / "a3-heal" / "merged")]
    assert _option(argv, "--repo") == [_PREFIX + "tool-jev"]
    assert _option(argv, "--run") == ["a3-heal"]
    assert _option(argv, "--results") == [str(report), str(edge)]
    assert _option(argv, "--licence-kind") == ["apache"]
    assert _option(argv, "--tool-call-parser") == ["qwen3_coder"]
    assert _option(argv, "--teacher-models") == [str(tmp_path / "teacher-models.json")]
    assert _option(argv, "--accepted") == [str(pipe.work / "aug" / "nvsh-accepted.jsonl")]
    assert _option(argv, "--train-augmented") == [str(pipe.work / "data" / "train-augmented.json")]
    assert _option(argv, "--data-summary") == ["n-records"]
    assert _option(argv, "--out") == [str(out)]
    assert "--scorer" not in argv and "--quantized-from" not in argv
    ((_, scan_argv),) = pipe.calls("scan_bundle.py")
    assert scan_argv[-2:] == ["scan", str(out)]
    meta = json.loads((pipe.work / "bundles" / "tool-jev.json").read_text(encoding="utf-8"))
    assert meta == {
        "kind": "bf16",
        "build": "a3-heal",
        "repo": _PREFIX + "tool-jev",
        "repo_type": "model",
    }


def test_bundle_gguf_ships_the_q4_k_m_file_of_its_run(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    pipe.ready(stock=False, run="a3-heal")
    _quant(pipe, "a3-heal")
    result = pipe.run("bundle", "gguf", "a3-heal.q4_k_m", "tool-jev-gguf", str(_report(tmp_path)))
    assert result.returncode == 0, result.stderr
    ((_, argv),) = pipe.calls("release_bundle.py")
    assert _option(argv, "--kind") == ["gguf"]
    assert _option(argv, "--gguf") == [str(pipe.work / "quant" / "a3-heal" / "model-q4_k_m.gguf")]
    assert _option(argv, "--merged") == [str(pipe.work / "runs" / "a3-heal" / "merged")]
    assert _option(argv, "--quantized-from") == [_PREFIX + "tool-jev"]
    assert _option(argv, "--run") == ["a3-heal"]


def test_bundle_awq_of_a_scorer_run_is_marked_a_scorer(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    pipe.ready(stock=False, run="scorer-b1")
    _mark_scorer_run(pipe, "scorer-b1")
    _quant(pipe, "scorer-b1")
    result = pipe.run(
        "bundle", "awq", "scorer-b1.awq", "tool-jev-scorer-awq", str(_report(tmp_path))
    )
    assert result.returncode == 0, result.stderr
    ((_, argv),) = pipe.calls("release_bundle.py")
    assert _option(argv, "--awq-dir") == [str(pipe.work / "quant" / "scorer-b1" / "awq")]
    assert "--scorer" in argv
    assert _option(argv, "--quantized-from") == [_PREFIX + "tool-jev-scorer"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["gguf", "a3-heal", "tool-jev-gguf"], "a3-heal.q4_k_m"),
        (["awq", "a3-heal.q4_k_m", "x"], "a3-heal.awq"),
        (["bf16", "a3-heal.awq", "x"], "run name"),
        (["fp8", "a3-heal", "x"], "bf16|gguf|awq"),
        (["bf16", "a3-heal", "Tool_Jev"], "repo suffix"),
        (["bf16", "a3-heal", "../x"], "repo suffix"),
        (["bf16", "a3-heal", "tool-jev"], "measure report"),
        (["bf16", "a3-heal", "tool-jev", "/no/such/report.md"], "/no/such/report.md"),
        (["bf16", "no-such-run", "tool-jev", "REPORT"], "no-such-run"),
    ],
)
def test_bundle_refuses_bad_arguments(tmp_path: Path, args: list, message: str) -> None:
    pipe = _bundle_pipeline(tmp_path)
    pipe.ready(stock=False, run="a3-heal")
    args = [str(_report(tmp_path)) if arg == "REPORT" else arg for arg in args]
    result = pipe.run("bundle", *args)
    assert result.returncode == 1
    assert message in result.stderr
    assert pipe.calls("release_bundle.py") == []


def test_bundle_needs_the_teacher_models_file(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path, extra_env="TEACHER_MODELS=\n")
    pipe.ready(stock=False, run="a3-heal")
    result = pipe.run("bundle", "bf16", "a3-heal", "tool-jev", str(_report(tmp_path)))
    assert result.returncode == 1
    assert "TEACHER_MODELS" in result.stderr


def test_bundle_stages_refuse_the_lfm_base(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path, extra_env="BASE=LiquidAI/LFM2.5-350M\n")
    for args in (
        ["bundle", "bf16", "a3-heal", "tool-jev", str(_report(tmp_path))],
        ["bundle-dataset", "tool-jev-data"],
    ):
        result = pipe.run(*args)
        assert result.returncode == 1
        assert "Qwen" in result.stderr


def test_bundle_dataset_builds_the_apache_only_data_set(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(
        tmp_path,
        extra_env=f'BUNDLE_REJECTED="{tmp_path}/r1.jsonl {tmp_path}/r2.jsonl"\n'
        'DATASET_MODEL_REPOS="tool-jev tool-jev-scorer"\n',
    )
    (pipe.work / "data").mkdir(parents=True)
    (pipe.work / "data" / "train-augmented.json").write_text("{}", encoding="utf-8")
    result = pipe.run("bundle-dataset", "tool-jev-dataset")
    assert result.returncode == 0, result.stderr
    ((_, argv),) = pipe.calls("dataset_bundle.py")
    out = pipe.work / "bundles" / "tool-jev-dataset"
    assert "--apache-only" in argv
    assert _option(argv, "--splits") == [str(pipe.work / "splits")]
    assert _option(argv, "--train-augmented") == [str(pipe.work / "data" / "train-augmented.json")]
    assert _option(argv, "--accepted") == [str(pipe.work / "aug" / "nvsh-accepted.jsonl")]
    rejected = argv[argv.index("--rejected") + 1 : argv.index("--rejected") + 3]
    assert rejected == [f"{tmp_path}/r1.jsonl", f"{tmp_path}/r2.jsonl"]
    assert _option(argv, "--licence") == [str(_REPO_ROOT / "LICENSE")]
    assert _option(argv, "--teacher-models") == [str(tmp_path / "teacher-models.json")]
    assert _option(argv, "--issue") == ["46"]
    assert _option(argv, "--model-repo") == [_PREFIX + "tool-jev", _PREFIX + "tool-jev-scorer"]
    assert _option(argv, "--out") == [str(out)]
    ((_, scan_argv),) = pipe.calls("scan_bundle.py")
    assert scan_argv[-2:] == ["scan", str(out)]
    meta = json.loads((pipe.work / "bundles" / "tool-jev-dataset.json").read_text())
    assert meta["kind"] == "dataset" and meta["repo_type"] == "dataset"
    assert meta["repo"] == _PREFIX + "tool-jev-dataset"


def test_bundle_dataset_needs_the_frozen_train_set(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    result = pipe.run("bundle-dataset", "tool-jev-dataset")
    assert result.returncode == 1
    assert "train-augmented.json" in result.stderr


def _scanned_bundle(pipe: "_Pipeline", suffix: str = "tool-jev", kind: str = "bf16") -> Path:
    bundle = _greedy_dir(pipe.work / "bundles" / suffix)
    (bundle / "README.md").write_text("# card\n", encoding="utf-8")
    (bundle / "model.safetensors").write_bytes(b"\x00w\x01")
    scan_bundle = _load_scan_bundle()
    scan_bundle.write_scan(bundle, scan_bundle._get_scan_secrets())  # noqa: SLF001
    meta = {"kind": kind, "build": "a3-heal", "repo": _PREFIX + suffix, "repo_type": "model"}
    (pipe.work / "bundles" / f"{suffix}.json").write_text(json.dumps(meta), encoding="utf-8")
    return bundle


def _upload_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    remote = tmp_path / "remote"
    remote.mkdir(exist_ok=True)
    return {
        "FINAL": "1",
        "HF_TOKEN": _FAKE_TOKEN,
        "EXPECTED_TOKEN": _FAKE_TOKEN,
        "FAKE_UV_REAL": "scan_bundle.py hub_upload.py",
        "FAKE_HUB_DIR": str(remote),
        **extra,
    }


def _hub_calls(tmp_path: Path) -> list[list]:
    log = tmp_path / "remote" / "calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_upload_bundle_uploads_privately_and_fetches_back_byte_identical(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    _scanned_bundle(pipe)
    result = pipe.run("upload-bundle", "tool-jev", **_upload_env(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "private=True" in result.stdout
    assert "byte-identical" in result.stdout
    assert _FAKE_TOKEN not in result.stdout + result.stderr
    calls = _hub_calls(tmp_path)
    assert [call[0] for call in calls] == [
        "HfApi",
        "create_repo",
        "update_repo_visibility",
        "upload_folder",
        "snapshot_download",
        "list_repo_files",
        "repo_info",
    ]
    assert calls[0] == ["HfApi", True]
    assert calls[1][1] == _PREFIX + "tool-jev"
    assert calls[1][2]["private"] is True
    assert calls[2][2]["private"] is True
    # hub_upload.py got the training stack's site-packages on PYTHONPATH
    ((pythonpath, _),) = pipe.calls("hub_upload.py")
    assert pythonpath.split(":")[0] == str(tmp_path / "fake-site")


def test_upload_bundle_fails_loudly_on_a_changed_fetch_back(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    _scanned_bundle(pipe)
    env = _upload_env(tmp_path, FAKE_HUB_TAMPER="model.safetensors")
    result = pipe.run("upload-bundle", "tool-jev", **env)
    assert result.returncode != 0
    assert "model.safetensors: sha256 differs" in result.stderr


def test_upload_bundle_refuses_without_final(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    _scanned_bundle(pipe)
    result = pipe.run("upload-bundle", "tool-jev", **_upload_env(tmp_path, FINAL="0"))
    assert result.returncode == 1
    assert "FINAL=1" in result.stderr
    assert _hub_calls(tmp_path) == []


@pytest.mark.parametrize("suffix", ["../tool-jev", "Tool", "a/b", ""])
def test_upload_bundle_refuses_a_bad_repo_suffix(tmp_path: Path, suffix: str) -> None:
    pipe = _bundle_pipeline(tmp_path)
    result = pipe.run("upload-bundle", suffix, **_upload_env(tmp_path))
    assert result.returncode == 1
    assert "repo suffix" in result.stderr or "upload-bundle <repo-suffix>" in result.stderr
    assert _hub_calls(tmp_path) == []


def test_upload_bundle_refuses_a_missing_bundle(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    result = pipe.run("upload-bundle", "tool-jev", **_upload_env(tmp_path))
    assert result.returncode == 1
    assert "run bundle" in result.stderr


def test_upload_bundle_refuses_a_bundle_changed_after_its_scan(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    bundle = _scanned_bundle(pipe)
    (bundle / "README.md").write_text("changed\n", encoding="utf-8")
    result = pipe.run("upload-bundle", "tool-jev", **_upload_env(tmp_path))
    assert result.returncode == 1
    assert "scan_bundle.py verify" in result.stderr
    assert _hub_calls(tmp_path) == []


def test_upload_bundle_refuses_a_model_bundle_without_greedy_decoding(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    bundle = _scanned_bundle(pipe)
    (bundle / "generation_config.json").unlink()
    scan_bundle = _load_scan_bundle()
    scan_bundle.write_scan(bundle, scan_bundle._get_scan_secrets())  # noqa: SLF001
    result = pipe.run("upload-bundle", "tool-jev", **_upload_env(tmp_path))
    assert result.returncode == 1
    assert "generation_config.json" in result.stderr
    assert _hub_calls(tmp_path) == []


def test_upload_bundle_of_a_gguf_needs_no_generation_config(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    bundle = _scanned_bundle(pipe, suffix="tool-jev-gguf", kind="gguf")
    (bundle / "generation_config.json").unlink()
    scan_bundle = _load_scan_bundle()
    scan_bundle.write_scan(bundle, scan_bundle._get_scan_secrets())  # noqa: SLF001
    result = pipe.run("upload-bundle", "tool-jev-gguf", **_upload_env(tmp_path))
    assert result.returncode == 0, result.stderr


def test_upload_bundle_refuses_an_unset_token(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    _scanned_bundle(pipe)
    env = _upload_env(tmp_path)
    del env["HF_TOKEN"]
    pipe.env.pop("HF_TOKEN", None)
    result = pipe.run("upload-bundle", "tool-jev", **env)
    assert result.returncode == 1
    assert "HF_TOKEN is not set" in result.stderr
    assert _hub_calls(tmp_path) == []


def test_upload_bundle_refuses_a_record_naming_another_repo(tmp_path: Path) -> None:
    pipe = _bundle_pipeline(tmp_path)
    _scanned_bundle(pipe)
    meta_path = pipe.work / "bundles" / "tool-jev.json"
    meta = json.loads(meta_path.read_text())
    meta["repo"] = "jetson-ai-lab/lfm2.5-350m-nvsh-triage"
    meta_path.write_text(json.dumps(meta))
    result = pipe.run("upload-bundle", "tool-jev", **_upload_env(tmp_path))
    assert result.returncode == 1
    assert _hub_calls(tmp_path) == []


def test_the_qwen_env_example_documents_teacher_models() -> None:
    text = _QWEN_ENV.read_text(encoding="utf-8")
    assert "\nTEACHER_MODELS=" in text
    assert "\nBUNDLE_DATA_SUMMARY=" in text
    assert "bundle-dataset" in text and "upload-bundle" in text


def _safetensors_file(path: Path, names: list[str]) -> None:
    import struct

    header = {"__metadata__": {"format": "pt"}}
    for index, name in enumerate(names):
        header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [2 * index, 2 * index + 2]}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0\0" * len(names))


def test_bundle_then_upload_bundle_end_to_end_with_the_real_scripts(tmp_path: Path) -> None:
    """The real release_bundle.py and scan_bundle.py build a clean bf16 bundle
    from a synthetic run; upload-bundle ships it to the fake hub, private."""
    pipe = _bundle_pipeline(tmp_path)
    (tmp_path / "teacher-models.json").write_text(
        json.dumps(
            {
                "worker": {"name": "Qwen 3.6 35B-A3B", "licence": "Apache-2.0"},
                "cortex": {"name": "Qwen 3.8 27B", "licence": "Apache-2.0"},
                "senses": {"name": "Gemma 4 26B-A4B", "licence": "Apache-2.0"},
            }
        )
    )
    snapshot = tmp_path / "hf-cache" / "hub" / "models--Qwen--Qwen3.5-0.8B" / "snapshots"
    snapshot = snapshot / _QWEN_BASE_REV
    snapshot.mkdir(parents=True)
    (snapshot / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n")
    (snapshot / "chat_template.jinja").write_text("T")
    merged = _greedy_dir(pipe.work / "runs" / "a3-heal" / "merged")
    (merged / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "mtp_num_hidden_layers": 1})
    )
    (merged / "chat_template.jinja").write_text("T")
    _safetensors_file(merged / "model.safetensors", ["model.layers.0.w"])
    (pipe.work / "aug").mkdir(parents=True)
    (pipe.work / "aug" / "nvsh-accepted.jsonl").write_text(
        json.dumps(
            {
                "id": "dev-a~v1",
                "models": {
                    "GENERATOR": "worker",
                    "CORRECTOR": "cortex",
                    "REVIEWER_A": "senses",
                    "REVIEWER_B": "cortex",
                },
                "decided_by": "reviewer_b",
                "verdicts": {"reviewer_a": {"accept": True}, "reviewer_b": {"accept": True}},
            }
        )
        + "\n"
    )
    (pipe.work / "data").mkdir(parents=True)
    (pipe.work / "data" / "train-augmented.json").write_text(
        json.dumps({"entries": [{"id": "dev-a"}, {"id": "dev-a~v1"}]})
    )
    report = tmp_path / "final-a3-heal.md"
    report.write_text(
        "# Tier 2 measurement, 2026-09-24: final-a3-heal\n\n## Issue 46 metrics\n\n"
        "| Metric | `a3-heal` |\n|---|---|\n| Right proposals (metrics.py) | 31 of 32 |\n"
    )
    real = {"FAKE_UV_REAL": "release_bundle.py scan_bundle.py hub_upload.py"}
    result = pipe.run("bundle", "bf16", "a3-heal", "tool-jev", str(report), **real)
    assert result.returncode == 0, result.stderr
    bundle = pipe.work / "bundles" / "tool-jev"
    assert json.loads((bundle / "scan.json").read_text())["clean"] is True
    assert json.loads((bundle / "config.json").read_text())["mtp_num_hidden_layers"] == 0
    assert json.loads((merged / "config.json").read_text())["mtp_num_hidden_layers"] == 1
    card = (bundle / "README.md").read_text()
    assert "| Right proposals (metrics.py) | 31 of 32 |" in card
    assert "Gemma 4 26B-A4B" in card and "Nemotron" not in card

    result = pipe.run("upload-bundle", "tool-jev", **_upload_env(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "private=True" in result.stdout
    assert (tmp_path / "remote" / (_PREFIX + "tool-jev") / "README.md").read_text() == card
