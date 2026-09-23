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
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _REPO_ROOT / "scripts" / "lfm-finetune" / "pipeline.sh"
_LFM_ENV = _REPO_ROOT / "scripts" / "lfm-finetune" / "pipeline.env.example"
_QWEN_ENV = _REPO_ROOT / "scripts" / "lfm-finetune" / "pipeline-qwen.env.example"

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
        ["shellcheck", "-x", str(_PIPELINE), str(capped)],
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
esac
exit 0
"""


class _Pipeline:
    """pipeline.sh against the Qwen example env, with `uv` replaced by a recorder."""

    def __init__(self, tmp_path: Path, extra_env: str = "") -> None:
        self.tmp = tmp_path
        self.work = tmp_path / "qwen-work"
        self.snapshot = tmp_path / "ground.json"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        uv = bin_dir / "uv"
        uv.write_text(_FAKE_UV, encoding="utf-8")
        uv.chmod(0o755)
        self.log = tmp_path / "uv.log"
        self.env_file = tmp_path / "test.env"
        self.env_file.write_text(
            _QWEN_ENV.read_text(encoding="utf-8")
            + f"\nGROUND_SNAPSHOT={self.snapshot}\n"
            + extra_env,
            encoding="utf-8",
        )
        self.env = {
            **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "UV_LOG": str(self.log),
            "REAL_PY": sys.executable,
        }

    def run(self, *stage_args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(_PIPELINE), "--env", str(self.env_file), *stage_args],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            timeout=60,
            env=self.env,
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
            (self.work / "stock").mkdir(parents=True)
            (self.work / "stock" / "config.json").write_text("{}", encoding="utf-8")
            gen_config = _REPO_ROOT / "scripts" / "lfm-finetune" / "gen_config.py"
            subprocess.run(
                [sys.executable, str(gen_config), "write", str(self.work / "stock")],
                check=True,
                capture_output=True,
            )
        if run:
            (self.work / "runs" / run).mkdir(parents=True)
            (self.work / "runs" / run / "revision").write_text("abc123\n", encoding="utf-8")


def _option(argv: list[str], name: str) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == name]


_QWEN_BASE_REV = "2fc06364715b967f1860aea9cf38778875588b17"


def test_measure_final_measures_stock_from_the_stock_copy(tmp_path: Path) -> None:
    """Finding #2a: the stock server gets the greedy generation_config.json."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-final", "a1")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == [
        str(pipe.work / "stock"),
        "jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev",
    ]
    assert _option(argv, "--revision") == [_QWEN_BASE_REV, "abc123"]
    assert "Qwen/Qwen3.5-0.8B" not in argv


@pytest.mark.parametrize("stage", ["measure-val", "measure-final", "measure-skills"])
def test_a_measure_stage_refuses_without_the_stock_copy(stage: str, tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    name = "stock" if stage == "measure-val" else "a1"
    result = pipe.run(stage, name)
    assert result.returncode == 1
    assert "stock-copy" in result.stderr
    assert not pipe.calls("measure.py") and not pipe.calls("measure_skills.py")


@pytest.mark.parametrize("stage", ["measure-val", "measure-final", "measure-skills"])
def test_a_measure_stage_refuses_a_stock_copy_without_greedy_decoding(
    stage: str, tmp_path: Path
) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    (pipe.work / "stock" / "generation_config.json").unlink()
    name = "stock" if stage == "measure-val" else "a1"
    result = pipe.run(stage, name)
    assert result.returncode == 1
    assert "generation_config.json" in result.stderr
    assert not pipe.calls("measure.py") and not pipe.calls("measure_skills.py")


def test_measure_val_measures_stock_from_the_stock_copy(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-val", "stock")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == [str(pipe.work / "stock")]
    assert _option(argv, "--revision") == [_QWEN_BASE_REV]


def test_measure_val_of_a_tuned_run_needs_no_stock_copy(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path)
    pipe.ready(stock=False)
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--model") == ["jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev"]
    assert _option(argv, "--revision") == ["abc123"]


@pytest.mark.parametrize(
    ("stage", "name"), [("measure-val", "a1"), ("measure-val", "stock"), ("measure-final", "a1")]
)
def test_measure_stages_pass_the_snapshot_thinking_and_extra_args(
    stage: str, name: str, tmp_path: Path
) -> None:
    """Finding #2b/#2c: fixed grounding, thinking off, and the operator's own flags."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run(stage, name, "--ctx", "2048", "--slice", "missing-candidate")
    assert result.returncode == 0, result.stderr
    [(_, argv)] = pipe.calls("measure.py")
    assert _option(argv, "--ground-snapshot") == [str(pipe.snapshot)]
    assert _option(argv, "--enable-thinking") == ["false"]
    assert _option(argv, "--ctx") == ["2048"]
    assert _option(argv, "--slice") == ["missing-candidate"]


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


def test_a_measure_stage_refuses_an_unset_ground_snapshot(tmp_path: Path) -> None:
    pipe = _Pipeline(tmp_path, "GROUND_SNAPSHOT=\n")
    pipe.ready()
    result = pipe.run("measure-val", "a1")
    assert result.returncode == 1
    assert "GROUND_SNAPSHOT" in result.stderr


def test_measure_skills_measures_stock_from_the_stock_copy_with_thinking_off(
    tmp_path: Path,
) -> None:
    """measure_skills.py has no --ground-snapshot (it grounds nothing); it gets
    the stock copy and --enable-thinking, and the margin reaches only the tuned run."""
    pipe = _Pipeline(tmp_path)
    pipe.ready()
    result = pipe.run("measure-skills", "a1", "--margin", "+15")
    assert result.returncode == 0, result.stderr
    (_, stock), (_, tuned) = pipe.calls("measure_skills.py")
    assert _option(stock, "--model") == [str(pipe.work / "stock")]
    assert _option(stock, "--model-revision") == [_QWEN_BASE_REV]
    assert _option(tuned, "--model") == ["jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev"]
    for argv in (stock, tuned):
        assert _option(argv, "--enable-thinking") == ["false"]
        assert "--ground-snapshot" not in argv
    assert "--margin" not in stock
    assert _option(tuned, "--margin") == ["+15"]


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
