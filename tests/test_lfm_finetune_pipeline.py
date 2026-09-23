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
