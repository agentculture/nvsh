#!/usr/bin/env python3
"""Quantization and healing tooling for the Qwen3.5-0.8B Tool-Jev (issue 46, task t12).

A development-machine tool, never imported by the nvsh package. It builds a
train-side-only calibration set (spec h27: calibration and healing data come
from the train split only, never val or test) and drives three external
tools as subprocesses -- llama.cpp's GGUF converter, its ``imatrix`` and
``quantize`` binaries, and ``llm-compressor`` for INT4 AWQ -- to produce a
text-only (no vision ``mmproj``) ``Q4_K_M`` GGUF and an INT4 AWQ export of a
merged checkpoint (spec c44). Every tool's path comes from an environment
variable (``LLAMA_CPP_CONVERT``, ``LLAMA_CPP_QUANTIZE``, ``LLAMA_CPP_IMATRIX``,
``LLM_COMPRESSOR``): none is hard-coded, and none of these tools is a
dependency of nvsh itself.

    LLAMA_CPP_CONVERT=<llama.cpp checkout>/convert_hf_to_gguf.py \\
    LLAMA_CPP_QUANTIZE=<llama.cpp build>/llama-quantize \\
    LLAMA_CPP_IMATRIX=<llama.cpp build>/llama-imatrix \\
    LLM_COMPRESSOR=<train venv>/bin/llmcompressor \\
        python scripts/lfm-finetune/quantize.py --model-dir runs/r1/merged \\
            --train out/train.json --val out/val.json --test out/test.json \\
            --work-dir quant/r1 --repo jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev

This script never runs any of the four tools in its own tests: every
subprocess call goes through an injected ``run`` seam (the same
``argv, timeout -> (returncode, output)`` shape ``measure.py`` uses), so
tests substitute a fake runner instead of invoking llama.cpp or
llm-compressor.

:func:`heal_needed` is the healing trigger from decisions c42/c43: a
quantized build is re-tuned only when it loses more than 3 percentage points
of right proposals against its bf16 checkpoint, or introduces a wrong
mutating proposal bf16 did not have. Nothing in this script decides *when*
to call it -- that is task t18's run, once measure.py (task t13) produces
comparable bf16/quant summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

DEFAULT_TIMEOUT = 3600.0
VERSION_TIMEOUT = 30.0

#: Percentage-point margin from decisions c42/c43: a drop of MORE than this
#: many points of right proposals triggers healing.
HEAL_MARGIN_POINTS = 3.0

#: Env var each tool path is read from (c44: no hard-coded tool paths).
ENV_VARS = {
    "convert": "LLAMA_CPP_CONVERT",
    "quantize": "LLAMA_CPP_QUANTIZE",
    "imatrix": "LLAMA_CPP_IMATRIX",
    "compressor": "LLM_COMPRESSOR",
}

RunFn = Callable[[list[str], float], "tuple[int, str]"]


class QuantizeError(Exception):
    """A refusal or a failed tool invocation: printed as one line, never a traceback."""


def default_run(argv: list[str], timeout: float) -> tuple[int, str]:  # pragma: no cover
    """Run *argv* (a fixed list, no shell) and return ``(exit code, output)``."""
    try:
        completed = subprocess.run(  # nosec B603 - fixed argv list, no shell=True
            argv, check=False, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return (127, f"{argv[0]}: not found")
    except (OSError, subprocess.SubprocessError) as exc:
        return (1, f"{type(exc).__name__}: {exc}")
    return (completed.returncode, (completed.stdout or "") + (completed.stderr or ""))


# ---------------------------------------------------------------------------
# Calibration set: train-side only (h27)
# ---------------------------------------------------------------------------


def _load_split(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("entries", []) if isinstance(raw, dict) else raw


def _source_ids(entries: Iterable[dict]) -> set[str]:
    return {entry.get("source_id", entry["id"]) for entry in entries}


def build_calibration_set(
    train: Path, val: Path, test: Path, limit: int | None = None
) -> list[str]:
    """Calibration text for imatrix/AWQ, drawn only from *train* (h27).

    Refuses if any train entry's ``source_id`` also appears on *val* or
    *test* -- belt-and-braces on top of ``split.py``'s own non-overlap
    guarantee, since a hand-edited split file could reintroduce one and
    silently leak validation/test data into calibration or a later heal run.
    """
    train_entries = _load_split(train)
    other_ids = _source_ids(_load_split(val)) | _source_ids(_load_split(test))
    overlap = sorted(_source_ids(train_entries) & other_ids)
    if overlap:
        raise QuantizeError(
            f"calibration set would use non-train source_id(s) {overlap}; calibration and"
            " healing data must be train-side only (h27)"
        )
    texts = [entry["text"] for entry in train_entries if "text" in entry]
    if limit is not None:
        texts = texts[:limit]
    return texts


def write_calibration_file(texts: Sequence[str], out_path: Path) -> Path:
    """Write one calibration text per line, for llama.cpp ``imatrix`` and AWQ's calibration."""
    content = "\n".join(texts)
    out_path.write_text(content + ("\n" if texts else ""), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Tool paths, read only from the environment (c44)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolPaths:
    """Where each external tool lives on this machine; never hard-coded."""

    convert: str
    quantize: str
    imatrix: str
    compressor: str


def tool_paths_from_env(env: Mapping[str, str]) -> ToolPaths:
    """Read every tool path from *env*; refuses if any of the four is unset."""
    missing = [var for var in ENV_VARS.values() if not env.get(var)]
    if missing:
        raise QuantizeError(f"missing environment variable(s): {', '.join(missing)}")
    return ToolPaths(**{name: env[var] for name, var in ENV_VARS.items()})


# ---------------------------------------------------------------------------
# GGUF conversion + imatrix + Q4_K_M (text-only, no mmproj)
# ---------------------------------------------------------------------------


def convert_gguf(
    run: RunFn,
    tools: ToolPaths,
    model_dir: Path,
    out_file: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Convert a merged HF checkpoint to an f16 GGUF. Refuses a vision-projector output.

    Qwen3.5-0.8B is a hybrid model with a vision tower; the served artifact
    (spec c44) is text-only, so a converter that writes a separate
    ``mmproj-*`` file alongside *out_file* anyway is a refusal, not silently
    accepted.
    """
    argv = [tools.convert, str(model_dir), "--outfile", str(out_file), "--outtype", "f16"]
    code, output = run(argv, timeout)
    if code != 0:
        raise QuantizeError(f"GGUF conversion failed (exit {code}): {output}")
    mmproj = out_file.parent / f"mmproj-{out_file.name}"
    if mmproj.exists():
        raise QuantizeError(
            f"conversion wrote a vision projector {mmproj}; this export is text-only, no mmproj"
        )
    return output


def compute_imatrix(
    run: RunFn,
    tools: ToolPaths,
    gguf_file: Path,
    calibration_file: Path,
    out_file: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Build an importance matrix from the train-side calibration file."""
    argv = [
        tools.imatrix,
        "-m",
        str(gguf_file),
        "-f",
        str(calibration_file),
        "-o",
        str(out_file),
    ]
    code, output = run(argv, timeout)
    if code != 0:
        raise QuantizeError(f"imatrix computation failed (exit {code}): {output}")
    return output


def quantize_q4_k_m(
    run: RunFn,
    tools: ToolPaths,
    gguf_file: Path,
    imatrix_file: Path,
    out_file: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Quantize an f16 GGUF to Q4_K_M, imatrix-guided."""
    argv = [
        tools.quantize,
        "--imatrix",
        str(imatrix_file),
        str(gguf_file),
        str(out_file),
        "Q4_K_M",
    ]
    code, output = run(argv, timeout)
    if code != 0:
        raise QuantizeError(f"Q4_K_M quantization failed (exit {code}): {output}")
    return output


def export_awq(
    run: RunFn,
    tools: ToolPaths,
    model_dir: Path,
    calibration_file: Path,
    out_dir: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Export an INT4 AWQ checkpoint via llm-compressor, calibrated from the train side only."""
    argv = [
        tools.compressor,
        "--model",
        str(model_dir),
        "--calibration",
        str(calibration_file),
        "--scheme",
        "AWQ",
        "--bits",
        "4",
        "--out",
        str(out_dir),
    ]
    code, output = run(argv, timeout)
    if code != 0:
        raise QuantizeError(f"INT4 AWQ export failed (exit {code}): {output}")
    return output


# ---------------------------------------------------------------------------
# Tool version recording (c44: every result names its exact tool version)
# ---------------------------------------------------------------------------


def tool_version(run: RunFn, path: str, timeout: float = VERSION_TIMEOUT) -> str:
    """The tool's own ``--version`` output, or ``"unknown (...)"`` if it fails."""
    code, output = run([path, "--version"], timeout)
    output = output.strip()
    if code != 0:
        return f"unknown ({output or code})"
    return output


def record_tool_versions(
    run: RunFn, tools: ToolPaths, timeout: float = VERSION_TIMEOUT
) -> dict[str, str]:
    """Every tool's version, keyed by name, for the run log."""
    return {
        "llama.cpp convert": tool_version(run, tools.convert, timeout),
        "llama.cpp imatrix": tool_version(run, tools.imatrix, timeout),
        "llama.cpp quantize": tool_version(run, tools.quantize, timeout),
        "llm-compressor": tool_version(run, tools.compressor, timeout),
    }


# ---------------------------------------------------------------------------
# heal_needed (decisions c42, c43)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSummary:
    """The two headline numbers ``heal_needed`` compares between bf16 and a quantized build."""

    right_pct: float
    wrong_mutating: int


def heal_needed(bf16: QuantSummary, quant: QuantSummary) -> bool:
    """True when *quant* has drifted enough from *bf16* to need a healing fine-tune.

    Two independent triggers (decisions c42, c43), either is sufficient:
    right proposals drop by MORE than :data:`HEAL_MARGIN_POINTS` percentage
    points, or *quant* has more wrong mutating proposals than *bf16* (a new
    one appeared). A drop of exactly the margin, or an unchanged/lower wrong-
    mutating count, does not trigger healing on its own.
    """
    if bf16.right_pct - quant.right_pct > HEAL_MARGIN_POINTS:
        return True
    if quant.wrong_mutating > bf16.wrong_mutating:
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--calibration-limit", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        tools = tool_paths_from_env(os.environ)
        texts = build_calibration_set(args.train, args.val, args.test, args.calibration_limit)
    except QuantizeError as exc:
        parser.error(str(exc))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    calibration_file = write_calibration_file(texts, args.work_dir / "calibration.txt")

    run = default_run
    gguf_f16 = args.work_dir / "model-f16.gguf"
    imatrix_file = args.work_dir / "imatrix.dat"
    q4_k_m = args.work_dir / "model-q4_k_m.gguf"
    awq_dir = args.work_dir / "awq"

    try:
        convert_gguf(run, tools, args.model_dir, gguf_f16)
        compute_imatrix(run, tools, gguf_f16, calibration_file, imatrix_file)
        quantize_q4_k_m(run, tools, gguf_f16, imatrix_file, q4_k_m)
        export_awq(run, tools, args.model_dir, calibration_file, awq_dir)
    except QuantizeError as exc:
        parser.error(str(exc))

    versions = record_tool_versions(run, tools)
    run_log = {
        "calibration_entries": len(texts),
        "gguf_q4_k_m": str(q4_k_m),
        "awq_dir": str(awq_dir),
        "tool_versions": versions,
    }
    (args.work_dir / "quantize-run.json").write_text(
        json.dumps(run_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_log, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
