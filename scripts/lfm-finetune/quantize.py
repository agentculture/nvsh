#!/usr/bin/env python3
"""Quantization and healing tooling for the Qwen3.5-0.8B Tool-Jev (issue 46, task t12).

A development-machine tool, never imported by the nvsh package. It builds a
train-side-only calibration set (spec h27: calibration and healing data come
from the train split only, never val or test) and drives llama.cpp's GGUF
converter plus its ``imatrix`` and ``quantize`` binaries to produce a
text-only (no vision ``mmproj``) ``bf16`` GGUF, imatrix-quantized to
``Q4_K_M`` (spec c44). Every llama.cpp tool's path comes from an environment
variable (``LLAMA_CPP_CONVERT``, ``LLAMA_CPP_QUANTIZE``, ``LLAMA_CPP_IMATRIX``):
none is hard-coded, and none of these tools is a dependency of nvsh itself.

Risk r14 (t16 spike): stock ``llm-compressor``'s ``AWQModifier`` needs
transformers >= 5.17.0, newer than the shared training venv's pin, so the
INT4 AWQ export never runs in-process here. It runs the proven recipe
(``awq_oneshot.py``, next to this script) as a **subprocess of a separate AWQ
venv's python**, named by the ``AWQ_PY`` environment variable -- not an
``llm-compressor`` command line. After the export, this script copies the
vLLM-serving files ``save_pretrained`` does not write and writes a
greedy-decoding ``generation_config.json`` (deviation d3).

    LLAMA_CPP_CONVERT=<llama.cpp checkout>/convert_hf_to_gguf.py \\
    LLAMA_CPP_QUANTIZE=<llama.cpp build>/llama-quantize \\
    LLAMA_CPP_IMATRIX=<llama.cpp build>/llama-imatrix \\
    AWQ_PY=<AWQ venv>/bin/python \\
        python scripts/lfm-finetune/quantize.py --model-dir runs/r1/merged \\
            --train out/train.json --val out/val.json --test out/test.json \\
            --work-dir quant/r1 --repo jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev

This script never runs any external tool in its own tests: every subprocess
call goes through an injected ``run`` seam (the same
``argv, timeout -> (returncode, output)`` shape ``measure.py`` uses), so
tests substitute a fake runner instead of invoking llama.cpp, the AWQ venv,
or git.

:func:`heal_needed` is the healing trigger from decisions c42/c43: a
quantized build is re-tuned only when it loses more than 3 percentage points
of right proposals against its bf16 checkpoint, or introduces a wrong
mutating proposal bf16 did not have. Nothing in this script decides *when*
to call it -- that is task t18's run, once measure.py (task t13) produces
comparable bf16/quant summaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess  # nosec B404 - fixed argv lists, never a shell
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # runnable from any directory

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A script next to this one, loaded by path (they are not a package)."""
    spec = importlib.util.spec_from_file_location(f"lfm_finetune_{name}", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


DEFAULT_TIMEOUT = 3600.0
VERSION_TIMEOUT = 30.0

#: awq_oneshot.py, run as a subprocess of AWQ_PY -- never imported in-process (risk r14).
_AWQ_ONESHOT_SCRIPT = _HERE / "awq_oneshot.py"

#: The env var naming the separate AWQ venv's python (risk r14: too new a
#: transformers for the shared training venv to carry).
AWQ_PY_VAR = "AWQ_PY"

#: llm-compressor's save_pretrained does not write these; vLLM needs them to
#: serve the AWQ export, so they are copied from the source model dir.
VLLM_SUPPORT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

#: nvsh's Tier 2 launcher cannot pass this vLLM flag today (known limitation,
#: recorded in the run log rather than silently dropped).
AWQ_SERVE_ARGS = ["--limit-mm-per-prompt", '{"image": 0, "video": 0}']

#: Where llama.cpp's checked-out commit is read from, if the operator set it.
LLAMA_CPP_DIR_VAR = "LLAMA_CPP_DIR"

#: Run inside the AWQ venv to report the two package versions llm-compressor needs.
_AWQ_VERSION_PROBE = (
    "import importlib.metadata as m; print(m.version('llmcompressor'), m.version('transformers'))"
)

#: Percentage-point margin from decisions c42/c43: a drop of MORE than this
#: many points of right proposals triggers healing.
HEAL_MARGIN_POINTS = 3.0

#: ``split.py``'s note in a side's header: ``Split '<side>' of <corpus> (seed=N).``
#: Mirrors ``train_scorer.py``'s own copy of this pattern.
_SPLIT_SIDE_RE = re.compile(r"Split '(\w+)' of ")

#: The held-out split's file name and the phrase its header opens with.
HELD_OUT_NAME = "held-out.json"
_HELD_OUT_MARKER = "held-out split"

#: Env var each llama.cpp tool path is read from (c44: no hard-coded tool paths).
ENV_VARS = {
    "convert": "LLAMA_CPP_CONVERT",
    "quantize": "LLAMA_CPP_QUANTIZE",
    "imatrix": "LLAMA_CPP_IMATRIX",
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


def _load_split(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {"header": "", "entries": raw}


def _entries_of(raw: dict) -> list[dict]:
    return raw.get("entries", [])


def _source_ids(entries: Iterable[dict]) -> set[str]:
    return {entry.get("source_id", entry["id"]) for entry in entries}


def _verify_split_side(path: Path, raw: dict, expected: str) -> None:
    """Refuse *path* unless its ``split.py`` header names *expected*.

    Checking only for disjoint ``source_id``\\ s (below) cannot catch a
    swapped ``--train``/``--val`` pair when both splits are otherwise
    ordinary and mutually disjoint -- each file's header must name the role
    it is used for. The held-out split is refused outright, by file name
    and by its header's own marker, exactly as ``train_scorer.py`` refuses
    it for training (Codex finding #3).
    """
    header = raw.get("header")
    header = header if isinstance(header, str) else ""
    if path.name == HELD_OUT_NAME or header.casefold().startswith(_HELD_OUT_MARKER):
        raise QuantizeError(
            f"{path}: the held-out split is never used for calibration or healing (h27)"
        )
    match = _SPLIT_SIDE_RE.search(header)
    found = match.group(1) if match else None
    if found != expected:
        raise QuantizeError(
            f"{path}: its header names side {found!r}, expected {expected!r} -- "
            "calibration refuses a file whose header does not match its --train/--val/--test role"
        )


def build_calibration_set(
    train: Path, val: Path, test: Path, limit: int | None = None
) -> list[str]:
    """Calibration text for imatrix/AWQ, drawn only from *train* (h27).

    Verifies each of *train*, *val* and *test* against its ``split.py``
    header before trusting which side it is: a swapped ``--train``/``--val``
    pair is refused even when the two files are otherwise ordinary, mutually
    disjoint splits (Codex finding #3), and the held-out split is refused
    outright. On top of that, refuses if any train entry's ``source_id``
    also appears on *val* or *test* -- belt-and-braces on top of
    ``split.py``'s own non-overlap guarantee, since a hand-edited split file
    could reintroduce one and silently leak validation/test data into
    calibration or a later heal run.
    """
    train_raw, val_raw, test_raw = _load_split(train), _load_split(val), _load_split(test)
    _verify_split_side(train, train_raw, "train")
    _verify_split_side(val, val_raw, "val")
    _verify_split_side(test, test_raw, "test")

    train_entries = _entries_of(train_raw)
    other_ids = _source_ids(_entries_of(val_raw)) | _source_ids(_entries_of(test_raw))
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
    """Where each llama.cpp tool lives on this machine; never hard-coded."""

    convert: str
    quantize: str
    imatrix: str


def tool_paths_from_env(env: Mapping[str, str]) -> ToolPaths:
    """Read every llama.cpp tool path from *env*; refuses if any of the three is unset."""
    missing = [var for var in ENV_VARS.values() if not env.get(var)]
    if missing:
        raise QuantizeError(f"missing environment variable(s): {', '.join(missing)}")
    return ToolPaths(**{name: env[var] for name, var in ENV_VARS.items()})


def awq_python_from_env(env: Mapping[str, str]) -> str:
    """The AWQ venv's python, from ``AWQ_PY``; refuses when unset (risk r14).

    The AWQ step never falls back to the training venv's python: stock
    llm-compressor needs a newer transformers than the training venv pins.
    """
    awq_py = env.get(AWQ_PY_VAR)
    if not awq_py:
        raise QuantizeError(f"missing environment variable: {AWQ_PY_VAR}")
    return awq_py


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
    """Convert a merged HF checkpoint to a bf16 GGUF. Refuses a vision-projector output.

    ``bf16``, not ``f16`` (risk r14 / t16 spike): Qwen3.5-0.8B's own weights
    are bf16, and converting through f16 first risks a silent range-overflow
    rounding pass that bf16 skips. Qwen3.5-0.8B is a hybrid model with a
    vision tower; the served artifact (spec c44) is text-only, so a converter
    that writes a separate ``mmproj-*`` file alongside *out_file* anyway is a
    refusal, not silently accepted.
    """
    argv = [tools.convert, str(model_dir), "--outfile", str(out_file), "--outtype", "bf16"]
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
    awq_py: str,
    model_dir: Path,
    calibration_file: Path,
    out_dir: Path,
    num_calibration_samples: int,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Run the proven AWQ recipe (risk r14 / t16 spike) as a subprocess of *awq_py*.

    Never an ``llm-compressor`` command line, and never imported in-process:
    ``awq_oneshot.py`` runs inside a separate venv (llm-compressor +
    transformers >= 5.17.0) that the shared training venv does not carry.
    """
    argv = [
        awq_py,
        str(_AWQ_ONESHOT_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--calibration-file",
        str(calibration_file),
        "--out-dir",
        str(out_dir),
        "--num-calibration-samples",
        str(num_calibration_samples),
    ]
    code, output = run(argv, timeout)
    if code != 0:
        raise QuantizeError(f"INT4 AWQ export failed (exit {code}): {output}")
    return output


def copy_vllm_support_files(model_dir: Path, out_dir: Path) -> list[str]:
    """Copy each of :data:`VLLM_SUPPORT_FILES` from *model_dir* to *out_dir* when present.

    ``save_pretrained(save_compressed=True)`` does not write these, but vLLM
    needs them to serve the export. Symlinks are resolved (a Hugging Face
    snapshot dir is a tree of symlinks into a shared blob store) so *out_dir*
    stays self-contained. Returns the names actually copied.
    """
    copied = []
    for name in VLLM_SUPPORT_FILES:
        src = model_dir / name
        if src.is_file():  # is_file() follows symlinks; a dangling one is skipped
            shutil.copyfile(src.resolve(), out_dir / name)
            copied.append(name)
    return copied


def finish_awq_export(model_dir: Path, out_dir: Path) -> dict:
    """Post-processing after ``awq_oneshot.py`` saves *out_dir* (risk r14).

    Copies the vLLM-serving files the AWQ save does not write, then writes a
    greedy-decoding ``generation_config.json`` (deviation d3) via
    ``gen_config.py``, imported by path. Returns what the run record needs:
    the files copied, and the vLLM serve flag nvsh's launcher cannot pass on
    its own -- recorded as a known limitation rather than silently dropped.
    """
    copied = copy_vllm_support_files(model_dir, out_dir)
    gen_config = _sibling("gen_config")
    gen_config.write(out_dir)
    return {"copied_files": copied, "serve_args": list(AWQ_SERVE_ARGS)}


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


def llama_cpp_commit(run: RunFn, env: Mapping[str, str], timeout: float = VERSION_TIMEOUT) -> str:
    """llama.cpp's checked-out commit, from ``LLAMA_CPP_DIR``'s git HEAD.

    ``"unknown (...)"`` when the operator did not set the env var, or when
    the git call itself fails -- never invented.
    """
    directory = env.get(LLAMA_CPP_DIR_VAR)
    if not directory:
        return f"unknown ({LLAMA_CPP_DIR_VAR} not set)"
    code, output = run(["git", "-C", directory, "rev-parse", "HEAD"], timeout)
    output = output.strip()
    if code != 0:
        return f"unknown ({output or code})"
    return output


def awq_tool_versions(run: RunFn, awq_py: str, timeout: float = VERSION_TIMEOUT) -> dict[str, str]:
    """llm-compressor's and transformers' versions, read from the AWQ venv itself."""
    code, output = run([awq_py, "-c", _AWQ_VERSION_PROBE], timeout)
    output = output.strip()
    if code != 0:
        unknown = f"unknown ({output or code})"
        return {"llm-compressor": unknown, "transformers": unknown}
    parts = output.split()
    if len(parts) != 2:
        unknown = f"unknown (unexpected output: {output!r})"
        return {"llm-compressor": unknown, "transformers": unknown}
    compressor, transformers = parts
    return {"llm-compressor": compressor, "transformers": transformers}


def record_tool_versions(
    run: RunFn,
    tools: ToolPaths,
    env: Mapping[str, str],
    awq_py: str,
    timeout: float = VERSION_TIMEOUT,
) -> dict[str, str]:
    """Every tool's version, keyed by name, for the run log."""
    versions = {
        "llama.cpp convert": tool_version(run, tools.convert, timeout),
        "llama.cpp imatrix": tool_version(run, tools.imatrix, timeout),
        "llama.cpp quantize": tool_version(run, tools.quantize, timeout),
        "llama.cpp commit": llama_cpp_commit(run, env, timeout),
    }
    versions.update(awq_tool_versions(run, awq_py, timeout))
    return versions


# ---------------------------------------------------------------------------
# heal_needed (decisions c42, c43)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantSummary:
    """What ``heal_needed`` compares between a bf16 build and a quantized one.

    ``wrong_mutating_ids`` is the *set* of entry ids with a wrong mutating
    proposal, not a count: two builds can have the same count of wrong
    mutating proposals while disagreeing on which entries they are wrong on
    (one build fixes an entry another build breaks), and only the set catches
    that (Codex finding #5).
    """

    right_pct: float
    wrong_mutating_ids: frozenset[str]


def heal_needed(bf16: QuantSummary, quant: QuantSummary) -> bool:
    """True when *quant* has drifted enough from *bf16* to need a healing fine-tune.

    Two independent triggers (decisions c42, c43), either is sufficient:
    right proposals drop by MORE than :data:`HEAL_MARGIN_POINTS` percentage
    points, or *quant* has a wrong mutating proposal on an entry id *bf16*
    did not have one on. Comparing counts alone misses this: quantization
    fixing one entry while breaking another leaves the count unchanged but
    still needs healing, since it is a new failure, not the old one
    persisting. A drop of exactly the margin, or a wrong-mutating id set
    that only loses members (no new ones), does not trigger healing on its
    own.
    """
    if bf16.right_pct - quant.right_pct > HEAL_MARGIN_POINTS:
        return True
    if quant.wrong_mutating_ids - bf16.wrong_mutating_ids:
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
        awq_py = awq_python_from_env(os.environ)
        texts = build_calibration_set(args.train, args.val, args.test, args.calibration_limit)
    except QuantizeError as exc:
        parser.error(str(exc))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    calibration_file = write_calibration_file(texts, args.work_dir / "calibration.txt")

    run = default_run
    gguf_bf16 = args.work_dir / "model-bf16.gguf"
    imatrix_file = args.work_dir / "imatrix.dat"
    q4_k_m = args.work_dir / "model-q4_k_m.gguf"
    awq_dir = args.work_dir / "awq"

    try:
        # Written on the source dir BEFORE conversion (deviation d3), so the
        # GGUF picks up the sampling defaults if the converter reads them --
        # ordered that way, never assumed of a specific converter version.
        _sibling("gen_config").write(args.model_dir)
        convert_gguf(run, tools, args.model_dir, gguf_bf16)
        compute_imatrix(run, tools, gguf_bf16, calibration_file, imatrix_file)
        quantize_q4_k_m(run, tools, gguf_bf16, imatrix_file, q4_k_m)
        export_awq(run, awq_py, args.model_dir, calibration_file, awq_dir, len(texts))
        awq_result = finish_awq_export(args.model_dir, awq_dir)
    except QuantizeError as exc:
        parser.error(str(exc))

    versions = record_tool_versions(run, tools, os.environ, awq_py)
    run_log = {
        "calibration_entries": len(texts),
        "gguf_q4_k_m": str(q4_k_m),
        "awq_dir": str(awq_dir),
        "awq_copied_files": awq_result["copied_files"],
        "awq_serve_args": awq_result["serve_args"],
        "awq_serve_args_note": "nvsh's Tier 2 launcher cannot pass these; set them by hand",
        "tool_versions": versions,
    }
    (args.work_dir / "quantize-run.json").write_text(
        json.dumps(run_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_log, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
