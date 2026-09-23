"""scripts/lfm-finetune/quantize.py (issue 46, task t12): quantization + healing tooling.

Covers c44 (train-side-only calibration, GGUF/imatrix/Q4_K_M and INT4 AWQ export as
subprocesses with paths from env, tool versions recorded) and h27 (heal_needed and the
calibration-set refusal of any val/test source_id). External tools are never invoked here:
every subprocess seam takes an injected fake ``run``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "quantize.py"


def _module():
    spec = importlib.util.spec_from_file_location("lfm_quantize", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lfm_quantize"] = module  # dataclass field resolution needs the module registered
    spec.loader.exec_module(module)
    return module


def _write_split(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"header": "h", "entries": entries}), encoding="utf-8")
    return path


def _entry(entry_id: str, source_id: str | None = None, text: str = "hi") -> dict:
    entry = {"id": entry_id, "text": text, "expect": {"operation": "thermal_stats"}}
    if source_id is not None:
        entry["source_id"] = source_id
    return entry


# ---------------------------------------------------------------------------
# Calibration set: train-side only (h27)
# ---------------------------------------------------------------------------


def test_calibration_set_uses_only_train_text(tmp_path) -> None:
    module = _module()
    train = _write_split(tmp_path / "train.json", [_entry("t1", text="train text")])
    val = _write_split(tmp_path / "val.json", [_entry("v1", text="val text")])
    test = _write_split(tmp_path / "test.json", [_entry("s1", text="test text")])
    texts = module.build_calibration_set(train, val, test)
    assert texts == ["train text"]


def test_calibration_set_refuses_a_val_source_id_in_train(tmp_path) -> None:
    module = _module()
    # A hand-edited train file that reintroduces a val source_id (h27's belt-and-braces check).
    train = _write_split(tmp_path / "train.json", [_entry("t1", source_id="shared")])
    val = _write_split(tmp_path / "val.json", [_entry("v1", source_id="shared")])
    test = _write_split(tmp_path / "test.json", [])
    with pytest.raises(module.QuantizeError, match="train-side only"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_refuses_a_test_source_id_in_train(tmp_path) -> None:
    module = _module()
    train = _write_split(tmp_path / "train.json", [_entry("t1", source_id="shared")])
    val = _write_split(tmp_path / "val.json", [])
    test = _write_split(tmp_path / "test.json", [_entry("s1", source_id="shared")])
    with pytest.raises(module.QuantizeError, match="train-side only"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_respects_a_limit(tmp_path) -> None:
    module = _module()
    train = _write_split(
        tmp_path / "train.json", [_entry(f"t{i}", text=f"text{i}") for i in range(5)]
    )
    val = _write_split(tmp_path / "val.json", [])
    test = _write_split(tmp_path / "test.json", [])
    texts = module.build_calibration_set(train, val, test, limit=2)
    assert texts == ["text0", "text1"]


def test_write_calibration_file_writes_one_text_per_line(tmp_path) -> None:
    module = _module()
    out = module.write_calibration_file(["a", "b"], tmp_path / "calib.txt")
    assert out.read_text(encoding="utf-8") == "a\nb\n"


def test_write_calibration_file_handles_no_texts(tmp_path) -> None:
    module = _module()
    out = module.write_calibration_file([], tmp_path / "calib.txt")
    assert out.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Tool paths from env (c44: never hard-coded)
# ---------------------------------------------------------------------------


def _env(**overrides) -> dict:
    base = {
        "LLAMA_CPP_CONVERT": "/opt/llama/convert.py",
        "LLAMA_CPP_QUANTIZE": "/opt/llama/quantize",
        "LLAMA_CPP_IMATRIX": "/opt/llama/imatrix",
        "LLM_COMPRESSOR": "/opt/venv/bin/llm-compressor",
    }
    base.update(overrides)
    return base


def test_tool_paths_from_env_reads_all_four() -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    assert tools.convert == "/opt/llama/convert.py"
    assert tools.quantize == "/opt/llama/quantize"
    assert tools.imatrix == "/opt/llama/imatrix"
    assert tools.compressor == "/opt/venv/bin/llm-compressor"


def test_tool_paths_from_env_refuses_a_missing_variable() -> None:
    module = _module()
    env = _env()
    del env["LLAMA_CPP_QUANTIZE"]
    with pytest.raises(module.QuantizeError, match="LLAMA_CPP_QUANTIZE"):
        module.tool_paths_from_env(env)


# ---------------------------------------------------------------------------
# GGUF conversion, imatrix, Q4_K_M, INT4 AWQ: subprocess seams, no real tools
# ---------------------------------------------------------------------------


def _recording_run(returncode: int = 0, output: str = "ok", side_effect=None):
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: float) -> tuple[int, str]:
        calls.append(argv)
        if side_effect is not None:
            side_effect(argv)
        return (returncode, output)

    return run, calls


def test_convert_gguf_calls_the_env_configured_tool(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, calls = _recording_run()
    out_file = tmp_path / "model-f16.gguf"
    module.convert_gguf(run, tools, tmp_path / "hf-model", out_file)
    assert calls[0][0] == tools.convert
    assert str(out_file) in calls[0]


def test_convert_gguf_raises_on_nonzero_exit(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, _ = _recording_run(returncode=1, output="boom")
    with pytest.raises(module.QuantizeError, match="boom"):
        module.convert_gguf(run, tools, tmp_path / "hf-model", tmp_path / "out.gguf")


def test_convert_gguf_refuses_a_vision_projector_output(tmp_path) -> None:
    """text-only, no mmproj: a tool that writes one anyway is a refusal, not silent success."""
    module = _module()
    tools = module.tool_paths_from_env(_env())
    out_file = tmp_path / "model-f16.gguf"

    def plants_mmproj(argv: list[str]) -> None:
        (tmp_path / "mmproj-model-f16.gguf").write_bytes(b"x")

    run, _ = _recording_run(side_effect=plants_mmproj)
    with pytest.raises(module.QuantizeError, match="mmproj"):
        module.convert_gguf(run, tools, tmp_path / "hf-model", out_file)


def test_compute_imatrix_calls_the_env_configured_tool(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, calls = _recording_run()
    module.compute_imatrix(
        run, tools, tmp_path / "model-f16.gguf", tmp_path / "calib.txt", tmp_path / "im.dat"
    )
    assert calls[0][0] == tools.imatrix


def test_compute_imatrix_raises_on_nonzero_exit(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, _ = _recording_run(returncode=2, output="bad calib")
    with pytest.raises(module.QuantizeError, match="bad calib"):
        module.compute_imatrix(
            run, tools, tmp_path / "model-f16.gguf", tmp_path / "calib.txt", tmp_path / "im.dat"
        )


def test_quantize_q4_k_m_calls_the_env_configured_tool_with_that_quant_name(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, calls = _recording_run()
    module.quantize_q4_k_m(
        run, tools, tmp_path / "model-f16.gguf", tmp_path / "im.dat", tmp_path / "model-q4km.gguf"
    )
    assert calls[0][0] == tools.quantize
    assert "Q4_K_M" in calls[0]


def test_quantize_q4_k_m_raises_on_nonzero_exit(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, _ = _recording_run(returncode=1, output="bad imatrix")
    with pytest.raises(module.QuantizeError, match="bad imatrix"):
        module.quantize_q4_k_m(
            run,
            tools,
            tmp_path / "model-f16.gguf",
            tmp_path / "im.dat",
            tmp_path / "model-q4km.gguf",
        )


def test_export_awq_calls_the_env_configured_tool(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, calls = _recording_run()
    module.export_awq(run, tools, tmp_path / "hf-model", tmp_path / "calib.txt", tmp_path / "awq")
    assert calls[0][0] == tools.compressor


def test_export_awq_raises_on_nonzero_exit(tmp_path) -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, _ = _recording_run(returncode=1, output="unsupported layer")
    with pytest.raises(module.QuantizeError, match="unsupported layer"):
        module.export_awq(
            run, tools, tmp_path / "hf-model", tmp_path / "calib.txt", tmp_path / "awq"
        )


# ---------------------------------------------------------------------------
# Tool version recording (c44)
# ---------------------------------------------------------------------------


def test_tool_version_reports_the_run_output() -> None:
    module = _module()
    run, _ = _recording_run(output="convert.py version 1.2.3")
    assert module.tool_version(run, "/opt/llama/convert.py") == "convert.py version 1.2.3"


def test_tool_version_reports_unknown_on_failure() -> None:
    module = _module()
    run, _ = _recording_run(returncode=127, output="not found")
    assert "unknown" in module.tool_version(run, "/opt/llama/missing")


def test_record_tool_versions_calls_all_four_tools() -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, calls = _recording_run(output="v1")
    versions = module.record_tool_versions(run, tools)
    assert len(calls) == 4
    assert all(value == "v1" for value in versions.values())
    assert len(versions) == 4


# ---------------------------------------------------------------------------
# heal_needed (c42, c43)
# ---------------------------------------------------------------------------


def test_heal_needed_false_when_quant_matches_bf16() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating=0)
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating=0)
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_right_proposals_drop_more_than_3_points() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating=0)
    quant = module.QuantSummary(right_pct=86.9, wrong_mutating=0)
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_false_at_exactly_the_3_point_margin() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating=0)
    quant = module.QuantSummary(right_pct=87.0, wrong_mutating=0)
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_a_new_wrong_mutating_proposal_appears() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating=0)
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating=1)
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_false_when_wrong_mutating_does_not_increase() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating=1)
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating=1)
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_right_proposals_improve_but_wrong_mutating_appears() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=80.0, wrong_mutating=0)
    quant = module.QuantSummary(right_pct=95.0, wrong_mutating=1)
    assert module.heal_needed(bf16, quant) is True
