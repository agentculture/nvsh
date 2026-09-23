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


def _write_split(path: Path, side: str, entries: list[dict], header: str | None = None) -> Path:
    """Write a split file with a ``split.py``-shaped header naming *side*."""
    if header is None:
        header = f"Split '{side}' of corpus.json (seed=1)."
    path.write_text(json.dumps({"header": header, "entries": entries}), encoding="utf-8")
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
    train = _write_split(tmp_path / "train.json", "train", [_entry("t1", text="train text")])
    val = _write_split(tmp_path / "val.json", "val", [_entry("v1", text="val text")])
    test = _write_split(tmp_path / "test.json", "test", [_entry("s1", text="test text")])
    texts = module.build_calibration_set(train, val, test)
    assert texts == ["train text"]


def test_calibration_set_refuses_a_val_source_id_in_train(tmp_path) -> None:
    module = _module()
    # A hand-edited train file that reintroduces a val source_id (h27's belt-and-braces check).
    train = _write_split(tmp_path / "train.json", "train", [_entry("t1", source_id="shared")])
    val = _write_split(tmp_path / "val.json", "val", [_entry("v1", source_id="shared")])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="train-side only"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_refuses_a_test_source_id_in_train(tmp_path) -> None:
    module = _module()
    train = _write_split(tmp_path / "train.json", "train", [_entry("t1", source_id="shared")])
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [_entry("s1", source_id="shared")])
    with pytest.raises(module.QuantizeError, match="train-side only"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_respects_a_limit(tmp_path) -> None:
    module = _module()
    train = _write_split(
        tmp_path / "train.json", "train", [_entry(f"t{i}", text=f"text{i}") for i in range(5)]
    )
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    texts = module.build_calibration_set(train, val, test, limit=2)
    assert texts == ["text0", "text1"]


def test_calibration_set_refuses_swapped_train_and_val(tmp_path) -> None:
    """Codex finding #3: --train/--val swapped, but otherwise ordinary disjoint splits."""
    module = _module()
    # These are ordinary, mutually disjoint splits: the swap is purely in which
    # path is passed as --train and which as --val, so a disjointness check alone
    # cannot catch it. The header on each file names its real side.
    real_train = _write_split(tmp_path / "train.json", "train", [_entry("t1", text="train text")])
    real_val = _write_split(tmp_path / "val.json", "val", [_entry("v1", text="val text")])
    test = _write_split(tmp_path / "test.json", "test", [_entry("s1", text="test text")])
    with pytest.raises(module.QuantizeError, match="val.*expected 'train'"):
        # --train is given val.json, --val is given train.json.
        module.build_calibration_set(real_val, real_train, test)


def test_calibration_set_refuses_a_file_whose_header_names_the_wrong_side(tmp_path) -> None:
    module = _module()
    train = _write_split(tmp_path / "train.json", "test", [_entry("t1")])
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="expected 'train'"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_refuses_the_held_out_split_by_file_name(tmp_path) -> None:
    module = _module()
    train = _write_split(
        tmp_path / "held-out.json", "train", [_entry("t1")]
    )  # header lies about its side; the file name alone must refuse it
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="held-out"):
        module.build_calibration_set(train, val, test)


def test_calibration_set_refuses_the_held_out_split_by_header(tmp_path) -> None:
    module = _module()
    train = _write_split(
        tmp_path / "train.json", "train", [_entry("t1")], header="Held-out split of corpus.json."
    )
    val = _write_split(tmp_path / "val.json", "val", [])
    test = _write_split(tmp_path / "test.json", "test", [])
    with pytest.raises(module.QuantizeError, match="held-out"):
        module.build_calibration_set(train, val, test)


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
    }
    base.update(overrides)
    return base


def test_tool_paths_from_env_reads_all_three() -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    assert tools.convert == "/opt/llama/convert.py"
    assert tools.quantize == "/opt/llama/quantize"
    assert tools.imatrix == "/opt/llama/imatrix"


def test_tool_paths_from_env_refuses_a_missing_variable() -> None:
    module = _module()
    env = _env()
    del env["LLAMA_CPP_QUANTIZE"]
    with pytest.raises(module.QuantizeError, match="LLAMA_CPP_QUANTIZE"):
        module.tool_paths_from_env(env)


# ---------------------------------------------------------------------------
# AWQ venv python from env (risk r14: a separate venv, never the training one)
# ---------------------------------------------------------------------------


def test_awq_python_from_env_returns_the_path() -> None:
    module = _module()
    assert module.awq_python_from_env({"AWQ_PY": "/opt/awq/bin/python"}) == "/opt/awq/bin/python"


def test_awq_python_from_env_refuses_when_unset() -> None:
    module = _module()
    with pytest.raises(module.QuantizeError, match="AWQ_PY"):
        module.awq_python_from_env({})


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
    out_file = tmp_path / "model-bf16.gguf"
    module.convert_gguf(run, tools, tmp_path / "hf-model", out_file)
    assert calls[0][0] == tools.convert
    assert str(out_file) in calls[0]


def test_convert_gguf_requests_bf16_not_f16(tmp_path) -> None:
    """Risk r14 / t16 spike: bf16, never f16."""
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, calls = _recording_run()
    module.convert_gguf(run, tools, tmp_path / "hf-model", tmp_path / "model-bf16.gguf")
    assert "bf16" in calls[0]
    assert "f16" not in calls[0]


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


def test_export_awq_invokes_awq_py_with_the_oneshot_script(tmp_path) -> None:
    """Risk r14: a subprocess of AWQ_PY running awq_oneshot.py, never an llm-compressor CLI."""
    module = _module()
    run, calls = _recording_run()
    model_dir, calib_file, out_dir = tmp_path / "hf-model", tmp_path / "calib.txt", tmp_path / "awq"
    module.export_awq(run, "/opt/awq/bin/python", model_dir, calib_file, out_dir, 128)
    assert calls[0][0] == "/opt/awq/bin/python"
    assert calls[0][1] == str(module._AWQ_ONESHOT_SCRIPT)
    assert calls[0][1].endswith("awq_oneshot.py")
    assert str(model_dir) in calls[0]
    assert str(calib_file) in calls[0]
    assert str(out_dir) in calls[0]
    assert "128" in calls[0]


def test_export_awq_raises_on_nonzero_exit(tmp_path) -> None:
    module = _module()
    run, _ = _recording_run(returncode=1, output="unsupported layer")
    with pytest.raises(module.QuantizeError, match="unsupported layer"):
        module.export_awq(
            run,
            "/opt/awq/bin/python",
            tmp_path / "hf-model",
            tmp_path / "calib.txt",
            tmp_path / "awq",
            128,
        )


# ---------------------------------------------------------------------------
# vLLM support files + generation_config.json after the AWQ save (risk r14)
# ---------------------------------------------------------------------------


def test_copy_vllm_support_files_copies_present_files_resolving_symlinks(tmp_path) -> None:
    module = _module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "awq"
    out_dir.mkdir()
    real = tmp_path / "blob-tokenizer.json"
    real.write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").symlink_to(real)
    (model_dir / "vocab.json").write_text("{}", encoding="utf-8")

    copied = module.copy_vllm_support_files(model_dir, out_dir)

    assert set(copied) == {"tokenizer.json", "vocab.json"}
    assert not (out_dir / "tokenizer.json").is_symlink()
    assert (out_dir / "tokenizer.json").read_text(encoding="utf-8") == "{}"


def test_copy_vllm_support_files_skips_files_not_present(tmp_path) -> None:
    module = _module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "awq"
    out_dir.mkdir()
    (model_dir / "merges.txt").write_text("a b", encoding="utf-8")

    copied = module.copy_vllm_support_files(model_dir, out_dir)

    assert copied == ["merges.txt"]
    assert not (out_dir / "tokenizer.json").exists()


def test_finish_awq_export_writes_generation_config_and_reports_serve_args(tmp_path) -> None:
    module = _module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "awq"
    out_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    result = module.finish_awq_export(model_dir, out_dir)

    assert result["copied_files"] == ["tokenizer_config.json"]
    assert result["serve_args"] == module.AWQ_SERVE_ARGS
    gen_config_path = out_dir / "generation_config.json"
    assert gen_config_path.is_file()
    payload = json.loads(gen_config_path.read_text(encoding="utf-8"))
    assert payload["temperature"] == 0.0
    assert payload["do_sample"] is False


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


def test_llama_cpp_commit_reads_git_head_when_dir_is_set() -> None:
    module = _module()
    run, calls = _recording_run(output="abc123\n")
    commit = module.llama_cpp_commit(run, {"LLAMA_CPP_DIR": "/opt/llama.cpp"})
    assert commit == "abc123"
    assert calls[0] == ["git", "-C", "/opt/llama.cpp", "rev-parse", "HEAD"]


def test_llama_cpp_commit_is_unknown_when_dir_is_not_set() -> None:
    module = _module()
    run, calls = _recording_run()
    commit = module.llama_cpp_commit(run, {})
    assert "unknown" in commit
    assert calls == []  # never runs git without a directory to point it at


def test_llama_cpp_commit_is_unknown_on_a_failed_git_call() -> None:
    module = _module()
    run, _ = _recording_run(returncode=128, output="not a git repository")
    commit = module.llama_cpp_commit(run, {"LLAMA_CPP_DIR": "/not/a/repo"})
    assert "unknown" in commit
    assert "not a git repository" in commit


def test_awq_tool_versions_parses_the_probe_output() -> None:
    module = _module()
    run, calls = _recording_run(output="0.14.0 5.17.0\n")
    versions = module.awq_tool_versions(run, "/opt/awq/bin/python")
    assert versions == {"llm-compressor": "0.14.0", "transformers": "5.17.0"}
    assert calls[0][0] == "/opt/awq/bin/python"
    assert calls[0][1] == "-c"


def test_awq_tool_versions_is_unknown_on_failure() -> None:
    module = _module()
    run, _ = _recording_run(returncode=1, output="ModuleNotFoundError")
    versions = module.awq_tool_versions(run, "/opt/awq/bin/python")
    assert "unknown" in versions["llm-compressor"]
    assert "unknown" in versions["transformers"]


def test_record_tool_versions_covers_llama_cpp_and_awq() -> None:
    module = _module()
    tools = module.tool_paths_from_env(_env())
    run, calls = _recording_run(output="v1 v2")
    versions = module.record_tool_versions(
        run, tools, {"LLAMA_CPP_DIR": "/opt/llama.cpp"}, "/opt/awq/bin/python"
    )
    # 3 llama.cpp --version calls + 1 git rev-parse + 1 AWQ venv probe.
    assert len(calls) == 5
    assert set(versions) == {
        "llama.cpp convert",
        "llama.cpp imatrix",
        "llama.cpp quantize",
        "llama.cpp commit",
        "llm-compressor",
        "transformers",
    }


# ---------------------------------------------------------------------------
# heal_needed (c42, c43)
# ---------------------------------------------------------------------------


def test_heal_needed_false_when_quant_matches_bf16() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_right_proposals_drop_more_than_3_points() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=86.9, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_false_at_exactly_the_3_point_margin() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=87.0, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_a_new_wrong_mutating_proposal_appears() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"b"}))
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_false_when_wrong_mutating_ids_do_not_change() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_false_when_a_wrong_mutating_id_is_fixed_and_none_added() -> None:
    """Fewer wrong mutating proposals, no new ones: the count drops, no healing needed."""
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset())
    assert module.heal_needed(bf16, quant) is False


def test_heal_needed_true_when_right_proposals_improve_but_wrong_mutating_appears() -> None:
    module = _module()
    bf16 = module.QuantSummary(right_pct=80.0, wrong_mutating_ids=frozenset())
    quant = module.QuantSummary(right_pct=95.0, wrong_mutating_ids=frozenset({"b"}))
    assert module.heal_needed(bf16, quant) is True


def test_heal_needed_true_when_the_wrong_mutating_set_changes_with_equal_counts() -> None:
    """Codex finding #5: bf16 gets A wrong, quant fixes A but breaks B -- counts tie at 1."""
    module = _module()
    bf16 = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"a"}))
    quant = module.QuantSummary(right_pct=90.0, wrong_mutating_ids=frozenset({"b"}))
    assert module.heal_needed(bf16, quant) is True
