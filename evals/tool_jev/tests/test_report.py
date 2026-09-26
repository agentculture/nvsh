"""Tests for ``evals/tool_jev/report.py`` (task t18, issue #64).

Builds a small, synthetic run dir (manifest + traces + real
``metrics_bridge.compute()`` output) under ``tmp_path`` and checks:

- the issue-64 table columns render for a candidate/baseline pair (model-only
  vs. model+harness) and a reference row;
- the page and JSON never contain a fixture's request/argument text
  (acceptance criterion 2);
- regenerating from the same run dir is byte-identical (acceptance
  criterion 1);
- absent permutation/judge inputs render as "not run" rather than raising;
- (if ``markdownlint-cli2`` is on PATH) the generated page lints clean.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evals.tool_jev import metrics_bridge as bridge
from evals.tool_jev import report
from evals.tool_jev import trace as trace_mod

SENTINEL = "zzq-sentinel-do-not-leak-4f9c2b1e"

ESC = "(escalate)"


def _op(operation_name: str, **args) -> dict:
    return {"operation": operation_name, "args": dict(args)}


def _pred_row(
    entry_id: str,
    expected: dict,
    outcome: str,
    operation: str | None = None,
    arguments: dict | None = None,
    candidates: dict | None = None,
) -> dict:
    return {
        "id": entry_id,
        "expected": expected,
        "outcome": outcome,
        "operation": operation,
        "arguments": arguments,
        "candidates": candidates,
        "tokens": 5,
        "ttfd_ms": 40.0,
        "latency_ms": 50.0,
    }


def _predictions(metrics_mod, rows: list[dict]):
    return [metrics_mod.Prediction.from_dict(r) for r in rows]


@pytest.fixture(scope="module")
def metrics_mod():
    return bridge.load_metrics_module()


@pytest.fixture(scope="module")
def gate_mod():
    return bridge.load_gate_module()


def _candidate_rows(with_distribution: bool = True) -> list[dict]:
    """A handful of prediction rows for one subject: mix of read-only/mutating,
    escalate/explain, and one missing-candidate row."""
    rows = [
        _pred_row(
            "c1",
            _op("memory_stats"),
            "propose",
            operation="memory_stats",
            arguments={},
            candidates=(
                {"memory_stats": 0.8, "service_status": 0.2} if with_distribution else None
            ),
        ),
        _pred_row(
            "c2",
            _op("service_restart", name="x"),
            "propose",
            operation="service_restart",
            arguments={"name": "x"},
            candidates=(
                {"service_restart": 0.4, "service_status": 0.3, ESC: 0.3}
                if with_distribution
                else None
            ),
        ),
        _pred_row(
            "c3",
            {"escalate": True},
            "escalate",
            candidates=({ESC: 0.9, "memory_stats": 0.1} if with_distribution else None),
        ),
        _pred_row(
            "c4-nocand",
            _op("service_logs", name="y"),
            "escalate",
            candidates=({ESC: 0.7, "memory_stats": 0.3} if with_distribution else None),
        ),
        _pred_row(
            "c5",
            _op("container_restart", name="z"),
            "abstain_uncertain",
            candidates=(
                {"container_restart": 0.34, "memory_stats": 0.33, "service_status": 0.33}
                if with_distribution
                else None
            ),
        ),
    ]
    return rows


def _write_metrics(
    run_dir: Path, metrics_mod, gate_mod, subject: str, policy: str, rows: list[dict]
):
    predictions = _predictions(metrics_mod, rows)
    output = bridge.compute(predictions, metrics_mod=metrics_mod, gate_mod=gate_mod)
    metrics_dir = run_dir / report.METRICS_DIRNAME
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = report.metrics_path(run_dir, subject, policy)
    path.write_text(json.dumps(output), encoding="utf-8")
    return output


def _write_traces(run_dir: Path, subject: str, rows: list[dict], sentinel_in_args: bool = False):
    traces = []
    for row in rows:
        arguments = dict(row.get("arguments") or {})
        if sentinel_in_args and arguments:
            arguments = {k: SENTINEL for k in arguments}
        raw = trace_mod.RawRecord.from_provider_answer(
            provider="fake",
            model="fake-model",
            returned_model="fake-model-v1",
            interface="tool_call",
            outcome=row["outcome"],
            operation=row["operation"],
            arguments=arguments or None,
            candidates=row["candidates"],
        )
        ground_truth = dict(row["expected"])
        if sentinel_in_args and "args" in ground_truth:
            ground_truth = {**ground_truth, "args": {k: SENTINEL for k in ground_truth["args"]}}
        traces.append(
            trace_mod.Trace(
                case_id=row["id"],
                split="test",
                raw=raw,
                ground_truth=ground_truth,
                subject=subject,
            )
        )
    traces_dir = run_dir / report.TRACES_DIRNAME
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_mod.write_traces(report.traces_path(run_dir, subject), traces)


def _write_manifest(run_dir: Path, run_id: str = "test-run-1", date: str = "2026-09-26"):
    manifest = {
        "run_id": run_id,
        "date": date,
        "subjects": [
            {"name": "cand-a", "kind": "candidate", "policies": ["raw", "scorer-r3b-shipped"]},
            {"name": "base-1", "kind": "baseline", "policies": ["raw"]},
            {"name": "gpt-ref", "kind": "reference", "policies": ["raw"]},
        ],
    }
    (run_dir / report.MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def basic_run_dir(tmp_path, metrics_mod, gate_mod):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_manifest(run_dir)

    cand_rows = _candidate_rows()
    _write_metrics(run_dir, metrics_mod, gate_mod, "cand-a", "raw", cand_rows)
    _write_metrics(run_dir, metrics_mod, gate_mod, "cand-a", "scorer-r3b-shipped", cand_rows)
    _write_traces(run_dir, "cand-a", cand_rows)

    base_rows = _candidate_rows()
    _write_metrics(run_dir, metrics_mod, gate_mod, "base-1", "raw", base_rows)
    _write_traces(run_dir, "base-1", base_rows)

    ref_rows = _candidate_rows(with_distribution=False)
    _write_metrics(run_dir, metrics_mod, gate_mod, "gpt-ref", "raw", ref_rows)
    _write_traces(run_dir, "gpt-ref", ref_rows)

    return run_dir


def test_manifest_loads(basic_run_dir):
    manifest = report.load_manifest(basic_run_dir)
    assert manifest.run_id == "test-run-1"
    assert manifest.date == "2026-09-26"
    assert [s.name for s in manifest.subjects] == ["cand-a", "base-1", "gpt-ref"]
    assert manifest.subjects[0].policies == ("raw", "scorer-r3b-shipped")


def test_result_has_model_only_and_model_harness_rows(basic_run_dir):
    result = report.build_result(basic_run_dir)
    variants = [(row["subject"], row["variant"], row["harness_policy"]) for row in result["rows"]]
    assert ("cand-a", "model-only", None) in variants
    assert ("cand-a", "model+harness", "scorer-r3b-shipped") in variants
    assert ("base-1", "model-only", None) in variants
    # Reference rows are kept separate from candidates/baselines.
    assert all(row["subject"] != "gpt-ref" for row in result["rows"])
    assert [row["subject"] for row in result["reference_rows"]] == ["gpt-ref"]
    assert result["reference_rows"][0]["variant"] == "reference"


def test_measurable_rows_have_numeric_top1_ece_brier(basic_run_dir):
    result = report.build_result(basic_run_dir)
    row = next(r for r in result["rows"] if r["subject"] == "cand-a" and r["policy"] == "raw")
    assert row["measurable"] is True
    assert isinstance(row["top1"]["rate"], float)
    assert isinstance(row["ece"], float)
    assert isinstance(row["brier"], float)
    assert 0.0 <= row["coverage"]["rate"] <= 1.0
    assert row["wrong_mutations"] >= 0


def test_reference_without_distribution_is_not_measurable(basic_run_dir):
    result = report.build_result(basic_run_dir)
    row = result["reference_rows"][0]
    assert row["measurable"] is False
    assert row["top1"]["rate"] is None
    assert row["ece"] is None
    assert row["brier"] is None
    # Coverage/abstain/missing-candidate never require a distribution.
    assert row["coverage"]["rate"] is not None


def test_markdown_shows_nm_for_not_measurable(basic_run_dir):
    result = report.build_result(basic_run_dir)
    markdown = report.render_markdown(result)
    assert report.NM_MARKER in markdown
    assert report.NM_FOOTNOTE in markdown


def test_main_table_column_headers(basic_run_dir):
    result = report.build_result(basic_run_dir)
    markdown = report.render_markdown(result)
    assert (
        "| Variant | Harness policy | Top-1 | ECE | Brier | Coverage | Abstain P/R "
        "| Missing-candidate | Wrong mutations |" in markdown
    )


def test_absent_permutation_and_judge_render_not_run(basic_run_dir):
    assert report.load_permutation(basic_run_dir) is None
    assert report.load_judge_results(basic_run_dir) is None
    result = report.build_result(basic_run_dir)
    assert result["judge_panel"] == "not_run"
    for row in result["rows"] + result["reference_rows"]:
        assert row["slices"]["permutation"] == "not_run"
    markdown = report.render_markdown(result)
    assert "Judge panel: not run." in markdown
    assert "Permutation slice: not run." in markdown


def test_present_permutation_and_judge_are_reflected(basic_run_dir):
    (basic_run_dir / report.PERMUTATION_FILENAME).write_text(
        json.dumps({"cand-a__raw": {"n": 5, "delta_top1": -0.02}}), encoding="utf-8"
    )
    (basic_run_dir / report.JUDGE_RESULTS_FILENAME).write_text(
        json.dumps(
            {
                "judges": ["judge-a"],
                "results": [
                    {
                        "subject": "cand-a",
                        "policy": "raw",
                        "judge": "judge-a",
                        "verdict": "looks fine",
                        "score": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = report.build_result(basic_run_dir)
    row = next(r for r in result["rows"] if r["subject"] == "cand-a" and r["policy"] == "raw")
    assert row["slices"]["permutation"] == {"n": 5, "delta_top1": -0.02}
    assert result["judge_panel"]["judges"] == ["judge-a"]
    assert result["judge_panel"]["results"][0]["verdict"] == "looks fine"
    markdown = report.render_markdown(result)
    assert "judge-a" in markdown
    assert "looks fine" in markdown


def test_candidate_count_slice_present(basic_run_dir):
    result = report.build_result(basic_run_dir)
    row = next(r for r in result["rows"] if r["subject"] == "cand-a" and r["policy"] == "raw")
    bucket_keys = set(row["slices"]["candidate_count"].keys())
    # c1/c4-nocand have 2 offered, c2/c3 have 3, c5 has 3.
    assert bucket_keys == {"2", "3"}


def test_regeneration_is_byte_identical(basic_run_dir):
    result1, markdown1 = report.generate(basic_run_dir)
    result2, markdown2 = report.generate(basic_run_dir)
    assert json.dumps(result1) == json.dumps(result2)
    assert markdown1 == markdown2


def test_generate_writes_files(basic_run_dir, tmp_path):
    out_json = tmp_path / "result.json"
    out_md = tmp_path / "report.md"
    result, markdown = report.generate(basic_run_dir, result_path=out_json, markdown_path=out_md)
    assert out_json.read_text(encoding="utf-8").strip() == json.dumps(result, indent=2) + ""
    assert out_md.read_text(encoding="utf-8") == markdown


# ---------------------------------------------------------------------------
# Criterion 2: no case request text anywhere in the outputs.
# ---------------------------------------------------------------------------


def test_no_case_text_leaks_into_json_or_markdown(tmp_path, metrics_mod, gate_mod):
    run_dir = tmp_path / "run_sentinel"
    run_dir.mkdir()
    _write_manifest(run_dir)

    cand_rows = _candidate_rows()
    _write_metrics(run_dir, metrics_mod, gate_mod, "cand-a", "raw", cand_rows)
    _write_metrics(run_dir, metrics_mod, gate_mod, "cand-a", "scorer-r3b-shipped", cand_rows)
    # The trace's argument values and ground-truth args carry the sentinel,
    # standing in for real per-case request/argument text.
    _write_traces(run_dir, "cand-a", cand_rows, sentinel_in_args=True)

    base_rows = _candidate_rows()
    _write_metrics(run_dir, metrics_mod, gate_mod, "base-1", "raw", base_rows)
    _write_traces(run_dir, "base-1", base_rows, sentinel_in_args=True)

    ref_rows = _candidate_rows(with_distribution=False)
    _write_metrics(run_dir, metrics_mod, gate_mod, "gpt-ref", "raw", ref_rows)
    _write_traces(run_dir, "gpt-ref", ref_rows, sentinel_in_args=True)

    # Sanity: the sentinel really is present in the raw trace file (private,
    # per-case data), so the assertion below is meaningful.
    raw_trace_text = report.traces_path(run_dir, "cand-a").read_text(encoding="utf-8")
    assert SENTINEL in raw_trace_text

    result, markdown = report.generate(run_dir)
    result_text = json.dumps(result)

    assert SENTINEL not in result_text
    assert SENTINEL not in markdown

    # Case ids never appear in the committed page either (counts only).
    for case_id in ("c1", "c2", "c3", "c4-nocand", "c5"):
        assert case_id not in markdown


# ---------------------------------------------------------------------------
# markdownlint, best-effort (only if the binary is present).
# ---------------------------------------------------------------------------


def test_markdown_passes_markdownlint_if_available(basic_run_dir, tmp_path):
    binary = shutil.which("markdownlint-cli2")
    if binary is None:
        pytest.skip("markdownlint-cli2 not installed; lint manually before merging")
    result = report.build_result(basic_run_dir)
    markdown = report.render_markdown(result)
    md_path = tmp_path / "report.md"
    md_path.write_text(markdown, encoding="utf-8")
    # Use this repo's own markdownlint-cli2 config (e.g. MD013 line-length is
    # deliberately off here, per CLAUDE.md) rather than the tool's defaults.
    repo_config = Path(__file__).resolve().parents[3] / ".markdownlint-cli2.yaml"
    shutil.copy(repo_config, tmp_path / ".markdownlint-cli2.yaml")
    proc = subprocess.run(
        [binary, str(md_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
