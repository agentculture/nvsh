"""Artifact labels on the report (h1, t17): each candidate row names its exact artifact.

The runner writes ``artifact`` (the predictions file's sha256, plus the repo
id and revision when the manifest has them) into ``manifest.json``'s
subject entries; ``report.py`` carries it onto the row and lists it on the
page. A run dir without it renders as before. Synthetic run dirs only.
"""

from __future__ import annotations

import json

from evals.tool_jev import report

_BRIDGE = {
    "rows": [],
    "metrics_compute": {"outcome_counts": {"propose": 1}, "calibration": {}},
    "top_k_accuracy": {},
    "abstain": {},
    "missing_candidate": {"n": 0, "N": 1, "rate": 0.0},
    "wrong_mutating": {"total": 0},
}


def _run_dir(tmp_path, artifact):
    subject = {"name": "cand.test", "kind": "candidate", "policies": ["raw"]}
    if artifact is not None:
        subject["artifact"] = artifact
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "date": "2026-09-26", "subjects": [subject]})
    )
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "cand.test__raw.json").write_text(json.dumps(_BRIDGE))
    return tmp_path


def test_artifact_is_carried_onto_the_row_and_the_page(tmp_path):
    artifact = {"predictions_sha256": "ab" * 32, "repo_id": "org/cand", "revision": "rev1"}
    result, markdown = report.generate(_run_dir(tmp_path, artifact))
    assert result["rows"][0]["artifact"] == artifact
    assert "## Artifacts" in markdown
    assert "| cand.test | " + "ab" * 32 + " | org/cand | rev1 |" in markdown


def test_no_artifact_keeps_the_old_shape(tmp_path):
    result, markdown = report.generate(_run_dir(tmp_path, None))
    assert "artifact" not in result["rows"][0]
    assert "## Artifacts" not in markdown
