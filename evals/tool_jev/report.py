"""Issue-64 report: JSON result + markdown comparison page, generated from one run dir.

This module is a **pure function of the run directory's files**. It never
calls a model, never reads request text, and never writes a timestamp of its
own — the only date/id fields in the output are the ``run_id``/``date``
values the run's own ``manifest.json`` carries as data. Given the same run
dir, :func:`generate` (and :func:`build_result` / :func:`render_markdown`)
produce byte-identical output every time (acceptance criterion 1).

Run directory layout this module reads (never writes; some other task's
runner is responsible for producing it)::

    <run_dir>/manifest.json          -- see :func:`load_manifest`
    <run_dir>/traces/<subject>.jsonl -- evals.tool_jev.trace.write_traces output
    <run_dir>/metrics/<subject>__<policy>.json
                                      -- evals.tool_jev.metrics_bridge.compute()
                                         output, json.dump'd verbatim
    <run_dir>/permutation.json        -- OPTIONAL, see :func:`load_permutation`
    <run_dir>/judge_results.json      -- OPTIONAL, see :func:`load_judge_results`

``manifest.json`` shape (this module's own minimal contract — no sibling
task defines a run manifest yet, so this is defined here and reported per
COMMON2's rule about defining the minimal thing locally)::

    {
      "run_id": "<opaque string, data, not generated here>",
      "date": "<opaque string, data, not generated here>",
      "subjects": [
        {"name": "<subject name>", "kind": "candidate"|"baseline"|"reference",
         "policies": ["raw", "scorer-r3b-shipped", ...]}
      ]
    }

``policies.py``'s own ``"raw"`` policy (no calibration, no gate: the bare
argmax) is always the *model-only* variant; any other policy name on the
same subject is a *model+harness* variant, so a candidate/baseline can
answer "did the model improve" (raw vs. raw for two checkpoints) separately
from "did the harness prevent mistakes" (raw vs. shipped for one
checkpoint) -- issue 64's own framing. A ``"reference"`` subject is always
rendered as a reference row regardless of which policy name it carries
(hosted comparison models are not gated by nvsh's harness).

No case request text and no case id ever reaches :func:`build_result` or
:func:`render_markdown`'s output: every figure here is a count, a rate or a
mean pulled from ``metrics_bridge.compute()``'s own output (which itself
never carries request text, see its module docstring) plus, for the
candidate-count slice only, the *length* of a trace's offered-candidates
mapping (never its keys' text) joined back to a metrics row by id purely
in-memory -- the id itself is never written to the result.

Nothing under ``nvsh/`` may import this module. No network, no subprocess.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.tool_jev import trace as trace_mod
from evals.tool_jev.metrics_bridge import NOT_MEASURABLE

#: Policy name that always means "no calibration, no gate" (see policies.py).
MODEL_ONLY_POLICY = "raw"

#: The three variant labels a row can carry.
VARIANT_MODEL_ONLY = "model-only"
VARIANT_MODEL_HARNESS = "model+harness"
VARIANT_REFERENCE = "reference"

#: Subject kinds a manifest entry may carry.
SUBJECT_KINDS = ("candidate", "baseline", "reference")

#: The footnote marker used next to a not-measurable cell, and its text.
NM_MARKER = "n/m"
NM_FOOTNOTE = (
    "n/m = not measurable: the provider returned no token logprobs for any row, so no "
    "probability distribution exists to score Top-1 / ECE / Brier against."
)

#: Filenames this module reads from a run dir (relative to run_dir).
MANIFEST_FILENAME = "manifest.json"
TRACES_DIRNAME = "traces"
METRICS_DIRNAME = "metrics"
PERMUTATION_FILENAME = "permutation.json"
JUDGE_RESULTS_FILENAME = "judge_results.json"


class ReportError(ValueError):
    """A run directory's files did not match this module's contract."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectSpec:
    name: str
    kind: str
    policies: tuple[str, ...]
    #: The exact artifact this subject's answers came from (h1), when the
    #: runner recorded one: ``{"predictions_sha256", "repo_id", "revision"}``.
    artifact: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    date: str
    subjects: tuple[SubjectSpec, ...]


def _require_str(data: Mapping, key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ReportError(f"{where}: missing or empty required string field {key!r}")
    return value


def load_manifest(run_dir: str | Path) -> RunManifest:
    """Read and validate ``<run_dir>/manifest.json``."""
    path = Path(run_dir) / MANIFEST_FILENAME
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    run_id = _require_str(data, "run_id", "manifest")
    date = _require_str(data, "date", "manifest")
    subjects = []
    for i, raw in enumerate(data.get("subjects", [])):
        where = f"manifest.subjects[{i}]"
        name = _require_str(raw, "name", where)
        kind = _require_str(raw, "kind", where)
        if kind not in SUBJECT_KINDS:
            raise ReportError(f"{where}: unknown kind {kind!r} (must be one of {SUBJECT_KINDS})")
        policies_raw = raw.get("policies")
        if not isinstance(policies_raw, list) or not policies_raw:
            raise ReportError(f"{where}: 'policies' must be a non-empty list of policy names")
        artifact = raw.get("artifact")
        if artifact is not None and not isinstance(artifact, Mapping):
            raise ReportError(f"{where}: 'artifact' must be an object")
        subjects.append(
            SubjectSpec(
                name=name,
                kind=kind,
                policies=tuple(str(p) for p in policies_raw),
                artifact=None if artifact is None else dict(artifact),
            )
        )
    return RunManifest(run_id=run_id, date=date, subjects=tuple(subjects))


# ---------------------------------------------------------------------------
# Per-(subject, policy) metrics_bridge.compute() output
# ---------------------------------------------------------------------------


def metrics_path(run_dir: str | Path, subject: str, policy: str) -> Path:
    return Path(run_dir) / METRICS_DIRNAME / f"{subject}__{policy}.json"


def load_bridge_output(run_dir: str | Path, subject: str, policy: str) -> dict:
    """Read one ``metrics_bridge.compute()`` output back from disk, verbatim."""
    path = metrics_path(run_dir, subject, policy)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def traces_path(run_dir: str | Path, subject: str) -> Path:
    return Path(run_dir) / TRACES_DIRNAME / f"{subject}.jsonl"


def load_traces(run_dir: str | Path, subject: str) -> tuple[trace_mod.Trace, ...]:
    """Read ``<run_dir>/traces/<subject>.jsonl`` back via ``trace.read_traces``.

    Returns an empty tuple when the file does not exist -- a subject with no
    trace file simply cannot support the candidate-count slice, which then
    reports 0 rows rather than raising.
    """
    path = traces_path(run_dir, subject)
    if not path.exists():
        return ()
    return tuple(trace_mod.read_traces(path))


def load_permutation(run_dir: str | Path) -> dict | None:
    """Read ``<run_dir>/permutation.json``, or ``None`` when absent (criterion: 'not run').

    Shape: ``{"<subject>__<policy>": {...opaque, numeric-only permutation figures...}}``.
    No sibling task defines this file yet, so this module defines the
    minimal shape it needs and reports doing so.
    """
    path = Path(run_dir) / PERMUTATION_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_judge_results(run_dir: str | Path) -> dict | None:
    """Read ``<run_dir>/judge_results.json``, or ``None`` when absent ('not run').

    Shape: ``{"judges": [<judge name>, ...],
              "results": [{"subject":, "policy":, "judge":, "verdict":, "score":}]}``.
    Explain text only (never a release bar): rendered in its own section.
    """
    path = Path(run_dir) / JUDGE_RESULTS_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Row-level figures
# ---------------------------------------------------------------------------


def _variant_for(kind: str, policy: str) -> str:
    if kind == "reference":
        return VARIANT_REFERENCE
    return VARIANT_MODEL_ONLY if policy == MODEL_ONLY_POLICY else VARIANT_MODEL_HARNESS


def _harness_policy_for(kind: str, policy: str) -> str | None:
    """The 'Harness policy' column value: ``None`` means 'no harness applied'.

    ``None`` renders as ``-`` for a model-only row and ``n/a`` for a
    reference row (:func:`_render_harness_policy`); a model+harness row
    carries the policy's own name.
    """
    if kind == "reference":
        return None
    if policy == MODEL_ONLY_POLICY:
        return None
    return policy


def _topk_rate(bridge: Mapping[str, Any], k: int) -> dict:
    top_k = bridge.get("top_k_accuracy", {})
    # json round-trips int keys to strings.
    entry = top_k.get(str(k)) or top_k.get(k) or {"n": 0, "N": 0, "rate": None}
    return {"n": entry.get("n", 0), "N": entry.get("N", 0), "rate": entry.get("rate")}


def _is_measurable(bridge: Mapping[str, Any]) -> bool:
    """True when at least one row in this bridge output carries a distribution."""
    calibration = bridge.get("metrics_compute", {}).get("calibration", {})
    return bool(calibration.get("n"))


def row_figures(bridge: Mapping[str, Any]) -> dict:
    """The issue-64 table's numeric figures for one (subject, policy) row.

    Top-1 / ECE / Brier are ``None`` (rendered ``n/m``) whenever no row in
    this bridge output carries a candidate distribution at all -- never an
    estimate. Coverage / abstain P/R / missing-candidate / wrong-mutations
    never depend on a distribution and are always a real number.
    """
    metrics_compute = bridge.get("metrics_compute", {})
    rows = bridge.get("rows", [])
    measurable = _is_measurable(bridge)

    top1 = _topk_rate(bridge, 1)
    calibration = metrics_compute.get("calibration", {})
    outcome_counts = metrics_compute.get("outcome_counts", {})
    n_rows = len(rows) or sum(outcome_counts.values())
    propose_n = outcome_counts.get("propose", 0)

    return {
        "measurable": measurable,
        "top1": (
            top1 if measurable else {"n": top1.get("n", 0), "N": top1.get("N", 0), "rate": None}
        ),
        "ece": calibration.get("ece") if measurable else None,
        "brier": calibration.get("brier") if measurable else None,
        "coverage": {
            "n": propose_n,
            "N": n_rows,
            "rate": (propose_n / n_rows) if n_rows else None,
        },
        "abstain_precision": bridge.get("abstain", {}).get("precision"),
        "abstain_recall": bridge.get("abstain", {}).get("recall"),
        "missing_candidate": bridge.get("missing_candidate", {"n": 0, "N": 0, "rate": None}),
        "wrong_mutations": bridge.get("wrong_mutating", {}).get("total", 0),
    }


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------


def _read_only_mutating_slice(bridge: Mapping[str, Any]) -> dict:
    slices = bridge.get("metrics_compute", {}).get("slices", {})
    result = {}
    for name in ("read_only", "mutating", "escalate_or_explain"):
        data = slices.get(name, {})
        calibration = data.get("calibration", {})
        missing_ci = data.get("missing_candidate", {}).get("rate") or {}
        result[name] = {
            "n": data.get("n", 0),
            "ece": calibration.get("ece") if calibration.get("n") else None,
            "brier": calibration.get("brier") if calibration.get("n") else None,
            "missing_candidate": missing_ci.get("value"),
        }
    return result


def _confidence_bucket_slice(bridge: Mapping[str, Any]) -> list[dict]:
    bins = bridge.get("metrics_compute", {}).get("calibration", {}).get("bins", [])
    return [
        {
            "lower": row.get("lower"),
            "upper": row.get("upper"),
            "n": row.get("n"),
            "confidence": row.get("confidence"),
            "accuracy": row.get("accuracy"),
        }
        for row in bins
    ]


def _missing_candidate_split_slice(bridge: Mapping[str, Any]) -> dict:
    """Top-1 rate conditioned on whether a row IS a missing-candidate row.

    Distinct from the top-level 'Missing-candidate' column (which is a
    single rate over every row): this splits every row into the
    missing-candidate bucket and the rest, and reports each bucket's own
    top-1 rate, so a reader can see whether missing-candidate rows are
    scored differently, not just how common they are.
    """
    rows = bridge.get("rows", [])
    buckets: dict[bool, list[dict]] = {True: [], False: []}
    for row in rows:
        buckets[bool(row.get("missing_candidate"))].append(row)
    return {
        "missing": _measurable_top1(buckets[True]),
        "not_missing": _measurable_top1(buckets[False]),
    }


def _measurable_top1(rows: Sequence[Mapping[str, Any]]) -> dict:
    measurable = [r for r in rows if r.get("top1_correct") != NOT_MEASURABLE]
    correct = sum(1 for r in measurable if r.get("top1_correct") is True)
    return {
        "n_rows": len(rows),
        "n_measurable": len(measurable),
        "top1_rate": (correct / len(measurable)) if measurable else None,
    }


def _semantic_vs_epistemic_slice(bridge: Mapping[str, Any]) -> dict:
    outcome_counts = bridge.get("metrics_compute", {}).get("outcome_counts", {})
    semantic = outcome_counts.get("escalate", 0)
    epistemic = outcome_counts.get("abstain_uncertain", 0)
    total = sum(outcome_counts.values())
    return {
        "semantic_escalation": {
            "n": semantic,
            "N": total,
            "rate": (semantic / total) if total else None,
        },
        "epistemic_abstention": {
            "n": epistemic,
            "N": total,
            "rate": (epistemic / total) if total else None,
        },
    }


def _candidate_count_slice(bridge: Mapping[str, Any], traces: Sequence[trace_mod.Trace]) -> dict:
    """Top-1 rate bucketed by how many candidates were offered (never by candidate text).

    Joins each bridge row to its trace purely by id, in memory, to read
    ``len(raw.candidates)``; ids are never written to the result, only the
    resulting bucket key (an integer, or ``"n/m"`` for a row with no
    distribution at all) and counts.
    """
    counts_by_id: dict[str, int | None] = {
        t.case_id: (None if t.raw.candidates is None else len(t.raw.candidates)) for t in traces
    }
    buckets: dict[str, list[dict]] = {}
    for row in bridge.get("rows", []):
        count = counts_by_id.get(row.get("id"))
        key = "n/m" if count is None else str(count)
        buckets.setdefault(key, []).append(row)
    return {key: _measurable_top1(rows) for key, rows in sorted(buckets.items())}


def _permutation_slice(permutation: Mapping[str, Any] | None, subject: str, policy: str) -> Any:
    if permutation is None:
        return "not_run"
    return permutation.get(f"{subject}__{policy}", "not_run")


def row_slices(
    bridge: Mapping[str, Any],
    traces: Sequence[trace_mod.Trace],
    permutation: Mapping[str, Any] | None,
    subject: str,
    policy: str,
) -> dict:
    return {
        "read_only_vs_mutating": _read_only_mutating_slice(bridge),
        "candidate_count": _candidate_count_slice(bridge, traces),
        "confidence_bucket": _confidence_bucket_slice(bridge),
        "missing_candidate_split": _missing_candidate_split_slice(bridge),
        "permutation": _permutation_slice(permutation, subject, policy),
        "semantic_vs_epistemic": _semantic_vs_epistemic_slice(bridge),
    }


# ---------------------------------------------------------------------------
# Judge panel
# ---------------------------------------------------------------------------


def build_judge_panel(judge_results: Mapping[str, Any] | None) -> Any:
    """The judge-panel section's data: ``"not_run"`` when the input is absent.

    Explain text, not a release bar: the panel is reported verbatim (judges,
    results) and never folds into any row's Top-1/ECE/Brier/coverage figure.
    """
    if judge_results is None:
        return "not_run"
    judges = list(judge_results.get("judges", []))
    results = [
        {
            "subject": r.get("subject"),
            "policy": r.get("policy"),
            "judge": r.get("judge"),
            "verdict": r.get("verdict"),
            "score": r.get("score"),
        }
        for r in judge_results.get("results", [])
    ]
    return {"judges": judges, "results": results}


# ---------------------------------------------------------------------------
# Building the result
# ---------------------------------------------------------------------------


def build_row(
    run_dir: str | Path,
    subject: SubjectSpec,
    policy: str,
    permutation: Mapping[str, Any] | None,
) -> dict:
    bridge = load_bridge_output(run_dir, subject.name, policy)
    traces = load_traces(run_dir, subject.name)
    figures = row_figures(bridge)
    row = {
        "subject": subject.name,
        "kind": subject.kind,
        "policy": policy,
        "variant": _variant_for(subject.kind, policy),
        "harness_policy": _harness_policy_for(subject.kind, policy),
        **figures,
        "slices": row_slices(bridge, traces, permutation, subject.name, policy),
    }
    if subject.artifact is not None:
        row["artifact"] = dict(subject.artifact)
    return row


def build_result(run_dir: str | Path) -> dict:
    """The full, deterministic ``result.json`` payload for a run dir.

    A pure function of the run dir's files: same files in, same dict out,
    key order stable (insertion order, subjects/policies walked in manifest
    order, never resorted by value). No case request text or case id
    anywhere in the result.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    permutation = load_permutation(run_dir)
    judge_results = load_judge_results(run_dir)

    rows: list[dict] = []
    reference_rows: list[dict] = []
    for subject in manifest.subjects:
        for policy in subject.policies:
            row = build_row(run_dir, subject, policy, permutation)
            if subject.kind == "reference":
                reference_rows.append(row)
            else:
                rows.append(row)

    return {
        "run_id": manifest.run_id,
        "date": manifest.date,
        "not_measurable_note": NM_FOOTNOTE,
        "rows": rows,
        "reference_rows": reference_rows,
        "judge_panel": build_judge_panel(judge_results),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_rate(value: float | None, *, percent: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if percent else f"{value:.3f}"


def _fmt_measurable(value: float | None, measurable: bool, *, digits: int = 3) -> str:
    if not measurable or value is None:
        return NM_MARKER
    return f"{value:.{digits}f}"


def _render_harness_policy(harness_policy: str | None, kind: str) -> str:
    if harness_policy is not None:
        return harness_policy
    return "n/a" if kind == "reference" else "-"


def _render_main_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Variant | Harness policy | Top-1 | ECE | Brier | Coverage | Abstain P/R "
        "| Missing-candidate | Wrong mutations |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        variant = f"{row['subject']} ({row['variant']})"
        harness_policy = _render_harness_policy(row["harness_policy"], row["kind"])
        top1 = _fmt_measurable(row["top1"]["rate"], row["measurable"], digits=3)
        ece = _fmt_measurable(row["ece"], row["measurable"])
        brier = _fmt_measurable(row["brier"], row["measurable"])
        coverage = _fmt_rate(row["coverage"]["rate"])
        abstain_pr = f"{_fmt_rate(row['abstain_precision'])} / {_fmt_rate(row['abstain_recall'])}"
        missing = _fmt_rate(row["missing_candidate"].get("rate"))
        wrong = str(row["wrong_mutations"])
        lines.append(
            f"| {variant} | {harness_policy} | {top1} | {ece} | {brier} | {coverage} | "
            f"{abstain_pr} | {missing} | {wrong} |"
        )
    return lines


def _render_slices(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        variant = f"{row['subject']} ({row['variant']})"
        slices = row["slices"]
        lines.append(f"### {variant}")
        lines.append("")

        lines.append("Read-only vs. mutating:")
        lines.append("")
        lines.append("| slice | n | ECE | Brier | missing-candidate |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, data in slices["read_only_vs_mutating"].items():
            measurable = data["ece"] is not None
            missing = _fmt_rate(data["missing_candidate"])
            lines.append(
                f"| {name} | {data['n']} | {_fmt_measurable(data['ece'], measurable)} | "
                f"{_fmt_measurable(data['brier'], measurable)} | {missing} |"
            )
        lines.append("")

        lines.append("Candidate count (offered candidates per case):")
        lines.append("")
        lines.append("| candidates offered | rows | measurable | top-1 rate |")
        lines.append("| --- | --- | --- | --- |")
        for key, data in slices["candidate_count"].items():
            lines.append(
                f"| {key} | {data['n_rows']} | {data['n_measurable']} | "
                f"{_fmt_rate(data['top1_rate'])} |"
            )
        lines.append("")

        lines.append("Confidence bucket:")
        lines.append("")
        lines.append("| bucket | n | mean confidence | accuracy |")
        lines.append("| --- | --- | --- | --- |")
        for bucket in slices["confidence_bucket"]:
            lo, hi = bucket["lower"], bucket["upper"]
            label = "n/a" if lo is None or hi is None else f"{lo:.1f} to {hi:.1f}"
            confidence = "-" if bucket["confidence"] is None else f"{bucket['confidence']:.3f}"
            accuracy = "-" if bucket["accuracy"] is None else f"{bucket['accuracy']:.3f}"
            lines.append(f"| {label} | {bucket['n']} | {confidence} | {accuracy} |")
        lines.append("")

        lines.append("Missing-candidate split:")
        lines.append("")
        lines.append("| bucket | rows | measurable | top-1 rate |")
        lines.append("| --- | --- | --- | --- |")
        for name, data in slices["missing_candidate_split"].items():
            lines.append(
                f"| {name} | {data['n_rows']} | {data['n_measurable']} | "
                f"{_fmt_rate(data['top1_rate'])} |"
            )
        lines.append("")

        lines.append("Semantic escalation vs. uncertainty abstention:")
        lines.append("")
        sem = slices["semantic_vs_epistemic"]
        lines.append("| kind | n | N | rate |")
        lines.append("| --- | --- | --- | --- |")
        sem_rate = _fmt_rate(sem["semantic_escalation"]["rate"])
        epi_rate = _fmt_rate(sem["epistemic_abstention"]["rate"])
        lines.append(
            f"| semantic escalation | {sem['semantic_escalation']['n']} | "
            f"{sem['semantic_escalation']['N']} | {sem_rate} |"
        )
        lines.append(
            f"| epistemic abstention | {sem['epistemic_abstention']['n']} | "
            f"{sem['epistemic_abstention']['N']} | {epi_rate} |"
        )
        lines.append("")

        permutation = slices["permutation"]
        if permutation == "not_run":
            lines.append("Permutation slice: not run.")
        else:
            lines.append(f"Permutation slice: `{json.dumps(permutation, sort_keys=True)}`")
        lines.append("")
    return lines


def _render_artifacts(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per subject that carries an artifact label (h1); nothing when none do."""
    seen: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("artifact") is not None:
            seen.setdefault(row["subject"], row["artifact"])
    if not seen:
        return []
    lines = [
        "## Artifacts",
        "",
        "| subject | predictions sha256 | repo id | revision |",
        "| --- | --- | --- | --- |",
    ]
    for subject, artifact in seen.items():
        lines.append(
            f"| {subject} | {artifact.get('predictions_sha256') or '-'} | "
            f"{artifact.get('repo_id') or '-'} | {artifact.get('revision') or '-'} |"
        )
    lines.append("")
    return lines


def _render_judge_panel(judge_panel: Any) -> list[str]:
    lines = ["## Judge panel (explain text, not a release bar)", ""]
    if judge_panel == "not_run":
        lines.append("Judge panel: not run.")
        lines.append("")
        return lines
    judges = judge_panel.get("judges", [])
    results = judge_panel.get("results", [])
    lines.append("Judges: " + (", ".join(judges) if judges else "(none listed)"))
    lines.append("")
    if results:
        lines.append("| subject | policy | judge | verdict | score |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in results:
            score = "-" if r.get("score") is None else f"{r['score']:.3f}"
            lines.append(
                f"| {r.get('subject')} | {r.get('policy')} | {r.get('judge')} | "
                f"{r.get('verdict')} | {score} |"
            )
        lines.append("")
    return lines


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render :func:`build_result`'s output as a markdown comparison page.

    Deterministic and generated only: never hand-edit the output of this
    function. No case request text or case id appears anywhere in it.
    """
    lines: list[str] = []
    lines.append(f"# Tool-Jev release-gate report — run {result['run_id']}")
    lines.append("")
    lines.append(f"Date: {result['date']}")
    lines.append("")
    lines.append(
        "Generated by `evals/tool_jev/report.py`. Do not hand-edit; regenerate from the run "
        "directory instead."
    )
    lines.append("")

    lines.append("## Candidates and baselines")
    lines.append("")
    lines.append(
        "Each candidate/baseline appears twice: model-only (policy `raw`, the bare argmax, "
        "no calibration or gate) and model+harness (the shipped policy) — so the table "
        "answers whether the model improved separately from whether the harness prevented "
        "mistakes."
    )
    lines.append("")
    lines.extend(_render_main_table(result["rows"]))
    lines.append("")
    lines.extend(_render_artifacts(result["rows"]))

    lines.append("## Reference models")
    lines.append("")
    lines.append("Hosted reference models, scored raw (no nvsh harness applies to them).")
    lines.append("")
    lines.extend(_render_main_table(result["reference_rows"]))
    lines.append("")

    lines.append(f"Note: {result['not_measurable_note']}")
    lines.append("")

    lines.append("## Slices")
    lines.append("")
    lines.extend(_render_slices(result["rows"]))
    lines.extend(_render_slices(result["reference_rows"]))

    lines.extend(_render_judge_panel(result["judge_panel"]))

    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def generate(
    run_dir: str | Path,
    *,
    result_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> tuple[dict, str]:
    """Build the JSON result and markdown page for *run_dir*.

    Purely reads *run_dir*'s files and returns ``(result, markdown)``;
    writes them to *result_path*/*markdown_path* only when given (never
    inside a git worktree by convention -- callers pass a private run dir
    path, exactly like ``trace.write_traces``).
    """
    result = build_result(run_dir)
    markdown = render_markdown(result)
    if result_path is not None:
        Path(result_path).write_text(
            json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
    if markdown_path is not None:
        Path(markdown_path).write_text(markdown, encoding="utf-8")
    return result, markdown
