"""Tests for evals/tool_jev/manifest.py (task t5, issue #64).

Covers the three acceptance criteria:

1. adding a checkpoint is one [[candidate]] table; it appears in the
   planned run with no code change,
2. the committed example manifest passes scripts/scan-secrets.py and
   contains no home or absolute private path,
3. manifest validation rejects an unknown provider, a duplicate
   (provider, model) pair, and a judge not in the roster.

All manifests here are inline TOML strings (synthetic fixtures) or the
committed example file — never real private data.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed argv below, no shell=True
import sys
import tomllib
from pathlib import Path

import pytest

from evals.tool_jev import manifest as manifest_mod
from evals.tool_jev.manifest import (
    ManifestError,
    load_manifest,
    parse_manifest,
    resolve_manifest_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_MANIFEST = REPO_ROOT / "evals" / "tool_jev" / "manifest.example.toml"

_BASE_MANIFEST = """
[[candidate]]
name = "a3-heal.q4_k_m"
track = "A"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/a3-heal.q4_k_m.jsonl"

[[baseline]]
name = "stock"
track = "A"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/stock.jsonl"

[[reference]]
provider = "openai"
model = "gpt-6-luna"
reasoning = "medium"
batch = true
capabilities = ["chat"]
api_key_env = "OPENAI_API_KEY"

[[reference]]
provider = "anthropic"
model = "claude-opus-5-5"
reasoning = "medium"
batch = true
capabilities = ["chat"]
api_key_env = "ANTHROPIC_API_KEY"

[[judge]]
provider = "anthropic"
model = "claude-opus-5-5"

[[case_set]]
name = "issue-53-test"
count = 198
split = "test"
path = "splits/issue-53-test.json"
include_heldout = false

[[case_set]]
name = "heldout-track-a"
count = 40
split = "heldout"
path = "splits/heldout-track-a.json"
include_heldout = true

[budget.openai]
usd_cap = 25.0
concurrency_cap = 4
"""


def _parse(text: str):
    return parse_manifest(tomllib.loads(text))


# ---------------------------------------------------------------------------
# Criterion 1: adding a checkpoint is one [[candidate]] table, no code
# change, and it shows up in the planned run.
# ---------------------------------------------------------------------------


def test_base_fixture_manifest_parses_and_plans_known_candidates():
    m = _parse(_BASE_MANIFEST)
    targets = m.plan_run()
    names = {t.name for t in targets}
    assert names == {"a3-heal.q4_k_m", "stock"}


def test_adding_a_new_candidate_table_appears_in_the_planned_run():
    extra_candidate = """
[[candidate]]
name = "new-checkpoint.q5_k_m"
track = "A"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/new-checkpoint.q5_k_m.jsonl"
"""
    m = _parse(_BASE_MANIFEST + extra_candidate)
    targets = m.plan_run()
    matches = [t for t in targets if t.name == "new-checkpoint.q5_k_m"]
    assert len(matches) == 1
    target = matches[0]
    assert target.kind == "candidate"
    assert target.track == "A"
    assert target.predictions_path.endswith("new-checkpoint.q5_k_m.jsonl")
    # And the previously-existing checkpoints are still present.
    names = {t.name for t in targets}
    assert {"a3-heal.q4_k_m", "stock", "new-checkpoint.q5_k_m"} <= names


def test_plan_run_marks_candidates_and_baselines_by_kind():
    m = _parse(_BASE_MANIFEST)
    by_name = {t.name: t.kind for t in m.plan_run()}
    assert by_name["a3-heal.q4_k_m"] == "candidate"
    assert by_name["stock"] == "baseline"


def test_sendable_case_sets_excludes_heldout():
    m = _parse(_BASE_MANIFEST)
    sendable_names = {cs.name for cs in m.sendable_case_sets()}
    assert sendable_names == {"issue-53-test"}
    all_names = {cs.name for cs in m.case_sets}
    assert all_names == {"issue-53-test", "heldout-track-a"}


# ---------------------------------------------------------------------------
# Criterion 2: the committed example manifest is clean.
# ---------------------------------------------------------------------------


def test_example_manifest_exists_and_loads():
    assert EXAMPLE_MANIFEST.exists()
    m = load_manifest(EXAMPLE_MANIFEST)
    assert len(m.candidates) == 2
    assert len(m.baselines) == 2
    assert len(m.references) == 16
    assert len(m.judges) == 5
    assert len(m.case_sets) == 4
    assert len(m.budgets) == 5


def test_example_manifest_has_no_home_or_absolute_private_paths():
    m = load_manifest(EXAMPLE_MANIFEST)
    for entry in (*m.candidates, *m.baselines):
        path = entry.predictions_path
        assert not path.startswith("/"), f"{entry.name}: absolute path {path!r}"
        assert "/home/" not in path, f"{entry.name}: home path {path!r}"
        assert "~" not in path, f"{entry.name}: home-shorthand path {path!r}"


def test_example_manifest_passes_scan_secrets():
    result = subprocess.run(  # nosec B603
        [sys.executable, "scripts/scan-secrets.py", str(EXAMPLE_MANIFEST)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"scan-secrets.py flagged manifest.example.toml (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_example_manifest_judges_are_all_roster_members():
    m = load_manifest(EXAMPLE_MANIFEST)
    roster = m.roster_keys()
    for judge in m.judges:
        assert judge.key in roster


def test_example_manifest_roster_has_no_duplicate_pairs_and_allows_shared_model_id():
    m = load_manifest(EXAMPLE_MANIFEST)
    pairs = [ref.key for ref in m.references]
    assert len(pairs) == len(set(pairs))
    # Deviation d3: kimi-k3 runs on build.nvidia.com only.
    kimi_providers = {p for p, model in pairs if model == "moonshotai/kimi-k3"}
    assert kimi_providers == {"nvidia"}
    # The same model id under two providers is still a legal roster.
    shared = _BASE_MANIFEST + """
[[reference]]
provider = "openrouter"
model = "moonshotai/kimi-k3"
api_key_env = "OPEN_ROUTER_API_KEY"

[[reference]]
provider = "nvidia"
model = "moonshotai/kimi-k3"
api_key_env = "NGC_API_KEY"
"""
    both = {ref.key for ref in _parse(shared).references}
    assert {("openrouter", "moonshotai/kimi-k3"), ("nvidia", "moonshotai/kimi-k3")} <= both


# ---------------------------------------------------------------------------
# Criterion 3: validation rejects bad input.
# ---------------------------------------------------------------------------


def test_rejects_unknown_provider_in_reference():
    bad = _BASE_MANIFEST + """
[[reference]]
provider = "totally-not-a-provider"
model = "some-model"
"""
    with pytest.raises(ManifestError, match="unknown provider"):
        _parse(bad)


def test_rejects_unknown_provider_in_budget():
    bad = _BASE_MANIFEST + """
[budget.not-a-real-provider]
usd_cap = 10.0
concurrency_cap = 2
"""
    with pytest.raises(ManifestError, match="unknown provider"):
        _parse(bad)


def test_rejects_duplicate_provider_model_pair_in_reference_roster():
    bad = _BASE_MANIFEST + """
[[reference]]
provider = "openai"
model = "gpt-6-luna"
reasoning = "medium"
batch = true
"""
    with pytest.raises(ManifestError, match="duplicate"):
        _parse(bad)


def test_rejects_judge_not_in_roster():
    bad = _BASE_MANIFEST + """
[[judge]]
provider = "openai"
model = "some-model-never-referenced"
"""
    with pytest.raises(ManifestError, match="not a member of the reference roster"):
        _parse(bad)


def test_rejects_missing_required_field():
    bad = """
[[candidate]]
name = "no-track-checkpoint"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/x.jsonl"
"""
    with pytest.raises(ManifestError, match="track"):
        _parse(bad)


def test_rejects_unknown_track():
    bad = """
[[candidate]]
name = "bad-track-checkpoint"
track = "C"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/x.jsonl"
"""
    with pytest.raises(ManifestError, match="unknown track"):
        _parse(bad)


def test_rejects_negative_budget_cap():
    bad = _BASE_MANIFEST + """
[budget.nvidia]
usd_cap = -5.0
concurrency_cap = 2
"""
    with pytest.raises(ManifestError, match="usd_cap"):
        _parse(bad)


def test_empty_manifest_parses_to_empty_manifest():
    m = _parse("")
    assert m.candidates == ()
    assert m.baselines == ()
    assert m.references == ()
    assert m.judges == ()
    assert m.case_sets == ()
    assert m.budgets == ()
    assert m.plan_run() == ()


# ---------------------------------------------------------------------------
# NVSH_EVALS_MANIFEST env resolution.
# ---------------------------------------------------------------------------


def test_resolve_manifest_path_reads_env_var():
    path = resolve_manifest_path({manifest_mod.ENV_MANIFEST_PATH: "/some/private/manifest.toml"})
    assert path == Path("/some/private/manifest.toml")


def test_resolve_manifest_path_raises_when_unset():
    with pytest.raises(ManifestError, match=manifest_mod.ENV_MANIFEST_PATH):
        resolve_manifest_path({})


def test_resolve_manifest_path_raises_when_empty():
    with pytest.raises(ManifestError):
        resolve_manifest_path({manifest_mod.ENV_MANIFEST_PATH: ""})


# ---------------------------------------------------------------------------
# Reviewer finding (codex wave 1, P2): the manifest must feed cases.py --
# case_set split tags + relative paths, resolved via case_sets_from_manifest.
# ---------------------------------------------------------------------------


def test_case_set_carries_split_and_relative_path():
    m = _parse(_BASE_MANIFEST)
    by_name = {cs.name: cs for cs in m.case_sets}
    assert by_name["issue-53-test"].split == "test"
    assert by_name["issue-53-test"].path == "splits/issue-53-test.json"
    assert by_name["heldout-track-a"].split == "heldout"
    assert by_name["heldout-track-a"].path == "splits/heldout-track-a.json"


def test_rejects_unknown_split_tag_in_case_set():
    bad = _BASE_MANIFEST + """
[[case_set]]
name = "bad-split-case-set"
count = 1
split = "not-a-real-split"
path = "splits/x.json"
include_heldout = false
"""
    with pytest.raises(ManifestError, match="unknown split"):
        _parse(bad)


def test_rejects_absolute_path_in_case_set():
    bad = _BASE_MANIFEST + """
[[case_set]]
name = "bad-path-case-set"
count = 1
split = "test"
path = "/srv/private/splits/x.json"
include_heldout = false
"""
    with pytest.raises(ManifestError, match="relative"):
        _parse(bad)


def test_rejects_home_shaped_path_in_case_set():
    bad = _BASE_MANIFEST + """
[[case_set]]
name = "bad-home-path-case-set"
count = 1
split = "test"
path = "~/splits/x.json"
include_heldout = false
"""
    with pytest.raises(ManifestError, match="relative"):
        _parse(bad)


def test_rejects_include_heldout_false_for_heldout_split():
    bad = _BASE_MANIFEST + """
[[case_set]]
name = "mislabeled-heldout"
count = 1
split = "heldout"
path = "splits/mislabeled.json"
include_heldout = false
"""
    with pytest.raises(ManifestError, match="include_heldout must be True"):
        _parse(bad)


def test_rejects_include_heldout_true_for_non_heldout_split():
    bad = _BASE_MANIFEST + """
[[case_set]]
name = "mislabeled-test"
count = 1
split = "test"
path = "splits/mislabeled.json"
include_heldout = true
"""
    with pytest.raises(ManifestError, match="include_heldout must be False"):
        _parse(bad)


def test_run_entry_train_split_defaults_to_none_and_is_optional():
    m = _parse(_BASE_MANIFEST)
    assert all(c.train_split is None for c in m.candidates)


def test_run_entry_train_split_parses_when_present():
    with_train_split = _BASE_MANIFEST + """
[[candidate]]
name = "with-train-split"
track = "A"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/with-train-split.jsonl"
train_split = "${NVSH_EVALS_PRIVATE}/splits/with-train-split.train.json"
"""
    m = _parse(with_train_split)
    by_name = {c.name: c for c in m.candidates}
    assert by_name["with-train-split"].train_split == (
        "${NVSH_EVALS_PRIVATE}/splits/with-train-split.train.json"
    )
    # plan_run() carries train_split through onto the RunTarget too.
    targets = {t.name: t for t in m.plan_run()}
    assert targets["with-train-split"].train_split == by_name["with-train-split"].train_split


def test_rejects_non_string_train_split():
    bad = _BASE_MANIFEST + """
[[candidate]]
name = "bad-train-split"
track = "A"
predictions_path = "${NVSH_EVALS_PRIVATE}/predictions/bad-train-split.jsonl"
train_split = 12345
"""
    with pytest.raises(ManifestError, match="train_split"):
        _parse(bad)


def test_example_manifest_case_sets_have_no_home_or_absolute_paths():
    m = load_manifest(EXAMPLE_MANIFEST)
    for cs in m.case_sets:
        assert not cs.path.startswith("/"), f"{cs.name}: absolute path {cs.path!r}"
        assert "~" not in cs.path, f"{cs.name}: home-shorthand path {cs.path!r}"
        assert cs.split in manifest_mod.SPLIT_TAGS


def test_example_manifest_case_sets_split_tags_match_include_heldout():
    m = load_manifest(EXAMPLE_MANIFEST)
    for cs in m.case_sets:
        assert cs.include_heldout == (cs.split in {"heldout", "heldout-mc"})


# ---------------------------------------------------------------------------
# case_sets_from_manifest + load_case_set end to end (a Manifest built from
# the example TOML, a tmp private root, a synthetic split loaded through it).
# ---------------------------------------------------------------------------


def test_case_sets_from_manifest_loads_a_synthetic_split_end_to_end(tmp_path):
    from evals.tool_jev import cases

    m = load_manifest(EXAMPLE_MANIFEST)
    issue_53_test = next(cs for cs in m.case_sets if cs.name == "issue-53-test")

    split_file = tmp_path / issue_53_test.path
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(
        json.dumps(
            {
                "header": "synthetic fixture split",
                "entries": [
                    {
                        "id": "t1",
                        "kind": "explicit",
                        "text": "show me the gpu stats",
                        "expect": {"operation": "gpu_stats", "args": {}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = cases.case_sets_from_manifest(m, tmp_path)
    assert resolved["test"] == str(tmp_path / issue_53_test.path)

    loaded = cases.load_case_set(resolved, "test")
    assert len(loaded) == 1
    assert loaded[0].id == "t1"
    assert loaded[0].text == "show me the gpu stats"
