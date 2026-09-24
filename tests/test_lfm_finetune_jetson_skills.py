"""Jetson skills validation set builder: scripts/lfm-finetune/jetson_skills.py.

Exercises the two eval shapes (device dict form, BSP list form) against a
small fixture copy of both repos, never over the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lfm-finetune/jetson_skills.py"
_FIXTURES = Path(__file__).resolve().parent / "fixtures/jetson_skills"
_DEVICE_SHA = "deadbeef" * 5
_BSP_SHA = "cafef00d" * 5


def _module():
    spec = importlib.util.spec_from_file_location("jetson_skills", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses' field-type resolution looks the module up in sys.modules
    # (needed to resolve `frozenset[tuple[str, ...]]`-style annotations).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _module()


@pytest.fixture
def result(mod):
    return mod.build_from_checkout(
        _FIXTURES / "device",
        _FIXTURES / "bsp",
        _DEVICE_SHA,
        _BSP_SHA,
    )


# ---------------------------------------------------------------------------
# frontmatter parsing
# ---------------------------------------------------------------------------


def test_parses_plain_scalar_description(mod):
    text = (_FIXTURES / "device/skills/jetson-diagnostic/SKILL.md").read_text()
    fm = mod.parse_frontmatter(text)
    assert fm["name"] == "jetson-diagnostic"
    assert fm["license"] == "Apache-2.0"
    assert "Read-only Jetson health snapshot" in fm["description"]


def test_parses_folded_block_scalar_description(mod):
    text = (_FIXTURES / "bsp/skills/jetson-customize-nvpmodel/SKILL.md").read_text()
    fm = mod.parse_frontmatter(text)
    assert fm["name"] == "jetson-customize-nvpmodel"
    # folded onto one line, no embedded newlines
    assert "\n" not in fm["description"]
    assert "nvpmodel power mode" in fm["description"]


# ---------------------------------------------------------------------------
# tool names
# ---------------------------------------------------------------------------


def test_tool_name_replaces_hyphens(mod):
    assert mod.to_tool_name("jetson-diagnostic") == "jetson_diagnostic"


def test_tool_name_is_a_valid_identifier(mod):
    for name in ["jetson-customize-nvpmodel", "jetson-video-setup", "jetson-print-device-info"]:
        tool_name = mod.to_tool_name(name)
        assert tool_name.isidentifier()


# ---------------------------------------------------------------------------
# discovery / tools -- one tool per skill (38 in the real repos; 4 here)
# ---------------------------------------------------------------------------


def test_discovers_all_fixture_skills(result):
    assert len(result.skills) == 4
    names = {s.name for s in result.skills}
    assert names == {
        "jetson-diagnostic",
        "jetson-memory-audit",
        "jetson-customize-nvpmodel",
        "jetson-customize-fan",
    }


def test_one_tool_per_skill_with_no_parameters(result):
    assert len(result.tools) == 4
    for record in result.tools:
        tool = record["tool"]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == record["tool"]["function"]["name"]
        assert tool["function"]["parameters"] == {"type": "object", "properties": {}}
        assert tool["function"]["description"]


def test_tool_name_mapping_is_recorded(result):
    by_skill = {r["skill"]: r["tool"]["function"]["name"] for r in result.tools}
    assert by_skill["jetson-diagnostic"] == "jetson_diagnostic"
    assert by_skill["jetson-customize-nvpmodel"] == "jetson_customize_nvpmodel"


# ---------------------------------------------------------------------------
# both eval shapes
# ---------------------------------------------------------------------------


def test_device_shape_evals_are_loaded(result):
    device_evals = [e for e in result.evals if e.repo == "device"]
    assert len(device_evals) == 3  # 2 + 1 across the two device fixture skills
    by_id = {e.id: e for e in device_evals}
    ev = by_id["jetson-diagnostic-001"]
    assert ev.expected_skill == "jetson-diagnostic"
    assert "SKU" in ev.text
    # device shape has no ground_truth; falls back to expected_output
    assert ev.ground_truth is not None


def test_bsp_shape_evals_are_loaded(result):
    bsp_evals = [e for e in result.evals if e.repo == "bsp"]
    assert len(bsp_evals) == 3  # 2 + 1 across the two BSP fixture skills
    by_id = {e.id: e for e in bsp_evals}
    ev = by_id["jetson-customize-nvpmodel-002"]
    assert ev.expected_skill == "jetson-customize-nvpmodel"
    assert "MAX_FREQ" in ev.text
    assert "silently rounded" in ev.ground_truth


def test_the_test_set_is_exactly_every_eval_found(result):
    assert len(result.evals) == 6  # 3 device + 3 bsp fixture evals
    ids = {e.id for e in result.evals}
    assert len(ids) == 6  # every id unique, nothing dropped or duplicated


# ---------------------------------------------------------------------------
# "names the skill outright"
# ---------------------------------------------------------------------------


def test_eval_naming_the_skill_with_a_leading_slash_is_flagged(result):
    by_id = {e.id: e for e in result.evals}
    assert by_id["jetson-memory-audit-001"].names_skill is True
    assert by_id["jetson-customize-nvpmodel-001"].names_skill is True


def test_eval_not_naming_the_skill_is_not_flagged(result):
    by_id = {e.id: e for e in result.evals}
    assert by_id["jetson-diagnostic-001"].names_skill is False
    assert by_id["jetson-customize-fan-001"].names_skill is False


def test_names_skill_matches_hyphen_and_underscore_forms(mod):
    assert mod.names_skill("run jetson_diagnostic now", "jetson-diagnostic")
    assert mod.names_skill("run /jetson-diagnostic now", "jetson-diagnostic")
    assert not mod.names_skill("diagnose my jetson", "jetson-diagnostic")


# ---------------------------------------------------------------------------
# manifest + README
# ---------------------------------------------------------------------------


def test_manifest_covers_every_tool_and_eval(result):
    kinds = [r["kind"] for r in result.manifest["records"]]
    assert kinds.count("tool") == 4
    assert kinds.count("eval") == 6


def test_manifest_records_repo_file_licence_and_transformed(result):
    tool_records = [r for r in result.manifest["records"] if r["kind"] == "tool"]
    for r in tool_records:
        assert r["repo"] in ("device", "bsp")
        assert r["file"].endswith("SKILL.md")
        assert r["licence"] == "Apache-2.0"
        assert r["transformed"] is True
        assert r["commit"] in (_DEVICE_SHA, _BSP_SHA)


def test_manifest_lists_both_repositories_with_licences(result):
    repos = {r["repo"]: r for r in result.manifest["repositories"]}
    assert repos["device"]["commit"] == _DEVICE_SHA
    assert repos["bsp"]["commit"] == _BSP_SHA
    for r in repos.values():
        assert r["licences"] == {"documentation": "CC-BY-4.0", "source": "Apache-2.0"}


def test_readme_attributes_both_repos_with_commits(result):
    readme = result.readme
    assert "jetson-device-skills" in readme
    assert "jetson-bsp-skills" in readme
    assert _DEVICE_SHA in readme
    assert _BSP_SHA in readme
    assert "CC-BY-4.0" in readme
    assert "Apache-2.0" in readme


# ---------------------------------------------------------------------------
# write_outputs / CLI plumbing
# ---------------------------------------------------------------------------


def test_write_outputs_creates_all_four_files(mod, result, tmp_path):
    mod.write_outputs(result, tmp_path)
    assert (tmp_path / "tools.json").is_file()
    assert (tmp_path / "test.jsonl").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "README.md").is_file()

    tools = json.loads((tmp_path / "tools.json").read_text())
    assert len(tools) == 4

    lines = (tmp_path / "test.jsonl").read_text().splitlines()
    assert len(lines) == 6
    first = json.loads(lines[0])
    assert set(first) == {
        "id",
        "repo",
        "skill",
        "text",
        "expected_skill",
        "ground_truth",
        "names_skill",
    }


# ---------------------------------------------------------------------------
# contamination scan
# ---------------------------------------------------------------------------


def test_scan_is_clean_when_training_shares_nothing(mod, result):
    training_texts = ["completely unrelated training example about npm installs"]
    hits = mod.scan_contamination(result.evals, training_texts)
    assert hits == []


def test_scan_catches_an_exact_prompt_copy(mod, result):
    contaminated_prompt = next(e for e in result.evals if e.repo == "device").text
    training_texts = [f"user: {contaminated_prompt}"]
    hits = mod.scan_contamination(result.evals, training_texts)
    assert any(h.reason == "exact" for h in hits)


def test_scan_catches_a_near_duplicate_ground_truth(mod, result):
    bsp_eval = next(e for e in result.evals if e.repo == "bsp" and e.ground_truth)
    # paraphrase: swap the last two words so it is no longer a contiguous
    # substring match, but almost every 5-token shingle is still shared.
    words = bsp_eval.ground_truth.split()
    words[-2], words[-1] = words[-1], words[-2]
    paraphrased = " ".join(words)
    hits = mod.scan_contamination(result.evals, [paraphrased])
    assert any(h.eval_id == bsp_eval.id and "near-duplicate" in h.reason for h in hits)


def test_scan_ignores_short_unrelated_training_text(mod, result):
    hits = mod.scan_contamination(result.evals, ["ok", "sure thing", ""])
    assert hits == []


def test_load_training_texts_reads_every_string_in_a_jsonl_file(mod, tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello world"}]})
        + "\n"
        + json.dumps({"a": {"b": ["nested string here"]}})
        + "\n"
    )
    texts = mod.load_training_texts(path)
    assert "hello world" in texts
    assert "nested string here" in texts


# ---------------------------------------------------------------------------
# CLI: scan subcommand end to end (no network)
# ---------------------------------------------------------------------------


def test_cli_scan_exits_nonzero_on_contamination(mod, result, tmp_path):
    mod.write_outputs(result, tmp_path)
    contaminated_prompt = next(e for e in result.evals if e.repo == "device").text
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"messages": [{"content": contaminated_prompt}]}) + "\n")

    rc = mod.main(
        [
            "scan",
            "--training",
            str(training),
            "--test",
            str(tmp_path / "test.jsonl"),
        ]
    )
    assert rc == 1


def test_cli_scan_exits_zero_when_clean(mod, result, tmp_path):
    mod.write_outputs(result, tmp_path)
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"messages": [{"content": "totally unrelated text"}]}) + "\n")

    rc = mod.main(
        [
            "scan",
            "--training",
            str(training),
            "--test",
            str(tmp_path / "test.jsonl"),
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# CLI: build subcommand with --skip-fetch (no network)
# ---------------------------------------------------------------------------


def test_cli_build_with_skip_fetch_writes_outputs(mod, tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    # lay out work_dir/device and work_dir/bsp as fetch_repo would have
    import shutil

    shutil.copytree(_FIXTURES / "device", work_dir / "device")
    shutil.copytree(_FIXTURES / "bsp", work_dir / "bsp")
    out_dir = tmp_path / "out"

    rc = mod.main(
        [
            "build",
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(out_dir),
            "--device-sha",
            _DEVICE_SHA,
            "--bsp-sha",
            _BSP_SHA,
            "--skip-fetch",
        ]
    )
    assert rc == 0
    assert (out_dir / "tools.json").is_file()
    assert (out_dir / "test.jsonl").is_file()
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "README.md").is_file()


def test_cli_build_fails_and_writes_nothing_when_training_contaminates(mod, tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    import shutil

    shutil.copytree(_FIXTURES / "device", work_dir / "device")
    shutil.copytree(_FIXTURES / "bsp", work_dir / "bsp")
    out_dir = tmp_path / "out"

    # build once (clean) to learn a real eval prompt
    clean_rc = mod.main(
        [
            "build",
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(tmp_path / "preview"),
            "--device-sha",
            _DEVICE_SHA,
            "--bsp-sha",
            _BSP_SHA,
            "--skip-fetch",
        ]
    )
    assert clean_rc == 0
    test_lines = (tmp_path / "preview/test.jsonl").read_text().splitlines()
    contaminated_prompt = json.loads(test_lines[0])["text"]

    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps({"messages": [{"content": contaminated_prompt}]}) + "\n")

    rc = mod.main(
        [
            "build",
            "--work-dir",
            str(work_dir),
            "--out-dir",
            str(out_dir),
            "--device-sha",
            _DEVICE_SHA,
            "--bsp-sha",
            _BSP_SHA,
            "--skip-fetch",
            "--training",
            str(training),
        ]
    )
    assert rc == 1
    assert not out_dir.exists()


_PROBE_EVAL = (
    "What is this Jetson? Tell me the SKU, how much memory it has, "
    "and what's currently using it."
)
_PROBE_PARAPHRASE = (
    "What Jetson is this? Tell me its SKU, how much memory it has, "
    "and what is using it right now."
)


def test_scan_flags_a_light_paraphrase_of_an_eval() -> None:
    module = _module()
    eval_text = _PROBE_EVAL
    training = _PROBE_PARAPHRASE
    hit = module._find_contamination(
        "e1", "text", eval_text, [training], [module.normalize_text(training)], 0.8
    )
    assert hit is not None
    assert hit.reason.startswith("paraphrase")


def test_scan_leaves_a_different_request_about_the_same_device_clean() -> None:
    module = _module()
    eval_text = _PROBE_EVAL
    training = "Tell me the power mode of this Jetson and whether jetson_clocks is on."
    hit = module._find_contamination(
        "e1", "text", eval_text, [training], [module.normalize_text(training)], 0.8
    )
    assert hit is None


def test_a_one_word_training_string_is_not_an_exact_copy() -> None:
    module = _module()
    hit = module._find_contamination("e1", "text", _PROBE_EVAL, ["it"], ["it"], 0.8)
    assert hit is None


def test_a_paraphrase_buried_in_unrelated_text_is_flagged() -> None:
    module = _module()
    padding = "Simmer the onions slowly and water the tomatoes twice a week in summer. " * 4
    training = padding + _PROBE_PARAPHRASE + " " + padding
    hit = module._find_contamination(
        "e1", "text", _PROBE_EVAL, [training], [module.normalize_text(training)], 0.8
    )
    assert hit is not None


def test_an_existing_checkout_at_another_commit_is_refused(tmp_path) -> None:
    import subprocess

    module = _module()
    repo = tmp_path / "device"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "f"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    module.verify_checkout(repo, head)
    with pytest.raises(ValueError, match="not the pinned"):
        module.fetch_repo("unused", "0" * 40, repo)
    (repo / "f").write_text("changed")
    with pytest.raises(ValueError, match="local changes"):
        module.verify_checkout(repo, head)


# ---------------------------------------------------------------------------
# bodies (s3 seeds)
# ---------------------------------------------------------------------------


def test_body_paragraphs_skip_frontmatter_and_code(mod):
    text = (
        "---\nname: x\ndescription: d\n---\n\n# Title\n\nFirst line\ncontinues.\n\n"
        "```bash\nrm -rf /\n```\n\nLast."
    )
    assert mod.body_paragraphs(text) == ["# Title", "First line continues.", "Last."]


def test_bodies_carry_the_tool_and_a_capped_excerpt(mod, result):
    bodies, _ = mod.build_bodies(result.skills, result.evals, limit=40)
    assert [b["skill"] for b in bodies] == [s.name for s in result.skills]
    assert all(b["tool"] == mod.build_tool(s) for b, s in zip(bodies, result.skills))
    assert all(len(b["body"]) <= 40 for b in bodies)
    full, _ = mod.build_bodies(result.skills, result.evals)
    assert any("unified, agent-friendly view" in b["body"] for b in full)


def test_a_body_paragraph_matching_an_eval_is_left_out(mod, result, tmp_path):
    skill = result.skills[0]
    eval_text = next(e.text for e in result.evals if e.skill == skill.name)
    copy = tmp_path / "SKILL.md"
    copy.write_text(skill.skill_md.read_text() + f"\n\nExample: {eval_text}\n\nKeep this one.\n")
    moved = type(skill)(
        repo=skill.repo,
        name=skill.name,
        skill_md=copy,
        evals_json=skill.evals_json,
        frontmatter=skill.frontmatter,
    )
    bodies, dropped = mod.build_bodies([moved], result.evals)
    assert dropped == 1
    assert eval_text not in bodies[0]["body"]
    assert "Keep this one." in bodies[0]["body"]


def test_write_outputs_also_writes_bodies(mod, result, tmp_path):
    mod.write_outputs(result, tmp_path)
    bodies = json.loads((tmp_path / "bodies.json").read_text())
    assert len(bodies) == 4
    assert all("body" in b and "tool" in b for b in bodies)


def test_scan_checks_each_distinct_training_string_once(mod, result, monkeypatch):
    # Issue 46: every rendered training row repeats the same long system prompt
    # (the tool definitions), so the scan re-windowed one text ~1,463 times.
    seen: list[str] = []
    real = mod._find_contamination

    def counting(eval_id, field, text, training_texts, training_normalized, threshold):
        seen.append(len(training_texts))
        return real(eval_id, field, text, training_texts, training_normalized, threshold)

    monkeypatch.setattr(mod, "_find_contamination", counting)
    repeated = ["the same long system prompt " * 20] * 50 + ["a different training string here"]
    mod.scan_contamination(result.evals, repeated)
    assert set(seen) == {2}


def test_scan_finds_a_copy_hidden_among_repeats(mod, result):
    contaminated_prompt = next(e for e in result.evals if e.repo == "device").text
    training = ["the same long system prompt"] * 30 + [f"user: {contaminated_prompt}"]
    hits = mod.scan_contamination(result.evals, training)
    assert any(h.reason == "exact" for h in hits)
