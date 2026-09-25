"""Model-free parts of scripts/lfm-finetune/draft_heldout.py (issue 46, t17)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from nvsh.ops import table

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "draft_heldout.py"


def _module():
    spec = importlib.util.spec_from_file_location("draft_heldout", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["draft_heldout"] = module
    spec.loader.exec_module(module)
    return module


def test_the_prompts_cover_every_operation_plus_escalate_and_explain() -> None:
    keys = [key for key, _ in _module().prompts(table)]
    assert keys == [f"op:{name}" for name in table.names()] + ["escalate", "explain"]


def test_the_prompts_show_the_table_and_nothing_from_the_corpus() -> None:
    corpus = (Path(__file__).resolve().parents[1] / "nvsh/tiers/corpus/dev.json").read_text()
    for _, prompt in _module().prompts(table):
        assert "Operations:" in prompt
        assert '"expect"' not in prompt
        assert "dev-" not in prompt
    assert '"expect"' in corpus  # the check above would notice a corpus leak


def test_a_json_reply_in_a_code_fence_is_parsed() -> None:
    reply = 'Here:\n```json\n[{"text": "a", "args": {}}]\n```'
    assert _module().parse_json_list(reply) == [{"text": "a", "args": {}}]


def test_the_seed_defaults_to_issue_46s_and_can_be_overridden() -> None:
    assert _module().parse_args(["out"]) == (Path("out"), 46)
    assert _module().parse_args(["--seed", "53", "out"]) == (Path("out"), 53)
    assert _module().parse_args(["out", "--seed", "7"]) == (Path("out"), 7)


def test_bare_string_items_become_text_only_items() -> None:
    module = _module()
    assert module.as_item("reboot the box") == {"text": "reboot the box"}
    assert module.as_item({"text": "a", "args": {}}) == {"text": "a", "args": {}}
    assert module.as_item(7) == {}


def test_a_seed_flag_without_a_value_is_a_usage_error() -> None:
    """PR #65 review: '--seed' as the last argument raised IndexError."""
    with pytest.raises(SystemExit):
        _module().parse_args(["out", "--seed"])


def test_the_draft_header_names_the_issue_it_was_drafted_for() -> None:
    """PR #65 review: an issue-53 draft (its own --seed) was labelled issue 46's."""
    module = _module()
    assert module.parse_args(["out", "--seed", "53", "--issue", "53"]) == (Path("out"), 53, 53)[:2]
    assert module.parse_issue(["out", "--seed", "53", "--issue", "53"]) == 53
    assert module.parse_issue(["out"]) == 46
    assert "Issue 53 sealed held-out draft" in module.draft_header(53, "snap", 53)
    assert "t17" not in module.draft_header(53, "snap", 53)
    assert "Issue 46 sealed held-out draft (task t17, decision c51)" in module.draft_header(
        46, "s", 46
    )
