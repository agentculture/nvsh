"""Model-free parts of scripts/lfm-finetune/draft_heldout.py (issue 46, t17)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
