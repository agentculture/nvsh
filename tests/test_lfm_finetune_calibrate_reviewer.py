"""calibrate_reviewer.py (issue 46, d10): the reviewer's known-good/known-bad probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "lfm-finetune" / "calibrate_reviewer.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("calibrate_reviewer", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_reviewer"] = module
    spec.loader.exec_module(module)
    return module


def _records():
    out = []
    reads = ["memory_stats", "gpu_stats", "swap_status", "disk_stats"]
    for i in range(8):
        out.append(
            {
                "id": f"r{i}~v1",
                "side": "train",
                "text": f"read {i}",
                "expect": {"operation": reads[i % 4], "args": {}},
            }
        )
        out.append(
            {
                "id": f"c{i}~v1",
                "side": "train",
                "text": f"restart vllm {i}",
                "expect": {"operation": "service_restart", "args": {"service": "vllm"}},
            }
        )
        out.append(
            {
                "id": f"e{i}~v1",
                "side": "train",
                "text": f"reflash the board {i}",
                "expect": {"escalate": True},
            }
        )
        out.append(
            {
                "id": f"x{i}~v1",
                "side": "train",
                "text": f"what is cuda {i}",
                "expect": {"explain": True, "answer": "a GPU platform"},
            }
        )
    out.append(
        {
            "id": "h1~v1",
            "side": "train",
            "text": "Escalate this to a human agent please",
            "expect": {"escalate": True},
        }
    )
    return out


def test_probe_is_deterministic_and_labels_every_wrong_pair_bad() -> None:
    module = _module()
    first = module.build_probe(_records(), per_kind=3)
    assert first == module.build_probe(_records(), per_kind=3)
    by_kind = {}
    for item in first:
        by_kind.setdefault(item["kind"], []).append(item)
    assert {"good-read", "good-change", "good-escalate", "good-explain"} <= set(by_kind)
    for item in by_kind["bad-other-check"]:
        original = next(r for r in _records() if r["id"] == item["id"])
        assert item["expect"]["operation"] != original["expect"]["operation"]
        assert item["label"] == "bad"
    for item in by_kind["bad-change-for-read"]:
        operation = module.get_operation(item["expect"]["operation"])
        assert operation is not None
        assert not operation.read_only  # a mutating change
        assert set(item["expect"]["args"]) == {arg.name for arg in operation.args}
    assert all(i["label"] == "bad" for k, v in by_kind.items() if k.startswith("bad") for i in v)
    assert [i["id"] for i in by_kind["bad-asks-for-handoff"]] == ["h1~v1"]
    # a hand-off-in-words request is never used as a good escalate example
    assert all(i["id"] != "h1~v1" for i in by_kind["good-escalate"])


def test_score_counts_false_accepts_and_false_rejects() -> None:
    module = _module()
    items = [
        {"id": "a", "kind": "good-read", "label": "good"},
        {"id": "b", "kind": "bad-other-check", "label": "bad"},
        {"id": "c", "kind": "bad-change-for-read", "label": "bad"},
    ]
    answers = [
        {"accepted": False, "tokens": 10},
        {"accepted": True, "tokens": 20},
        {"accepted": False, "tokens": 30},
    ]
    result = module.score(items, answers)
    assert result["false_accepts"] == ["b:bad-other-check"]
    assert result["false_rejects"] == ["a:good-read"]
    assert result["tokens_mean"] == 20


def test_default_effort_sends_no_reasoning_effort() -> None:
    module = _module()
    assert module._effort_value("default") is None
    assert module._effort_value("xhigh") == "xhigh"
