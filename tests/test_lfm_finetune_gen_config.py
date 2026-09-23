"""scripts/lfm-finetune/gen_config.py (issue 46, deviation d3): every served
Qwen model directory must carry a generation_config.json with temperature 0,
because Qwen3.5-0.8B ships none and nvsh's Tier 2 request sets no
temperature, so vLLM would otherwise sample at its default 1.0.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lfm-finetune" / "gen_config.py"


def _module():
    spec = importlib.util.spec_from_file_location("gen_config", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------


def test_write_keeps_other_keys_and_forces_temperature_zero(tmp_path):
    module = _module()
    gen_path = tmp_path / "generation_config.json"
    _write_json(gen_path, {"temperature": 0.7, "do_sample": True, "eos_token_id": 999})

    module.write(tmp_path)

    payload = json.loads(gen_path.read_text(encoding="utf-8"))
    assert payload["temperature"] == 0.0
    assert payload["do_sample"] is False
    assert payload["eos_token_id"] == 999


def test_write_creates_the_file_from_config_json_top_level_token_ids(tmp_path):
    module = _module()
    _write_json(
        tmp_path / "config.json",
        {"eos_token_id": 1, "bos_token_id": 2, "pad_token_id": 3},
    )

    module.write(tmp_path)

    payload = json.loads((tmp_path / "generation_config.json").read_text(encoding="utf-8"))
    assert payload["temperature"] == 0.0
    assert payload["do_sample"] is False
    assert payload["eos_token_id"] == 1
    assert payload["bos_token_id"] == 2
    assert payload["pad_token_id"] == 3


def test_write_creates_the_file_from_config_json_text_config_token_ids(tmp_path):
    """Qwen3.5's own config.json nests eos_token_id under text_config (no top-level id)."""
    module = _module()
    _write_json(
        tmp_path / "config.json",
        {"model_type": "qwen3_5", "text_config": {"eos_token_id": 248044}},
    )

    module.write(tmp_path)

    payload = json.loads((tmp_path / "generation_config.json").read_text(encoding="utf-8"))
    assert payload["temperature"] == 0.0
    assert payload["do_sample"] is False
    assert payload["eos_token_id"] == 248044
    assert "bos_token_id" not in payload
    assert "pad_token_id" not in payload


def test_write_with_no_config_json_at_all_still_writes_the_two_required_keys(tmp_path):
    module = _module()

    module.write(tmp_path)

    payload = json.loads((tmp_path / "generation_config.json").read_text(encoding="utf-8"))
    assert payload == {"temperature": 0.0, "do_sample": False}


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


def test_check_reports_a_missing_file(tmp_path):
    module = _module()
    reason = module.check(tmp_path)
    assert reason is not None
    assert "generation_config.json" in reason


def test_check_reports_a_nonzero_temperature(tmp_path):
    module = _module()
    _write_json(tmp_path / "generation_config.json", {"temperature": 0.7, "do_sample": False})
    reason = module.check(tmp_path)
    assert reason is not None
    assert "temperature" in reason


def test_check_reports_do_sample_true(tmp_path):
    module = _module()
    _write_json(tmp_path / "generation_config.json", {"temperature": 0.0, "do_sample": True})
    reason = module.check(tmp_path)
    assert reason is not None
    assert "do_sample" in reason


def test_check_passes_a_valid_file(tmp_path):
    module = _module()
    _write_json(tmp_path / "generation_config.json", {"temperature": 0.0, "do_sample": False})
    assert module.check(tmp_path) is None


# ---------------------------------------------------------------------------
# stock_copy()
# ---------------------------------------------------------------------------


def test_stock_copy_resolves_symlinks_and_writes_generation_config(tmp_path):
    module = _module()
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    blob = blobs / "abc123"
    blob.write_text("weights", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to(blob)
    _write_json(snapshot / "config.json", {"eos_token_id": 7})

    out_dir = tmp_path / "out"
    module.stock_copy(snapshot, out_dir)

    copied = out_dir / "model.safetensors"
    assert copied.is_file()
    assert not copied.is_symlink()
    assert copied.read_text(encoding="utf-8") == "weights"
    assert module.check(out_dir) is None
    payload = json.loads((out_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert payload["eos_token_id"] == 7


def test_stock_copy_refuses_a_nonempty_out_dir_without_force(tmp_path):
    module = _module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "file.txt").write_text("x", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("y", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module.stock_copy(snapshot, out_dir)

    # Nothing was touched.
    assert [p.name for p in out_dir.iterdir()] == ["existing.txt"]


def test_stock_copy_with_force_overwrites_a_nonempty_out_dir(tmp_path):
    module = _module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "file.txt").write_text("x", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("y", encoding="utf-8")

    module.stock_copy(snapshot, out_dir, force=True)

    assert (out_dir / "file.txt").read_text(encoding="utf-8") == "x"
    assert not (out_dir / "existing.txt").exists()
    assert module.check(out_dir) is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_write_then_check(tmp_path, capsys):
    module = _module()
    assert module.main(["write", str(tmp_path)]) == 0
    assert module.main(["check", str(tmp_path)]) == 0


def test_cli_check_exits_1_with_reason_on_stderr(tmp_path, capsys):
    module = _module()
    code = module.main(["check", str(tmp_path)])
    assert code == 1
    captured = capsys.readouterr()
    assert "generation_config.json" in captured.err


def test_cli_stock_copy(tmp_path):
    module = _module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "file.txt").write_text("x", encoding="utf-8")
    out_dir = tmp_path / "out"

    assert module.main(["stock-copy", str(snapshot), str(out_dir)]) == 0
    assert module.main(["check", str(out_dir)]) == 0


def test_cli_stock_copy_refuses_without_force(tmp_path):
    module = _module()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "file.txt").write_text("x", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("y", encoding="utf-8")

    assert module.main(["stock-copy", str(snapshot), str(out_dir)]) == 1

    assert module.main(["stock-copy", str(snapshot), str(out_dir), "--force"]) == 0
