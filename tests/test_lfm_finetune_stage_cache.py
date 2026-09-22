"""scripts/lfm-finetune/stage_cache.py (issue 39): a checkpoint staged as a cached repo."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune" / "stage_cache.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage_cache", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(directory: Path, template: str, weights: bytes = b"w") -> Path:
    directory.mkdir(parents=True)
    (directory / "chat_template.jinja").write_text(template)
    (directory / "model.safetensors").write_bytes(weights)
    return directory


def test_a_checkpoint_is_staged_with_refs_main_naming_its_snapshot(tmp_path) -> None:
    module = _module()
    base = _model(tmp_path / "base", "T")
    merged = _model(tmp_path / "merged", "T", b"tuned")
    revision = module.stage(merged, "org/model", tmp_path / "cache", base)
    repo = tmp_path / "cache" / "hub" / "models--org--model"
    assert (repo / "refs" / "main").read_text() == revision
    assert (repo / "snapshots" / revision / "model.safetensors").read_bytes() == b"tuned"
    assert len(revision) == 40


def test_the_revision_follows_the_weights(tmp_path) -> None:
    module = _module()
    one = _model(tmp_path / "one", "T", b"a")
    same = _model(tmp_path / "same", "T", b"a")
    other = _model(tmp_path / "other", "T", b"b")
    assert module.revision_of(one) == module.revision_of(same) != module.revision_of(other)


def test_a_changed_chat_template_is_refused(tmp_path) -> None:
    module = _module()
    base = _model(tmp_path / "base", "T")
    merged = _model(tmp_path / "merged", "T changed")
    with pytest.raises(ValueError, match="chat template differs"):
        module.stage(merged, "org/model", tmp_path / "cache", base)


def test_a_template_inside_tokenizer_config_is_read(tmp_path) -> None:
    module = _module()
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "T"}))
    assert module.chat_template(directory) == "T"


def test_a_repo_id_must_be_owner_slash_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="owner/name"):
        _module().repo_dir(tmp_path, "just-a-name")
