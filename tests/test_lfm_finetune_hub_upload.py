"""scripts/lfm-finetune/hub_upload.py (issue 46, t27): private upload, byte-identical fetch-back.

``huggingface_hub`` is never imported here: every test passes a fake hub that
keeps "remote" repositories in a temporary folder, so no network call, token
or real repository is ever involved.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

_DIR = Path(__file__).resolve().parents[1] / "scripts" / "lfm-finetune"
_REPO = "jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev"
_TOKEN = "hf_fake_token_value_for_tests_only"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"t27_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module():
    return _load("hub_upload")


class FakeHub:
    """Stands in for the huggingface_hub module: HfApi + snapshot_download."""

    def __init__(self, remote: Path, *, private: bool = True, tamper=None) -> None:
        self.remote = remote
        self.calls: list[tuple] = []
        self.private = private
        self.tamper = tamper  # callable(local_dir) run after a download
        hub = self

        class HfApi:
            def __init__(self, token=None) -> None:
                hub.calls.append(("HfApi", token == _TOKEN))

            def create_repo(self, repo_id, **kwargs):
                hub.calls.append(("create_repo", repo_id, kwargs))
                folder = hub.remote / repo_id
                folder.mkdir(parents=True, exist_ok=True)
                (folder / ".gitattributes").write_text("*.gguf filter=lfs\n")

            def update_repo_visibility(self, repo_id, **kwargs):
                hub.calls.append(("update_repo_visibility", repo_id, kwargs))

            def upload_folder(self, *, folder_path, repo_id, **kwargs):
                hub.calls.append(("upload_folder", repo_id, kwargs))
                shutil.copytree(folder_path, hub.remote / repo_id, dirs_exist_ok=True)
                return SimpleNamespace(oid="c0ffee")

            def repo_info(self, repo_id, **kwargs):
                hub.calls.append(("repo_info", repo_id, kwargs))
                return SimpleNamespace(private=hub.private)

            def list_repo_files(self, repo_id, **kwargs):
                hub.calls.append(("list_repo_files", repo_id, kwargs))
                root = hub.remote / repo_id
                return sorted(
                    p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                )

        self.HfApi = HfApi

    def snapshot_download(self, repo_id, *, local_dir, **kwargs):
        self.calls.append(("snapshot_download", repo_id, kwargs))
        shutil.copytree(self.remote / repo_id, local_dir, dirs_exist_ok=True)
        cache = Path(local_dir) / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "download.lock").write_text("x")
        if self.tamper:
            self.tamper(Path(local_dir))
        return str(local_dir)


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundles" / "tool-jev"
    (bundle / "sub").mkdir(parents=True)
    (bundle / "README.md").write_text("---\nlicense: apache-2.0\n---\n# card\n")
    (bundle / "model.safetensors").write_bytes(b"\x00weights\x01")
    (bundle / "sub" / "tokenizer.json").write_text("{}")
    scan_bundle = _load("scan_bundle")
    scan_bundle.write_scan(bundle, scan_bundle._get_scan_secrets())  # noqa: SLF001
    return bundle


def _env(**extra: str) -> dict[str, str]:
    return {"FINAL": "1", "HF_TOKEN": _TOKEN, **extra}


def _upload(tmp_path: Path, hub: FakeHub, **overrides):
    kwargs = dict(
        bundle=overrides.pop("bundle", None) or _bundle(tmp_path),
        repo=_REPO,
        repo_type="model",
        token_env="HF_TOKEN",
        hub=hub,
        environ=_env(),
    )
    kwargs.update(overrides)
    return _module().upload(**kwargs)


@pytest.mark.parametrize(
    "repo",
    [
        "jetson-ai-lab/lfm2.5-350m-nvsh-triage",
        "someone/qwen3.5-0.8b-nvsh-tool-jev",
        "jetson-ai-lab/qwen3.5-0.8b-nvsh-",
        "jetson-ai-lab/qwen3.5-0.8b-nvsh-Tool",
        "jetson-ai-lab/qwen3.5-0.8b-nvsh-a/b",
        "jetson-ai-lab/qwen3.5-0.8b-nvsh-../x",
        "jetson-ai-lab/qwen3.5-0.8b-nvsh-tool-jev ",
    ],
)
def test_a_repo_outside_the_nvsh_qwen_namespace_is_refused(tmp_path, repo: str) -> None:
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(ValueError, match="jetson-ai-lab/qwen3.5-0.8b-nvsh-"):
        _upload(tmp_path, hub, repo=repo)
    assert hub.calls == []


def test_upload_refuses_without_final(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(ValueError, match="FINAL=1"):
        _upload(tmp_path, hub, environ={"HF_TOKEN": _TOKEN})
    assert hub.calls == []


def test_upload_refuses_an_unset_token_variable(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(ValueError, match="HF_TOKEN is not set"):
        _upload(tmp_path, hub, environ={"FINAL": "1"})
    assert hub.calls == []


def test_upload_refuses_a_bundle_changed_after_its_scan(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote")
    bundle = _bundle(tmp_path)
    (bundle / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="scan_bundle.py verify"):
        _upload(tmp_path, hub, bundle=bundle)
    assert hub.calls == []


def test_an_unknown_repo_type_is_refused(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(ValueError, match="repo type"):
        _upload(tmp_path, hub, repo_type="space")


def test_upload_is_private_and_the_fetch_back_is_byte_identical(tmp_path, capsys) -> None:
    hub = FakeHub(tmp_path / "remote")
    result = _upload(tmp_path, hub)
    names = [call[0] for call in hub.calls]
    assert names == [
        "HfApi",
        "create_repo",
        "update_repo_visibility",
        "upload_folder",
        "snapshot_download",
        "list_repo_files",
        "repo_info",
    ]
    assert hub.calls[0] == ("HfApi", True)  # the token came from the named variable
    assert hub.calls[1][2]["private"] is True and hub.calls[1][2]["exist_ok"] is True
    assert hub.calls[2][2]["private"] is True
    assert hub.calls[4][2]["revision"] == "c0ffee"  # exactly the commit just made
    for call in hub.calls[1:]:
        assert call[2]["repo_type"] == "model"
    assert result["private"] is True
    assert result["files"] == 4  # README, weights, tokenizer, scan.json
    out = capsys.readouterr()
    assert "private=True" in out.out
    assert "byte-identical" in out.out
    assert _TOKEN not in out.out + out.err
    assert not list((tmp_path / "bundles").glob(".fetch-*"))  # the fetch dir is removed


def test_a_dataset_bundle_uploads_as_a_dataset_repo(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote")
    _upload(tmp_path, hub, repo=_REPO + "-dataset", repo_type="dataset")
    assert all(call[2]["repo_type"] == "dataset" for call in hub.calls[1:])


def _flip_a_byte(local_dir: Path) -> None:
    path = local_dir / "model.safetensors"
    data = bytearray(path.read_bytes())
    data[1] ^= 0xFF
    path.write_bytes(bytes(data))


def test_a_changed_file_in_the_fetch_back_fails_loudly(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote", tamper=_flip_a_byte)
    with pytest.raises(ValueError, match="model.safetensors"):
        _upload(tmp_path, hub)


def test_a_missing_file_in_the_fetch_back_fails_loudly(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote", tamper=lambda d: (d / "sub" / "tokenizer.json").unlink())
    with pytest.raises(ValueError, match="sub/tokenizer.json"):
        _upload(tmp_path, hub)


def test_a_stale_remote_file_fails_loudly(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote")
    (tmp_path / "remote" / _REPO).mkdir(parents=True)
    (tmp_path / "remote" / _REPO / "old-weights.bin").write_bytes(b"old")
    with pytest.raises(ValueError, match="old-weights.bin"):
        _upload(tmp_path, hub)


def test_a_repo_the_hub_reports_public_fails_loudly(tmp_path) -> None:
    hub = FakeHub(tmp_path / "remote", private=False)
    with pytest.raises(ValueError, match="not private"):
        _upload(tmp_path, hub)


def test_the_script_never_asks_for_a_public_repo() -> None:
    source = (_DIR / "hub_upload.py").read_text(encoding="utf-8")
    assert "private=False" not in source
    assert "private=True" in source


def test_main_reports_a_refusal_without_the_token(tmp_path, monkeypatch, capsys) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setenv("HF_TOKEN", _TOKEN)
    monkeypatch.delenv("FINAL", raising=False)
    code = _module().main(
        [
            "--bundle",
            str(bundle),
            "--repo",
            _REPO,
            "--repo-type",
            "model",
            "--token-env",
            "HF_TOKEN",
        ]
    )
    assert code == 1
    out = capsys.readouterr()
    assert "FINAL=1" in out.err
    assert _TOKEN not in out.out + out.err


def test_compare_ignores_the_download_cache_and_a_hub_gitattributes(tmp_path) -> None:
    module = _module()
    local = tmp_path / "local"
    local.mkdir()
    (local / "a.txt").write_text("a")
    remote = tmp_path / "remote"
    (remote / ".cache" / "huggingface").mkdir(parents=True)
    (remote / "a.txt").write_text("a")
    (remote / ".gitattributes").write_text("x")
    (remote / ".cache" / "huggingface" / "x.lock").write_text("")
    assert module.compare(local, remote) == []
    (remote / "a.txt").write_text("b")
    assert module.compare(local, remote) == ["a.txt: sha256 differs"]


def test_the_result_is_json_serialisable(tmp_path) -> None:
    result = _upload(tmp_path, FakeHub(tmp_path / "remote"))
    json.dumps(result)


def test_a_symlink_in_the_bundle_is_refused_before_any_hub_call(tmp_path) -> None:
    """Codex on t27: a link to a file outside the bundle (e.g. the sealed held-out)
    would otherwise be followed by the scan and the upload."""
    outside = tmp_path / "held-out-q46.sealed.json"
    outside.write_text('{"entries": []}')
    bundle = _bundle(tmp_path)
    (bundle / "extra.json").symlink_to(outside)
    hub = FakeHub(tmp_path / "remote")
    with pytest.raises(ValueError, match="symlink"):
        _upload(tmp_path, hub, bundle=bundle)
    assert hub.calls == []


def test_an_extra_remote_file_under_cache_fails_the_fetch_back(tmp_path) -> None:
    """Codex on t27: the download's own .cache/ metadata is skipped when hashing,
    so the repository's file list is what proves nothing extra is there."""
    hub = FakeHub(tmp_path / "remote")
    stale = tmp_path / "remote" / _REPO / ".cache"
    stale.mkdir(parents=True)
    (stale / "held-out.json").write_text("{}")
    with pytest.raises(ValueError, match=r"\.cache/held-out\.json"):
        _upload(tmp_path, hub)


def test_a_bundle_that_ships_its_own_gitattributes_passes_the_inventory(tmp_path) -> None:
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / ".gitattributes").write_text("*.gguf filter=lfs\n")
    (bundle / "README.md").write_text("x")
    assert _module().check_inventory(bundle, [".gitattributes", "README.md"]) == []
