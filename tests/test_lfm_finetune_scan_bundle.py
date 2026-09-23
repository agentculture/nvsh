"""Pre-upload bundle scan: scripts/lfm-finetune/scan_bundle.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lfm-finetune" / "scan_bundle.py"


def _module():
    spec = importlib.util.spec_from_file_location("scan_bundle", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Lazily load the scan_secrets helper so we can call its private functions
# from test fixtures that produce known findings.
def _load_scan_secrets():
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "scan_secrets",
        repo_root / "scripts" / "scan-secrets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_file(folder: Path, name: str, content: str) -> Path:
    path = folder / name
    path.write_text(content, encoding="utf-8")
    return path


def test_scan_clean(tmp_path):
    """A folder with one harmless text file scans clean: exit 0, scan.json with clean true."""
    module = _module()
    _write_file(tmp_path, "readme.txt", "This is a harmless file.\n")
    result = module.main(["scan", str(tmp_path)])
    assert result == 0
    scan_path = tmp_path / "scan.json"
    assert scan_path.is_file()
    payload = json.loads(scan_path.read_text(encoding="utf-8"))
    assert payload["clean"] is True


def test_scan_with_credential(tmp_path):
    """A folder with a file containing HF_TOKEN=hf_<40+ chars> has at least one finding, exit 1."""
    module = _module()
    _write_file(tmp_path, "env.txt", "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123456789\n")
    result = module.main(["scan", str(tmp_path)])
    assert result == 1
    scan_path = tmp_path / "scan.json"
    assert scan_path.is_file()
    payload = json.loads(scan_path.read_text(encoding="utf-8"))
    assert payload["clean"] is False
    assert len(payload["findings"]) >= 1
    # The redact module should fire 'hf_token' for this token shape.
    kinds = {f["kind"] for f in payload["findings"]}
    assert "redact" in kinds or "credential" in kinds


def test_verify_clean_after_scan(tmp_path):
    """verify returns None right after a clean scan."""
    module = _module()
    _write_file(tmp_path, "readme.txt", "safe content\n")
    module.main(["scan", str(tmp_path)])
    result = module.main(["verify", str(tmp_path)])
    assert result == 0


def test_verify_folder_changed(tmp_path):
    """verify returns 'folder changed after scan' after a file is modified."""
    module = _module()
    _write_file(tmp_path, "a.txt", "original\n")
    module.main(["scan", str(tmp_path)])
    # Modify a file
    _write_file(tmp_path, "a.txt", "modified\n")
    reason = module.verify(tmp_path)
    assert reason == "folder changed after scan"


def test_verify_no_scan_json(tmp_path):
    """verify returns 'no scan.json' for a folder never scanned."""
    module = _module()
    reason = module.verify(tmp_path)
    assert reason == "no scan.json"


def test_folder_hash_ignores_scan_json(tmp_path):
    """folder_hash is the same before and after write_scan (scan.json excluded)."""
    module = _module()
    _write_file(tmp_path, "a.txt", "data\n")
    h_before = module.folder_hash(tmp_path)
    module.main(["scan", str(tmp_path)])
    h_after = module.folder_hash(tmp_path)
    assert h_before == h_after


def test_verify_scan_has_findings(tmp_path):
    """verify returns 'scan has findings' when the scan detected issues."""
    module = _module()
    _write_file(tmp_path, "env.txt", "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123456789\n")
    module.main(["scan", str(tmp_path)])
    reason = module.verify(tmp_path)
    assert reason == "scan has findings"


def test_scan_json_payload_structure(tmp_path):
    """scan.json contains hash, findings, clean keys in the expected format."""
    module = _module()
    _write_file(tmp_path, "file.txt", "hello\n")
    module.main(["scan", str(tmp_path)])
    payload = json.loads((tmp_path / "scan.json").read_text(encoding="utf-8"))
    assert "hash" in payload
    assert "findings" in payload
    assert "clean" in payload
    assert isinstance(payload["hash"], str) and len(payload["hash"]) == 64
    assert isinstance(payload["findings"], list)
    assert isinstance(payload["clean"], bool)


def test_finding_dicts_have_required_keys(tmp_path):
    """Every finding dict in scan.json has path, line, kind, detail."""
    module = _module()
    _write_file(tmp_path, "env.txt", "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123456789\n")
    module.main(["scan", str(tmp_path)])
    payload = json.loads((tmp_path / "scan.json").read_text(encoding="utf-8"))
    required_keys = {"path", "line", "kind", "detail"}
    for finding in payload["findings"]:
        assert required_keys.issubset(set(finding.keys())), finding
