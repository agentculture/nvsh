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
    assert isinstance(payload["hash"], str)
    assert len(payload["hash"]) == 64
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


def test_a_private_host_in_free_text_is_a_finding(tmp_path):
    """Model and dataset cards are prose: a private address there must be caught too."""
    module = _module()
    _write_file(tmp_path, "README.md", "Served at http://192.168.1.138:8000/v1 during the run.\n")
    _write_file(tmp_path, "notes.txt", "gateway on 10.0.0.5 and 100.93.248.8, box spark2.local\n")
    findings = module.scan_folder(tmp_path, _load_scan_secrets())
    hosts = {f["detail"] for f in findings if f["kind"] == "private_host"}
    assert hosts == {"192.168.1.138", "10.0.0.5", "100.93.248.8", "spark2.local"}
    assert module.main(["scan", str(tmp_path)]) == 1


def test_localhost_public_hosts_and_versions_are_not_private_hosts(tmp_path):
    module = _module()
    _write_file(
        tmp_path,
        "README.md",
        "Try http://localhost:8000 or 127.0.0.1; see https://huggingface.co/Qwen and"
        " https://github.com/agentculture/nvsh; transformers 5.5.0, torch 2.12.1, 8.8.8.8.\n",
    )
    findings = module.scan_folder(tmp_path, _load_scan_secrets())
    assert [f for f in findings if f["kind"] == "private_host"] == []


def test_json_file_scans_decoded_unicode_escaped_string_value(tmp_path):
    """A \\u-escaped credential inside a JSON string value is invisible to a raw-byte
    scan (the literal bytes never contain "hf_") but must be caught once decoded."""
    module = _module()
    _write_file(
        tmp_path,
        "record.json",
        '{"field": "\\u0068\\u0066_abcdefghijklmnopqrstuvwxyz0123456789"}\n',
    )
    findings = module.scan_folder(tmp_path, _load_scan_secrets())
    assert any(f["kind"] == "redact" and f["detail"] == "hf_token" for f in findings)
    assert all(f["path"] == "record.json" for f in findings)


def test_jsonl_file_scans_decoded_string_value_per_line(tmp_path):
    """Each JSONL line that parses as JSON gets its decoded string values scanned,
    and the finding's line number is the physical line inside the file."""
    module = _module()
    _write_file(
        tmp_path,
        "records.jsonl",
        '{"ok": "nothing here"}\n'
        '{"field": "\\u0068\\u0066_abcdefghijklmnopqrstuvwxyz0123456789"}\n',
    )
    findings = module.scan_folder(tmp_path, _load_scan_secrets())
    hits = [f for f in findings if f["kind"] == "redact" and f["detail"] == "hf_token"]
    assert len(hits) == 1
    assert hits[0]["line"] == 2


def test_json_recursive_scan_covers_nested_objects_and_arrays(tmp_path):
    """The decoded-string scan recurses through nested dicts and lists, not just
    top-level values."""
    module = _module()
    _write_file(
        tmp_path,
        "nested.json",
        json.dumps(
            {
                "outer": [
                    {"inner": "hf_abcdefghijklmnopqrstuvwxyz0123456789"},
                ]
            }
        ),
    )
    findings = module.scan_folder(tmp_path, _load_scan_secrets())
    assert any(f["kind"] == "redact" and f["detail"] == "hf_token" for f in findings)


def test_non_utf8_file_is_an_unscanned_binary_finding(tmp_path):
    """A non-UTF-8 file that is not an expected weight file must be visible as a
    finding, not silently skipped."""
    module = _module()
    (tmp_path / "mystery.dat").write_bytes(b"\xff\xfe\x00\x01garbage")
    findings = module.scan_folder(tmp_path, _load_scan_secrets())
    kinds = [f for f in findings if f["kind"] == "unscanned_binary"]
    assert len(kinds) == 1
    assert kinds[0]["path"] == "mystery.dat"
    module_result = module.main(["scan", str(tmp_path)])
    assert module_result == 1


def test_expected_binary_extensions_are_listed_not_flagged(tmp_path):
    """*.safetensors / *.gguf / *.bin are expected binaries: listed under scan.json's
    'binaries' key, never reported as findings and never as unscanned_binary."""
    module = _module()
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x01\x02\x03")
    (tmp_path / "adapter.gguf").write_bytes(b"\x00\x01\x02\x03")
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01\x02\x03")
    _write_file(tmp_path, "readme.txt", "harmless\n")
    findings = module.scan_folder(tmp_path, _load_scan_secrets())
    assert findings == []
    payload = module.write_scan(tmp_path, _load_scan_secrets())
    assert payload["clean"] is True
    assert payload["binaries"] == ["adapter.gguf", "model.safetensors", "weights.bin"]
    assert module.main(["scan", str(tmp_path)]) == 0
