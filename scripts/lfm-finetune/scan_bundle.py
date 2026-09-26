"""Pre-upload scan for LFM2.5 fine-tune bundles (issue 39).

Before a merged checkpoint is pushed to the Hugging Face Hub this script
scans the bundle directory for leaked credentials, non-localhost endpoints,
and anything else ``nvsh.redact`` would redact, producing a ``scan.json``
that downstream CI and the stage-cache script can verify to be clean.
``.json``/``.jsonl`` files are also parsed and every decoded string value is
scanned recursively, so an escape sequence hiding a credential from the raw
byte scan does not slip through. Model-weight binaries (``*.safetensors``,
``*.gguf``) are listed under ``scan.json``'s ``binaries`` key instead of being
decoded as text, but only once their contents prove the format: a safetensors
file needs a valid JSON header (whose string values are scanned like any JSON
file's), a GGUF file the ``GGUF`` magic; either one without it is an
``unrecognised_binary`` finding. Any other file that is not valid UTF-8 --
``*.bin`` included, since a torch pickle such as ``training_args.bin`` can hold
a Hub token -- is reported as an ``unscanned_binary`` finding rather than
skipped.

    python scripts/lfm-finetune/scan_bundle.py scan <folder>
    python scripts/lfm-finetune/scan_bundle.py verify <folder>
"""

from __future__ import annotations

import hashlib
import importlib.util
import ipaddress
import json
import re
import struct
import sys
from pathlib import Path

from nvsh.redact import redact_report

# Load the shared secrets scanner (importlib because scan-secrets.py
# contains a hyphen in its name, which importlib can handle).
_scan_secrets: object | None = None


def _get_scan_secrets():
    """Lazily load the scan-secrets module so callers can access
    ``_scan_credentials``, ``_scan_endpoints`` and ``redact_report``."""
    global _scan_secrets
    if _scan_secrets is None:
        _scan_secrets = _load_scan_secrets()
    return _scan_secrets


def _load_scan_secrets():
    repo_root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "scan_secrets",
        repo_root / "scripts" / "scan-secrets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scan_secrets"] = mod  # so dataclass annotation lookup works
    spec.loader.exec_module(mod)
    return mod


def folder_hash(folder: Path) -> str:
    """A stable SHA-256 hex digest over every regular file under *folder*.

    Excludes ``scan.json`` so the hash is stable across ``write_scan`` calls.
    Each file contributes its relative path (UTF-8), a NUL, and its bytes,
    all in sorted POSIX order.
    """
    digest = hashlib.sha256()
    for path in sorted(
        (f for f in folder.rglob("*") if f.is_file()),
        key=lambda p: p.relative_to(folder).as_posix(),
    ):
        rel = path.relative_to(folder).as_posix()
        if rel == "scan.json":
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\x00")
    return digest.hexdigest()


#: An IPv4 address or a host name on a private-only suffix, found anywhere in prose.
_IPV4_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.])")
_PRIVATE_NAME_RE = re.compile(
    r"(?<![\w.-])([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:local|lan|internal|home))\b", re.I
)
#: Tailscale and carrier-grade NAT (100.64.0.0/10) are private in practice, not in ipaddress.
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
#: RFC 5737 documentation networks: never a real host, so never a finding
#: (Python's ``is_private`` counts them private; issue 53 t21).
_DOCUMENTATION = tuple(
    ipaddress.ip_network(net) for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
)

#: Model weight files are expected binaries: listed under scan.json's ``binaries``
#: key rather than decoded as text -- once their contents prove the format
#: (``_recognised_binary``), not by name alone (PR #52 review). ``.bin`` is not
#: one: a torch pickle of weights cannot be told from one of training state.
_EXPECTED_BINARY_EXTS = {".safetensors", ".gguf"}

_GGUF_MAGIC = b"GGUF"

#: The safetensors format caps its JSON header at 100 MB.
_SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024


def _safetensors_header(path: Path) -> dict | None:
    """The decoded JSON header of a well-formed safetensors file, else ``None``.

    The format is an 8-byte little-endian header length, then that many bytes
    of UTF-8 JSON (an object), then the tensor data."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            (size,) = struct.unpack("<Q", prefix)
            if size > min(_SAFETENSORS_MAX_HEADER, path.stat().st_size - 8):
                return None
            header = json.loads(handle.read(size).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return header if isinstance(header, dict) else None


def _has_gguf_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(len(_GGUF_MAGIC)) == _GGUF_MAGIC
    except OSError:
        return False


def _scan_weight_file(rel: str, path: Path, scan_secrets: object) -> list[dict]:
    """Findings for a ``*.safetensors`` / ``*.gguf`` file: one ``unrecognised_binary``
    when its contents are not that format, else (safetensors) whatever its header's
    string values hold -- ``__metadata__`` is free-form text."""
    if path.suffix.lower() == ".gguf":
        if _has_gguf_magic(path):
            return []
        detail = "named .gguf but has no GGUF magic"
        return [{"path": rel, "line": 0, "kind": "unrecognised_binary", "detail": detail}]
    header = _safetensors_header(path)
    if header is None:
        detail = "named .safetensors but has no valid safetensors header"
        return [{"path": rel, "line": 0, "kind": "unrecognised_binary", "detail": detail}]
    findings: list[dict] = []
    for fragment in _iter_json_strings(header):
        findings.extend(_scan_fragment(rel, 0, fragment, scan_secrets))
    return findings


def private_hosts(text: str) -> list[tuple[int, str]]:
    """``(line, host)`` for every private address or private-suffix host name in *text*.

    Model and dataset cards are prose, where scan-secrets' JSON endpoint check
    never looks. Loopback and unspecified addresses are allowed, as there.
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in _IPV4_RE.finditer(line):
            try:
                address = ipaddress.ip_address(match.group(1))
            except ValueError:
                continue
            if address.is_loopback or address.is_unspecified:
                continue
            if any(address in net for net in _DOCUMENTATION):
                continue
            if address.is_private or address in _CGNAT:
                found.append((number, match.group(1)))
        for match in _PRIVATE_NAME_RE.finditer(line):
            found.append((number, match.group(1)))
    return found


def _iter_json_strings(obj):
    """Yield every string leaf reachable from *obj* through nested dicts/lists."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_json_strings(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_json_strings(item)


def _scan_fragment(rel: str, base_line: int, fragment: str, scan_secrets: object) -> list[dict]:
    """Run the credential/redact/private-host checks against a decoded JSON string.

    *base_line* is the physical line the JSON value came from (the record's line
    for JSONL, or 1 for a whole-file JSON parse); a fragment's own internal line
    offset (for a multi-line decoded string) is added on top of it.
    """
    findings: list[dict] = []
    for finding in scan_secrets._scan_credentials(rel, fragment):  # noqa: SLF001
        findings.append(
            {
                "path": rel,
                "line": base_line + finding.line - 1,
                "kind": finding.kind,
                "detail": finding.detail,
            }
        )
    for offset, host in private_hosts(fragment):
        findings.append(
            {
                "path": rel,
                "line": base_line + offset - 1,
                "kind": "private_host",
                "detail": host,
            }
        )
    _, fired_rules = redact_report(fragment.encode("utf-8"))
    for rule_name in fired_rules:
        findings.append({"path": rel, "line": base_line, "kind": "redact", "detail": rule_name})
    return findings


def _scan_json_strings(rel: str, suffix: str, text: str, scan_secrets: object) -> list[dict]:
    """Recursively scan decoded JSON string values for ``.json``/``.jsonl`` files.

    Serialized bytes already went through the byte/text scan above; this covers
    values an escape sequence (``\\uXXXX``, ``\\n``, ...) hides from that scan
    until the JSON is actually decoded.
    """
    findings: list[dict] = []
    if suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError:
            return findings
        for fragment in _iter_json_strings(data):
            findings.extend(_scan_fragment(rel, 1, fragment, scan_secrets))
    elif suffix == ".jsonl":
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            for fragment in _iter_json_strings(record):
                findings.extend(_scan_fragment(rel, lineno, fragment, scan_secrets))
    return findings


def list_binaries(folder: Path) -> list[str]:
    """Relative POSIX paths of expected weight-file binaries under *folder*."""
    return sorted(
        f.relative_to(folder).as_posix()
        for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in _EXPECTED_BINARY_EXTS
    )


def scan_folder(folder: Path, scan_secrets: object) -> list[dict]:
    """Return one finding dict per credential / endpoint / redact issue.

    Scans UTF-8 decodable regular files, skipping ``scan.json``. Weight files
    (see ``list_binaries``) are checked by content instead (``_scan_weight_file``).
    Any other file that fails UTF-8 decoding is reported as an
    ``unscanned_binary`` finding instead of being silently skipped, so an
    unexpected binary upload stays visible.

    Each dict has keys: ``path``, ``line``, ``kind``, ``detail`` — all
    ``path`` values are relative to *folder* as POSIX strings.
    """
    findings: list[dict] = []
    for path in sorted(
        (f for f in folder.rglob("*") if f.is_file()),
        key=lambda p: p.relative_to(folder).as_posix(),
    ):
        rel = path.relative_to(folder).as_posix()
        if rel == "scan.json":
            continue
        if path.suffix.lower() in _EXPECTED_BINARY_EXTS:
            findings.extend(_scan_weight_file(rel, path, scan_secrets))
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(
                {"path": rel, "line": 0, "kind": "unscanned_binary", "detail": "non-UTF-8 file"}
            )
            continue
        except OSError:
            continue

        # _scan_credentials expects (path, text) and returns Finding objects.
        for finding in scan_secrets._scan_credentials(rel, text):  # noqa: SLF001
            findings.append(
                {
                    "path": rel,
                    "line": finding.line,
                    "kind": finding.kind,
                    "detail": finding.detail,
                }
            )

        # _scan_endpoints is identical.
        for finding in scan_secrets._scan_endpoints(rel, text):  # noqa: SLF001
            findings.append(
                {
                    "path": rel,
                    "line": finding.line,
                    "kind": finding.kind,
                    "detail": finding.detail,
                }
            )

        for line, host in private_hosts(text):
            findings.append({"path": rel, "line": line, "kind": "private_host", "detail": host})

        # redact_report: returns (redacted_bytes, list_of_rule_names_that_fired)
        file_bytes = path.read_bytes()
        _, fired_rules = redact_report(file_bytes)
        for rule_name in fired_rules:
            findings.append(
                {
                    "path": rel,
                    "line": 0,
                    "kind": "redact",
                    "detail": rule_name,
                }
            )

        findings.extend(_scan_json_strings(rel, path.suffix.lower(), text, scan_secrets))

    return findings


def write_scan(folder: Path, scan_secrets: object) -> dict:
    """Compute findings, hash and the binaries list, write ``scan.json``, return it."""
    findings = scan_folder(folder, scan_secrets)
    hash_hex = folder_hash(folder)
    payload = {
        "hash": hash_hex,
        "findings": findings,
        "clean": len(findings) == 0,
        "binaries": list_binaries(folder),
    }
    (folder / "scan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify(folder: Path) -> str | None:
    """Return ``None`` when the folder is clean and unmodified, else a reason."""
    scan_path = folder / "scan.json"
    if not scan_path.is_file():
        return "no scan.json"
    payload = json.loads(scan_path.read_text(encoding="utf-8"))
    if not payload.get("clean"):
        return "scan has findings"
    current = folder_hash(folder)
    if payload.get("hash") != current:
        return "folder changed after scan"
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``scan FOLDER`` or ``verify FOLDER``."""
    scan_secrets = _get_scan_secrets()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("usage: scan_bundle.py {scan,verify} FOLDER", file=sys.stderr)
        return 1

    subcommand, *rest = argv
    if len(rest) < 1:
        print(f"usage: scan_bundle.py {subcommand} FOLDER", file=sys.stderr)
        return 1

    folder = Path(rest[0]).resolve()
    if not folder.is_dir():
        print(f"{folder} is not a directory", file=sys.stderr)
        return 1

    if subcommand == "scan":
        payload = write_scan(folder, scan_secrets)
        num_findings = len(payload["findings"])
        print(f"{num_findings} finding(s)")
        return 1 if num_findings else 0

    if subcommand == "verify":
        reason = verify(folder)
        if reason is None:
            print("ok")
            return 0
        print(reason)
        return 1

    print(f"unknown subcommand: {subcommand}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
