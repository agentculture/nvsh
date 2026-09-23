"""Pre-upload scan for LFM2.5 fine-tune bundles (issue 39).

Before a merged checkpoint is pushed to the Hugging Face Hub this script
scans the bundle directory for leaked credentials, non-localhost endpoints,
and anything else ``nvsh.redact`` would redact, producing a ``scan.json``
that downstream CI and the stage-cache script can verify to be clean.

    python scripts/lfm-finetune/scan_bundle.py scan <folder>
    python scripts/lfm-finetune/scan_bundle.py verify <folder>
"""

from __future__ import annotations

import hashlib
import importlib.util
import ipaddress
import json
import re
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
            if address.is_private or address in _CGNAT:
                found.append((number, match.group(1)))
        for match in _PRIVATE_NAME_RE.finditer(line):
            found.append((number, match.group(1)))
    return found


def scan_folder(folder: Path, scan_secrets: object) -> list[dict]:
    """Return one finding dict per credential / endpoint / redact issue.

    Only scans UTF-8 decodable regular files, skipping ``scan.json``.

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

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
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

    return findings


def write_scan(folder: Path, scan_secrets: object) -> dict:
    """Compute findings and hash, write ``scan.json``, return the payload."""
    findings = scan_folder(folder, scan_secrets)
    hash_hex = folder_hash(folder)
    payload = {
        "hash": hash_hex,
        "findings": findings,
        "clean": len(findings) == 0,
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
