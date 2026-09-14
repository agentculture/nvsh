"""Fixture hygiene: no leaked personal data, and recorded transcripts carry
a provenance header.

Two independent checks over ``tests/fakes/`` and ``tests/fixtures/``:

1. **No PII.** Nothing in those trees may contain a ``/home/<user>`` path,
   a ``~/`` path, an e-mail address, or a 32-hex-character id (session ids,
   UUIDs-without-dashes, etc.) — all things a real recorded session could
   leak but a checked-in fixture must never carry. ``tests/fixtures/
   redact_corpus.txt`` is exempt: it is already a deliberately-fake
   secret-shaped corpus (see its own header and ``scripts/scan-secrets.py``'s
   ``SELF_EXCLUDE``), not real captured data, so applying the same
   real-data rule to it would be testing the wrong thing.

2. **Recorded transcripts carry a header.** A "recorded transcript" is a
   transcript *data* file — ``*.jsonl`` / ``*.ndjson`` / ``*.txt`` — sitting
   directly under ``tests/fakes/`` or ``tests/fixtures/``, as opposed to the
   executable fake CLI binaries in ``tests/fakes/`` (no extension, not data)
   or the structured platform/capture fixtures that were never "recorded
   from a CLI" in the transcript sense. Every such file must contain a line
   shaped ``# recorded-from: <cli> <version>`` so a reader can tell what
   produced it and against which version. The exact rule, including the
   directory exemptions, is documented in ``tests/fixtures/README.md`` —
   keep the two in sync.

   Exempt directories/files (not transcript data, or already covered by
   their own rationale above): ``tests/fixtures/platform/`` (recorded
   *device* state — sysfs/proc/subprocess snapshots, not CLI transcripts),
   ``tests/fixtures/capture/`` (raw terminal capture fixtures for the
   capture-parsing tests, predating this rule and not CLI transcripts
   either), and ``tests/fixtures/redact_corpus.txt`` (see above).

Both checks have two kinds of test: one that runs the real scan over this
repo's actual ``tests/fakes``/``tests/fixtures`` trees and must currently
pass, and unit tests that point the same scanning functions at a temporary
directory holding a deliberately dirty fixture, to prove each rule actually
fires.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SECRETS_SCRIPT = REPO_ROOT / "scripts" / "scan-secrets.py"

# Loaded the same way tests/test_scan_secrets.py loads it, so this module can
# prove scripts/scan-secrets.py actually covers tests/fixtures (acceptance
# criterion 2) without duplicating that file's own test suite.
_spec = importlib.util.spec_from_file_location("scan_secrets_fixture_hygiene", SCAN_SECRETS_SCRIPT)
scan_secrets = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules.setdefault("scan_secrets_fixture_hygiene", scan_secrets)
_spec.loader.exec_module(scan_secrets)

FIXTURE_DIRS = ("tests/fakes", "tests/fixtures")

# Deliberately-fake corpora and this module's own documentation, neither of
# which is real captured data. Kept as a set of repo-relative POSIX paths,
# same shape as scripts/scan-secrets.py's SELF_EXCLUDE, and for the same
# reason: tests/fixtures/README.md documents the ``~/`` rule in prose (it
# has to say the literal pattern to describe it) and redact_corpus.txt is
# already a deliberately fake secret corpus.
PII_EXEMPT_FILES = {"tests/fixtures/redact_corpus.txt", "tests/fixtures/README.md"}

TRANSCRIPT_SUFFIXES = (".jsonl", ".ndjson", ".txt")
TRANSCRIPT_EXEMPT_DIRS = ("tests/fixtures/platform", "tests/fixtures/capture")
TRANSCRIPT_EXEMPT_FILES = {"tests/fixtures/redact_corpus.txt"}

HEADER_RE = re.compile(r"^#\s*recorded-from:\s*\S+\s+\S+", re.MULTILINE)

_HOME_RE = re.compile(r"/home/[A-Za-z0-9_.-]+")
_TILDE_RE = re.compile(r"~/")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HEX32_RE = re.compile(r"\b[0-9a-fA-F]{32}\b")

_PII_CHECKS = (
    ("home-path", _HOME_RE),
    ("tilde-path", _TILDE_RE),
    ("email", _EMAIL_RE),
    ("hex32-id", _HEX32_RE),
)


@dataclass(frozen=True)
class PiiFinding:
    path: str
    line: int
    kind: str
    snippet: str

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"{self.path}:{self.line}: [{self.kind}] {self.snippet!r}"


def _iter_files(root: Path, dirs: tuple[str, ...]):
    """Yield (relpath, absolute Path) for every regular file under ``dirs``.

    ``relpath`` is POSIX-style and relative to ``root``, so the same
    exemption sets work whether ``root`` is the real repo or a tmp_path
    fixture laid out with the same ``tests/fakes``/``tests/fixtures``
    substructure.
    """
    for rel_dir in dirs:
        base = root / rel_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            yield path.relative_to(root).as_posix(), path


def scan_pii(root: Path, dirs: tuple[str, ...] = FIXTURE_DIRS) -> list[PiiFinding]:
    """Scan ``dirs`` under ``root`` for leaked personal data.

    Skips files listed in ``PII_EXEMPT_FILES`` and anything that fails to
    decode as UTF-8 text (binary fixtures aren't expected here, but a
    stray one must not crash the scan).
    """
    findings: list[PiiFinding] = []
    for relpath, path in _iter_files(root, dirs):
        if relpath in PII_EXEMPT_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in _PII_CHECKS:
                match = pattern.search(line)
                if match:
                    findings.append(PiiFinding(relpath, lineno, kind, match.group(0)))
    return findings


def is_transcript_path(relpath: str) -> bool:
    """Is ``relpath`` a recorded-transcript data file subject to the header rule?

    True for ``*.jsonl``/``*.ndjson``/``*.txt`` files under
    ``tests/fakes/``/``tests/fixtures/``, excluding the directories and
    files this module documents as exempt.
    """
    if not relpath.startswith(FIXTURE_DIRS):
        return False
    if relpath in TRANSCRIPT_EXEMPT_FILES:
        return False
    if any(relpath == d or relpath.startswith(d + "/") for d in TRANSCRIPT_EXEMPT_DIRS):
        return False
    return relpath.endswith(TRANSCRIPT_SUFFIXES)


def has_recorded_header(text: str) -> bool:
    return bool(HEADER_RE.search(text))


def scan_missing_headers(root: Path, dirs: tuple[str, ...] = FIXTURE_DIRS) -> list[str]:
    """Return the relpaths of transcript files missing the recorded-from header."""
    missing: list[str] = []
    for relpath, path in _iter_files(root, dirs):
        if not is_transcript_path(relpath):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            missing.append(relpath)
            continue
        if not has_recorded_header(text):
            missing.append(relpath)
    return missing


# ---------------------------------------------------------------------------
# Real-tree tests: the actual repo, scanned the way this module scans it,
# must currently pass both checks.
# ---------------------------------------------------------------------------


def test_repo_fixtures_have_no_pii() -> None:
    findings = scan_pii(REPO_ROOT)
    assert findings == [], [str(f) for f in findings]


def test_repo_transcripts_have_recorded_from_header() -> None:
    missing = scan_missing_headers(REPO_ROOT)
    assert missing == [], missing


# ---------------------------------------------------------------------------
# Unit tests: point the scanners at a temporary directory holding a
# deliberately dirty fixture and assert each rule actually fires.
# ---------------------------------------------------------------------------


def _make_tree(tmp_path: Path) -> Path:
    (tmp_path / "tests" / "fakes").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    return tmp_path


def test_home_path_is_caught(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    (root / "tests" / "fixtures" / "dirty.txt").write_text(
        "cd /home/alice/project && run\n", encoding="utf-8"
    )
    findings = scan_pii(root)
    assert any(f.kind == "home-path" for f in findings), findings


def test_tilde_path_is_caught(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    (root / "tests" / "fixtures" / "dirty.txt").write_text(
        "source ~/.bashrc before running\n", encoding="utf-8"
    )
    findings = scan_pii(root)
    assert any(f.kind == "tilde-path" for f in findings), findings


def test_email_address_is_caught(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    (root / "tests" / "fixtures" / "dirty.txt").write_text(
        "contact operator at jane.doe@example.com for access\n", encoding="utf-8"
    )
    findings = scan_pii(root)
    assert any(f.kind == "email" for f in findings), findings


def test_hex32_id_is_caught(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    (root / "tests" / "fixtures" / "dirty.txt").write_text(
        "session=deadbeefdeadbeefdeadbeefdeadbeef\n", encoding="utf-8"
    )
    findings = scan_pii(root)
    assert any(f.kind == "hex32-id" for f in findings), findings


def test_transcript_missing_header_is_caught(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    (root / "tests" / "fakes" / "session.jsonl").write_text(
        '{"kind": "text_delta", "text": "hi"}\n', encoding="utf-8"
    )
    missing = scan_missing_headers(root)
    assert "tests/fakes/session.jsonl" in missing


def test_transcript_with_header_is_not_flagged(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    (root / "tests" / "fakes" / "session.jsonl").write_text(
        "# recorded-from: claude 1.2.3\n" '{"kind": "text_delta", "text": "hi"}\n',
        encoding="utf-8",
    )
    missing = scan_missing_headers(root)
    assert "tests/fakes/session.jsonl" not in missing


def test_platform_fixture_is_exempt_from_header_rule(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    platform_dir = root / "tests" / "fixtures" / "platform" / "orin" / "proc"
    platform_dir.mkdir(parents=True)
    (platform_dir / "meminfo.txt").write_text("MemTotal: 65850000 kB\n", encoding="utf-8")
    missing = scan_missing_headers(root)
    assert missing == []


def test_capture_fixture_is_exempt_from_header_rule(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    capture_dir = root / "tests" / "fixtures" / "capture"
    capture_dir.mkdir(parents=True)
    (capture_dir / "session.txt").write_text("raw terminal bytes\n", encoding="utf-8")
    missing = scan_missing_headers(root)
    assert missing == []


def test_redact_corpus_is_exempt_from_both_rules(tmp_path: Path) -> None:
    root = _make_tree(tmp_path)
    fixtures_dir = root / "tests" / "fixtures"
    (fixtures_dir / "redact_corpus.txt").write_text(
        "postgres://dbuser:s3cr3tEXAMPLEpw000@db.example.com:5432/app\n", encoding="utf-8"
    )
    assert scan_pii(root) == []
    assert scan_missing_headers(root) == []


def test_executable_fake_binary_without_extension_is_not_a_transcript() -> None:
    # tests/fakes/* today are executable fake CLI binaries with no
    # extension; the header rule must not apply to them.
    assert is_transcript_path("tests/fakes/claude") is False


def test_jsonl_and_ndjson_and_txt_under_fakes_or_fixtures_are_transcripts() -> None:
    assert is_transcript_path("tests/fakes/session.jsonl") is True
    assert is_transcript_path("tests/fixtures/session.ndjson") is True
    assert is_transcript_path("tests/fixtures/session.txt") is True


def test_files_outside_the_fixture_dirs_are_never_transcripts() -> None:
    assert is_transcript_path("nvsh/agent/claude.py") is False
    assert is_transcript_path("docs/session.jsonl") is False


# ---------------------------------------------------------------------------
# Acceptance criterion 2: scripts/scan-secrets.py covers tests/fixtures. A
# deliberately dirty fixture placed at a tests/fixtures-shaped path in a temp
# dir must be flagged, and the CLI invocation over it must exit non-zero.
# ---------------------------------------------------------------------------


def test_scan_secrets_covers_a_dirty_fixture_under_tests_fixtures(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "tests" / "fixtures" / "platform" / "planted"
    fixtures_dir.mkdir(parents=True)
    planted = fixtures_dir / "leaked.json"
    # Assembled at runtime so the planted key never sits literally in this
    # file: scripts/scan-secrets.py scans the whole tracked tree, this test
    # included, and must not trip over its own bait.
    key_name = "api" + "Key"
    planted_value = "sk-" + "live" + "AbCdEfGhIjKlMnOpQrStUvWxYz1234"
    planted.write_text(json.dumps({key_name: planted_value}) + "\n", encoding="utf-8")

    findings = scan_secrets.scan_paths([str(planted)])
    assert any(f.kind == "credential" for f in findings), findings


def test_scan_secrets_cli_fails_on_dirty_fixture_under_tests_fixtures(tmp_path: Path) -> None:
    import subprocess

    fixtures_dir = tmp_path / "tests" / "fixtures" / "platform" / "planted"
    fixtures_dir.mkdir(parents=True)
    planted = fixtures_dir / "leaked.json"
    planted.write_text('{"token": "ghp_' + "A" * 40 + '"}\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCAN_SECRETS_SCRIPT), str(planted)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "finding" in result.stderr
