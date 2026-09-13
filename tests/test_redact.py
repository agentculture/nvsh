"""Tests for nvsh.redact — the single choke point for outbound secret redaction.

Covers plan task t3 (spec claim c11, honesty condition h10):

* a corpus fixture exercising every documented secret shape, asserting the
  typed marker replaces it and no secret substring survives;
* ``redact(bytes) -> bytes`` never raises on invalid UTF-8 or escape
  sequences, and is idempotent (a stdlib property-style test — no
  ``hypothesis`` dependency, since the runtime stays dependency-free);
* the collector API does not exist yet (t9/t8 add it later) and, per the
  task instruction, never reads shell rc files — asserted here by scanning
  ``nvsh/`` source so the invariant stays true as later tasks land.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from nvsh.redact import PATTERNS, redact, redact_report

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_PATH = FIXTURES_DIR / "redact_corpus.txt"

# Exact secret substrings planted in tests/fixtures/redact_corpus.txt. None of
# these are real credentials (see the corpus file header); every one of them
# must be gone from the redacted output.
CORPUS_SECRETS: list[str] = [
    "hf_EXAMPLE00000000000000000000000000",
    "sk-EXAMPLE0000000000000000000000000000000000",
    "abcdEXAMPLEsecretvalue0000000000",
    "hunter2EXAMPLEpassword000",
    "AKIAEXAMPLE00000000",
    "ghp_EXAMPLE0000000000000000000000000000",
    "sk_test_EXAMPLE00000000000000000000",
    "sk-EXAMPLE0000000000000000000000000001",
    "ghp_EXAMPLE0000000000000000000000000001",
    "sk-EXAMPLE0000000000000000000000000002",
    "ZXhhbXBsZTpwYXNzd29yZC1FWEFNUExFLTAwMDA=",
    "sk-EXAMPLE0000000000000000000000000003",
    "sk-EXAMPLE0000000000000000000000000004",
    "ghp_EXAMPLE0000000000000000000000000002",
    "EXAMPLEsecretvalue00000000000000",
    "hunter2EXAMPLEpassword001",
    "hunter2EXAMPLEpw",
    "s3cr3tEXAMPLEpw000",
    "xoxb-EXAMPLE0000000000-EXAMPLE0000000000-EXAMPLE00000000000000000000",
]

# The SSH private key block body (base64-ish payload lines) must not survive
# either; checked separately since it spans multiple lines.
CORPUS_PRIVATE_KEY_LINES = [
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW",
    "QyNTUxOQAAACBFWEFNUExFRVhBTVBMRUVYQU1QTEVFWEFNUExFRVhBTVBMRQAAAKBEXAMP",
    "LEEXAMPLEEXAMPLEEXAMPLEEXAMPLEEXAMPLEEXAMPLEEXAMPLEEXAMPLEEXAMPLE0000A",
]


def _load_corpus_bytes() -> bytes:
    return CORPUS_PATH.read_bytes()


# --- corpus: every secret shape gets redacted -------------------------------


def test_corpus_file_exists_and_is_nonempty() -> None:
    assert CORPUS_PATH.is_file()
    assert len(_load_corpus_bytes()) > 0


def test_corpus_no_secret_substring_survives() -> None:
    redacted = redact(_load_corpus_bytes())
    text = redacted.decode("utf-8")
    for secret in CORPUS_SECRETS:
        assert secret not in text, f"secret substring survived redaction: {secret!r}"
    for line in CORPUS_PRIVATE_KEY_LINES:
        assert line not in text, f"private key body line survived redaction: {line!r}"


def test_corpus_produces_typed_markers() -> None:
    redacted, fired = redact_report(_load_corpus_bytes())
    text = redacted.decode("utf-8")
    # Every documented rule should have fired at least once against the
    # corpus (the corpus is written to exercise all of them).
    fired_names = set(fired)
    expected_names = {name for name, _pattern in PATTERNS}
    missing = expected_names - fired_names
    assert not missing, f"rules that never fired against the corpus: {missing}"
    # Typed markers, not a blank / generic "***".
    assert re.search(r"<REDACTED:[a-z_]+>", text)


def test_env_assignment_redacted_with_typed_marker() -> None:
    redacted = redact(b"HF_TOKEN=hf_EXAMPLE00000000000000000000000000\n")
    text = redacted.decode("utf-8")
    assert "hf_EXAMPLE00000000000000000000000000" not in text
    assert "HF_TOKEN=<REDACTED:env_assignment>" in text


def test_cli_flag_api_key_space_form() -> None:
    redacted = redact(b"curl --api-key sk-EXAMPLE0000000000000000000000000000 https://x\n")
    text = redacted.decode("utf-8")
    assert "sk-EXAMPLE0000000000000000000000000000" not in text
    assert "--api-key <REDACTED:cli_flag_secret>" in text


def test_cli_flag_api_key_equals_form() -> None:
    redacted = redact(b"curl --api-key=sk-EXAMPLE0000000000000000000000000001 https://x\n")
    text = redacted.decode("utf-8")
    assert "sk-EXAMPLE0000000000000000000000000001" not in text
    assert "--api-key=<REDACTED:cli_flag_secret>" in text


def test_authorization_bearer_redacted() -> None:
    redacted = redact(b"Authorization: Bearer sk-EXAMPLE0000000000000000000000000002\n")
    text = redacted.decode("utf-8")
    assert "sk-EXAMPLE0000000000000000000000000002" not in text
    assert "Authorization: Bearer <REDACTED:authorization_header>" in text


def test_authorization_basic_redacted() -> None:
    redacted = redact(b"Authorization: Basic ZXhhbXBsZTpwYXNzd29yZC1FWEFNUExFLTAwMDA=\n")
    text = redacted.decode("utf-8")
    assert "ZXhhbXBsZTpwYXNzd29yZC1FWEFNUExFLTAwMDA=" not in text
    assert "Authorization: Basic <REDACTED:authorization_header>" in text


def test_private_key_block_redacted() -> None:
    block = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        + "\n".join(CORPUS_PRIVATE_KEY_LINES)
        + "\n-----END OPENSSH PRIVATE KEY-----\n"
    )
    redacted = redact(block.encode("utf-8"))
    text = redacted.decode("utf-8")
    for line in CORPUS_PRIVATE_KEY_LINES:
        assert line not in text
    assert "<REDACTED:private_key_block>" in text
    assert "BEGIN" not in text
    assert "END" not in text


def test_json_api_key_field_redacted() -> None:
    redacted = redact(b'{"apiKey": "sk-EXAMPLE0000000000000000000000000003"}')
    text = redacted.decode("utf-8")
    assert "sk-EXAMPLE0000000000000000000000000003" not in text
    assert '"apiKey": "<REDACTED:json_secret_field>"' in text


def test_json_snake_case_fields_redacted() -> None:
    for field, value in (
        ("api_key", "sk-EXAMPLE0000000000000000000000000004"),
        ("token", "ghp_EXAMPLE0000000000000000000000000002"),
        ("secret", "EXAMPLEsecretvalue00000000000000"),
        ("password", "hunter2EXAMPLEpassword001"),
    ):
        payload = f'{{"{field}": "{value}"}}'.encode("utf-8")
        redacted = redact(payload)
        text = redacted.decode("utf-8")
        assert value not in text, f"{field} value survived: {value!r}"
        assert f'"{field}": "<REDACTED:json_secret_field>"' in text


def test_url_credentials_redacted() -> None:
    redacted = redact(b"https://admin:hunter2EXAMPLEpw@internal.example.com/api\n")
    text = redacted.decode("utf-8")
    assert "admin:hunter2EXAMPLEpw" not in text
    assert "hunter2EXAMPLEpw" not in text
    assert "https://<REDACTED:url_credentials>@internal.example.com/api" in text


def test_token_shapes_redacted_standalone() -> None:
    cases = {
        "hf_EXAMPLE00000000000000000000000000": "hf_token",
        "sk-EXAMPLE0000000000000000000000000000000000": "openai_key",
        "ghp_EXAMPLE0000000000000000000000000000": "github_token",
        "AKIAEXAMPLE00000000": "aws_access_key",
        "xoxb-EXAMPLE0000000000-EXAMPLE0000000000-EXAMPLE00000000000000000000": "slack_token",
    }
    for secret, rule in cases.items():
        redacted = redact(secret.encode("utf-8"))
        text = redacted.decode("utf-8")
        assert secret not in text
        assert f"<REDACTED:{rule}>" in text


def test_non_secret_text_passes_through_unchanged() -> None:
    text = b"ls -la /home/spark && echo done\n"
    assert redact(text) == text


# --- redact(bytes) -> bytes: never raises ----------------------------------


def test_redact_handles_invalid_utf8_without_raising() -> None:
    data = b"\xff\xfe\x00invalid utf-8 HF_TOKEN=hf_EXAMPLE00000000000000000000000000\x80\x81"
    redacted = redact(data)  # must not raise
    assert isinstance(redacted, bytes)
    assert b"hf_EXAMPLE00000000000000000000000000" not in redacted


def test_redact_handles_escape_sequences_without_raising() -> None:
    data = b"line1\\nline2\\t\x1b[31mred\x1b[0m HF_TOKEN=hf_EXAMPLE00000000000000000000000000\n"
    redacted = redact(data)  # must not raise
    assert isinstance(redacted, bytes)
    assert b"hf_EXAMPLE00000000000000000000000000" not in redacted


def test_redact_report_matches_redact_output() -> None:
    data = _load_corpus_bytes()
    redacted_bytes, fired = redact_report(data)
    assert redacted_bytes == redact(data)
    assert fired  # the corpus fires at least one rule
    assert set(fired).issubset({name for name, _pattern in PATTERNS})


# --- property test: idempotence over generated inputs -----------------------

_RANDOM_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" "_-=:./@\"'{} \t\n"
)

_SEED_SNIPPETS = [
    b"",
    b"plain text with no secrets at all",
    b"HF_TOKEN=hf_EXAMPLE00000000000000000000000000",
    b'{"apiKey": "sk-EXAMPLE0000000000000000000000000003"}',
    b"Authorization: Bearer sk-EXAMPLE0000000000000000000000000002",
    b"curl --api-key=sk-EXAMPLE0000000000000000000000000001 https://x",
    b"https://admin:hunter2EXAMPLEpw@internal.example.com/api",
    b"-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\n",
    b"\xff\xfe not valid utf-8 \x80\x81",
    b"nested <REDACTED:hf_token> marker text already present",
]


def _random_bytes(rng: random.Random, length: int) -> bytes:
    """Generate length random bytes, including values outside valid UTF-8."""
    return bytes(rng.randrange(0, 256) for _ in range(length))


def _random_text_snippet(rng: random.Random, length: int) -> bytes:
    return "".join(rng.choice(_RANDOM_ALPHABET) for _ in range(length)).encode("utf-8")


def test_redact_is_idempotent_over_generated_inputs() -> None:
    """Stdlib property-style test (no hypothesis dependency): for many
    generated inputs, redacting an already-redacted payload changes nothing.
    """
    rng = random.Random(20260913)  # fixed seed: deterministic, reproducible CI

    inputs: list[bytes] = list(_SEED_SNIPPETS)
    inputs.append(_load_corpus_bytes())
    for _ in range(200):
        length = rng.randrange(0, 120)
        if rng.random() < 0.5:
            inputs.append(_random_bytes(rng, length))
        else:
            inputs.append(_random_text_snippet(rng, length))

    for data in inputs:
        once = redact(data)
        twice = redact(once)
        assert twice == once, f"redact not idempotent for input {data!r}: {once!r} != {twice!r}"


def test_redact_report_never_raises_on_arbitrary_bytes() -> None:
    rng = random.Random(9)
    for _ in range(100):
        data = _random_bytes(rng, rng.randrange(0, 64))
        redacted, fired = redact_report(data)  # must not raise
        assert isinstance(redacted, bytes)
        assert isinstance(fired, list)


# --- t9: the (not-yet-built) collector must never read shell rc files ------


_RC_FILE_NAMES = (".bashrc", ".bash_profile", ".profile", ".zshrc")

#: The sanctioned rc *editor* (task t21: `nvsh setup`/`uninstall`, and its
#: pure-function half in nvsh/rcfile.py) legitimately names ".bashrc" as its
#: default target -- inserting/removing the marked hook block is the whole
#: point of it, and it is not the device-context collector this invariant
#: guards against. Everything else under nvsh/ still must never reference a
#: shell rc file, so a future collector module trips this the moment it does.
_RC_EDITOR_FILES = frozenset({"nvsh/rcfile.py", "nvsh/cli/_commands/setup.py"})


def test_no_source_reads_shell_rc_files() -> None:
    """The device-context collector (added in a later task, t9/t8) must never
    open or read shell rc files — the collector reads process/proc/sysfs
    state, not the user's shell configuration. Since the collector module
    does not exist yet, this scans all of nvsh/ (excluding the sanctioned rc
    *editor*, see `_RC_EDITOR_FILES`) so the invariant is caught the moment
    it lands, rather than only once a collector module exists.
    """
    repo_root = Path(__file__).parent.parent
    nvsh_src = repo_root / "nvsh"
    assert nvsh_src.is_dir()

    offenders: list[str] = []
    for path in nvsh_src.rglob("*.py"):
        rel = path.relative_to(repo_root).as_posix()
        if rel in _RC_EDITOR_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for rc_name in _RC_FILE_NAMES:
            if rc_name in text:
                offenders.append(f"{rel}: references {rc_name!r}")

    assert not offenders, "nvsh/ source references a shell rc file:\n" + "\n".join(offenders)


def test_rc_editor_exemption_list_still_matches_real_files() -> None:
    """Guard the exemption itself: it must name real files, not a typo that
    silently exempts nothing (or everything, if too broad).
    """
    repo_root = Path(__file__).parent.parent
    for rel in _RC_EDITOR_FILES:
        assert (repo_root / rel).is_file(), f"exempted path does not exist: {rel}"
