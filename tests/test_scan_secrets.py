"""Tests for scripts/scan-secrets.py.

The scan is the real deliverable behind t6: it must catch API-key-shaped
strings and non-localhost endpoints in tracked config, without flagging
every URL in prose docs. Both directions are exercised here:

- a positive case that plants a real-shaped secret/endpoint and asserts the
  scanner catches it, and
- a negative case that runs the scanner over this repo's own tracked files
  (the actual CI invocation) and asserts it comes back clean.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "scan-secrets.py"

_spec = importlib.util.spec_from_file_location("scan_secrets", SCAN_SCRIPT)
scan_secrets = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["scan_secrets"] = scan_secrets
_spec.loader.exec_module(scan_secrets)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Negative case: the real repo, scanned the way CI scans it, is clean.
# ---------------------------------------------------------------------------


def test_clean_repo_passes_via_cli() -> None:
    """This is exactly the CI invocation: no arguments, all tracked files."""
    result = _run_cli()
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_clean_repo_has_no_findings_via_api() -> None:
    findings = scan_secrets.scan_paths(scan_secrets._tracked_files())
    assert findings == [], [str(f) for f in findings]


# ---------------------------------------------------------------------------
# Positive case: a planted secret is caught. These write into a throwaway
# tmp_path fixture, never into the tracked tree, and run scan_paths directly
# against that fixture file (an absolute path, so it never touches git
# ls-files or the real repo).
# ---------------------------------------------------------------------------


def _write_and_scan(tmp_path: Path, content: str) -> list:
    planted = tmp_path / "planted.json"
    planted.write_text(content, encoding="utf-8")
    text = planted.read_text(encoding="utf-8")
    findings = []
    findings.extend(scan_secrets._scan_credentials(str(planted), text))
    findings.extend(scan_secrets._scan_endpoints(str(planted), text))
    return findings


def test_planted_aws_key_is_caught(tmp_path: Path) -> None:
    findings = _write_and_scan(tmp_path, 'aws_key = "AKIAABCDEFGHIJKLMNOP"\n')
    assert any(f.kind == "credential" for f in findings)


def test_planted_generic_api_key_assignment_is_caught(tmp_path: Path) -> None:
    findings = _write_and_scan(tmp_path, '{"apiKey": "sk-liveAbCdEfGhIjKlMnOpQrStUvWxYz1234"}\n')
    assert any(f.kind == "credential" for f in findings)


def test_planted_private_key_block_is_caught(tmp_path: Path) -> None:
    findings = _write_and_scan(
        tmp_path, "-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n-----END RSA PRIVATE KEY-----\n"
    )
    assert any(f.kind == "credential" for f in findings)


def test_planted_non_localhost_base_url_is_caught(tmp_path: Path) -> None:
    findings = _write_and_scan(
        tmp_path,
        """{
          "modelProviders": {
            "openai": [
              {"id": "leaked", "envKey": "X", "baseUrl": "https://internal.corp.example.net/v1"}
            ]
          }
        }
        """,
    )
    assert any(f.kind == "endpoint" for f in findings)


def test_planted_secret_fails_the_cli(tmp_path: Path) -> None:
    """End-to-end: pointing the CLI at a planted file exits non-zero."""
    planted = tmp_path / "leak.json"
    planted.write_text('{"token": "ghp_' + "A" * 40 + '"}\n', encoding="utf-8")
    result = _run_cli(str(planted))
    assert result.returncode == 1
    assert "finding" in result.stderr


# ---------------------------------------------------------------------------
# Negative-shaped inputs that must NOT be flagged (placeholders, env refs,
# localhost, CI expression syntax, docs prose).
# ---------------------------------------------------------------------------


def test_localhost_base_url_is_allowed(tmp_path: Path) -> None:
    findings = _write_and_scan(
        tmp_path,
        '{"modelProviders": {"openai": [{"id": "x", "baseUrl": "http://localhost:8000/v1"}]}}\n',
    )
    assert findings == []


def test_env_var_reference_is_not_flagged(tmp_path: Path) -> None:
    findings = _write_and_scan(tmp_path, '{"apiKey": "$MY_API_TOKEN"}\n')
    assert findings == []


def test_github_actions_secrets_expression_is_not_flagged(tmp_path: Path) -> None:
    planted = tmp_path / "workflow.txt"
    planted.write_text("SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}\n", encoding="utf-8")
    text = planted.read_text(encoding="utf-8")
    findings = scan_secrets._scan_credentials(str(planted), text)
    assert findings == []


def test_placeholder_value_is_not_flagged(tmp_path: Path) -> None:
    findings = _write_and_scan(tmp_path, '{"apiKey": "your-api-key-here-please-fill-in"}\n')
    assert findings == []


def test_prose_markdown_with_many_urls_is_not_flagged(tmp_path: Path) -> None:
    """A .md file never parses as JSON, so the endpoint check never applies."""
    planted = tmp_path / "doc.md"
    text = (
        "See https://github.com/agentculture/guildmaster and "
        "https://sonarcloud.io and https://example.com/api for more.\n"
    )
    planted.write_text(text, encoding="utf-8")
    findings = []
    findings.extend(scan_secrets._scan_credentials(str(planted), text))
    findings.extend(scan_secrets._scan_endpoints(str(planted), text))
    assert findings == []


# ---------------------------------------------------------------------------
# The shipped .qwen/settings.json.example itself must be clean and must not
# be a live, tracked settings.json (it's git-ignored once created for real).
# ---------------------------------------------------------------------------


def test_qwen_settings_example_is_tracked_and_clean() -> None:
    example = REPO_ROOT / ".qwen" / "settings.json.example"
    assert example.is_file()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".qwen/settings.json.example"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, ".qwen/settings.json.example must be tracked"
    text = example.read_text(encoding="utf-8")
    findings = []
    findings.extend(scan_secrets._scan_credentials(".qwen/settings.json.example", text))
    findings.extend(scan_secrets._scan_endpoints(".qwen/settings.json.example", text))
    assert findings == []


def test_qwen_settings_real_file_is_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".qwen/settings.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, ".qwen/settings.json (the real file) must be gitignored"


# ---------------------------------------------------------------------------
# Credential values whose alphabet includes ordinary punctuation. A value
# regex restricted to a base64-ish alphabet stopped matching at the first
# `@` / `:` / `%` / `=` / `?`, so these walked straight through the gate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "Tr0ub4dor&3@Correct-Horse!",
        "p%40ssw0rd%3Avalue%2Fwith%2Fescapes",
        "db://admin:s3cr3t@10.0.0.4:5432/prod",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0=?sig=abc",
        "aB3$dE6^gH9*jK2!mN5@pQ8&",
    ],
)
def test_punctuation_bearing_credentials_are_caught(tmp_path: Path, value: str) -> None:
    """Punctuation before the 20th character must not buy an exemption."""
    findings = _write_and_scan(tmp_path, f'password = "{value}"\n')
    assert any(f.kind == "credential" for f in findings), value


def test_an_unquoted_punctuation_bearing_credential_is_caught(tmp_path: Path) -> None:
    planted = tmp_path / "app.env"
    text = "API_KEY=aB3$dE6^gH9*jK2!mN5@pQ8&rS1\n"
    planted.write_text(text, encoding="utf-8")
    findings = scan_secrets._scan_credentials(str(planted), text)
    assert any(f.kind == "credential" for f in findings)


# ---------------------------------------------------------------------------
# Placeholder exemption is a whole-value judgement. A high-entropy literal
# that merely CONTAINS an English label like `example` or `fake` is still a
# committed secret.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "fake-Xk9Lm2Pq7Rt4Vw8Zb3Nc6Yd1",
        "exampleXk9Lm2Pq7Rt4Vw8Zb3Nc6Y",
        "dummy_A1b2C3d4E5f6G7h8I9j0K1l2",
        "PlaceholderZq7Wx2Ey9Rt4Uv8Ip3",
        "your-key-A1b2C3d4E5f6G7h8I9j0",
    ],
)
def test_a_real_secret_containing_a_placeholder_word_is_still_caught(
    tmp_path: Path, value: str
) -> None:
    findings = _write_and_scan(tmp_path, f'{{"apiKey": "{value}"}}\n')
    assert any(f.kind == "credential" for f in findings), value


@pytest.mark.parametrize(
    "value",
    [
        "your-api-key-here-please-fill-in",
        "changeme-changeme-changeme-x",
        "replace.with.your.token.value",
        "xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "00000000000000000000000000",
    ],
)
def test_written_out_placeholders_stay_exempt(tmp_path: Path, value: str) -> None:
    """The exemption must still cover the samples docs actually ship."""
    findings = _write_and_scan(tmp_path, f'{{"apiKey": "{value}"}}\n')
    assert findings == [], (value, [str(f) for f in findings])


# ---------------------------------------------------------------------------
# Endpoint hosts are parsed, not pattern-matched: a bracketed IPv6 authority
# took the no-match branch under the old host regex and passed unexamined.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://[2001:db8::1]:8080/v1",
        "https://[fd00:1234::abcd]/v1",
        "http://[2001:db8::1]/v1",
    ],
)
def test_remote_ipv6_endpoints_are_caught(tmp_path: Path, url: str) -> None:
    findings = _write_and_scan(tmp_path, '{"baseUrl": "' + url + '"}\n')
    assert any(f.kind == "endpoint" for f in findings), url


@pytest.mark.parametrize(
    "url",
    [
        "http://[::1]:8000/v1",
        "http://[::1]/v1",
        "http://127.0.0.1:8000/v1",
        "http://LOCALHOST:8000/v1",
    ],
)
def test_loopback_endpoints_are_allowed(tmp_path: Path, url: str) -> None:
    findings = _write_and_scan(tmp_path, '{"baseUrl": "' + url + '"}\n')
    assert findings == [], (url, [str(f) for f in findings])


def test_a_non_http_scheme_is_not_treated_as_an_endpoint(tmp_path: Path) -> None:
    """The endpoint check is scoped to http(s) URLs, as documented."""
    findings = _write_and_scan(tmp_path, '{"endpoint": "unix:///var/run/thing.sock"}\n')
    assert findings == []
