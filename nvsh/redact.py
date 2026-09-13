"""Secret redaction for anything nvsh reports outward (device context, logs).

``redact()`` is the single choke point required by the device-context rule in
``CLAUDE.md``: "Redact tokens (HF_TOKEN=, --api-key, Authorization:, .env)
before anything leaves the process." Every caller that assembles text bound
for the agent backend, ``--show-context``, or a log line must pass it through
here first.

Stdlib ``re`` only (no third-party dependency; ``dependencies = []`` in
``pyproject.toml`` stays empty). Rules are declared as a flat list of
``(name, pattern)`` pairs in :data:`PATTERNS` so ``--show-context`` (or any
caller) can report which rules fired via :func:`redact_report`, without
re-deriving the rule set.

Design notes:

* Every rule replaces what it matches with a **typed marker**,
  ``<REDACTED:rule_name>``, never a blank or a generic ``***``. That keeps
  the redacted text legible (an operator or agent can see *that* a secret
  was there and *what kind*) while guaranteeing no secret substring
  survives.
* Rules that replace a value in place (env assignments, CLI flags,
  ``Authorization:`` headers, JSON fields, URL credentials) keep their
  surrounding structure (the key name, the flag, the scheme) and only
  swap out the secret portion, so the marker reads in context.
* Because every rule's replacement text is a **fixed string** derived only
  from the rule's own name (never from the matched content), re-running
  :func:`redact` on already-redacted text is a no-op: a rule either no
  longer finds the structural shape it needs (the secret is gone) or it
  matches the marker itself and replaces it with the identical marker.
  This is what makes ``redact`` idempotent (see
  ``tests/test_redact.py::test_redact_is_idempotent``).
* :func:`redact` takes and returns ``bytes`` and must never raise, even on
  invalid UTF-8. It round-trips through ``str`` using the
  ``surrogateescape`` error handler (the same trick ``os.fsdecode`` /
  ``os.fsencode`` use), so arbitrary bytes decode without loss and without
  exception, get scanned as text, and re-encode losslessly for any byte
  sequence the rules didn't touch.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# A rule's replacement function receives the ``re.Match`` and returns the
# full text that should stand in place of the match.
_ReplFunc = Callable[[re.Match[str]], str]


def _marker(name: str) -> str:
    return f"<REDACTED:{name}>"


def _fixed_replacer(name: str) -> _ReplFunc:
    """Replace the whole match with the rule's fixed marker."""
    marker = _marker(name)
    return lambda match: marker


def _private_key_repl(match: re.Match[str]) -> str:
    return _marker("private_key_block")


def _json_field_repl(match: re.Match[str]) -> str:
    return f'"{match.group(1)}": "{_marker("json_secret_field")}"'


def _authorization_repl(match: re.Match[str]) -> str:
    return f'{match.group(1)} {_marker("authorization_header")}'


def _cli_flag_repl(match: re.Match[str]) -> str:
    return f'{match.group(1)}{match.group(2)}{_marker("cli_flag_secret")}'


def _url_credentials_repl(match: re.Match[str]) -> str:
    return f'{match.group(1)}{_marker("url_credentials")}@'


def _env_assignment_repl(match: re.Match[str]) -> str:
    prefix, name, eq = match.group("prefix"), match.group("name"), match.group("eq")
    return f'{prefix}{name}{eq}{_marker("env_assignment")}'


# ---------------------------------------------------------------------------
# Rules, in application order. Order matters: rules that need surrounding
# context (JSON fields, headers, CLI flags, URLs, env assignments) run
# first, so they redact the *whole* secret in its context before the bare
# token-shape rules would otherwise only catch a fragment of it.
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, re.Pattern[str], _ReplFunc]] = [
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
            r".*?"
            r"-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        _private_key_repl,
    ),
    (
        "json_secret_field",
        re.compile(r'(?i)"(apiKey|api_key|token|secret|password)"\s*:\s*"([^"]*)"'),
        _json_field_repl,
    ),
    (
        "authorization_header",
        re.compile(r"(?i)\b(Authorization:\s*(?:Bearer|Basic))\s+(\S+)"),
        _authorization_repl,
    ),
    (
        "cli_flag_secret",
        re.compile(r"(--api-key|--token)(=|\s+)(\S+)"),
        _cli_flag_repl,
    ),
    (
        "url_credentials",
        re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)([^\s:/@]+):([^\s@]+)@"),
        _url_credentials_repl,
    ),
    (
        "env_assignment",
        re.compile(
            r"(?im)^(?P<prefix>[ \t]*(?:export[ \t]+)?)"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|PASS)[A-Za-z0-9_]*)"
            r"(?P<eq>[ \t]*=[ \t]*)"
            r"(?P<value>\S+)"
        ),
        _env_assignment_repl,
    ),
    (
        "hf_token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        _fixed_replacer("hf_token"),
    ),
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        _fixed_replacer("openai_key"),
    ),
    (
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        _fixed_replacer("github_token"),
    ),
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{8,}\b"),
        _fixed_replacer("aws_access_key"),
    ),
    (
        "slack_token",
        re.compile(r"\bxox[abp]-[A-Za-z0-9-]{10,}\b"),
        _fixed_replacer("slack_token"),
    ),
]

#: Public, introspectable rule set: ``[(name, compiled_pattern), ...]`` in
#: application order. ``--show-context`` (and tests) use this to report which
#: rules exist / fired, without reaching into the private replacement
#: functions above.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [(name, pattern) for name, pattern, _ in _RULES]


def redact_report(data: bytes) -> tuple[bytes, list[str]]:
    """Redact ``data`` and report which rule names matched at least once.

    Never raises: invalid UTF-8 is round-tripped losslessly via the
    ``surrogateescape`` error handler rather than raising or dropping bytes.
    """
    text = data.decode("utf-8", errors="surrogateescape")
    fired: list[str] = []
    for name, pattern, repl in _RULES:
        new_text, count = pattern.subn(repl, text)
        if count:
            fired.append(name)
        text = new_text
    return text.encode("utf-8", errors="surrogateescape"), fired


def redact(data: bytes) -> bytes:
    """Redact secrets from ``data``, returning the redacted bytes.

    Idempotent: ``redact(redact(data)) == redact(data)`` for any ``data``,
    because every rule's replacement is a fixed marker string derived only
    from the rule's name (see module docstring). Never raises.
    """
    redacted, _fired = redact_report(data)
    return redacted
