# Fixture hygiene

`tests/test_fixture_hygiene.py` enforces two rules over `tests/fakes/` and
`tests/fixtures/`. Both are exercised against the real trees below and,
separately, against temporary dirty fixtures constructed in the test's own
unit tests — see that file for the exact scanning logic.

## No leaked personal data

Nothing under `tests/fakes/` or `tests/fixtures/` may contain a
`/home/<user>` path, a `~/` path, an e-mail address, or a 32-hex-character
id (session ids, dash-less UUIDs, etc.). These are exactly the things a
*real* recorded session could leak but a checked-in fixture must never
carry.

`tests/fixtures/redact_corpus.txt` is exempt from this rule: it is already
a deliberately fake secret-shaped corpus documented in its own header and
excluded by `scripts/scan-secrets.py`'s `SELF_EXCLUDE`, not real captured
data — the same reasoning applies here.

## Recorded transcripts need a `# recorded-from:` header

A **recorded transcript** is a transcript *data* file — `*.jsonl`,
`*.ndjson`, or `*.txt` — sitting under `tests/fakes/` or `tests/fixtures/`.
Every recorded transcript must contain a line shaped:

```text
# recorded-from: <cli> <version>
```

so a reader can tell what produced the file and against which version,
without needing to re-derive it from git blame.

This rule does **not** apply to:

- `tests/fakes/*` **executable fake binaries** (`claude`, `codex`, `nvsh`,
  `pi`, `pi_scripted`, `qwen`). These are hand-written fake CLIs used to
  drive the conformance suite without a real backend installed — they are
  Python source, not recorded output, and carry no filename extension.
- `tests/fixtures/platform/**` — recorded *device* state (sysfs/proc/
  subprocess snapshots for platform detection), not a CLI transcript.
- `tests/fixtures/capture/**` — raw terminal capture fixtures for the
  output-capture tests, not a CLI transcript.
- `tests/fixtures/redact_corpus.txt` — the deliberately fake secret corpus
  described above.

Any *new* transcript data file added under `tests/fakes/` or
`tests/fixtures/` outside those exemptions must carry the header, or
`tests/test_fixture_hygiene.py::test_repo_transcripts_have_recorded_from_header`
fails.

## Secret scanning

`scripts/scan-secrets.py` scans all tracked files, `tests/fixtures/**`
included (its `SELF_EXCLUDE` list only carves out its own source, its own
test, and the redaction corpus + test above) — see that script's module
docstring for what it checks.
