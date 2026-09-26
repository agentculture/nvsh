"""README acceptance test (issue #64, task t19).

``evals/README.md``'s "Fixture run" section is not just documentation: this
test extracts exactly the fenced ``bash`` commands under that heading and
runs them for real, so a reader really can go from a clean checkout to a
complete gate run using only the commands in the README (acceptance
criterion 2). The fixture data those commands point at
(``evals/tool_jev/fixtures/readme/``) is entirely synthetic — three made-up
cases and made-up saved predictions, never real request text — and the
fixture manifest carries no reference models, no judges and no budgets, so
the whole run needs no network call and no provider key.

The run directory the extracted commands create is a fresh temporary
directory outside this repository (``mktemp -d``, run by the extracted
script itself): the runner refuses a run dir inside any git worktree, so
this is also a regression check that the extracted commands still satisfy
that rule.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404 - fixed argv below, no shell=True; script text is our own README
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "evals" / "README.md"
HAS_UV = shutil.which("uv") is not None

HEADING = "## Fixture run"


def _fixture_run_section() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(HEADING)
    rest = text[start + len(HEADING) :]
    match = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: match.start()] if match else rest


def _fenced_bash_blocks(section: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", section, re.DOTALL)


def test_readme_has_a_fixture_run_section_with_commands():
    section = _fixture_run_section()
    blocks = _fenced_bash_blocks(section)
    assert blocks, "evals/README.md's Fixture run section has no fenced bash command block"
    script = "\n".join(blocks)
    assert "NVSH_EVALS_RUN_DIR" in script
    assert "NVSH_EVALS_PRIVATE_ROOT" in script
    assert "NVSH_EVALS_MANIFEST" in script
    # This is the acceptance test for the "no ~ or /home paths" rule: the
    # README's own commands must not lean on a home-directory shortcut.
    assert "~/" not in script
    assert "/home" not in script


def test_readme_has_no_tilde_or_home_paths():
    """No home-shorthand (``~/...``) or ``/home/...`` path anywhere in the README.

    A bare ``~`` used as "approximately" in the cost table is fine; only a
    home-directory *path* is disallowed.
    """
    text = README.read_text(encoding="utf-8")
    assert "~/" not in text
    assert "/home" not in text


def test_fixture_run_commands_reach_a_real_run_outside_the_repo(tmp_path):
    """Run the README's own commands for real: acceptance criterion 2."""
    if not HAS_UV:
        pytest.fail("`uv` is not on PATH — cannot run the README's fixture-run commands.")
    script = "\n".join(_fenced_bash_blocks(_fixture_run_section()))
    marker = tmp_path / "run-dir.txt"
    # The extracted script already exports NVSH_EVALS_RUN_DIR to a fresh
    # `mktemp -d` directory outside the repo; capture that path afterwards
    # so this test can assert on result.json/report.md directly, in
    # addition to the `ls` the README's own script already runs.
    wrapped = script + f'\nprintf "%s" "$NVSH_EVALS_RUN_DIR" > "{marker}"\n'
    result = subprocess.run(  # nosec B603 - fixed argv, no shell=True; input is our own README
        ["bash", "-euo", "pipefail", "-c", wrapped],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"README fixture-run commands failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "complete:" in result.stdout
    run_dir = Path(marker.read_text(encoding="utf-8").strip())
    try:
        assert (run_dir / "result.json").is_file()
        assert (run_dir / "report.md").is_file()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
