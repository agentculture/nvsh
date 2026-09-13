"""Invariants t18 (CI integration gate) must hold on every commit.

Four things, per the converged spec (c17, c20, c21, c13) and CLAUDE.md's
mesh-identity rules:

(a) the mesh-identity files (``culture.yaml``, ``.claude/skills``,
    ``.pi/skills``) are byte-identical to ``main`` — no task agent may touch
    them;
(b) ``doctor.py``'s ``_PROMPT_FILE`` mapping has not silently drifted from
    the value pinned here (``tests/test_harness_registries.py`` cross-checks
    the same table against the vendored fingerprint registry);
(c) the two registry/identity test modules this repo depends on are present;
(d) with ``pi``, ``tmux``, ``fzf`` and ``spark`` removed from ``PATH``
    entirely (not just failing to run — absent), every registered verb still
    runs to a clean diagnostic, never a Python traceback.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
from pathlib import Path

import pytest

from nvsh.cli import main
from nvsh.cli._commands.doctor import _PROMPT_FILE

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# (a) mesh-identity files untouched relative to main
# ---------------------------------------------------------------------------


def _main_ref_exists() -> bool:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "-q", "main"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


@pytest.mark.skipif(not _main_ref_exists(), reason="no local 'main' ref to diff against")
def test_mesh_identity_files_match_main() -> None:
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "diff",
            "main",
            "--",
            "culture.yaml",
            ".claude/skills",
            ".pi/skills",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"git diff failed: {proc.stderr}"
    assert proc.stdout == "", (
        "culture.yaml, .claude/skills or .pi/skills differ from main — "
        f"these must never change:\n{proc.stdout}"
    )


# ---------------------------------------------------------------------------
# (b) doctor._PROMPT_FILE pinned to its known-good value
# ---------------------------------------------------------------------------

#: Copied verbatim from ``nvsh/cli/_commands/doctor.py``. Keep in lockstep —
#: this is a tripwire, not a derived value: if the mapping in doctor.py ever
#: changes (an added/removed backend, a renamed prompt file), this literal
#: must be updated deliberately, in the same review, not silently.
_EXPECTED_PROMPT_FILE = {
    "claude": ("CLAUDE.md",),
    "colleague": ("AGENTS.colleague.md", "AGENTS.override.md", ".pi/SYSTEM.md"),
    "acp": ("AGENTS.md", "QWEN.md"),
    "codex": ("AGENTS.md",),
    "copilot": ("AGENTS.md",),
    "gemini": ("GEMINI.md",),
}


def test_doctor_prompt_file_pinned() -> None:
    assert _PROMPT_FILE == _EXPECTED_PROMPT_FILE, (
        "doctor.py's _PROMPT_FILE has drifted from the pinned mapping. "
        "tests/test_harness_registries.py checks it against the vendored "
        "fingerprint registry; update both deliberately if this is intended.\n"
        f"actual:   {_PROMPT_FILE}\nexpected: {_EXPECTED_PROMPT_FILE}"
    )


# ---------------------------------------------------------------------------
# (c) the sibling registry/identity test modules are present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_module",
    ["tests/test_pi_settings.py", "tests/test_harness_registries.py"],
)
def test_sibling_invariant_module_present(test_module: str) -> None:
    # Not re-run here (that would duplicate the suite) — just confirmed
    # present, since this invariants test assumes their guarantees hold.
    path = REPO_ROOT / test_module
    assert path.is_file(), f"{test_module} is missing; the mesh-identity invariants depend on it"


# ---------------------------------------------------------------------------
# (d) every verb still works with pi/tmux/fzf/spark absent from PATH
# ---------------------------------------------------------------------------

_STRIPPED_NAMES = {"pi", "tmux", "fzf", "spark"}
_SOURCE_DIRS = ("/usr/bin", "/bin")


def _build_stripped_path(tmp_path: Path) -> str:
    """A PATH containing only a tmp bin dir that symlinks every real
    /usr/bin and /bin binary EXCEPT pi, tmux, fzf and spark — so those four
    tools are genuinely absent (not just failing), while bash, python3,
    script and coreutils remain available.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for src_dir in _SOURCE_DIRS:
        src_path = Path(src_dir)
        if not src_path.is_dir():
            continue
        for entry in src_path.iterdir():
            if entry.name in _STRIPPED_NAMES:
                continue
            dst = bindir / entry.name
            if dst.exists():
                continue
            try:
                dst.symlink_to(entry)
            except OSError:
                continue
    for missing in _STRIPPED_NAMES:
        assert not (bindir / missing).exists()
    return str(bindir)


_VERBS: list[list[str]] = [
    ["doctor", "--json"],
    ["whoami", "--json"],
    ["overview", "--json"],
    ["cli", "overview", "--json"],
    ["agent", "list", "--json"],
    ["approve", "list", "--json"],
    ["capture", "--show", "--json"],
    ["complete", "--json", "--"],
    ["daemon", "status", "--json"],
    ["context", "--show", "--json"],
    ["setup", "--json", "--no-install", "--rc", "__RC__"],
    ["uninstall", "--json", "--rc", "__RC__"],
    ["slash", "/help", "--json"],
    ["explain", "doctor", "--json"],
    ["learn", "--json"],
]


def test_every_verb_works_without_pi_tmux_fzf_spark_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stripped_path = _build_stripped_path(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    rc_file = tmp_path / "rc"

    monkeypatch.setenv("PATH", stripped_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / ".state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".data"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(home / ".runtime"))
    for name in list(os.environ):
        if name.startswith("NVSH_"):
            monkeypatch.delenv(name, raising=False)

    import shutil

    assert shutil.which("pi") is None
    assert shutil.which("tmux") is None
    assert shutil.which("fzf") is None
    assert shutil.which("spark") is None
    assert shutil.which("bash") is not None
    assert shutil.which("python3") is not None

    for verb in _VERBS:
        argv = [rc_file.as_posix() if a == "__RC__" else a for a in verb]
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = main(list(argv))
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1

        stderr_text = err.getvalue()
        assert (
            "Traceback" not in stderr_text
        ), f"{argv}: unhandled traceback on stderr:\n{stderr_text}"
        # 0 = healthy/success, 1 = a reported diagnostic (e.g. doctor finds
        # the configured agent backend missing from PATH — an accurate,
        # structured finding, not a crash), 2 = environment error. All three
        # are "working with diagnostics only"; anything else, or a
        # traceback, is not.
        assert rc in (0, 1, 2), f"{argv}: unexpected exit code {rc}\nstdout: {out.getvalue()}"
