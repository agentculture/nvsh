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

import ast
import contextlib
import io
import os
import re
import subprocess
from pathlib import Path

import pytest

from nvsh.agent import acp as acp_module
from nvsh.agent.acp import build as acp_build
from nvsh.cli import main
from nvsh.cli._commands.doctor import _PROMPT_FILE
from tests import _fake_adapters

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
    rc_file = home / "rc"

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


# ---------------------------------------------------------------------------
# (e) no adapter ever asks a harness to skip its own permission gate
# ---------------------------------------------------------------------------

#: Every flag or mode token that makes a harness run tools without asking.
#: Spec c7 / "propose, don't run": nvsh proposes and the operator approves,
#: so none of these may reach a child -- not on argv, not in a wire frame,
#: not behind a config key.
_AUTO_APPROVE_TOKENS = (
    "dangerously-skip-permissions",
    "yolo",
    "danger-full-access",
    "--full-auto",
    "--trust-all-tools",
)


@pytest.mark.parametrize("case", _fake_adapters.CASES, ids=lambda case: case.name)
def test_no_adapter_tells_its_child_to_auto_approve(case) -> None:
    """Everything an adapter says to its child, scanned for a bypass token.

    ``effort_words`` returns the child's argv and -- for the ACP adapters,
    whose settings travel in-session -- every JSON-RPC frame the client
    sent, so this covers both ways a bypass could be requested.
    """
    words = case.effort_words(case.effort_probe)
    assert words, f"{case.name}: nothing was captured, so nothing was checked"
    for token in _AUTO_APPROVE_TOKENS:
        offenders = [word for word in words if token in word]
        assert not offenders, f"{case.name} sent {token!r} to its child: {offenders}"


@pytest.mark.parametrize("token", ["--trust-all-tools", "--yolo", "--dangerously-skip-permissions"])
def test_acp_refuses_a_bypass_flag_at_construction(token: str) -> None:
    """Refused outright, not merely left unset (nvsh/agent/acp.py)."""
    with pytest.raises(ValueError):
        acp_build("qwen", {"extra_args": [token]})


@pytest.mark.parametrize("mode", sorted(acp_module.FORBIDDEN_MODES))
def test_acp_refuses_a_bypass_mode_at_construction(mode: str) -> None:
    with pytest.raises(ValueError):
        acp_module.AcpAgent(["qwen", "--acp"], "qwen", mode=mode)


# ---------------------------------------------------------------------------
# (f) nvsh never writes a harness's own settings or trust file
# ---------------------------------------------------------------------------

#: Files that belong to a harness's permission model, not to nvsh. Writing
#: any of them would move an approval decision into a store nvsh's own
#: approve loop can neither see nor revoke (spec c7). ``models.json`` is
#: pi's provider/credential table: ``nvsh/doctor_checks.py`` reads it to
#: report what pi is pointed at, and must never write it.
_HARNESS_SETTINGS_RE = re.compile(
    r"settings(\.local)?\.json"
    r"|trusted[-_]folders"
    r"|\.claude/settings"
    r"|\.qwen/settings"
    r"|\.gemini/settings"
    r"|\.codex/config\.toml"
    r"|models\.json"
    r"|trust-all-tools",
    re.IGNORECASE,
)

#: Method calls that create or replace a *file* on disk. Deliberately not
#: ``.write``/``.replace``: the first is what every pipe and stream write
#: uses (``nvsh/agent/acp.py`` writes JSON-RPC frames to a child's stdin)
#: and the second is ``str.replace`` far more often than ``os.replace``.
#: File-replacing moves are matched by qualified name instead.
_WRITE_METHODS = {"write_text", "write_bytes", "writelines", "touch", "unlink"}

#: ``<module>.<attr>`` calls that move or copy a file over another.
_WRITE_QUALIFIED = {
    ("os", "replace"),
    ("os", "rename"),
    ("shutil", "copy"),
    ("shutil", "copy2"),
    ("shutil", "copyfile"),
    ("shutil", "move"),
}


def _nvsh_sources() -> list[Path]:
    return sorted((REPO_ROOT / "nvsh").rglob("*.py"))


def _docstring_ids(tree: ast.AST) -> set[int]:
    """The ids of every docstring constant in ``tree``.

    Computed over the whole module, not over the node being scanned: a
    module docstring is a statement in the module body, so a scan that
    starts *at* that statement would not recognise it as one.
    """
    ids = set()
    for sub in ast.walk(tree):
        if not isinstance(sub, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(sub, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _literals(node: ast.AST, docstrings: set[int]) -> list[str]:
    """Every string constant under ``node`` that is not a docstring."""
    return [
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant)
        and isinstance(sub.value, str)
        and id(sub) not in docstrings
    ]


def _writes(node: ast.AST) -> list[str]:
    """Names of the write-capable calls under ``node``."""
    found = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute) and func.attr in _WRITE_METHODS:
            found.append(func.attr)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and (func.value.id, func.attr) in _WRITE_QUALIFIED
        ):
            found.append(f"{func.value.id}.{func.attr}")
        elif isinstance(func, ast.Name) and func.id == "open":
            modes = [
                arg.value
                for arg in list(sub.args[1:]) + [kw.value for kw in sub.keywords]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if any(set("wax+") & set(mode) for mode in modes):
                found.append("open")
    return found


def test_no_nvsh_code_path_opens_a_harness_settings_file_for_writing() -> None:
    """AST-level, not grep-level: a *write* and a harness settings name
    never appear in the same function, and a module that names one at module
    scope holds no write at all."""
    offenders: list[str] = []
    for path in _nvsh_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(REPO_ROOT).as_posix()
        docstrings = _docstring_ids(tree)

        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        function_ids = {id(node) for node in functions}
        module_level = [
            literal
            for node in tree.body
            if id(node) not in function_ids and not isinstance(node, ast.ClassDef)
            for literal in _literals(node, docstrings)
            if _HARNESS_SETTINGS_RE.search(literal)
        ]

        for node in functions:
            named = [lit for lit in _literals(node, docstrings) if _HARNESS_SETTINGS_RE.search(lit)]
            writes = _writes(node)
            if writes and (named or module_level):
                offenders.append(
                    f"{rel}:{node.lineno} {node.name}() writes ({sorted(set(writes))}) "
                    f"while naming {sorted(set(named + module_level))}"
                )
    assert not offenders, "harness settings/trust files must never be written:\n" + "\n".join(
        offenders
    )


def test_harness_settings_filenames_appear_only_in_prose() -> None:
    """The names themselves are absent from executable nvsh source.

    ``nvsh/agent/agy.py`` *explains* in its docstring why it never touches
    ``settings.json``; the code below that docstring must not contain the
    string at all. Two deliberate exclusions from the pattern:
    ``models.json``, because ``nvsh/doctor_checks.py`` legitimately reads
    pi's copy (the write ban for it is the AST test above), and the bypass
    *flags*, because ``nvsh/agent/acp.py`` must name them in
    ``FORBIDDEN_ARGS`` in order to refuse them -- that refusal is asserted
    directly in ``test_acp_refuses_a_bypass_flag_at_construction``.
    """
    banned = re.compile(r"settings(\.local)?\.json|trusted[-_]folders", re.I)
    offenders = []
    for path in _nvsh_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for literal in _literals(tree, _docstring_ids(tree)):
            if banned.search(literal):
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {literal!r}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# (g) the CLI version each fake was recorded from, per adapter
# ---------------------------------------------------------------------------

_RECORDED_FROM_RE = re.compile(r"^#\s*recorded-from:\s*(\S+)\s+(\S+)", re.MULTILINE)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for chunk in text.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _in_range(version: str, low: str, high: str) -> bool:
    return _version_tuple(low) <= _version_tuple(version) < _version_tuple(high)


@pytest.mark.parametrize(
    "case",
    [case for case in _fake_adapters.CASES if case.stamp_file],
    ids=lambda case: case.name,
)
def test_each_fake_records_the_cli_version_it_was_taken_from(case) -> None:
    """Provenance, and a supported range, per adapter.

    ``tests/test_fixture_hygiene.py`` already enforces that every recorded
    *transcript* carries a ``# recorded-from: <cli> <version>`` header; this
    is the other half -- that the version in that header is the one the
    adapter was written against. No adapter exposes a ``supported_versions``
    constant today (see the test below), so the range lives in
    ``tests/_fake_adapters.py``'s case table and is checked against the
    header rather than against the adapter.
    """
    path = REPO_ROOT / case.stamp_file
    assert path.is_file(), f"{case.name}: {case.stamp_file} is missing"
    match = _RECORDED_FROM_RE.search(path.read_text(encoding="utf-8"))
    assert match, f"{case.name}: no '# recorded-from:' header in {case.stamp_file}"

    cli, version = match.group(1), match.group(2)
    assert cli == case.cli, f"{case.name}: recorded from {cli!r}, expected {case.cli!r}"
    low, high = case.version_range
    assert _in_range(version, low, high), (
        f"{case.name}: {case.cli} {version} is outside the supported range "
        f"[{low}, {high}) -- re-record the fixture or widen the range deliberately"
    )


def test_the_adapters_without_a_recorded_version_are_the_pinned_ones() -> None:
    """pi and kiro have no recorded-from stamp anywhere in the tree.

    ``tests/fakes/pi``/``pi_scripted`` are hand-written rather than
    recorded, and the kiro 2.21.4 session ``nvsh/agent/acp.py`` cites in
    prose was never captured as a fixture (``tests/fakes/acp`` replays the
    qwen recording). Pinned so that landing a stamp -- or landing another
    unstamped fake -- is a deliberate edit.
    """
    unstamped = {case.name for case in _fake_adapters.CASES if not case.stamp_file}
    assert unstamped == set(_fake_adapters.UNSTAMPED_ADAPTERS)


def test_no_adapter_declares_its_own_supported_cli_versions() -> None:
    """The supported range lives in the tests, not in the adapters.

    Recorded here as a known gap rather than left implicit: if an adapter
    grows a ``supported_versions``-style constant, this test fails and the
    range above should move there, where the runtime can also use it.
    """
    declared = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _nvsh_sources()
        if re.search(r"\bsupported_versions\b|\bSUPPORTED_VERSIONS\b", path.read_text("utf-8"))
    ]
    assert declared == [], f"an adapter now declares its supported versions: {declared}"
