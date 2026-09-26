"""Scaffold-level tests for the evals/ tree (task t1, issue #64).

These cover the four acceptance criteria for the scaffolding task:

1. the built wheel ships no ``evals/`` file and gains no ``Requires-Dist``
   for ``deepeval``,
2. ``uv run pytest -n auto`` from the repo root collects zero evals tests
   and never loads the deepeval pytest plugin,
3. importing ``evals.tool_jev`` sets ``DEEPEVAL_TELEMETRY_OPT_OUT=1`` before
   ``deepeval`` is ever imported, and refuses to run when
   ``CONFIDENT_API_KEY`` is set,
4. nothing under ``nvsh/`` mentions ``deepeval`` or ``evals``.

Nothing here is silently skipped: if a tool this suite needs (``uv``) is
missing, the affected tests fail with a clear message instead of skipping.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - fixed argv below, no shell=True
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HAS_UV = __import__("shutil").which("uv") is not None


def _require_uv() -> None:
    if not HAS_UV:
        pytest.fail("`uv` is not on PATH — cannot verify the built wheel without it.")


# ---------------------------------------------------------------------------
# Criterion 1: the built wheel ships no evals/ file, and gains no
# Requires-Dist for deepeval.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    _require_uv()
    out_dir = tmp_path_factory.mktemp("evals-wheel-out")
    subprocess.run(  # nosec B603
        ["uv", "build", "--wheel", "-o", str(out_dir)],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    wheels = list(out_dir.glob("*.whl"))
    assert wheels, "uv build produced no wheel"
    return wheels[0]


def test_wheel_contains_no_evals_files(built_wheel):
    with zipfile.ZipFile(built_wheel) as zf:
        names = zf.namelist()
    evals_names = [n for n in names if n.startswith("evals/") or "/evals/" in n]
    assert evals_names == [], f"wheel packages evals/ files it must not: {evals_names}"


def test_wheel_metadata_has_no_deepeval_requires_dist(built_wheel):
    with zipfile.ZipFile(built_wheel) as zf:
        metadata_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1, metadata_names
        metadata = zf.read(metadata_names[0]).decode("utf-8")
    requires_dist_lines = [
        line for line in metadata.splitlines() if line.startswith("Requires-Dist:")
    ]
    deepeval_lines = [line for line in requires_dist_lines if "deepeval" in line.lower()]
    assert deepeval_lines == [], f"wheel METADATA gained a deepeval requirement: {deepeval_lines}"


# ---------------------------------------------------------------------------
# Criterion 2: `uv run pytest -n auto` from the repo root collects zero
# evals tests, and the deepeval pytest plugin never loads.
# ---------------------------------------------------------------------------


def test_root_suite_collects_zero_evals_tests():
    _require_uv()
    result = subprocess.run(  # nosec B603
        ["uv", "run", "pytest", "-n", "auto", "--collect-only", "-q"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"root collection failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    collected_evals_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("evals/") or "::evals" in line
    ]
    assert (
        collected_evals_lines == []
    ), "root pytest run collected evals/ tests it must not:\n" + "\n".join(collected_evals_lines)


def test_root_suite_never_loads_the_deepeval_plugin():
    _require_uv()
    result = subprocess.run(  # nosec B603
        ["uv", "run", "pytest", "-q", "--collect-only", "--trace-config"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"root collection with --trace-config failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    deepeval_plugin_lines = [
        line
        for line in result.stdout.splitlines()
        if "PLUGIN registered" in line and "deepeval" in line.lower()
    ]
    assert (
        deepeval_plugin_lines == []
    ), "the deepeval pytest11 plugin loaded into the root suite:\n" + "\n".join(
        deepeval_plugin_lines
    )


# ---------------------------------------------------------------------------
# Criterion 3: importing evals.tool_jev sets DEEPEVAL_TELEMETRY_OPT_OUT=1
# before deepeval import, and refuses to run if CONFIDENT_API_KEY is set.
# ---------------------------------------------------------------------------


def _run_probe(code: str, env_overrides: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for key in (
        "DEEPEVAL_TELEMETRY_OPT_OUT",
        "DEEPEVAL_DISABLE_DOTENV",
        "CONFIDENT_API_KEY",
    ):
        env.pop(key, None)
    env.update(env_overrides)
    # Run from a scratch dir, never the repo root: importing deepeval creates
    # a relative .deepeval/ directory in the cwd. PYTHONPATH keeps
    # `import evals.tool_jev` resolvable from there.
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    scratch = tempfile.mkdtemp(prefix="evals-probe-")
    return subprocess.run(  # nosec B603
        [sys.executable, "-c", code],
        cwd=scratch,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_importing_tool_jev_sets_telemetry_opt_out_before_any_deepeval_import():
    code = (
        "import sys, os\n"
        "assert 'deepeval' not in sys.modules\n"
        "import evals.tool_jev\n"
        "assert 'deepeval' not in sys.modules, ("
        "'evals.tool_jev must not import deepeval itself')\n"
        "assert os.environ.get('DEEPEVAL_TELEMETRY_OPT_OUT') == '1'\n"
        "assert os.environ.get('DEEPEVAL_DISABLE_DOTENV') == '1'\n"
        "import deepeval\n"
        "print('OK')\n"
    )
    result = _run_probe(code, {})
    assert result.returncode == 0, (
        f"probe failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.parametrize("name", ["DEEPEVAL_TELEMETRY_OPT_OUT", "DEEPEVAL_DISABLE_DOTENV"])
def test_importing_tool_jev_refuses_a_conflicting_protective_value(name):
    # An explicit "0" would re-enable telemetry or .env loading (which can
    # smuggle CONFIDENT_API_KEY in after the guard checked it): refuse it.
    result = _run_probe("import evals.tool_jev\n", {name: "0"})
    assert result.returncode != 0, f"import must fail when {name}=0"
    assert name in result.stderr


def test_evals_pytest_run_never_loads_the_deepeval_plugin():
    # The real startup path: pytest itself would import deepeval's pytest11
    # plugin before collection reaches evals.tool_jev's guard.
    env = dict(os.environ)
    env.pop("CONFIDENT_API_KEY", None)
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "evals/pytest.ini",
            "--rootdir=.",
            "--trace-config",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "evals/tool_jev/tests/test_scaffold.py",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deepeval.plugins" not in result.stdout + result.stderr


def test_importing_tool_jev_raises_if_confident_api_key_is_set():
    code = "import evals.tool_jev\n"
    result = _run_probe(code, {"CONFIDENT_API_KEY": "sk-should-never-be-set"})
    assert result.returncode != 0, "import must fail when CONFIDENT_API_KEY is set"
    assert "CONFIDENT_API_KEY" in result.stderr


def test_importing_tool_jev_succeeds_when_confident_api_key_is_absent():
    code = "import evals.tool_jev\nprint('OK')\n"
    result = _run_probe(code, {})
    assert result.returncode == 0, (
        f"probe failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Criterion 4: nvsh/ never mentions deepeval or evals.
# ---------------------------------------------------------------------------


def test_nvsh_package_never_mentions_deepeval_or_evals():
    result = subprocess.run(  # nosec B603
        ["grep", "-rln", "-E", "deepeval|evals", str(REPO_ROOT / "nvsh")],
        capture_output=True,
        text=True,
    )
    # grep exit code 1 = no matches (what we want); 0 = matches found (fail).
    assert result.returncode == 1, (
        f"nvsh/ mentions deepeval/evals in:\n{result.stdout}\n"
        f"(grep rc={result.returncode}, stderr={result.stderr})"
    )


# ---------------------------------------------------------------------------
# Scaffold layout sanity (not one of the four numbered criteria, but the
# structural contract later tasks build on).
# ---------------------------------------------------------------------------


def test_evals_tool_jev_tests_dir_has_no_init_file():
    tests_dir = REPO_ROOT / "evals" / "tool_jev" / "tests"
    assert not (tests_dir / "__init__.py").exists(), (
        "evals/tool_jev/tests must stay a non-package dir (importmode=importlib): "
        "later tasks only add their own module + test file here."
    )


def test_evals_pytest_ini_uses_importlib_import_mode():
    ini_text = (REPO_ROOT / "evals" / "pytest.ini").read_text()
    assert "--import-mode=importlib" in ini_text
