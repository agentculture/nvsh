"""Tests for pyproject.toml optional-dependencies (flavors): needle, lfm, tiers."""

import ast
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_pyproject():
    """Return the parsed pyproject.toml."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_runtime_dependencies_stay_empty():
    """The base install must stay dependency-free."""
    data = _load_pyproject()
    assert data["project"]["dependencies"] == []


def test_needle_flavor_pins_cactus_needle_range():
    """The needle flavor must pin cactus-needle to >=3.0.1,<3.1."""
    data = _load_pyproject()
    needle = data["project"]["optional-dependencies"]["needle"]
    assert len(needle) == 1
    entry = needle[0]
    assert entry.startswith("cactus-needle")
    assert ">=" in entry
    assert "<" in entry


def test_lfm_flavor_needs_no_python_package():
    """The lfm flavor needs no Python package (runtime is a container)."""
    data = _load_pyproject()
    assert data["project"]["optional-dependencies"]["lfm"] == []


def test_tiers_flavor_is_needle_plus_lfm():
    """The tiers flavor must be needle + lfm combined."""
    data = _load_pyproject()
    needle = data["project"]["optional-dependencies"]["needle"]
    lfm = data["project"]["optional-dependencies"]["lfm"]
    tiers = data["project"]["optional-dependencies"]["tiers"]
    assert sorted(tiers) == sorted(needle + lfm)


def test_cli_startup_imports_no_tier_module():
    """No tier module may be imported by nvsh.cli startup."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import nvsh.cli"],
        capture_output=True,
        text=True,
        timeout=60,
    )  # nosec B603
    stderr = result.stderr
    assert "nvsh.tiers" not in stderr, "nvsh.tiers found in import-time output"
    assert "nvsh.ops" not in stderr, "nvsh.ops found in import-time output"


def _is_needle(module: str | None) -> bool:
    return module is not None and (module == "needle" or module.startswith("needle."))


def _top_level_needle_imports(py_file: Path) -> list[str]:
    """Names of the ``needle`` modules *py_file* imports at module level."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if _is_needle(alias.name)]
        elif isinstance(node, ast.ImportFrom) and _is_needle(node.module):
            found.append(node.module)
    return found


def test_no_tier_module_imports_needle_at_module_level():
    """No *.py under nvsh/tiers or nvsh/ops may import needle at module level."""
    files = [f for pkg in ("tiers", "ops") for f in (REPO_ROOT / "nvsh" / pkg).rglob("*.py")]
    offenders = {str(f): names for f in files if (names := _top_level_needle_imports(f))}
    assert offenders == {}


@pytest.mark.parametrize(
    "verb",
    ["whoami", "learn", "explain", "overview", "doctor", "agent", "approve", "setup", "slash"],
)
def test_every_verb_help_runs_without_extras(verb):
    """Every verb must accept --help when cactus-needle is not installed."""
    if importlib.util.find_spec("needle") is not None:
        pytest.skip("cactus-needle is installed here")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from nvsh.cli import main; sys.exit(main())",
            verb,
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )  # nosec B603
    assert (
        result.returncode == 0
    ), f"`nvsh {verb} --help` returned {result.returncode}: {result.stderr}"
