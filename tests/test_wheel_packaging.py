"""Real wheel-build tests, extending the string-level check in
``tests/test_readline_bash.py::test_bash_file_ships_in_the_wheel``.

These actually build the wheel (via ``uv build`` / ``python -m build``) and
inspect its contents, then — when ``uv`` is available — install it into a
throwaway venv and run ``nvsh setup`` from it, proving the clean-checkout
packaging story task t21 promises: the bash files ship, any ``pi_ext/*.ts``
file ships too, and a wheel install can render + insert the rc block with no
``nvsh/shell`` source directory anywhere on disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - fixed argv below, no shell=True
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HAS_UV = shutil.which("uv") is not None


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    if not HAS_UV:
        pytest.skip("uv not available")
    out_dir = tmp_path_factory.mktemp("wheel-out")
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


def test_wheel_contains_shell_bash_files(built_wheel):
    with zipfile.ZipFile(built_wheel) as zf:
        names = set(zf.namelist())
    assert "nvsh/shell/hook.bash" in names
    assert "nvsh/shell/readline.bash" in names


def test_wheel_contains_tier_fetch_pins(built_wheel):
    with zipfile.ZipFile(built_wheel) as zf:
        names = set(zf.namelist())
    assert "nvsh/tiers/pins.json" in names


def test_wheel_contains_tier_bench_corpus(built_wheel):
    with zipfile.ZipFile(built_wheel) as zf:
        names = set(zf.namelist())
    assert "nvsh/tiers/corpus/dev.json" in names
    assert "nvsh/tiers/corpus/held-out.json" in names


def test_wheel_contains_every_pi_ext_ts_file(built_wheel):
    pi_ext_dir = REPO_ROOT / "nvsh" / "agent" / "pi_ext"
    ts_files = sorted(pi_ext_dir.glob("*.ts")) if pi_ext_dir.is_dir() else []
    with zipfile.ZipFile(built_wheel) as zf:
        names = set(zf.namelist())
    for ts_file in ts_files:
        rel = ts_file.relative_to(REPO_ROOT).as_posix()
        assert rel in names, f"{rel} built but missing from the wheel"
    if not ts_files:
        pytest.skip("nvsh/agent/pi_ext/*.ts does not exist yet in this checkout")


@pytest.mark.skipif(not HAS_UV, reason="uv not available")
def test_clean_venv_install_runs_nvsh_setup(built_wheel, tmp_path):
    venv_dir = tmp_path / "venv"
    subprocess.run(  # nosec B603
        ["uv", "venv", str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    python = venv_dir / "bin" / "python"
    subprocess.run(  # nosec B603
        ["uv", "pip", "install", "--python", str(python), str(built_wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )

    home = tmp_path / "home"
    home.mkdir()
    rc = home / "fakerc"
    rc.write_text(
        "# If not running interactively, don't do anything\n"
        "case $- in\n    *i*) ;;\n      *) return;;\nesac\n"
    )
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        XDG_DATA_HOME=str(tmp_path / "xdg-data"),
        XDG_CONFIG_HOME=str(tmp_path / "xdg-config"),
        XDG_RUNTIME_DIR=str(tmp_path / "run"),
        PATH=f"{venv_dir / 'bin'}:{os.environ.get('PATH', '')}",
    )
    nvsh_bin = venv_dir / "bin" / "nvsh"
    result = subprocess.run(  # nosec B603
        [str(nvsh_bin), "setup", "--rc", str(rc), "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "# >>> nvsh setup >>>" in rc.read_text()
    assert (tmp_path / "xdg-data" / "nvsh" / "shell" / "hook.bash").is_file()
