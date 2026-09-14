"""Static checks for scripts/demo-render.sh.

No network calls here — these only inspect the script's own text and its
argv-validation exit code, so the suite stays fast and offline-safe. The
real render (npx fetching svg-term-cli and producing an SVG from a
committed .cast) is a manual/CI-adjacent check, not something this test
suite performs.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "demo-render.sh"


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_script_is_executable() -> None:
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "scripts/demo-render.sh must be executable"


def test_script_pins_svg_term_version() -> None:
    text = SCRIPT.read_text()
    assert "svg-term-cli@" in text
    assert "SVG_TERM_VERSION=" in text


def test_script_documents_agg_fallback() -> None:
    text = SCRIPT.read_text()
    assert "agg" in text
    assert "asciinema-agg" in text or "agg " in text
    assert "--cols" in text
    assert "--rows" in text
    assert "github.com/asciinema/agg" in text


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_script_no_args_exits_nonzero() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "Usage" in result.stderr or "usage" in result.stderr
