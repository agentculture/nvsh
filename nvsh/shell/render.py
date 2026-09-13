"""Render ``nvsh/shell/*.bash`` package resources into the user's data dir.

``nvsh setup`` (task t21) calls :func:`render_shell_files` to copy
``hook.bash`` and ``readline.bash`` out of the installed package (via
``importlib.resources``, so this works from a wheel install with no
``nvsh/shell`` directory on disk) into
``$XDG_DATA_HOME/nvsh/shell/`` (default ``~/.local/share/nvsh/shell``),
each prefixed with a two-line stamp naming the nvsh version that rendered
it. The marked rc block sources these rendered copies, never the package's
own files, so ``nvsh setup`` is what refreshes them after an upgrade.
"""

from __future__ import annotations

import os
import shutil
import sys
from importlib import resources
from pathlib import Path

from nvsh import __version__

#: The files copied from ``nvsh.shell`` into the rendered data dir.
SHELL_FILES = ("hook.bash", "readline.bash")


def data_dir() -> Path:
    """``$XDG_DATA_HOME/nvsh`` (default ``~/.local/share/nvsh``)."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "nvsh"


def render_shell_files(base: Path | None = None) -> Path:
    """Copy :data:`SHELL_FILES` into ``<base or data_dir()>/shell/``, version-stamped.

    Returns the shell directory path. Idempotent: re-rendering the same
    version produces byte-identical files.
    """
    target = base if base is not None else data_dir()
    shell_dir = target / "shell"
    shell_dir.mkdir(parents=True, exist_ok=True)

    src_root = resources.files("nvsh.shell")
    header = f"# nvsh hook version {__version__}\nexport NVSH_HOOK_VERSION={__version__}\n"
    for name in SHELL_FILES:
        content = (src_root / name).read_text(encoding="utf-8")
        (shell_dir / name).write_text(header + content, encoding="utf-8")

    return shell_dir


def resolve_nvsh_bin() -> str:
    """Absolute path of the running ``nvsh`` entrypoint.

    Tries ``shutil.which("nvsh")`` first (works for a ``uv tool install`` or
    any PATH install); falls back to ``sys.argv[0]`` resolved to an absolute
    path when it looks like a real executable path; falls back to the
    literal string ``"nvsh"`` (so the rendered block still works once nvsh
    ends up on PATH some other way) when neither resolves to anything.
    """
    found = shutil.which("nvsh")
    if found:
        return str(Path(found).resolve())

    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and Path(argv0).name not in ("", "-c"):
        resolved = shutil.which(argv0) or argv0
        candidate = Path(resolved)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    return "nvsh"
