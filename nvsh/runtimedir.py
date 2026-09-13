"""Where nvsh's per-session scratch directory lives, and proof that it is ours.

nvsh keeps its session log, the hook's refresh notice, the daemon socket and
the session-approval store in ``$XDG_RUNTIME_DIR/nvsh``. That variable is
routinely missing on a Jetson reached with a bare ``ssh host``, so there has
to be a fallback — and the only fallback available on such a box is the system
temp directory, which is world-writable (SonarCloud ``python:S5443``).

Two rules make that safe, and this module is the one place both live so
:mod:`nvsh.capture`, ``nvsh setup``, :mod:`nvsh.daemon` and
:mod:`nvsh.approvals` cannot drift apart (the fallback path must agree with
``nvsh/shell/hook.bash``'s too, which is what ``tests/test_hook_bash.py``
pins):

1. The fallback is **per-uid** (``<tmp>/nvsh-<uid>``), never a shared
   ``<tmp>/nvsh`` another user could create first.
2. Nothing is ever written into it until :func:`ensure_private` has proved
   the directory is a real directory (not a symlink), owned by the calling
   uid, and mode ``0700``. A directory that fails any of those is refused
   with :class:`RuntimeDirError` rather than used.

Stdlib only.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Mapping


class RuntimeDirError(RuntimeError):
    """The runtime directory exists but is not safe to write into."""


def fallback_dir(uid: int | None = None) -> Path:
    """``<tmp>/nvsh-<uid>`` — the per-uid fallback when ``XDG_RUNTIME_DIR`` is unset."""
    resolved = os.getuid() if uid is None else uid
    return Path(tempfile.gettempdir()) / f"nvsh-{resolved}"


def runtime_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$XDG_RUNTIME_DIR/nvsh``, else :func:`fallback_dir`. Creates nothing."""
    resolved = os.environ if env is None else env
    xdg = resolved.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "nvsh"
    return fallback_dir()


def ensure_private(path: Path) -> Path:
    """Create *path* mode ``0700`` if needed, then prove it is the caller's own.

    Raises :class:`RuntimeDirError` — never silently proceeds — when the path
    is a symlink, is not a directory, is owned by another uid, or is still
    group/other-accessible after an attempt to tighten it. Returns *path* so
    it can be used inline.
    """
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise RuntimeDirError(f"cannot create runtime directory {path}: {exc}") from exc

    info = _lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeDirError(
            f"refusing to use runtime directory {path}: it is a symlink, not a directory"
        )
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeDirError(f"refusing to use runtime directory {path}: it is not a directory")
    if info.st_uid != os.getuid():
        raise RuntimeDirError(
            f"refusing to use runtime directory {path}: it is owned by uid {info.st_uid}, "
            f"not by you (uid {os.getuid()})"
        )

    if stat.S_IMODE(info.st_mode) != 0o700:
        try:
            os.chmod(path, 0o700)
        except OSError as exc:
            raise RuntimeDirError(f"cannot set {path} to mode 0700: {exc}") from exc
        info = _lstat(path)
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise RuntimeDirError(
                f"refusing to use runtime directory {path}: its permissions are "
                f"{stat.S_IMODE(info.st_mode):04o}, not 0700"
            )
    return path


def _lstat(path: Path) -> os.stat_result:
    try:
        return os.lstat(path)
    except OSError as exc:  # pragma: no cover - makedirs just succeeded
        raise RuntimeDirError(f"cannot inspect runtime directory {path}: {exc}") from exc
