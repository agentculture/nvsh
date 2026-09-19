"""Stage the pinned Needle3 engine into a private home, offline (task t9).

``cactus-needle`` resolves its native engine and its base weights by
*downloading* them when they are not already where it expects: the engine
from ``NEEDLE3_LIB_PATH``, else ``<package>/libneedle3.so``, else
``~/.cache/cactus-needle/v3/<engine version>/libneedle.so``; the base
weights from ``needle.agent.fetch.cache_dir(3)/base_weights(3)`` under the
same ``~``. nvsh never wants that download: the bytes it trusts are the
ones :mod:`nvsh.tiers.fetch` already verified against ``pins.json``.

:func:`stage` closes that gap without a socket. It takes the verified wheel
and weights, extracts the one ``needle/libneedle*.so`` member out of the
wheel into nvsh's own cache, and hands back a ``home`` directory the worker
points ``$HOME`` at -- so every path cactus-needle derives from ``~``
lands inside nvsh's cache, next to files whose sha256 is pinned, and the
operator's real ``~/.cache`` is never touched.

Nothing here opens a socket or imports ``needle``: it is ``zipfile`` and
``os`` over files :mod:`nvsh.tiers.fetch` already verified.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import fetch
from .fetch import FetchProblem

#: The wheel member holding the native engine. Anchored at both ends, so a
#: member with a leading ``/`` or a ``..`` segment cannot match.
_LIB_MEMBER = re.compile(r"^needle/libneedle[A-Za-z0-9._-]*\.so$")

#: Refuse to extract a "shared library" bigger than this. The pinned engine
#: measures 1.2 MB; 16 MiB is room for growth and still a bound.
MAX_LIB_BYTES = 16 * 1024 * 1024

_CHUNK_BYTES = 1024 * 1024

#: Layout under the nvsh tiers cache.
_HOME_ROOT = "needle-home"
_STAMP_NAME = "engine.stamp"


@dataclass(frozen=True)
class NeedleHome:
    """Where the worker finds a staged engine, its weights and its ``$HOME``."""

    lib: Path
    weights: Path
    home: Path


def _stamp_of(path: Path) -> dict[str, int]:
    """Identity of the wheel on disk: size, mtime and inode.

    Cheap enough to check on every call, and all three change when the file
    is replaced -- so a re-download (or a hand-edit) re-extracts, while a
    warm cache costs one ``stat``.
    """
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _read_stamp(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _engine_member(archive: zipfile.ZipFile) -> str | FetchProblem:
    """The single engine member of the wheel, or why there is not exactly one."""
    members = [name for name in archive.namelist() if _LIB_MEMBER.match(name)]
    if len(members) != 1:
        return FetchProblem(
            item="engine",
            code="bad_engine",
            message=f"wheel holds {len(members)} needle/libneedle*.so members, expected exactly 1",
        )
    return members[0]


def _extract_member(archive: zipfile.ZipFile, member: str, target: Path) -> FetchProblem | None:
    """Extract *member* to *target* atomically, refusing anything oversized."""
    info = archive.getinfo(member)
    if info.file_size > MAX_LIB_BYTES:
        return FetchProblem(
            item="engine",
            code="too_large",
            message=f"{member} is {info.file_size} bytes, over the {MAX_LIB_BYTES} byte bound",
        )

    _secure_dir(target.parent)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".engine-")
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        written = _copy_bounded(archive, member, tmp_path)
        if isinstance(written, FetchProblem):
            return written
        os.chmod(tmp_path, 0o700)
        os.replace(tmp_path, target)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def _copy_bounded(archive: zipfile.ZipFile, member: str, tmp_path: Path) -> int | FetchProblem:
    """Stream *member* into *tmp_path*, stopping past :data:`MAX_LIB_BYTES`.

    The declared size was checked already; this bounds the *decompressed*
    stream too, so a zip bomb cannot fill the cache directory.
    """
    written = 0
    with archive.open(member) as source, open(tmp_path, "wb") as out:
        while True:
            chunk = source.read(_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_LIB_BYTES:
                return FetchProblem(
                    item="engine",
                    code="too_large",
                    message=f"{member} expands past the {MAX_LIB_BYTES} byte bound",
                )
            out.write(chunk)
    return written


def _stage_engine(wheel: Path, lib_dir: Path) -> Path | FetchProblem:
    """The staged ``libneedle*.so``, extracting it only when the wheel changed."""
    stamp_path = lib_dir / _STAMP_NAME
    stamp = _stamp_of(wheel)

    try:
        with zipfile.ZipFile(wheel) as archive:
            member = _engine_member(archive)
            if isinstance(member, FetchProblem):
                return member
            target = lib_dir / Path(member).name
            if target.is_file() and _read_stamp(stamp_path) == stamp:
                return target
            problem = _extract_member(archive, member, target)
    except (OSError, zipfile.BadZipFile) as exc:
        return FetchProblem(item="engine", code="bad_engine", message=f"unreadable wheel: {exc}")

    if problem is not None:
        return problem
    _write_stamp(stamp_path, stamp)
    return target


def _write_stamp(stamp_path: Path, stamp: dict[str, int]) -> None:
    try:
        with open(stamp_path, "w", encoding="utf-8") as handle:
            json.dump(stamp, handle)
    except OSError:
        pass  # a stamp that cannot be written only costs a re-extraction


def stage(
    cache_dir: Path | None = None,
    *,
    pins: dict | None = None,
    platform_tag: str | None = None,
) -> NeedleHome | FetchProblem:
    """Make the verified engine and weights usable by ``cactus-needle``.

    Returns a :class:`NeedleHome` when both pinned files are present and the
    engine is staged, or the first :class:`~nvsh.tiers.fetch.FetchProblem`
    that stopped it -- never raises, never downloads. ``nvsh tiers
    prefetch`` is what puts the files there in the first place.
    """
    cache_dir = cache_dir if cache_dir is not None else fetch.default_cache_dir()

    weights = fetch.resolve("weights", pins=pins, cache_dir=cache_dir, platform_tag=platform_tag)
    if isinstance(weights, FetchProblem):
        return weights

    wheel = fetch.resolve("engine", pins=pins, cache_dir=cache_dir, platform_tag=platform_tag)
    if isinstance(wheel, FetchProblem):
        return wheel

    root = Path(cache_dir) / _HOME_ROOT
    lib_dir = root / "lib"
    home = root / "home"
    try:
        _secure_dir(lib_dir)
        _secure_dir(home)
    except OSError as exc:
        return FetchProblem(item="engine", code="unwritable", message=f"{root}: {exc}")

    lib = _stage_engine(Path(wheel), lib_dir)
    if isinstance(lib, FetchProblem):
        return lib

    return NeedleHome(lib=lib, weights=Path(weights), home=home)
