"""Pinned fetch and prefetch for the local-tier engine, weights and image.

Task t8. Style follows ``nvsh/installers.py``: planning is pure (no
``subprocess``, no network, no ``urlopen``) and exactly one function ever
executes anything (:func:`prefetch`, plus the docker-pull step it drives).
Everything the executing function touches -- the URL opener, the argv
runner, the confirmation callback -- is injectable so tests never hit the
real network or a real ``docker``.

nvsh deliberately does NOT import ``huggingface_hub`` or ``cactus-needle``:
the upstream ``cactus-needle`` package downloads its weights via
``huggingface_hub`` and, as a side effect of every fetch, also calls
``hf_hub_download(..., filename="config.json", force_download=True)``
purely as a download counter (see the module docstring's scope note s15),
and ships telemetry on by default. Pulling files ourselves with stdlib
``urllib`` avoids both: nvsh only ever *requests*
``https://huggingface.co/<repo>/resolve/<revision>/<filename>`` URLs built
from :data:`pins.json <PINS_PATH>`, and only when :func:`prefetch` is asked
to fetch something missing. huggingface.co answers those with a redirect to
its storage CDN, which is followed (https only); the bytes are trusted
because their size and sha256 match the pin, not because of the host that
served them.

Cache layout: ``$XDG_CACHE_HOME/nvsh/tiers/`` (falling back to
``$HOME/.cache/nvsh/tiers/``), directory mode ``0700``. Downloads land in a
temp file in that same directory, get sha256- and size-verified, then
``os.replace()`` into their final name -- so a reader never observes a
partial file, and a failed download never leaves one lying around under the
real name.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

PINS_PATH = Path(__file__).resolve().parent / "pins.json"

#: Only host nvsh will ever build a download URL for. Checked before every
#: ``urlopen`` call (see :func:`_download`), so a malformed pin can never
#: send a request anywhere else.
_ALLOWED_HOST = "huggingface.co"
_ALLOWED_SCHEME = "https"

#: Chunk size used while streaming a download to disk and while hashing a
#: cached file for verification.
_CHUNK_BYTES = 1024 * 1024

OpenerFn = Callable[[Request], object]
RunnerFn = Callable[[list[str]], int]
ConfirmFn = Callable[["PrefetchItem"], bool]


def default_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve ``$XDG_CACHE_HOME/nvsh/tiers`` through an injectable env mapping.

    Falls back to ``$HOME/.cache`` when ``XDG_CACHE_HOME`` is unset, per the
    XDG base directory spec -- the same pattern
    :func:`nvsh.agent.audit.default_audit_path` uses for state.
    """
    resolved_env = os.environ if env is None else env
    xdg_cache_home = resolved_env.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        base = Path(xdg_cache_home)
    else:
        home = resolved_env.get("HOME") or os.path.expanduser("~")
        base = Path(home) / ".cache"
    return base / "nvsh" / "tiers"


def _ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o700)


def load_pins(path: str | Path | None = None) -> dict:
    """Load the pin table (``pins.json`` by default). Pure: no network."""
    resolved = Path(path) if path is not None else PINS_PATH
    with open(resolved, encoding="utf-8") as handle:
        return json.load(handle)


def _default_platform_tag() -> str:
    return _platform.machine()


@dataclass(frozen=True)
class PrefetchItem:
    """One thing :func:`plan_prefetch` says should exist locally.

    ``present`` reflects only a local file-existence/size check at plan
    time (no hashing -- see :func:`plan_prefetch`'s docstring for why).
    ``source`` is a human-readable origin: the HF ``resolve`` URL for
    ``engine``/``weights``, or ``ref@digest`` for ``image``.
    """

    kind: str  # "engine" | "weights" | "image"
    name: str
    source: str
    sha256: str
    size_bytes: int
    present: bool


@dataclass(frozen=True)
class FetchProblem:
    """Something :func:`prefetch`, :func:`resolve` or :func:`verify_all` found wrong."""

    item: str
    code: str  # "hash_mismatch" | "size_mismatch" | "download_failed" | "no_pin_for_platform" | "missing"  # noqa: E501
    message: str


def _hf_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _cache_path(cache_dir: Path, filename: str) -> Path:
    # Only the basename matters for the on-disk cache -- pinned filenames
    # may contain a subdirectory (e.g. "python/cactus_needle-...whl").
    return cache_dir / Path(filename).name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _engine_pin(pins: dict, platform_tag: str) -> dict | None:
    return pins.get("needle3", {}).get("engines", {}).get(platform_tag)


def _weights_pin(pins: dict) -> dict | None:
    return pins.get("needle3", {}).get("weights")


def _needle3_repo_revision(pins: dict) -> tuple[str, str]:
    needle3 = pins.get("needle3", {})
    return needle3.get("repo", ""), needle3.get("revision", "")


def plan_prefetch(
    pins: dict | None = None,
    *,
    platform_tag: str | None = None,
    cache_dir: Path | None = None,
    image_present: Callable[[dict], bool] | None = None,
) -> list[PrefetchItem]:
    """Compute what prefetch would fetch, without fetching anything.

    Pure: only ``Path.exists``/``Path.stat`` local filesystem checks and
    dict lookups -- never opens a socket, never imports ``urllib``'s
    connection machinery. ``present`` is a cheap existence+size check, not a
    hash verification (that is :func:`resolve`'s job, cached per
    ``(path, mtime, size)``) -- planning stays fast even with a populated
    cache.

    A platform with no pinned engine (x86_64 today) still gets a
    ``PrefetchItem`` with an empty ``sha256`` and ``source`` so the caller
    can report "no pinned engine for this platform" (:data:`FetchProblem`
    code ``no_pin_for_platform``) rather than silently omitting it.
    """
    pins = pins if pins is not None else load_pins()
    platform_tag = platform_tag if platform_tag is not None else _default_platform_tag()
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    image_present = image_present if image_present is not None else (lambda _img: False)

    items: list[PrefetchItem] = []

    repo, revision = _needle3_repo_revision(pins)

    weights = _weights_pin(pins)
    if weights:
        filename = weights["filename"]
        local = _cache_path(cache_dir, filename)
        present = local.is_file() and local.stat().st_size == weights["size_bytes"]
        items.append(
            PrefetchItem(
                kind="weights",
                name=filename,
                source=_hf_url(repo, revision, filename),
                sha256=weights["sha256"],
                size_bytes=weights["size_bytes"],
                present=present,
            )
        )

    engine = _engine_pin(pins, platform_tag)
    if engine is not None:
        filename = engine["filename"]
        local = _cache_path(cache_dir, filename)
        present = local.is_file() and local.stat().st_size == engine["size_bytes"]
        items.append(
            PrefetchItem(
                kind="engine",
                name=filename,
                source=_hf_url(repo, revision, filename),
                sha256=engine["sha256"],
                size_bytes=engine["size_bytes"],
                present=present,
            )
        )
    else:
        items.append(
            PrefetchItem(
                kind="engine",
                name=f"needle3-engine-{platform_tag}",
                source="",
                sha256="",
                size_bytes=0,
                present=False,
            )
        )

    for image in pins.get("images", []):
        items.append(
            PrefetchItem(
                kind="image",
                name=image["name"],
                source=f"{image['ref']}@{image['digest']}",
                sha256=image["digest"].split(":", 1)[-1],
                size_bytes=image.get("size_bytes", 0),
                present=bool(image_present(image)),
            )
        )

    return items


def _validate_hf_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != _ALLOWED_SCHEME or parts.netloc != _ALLOWED_HOST:
        raise ValueError(f"refusing to fetch from untrusted URL: {url!r}")


def _download(item: PrefetchItem, dest_dir: Path, opener: OpenerFn) -> FetchProblem | None:
    """Download ``item.source`` into ``dest_dir`` and verify it. The only
    place ``opener`` (an injectable urllib-style opener) is ever called.
    """
    _validate_hf_url(item.source)
    _ensure_cache_dir(dest_dir)
    final_path = _cache_path(dest_dir, item.name)

    fd, tmp_name = tempfile.mkstemp(dir=str(dest_dir), prefix=".fetch-")
    tmp_path = Path(tmp_name)
    os.close(fd)  # reopened below via a plain path-based write

    try:
        request = Request(item.source, headers={"User-Agent": "nvsh-tiers-fetch"})
        try:
            response = opener(request)  # nosec B310 - scheme/host validated above
        except Exception as exc:  # noqa: BLE001
            # Any transport failure is reported, not raised.
            return FetchProblem(
                item=item.name, code="download_failed", message=f"download failed: {exc}"
            )

        try:
            digest = hashlib.sha256()
            size = 0
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > item.size_bytes:
                        # Stop at the pinned size: a server that keeps
                        # streaming must not be able to fill the disk.
                        break
                    out.write(chunk)
                    digest.update(chunk)
        except Exception as exc:  # noqa: BLE001
            # http.client.IncompleteRead and friends are not OSError; whatever
            # breaks the body read is reported, never raised.
            return FetchProblem(
                item=item.name, code="download_failed", message=f"download failed: {exc}"
            )
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

        if size != item.size_bytes:
            return FetchProblem(
                item=item.name,
                code="size_mismatch",
                message=f"expected {item.size_bytes} bytes, got {size}",
            )
        if digest.hexdigest() != item.sha256:
            return FetchProblem(
                item=item.name,
                code="hash_mismatch",
                message=f"expected sha256 {item.sha256}, got {digest.hexdigest()}",
            )

        os.replace(tmp_path, final_path)
        return None
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def prefetch(
    items: list[PrefetchItem],
    *,
    cache_dir: Path | None = None,
    opener: OpenerFn | None = None,
    runner: RunnerFn | None = None,
    confirm: ConfirmFn | None = None,
) -> list[FetchProblem]:
    """Fetch every item in ``items`` that is not already ``present``.

    The one executing function in this module. ``opener`` downloads engine
    and weights files (stdlib ``urllib``-shaped: called with a
    :class:`urllib.request.Request`, returns an object with ``.read()`` and
    ``.close()`` -- ``urllib.request.urlopen`` fits directly).  ``runner``
    executes ``["docker", "pull", f"{ref}@{digest}"]`` for an ``image`` item
    and returns its exit code. ``confirm`` (default: always ``True``) is
    asked once per item before it is fetched -- a caller that wants an
    interactive prompt or a ``--yes`` flag decides what that means, exactly
    like :func:`nvsh.installers.run_install`.

    An item with no pin for this platform (empty ``source``) is reported as
    ``no_pin_for_platform`` and never fetched, confirmed or not.
    """
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    opener = opener if opener is not None else _default_opener
    runner = runner if runner is not None else _default_runner
    confirm = confirm if confirm is not None else (lambda _item: True)

    wanted = (item for item in items if not item.present)
    found = (_fetch_one(item, cache_dir, opener, runner, confirm) for item in wanted)
    return [problem for problem in found if problem is not None]


def _fetch_one(
    item: PrefetchItem,
    cache_dir: Path,
    opener: OpenerFn,
    runner: RunnerFn,
    confirm: ConfirmFn,
) -> FetchProblem | None:
    if not item.source:
        return FetchProblem(
            item=item.name,
            code="no_pin_for_platform",
            message=f"no pinned {item.kind} for this platform",
        )
    if not confirm(item):
        return None
    if item.kind == "image":
        return _pull_image(item, runner)
    return _download(item, cache_dir, opener)


def _pull_image(item: PrefetchItem, runner: RunnerFn) -> FetchProblem | None:
    ref, _, digest = item.source.partition("@")
    try:
        returncode = runner(["docker", "pull", f"{ref}@{digest}"])
    except Exception as exc:  # noqa: BLE001
        # Report, never raise.
        return FetchProblem(
            item=item.name, code="download_failed", message=f"docker pull failed: {exc}"
        )
    if returncode != 0:
        return FetchProblem(
            item=item.name, code="download_failed", message=f"docker pull exited {returncode}"
        )
    return None


class _HttpsOnlyRedirects(HTTPRedirectHandler):
    """Follow a redirect only to another https:// URL.

    huggingface.co answers a ``resolve`` URL with a redirect to its storage
    CDN, so redirects have to be followed and the final host is not pinned.
    What is pinned is the content: the size and sha256 from ``pins.json`` are
    checked before a file is accepted, whichever host served it. The scheme
    is still held to https so the transfer cannot be downgraded to plain http.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: PLR0913
        if urlsplit(newurl).scheme != "https":
            raise HTTPError(newurl, code, "redirect to a non-https URL refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_opener(request: Request):  # pragma: no cover - exercised only via real network
    opener = build_opener(_HttpsOnlyRedirects)
    return opener.open(request, timeout=60)  # nosec B310


def _default_runner(argv: list[str]) -> int:  # pragma: no cover - exercised only with real docker
    import subprocess  # nosec B404 - fixed argv, no shell=True

    completed = subprocess.run(argv, check=False)  # nosec B603
    return completed.returncode


# ---------------------------------------------------------------------------
# resolve() / verify_all(): local-only, never touch the network.
# ---------------------------------------------------------------------------

#: Per-(path, mtime, size) verification cache, so repeated resolve() calls
#: on an unchanged file don't re-hash it every time.
_verify_cache: dict[tuple[str, float, int], str] = {}


def _verified_sha256(path: Path) -> str:
    stat = path.stat()
    # st_ctime_ns and st_ino cannot be set from user space, unlike mtime: a
    # file swapped or edited in place always changes at least one of them, so
    # a cached verdict can never vouch for different bytes.
    key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _verify_cache.get(key)
    if cached is not None:
        return cached
    digest = _sha256_file(path)
    _verify_cache[key] = digest
    return digest


def resolve(
    kind: str,
    *,
    pins: dict | None = None,
    cache_dir: Path | None = None,
    platform_tag: str | None = None,
) -> Path | FetchProblem:
    """Resolve ``kind`` ("weights" or "engine") to its verified local path.

    Local only: stats and (cache-permitting) hashes a file already on disk.
    Never calls into ``urllib`` or opens a socket. The verified sha256 is
    cached per ``(path, mtime, size)`` in :data:`_verify_cache`, so repeated
    calls after the first are just a ``stat()``.
    """
    pins = pins if pins is not None else load_pins()
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    platform_tag = platform_tag if platform_tag is not None else _default_platform_tag()

    if kind == "weights":
        pin = _weights_pin(pins)
        if pin is None:
            return FetchProblem(item="weights", code="missing", message="no weights pin")
    elif kind == "engine":
        pin = _engine_pin(pins, platform_tag)
        if pin is None:
            return FetchProblem(
                item=f"engine-{platform_tag}",
                code="no_pin_for_platform",
                message="no pinned engine for this platform",
            )
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected 'weights' or 'engine')")

    filename = pin["filename"]
    local = _cache_path(cache_dir, filename)
    if not local.is_file():
        return FetchProblem(item=filename, code="missing", message=f"{filename} not cached")

    stat = local.stat()
    if stat.st_size != pin["size_bytes"]:
        return FetchProblem(
            item=filename,
            code="size_mismatch",
            message=f"expected {pin['size_bytes']} bytes, got {stat.st_size}",
        )

    digest = _verified_sha256(local)
    if digest != pin["sha256"]:
        return FetchProblem(
            item=filename,
            code="hash_mismatch",
            message=f"expected sha256 {pin['sha256']}, got {digest}",
        )

    return local


def verify_all(
    pins: dict | None = None,
    *,
    cache_dir: Path | None = None,
    platform_tag: str | None = None,
    image_present: Callable[[dict], bool] | None = None,
) -> list[FetchProblem]:
    """Verify every pinned item this platform cares about. Local only.

    Used by ``nvsh doctor``. Returns one :class:`FetchProblem` per item that
    is missing, mismatched, or -- for the engine -- unpinned on this
    platform; a healthy cache returns an empty list. Image entries are
    reported ``missing`` unless ``image_present`` (an injectable "does this
    digest exist locally" check) says otherwise -- ``verify_all`` never
    shells out to ``docker`` itself.
    """
    pins = pins if pins is not None else load_pins()
    cache_dir = cache_dir if cache_dir is not None else default_cache_dir()
    platform_tag = platform_tag if platform_tag is not None else _default_platform_tag()
    image_present = image_present if image_present is not None else (lambda _img: False)

    problems: list[FetchProblem] = []

    for kind in ("weights", "engine"):
        result = resolve(kind, pins=pins, cache_dir=cache_dir, platform_tag=platform_tag)
        if isinstance(result, FetchProblem):
            problems.append(result)

    for image in pins.get("images", []):
        if not image_present(image):
            problems.append(
                FetchProblem(
                    item=image["name"],
                    code="missing",
                    message=f"{image['name']} not pulled",
                )
            )

    return problems
