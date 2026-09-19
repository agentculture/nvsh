"""Tests for nvsh.tiers.fetch — task t8: pinned fetch and prefetch.

Acceptance criteria covered:
- A cached file whose sha256 differs from pins.json is refused and reported
  (resolve()); a matching file is accepted.
- plan_prefetch() lists each item (engine, weights, image) with its size
  before anything downloads, and is pure (never opens a socket).
- After prefetch(), resolve() makes no network call — asserted with a
  socket guard that raises if socket.socket/create_connection is used.
- pins.json ships in the built wheel (extends test_wheel_packaging.py).

Everything that could hit the network or a real docker is injected with a
fake in every test; nothing here ever calls the real huggingface.co or a
real ``docker`` binary.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket

import pytest

from nvsh.tiers import fetch

WEIGHTS_BYTES = b"weights-payload-bytes-needle3" * 100
ENGINE_BYTES = b"engine-wheel-payload-bytes" * 50


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_pins() -> dict:
    return {
        "needle3": {
            "repo": "Cactus-Compute/needle3",
            "revision": "deadbeef",
            "weights": {
                "filename": "needle3.cact",
                "sha256": _sha256(WEIGHTS_BYTES),
                "size_bytes": len(WEIGHTS_BYTES),
            },
            "engines": {
                "aarch64": {
                    "filename": "cactus_needle-3.0.1-py3-none-aarch64.whl",
                    "sha256": _sha256(ENGINE_BYTES),
                    "size_bytes": len(ENGINE_BYTES),
                }
            },
        },
        "images": [
            {
                "name": "lfm-runtime",
                "engine": "llama-server",
                "ref": "ghcr.io/ggml-org/llama.cpp",
                "digest": "sha256:" + ("ab" * 32),
                "size_bytes": 123456789,
            }
        ],
    }


class _FakeResponse:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.closed = False

    def read(self, n=-1):
        return self._buf.read(n)

    def close(self):
        self.closed = True


def _opener_for(mapping: dict[str, bytes]):
    def _opener(request):
        url = request.full_url
        for suffix, data in mapping.items():
            if url.endswith(suffix):
                return _FakeResponse(data)
        raise AssertionError(f"unexpected fetch URL: {url}")

    return _opener


def _forbidden_opener(request):
    raise AssertionError(f"network access attempted: {request.full_url}")


class _SocketGuard:
    """Monkeypatches socket.socket/create_connection to raise if touched."""

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def __enter__(self):
        def _boom(*args, **kwargs):
            raise AssertionError("socket access attempted")

        self._monkeypatch.setattr(socket, "socket", _boom)
        self._monkeypatch.setattr(socket, "create_connection", _boom)
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# load_pins / pins.json shape
# ---------------------------------------------------------------------------


def test_load_pins_reads_real_pins_json():
    pins = fetch.load_pins()
    assert pins["needle3"]["weights"]["filename"] == "needle3.cact"
    assert pins["needle3"]["weights"]["size_bytes"] == 35335380
    assert len(pins["needle3"]["weights"]["sha256"]) == 64
    assert "aarch64" in pins["needle3"]["engines"]
    assert pins["images"] == []


def test_real_pins_json_is_valid_json_on_disk():
    with open(fetch.PINS_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# plan_prefetch: pure, lists sizes, never opens a socket
# ---------------------------------------------------------------------------


def test_plan_prefetch_lists_weights_engine_and_image_with_sizes(tmp_path):
    pins = _fake_pins()
    items = fetch.plan_prefetch(
        pins, platform_tag="aarch64", cache_dir=tmp_path, image_present=lambda _i: False
    )
    kinds = {item.kind: item for item in items}
    assert set(kinds) == {"weights", "engine", "image"}
    assert kinds["weights"].size_bytes == len(WEIGHTS_BYTES)
    assert kinds["weights"].present is False
    assert kinds["engine"].size_bytes == len(ENGINE_BYTES)
    assert kinds["image"].size_bytes == 123456789
    assert kinds["image"].source == "ghcr.io/ggml-org/llama.cpp@sha256:" + ("ab" * 32)


def test_plan_prefetch_marks_present_when_cached_file_matches_size(tmp_path):
    pins = _fake_pins()
    (tmp_path / "needle3.cact").write_bytes(WEIGHTS_BYTES)
    items = fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
    weights_item = next(i for i in items if i.kind == "weights")
    assert weights_item.present is True


def test_plan_prefetch_reports_no_pin_for_unknown_platform(tmp_path):
    pins = _fake_pins()
    items = fetch.plan_prefetch(pins, platform_tag="x86_64", cache_dir=tmp_path)
    engine_item = next(i for i in items if i.kind == "engine")
    assert engine_item.present is False
    assert engine_item.source == ""
    assert engine_item.sha256 == ""


def test_plan_prefetch_never_touches_a_socket(tmp_path, monkeypatch):
    pins = _fake_pins()
    with _SocketGuard(monkeypatch):
        items = fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
    assert len(items) == 3


# ---------------------------------------------------------------------------
# prefetch(): the one executing function
# ---------------------------------------------------------------------------


def test_prefetch_downloads_missing_items_and_verifies_them(tmp_path):
    pins = _fake_pins()
    items = fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
    opener = _opener_for(
        {
            "needle3.cact": WEIGHTS_BYTES,
            "cactus_needle-3.0.1-py3-none-aarch64.whl": ENGINE_BYTES,
        }
    )
    calls = []

    def runner(argv):
        calls.append(argv)
        return 0

    problems = fetch.prefetch(items, cache_dir=tmp_path, opener=opener, runner=runner)
    assert problems == []
    assert (tmp_path / "needle3.cact").read_bytes() == WEIGHTS_BYTES
    assert (tmp_path / "cactus_needle-3.0.1-py3-none-aarch64.whl").read_bytes() == ENGINE_BYTES
    assert calls == [["docker", "pull", "ghcr.io/ggml-org/llama.cpp@sha256:" + ("ab" * 32)]]


def test_prefetch_skips_items_already_present(tmp_path):
    pins = _fake_pins()
    (tmp_path / "needle3.cact").write_bytes(WEIGHTS_BYTES)
    (tmp_path / "cactus_needle-3.0.1-py3-none-aarch64.whl").write_bytes(ENGINE_BYTES)
    items = fetch.plan_prefetch(
        pins, platform_tag="aarch64", cache_dir=tmp_path, image_present=lambda _i: True
    )
    problems = fetch.prefetch(
        items, cache_dir=tmp_path, opener=_forbidden_opener, runner=lambda argv: 0
    )
    assert problems == []


def test_prefetch_reports_hash_mismatch_and_does_not_leave_a_partial_file(tmp_path):
    pins = _fake_pins()
    items = fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
    bad_bytes = b"not-the-right-payload" * 10
    opener = _opener_for(
        {
            "needle3.cact": bad_bytes,
            "cactus_needle-3.0.1-py3-none-aarch64.whl": ENGINE_BYTES,
        }
    )
    problems = fetch.prefetch(items, cache_dir=tmp_path, opener=opener, runner=lambda argv: 0)
    codes = {p.item: p.code for p in problems}
    assert codes["needle3.cact"] in ("hash_mismatch", "size_mismatch")
    assert not (tmp_path / "needle3.cact").exists()


def test_prefetch_reports_no_pin_for_platform_without_fetching(tmp_path):
    pins = _fake_pins()
    items = fetch.plan_prefetch(pins, platform_tag="x86_64", cache_dir=tmp_path)
    problems = fetch.prefetch(
        items, cache_dir=tmp_path, opener=_forbidden_opener, runner=lambda argv: 0
    )
    engine_problems = [p for p in problems if p.item.startswith("needle3-engine")]
    assert engine_problems and engine_problems[0].code == "no_pin_for_platform"


def test_prefetch_docker_pull_uses_exact_argv_with_digest(tmp_path):
    pins = _fake_pins()
    items = [
        i
        for i in fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
        if i.kind == "image"
    ]
    calls = []

    def runner(argv):
        calls.append(argv)
        return 0

    problems = fetch.prefetch(items, cache_dir=tmp_path, opener=_forbidden_opener, runner=runner)
    assert problems == []
    assert calls == [["docker", "pull", "ghcr.io/ggml-org/llama.cpp@sha256:" + ("ab" * 32)]]


def test_prefetch_docker_pull_nonzero_exit_is_reported(tmp_path):
    pins = _fake_pins()
    items = [
        i
        for i in fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
        if i.kind == "image"
    ]
    problems = fetch.prefetch(
        items, cache_dir=tmp_path, opener=_forbidden_opener, runner=lambda argv: 1
    )
    assert len(problems) == 1
    assert problems[0].code == "download_failed"


def test_prefetch_only_fetches_confirmed_items(tmp_path):
    pins = _fake_pins()
    items = fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
    opener = _opener_for(
        {
            "needle3.cact": WEIGHTS_BYTES,
            "cactus_needle-3.0.1-py3-none-aarch64.whl": ENGINE_BYTES,
        }
    )
    seen = []

    def confirm(item):
        seen.append(item.kind)
        return item.kind != "weights"

    problems = fetch.prefetch(
        items, cache_dir=tmp_path, opener=opener, runner=lambda argv: 0, confirm=confirm
    )
    assert not (tmp_path / "needle3.cact").exists()
    assert (tmp_path / "cactus_needle-3.0.1-py3-none-aarch64.whl").exists()
    assert problems == []
    assert "weights" in seen


def test_download_rejects_non_huggingface_url(tmp_path):
    item = fetch.PrefetchItem(
        kind="weights",
        name="evil",
        source="http://example.com/evil",
        sha256="0" * 64,
        size_bytes=1,
        present=False,
    )
    with pytest.raises(ValueError):
        fetch._download(item, tmp_path, opener=_forbidden_opener)


def test_download_raises_before_opening_for_non_https_scheme(tmp_path):
    item = fetch.PrefetchItem(
        kind="weights",
        name="evil",
        source="ftp://huggingface.co/x",
        sha256="0" * 64,
        size_bytes=1,
        present=False,
    )
    with pytest.raises(ValueError):
        fetch._download(item, tmp_path, opener=_forbidden_opener)


def test_download_raises_before_opening_for_wrong_host(tmp_path):
    item = fetch.PrefetchItem(
        kind="weights",
        name="evil",
        source="https://evil.example.com/x",
        sha256="0" * 64,
        size_bytes=1,
        present=False,
    )
    with pytest.raises(ValueError):
        fetch._download(item, tmp_path, opener=_forbidden_opener)


# ---------------------------------------------------------------------------
# resolve(): local only, verifies sha256, never touches the network
# ---------------------------------------------------------------------------


def test_resolve_accepts_matching_file(tmp_path):
    pins = _fake_pins()
    (tmp_path / "needle3.cact").write_bytes(WEIGHTS_BYTES)
    result = fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    assert result == tmp_path / "needle3.cact"


def test_resolve_refuses_file_whose_sha256_differs(tmp_path):
    pins = _fake_pins()
    (tmp_path / "needle3.cact").write_bytes(b"totally different bytes, same length as needed!!")
    result = fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    assert isinstance(result, fetch.FetchProblem)
    assert result.code in ("hash_mismatch", "size_mismatch")


def test_resolve_reports_missing_when_not_cached(tmp_path):
    pins = _fake_pins()
    result = fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    assert isinstance(result, fetch.FetchProblem)
    assert result.code == "missing"


def test_resolve_reports_no_pin_for_platform(tmp_path):
    pins = _fake_pins()
    result = fetch.resolve("engine", pins=pins, cache_dir=tmp_path, platform_tag="x86_64")
    assert isinstance(result, fetch.FetchProblem)
    assert result.code == "no_pin_for_platform"


def test_resolve_rejects_unknown_kind(tmp_path):
    pins = _fake_pins()
    with pytest.raises(ValueError):
        fetch.resolve("bogus", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")


def test_resolve_never_touches_a_socket(tmp_path, monkeypatch):
    pins = _fake_pins()
    (tmp_path / "needle3.cact").write_bytes(WEIGHTS_BYTES)
    with _SocketGuard(monkeypatch):
        result = fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    assert result == tmp_path / "needle3.cact"


def test_prefetch_then_resolve_makes_no_network_call(tmp_path, monkeypatch):
    """Criterion 2: after prefetch, resolve() makes no network call."""
    pins = _fake_pins()
    items = fetch.plan_prefetch(pins, platform_tag="aarch64", cache_dir=tmp_path)
    opener = _opener_for(
        {
            "needle3.cact": WEIGHTS_BYTES,
            "cactus_needle-3.0.1-py3-none-aarch64.whl": ENGINE_BYTES,
        }
    )
    problems = fetch.prefetch(items, cache_dir=tmp_path, opener=opener, runner=lambda argv: 0)
    assert problems == []

    with _SocketGuard(monkeypatch):
        weights_path = fetch.resolve(
            "weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64"
        )
        engine_path = fetch.resolve("engine", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    assert weights_path == tmp_path / "needle3.cact"
    assert engine_path == tmp_path / "cactus_needle-3.0.1-py3-none-aarch64.whl"


def test_resolve_caches_verification_per_path_mtime_size(tmp_path, monkeypatch):
    pins = _fake_pins()
    (tmp_path / "needle3.cact").write_bytes(WEIGHTS_BYTES)
    calls = []
    real_sha256_file = fetch._sha256_file

    def counting_sha256(path):
        calls.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(fetch, "_sha256_file", counting_sha256)
    fetch._verify_cache.clear()
    fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# verify_all()
# ---------------------------------------------------------------------------


def test_verify_all_reports_all_missing_when_cache_empty(tmp_path):
    pins = _fake_pins()
    problems = fetch.verify_all(pins, cache_dir=tmp_path, platform_tag="aarch64")
    codes = {p.item: p.code for p in problems}
    assert codes["needle3.cact"] == "missing"
    assert codes["cactus_needle-3.0.1-py3-none-aarch64.whl"] == "missing"
    assert codes["lfm-runtime"] == "missing"


def test_verify_all_reports_clean_when_everything_matches_and_image_present(tmp_path):
    pins = _fake_pins()
    (tmp_path / "needle3.cact").write_bytes(WEIGHTS_BYTES)
    (tmp_path / "cactus_needle-3.0.1-py3-none-aarch64.whl").write_bytes(ENGINE_BYTES)
    problems = fetch.verify_all(
        pins, cache_dir=tmp_path, platform_tag="aarch64", image_present=lambda _i: True
    )
    assert problems == []


class _EndlessResponse:
    """A server that never stops streaming."""

    def __init__(self):
        self.reads = 0

    def read(self, n=-1):
        self.reads += 1
        if self.reads > 10_000:
            raise AssertionError("download was not bounded by the pinned size")
        return b"x" * max(n, 1)

    def close(self):
        pass


def test_download_stops_at_the_pinned_size(tmp_path):
    items = fetch.plan_prefetch(_fake_pins(), platform_tag="aarch64", cache_dir=tmp_path)
    weights = [i for i in items if i.kind == "weights"]
    problems = fetch.prefetch(
        weights, cache_dir=tmp_path, opener=lambda request: _EndlessResponse(), runner=lambda a: 0
    )
    assert [p.code for p in problems] == ["size_mismatch"]
    assert [p.name for p in tmp_path.iterdir()] == []


class _BrokenResponse:
    def read(self, n=-1):
        raise ConnectionResetError("reset mid-stream")

    def close(self):
        pass


def test_mid_stream_error_is_reported_not_raised(tmp_path):
    items = fetch.plan_prefetch(_fake_pins(), platform_tag="aarch64", cache_dir=tmp_path)
    weights = [i for i in items if i.kind == "weights"]
    problems = fetch.prefetch(
        weights, cache_dir=tmp_path, opener=lambda request: _BrokenResponse(), runner=lambda a: 0
    )
    assert [p.code for p in problems] == ["download_failed"]
    assert [p.name for p in tmp_path.iterdir()] == []


def test_verdict_cache_is_not_fooled_by_a_same_size_same_mtime_swap(tmp_path):
    """qodo 1 on PR #32: swapping bytes while restoring size and mtime must be caught."""
    pins = _fake_pins()
    target = tmp_path / "needle3.cact"
    target.write_bytes(WEIGHTS_BYTES)
    assert fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64") == target
    before = target.stat()
    target.write_bytes(bytes(len(WEIGHTS_BYTES)))  # same size, different content
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    result = fetch.resolve("weights", pins=pins, cache_dir=tmp_path, platform_tag="aarch64")
    assert isinstance(result, fetch.FetchProblem)
    assert result.code == "hash_mismatch"


class _IncompleteResponse:
    def read(self, n=-1):
        import http.client

        raise http.client.IncompleteRead(b"partial")

    def close(self):
        pass


def test_a_non_oserror_body_failure_is_reported_not_raised(tmp_path):
    """qodo 5 on PR #32: IncompleteRead is an HTTPException, not an OSError."""
    items = fetch.plan_prefetch(_fake_pins(), platform_tag="aarch64", cache_dir=tmp_path)
    weights = [i for i in items if i.kind == "weights"]
    problems = fetch.prefetch(
        weights,
        cache_dir=tmp_path,
        opener=lambda request: _IncompleteResponse(),
        runner=lambda a: 0,
    )
    assert [p.code for p in problems] == ["download_failed"]


def test_redirects_are_followed_only_to_https():
    """qodo 7 on PR #32: the CDN hop is allowed, a downgrade to http is not."""
    from urllib.error import HTTPError
    from urllib.request import Request

    handler = fetch._HttpsOnlyRedirects()
    request = Request("https://huggingface.co/x/resolve/abc/f.bin")
    followed = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example/f")
    assert followed.full_url == "https://cdn.example/f"
    with pytest.raises(HTTPError):
        handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example/f")
