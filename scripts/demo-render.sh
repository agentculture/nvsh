#!/usr/bin/env bash
# Render a committed asciicast (.cast) recording into a static SVG for the
# README, via svg-term-cli run through `npx --yes` (no runtime dependency
# added to pyproject.toml or nvsh/ — this is a docs-build-time tool only).
#
# Usage:
#   scripts/demo-render.sh <in.cast> <out.svg>
#
# Example:
#   scripts/demo-render.sh docs/demos/spark-cuda-oom.cast docs/demos/spark-cuda-oom.svg
#
# Requires `npx` on PATH (ships with Node.js) and network access the first
# time, to fetch the pinned svg-term-cli version into npm's local cache;
# subsequent runs reuse that cache.
#
# ---------------------------------------------------------------------------
# GIF fallback (offline-friendly alternative to svg-term-cli)
# ---------------------------------------------------------------------------
# If svg-term-cli cannot be installed (no network, or npm/npx unavailable),
# render a GIF instead with `agg` (asciinema-agg), a standalone Rust binary
# with no npm/node dependency:
#
#   agg in.cast out.gif --cols 100 --rows 34
#
# Prebuilt binaries: https://github.com/asciinema/agg/releases
#
# ---------------------------------------------------------------------------

set -euo pipefail

SVG_TERM_VERSION="2.1.1"

usage() {
    echo "Usage: $(basename "$0") <in.cast> <out.svg>" >&2
    echo "Renders an asciicast recording into an SVG via svg-term-cli@${SVG_TERM_VERSION}." >&2
}

if [ "$#" -ne 2 ]; then
    usage
    exit 1
fi

IN_CAST="$1"
OUT_SVG="$2"

if ! command -v npx >/dev/null 2>&1; then
    echo "error: npx not found on PATH" >&2
    echo "hint: install Node.js (npx ships with it) or render offline with agg — see the fallback comment in this script" >&2
    exit 2
fi

if [ ! -f "$IN_CAST" ]; then
    echo "error: input cast file not found: $IN_CAST" >&2
    echo "hint: pass a path to a recorded .cast file, e.g. docs/demos/spark-cuda-oom.cast" >&2
    exit 1
fi

OUT_DIR="$(dirname -- "$OUT_SVG")"
mkdir -p -- "$OUT_DIR"

npx --yes "svg-term-cli@${SVG_TERM_VERSION}" \
    --in "$IN_CAST" \
    --out "$OUT_SVG" \
    --no-window \
    --width 100 \
    --height 34

echo "wrote $OUT_SVG"
