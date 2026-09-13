#!/usr/bin/env bash
# scope.sh — explore the scope of a vague idea before framing (the /scope skill).
#
# The skill is named `scope`; the product/CLI it drives is `devague`. /scope is
# the optional opening move ahead of /think: survey the surfaces an idea touches
# (code, docs, skills, CI, sibling repos) and seed the coming Announcement Frame
# with boundary / non-goal / assumption claims that cite what was actually
# explored (provenance, not generic disclaimers).
#
# Because /scope uses several top-level devague moves — `scope` (and `--list`),
# `capture --kind boundary|non_goal|assumption`, `new`, `question`, `show` — this
# wrapper forwards every argument to `devague <args>` verbatim (the think.sh
# shape), so the CLI's own parser owns the surface and new moves work without
# editing this script.
#
# Origin: authored and maintained in agentculture/devague. guildmaster pulls
# this skill from there and broadcasts it to the AgentCulture mesh, so it is
# written to run anywhere — portable bash, no devague-checkout assumptions.
#
# Frames persist under .devague/ in the current directory — run from the repo
# whose idea you are scoping.

set -euo pipefail

# ── resolve the devague CLI (mesh-first, then local-dev fallback) ───────────
DEVAGUE=()
resolve_devague() {
    if command -v devague >/dev/null 2>&1; then
        DEVAGUE=(devague)            # installed tool — the normal mesh case
        return 0
    fi
    # Local-dev fallback: inside the devague checkout, run via uv.
    local dir="$PWD"
    while [ -n "$dir" ] && [ "$dir" != "/" ]; do
        if [ -f "$dir/pyproject.toml" ] \
            && grep -q '^name = "devague"' "$dir/pyproject.toml" 2>/dev/null; then
            if command -v uv >/dev/null 2>&1; then
                DEVAGUE=(uv run devague)
                return 0
            fi
            break
        fi
        dir=$(dirname "$dir")
    done
    cat >&2 <<'EOF'
error: devague CLI not found.
hint: install it with `uv tool install devague` (or `pipx install devague`),
      or run from inside the devague checkout with `uv` available.
      https://github.com/agentculture/devague
EOF
    return 1
}

usage() {
    cat <<'EOF'
scope.sh — explore the scope of a vague idea before framing (the /scope skill).

Usage:
  scope.sh scope [--list] [--json]   manage/list scope findings for the frame
  scope.sh capture --kind boundary|non_goal|assumption --origin llm ...
  scope.sh <move> [args...]          any other devague move (new / question / show)
  scope.sh help                      this help

Every argument is forwarded verbatim as `devague <args>`, so the CLI owns the
surface and new moves work without editing this script. Run `devague explain
scope` for the surface, `devague learn` for the method.

Next leg: hand off to the /think skill to build the Announcement Frame from the
scope you seeded.
EOF
}

main() {
    case "${1:-help}" in
        help | -h | --help)
            usage
            return 0
            ;;
        *)
            # Forward every move to `devague <move>` verbatim (think.sh shape) —
            # /scope orchestrates several top-level moves, so no fixed prefix.
            resolve_devague
            exec "${DEVAGUE[@]}" "$@"
            ;;
    esac
}

main "$@"
