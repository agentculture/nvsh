#!/usr/bin/env bash
# deviate.sh — record/inspect in-flight plan deviations (the /deviate skill).
#
# The skill is named `deviate`; the product/CLI it drives is `devague`. This
# wrapper is the single-verb analog of spec-to-plan.sh: it forwards every
# argument to `devague deviate <args>` verbatim, so the CLI's own parser owns
# the deviation surface (--confirm / --reject / --list / --json and any future
# flags) without editing this script.
#
# Stop an assign-to-workforce run the moment execution must diverge from the
# confirmed plan, get explicit human approval, and record the divergence as a
# first-class, append-only deviation record — never fold a deviation silently
# into drift after the fact.
#
# Origin: authored and maintained in agentculture/devague. guildmaster pulls
# this skill from there and broadcasts it to the AgentCulture mesh, so it is
# written to run anywhere — portable bash, no devague-checkout assumptions.
#
# Plans/deviations persist under .devague/ in the current directory — run from
# the repo whose plan you are executing.

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
deviate.sh — record/inspect in-flight plan deviations (the /deviate skill).

Usage:
  deviate.sh <reason> [args...]   record a deviation (forwarded to `devague deviate`)
  deviate.sh --list [--json]      list recorded deviations for the plan
  deviate.sh --confirm <id>       confirm a proposed deviation (USER-only decision)
  deviate.sh --reject <id>        reject a proposed deviation
  deviate.sh help                 this help

Every argument is forwarded verbatim as `devague deviate <args>`, so new flags
work without editing this script. Run `devague explain deviate` for the surface.

A deviation is a first-class, append-only record — approve it explicitly before
resuming a workforce run; do not let a divergence become silent drift.
EOF
}

main() {
    case "${1:-help}" in
        help | -h | --help)
            usage
            return 0
            ;;
        *)
            # Forward every argument to `devague deviate <args>` verbatim so the
            # CLI's own parser owns the deviation surface.
            resolve_devague
            exec "${DEVAGUE[@]}" deviate "$@"
            ;;
    esac
}

main "$@"
