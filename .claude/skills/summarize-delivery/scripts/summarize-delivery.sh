#!/usr/bin/env bash
# summarize-delivery.sh — close the loop after an assign-to-workforce run (the
# /summarize-delivery skill).
#
# The skill is named `summarize-delivery`; the product/CLI it drives is
# `devague`. It turns what actually happened into an accountability artifact:
# planned-versus-actual delivery, mid-work decisions, plan drift, evidence-backed
# delivery claims, and remaining work. The confirmed plan is the contract; this
# records where execution obeyed it, where it changed, and what is genuinely safe
# to claim as delivered. Runs on complete, partial, AND failed runs — failure is
# reported faithfully, never smoothed over.
#
# It reads state across several top-level devague moves — `summary`, `plan`
# (`plan waves --json`, `plan show`), `deviate --list`, `scope --list`, `show` —
# so this wrapper forwards every argument to `devague <args>` verbatim (the
# think.sh shape), letting the CLI's own parser own the surface.
#
# Origin: authored and maintained in agentculture/devague. guildmaster pulls
# this skill from there and broadcasts it to the AgentCulture mesh, so it is
# written to run anywhere — portable bash, no devague-checkout assumptions.
#
# Plans persist under .devague/ in the current directory — run from the repo
# whose delivery you are summarizing.

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
summarize-delivery.sh — close the loop after an assign-to-workforce run (the
/summarize-delivery skill).

Usage:
  summarize-delivery.sh summary [args...]     render the delivery accountability artifact
  summarize-delivery.sh plan waves --json     read the wave graph the run executed
  summarize-delivery.sh deviate --list        list the deviations recorded mid-run
  summarize-delivery.sh <move> [args...]      any other devague move (plan show / scope --list / show)
  summarize-delivery.sh help                  this help

Every argument is forwarded verbatim as `devague <args>`, so the CLI owns the
surface and new moves work without editing this script. The confirmed plan is
the contract — report complete, partial, AND failed runs faithfully; never
smooth over failure.
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
            # the summary reads across several top-level moves, no fixed prefix.
            resolve_devague
            exec "${DEVAGUE[@]}" "$@"
            ;;
    esac
}

main "$@"
