#!/usr/bin/env bash
# validate-delivery.sh — file behavioral obligations, evidence, and deltas after
# a workforce run (the /validate-delivery skill).
#
# The skill is named `validate-delivery`; the product/CLI it drives is
# `devague`. It runs between /assign-to-workforce (which merges the waves) and
# /summarize-delivery (which closes the loop): the agent executes the confirmed
# plan's behavioral tests itself, then records what was found — evidence for
# what passed, behavioral deltas for what the run added, amended, or removed.
#
# Record-only: the devague CLI never runs a test. This wrapper only files the
# results the agent already obtained.
#
# Because /validate-delivery uses several top-level devague moves — `oblige`,
# `evidence`, `delta`, `summary` — this wrapper forwards every argument to
# `devague <args>` verbatim (the scope.sh shape), so the CLI's own parser owns
# the surface and new moves work without editing this script.
#
# Origin: authored and maintained in agentculture/devague. guildmaster pulls
# this skill from there and broadcasts it to the AgentCulture mesh, so it is
# written to run anywhere — portable bash, no devague-checkout assumptions.
#
# Plans/records persist under .devague/ in the current directory — run from the
# repo whose plan you executed.

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
validate-delivery.sh — file obligations, evidence, and behavioral deltas
after a workforce run (the /validate-delivery skill).

Usage:
  validate-delivery.sh oblige <cN> --seam "<seam>" --behavior "<behavior>"
  validate-delivery.sh evidence --obligation <oN> --test "<ref>" ... --outcome pass|fail
  validate-delivery.sh delta --kind added|amended|removed --behavior "<text>" --caused-by <cN|dN>
  validate-delivery.sh summary [--pr] [--json]
  validate-delivery.sh <move> [args...]   any other devague move
  validate-delivery.sh help               this help

Every argument is forwarded verbatim as `devague <args>`, so the CLI owns the
surface and new moves work without editing this script. Run `devague explain
<move>` for a surface, `devague learn` for the method.

Record-only: run the plan's behavioral tests yourself, then file what you
found. The CLI never executes a test on your behalf.

Next leg: hand off to the /summarize-delivery skill to close the loop.
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
            # /validate-delivery drives several top-level moves, so no fixed prefix.
            resolve_devague
            exec "${DEVAGUE[@]}" "$@"
            ;;
    esac
}

main "$@"
