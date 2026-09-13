#!/usr/bin/env bash
# challenge.sh — risk-scaled blind-spot pass over a converged frame (the
# /challenge skill).
#
# The skill is named `challenge`; the product/CLI it drives is `devague`.
# /challenge runs BETWEEN /think and /spec-to-plan: pressure-test the exported
# frame through structured lenses, then route every finding back through the
# existing deterministic moves as proposed-only content the human adjudicates.
# On a clean pass it records the examined lenses/surfaces and residual
# uncertainty — never a claim that there are no unknown unknowns.
#
# There is no single `devague challenge` verb: the pass is orchestrated from
# several top-level moves — `capture` (proposed findings), `interrogate`,
# `question` / `question --resolve`, `park`, `confirm`, `converge`, `export`,
# and `plan new` / `plan risk` on the plan side. So this wrapper forwards every
# argument to `devague <args>` verbatim (the think.sh shape), letting the CLI's
# own parser own the surface.
#
# Origin: authored and maintained in agentculture/devague. guildmaster pulls
# this skill from there and broadcasts it to the AgentCulture mesh, so it is
# written to run anywhere — portable bash, no devague-checkout assumptions.
#
# Frames persist under .devague/ in the current directory — run from the repo
# whose frame you are challenging.

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
challenge.sh — risk-scaled blind-spot pass over a converged frame (the
/challenge skill).

Usage:
  challenge.sh <move> [args...]   forward a devague move (capture / interrogate /
                                  question / park / confirm / converge / export /
                                  plan new / plan risk ...)
  challenge.sh question --resolve <id> ...   resolve a raised challenge question
  challenge.sh help               this help

Every argument is forwarded verbatim as `devague <args>`. There is no single
`challenge` verb — the pass routes findings back through the existing
deterministic moves as proposed-only content the human adjudicates. Run
`devague learn` for the method.

Flow: /think exports a frame → /challenge (this skill) → /spec-to-plan.
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
            # the challenge pass orchestrates many top-level moves, no fixed prefix.
            resolve_devague
            exec "${DEVAGUE[@]}" "$@"
            ;;
    esac
}

main "$@"
