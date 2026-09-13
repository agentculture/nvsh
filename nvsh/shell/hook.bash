# shellcheck shell=bash
# nvsh bash hook core.
#
# Sourced from the operator's ~/.bashrc (see `nvsh setup`). It installs one
# PROMPT_COMMAND element, `__nvsh_hook`, as the FIRST element of the array so
# that `$?` and `PIPESTATUS` are still the user's command's when it runs: a
# hook placed after another integration's entry (Ghostty's `__ghostty_hook`,
# for instance) sees PIPESTATUS already clobbered.
#
# Rules this file obeys:
#   * pure bash 5.1+, no external processes on the success path (no fork),
#   * every function ends with `|| return 0` semantics - nothing here may
#     abort the operator's shell, and `set -e` is never used,
#   * no DEBUG trap is installed; the OSC 133 "command start" marker rides on
#     PS0, exactly as Ghostty's own integration does on bash >= 5.2,
#   * NVSH_DISABLE=1 makes the whole file a no-op (no hook, no capture).
#
# Environment knobs:
#   NVSH_DISABLE                 1 -> this file does nothing at all
#   NVSH_CAPTURE                 0 -> no session-log capture (no script(1))
#   NVSH_AUTO                    0 -> never auto-call the agent
#   NVSH_LOG                     path of the session typescript (exported)
#   NVSH_WRAPPED                 1 -> already running under the capture exec
#   NVSH_INTERACTIVE_PROGRAMS    space-separated first-word skip list
#   NVSH_BIN                     name/path of the nvsh entrypoint
#   NVSH_HOOK_DEBUG_FILE         append "<exit>\t<PIPESTATUS>" per prompt
#
# Shell variables (not environment knobs):
#   __NVSH_SLASH_DISPATCH        set by readline.bash on the hidden
#                                ` nvsh slash ...` line it accepts for the
#                                operator; consumed here, never auto-triggers

# --- kill switch and preconditions ----------------------------------------

[[ -n ${NVSH_DISABLE:-} && ${NVSH_DISABLE} != 0 ]] && return 0
[[ $- == *i* ]] || return 0
((BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1))) || return 0
[[ -n ${__NVSH_HOOK_LOADED:-} ]] && return 0

# --- the one source of truth for the bash-side skip list ------------------
# Kept in sync with nvsh.triggers.INTERACTIVE_PROGRAMS by
# tests/test_hook_bash.py. `nvsh setup` (t21) may render an override into
# NVSH_INTERACTIVE_PROGRAMS; this stays the fallback.
__NVSH_INTERACTIVE_PROGRAMS_DEFAULT='docker htop jtop kubectl less screen ssh su tmux vim'

# --- session capture -------------------------------------------------------

__nvsh_capture_cleanup() {
    [[ -n ${NVSH_LOG:-} && -f ${NVSH_LOG} ]] && rm -f -- "${NVSH_LOG}"
    return 0
}

# Start (or join) the per-session log the Python log reader slices with the
# OSC 133 C/D markers. Inside tmux we tee the pane instead of exec'ing, and
# when script(1) is missing we simply continue without capture.
__nvsh_capture_start() {
    [[ ${NVSH_CAPTURE:-1} == 0 ]] && return 0
    [[ -n ${NVSH_WRAPPED:-} ]] && return 0
    [[ -n ${NVSH_LOG:-} ]] && return 0

    local __nvsh_dir=${XDG_RUNTIME_DIR:-/tmp}/nvsh
    (umask 077 && mkdir -p "${__nvsh_dir}") || return 0
    local __nvsh_log=${__nvsh_dir}/$$.log
    (umask 077 && : >"${__nvsh_log}") || return 0
    export NVSH_LOG=${__nvsh_log}
    trap '__nvsh_capture_cleanup' EXIT

    if [[ -n ${TMUX:-} ]]; then
        command -v tmux >/dev/null 2>&1 || return 0
        tmux pipe-pane -o "cat >> '${NVSH_LOG}'" >/dev/null 2>&1
        return 0
    fi

    command -v script >/dev/null 2>&1 || return 0
    export NVSH_WRAPPED=1
    exec script -qfc "${BASH}" "${NVSH_LOG}"
    return 0
}

# --- OSC 133 composition ---------------------------------------------------

# Emit our own "command start" marker on PS0 when no terminal integration
# already provides one. No DEBUG trap: PS0 is expanded after the line is read
# and before it runs, which is exactly the preexec point we need.
__nvsh_osc133_init() {
    [[ ${PS0:-} == *'133;C'* ]] || PS0=${PS0:-}'\e]133;C\a'
    __NVSH_OSC133_OWNED=1
    return 0
}

__nvsh_ghostty_init() {
    if [[ ${TERM_PROGRAM:-} == ghostty ]] && ! declare -F __ghostty_hook >/dev/null 2>&1; then
        local __nvsh_res=${GHOSTTY_RESOURCES_DIR:-}
        if [[ -n ${__nvsh_res} && -r ${__nvsh_res}/shell-integration/bash/ghostty.bash ]]; then
            builtin source "${__nvsh_res}/shell-integration/bash/ghostty.bash" 2>/dev/null
        fi
    fi
    declare -F __ghostty_hook >/dev/null 2>&1 && return 0
    __nvsh_osc133_init
    return 0
}

# --- the hook itself -------------------------------------------------------

__nvsh_hook() {
    # MUST be the first statement: any other command would overwrite both $?
    # and PIPESTATUS. One `local` command, so both expansions still see the
    # user's command's values.
    local __nvsh_status=$? __nvsh_pipe=("${PIPESTATUS[@]}")

    # bash-preexec (loaded by Ghostty's own integration on bash < 5.3, and by
    # kiro-cli / fig / amazon-q on any bash) rewrites PROMPT_COMMAND on its first prompt so that its
    # `__bp_precmd_invoke_cmd` runs first and every element that was there
    # before is folded in behind it as one newline-joined string. It restores
    # `$?` for each folded command via `__bp_set_ret_value`, but that `return`
    # leaves PIPESTATUS with exactly one element; the real per-stage statuses
    # survive only in its own copy, the global BP_PIPESTATUS. Take the copy.
    #
    # Staleness guard: BP_PIPESTATUS is refreshed by the first statement of
    # `__bp_precmd_invoke_cmd`, so it is this prompt's value only if that
    # function has already run in this prompt. `${PROMPT_COMMAND[0]}` starting
    # with it proves exactly that (it is where bash-preexec installs itself,
    # and it is also where we would have to be folded in to have lost
    # PIPESTATUS in the first place). If nvsh ever ran first instead, the
    # prefix test fails and we keep our own capture, which is then correct.
    if [[ -n ${BP_PIPESTATUS+x} ]] && ((${#__nvsh_pipe[@]} == 1)) &&
        [[ ${PROMPT_COMMAND[0]-} == __bp_precmd_invoke_cmd* ]] &&
        declare -F __bp_precmd_invoke_cmd >/dev/null 2>&1; then
        __nvsh_pipe=("${BP_PIPESTATUS[@]}")
    fi

    local __nvsh_ps="${__nvsh_pipe[*]}"

    if [[ -n ${NVSH_HOOK_DEBUG_FILE:-} ]]; then
        printf '%s\t%s\n' "${__nvsh_status}" "${__nvsh_ps}" \
            >>"${NVSH_HOOK_DEBUG_FILE}" 2>/dev/null
    fi

    [[ -n ${__NVSH_OSC133_OWNED:-} ]] && printf '\033]133;D;%s\a' "${__nvsh_status}"

    # nvsh's own hidden slash dispatch is never an operator command failing.
    # `__nvsh_enter` / `__nvsh_ctrl_g` (readline.bash) set this flag on the
    # line they rewrite to ` nvsh slash ...`, immediately before accept-line,
    # so the very next prompt is that dispatch's. Consume it (one-shot: a
    # plain assignment, no fork) and stop, or a `/doctor` that reports an
    # unhealthy check would make nvsh answer its own diagnostic with a full
    # agent turn.
    if [[ -n ${__NVSH_SLASH_DISPATCH:-} ]]; then
        __NVSH_SLASH_DISPATCH=
        return 0
    fi

    # Cheap bash-side pre-filter. Everything below the exit-code test runs
    # only on a failure, so a successful command pays one function call.
    case ${__nvsh_status} in
    0 | 130 | 141) return 0 ;;
    esac
    [[ ${NVSH_AUTO:-1} == 0 ]] && return 0

    # "Not inside a sourced script": at a real prompt the hook is called by
    # bash itself, so FUNCNAME holds only this function, BASH_SOURCE holds
    # only this file (its definition site), and BASH_SUBSHELL is 0. A call
    # from a sourced script or a subshell fails at least one of the three.
    ((${#FUNCNAME[@]} <= 1)) || return 0
    ((${#BASH_SOURCE[@]} <= 1)) || return 0
    ((BASH_SUBSHELL == 0)) || return 0

    # Re-drawing the prompt must not re-fire for the same command.
    local __nvsh_hc=${HISTCMD:-0}
    [[ ${__nvsh_hc} == "${__NVSH_LAST_HISTCMD:-}" ]] && return 0
    __NVSH_LAST_HISTCMD=${__nvsh_hc}

    local __nvsh_line
    __nvsh_line=$(HISTTIMEFORMAT= builtin history 1 2>/dev/null)
    [[ ${__nvsh_line} =~ ^[[:space:]]*[0-9]+[[:space:]]+(.*)$ ]] &&
        __nvsh_line=${BASH_REMATCH[1]}
    [[ -n ${__nvsh_line} ]] || return 0

    local __nvsh_first=${__nvsh_line%%[[:space:]]*}
    __nvsh_first=${__nvsh_first##*/}
    local __nvsh_progs=${NVSH_INTERACTIVE_PROGRAMS:-${__NVSH_INTERACTIVE_PROGRAMS_DEFAULT}}
    [[ " ${__nvsh_progs} " == *" ${__nvsh_first} "* ]] && return 0

    local __nvsh_bin=${NVSH_BIN:-nvsh}
    command -v "${__nvsh_bin}" >/dev/null 2>&1 || return 0
    "${__nvsh_bin}" hook \
        --exit "${__nvsh_status}" \
        --pipestatus "${__nvsh_ps}" \
        --line "${__nvsh_line}" \
        --cwd "${PWD}" \
        --log "${NVSH_LOG:-}" || return 0
    return 0
}

# --- PROMPT_COMMAND install / unload --------------------------------------

# Drop every standalone `__nvsh_hook` command from one PROMPT_COMMAND element
# and leave the remainder in __NVSH_STRIPPED (empty when nothing is left).
# A single element can hold several commands: bash-preexec's `__bp_install`
# folds the array it found into one newline-joined string, so a plain
# `== __nvsh_hook` comparison would miss our own entry and duplicate it. Only
# newlines (and a trailing `;`) are treated as separators - that is how bash
# joins them - so nothing inside a quoted argument is ever touched. Runs at
# install/unload time only, never on the prompt path.
__NVSH_STRIPPED=''
__nvsh_strip_hook() {
    local __nvsh_line __nvsh_t
    __NVSH_STRIPPED=''
    while IFS= read -r __nvsh_line || [[ -n ${__nvsh_line} ]]; do
        __nvsh_t=${__nvsh_line#"${__nvsh_line%%[![:space:]]*}"}
        __nvsh_t=${__nvsh_t%"${__nvsh_t##*[![:space:]]}"}
        __nvsh_t=${__nvsh_t%;}
        [[ -z ${__nvsh_t} || ${__nvsh_t} == __nvsh_hook ]] && continue
        __NVSH_STRIPPED+=${__NVSH_STRIPPED:+$'\n'}${__nvsh_line}
    done <<<"$1"
    return 0
}

# Prepend __nvsh_hook, keeping every existing element (array or string form)
# in order behind it.
__nvsh_prompt_command_install() {
    local -a __nvsh_rest=()
    local __nvsh_e
    if [[ -n ${PROMPT_COMMAND+x} ]]; then
        if [[ ${PROMPT_COMMAND@a} == *a* ]]; then
            for __nvsh_e in "${PROMPT_COMMAND[@]}"; do
                __nvsh_strip_hook "${__nvsh_e}"
                [[ -z ${__NVSH_STRIPPED} ]] && continue
                __nvsh_rest+=("${__NVSH_STRIPPED}")
            done
        elif [[ -n ${PROMPT_COMMAND} ]]; then
            __nvsh_strip_hook "${PROMPT_COMMAND}"
            [[ -n ${__NVSH_STRIPPED} ]] && __nvsh_rest+=("${__NVSH_STRIPPED}")
        fi
    fi
    PROMPT_COMMAND=(__nvsh_hook "${__nvsh_rest[@]}")
    return 0
}

__nvsh_hook_install() {
    __nvsh_ghostty_init
    __nvsh_prompt_command_install
    __NVSH_HOOK_LOADED=1
    return 0
}

# `nvsh off`: drop our element, leave every other integration's alone.
__nvsh_hook_unload() {
    local -a __nvsh_rest=()
    local __nvsh_e
    if [[ -n ${PROMPT_COMMAND+x} && ${PROMPT_COMMAND@a} == *a* ]]; then
        for __nvsh_e in "${PROMPT_COMMAND[@]}"; do
            __nvsh_strip_hook "${__nvsh_e}"
            [[ -z ${__NVSH_STRIPPED} ]] && continue
            __nvsh_rest+=("${__NVSH_STRIPPED}")
        done
        PROMPT_COMMAND=("${__nvsh_rest[@]}")
    fi
    unset __NVSH_HOOK_LOADED __NVSH_OSC133_OWNED __NVSH_LAST_HISTCMD __NVSH_SLASH_DISPATCH
    return 0
}

# --- wire it up ------------------------------------------------------------

__nvsh_capture_start
__nvsh_hook_install
