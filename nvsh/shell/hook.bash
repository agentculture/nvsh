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
    local __nvsh_ps="${__nvsh_pipe[*]}"

    if [[ -n ${NVSH_HOOK_DEBUG_FILE:-} ]]; then
        printf '%s\t%s\n' "${__nvsh_status}" "${__nvsh_ps}" \
            >>"${NVSH_HOOK_DEBUG_FILE}" 2>/dev/null
    fi

    [[ -n ${__NVSH_OSC133_OWNED:-} ]] && printf '\033]133;D;%s\a' "${__nvsh_status}"

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

# Prepend __nvsh_hook, keeping every existing element (array or string form)
# in order behind it.
__nvsh_prompt_command_install() {
    local -a __nvsh_rest=()
    local __nvsh_e
    if [[ -n ${PROMPT_COMMAND+x} ]]; then
        if [[ ${PROMPT_COMMAND@a} == *a* ]]; then
            for __nvsh_e in "${PROMPT_COMMAND[@]}"; do
                [[ -z ${__nvsh_e} || ${__nvsh_e} == __nvsh_hook ]] && continue
                __nvsh_rest+=("${__nvsh_e}")
            done
        elif [[ -n ${PROMPT_COMMAND} ]]; then
            __nvsh_rest+=("${PROMPT_COMMAND}")
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
            [[ ${__nvsh_e} == __nvsh_hook ]] && continue
            __nvsh_rest+=("${__nvsh_e}")
        done
        PROMPT_COMMAND=("${__nvsh_rest[@]}")
    fi
    unset __NVSH_HOOK_LOADED __NVSH_OSC133_OWNED __NVSH_LAST_HISTCMD
    return 0
}

# --- wire it up ------------------------------------------------------------

__nvsh_capture_start
__nvsh_hook_install
