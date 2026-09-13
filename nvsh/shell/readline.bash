# nvsh readline layer -- sourced into an interactive bash (see
# docs/shell-integration.md for the bash-side contract this file implements).
#
# It binds three things, in the emacs, vi-insert and vi-command keymaps:
#
#   Enter (C-m)  a macro that runs a bind -x function and then accept-line,
#                so a registered slash line is rewritten to a hidden
#                `nvsh slash <line>` call while the original stays in history.
#   Tab          initial-word completion (complete -I) merging the slash
#                palette with real path completion, plus per-command argument
#                completion, both fed only by `nvsh complete --json`.
#   C-g          the on-demand agent panel hook, restoring the typed line.
#
# Rules this file keeps: no `set -e`, every function ends `|| return 0`, a
# failure degrades to plain bash and never locks the terminal, and the
# command list is never duplicated here.

if [[ -n ${__NVSH_READLINE_LOADED:-} ]]; then
    return 0
fi
if [[ ${NVSH_DISABLE:-} == 1 ]]; then
    return 0
fi
case $- in
    *i*) ;;
    *) return 0 ;;
esac

__NVSH_READLINE_LOADED=1
__NVSH_KEYMAPS=(emacs vi-insert vi-command)
# An otherwise unused sequence: the Enter macro runs this, then accept-line.
__NVSH_DISPATCH_SEQ='\C-x\C-n'
__NVSH_ITEMS=()

# The hidden dispatch line is prefixed with one space, which only stays out
# of history when HISTCONTROL asks for it. Append ignorespace if neither it
# nor ignoreboth is already there; never clobber the operator's value.
__nvsh_histcontrol() {
    case ":${HISTCONTROL:-}:" in
        *:ignorespace:* | *:ignoreboth:*) ;;
        *) HISTCONTROL="${HISTCONTROL:+${HISTCONTROL}:}ignorespace" ;;
    esac
    return 0
}

# Fill __NVSH_ITEMS from `nvsh complete --json [args...]`. The payload is
# {"items":[{"value":"...","description":"..."}]}; values are plain words, so
# a bash regex scan is enough and costs no extra fork.
__nvsh_items() {
    __NVSH_ITEMS=()
    local json rest
    json=$("${NVSH_BIN:-nvsh}" complete --json "$@" 2>/dev/null) || return 0
    rest=$json
    while [[ $rest =~ \"value\"[[:space:]]*:[[:space:]]*\"([^\"]*)\" ]]; do
        __NVSH_ITEMS+=("${BASH_REMATCH[1]}")
        rest=${rest#*"\"${BASH_REMATCH[1]}\""}
    done
    return 0
}

# --- Enter -------------------------------------------------------------

__nvsh_enter() {
    local line=${READLINE_LINE:-} word item
    # Ordinary commands leave here before anything forks.
    [[ $line == /* ]] || return 0
    word=${line%%[[:space:]]*}
    # A path (/tmp/x) or an option-looking word is not a slash command.
    [[ $word =~ ^/[A-Za-z][A-Za-z0-9_-]*$ ]] || return 0
    __nvsh_items || return 0
    for item in "${__NVSH_ITEMS[@]}"; do
        if [[ $item == "$word" ]]; then
            history -s -- "$line"
            READLINE_LINE=" ${NVSH_BIN:-nvsh} slash ${line@Q}"
            READLINE_POINT=${#READLINE_LINE}
            __NVSH_ITEMS=()
            return 0
        fi
    done
    __NVSH_ITEMS=()
    return 0
}

# --- Ctrl+G ------------------------------------------------------------

__nvsh_ctrl_g() {
    local line=${READLINE_LINE:-} point=${READLINE_POINT:-0}
    printf '\n⚡ nvsh (Ctrl+G): asking the agent\n'
    NVSH_DRAFT=$line "${NVSH_BIN:-nvsh}" slash "/ask" >/dev/null 2>&1
    READLINE_LINE=$line
    READLINE_POINT=$point
    return 0
}

# --- completion --------------------------------------------------------

# First word: merge the slash palette with normal path completion, and fall
# through to bash's default completion for everything else.
__nvsh_complete_initial() {
    local cur=${COMP_WORDS[COMP_CWORD]} item
    COMPREPLY=()
    if [[ $cur == /* && $cur != */*/* ]]; then
        __nvsh_items || return 0
        for item in "${__NVSH_ITEMS[@]}"; do
            [[ $item == "$cur"* ]] && COMPREPLY+=("$item")
        done
        __NVSH_ITEMS=()
        while IFS= read -r item; do
            [[ -n $item ]] && COMPREPLY+=("$item")
        done < <(compgen -f -- "$cur" 2>/dev/null)
        compopt -o filenames 2>/dev/null
        return 0
    fi
    compopt -o bashdefault -o default 2>/dev/null
    return 0
}

# Arguments of one slash command, straight from `nvsh complete --json --`.
__nvsh_complete_args() {
    local cur=${COMP_WORDS[COMP_CWORD]} cmd=${COMP_WORDS[0]} item
    COMPREPLY=()
    __nvsh_items -- "$cmd" "$cur" || return 0
    for item in "${__NVSH_ITEMS[@]}"; do
        [[ $item == "$cur"* ]] && COMPREPLY+=("$item")
    done
    __NVSH_ITEMS=()
    return 0
}

# --- bind / unbind -----------------------------------------------------

__nvsh_readline_bind() {
    local keymap item
    __nvsh_histcontrol
    for keymap in "${__NVSH_KEYMAPS[@]}"; do
        bind -m "$keymap" -x "\"${__NVSH_DISPATCH_SEQ}\": __nvsh_enter" 2>/dev/null
        bind -m "$keymap" "\"\\C-m\": \"${__NVSH_DISPATCH_SEQ}\\C-j\"" 2>/dev/null
        bind -m "$keymap" -x '"\C-g": __nvsh_ctrl_g' 2>/dev/null
    done
    complete -I -F __nvsh_complete_initial 2>/dev/null
    # One `nvsh complete` call per shell, at source time, to register the
    # per-command argument completers: readline consults `complete -F <cmd>`
    # for a non-initial word and never calls back into the -I function.
    __nvsh_items || return 0
    for item in "${__NVSH_ITEMS[@]}"; do
        complete -F __nvsh_complete_args "$item" 2>/dev/null
    done
    __NVSH_ITEMS=()
    return 0
}

__nvsh_readline_unbind() {
    local keymap item
    __nvsh_items || __NVSH_ITEMS=()
    for keymap in "${__NVSH_KEYMAPS[@]}"; do
        bind -m "$keymap" '"\C-m": accept-line' 2>/dev/null
        bind -m "$keymap" -r "${__NVSH_DISPATCH_SEQ}" 2>/dev/null
        bind -m "$keymap" -r '\C-g' 2>/dev/null
    done
    bind -m emacs '"\C-g": abort' 2>/dev/null
    bind -m vi-insert '"\C-g": abort' 2>/dev/null
    bind -m vi-command '"\C-g": abort' 2>/dev/null
    complete -r -I 2>/dev/null
    for item in "${__NVSH_ITEMS[@]}"; do
        complete -r "$item" 2>/dev/null
    done
    __NVSH_ITEMS=()
    return 0
}

__nvsh_readline_bind
