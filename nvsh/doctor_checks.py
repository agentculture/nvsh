"""``nvsh doctor``'s extension checks: platform, agent backend, in-shell hook health.

Task t17. Each ``check_*`` function is a pure function (plus small
injectables — ``which``, ``run``, a fake home directory) that returns exactly
one rubric-shaped dict: ``{id, passed, severity, message, remediation}``,
matching the shape ``nvsh.cli._commands.doctor._diagnose`` already returns
for the mesh-identity checks. :func:`collect_checks` runs all of them and is
the one function ``doctor.py`` calls, so its own diff stays small.

**In-shell state.** Three of these checks (``hook_first_in_prompt_command``,
``bindings_present``, and — via ``NVSH_HOOK_VERSION``/``NVSH_LOG`` — indirectly
``hook_sourced``/``capture_active``) need state only a hooked bash session can
produce: the live ``PROMPT_COMMAND`` array and ``bind -p`` output. Bash passes
that state to ``nvsh doctor`` as arguments (see ``docs/shell-integration.md``'s
"/doctor" section for the exact invocation); this module never shells out to
read it itself. When that state is simply absent — nvsh is running from a
plain terminal, a script, or a wheel install with no hook sourced — these
checks report ``passed=False, severity="info"`` with the remediation "run
/doctor from a hooked shell". Info severity is deliberate: it is not a
failure to run ``nvsh doctor`` outside a hooked shell, so
:func:`nvsh.cli._commands.doctor._diagnose`'s ``healthy`` computation ignores
info-severity checks entirely, the same way it already treats
``harness_prompts`` and ``source_checkout`` as non-blocking.
"""

from __future__ import annotations

import json
import os
import re
import socket as socket_lib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, NamedTuple

from nvsh import capture as capture_mod
from nvsh import daemon as daemon_mod
from nvsh.agent import registry as agent_registry
from nvsh.config import (
    DEFAULT_KEY_FILE_DISPLAY,
    NO_BEARER_NOTE,
    BearerResolution,
    Config,
    resolve_bearer,
)
from nvsh.platform import Platform
from nvsh.platform._subprocess import Runner, Which, default_run, default_which

RUN_TO_HOOK_REMEDIATION = "run /doctor from a hooked shell"

#: Marker line separating ``bind -p`` from the ``bind -s``/``bind -X`` dumps
#: in the payload a hooked bash exports (see ``check_bindings_present``).
BIND_SECTION_MARKER = "# nvsh: bind -s/-X follow"

#: Source label for a base_url that came from pi's own models.json.
PI_MODELS_JSON_SOURCE = "pi models.json"


def _check(check_id: str, passed: bool, severity: str, message: str, remediation: str) -> dict:
    return {
        "id": check_id,
        "passed": passed,
        "severity": severity,
        "message": message,
        "remediation": remediation,
    }


# ---------------------------------------------------------------------------
# platform_detected
# ---------------------------------------------------------------------------


def check_platform_detected(platform: Platform) -> dict:
    if platform.kind != "generic":
        return _check(
            "platform_detected",
            True,
            "info",
            f"detected platform: {platform.kind}",
            "",
        )
    sources = ", ".join(dict.fromkeys(value.source for value in platform.values))
    sources = sources or "no sources checked"
    return _check(
        "platform_detected",
        False,
        "warning",
        f"platform not detected (generic); sources checked: {sources}",
        "run on a supported NVIDIA platform (Jetson/DGX Spark/RTX Spark); "
        "see docs/platforms.md for the full source list",
    )


# ---------------------------------------------------------------------------
# agent_configured
# ---------------------------------------------------------------------------


def check_agent_configured(config: Config | None, config_error: str | None) -> dict:
    if config_error is not None:
        return _check(
            "agent_configured",
            False,
            "error",
            f"config.toml failed to load: {config_error}",
            "fix or remove $XDG_CONFIG_HOME/nvsh/config.toml",
        )
    assert config is not None  # config_error is None -> a Config was loaded
    provider = config.agent_provider
    if provider in agent_registry.ADAPTERS:
        return _check(
            "agent_configured",
            True,
            "info",
            f"agent provider configured: {provider}",
            "",
        )
    return _check(
        "agent_configured",
        False,
        "error",
        f"configured provider '{provider}' is not a known adapter",
        f"set [agent] provider to one of: {', '.join(sorted(agent_registry.ADAPTERS))}",
    )


# ---------------------------------------------------------------------------
# agent_reachable
# ---------------------------------------------------------------------------


#: Literal-string form used in every user-facing message/remediation --
#: never a real ``~/``-expanded path (the steward portability check flags
#: ``~/.`` paths in committed text; see docs/... and memory
#: "steward-portability-home-paths").
_PI_MODELS_JSON = "$HOME/.pi/agent/models.json"

#: Matches an env-ref apiKey value: ``$VAR`` or ``${VAR}``. ``re.ASCII``
#: keeps ``\w`` to ``[A-Za-z0-9_]``, the shell's own variable-name alphabet.
_ENV_REF_RE = re.compile(r"^\$\{?([A-Za-z_]\w*)\}?$", re.ASCII)


def _provider_from_mapping(mapping: object, provider_name: object) -> dict | None:
    """``mapping[provider_name]``, when both are the right shape."""
    if not isinstance(mapping, dict) or not provider_name:
        return None
    entry = mapping.get(provider_name)
    return entry if isinstance(entry, dict) else None


def _provider_from_list(items: object, provider_name: object) -> dict | None:
    """The first entry in *items* whose ``id``/``name`` is *provider_name*."""
    if not isinstance(items, list):
        return None
    for entry in items:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") == provider_name or entry.get("name") == provider_name:
            return entry
    return None


def _find_provider_entry(data: object, provider_name: object) -> dict | None:
    """Best-effort lookup of one provider's table in pi's models.json.

    pi's own schema for ``$HOME/.pi/agent/models.json`` is not part of
    nvsh's contract (nvsh only ever reads it, read-only, and never prints a
    key), so this accepts the handful of shapes a provider table plausibly
    takes: a top-level or ``providers`` mapping keyed by provider name, or a
    list of provider objects carrying ``id``/``name``.
    """
    if isinstance(data, dict):
        found = _provider_from_mapping(data, provider_name)
        if found is not None:
            return found
        providers = data.get("providers")
        found = _provider_from_mapping(providers, provider_name) or _provider_from_list(
            providers, provider_name
        )
        if found is not None:
            return found
    return _provider_from_list(data, provider_name)


def _load_pi_models(home: Path) -> object | None:
    """Read and parse ``$HOME/.pi/agent/models.json``, or ``None`` on any miss."""
    models_path = home / ".pi" / "agent" / "models.json"
    try:
        text = models_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _resolve_pi_api_key(raw: object) -> tuple[str | None, str | None]:
    """Resolve a models.json ``apiKey`` field to ``(bearer, source_label)``.

    Supports a literal key or an env-ref (``$VAR`` / ``${VAR}``, resolved
    from ``os.environ``). ``source_label`` describes where the bearer came
    from for a check message -- ``"models.json"`` or ``"$VAR"`` -- and never
    carries the key value or any prefix of it. Returns ``(None, None)`` when
    there is nothing usable (missing, empty, or an env-ref to an unset var).
    """
    if not raw:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    match = _ENV_REF_RE.match(text)
    if match:
        var = match.group(1)
        value = os.environ.get(var)
        return (value, f"${var}") if value else (None, None)
    return text, PI_MODELS_JSON_SOURCE


def _pi_endpoint_info(
    config: Config, home: Path
) -> tuple[str | None, str, str | None, str | None, str | None]:
    """Return ``(base_url, base_url_source, bearer, bearer_source, provider_name)``.

    ``base_url`` prefers config's own ``[agents.pi] base_url`` over
    models.json's, and ``base_url_source`` says which of the two it came from
    (d4b: that label, never the URL, is what reaches a check message).
    ``bearer``/``bearer_source`` come only from models.json's ``apiKey`` for
    the configured provider (pi's config.toml has no ``api_key_env`` -- its
    bearer always lives in models.json).
    """
    pi_settings = config.agents.get("pi", {})
    provider_name = pi_settings.get("provider")
    base_url_raw = pi_settings.get("base_url")
    base_url = str(base_url_raw) if base_url_raw else None
    base_url_source = "[agents.pi]" if base_url else PI_MODELS_JSON_SOURCE

    entry = _find_provider_entry(_load_pi_models(home), provider_name)

    if not base_url and entry:
        raw_base_url = entry.get("baseUrl")
        if raw_base_url:
            base_url = str(raw_base_url)

    bearer, bearer_source = (None, None)
    if entry:
        bearer, bearer_source = _resolve_pi_api_key(entry.get("apiKey"))

    return base_url, base_url_source, bearer, bearer_source, provider_name


def _probe_endpoint(
    base_url: str,
    base_url_source: str,
    bearer: str | None,
    bearer_note: str | None,
    timeout: float,
    *,
    remediation_401: str,
) -> dict:
    """Probe ``base_url`` and report the result **without ever printing it**.

    Deviation d4b: the endpoint's host is machine-identifying and has no
    place in a doctor line an operator may paste anywhere. Every message and
    remediation below names only *where* the URL was configured
    (``base_url_source``) and where the bearer came from (``bearer_note``: an
    env var name, ``api_key_file``, the default key file, or "no bearer
    configured") -- never the URL, host or port themselves, and never the
    key or the directory a key file lives in.
    """
    source_note = f"base_url from {base_url_source}"
    if bearer_note:
        source_note += f", {bearer_note}"

    url = base_url.rstrip("/") + "/models"
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in ("http", "https"):
        return _check(
            "agent_reachable",
            False,
            "error",
            f"unsupported URL scheme in configured endpoint: {scheme!r}",
            "fix base_url in config.toml (must be http:// or https://)",
        )

    headers: dict[str, str] = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        # scheme validated above (http/https only); base_url is
        # config-supplied, not attacker/network-controlled input.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except OSError:  # URLError and TimeoutError both derive from OSError
        return _check(
            "agent_reachable",
            False,
            "error",
            f"endpoint unreachable (endpoint-unreachable; {source_note})",
            "check that the configured endpoint is running and reachable from this "
            f"machine ({source_note})",
        )

    if status == 401:
        return _check(
            "agent_reachable",
            False,
            "error",
            f"endpoint returned 401 Unauthorized (endpoint-401; {source_note})",
            remediation_401,
        )
    if status == 200:
        return _check("agent_reachable", True, "info", f"endpoint reachable ({source_note})", "")
    return _check(
        "agent_reachable",
        False,
        "warning",
        f"endpoint responded with status {status} ({source_note})",
        "",
    )


def _openai_compat_401_remediation(outcome: BearerResolution) -> str:
    """What to do about a 401, naming a key *location* and never a key.

    ``DEFAULT_KEY_FILE_DISPLAY`` is the placeholder spelling of the default
    key file, not a resolved path -- the same rule the endpoint URL follows
    (d4): a doctor line an operator pastes anywhere carries no machine- or
    user-identifying path.
    """
    if outcome.source and outcome.source.startswith("$"):
        return f"set {outcome.source[1:]} to a valid API key"
    if outcome.source or outcome.diagnostic:
        return "put a valid API key in the configured key file (mode 0600)"
    return (
        f"put the gateway's key in {DEFAULT_KEY_FILE_DISPLAY} (mode 0600), "
        "or point api_key_file/api_key_env at one in config.toml"
    )


class _ProbeInputs(NamedTuple):
    """What :func:`_probe_endpoint` needs, or the check to report instead."""

    base_url: str | None = None
    base_url_source: str = ""
    bearer: str | None = None
    bearer_note: str | None = None
    remediation_401: str = ""
    refusal: dict | None = None


def _pi_probe_inputs(config: Config, home: Path, which: Which) -> _ProbeInputs:
    """Probe inputs for the ``pi`` provider, or a refusal when pi is missing."""
    if which("pi") is None:
        return _ProbeInputs(
            refusal=_check(
                "agent_reachable",
                False,
                "error",
                "configured provider is pi, but 'pi' is not on PATH (pi-missing)",
                "nvsh agent install pi, or nvsh agent use openai-compat",
            )
        )
    base_url, base_url_source, bearer, bearer_source, provider_name = _pi_endpoint_info(
        config, home
    )
    provider_label = provider_name or "the configured provider"
    verb = "update" if bearer else "add"
    return _ProbeInputs(
        base_url=base_url,
        base_url_source=base_url_source,
        bearer=bearer,
        bearer_note=f"bearer from {bearer_source}" if bearer_source else None,
        remediation_401=f"{verb} apiKey for provider {provider_label} in {_PI_MODELS_JSON}",
    )


def _openai_compat_probe_inputs(config: Config) -> _ProbeInputs:
    """Probe inputs for the ``openai-compat`` provider."""
    settings = config.agents.get("openai-compat", {})
    base_url_raw = settings.get("base_url")
    outcome = resolve_bearer(settings)
    bearer_note = f"bearer from {outcome.source}" if outcome.source else NO_BEARER_NOTE
    if outcome.diagnostic:
        bearer_note = outcome.diagnostic
    return _ProbeInputs(
        base_url=str(base_url_raw) if base_url_raw else None,
        base_url_source="[agents.openai-compat]",
        bearer=outcome.bearer,
        bearer_note=bearer_note,
        remediation_401=_openai_compat_401_remediation(outcome),
    )


def check_agent_reachable(
    config: Config,
    which: Which = default_which,
    home: Path | None = None,
    timeout: float = 3.0,
) -> dict:
    home = home if home is not None else Path.home()
    provider = config.agent_provider

    if provider == "pi":
        inputs = _pi_probe_inputs(config, home, which)
    elif provider == "openai-compat":
        inputs = _openai_compat_probe_inputs(config)
    else:
        return _check(
            "agent_reachable",
            False,
            "warning",
            f"configured provider '{provider}' has no reachability probe",
            "",
        )

    if inputs.refusal is not None:
        return inputs.refusal

    if not inputs.base_url:
        return _check(
            "agent_reachable",
            False,
            "warning",
            f"endpoint unknown for provider '{provider}'; nothing to probe",
            f"set base_url in config.toml, or {_PI_MODELS_JSON} for pi",
        )

    return _probe_endpoint(
        inputs.base_url,
        inputs.base_url_source,
        inputs.bearer,
        inputs.bearer_note,
        timeout,
        remediation_401=inputs.remediation_401,
    )


# ---------------------------------------------------------------------------
# hook_sourced
# ---------------------------------------------------------------------------


def check_hook_sourced(env: Mapping[str, str], current_version: str) -> dict:
    hook_version = env.get("NVSH_HOOK_VERSION")
    if not hook_version:
        return _check(
            "hook_sourced",
            False,
            "info",
            "NVSH_HOOK_VERSION not set; not running in a hooked shell",
            RUN_TO_HOOK_REMEDIATION,
        )
    if hook_version == current_version:
        return _check(
            "hook_sourced",
            True,
            "info",
            f"hook sourced, version {hook_version} matches installed nvsh",
            "",
        )
    return _check(
        "hook_sourced",
        False,
        "warning",
        f"hook version {hook_version} does not match installed nvsh {current_version}",
        "run nvsh setup",
    )


# ---------------------------------------------------------------------------
# hook_first_in_prompt_command
# ---------------------------------------------------------------------------

_DECLARE_ARRAY_RE = re.compile(r"declare\s+-a\s+PROMPT_COMMAND=")
#: ``declare -p`` prints an element either double-quoted or, when it holds a
#: newline (which is how bash-preexec folds several commands into one), in
#: ANSI-C ``$'...'`` form.
_ARRAY_ELEM_RE = re.compile(r"\[(\d+)\]=(?:\"((?:[^\"\\]|\\.)*)\"|\$'((?:[^'\\]|\\.)*)')")
_DECLARE_STRING_RE = re.compile(r'declare\s+--\s*PROMPT_COMMAND="((?:[^"\\]|\\.)*)"')

#: The function bash-preexec installs as the first PROMPT_COMMAND command.
BASH_PREEXEC_ENTRY = "__bp_precmd_invoke_cmd"

#: The exact fix command reported when ``__nvsh_hook`` is not first.
HOOK_FIX_COMMAND = 'PROMPT_COMMAND=(__nvsh_hook "${PROMPT_COMMAND[@]/__nvsh_hook}")  # or: nvsh on'


def _unescape_declare(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


def _unescape_ansi_c(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace("\\\\", "\\")


def _parse_prompt_command(text: str) -> list[str] | None:
    if _DECLARE_ARRAY_RE.search(text) or "PROMPT_COMMAND=(" in text:
        pairs = [
            (
                int(match.group(1)),
                (
                    _unescape_ansi_c(match.group(3))
                    if match.group(3) is not None
                    else _unescape_declare(match.group(2))
                ),
            )
            for match in _ARRAY_ELEM_RE.finditer(text)
        ]
        if pairs:
            pairs.sort(key=lambda pair: pair[0])
            return [value for _, value in pairs]
        return None
    match = _DECLARE_STRING_RE.search(text)
    if match:
        return [_unescape_declare(match.group(1))]
    return None


def check_hook_first_in_prompt_command(prompt_command_text: str | None) -> dict:
    if not prompt_command_text:
        return _check(
            "hook_first_in_prompt_command",
            False,
            "info",
            "no --prompt-command state passed; not running in a hooked shell",
            RUN_TO_HOOK_REMEDIATION,
        )
    elements = _parse_prompt_command(prompt_command_text)
    if not elements:
        return _check(
            "hook_first_in_prompt_command",
            False,
            "warning",
            "could not parse the --prompt-command state",
            RUN_TO_HOOK_REMEDIATION,
        )
    commands = [[part.strip().rstrip(";") for part in element.split("\n")] for element in elements]
    occurrences = sum(part == "__nvsh_hook" for parts in commands for part in parts)
    if occurrences > 1:
        return _check(
            "hook_first_in_prompt_command",
            False,
            "error",
            f"__nvsh_hook appears {occurrences} times in PROMPT_COMMAND, "
            f"it must appear once (order: {elements})",
            HOOK_FIX_COMMAND,
        )
    if occurrences == 1 and commands[0][0] == "__nvsh_hook":
        return _check(
            "hook_first_in_prompt_command",
            True,
            "info",
            "__nvsh_hook is first in PROMPT_COMMAND",
            "",
        )
    # bash-preexec (Ghostty on bash < 5.3, fig/amazon-q) installs its own entry
    # first by design and folds ours in behind it; it restores `$?` for us and
    # the hook reads per-stage statuses from BP_PIPESTATUS, so this layout is
    # healthy. See docs/architecture.md, "PROMPT_COMMAND ordering".
    if occurrences == 1 and commands[0][:2] == [BASH_PREEXEC_ENTRY, "__nvsh_hook"]:
        return _check(
            "hook_first_in_prompt_command",
            True,
            "info",
            f"__nvsh_hook runs directly after {BASH_PREEXEC_ENTRY} in PROMPT_COMMAND "
            "(bash-preexec layout; $? is restored and PIPESTATUS is read from "
            "BP_PIPESTATUS)",
            "",
        )
    return _check(
        "hook_first_in_prompt_command",
        False,
        "error",
        f"__nvsh_hook is not first in PROMPT_COMMAND (order: {elements})",
        HOOK_FIX_COMMAND,
    )


# ---------------------------------------------------------------------------
# bindings_present
# ---------------------------------------------------------------------------

#: The three bindings ``readline.bash`` installs, each as
#: ``(label, pattern)``. The patterns allow for **both** dump formats, which
#: is the whole of deviation d4a: ``bind -p`` lists neither ``bind -x``
#: functions nor macros, so the ``\C-x\C-n``/``\C-g`` handlers show up only
#: in ``bind -X`` (which *quotes* the shell command --
#: ``"\C-x\C-n": "__nvsh_enter"``) and the Enter macro only in ``bind -s``.
#: A payload that merges the three dumps therefore carries the quoted form,
#: while hand-written/older payloads carry the bare one; both must parse.
_REQUIRED_BINDINGS = (
    (r'"\C-x\C-n": __nvsh_enter', re.compile(r'"\\C-x\\C-n"\s*:\s*"?__nvsh_enter"?')),
    (r'"\C-x\C-g": __nvsh_ctrl_g', re.compile(r'"\\C-x\\C-g"\s*:\s*"?__nvsh_ctrl_g"?')),
    (r'"\C-g": "\C-x\C-g\C-j"', re.compile(r'"\\C-g"\s*:\s*"\\C-x\\C-g\\C-j"')),
    (r'"\C-m": "\C-x\C-n\C-j"', re.compile(r'"\\C-m"\s*:\s*"\\C-x\\C-n\\C-j"')),
)

#: What a payload that only carries ``bind -p`` can honestly be reported as.
_BIND_P_ONLY_MESSAGE = (
    "cannot verify the nvsh readline bindings for {label}: this shell exported "
    "only 'bind -p', which lists neither bind -x functions nor macros"
)
_BIND_P_ONLY_REMEDIATION = (
    "run nvsh setup to refresh the hook so the shell exports bind -s and bind -X too"
)


def check_bindings_present(bind_p_text: str | None, keymap: str | None) -> dict:
    if not bind_p_text:
        return _check(
            "bindings_present",
            False,
            "info",
            "no --bind-p state passed; not running in a hooked shell",
            RUN_TO_HOOK_REMEDIATION,
        )
    label = keymap or "active keymap"
    missing = [label_ for label_, pattern in _REQUIRED_BINDINGS if not pattern.search(bind_p_text)]
    if not missing:
        return _check(
            "bindings_present",
            True,
            "info",
            f"nvsh readline bindings present ({label})",
            "",
        )
    # Nothing found at all, and no sign the payload carries the bind -s /
    # bind -X tables: the state needed to judge this simply is not here, so
    # report that honestly (info) instead of a false FAIL (d4a).
    complete_payload = BIND_SECTION_MARKER in bind_p_text or len(missing) < len(_REQUIRED_BINDINGS)
    if not complete_payload:
        return _check(
            "bindings_present",
            False,
            "info",
            _BIND_P_ONLY_MESSAGE.format(label=label),
            _BIND_P_ONLY_REMEDIATION,
        )
    return _check(
        "bindings_present",
        False,
        "error",
        f"missing nvsh readline bindings for {label}: {', '.join(missing)}",
        "run nvsh on (or re-source readline.bash)",
    )


# ---------------------------------------------------------------------------
# capture_active
# ---------------------------------------------------------------------------


def check_capture_active(
    env: Mapping[str, str], stat_fn: Callable[[str], os.stat_result] = os.stat
) -> dict:
    log = env.get("NVSH_LOG")
    if not log:
        hooked = bool(env.get("NVSH_HOOK_VERSION"))
        return _check(
            "capture_active",
            False,
            "warning" if hooked else "info",
            "NVSH_LOG not set; output capture is not active",
            RUN_TO_HOOK_REMEDIATION,
        )
    source = capture_mod.capture_source(dict(env))
    try:
        mode = stat_fn(log).st_mode & 0o777
    except OSError:
        return _check(
            "capture_active",
            False,
            "warning",
            f"NVSH_LOG={log} is set but the file is missing",
            "restart the shell so the hook re-opens the capture log",
        )
    if mode != 0o600:
        return _check(
            "capture_active",
            False,
            "warning",
            f"{source}: {log} has mode {oct(mode)}, expected 0600",
            f"chmod 600 {log}",
        )
    return _check("capture_active", True, "info", f"{source}: {log}", "")


# ---------------------------------------------------------------------------
# daemon_status
# ---------------------------------------------------------------------------


def check_daemon_status(
    env: Mapping[str, str],
    is_running: Callable[[Mapping[str, str]], bool] = daemon_mod.is_running,
    socket_path: Callable[[Mapping[str, str]], Path] = daemon_mod.socket_path,
) -> dict:
    running = is_running(env)
    path = socket_path(env)
    state = "running" if running else "not running (normal — the daemon starts on demand)"
    return _check("daemon_status", True, "info", f"daemon {state} (socket: {path})", "")


# ---------------------------------------------------------------------------
# terminfo_present
# ---------------------------------------------------------------------------

_TERMINFO_DIRS_DEFAULT = ("/usr/share/terminfo", "/lib/terminfo", "/etc/terminfo")


def _terminfo_search_dirs(env: Mapping[str, str]) -> list[str]:
    """Directories to look for a compiled terminfo entry under, in order.

    Mirrors ncurses' own precedence: ``$TERMINFO`` first, then
    ``~/.terminfo``, then ``$TERMINFO_DIRS`` (which — when explicitly set —
    REPLACES the compiled-in system list rather than adding to it, so a
    test can point it at an empty tree to simulate a host with no terminfo
    database without the real system one leaking through); only when
    ``$TERMINFO_DIRS`` is unset do the standard system locations apply.
    """
    dirs: list[str] = []
    single = env.get("TERMINFO")
    if single:
        dirs.append(single)
    home = env.get("HOME")
    if home:
        dirs.append(str(Path(home) / ".terminfo"))
    dirs_env = env.get("TERMINFO_DIRS")
    if dirs_env is not None:
        dirs.extend(part for part in dirs_env.split(":") if part)
    else:
        dirs.extend(_TERMINFO_DIRS_DEFAULT)
    return dirs


def _terminfo_file_exists(term: str, env: Mapping[str, str]) -> bool:
    if not term:
        return False
    first = term[0]
    for directory in _terminfo_search_dirs(env):
        if (Path(directory) / first / term).exists():
            return True
        if (Path(directory) / f"{ord(first):x}" / term).exists():
            return True
    return False


def check_terminfo_present(env: Mapping[str, str], run: Runner = default_run) -> dict:
    term = env.get("TERM") or "xterm"
    returncode, _stdout, _stderr = run(["infocmp", term], 2.0)
    if returncode == 0 or _terminfo_file_exists(term, env):
        return _check("terminfo_present", True, "info", f"terminfo present for TERM={term}", "")

    host = env.get("HOSTNAME") or socket_lib.gethostname()
    remediation = f"infocmp -x {term} | ssh {host} -- tic -x -"
    note = f" (run this from the machine whose terminal advertises TERM={term}, not here)"
    return _check(
        "terminfo_present",
        False,
        "warning",
        f"no terminfo entry for TERM={term}{note}",
        remediation,
    )


# ---------------------------------------------------------------------------
# collect_checks
# ---------------------------------------------------------------------------


def collect_checks(
    *,
    env: Mapping[str, str],
    current_version: str,
    config: Config | None,
    config_error: str | None,
    platform: Platform,
    which: Which = default_which,
    run: Runner = default_run,
    home: Path | None = None,
    prompt_command_text: str | None = None,
    bind_p_text: str | None = None,
    keymap: str | None = None,
    is_running: Callable[[Mapping[str, str]], bool] = daemon_mod.is_running,
    socket_path: Callable[[Mapping[str, str]], Path] = daemon_mod.socket_path,
) -> list[dict]:
    """Run every doctor extension check and return their rubric-shaped dicts.

    Called unconditionally by ``doctor.py``'s ``_diagnose()`` — including in
    the wheel-install branch where no ``culture.yaml`` exists, since these
    checks are about the shell/backend, not the mesh-identity invariants.
    """
    checks = [check_platform_detected(platform)]
    checks.append(check_agent_configured(config, config_error))
    if config is not None:
        checks.append(check_agent_reachable(config, which=which, home=home))
    else:
        checks.append(
            _check(
                "agent_reachable",
                False,
                "error",
                "agent config failed to load; skipping reachability probe",
                "fix or remove $XDG_CONFIG_HOME/nvsh/config.toml",
            )
        )
    checks.append(check_hook_sourced(env, current_version))
    checks.append(check_hook_first_in_prompt_command(prompt_command_text))
    checks.append(check_bindings_present(bind_p_text, keymap))
    checks.append(check_capture_active(env))
    checks.append(check_daemon_status(env, is_running=is_running, socket_path=socket_path))
    checks.append(check_terminfo_present(env, run=run))
    return checks
