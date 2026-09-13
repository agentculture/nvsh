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
from typing import Callable, Mapping

from nvsh import capture as capture_mod
from nvsh import daemon as daemon_mod
from nvsh.agent import registry as agent_registry
from nvsh.config import Config
from nvsh.platform import Platform
from nvsh.platform._subprocess import Runner, Which, default_run, default_which

RUN_TO_HOOK_REMEDIATION = "run /doctor from a hooked shell"


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


def _extract_base_url(data: object, provider_name: object) -> str | None:
    """Best-effort extraction of one provider's ``baseUrl`` from pi's models.json.

    pi's own schema for ``~/.pi/agent/models.json`` is not part of nvsh's
    contract (nvsh only ever reads it, read-only, and never prints a key),
    so this accepts the handful of shapes a provider table plausibly takes:
    a top-level or ``providers`` mapping keyed by provider name, or a list of
    provider objects carrying ``id``/``name`` and ``baseUrl``.
    """

    def _from_mapping(mapping: object) -> str | None:
        if not isinstance(mapping, dict) or not provider_name:
            return None
        entry = mapping.get(provider_name)
        if isinstance(entry, dict) and entry.get("baseUrl"):
            return str(entry["baseUrl"])
        return None

    def _from_list(items: object) -> str | None:
        if not isinstance(items, list):
            return None
        for entry in items:
            if not isinstance(entry, dict):
                continue
            if entry.get("id") == provider_name or entry.get("name") == provider_name:
                if entry.get("baseUrl"):
                    return str(entry["baseUrl"])
        return None

    if isinstance(data, dict):
        found = _from_mapping(data)
        if found:
            return found
        providers = data.get("providers")
        found = _from_mapping(providers) or _from_list(providers)
        if found:
            return found
    found = _from_list(data)
    if found:
        return found
    return None


def _pi_base_url(config: Config, home: Path) -> str | None:
    pi_settings = config.agents.get("pi", {})
    base_url = pi_settings.get("base_url")
    if base_url:
        return str(base_url)

    provider_name = pi_settings.get("provider")
    models_path = home / ".pi" / "agent" / "models.json"
    try:
        text = models_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return _extract_base_url(data, provider_name)


def _probe_endpoint(base_url: str, api_key_env: str | None, timeout: float) -> dict:
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
    if api_key_env:
        key = os.environ.get(api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        # scheme validated above (http/https only); base_url is
        # config-supplied, not attacker/network-controlled input.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return _check(
            "agent_reachable",
            False,
            "error",
            f"endpoint unreachable (endpoint-unreachable): {base_url}",
            f"check that the endpoint at {base_url} is running and reachable " "from this machine",
        )

    if status == 401:
        remediation = (
            f"set {api_key_env} to a valid API key"
            if api_key_env
            else "configure api_key_env in config.toml (or update the bearer key in "
            "~/.pi/agent/models.json for pi) with a valid key"
        )
        return _check(
            "agent_reachable",
            False,
            "error",
            f"endpoint returned 401 Unauthorized (endpoint-401): {base_url}",
            remediation,
        )
    if status == 200:
        return _check("agent_reachable", True, "info", f"endpoint reachable: {base_url}", "")
    return _check(
        "agent_reachable",
        False,
        "warning",
        f"endpoint {base_url} responded with HTTP {status}",
        "",
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
        if which("pi") is None:
            return _check(
                "agent_reachable",
                False,
                "error",
                "configured provider is pi, but 'pi' is not on PATH (pi-missing)",
                "nvsh agent install pi, or nvsh agent use openai-compat",
            )
        base_url = _pi_base_url(config, home)
        # pi's own bearer key lives in ~/.pi/agent/models.json, never in
        # nvsh's config.toml, so there is no api_key_env to name here.
        api_key_env = None
    elif provider == "openai-compat":
        settings = config.agents.get("openai-compat", {})
        base_url_raw = settings.get("base_url")
        base_url = str(base_url_raw) if base_url_raw else None
        api_key_env = settings.get("api_key_env")
    else:
        return _check(
            "agent_reachable",
            False,
            "warning",
            f"configured provider '{provider}' has no reachability probe",
            "",
        )

    if not base_url:
        return _check(
            "agent_reachable",
            False,
            "warning",
            f"endpoint unknown for provider '{provider}'; nothing to probe",
            "set base_url in config.toml, or ~/.pi/agent/models.json for pi",
        )

    return _probe_endpoint(base_url, api_key_env, timeout)


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
_ARRAY_ELEM_RE = re.compile(r'\[(\d+)\]="((?:[^"\\]|\\.)*)"')
_DECLARE_STRING_RE = re.compile(r'declare\s+--\s*PROMPT_COMMAND="((?:[^"\\]|\\.)*)"')

#: The exact fix command reported when ``__nvsh_hook`` is not first.
HOOK_FIX_COMMAND = 'PROMPT_COMMAND=(__nvsh_hook "${PROMPT_COMMAND[@]/__nvsh_hook}")  # or: nvsh on'


def _unescape_declare(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


def _parse_prompt_command(text: str) -> list[str] | None:
    if _DECLARE_ARRAY_RE.search(text) or "PROMPT_COMMAND=(" in text:
        pairs = [
            (int(index), _unescape_declare(value)) for index, value in _ARRAY_ELEM_RE.findall(text)
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
    if elements[0] == "__nvsh_hook":
        return _check(
            "hook_first_in_prompt_command",
            True,
            "info",
            "__nvsh_hook is first in PROMPT_COMMAND",
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

_REQUIRED_BIND_SUBSTRINGS = (
    r'"\C-x\C-n": __nvsh_enter',
    r'"\C-g": __nvsh_ctrl_g',
)
_ENTER_MACRO_RE = re.compile(r'"\\C-m":\s*"\\C-x\\C-n\\C-j"')
_ENTER_MACRO_LABEL = r'"\C-m": "\C-x\C-n\C-j"'


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
    missing = [entry for entry in _REQUIRED_BIND_SUBSTRINGS if entry not in bind_p_text]
    if not _ENTER_MACRO_RE.search(bind_p_text):
        missing.append(_ENTER_MACRO_LABEL)
    if not missing:
        return _check(
            "bindings_present",
            True,
            "info",
            f"nvsh readline bindings present ({label})",
            "",
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
        return _check(
            "capture_active",
            False,
            "warning",
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
