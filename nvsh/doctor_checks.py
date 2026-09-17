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

**CLI-harness reachability never makes a model call (task t17).** For
claude/codex/qwen/qwen-p/agy/kiro, :func:`check_agent_reachable` shells out
only to ``<binary> --version`` and, for kiro-cli alone, the verified
read-only ``kiro-cli whoami``. It deliberately does **not** probe agy's own
"Print mode: not authenticated" text by invoking agy's real print mode
(``agy -p <prompt> --output-format stream-json``): that text was recorded
live from an *unauthenticated* agy, but the same invocation against an
*authenticated* one would be a real, possibly paid, call to a model --
exactly what this task forbids. Rather than gamble on the account's auth
state to decide whether a probe is safe, this module never runs that
invocation at all; agy's (and claude/codex/qwen's, whose auth-status
subcommands are unverified here) auth state is reported as "not verified"
instead of guessed. This is a real capability gap versus the fuller spec
text in docs/specs/2026-09-14-first-class-multi-harness-with-aliases.md
("doctor's per-harness reachability check reports auth state without a
model call" for every harness) -- recorded here rather than silently
narrowed, per this task's own accuracy requirement.
"""

from __future__ import annotations

import json
import os
import re
import socket as socket_lib
import subprocess  # nosec B404 - fixed argv lists below, no shell=True
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, NamedTuple

from nvsh import capture as capture_mod
from nvsh import client_transport
from nvsh import daemon as daemon_mod
from nvsh.agent import registry as agent_registry
from nvsh.agent.demo import FIXTURE_PATH as DEMO_FIXTURE_PATH
from nvsh.config import (
    DEFAULT_ALIAS,
    DEFAULT_KEY_FILE_DISPLAY,
    NO_BEARER_NOTE,
    BearerResolution,
    Config,
    ConfigError,
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

    # The harness nvsh will actually run is the default alias's backend,
    # which may differ from the legacy [agent] provider (mirrors
    # check_agent_reachable's own resolve_target(DEFAULT_ALIAS) call).
    try:
        provider = config.resolve_target(DEFAULT_ALIAS)[0]
    except ConfigError as exc:
        return _check(
            "agent_configured",
            False,
            "error",
            f"the default target does not resolve: {exc}",
            "fix [aliases].default in config.toml (nvsh agent use <name>)",
        )

    source = "[aliases].default" if DEFAULT_ALIAS in config.aliases else "[agent] provider"

    if provider in agent_registry.ADAPTERS:
        return _check(
            "agent_configured",
            True,
            "info",
            f"agent provider configured: {provider} (via {source})",
            "",
        )

    known = ", ".join(sorted(agent_registry.ADAPTERS))
    if source == "[aliases].default":
        # resolve_target() consults [aliases].default before the legacy
        # [agent] provider (nvsh/config.py's Config.resolve_target), so
        # telling the operator to edit [agent] provider here cannot fix
        # this -- the remediation has to point at the alias that is
        # actually wrong.
        remediation = (
            f"fix or remove [aliases].default in config.toml, or run "
            f"`nvsh agent use <name>`, with one of: {known}"
        )
    else:
        remediation = f"set [agent] provider to one of: {known}"
    return _check(
        "agent_configured",
        False,
        "error",
        f"configured provider '{provider}' (via {source}) is not a known adapter",
        remediation,
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


def _check_demo_reachable(config: Config) -> dict:
    """``demo``'s "reachability" is just: can the fixture be read?

    ``demo`` runs no subprocess and opens no socket (see
    ``nvsh/agent/demo.py``'s module docstring), so there is nothing to probe
    the way a CLI harness or an HTTP endpoint is probed above. What *can*
    fail is the fixture itself -- an ``[agents.demo] fixture`` override
    pointing at a path that does not exist or is not readable -- so this
    reports that instead, never the file's contents.
    """
    settings = config.agents.get("demo", {})
    fixture = settings.get("fixture")
    path = Path(str(fixture)) if fixture else DEMO_FIXTURE_PATH
    try:
        from .agent.demo import load_fixture

        load_fixture(path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _check(
            "agent_reachable",
            False,
            "error",
            f"demo fixture unusable ({path}): {exc}",
            "point [agents.demo] fixture at a readable JSON file, or remove the "
            "override to use the committed fixture",
        )
    return _check(
        "agent_reachable",
        True,
        "info",
        f"demo reachable via fixture ({path})",
        "",
    )


#: Harnesses driven as a plain subprocess CLI (as opposed to pi's rpc mode or
#: openai-compat's HTTP probe), dispatched by :func:`check_agent_reachable`
#: to :func:`_check_cli_harness_reachable`. Keys are ``registry.ADAPTERS``
#: names; the binary to run for each comes from the registry itself
#: (``AdapterSpec.binary``) so this module never re-states it.
_CLI_HARNESS_PROVIDERS = frozenset({"claude", "codex", "qwen", "qwen-p", "agy", "kiro"})

#: Short timeout for a CLI reachability probe (``--version`` and, where one
#: exists, a dedicated auth-probe subcommand). Deliberately shorter than the
#: openai-compat network probe's default: these are local process spawns,
#: not network round-trips, and a hang here must not stall `nvsh doctor`.
CLI_PROBE_TIMEOUT = 5.0

#: (returncode, stdout, stderr) for one CLI invocation, mirroring
#: ``platform._subprocess.Runner`` -- but, unlike that Runner, this one
#: *raises* ``subprocess.TimeoutExpired`` on a hang instead of swallowing it,
#: because a hang must become its own failed check with its own remediation
#: (task t17), not the generic "exit 1" a timeout collapses to there.
CliRunner = Callable[[list, float], tuple]


def _default_cli_run(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """Real subprocess runner for CLI reachability probes.

    Fixed argv, no shell, text mode. ``subprocess.TimeoutExpired`` and
    ``OSError`` (missing binary, not executable, ...) are left to propagate
    -- :func:`_check_cli_harness_reachable` turns each into its own check.
    """
    proc = subprocess.run(  # nosec B603 - argv is a fixed list, no shell
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


#: Matched per whitespace-separated token (never against the whole
#: ``--version`` output) so the pattern is anchored at both ends and cannot
#: backtrack across the input the way a bare ``search()`` over the full text
#: could.
#: The optional suffix must start with a non-digit (``-rc1``, ``+build``,
#: ``)``), so the third group and the suffix can never compete for the same
#: characters and the match is linear.
_VERSION_TOKEN_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:[^\d\s]\S*)?")


def _parse_cli_version(text: str) -> tuple[int, int, int] | None:
    for token in text.split():
        match = _VERSION_TOKEN_RE.fullmatch(token)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


#: Floor of the version range this module's parsing/probing has been
#: exercised against, per adapter (``registry.ADAPTERS`` name -> minimum,
#: maximum|None). No adapter here declares a ceiling today -- these floors
#: are the earliest version cited in docs/specs/2026-09-14-first-class-
#: multi-harness-with-aliases.md's per-adapter scope-exploration entries
#: (s4-s8, s26), re-verified against what is actually installed on this box
#: on 2026-09-14 per the task handoff: claude 2.1.270, codex-cli 0.147.0,
#: qwen 0.23.3, agy 1.2.2 (agy's floor stays 1.0.1's minor, 1.0.0, since
#: that is the version the adapter's probed behavior is documented against),
#: kiro-cli 2.0.0. A version below the floor is a warning, not an error --
#: an older CLI is usually still usable, just unverified.
_SUPPORTED_VERSIONS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int] | None]] = {
    "claude": ((2, 0, 0), None),
    "codex": ((0, 100, 0), None),
    "qwen": ((0, 20, 0), None),
    "qwen-p": ((0, 20, 0), None),
    "agy": ((1, 0, 0), None),
    "kiro": ((2, 0, 0), None),
}


@dataclass(frozen=True)
class _AuthProbe:
    """A dedicated, read-only subcommand that reveals a harness's auth state
    without ever reaching a model.

    Only ``kiro-cli whoami`` is a *verified* safe probe (its exact
    unauthenticated text -- "You are not logged in, please log in with
    kiro-cli login" -- was recorded live on 2026-09-14; see
    tests/fixtures/doctor/). claude, codex, qwen and agy have no dedicated
    probe registered here: each CLI's real auth-status subcommand was not
    independently verified for this task, and inventing one risks a false
    read (or, for agy's print-mode probe specifically, a real paid call once
    authenticated -- see the module docstring's honesty note). Those
    harnesses still get a version check; their auth state is reported as
    "not verified" rather than guessed.
    """

    args: tuple[str, ...]
    login_command: str


_AUTH_PROBES: dict[str, _AuthProbe] = {
    "kiro": _AuthProbe(args=("whoami",), login_command="kiro-cli login"),
}

#: Substrings (checked case-insensitively) that mean "not authenticated" in
#: a harness CLI's own stdout/stderr. Deliberately generic -- shared across
#: every harness rather than one hard-coded sentence per CLI -- because only
#: kiro-cli's exact text is verified (see :class:`_AuthProbe`); the others
#: fall back to this list against whatever a probe actually printed.
_UNAUTH_MARKERS = (
    "not logged in",
    "not authenticated",
    "please log in",
    "please login",
    "log in with",
    "not signed in",
)


def _cli_harness_login_command(display_name: str, binary: str) -> str:
    probe = _AUTH_PROBES.get(display_name)
    if probe is not None:
        return probe.login_command
    # Best-effort guess, not independently verified for this CLI (see
    # _AuthProbe's docstring) -- still better than no remediation at all.
    return f"{binary} login"


def _probe_cli_version(
    display_name: str,
    binary: str,
    run: CliRunner,
    timeout: float,
) -> tuple[dict | None, str]:
    """Step one: run ``<binary> --version``.

    Returns either ``(finished failed check, "")`` or ``(None, combined
    stdout/stderr)`` for the next step to parse.
    """
    try:
        returncode, stdout, stderr = run([binary, "--version"], timeout)
    except subprocess.TimeoutExpired:
        return (
            _check(
                "agent_reachable",
                False,
                "error",
                f"'{binary} --version' timed out after {timeout}s ({display_name}-hung)",
                f"the {binary} CLI is hanging; check it manually ('{binary} --version')",
            ),
            "",
        )
    except OSError as exc:
        return (
            _check(
                "agent_reachable",
                False,
                "error",
                f"'{binary} --version' could not be run: {exc.__class__.__name__} "
                f"({display_name}-unreachable)",
                f"check the {binary} installation ('{binary} --version')",
            ),
            "",
        )

    version_text = f"{stdout}\n{stderr}"
    if returncode != 0 and _parse_cli_version(version_text):
        # A parseable version printed by a failing process is not "reachable";
        # an unparseable one keeps the softer "could not determine" warning.
        return (
            _check(
                "agent_reachable",
                False,
                "error",
                f"'{binary} --version' exited {returncode} ({display_name}-unreachable)",
                f"check the {binary} installation ('{binary} --version')",
            ),
            "",
        )
    return None, version_text


def _version_range_note(display_name: str, version: tuple[int, int, int] | None) -> str:
    """Step two: the `` (outside the supported range ...)`` suffix, or ``""``.

    Empty when the version could not be parsed, when this harness declares no
    supported range, or when the version falls inside it.
    """
    version_range = _SUPPORTED_VERSIONS.get(display_name)
    if not version or not version_range:
        return ""
    minimum, maximum = version_range
    below_min = version < minimum
    above_max = maximum is not None and version > maximum
    if not (below_min or above_max):
        return ""
    range_note = f" (outside the supported range >= {_format_version(minimum)}"
    range_note += f" <= {_format_version(maximum)})" if maximum else ")"
    return range_note


def _probe_cli_auth(
    display_name: str,
    binary: str,
    run: CliRunner,
    timeout: float,
    version_text: str,
) -> tuple[dict | None, str]:
    """Step three: run this harness's verified auth probe, if it has one.

    Returns either ``(finished failed check, "")`` or ``(None, the text to
    scan for :data:`_UNAUTH_MARKERS`)`` -- ``version_text`` itself when there
    is no registered probe, or when running it raised ``OSError``.
    """
    auth_probe = _AUTH_PROBES.get(display_name)
    if auth_probe is None:
        return None, version_text
    try:
        _rc, out, err = run([binary, *auth_probe.args], timeout)
    except subprocess.TimeoutExpired:
        probe_cmd = " ".join((binary, *auth_probe.args))
        return (
            _check(
                "agent_reachable",
                False,
                "error",
                f"'{probe_cmd}' timed out after {timeout}s ({display_name}-hung)",
                f"the {binary} CLI is hanging; check it manually ('{probe_cmd}')",
            ),
            "",
        )
    except OSError:
        return None, version_text  # fall back to scanning --version's own output
    return None, f"{out}\n{err}"


def _cli_harness_verdict(
    display_name: str,
    binary: str,
    version: tuple[int, int, int] | None,
    range_note: str,
    auth_text: str,
) -> dict:
    """Turn the three steps' collected facts into the one rubric-shaped dict."""
    version_note = f"version {_format_version(version)}" if version else "version unknown"

    if any(marker in auth_text.lower() for marker in _UNAUTH_MARKERS):
        return _check(
            "agent_reachable",
            False,
            "error",
            f"{display_name} is not authenticated ({version_note}{range_note})",
            _cli_harness_login_command(display_name, binary),
        )

    if not version:
        return _check(
            "agent_reachable",
            False,
            "warning",
            f"could not determine {display_name} version ('{binary} --version' unparseable)",
            f"run '{binary} --version' manually to check the installation",
        )

    if range_note:
        return _check(
            "agent_reachable",
            False,
            "warning",
            f"{display_name} {version_note}{range_note}",
            f"install a supported {display_name} version",
        )

    auth_suffix = (
        ""
        if display_name in _AUTH_PROBES
        else "; auth state not verified (no safe non-model probe)"
    )
    return _check(
        "agent_reachable",
        True,
        "info",
        f"{display_name} reachable ({version_note}{auth_suffix})",
        "",
    )


def _check_cli_harness_reachable(
    display_name: str,
    binary: str,
    which: Which,
    run: CliRunner,
    timeout: float,
) -> dict:
    """Binary present, version parsed + range-checked, auth state where a
    verified probe exists -- never a model call (task t17).
    """
    if which(binary) is None:
        return _check(
            "agent_reachable",
            False,
            "error",
            f"'{binary}' is not on PATH ({display_name}-missing)",
            f"nvsh agent install {display_name}, or nvsh agent use openai-compat",
        )

    failure, version_text = _probe_cli_version(display_name, binary, run, timeout)
    if failure is not None:
        return failure
    version = _parse_cli_version(version_text)

    failure, auth_text = _probe_cli_auth(display_name, binary, run, timeout, version_text)
    if failure is not None:
        return failure

    return _cli_harness_verdict(
        display_name,
        binary,
        version,
        _version_range_note(display_name, version),
        auth_text,
    )


def check_agent_reachable(
    config: Config,
    which: Which = default_which,
    home: Path | None = None,
    timeout: float = 3.0,
    run: CliRunner = _default_cli_run,
    cli_timeout: float = CLI_PROBE_TIMEOUT,
) -> dict:
    home = home if home is not None else Path.home()
    provider = config.agent_provider
    try:
        # The harness nvsh will actually run is the default alias's backend,
        # which may differ from the legacy ``[agent] provider``.
        provider = config.resolve_target(DEFAULT_ALIAS)[0]
    except ConfigError as exc:
        return _check(
            "agent_reachable",
            False,
            "error",
            f"the default target does not resolve: {exc}",
            "fix [aliases].default in config.toml (nvsh agent use <name>)",
        )

    if provider == "pi":
        inputs = _pi_probe_inputs(config, home, which)
    elif provider == "openai-compat":
        inputs = _openai_compat_probe_inputs(config)
    elif provider == "demo":
        return _check_demo_reachable(config)
    elif provider in _CLI_HARNESS_PROVIDERS:
        spec = agent_registry.ADAPTERS.get(provider)
        binary = spec.binary if spec is not None else None
        if not binary:
            return _check(
                "agent_reachable",
                False,
                "warning",
                f"configured provider '{provider}' has no reachability probe",
                "",
            )
        return _check_cli_harness_reachable(provider, binary, which, run, cli_timeout)
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
# default_target_not_demo
# ---------------------------------------------------------------------------


def check_default_target_not_demo(config: Config | None) -> dict:
    """Fail when the resolved default target is ``demo``.

    ``demo`` is a scripted fixture replay (:data:`nvsh.agent.registry.DEMO_DEFAULT_MESSAGE`),
    never a real backend -- ``nvsh agent use demo`` and ``nvsh setup --agent
    demo`` already refuse to write it as ``[aliases].default``, but nothing
    stops an operator from hand-editing ``config.toml``. This is the doctor
    check that catches that: a hand-edited default resolving to ``demo``
    means every ordinary failure on this machine would replay the same
    canned fixture instead of calling a real backend.

    Passes -- with an info message, never a failure -- when there is no
    ``config`` to check (a wheel install with no ``config.toml``, or one
    that failed to load and is already reported by ``agent_configured``) and
    when the default target does not resolve at all (``agent_configured``
    and ``agent_reachable`` already report that failure; this check has
    nothing more useful to add).
    """
    if config is None:
        return _check(
            "default_target_not_demo",
            True,
            "info",
            "no config.toml loaded; nothing resolves to demo",
            "",
        )
    try:
        backend = config.resolve_target(DEFAULT_ALIAS)[0]
    except ConfigError:
        return _check(
            "default_target_not_demo",
            True,
            "info",
            "default target does not resolve; see agent_configured",
            "",
        )
    if backend == "demo":
        return _check(
            "default_target_not_demo",
            False,
            "error",
            "[aliases].default resolves to demo, a scripted fixture -- not a real backend",
            "run `nvsh agent use <name>` with a real backend (see `nvsh agent list`)",
        )
    return _check(
        "default_target_not_demo",
        True,
        "info",
        f"default target resolves to {backend!r}, not demo",
        "",
    )


# ---------------------------------------------------------------------------
# agent_allowlist
# ---------------------------------------------------------------------------


def _json_list_present(text: str, *path: str) -> bool:
    """Does the JSON in *text* have a non-empty list at the dotted *path*?"""
    try:
        data = json.loads(text)
    except ValueError:
        return False
    node: object = data
    for key in path:
        if not isinstance(node, dict):
            return False
        node = node.get(key)
    return isinstance(node, list) and len(node) > 0


def _toml_key_present(text: str, key: str) -> bool:
    """Does the TOML in *text* set *key* at the top level, to anything?"""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return False
    return key in data


@dataclass(frozen=True)
class _AllowlistFile:
    """One harness-side persistent-allowlist file doctor knows how to read.

    ``relpath`` is relative to *home* (never a hard-coded absolute path, so
    tests point it at a ``tmp_path``). ``configured`` decides whether the
    file, once read, actually carries an allow-rule worth warning about (an
    empty/default file is not).
    """

    display_name: str
    relpath: str
    configured: Callable[[str], bool]


#: Read-only, doctor never writes to any of these (scope boundary in
#: docs/specs/2026-09-14-first-class-multi-harness-with-aliases.md:
#: "nvsh never edits, creates or overrides a harness's own settings or trust
#: files"). Paths recorded live on this box 2026-09-14 (spec s28) for agy
#: and observed conventions for claude/codex; kiro's trust-settings file
#: location is *not* independently verified (kiro-cli's own --trust-tools
#: is a launch flag, not a confirmed persisted file) -- it is included on a
#: best-effort basis per the task's own instruction and reports nothing if
#: the guessed path is absent, which is the honest outcome until verified.
_ALLOWLIST_FILES: tuple[_AllowlistFile, ...] = (
    _AllowlistFile(
        "claude",
        ".claude/settings.json",
        lambda text: _json_list_present(text, "permissions", "allow"),
    ),
    _AllowlistFile(
        "agy",
        ".gemini/antigravity-cli/settings.json",
        lambda text: _json_list_present(text, "permissions", "allow"),
    ),
    _AllowlistFile(
        "codex",
        ".codex/config.toml",
        lambda text: _toml_key_present(text, "approval_policy"),
    ),
    _AllowlistFile(
        "kiro",
        ".kiro/settings.json",
        lambda text: _json_list_present(text, "trustedTools") or _json_list_present(text, "trust"),
    ),
)


def check_agent_allowlist(home: Path | None = None) -> dict:
    """Warn, naming the file, for any harness-side allowlist doctor can read.

    Never writes to any of these files -- read-only, `Path.read_text` only.
    Runs regardless of the configured provider: an operator switching
    harnesses should still see what any installed harness would let run
    unmediated.
    """
    home = home if home is not None else Path.home()
    found: list[str] = []
    for entry in _ALLOWLIST_FILES:
        path = home / entry.relpath
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if entry.configured(text):
            found.append(str(path))

    if not found:
        return _check(
            "agent_allowlist",
            True,
            "info",
            "no harness-side command allowlist files found",
            "",
        )

    return _check(
        "agent_allowlist",
        False,
        "warning",
        "harness-side allowlist(s) let commands run without nvsh approve: " + ", ".join(found),
        "review and tighten the allowlist(s) yourself; nvsh never edits harness settings files",
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
# agent_turn_not_hung
# ---------------------------------------------------------------------------

#: Connect timeout for the read-only status probe this check makes. Short,
#: like ``overview.py``'s ``_ACTIVITY_CONNECT_TIMEOUT`` -- a stale socket
#: file with nothing listening must not make plain ``nvsh doctor`` hang.
_ACTIVE_TURN_CONNECT_TIMEOUT = 0.5


def _default_active_turn(env: Mapping[str, str]) -> Mapping[str, object] | None:
    """The daemon's current active turn, or ``None`` when idle/unreachable.

    Read-only: goes through :func:`nvsh.client_transport.status`, which
    sends a ``status`` control message and never autostarts a daemon or
    mutates anything. This is the only network call
    :func:`check_agent_turn_not_hung` makes, so plain ``nvsh doctor`` (no
    ``--apply``) never sends a mutating control message.
    """
    state = client_transport.status(env=env, timeout=_ACTIVE_TURN_CONNECT_TIMEOUT)
    if not state.get("running"):
        return None
    active = state.get("active_turn")
    return active if isinstance(active, Mapping) else None


def check_agent_turn_not_hung(
    active_turn: Mapping[str, object] | None,
    *,
    threshold: float,
    pid_gone: Callable[[str], bool] = daemon_mod.shell_pid_gone,
) -> dict:
    """Is the daemon's active turn (if any) still making progress?

    Fails (``severity=warning``) when the turn's owner shell no longer
    exists (its pid is gone -- the shell that started it exited or crashed
    without ever ending the turn) or when its elapsed time exceeds
    *threshold* (the daemon's own turn cap by default, so this flags a turn
    the daemon's own watchdog should have already ended). Passes trivially
    when there is no active turn at all. The remediation always names
    ``nvsh doctor --apply``, which is the only thing that acts on this
    check (task t19); this function itself only reports.
    """
    if not active_turn:
        return _check("agent_turn_not_hung", True, "info", "no active agent turn", "")
    shell = str(active_turn.get("shell", "") or "")
    try:
        elapsed = float(active_turn.get("elapsed", 0.0) or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    dead = pid_gone(shell)
    if not dead and elapsed <= threshold:
        return _check(
            "agent_turn_not_hung",
            True,
            "info",
            f"active turn for shell {shell}, {elapsed:.0f}s elapsed",
            "",
        )
    reason = (
        f"owner shell {shell} is gone"
        if dead
        else f"{elapsed:.0f}s elapsed exceeds the {threshold:.0f}s turn cap"
    )
    return _check(
        "agent_turn_not_hung",
        False,
        "warning",
        f"active turn for shell {shell} looks hung ({reason})",
        "nvsh doctor --apply",
    )


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


@dataclass(frozen=True)
class Probes:
    """The injectables :func:`collect_checks` passes straight through.

    Grouped into one frozen bundle (rather than one keyword argument each) so
    ``collect_checks`` stays inside the parameter limit; every field keeps the
    same default it had as a standalone keyword, and nothing in-tree overrides
    them, so the defaults are what production always uses.

    ``cli_run`` is deliberately *not* ``collect_checks``'s ``run``: ``run``
    (``platform._subprocess.Runner``) feeds ``check_terminfo_present`` and
    always returns a tuple, even on a timeout; ``cli_run`` feeds the
    per-harness CLI reachability probes inside ``check_agent_reachable`` and
    *raises* ``subprocess.TimeoutExpired`` on a hang, because that probe
    reports a hang as its own failed check (task t17), not a generic
    ``(1, "", "")``.
    """

    cli_run: CliRunner = _default_cli_run
    is_running: Callable[[Mapping[str, str]], bool] = daemon_mod.is_running
    socket_path: Callable[[Mapping[str, str]], Path] = daemon_mod.socket_path
    active_turn: Callable[[Mapping[str, str]], Mapping[str, object] | None] = _default_active_turn


_DEFAULT_PROBES = Probes()


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
    probes: Probes = _DEFAULT_PROBES,
) -> list[dict]:
    """Run every doctor extension check and return their rubric-shaped dicts.

    Called unconditionally by ``doctor.py``'s ``_diagnose()`` — including in
    the wheel-install branch where no ``culture.yaml`` exists, since these
    checks are about the shell/backend, not the mesh-identity invariants.

    ``run`` stays a keyword of its own; the remaining injectables live on
    :class:`Probes`, which documents why its ``cli_run`` is not this ``run``.
    """
    checks = [check_platform_detected(platform)]
    checks.append(check_agent_configured(config, config_error))
    if config is not None:
        checks.append(check_agent_reachable(config, which=which, home=home, run=probes.cli_run))
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
    checks.append(check_default_target_not_demo(config))
    checks.append(check_agent_allowlist(home=home))
    checks.append(check_hook_sourced(env, current_version))
    checks.append(check_hook_first_in_prompt_command(prompt_command_text))
    checks.append(check_bindings_present(bind_p_text, keymap))
    checks.append(check_capture_active(env))
    checks.append(
        check_daemon_status(env, is_running=probes.is_running, socket_path=probes.socket_path)
    )
    checks.append(
        check_agent_turn_not_hung(probes.active_turn(env), threshold=daemon_mod.turn_timeout(env))
    )
    checks.append(check_terminfo_present(env, run=run))
    return checks
