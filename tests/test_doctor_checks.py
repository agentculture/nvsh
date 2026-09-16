"""Tests for nvsh.doctor_checks — the doctor-extension pure functions (task t17).

Every check function takes plain data (env mapping, config, platform, or raw
bash-side state strings) plus injectables (``which``, ``run``) and returns a
single ``{id, passed, severity, message, remediation}`` dict, mirroring the
shape ``nvsh.cli._commands.doctor._diagnose`` already returns. This file
tests the pure functions directly; ``tests/test_cli_doctor_extensions.py``
covers the CLI wiring and the wheel-install branch.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from nvsh import doctor_checks
from nvsh.config import Config
from nvsh.platform import Platform, Value

FIXTURES = Path(__file__).parent / "fixtures" / "doctor"


def _fixture_text(name: str) -> str:
    """A recorded doctor fixture's body, with its '# recorded-from:' header
    line stripped -- the header is provenance for the fixture file itself,
    not part of what a real CLI invocation would print on stdout/stderr."""
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(line for line in lines if not line.startswith("#"))


@pytest.fixture(autouse=True)
def _isolated_config_home(tmp_path, monkeypatch):
    """Keep a real ``$XDG_CONFIG_HOME/nvsh/api_key`` out of these checks (d10)."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


def _check(checks, check_id):
    return next(c for c in checks if c["id"] == check_id)


# --- platform_detected ------------------------------------------------------


def test_platform_detected_passes_for_a_known_kind():
    plat = Platform(kind="dgx-spark", values=(Value("x", "1", "/etc/x", "file", True),))
    check = doctor_checks.check_platform_detected(plat)
    assert check["id"] == "platform_detected"
    assert check["passed"] is True
    assert check["severity"] == "info"


def test_platform_detected_warns_for_generic_and_lists_sources():
    plat = Platform(
        kind="generic",
        values=(
            Value("dgx_name", None, "/etc/dgx-release", "file", False),
            Value("tegra", None, "/etc/nv_tegra_release", "file", False),
        ),
    )
    check = doctor_checks.check_platform_detected(plat)
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "/etc/dgx-release" in check["message"]
    assert "/etc/nv_tegra_release" in check["message"]


# --- agent_configured --------------------------------------------------------


def test_agent_configured_passes_for_a_known_provider():
    cfg = Config(agent_provider="pi")
    check = doctor_checks.check_agent_configured(cfg, None)
    assert check["passed"] is True
    assert check["severity"] == "info"


def test_agent_configured_fails_for_an_unknown_provider():
    cfg = Config(agent_provider="not-a-real-adapter")
    check = doctor_checks.check_agent_configured(cfg, None)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "not-a-real-adapter" in check["message"]


def test_agent_configured_unknown_via_aliases_default_points_at_the_alias():
    # Qodo #10: [aliases].default resolves before the legacy [agent]
    # provider (Config.resolve_target), so telling the operator to set
    # [agent] provider cannot fix an unknown [aliases].default target.
    cfg = Config(aliases={"default": "nosuch"})
    check = doctor_checks.check_agent_configured(cfg, None)
    assert check["passed"] is False
    assert "[aliases].default" in check["remediation"]
    assert "nvsh agent use" in check["remediation"]
    assert "[agent] provider" not in check["remediation"]


def test_agent_configured_unknown_via_agent_provider_still_names_agent_provider():
    cfg = Config(agent_provider="nosuch")
    check = doctor_checks.check_agent_configured(cfg, None)
    assert check["passed"] is False
    assert "[agent] provider" in check["remediation"]


def test_agent_configured_fails_when_config_toml_failed_to_load():
    check = doctor_checks.check_agent_configured(None, "malformed TOML")
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "malformed TOML" in check["message"]


def test_agent_configured_reports_the_resolved_default_via_agent_provider():
    # No [aliases].default set -> resolve_target('default') falls back to
    # the legacy [agent] provider.
    cfg = Config(agent_provider="claude")
    check = doctor_checks.check_agent_configured(cfg, None)
    assert check["passed"] is True
    assert "claude" in check["message"]
    assert "via [agent] provider" in check["message"]


def test_agent_configured_reports_the_resolved_default_via_aliases_default():
    cfg = Config(agent_provider="pi", aliases={"default": "codex"})
    check = doctor_checks.check_agent_configured(cfg, None)
    assert check["passed"] is True
    assert "codex" in check["message"]
    assert "via [aliases].default" in check["message"]


def test_agent_configured_fails_when_default_alias_does_not_resolve():
    cfg = Config(agent_provider="pi", aliases={"default": "not-a-real-adapter"})
    check = doctor_checks.check_agent_configured(cfg, None)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "not-a-real-adapter" in check["message"]


# --- agent_reachable ---------------------------------------------------------


def test_agent_reachable_pi_missing(monkeypatch):
    cfg = Config(agent_provider="pi")
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: None)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "pi-missing" in check["message"]
    assert "nvsh agent install pi" in check["remediation"]


def test_agent_reachable_pi_endpoint_unknown(tmp_path):
    cfg = Config(agent_provider="pi")
    # No base_url in config, and no ~/.pi/agent/models.json on this fake home.
    check = doctor_checks.check_agent_reachable(
        cfg, which=lambda name: "/usr/bin/pi", home=tmp_path
    )
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "endpoint unknown" in check["message"]


def test_agent_reachable_pi_reads_base_url_from_models_json(tmp_path):
    models = tmp_path / ".pi" / "agent" / "models.json"
    models.parent.mkdir(parents=True)
    models.write_text(
        json.dumps({"providers": {"nemotron": {"baseUrl": "http://127.0.0.1:1/v1"}}}),
        encoding="utf-8",
    )
    cfg = Config(agent_provider="pi", agents={"pi": {"provider": "nemotron", "model": "associate"}})
    check = doctor_checks.check_agent_reachable(
        cfg, which=lambda name: "/usr/bin/pi", home=tmp_path
    )
    # Nothing listens on port 1 -> unreachable. The check names *where* the
    # base_url came from; it never prints the URL itself (d4b).
    assert check["passed"] is False
    assert doctor_checks.PI_MODELS_JSON_SOURCE in check["message"]
    assert "127.0.0.1" not in _all_check_text(check)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Handler(http.server.BaseHTTPRequestHandler):
    status_code = 200

    def do_GET(self):  # noqa: N802 - stdlib method name
        self.send_response(self.status_code)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):  # silence test output
        pass


def _serve(status_code: int):
    port = _free_port()
    handler = type("Handler", (_Handler,), {"status_code": status_code})
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}"


def test_agent_reachable_openai_compat_200_passes():
    server, thread, base_url = _serve(200)
    try:
        cfg = Config(
            agent_provider="openai-compat", agents={"openai-compat": {"base_url": base_url}}
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert check["passed"] is True
        assert check["severity"] == "info"
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_openai_compat_401_reports_endpoint_401(monkeypatch):
    server, thread, base_url = _serve(401)
    try:
        monkeypatch.setenv("NVSH_TEST_KEY", "secret-value")
        cfg = Config(
            agent_provider="openai-compat",
            agents={"openai-compat": {"base_url": base_url, "api_key_env": "NVSH_TEST_KEY"}},
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert check["passed"] is False
        assert "endpoint-401" in check["message"]
        assert "NVSH_TEST_KEY" in check["remediation"]
        assert "secret-value" not in check["remediation"]
        assert "secret-value" not in check["message"]
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_unreachable_closed_port():
    port = _free_port()  # nothing listening on it
    cfg = Config(
        agent_provider="openai-compat",
        agents={"openai-compat": {"base_url": f"http://127.0.0.1:{port}"}},
    )
    check = doctor_checks.check_agent_reachable(cfg, timeout=1.0)
    assert check["passed"] is False
    assert "endpoint-unreachable" in check["message"]
    assert "[agents.openai-compat]" in check["message"]
    assert str(port) not in _all_check_text(check)


# --- agent_reachable: the endpoint URL never leaves the process (d4b) --------


def _assert_no_endpoint_in_check(check: dict, base_url: str) -> None:
    """No URL, scheme, host or port in *any* user-facing field of the check."""
    text = _all_check_text(check)
    assert "http" not in text.lower(), text
    assert "127.0.0.1" not in text, text
    assert urlsplit(base_url).netloc not in text, text


def test_agent_reachable_200_never_prints_the_endpoint_url():
    server, thread, base_url = _serve(200)
    try:
        cfg = Config(
            agent_provider="openai-compat", agents={"openai-compat": {"base_url": base_url}}
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert check["passed"] is True
        assert check["message"] == (
            "endpoint reachable (base_url from [agents.openai-compat], no bearer configured)"
        )
        _assert_no_endpoint_in_check(check, base_url)
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_200_with_bearer_names_the_env_var_not_the_url(monkeypatch):
    server, thread, base_url = _serve(200)
    try:
        monkeypatch.setenv("NVSH_TEST_KEY", "example-fake-bearer-value")
        cfg = Config(
            agent_provider="openai-compat",
            agents={"openai-compat": {"base_url": base_url, "api_key_env": "NVSH_TEST_KEY"}},
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert check["message"] == (
            "endpoint reachable (base_url from [agents.openai-compat], "
            "bearer from $NVSH_TEST_KEY)"
        )
        _assert_no_endpoint_in_check(check, base_url)
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_401_never_prints_the_endpoint_url():
    server, thread, base_url = _serve(401)
    try:
        cfg = Config(
            agent_provider="openai-compat", agents={"openai-compat": {"base_url": base_url}}
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert "endpoint-401" in check["message"]
        _assert_no_endpoint_in_check(check, base_url)
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_other_status_never_prints_the_endpoint_url():
    server, thread, base_url = _serve(503)
    try:
        cfg = Config(
            agent_provider="openai-compat", agents={"openai-compat": {"base_url": base_url}}
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert check["passed"] is False
        assert "503" in check["message"]
        _assert_no_endpoint_in_check(check, base_url)
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_unreachable_never_prints_the_endpoint_url():
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    cfg = Config(agent_provider="openai-compat", agents={"openai-compat": {"base_url": base_url}})
    check = doctor_checks.check_agent_reachable(cfg, timeout=1.0)
    assert check["passed"] is False
    _assert_no_endpoint_in_check(check, base_url)


# --- agent_reachable: pi's apiKey in models.json (coordinator follow-up) -----


class _BearerGatedHandler(http.server.BaseHTTPRequestHandler):
    """200 only with the exact expected bearer; 401 otherwise (no bearer, wrong bearer)."""

    expected_bearer = ""

    def do_GET(self):  # noqa: N802 - stdlib method name
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {self.expected_bearer}":
            self.send_response(200)
        else:
            self.send_response(401)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):  # silence test output
        pass


def _serve_bearer_gated(expected_bearer: str):
    port = _free_port()
    handler = type(
        "BearerGatedHandler", (_BearerGatedHandler,), {"expected_bearer": expected_bearer}
    )
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{port}"


def _write_pi_models_json(home: Path, provider: str, api_key: str, base_url: str) -> None:
    models = home / ".pi" / "agent" / "models.json"
    models.parent.mkdir(parents=True)
    models.write_text(
        json.dumps({"providers": {provider: {"baseUrl": base_url, "apiKey": api_key}}}),
        encoding="utf-8",
    )


def _all_check_text(check: dict) -> str:
    return json.dumps(check)


def test_agent_reachable_pi_uses_literal_api_key_from_models_json(tmp_path):
    secret = "example-fake-bearer-not-a-real-key-value"
    server, thread, base_url = _serve_bearer_gated(secret)
    try:
        _write_pi_models_json(tmp_path, "nemotron", secret, base_url)
        cfg = Config(
            agent_provider="pi", agents={"pi": {"provider": "nemotron", "model": "associate"}}
        )

        check = doctor_checks.check_agent_reachable(
            cfg, which=lambda name: "/usr/bin/pi", home=tmp_path
        )

        assert check["passed"] is True
        assert check["severity"] == "info"
        assert secret not in _all_check_text(check)
        assert "models.json" in check["message"]
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_pi_resolves_env_ref_api_key_from_models_json(tmp_path, monkeypatch):
    secret = "example-fake-envref-not-a-real-key-value"
    monkeypatch.setenv("NVSH_PI_TEST_KEY", secret)
    server, thread, base_url = _serve_bearer_gated(secret)
    try:
        _write_pi_models_json(tmp_path, "nemotron", "$NVSH_PI_TEST_KEY", base_url)
        cfg = Config(
            agent_provider="pi", agents={"pi": {"provider": "nemotron", "model": "associate"}}
        )

        check = doctor_checks.check_agent_reachable(
            cfg, which=lambda name: "/usr/bin/pi", home=tmp_path
        )

        assert check["passed"] is True
        assert check["severity"] == "info"
        assert secret not in _all_check_text(check)
        assert "NVSH_PI_TEST_KEY" in check["message"]
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_pi_without_bearer_gets_401_without_leaking_absence_of_key(tmp_path):
    server, thread, base_url = _serve_bearer_gated("only-this-exact-key-passes")
    try:
        _write_pi_models_json(tmp_path, "nemotron", "wrong-key", base_url)
        cfg = Config(
            agent_provider="pi", agents={"pi": {"provider": "nemotron", "model": "associate"}}
        )

        check = doctor_checks.check_agent_reachable(
            cfg, which=lambda name: "/usr/bin/pi", home=tmp_path
        )

        assert check["passed"] is False
        assert "endpoint-401" in check["message"]
        assert "wrong-key" not in _all_check_text(check)
        assert "only-this-exact-key-passes" not in _all_check_text(check)
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_pi_401_with_no_api_key_names_models_json(tmp_path):
    server, thread, base_url = _serve(401)
    try:
        models = tmp_path / ".pi" / "agent" / "models.json"
        models.parent.mkdir(parents=True)
        models.write_text(
            json.dumps({"providers": {"nemotron": {"baseUrl": base_url}}}), encoding="utf-8"
        )
        cfg = Config(
            agent_provider="pi", agents={"pi": {"provider": "nemotron", "model": "associate"}}
        )

        check = doctor_checks.check_agent_reachable(
            cfg, which=lambda name: "/usr/bin/pi", home=tmp_path
        )

        assert check["passed"] is False
        assert "endpoint-401" in check["message"]
        assert "models.json" in check["remediation"]
        assert "nemotron" in check["remediation"]
        assert "$HOME" in check["remediation"]
        assert "~" not in check["remediation"]
    finally:
        server.shutdown()
        thread.join()


# --- hook_sourced -------------------------------------------------------------


def test_hook_sourced_absent_is_info_and_does_not_fail_healthy():
    check = doctor_checks.check_hook_sourced({}, "1.2.3")
    assert check["passed"] is False
    assert check["severity"] == "info"
    assert "hooked shell" in check["remediation"]


def test_hook_sourced_matching_version_passes():
    check = doctor_checks.check_hook_sourced({"NVSH_HOOK_VERSION": "1.2.3"}, "1.2.3")
    assert check["passed"] is True


def test_hook_sourced_stale_version_warns_with_setup_remediation():
    check = doctor_checks.check_hook_sourced({"NVSH_HOOK_VERSION": "1.0.0"}, "1.2.3")
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "nvsh setup" in check["remediation"]


# --- hook_first_in_prompt_command --------------------------------------------


def test_hook_first_in_prompt_command_absent_is_info():
    check = doctor_checks.check_hook_first_in_prompt_command(None)
    assert check["passed"] is False
    assert check["severity"] == "info"


def test_hook_first_in_prompt_command_passes_when_first():
    text = 'declare -a PROMPT_COMMAND=([0]="__nvsh_hook" [1]="__ghostty_hook")'
    check = doctor_checks.check_hook_first_in_prompt_command(text)
    assert check["passed"] is True


def test_hook_first_in_prompt_command_fails_with_fix_command_when_moved_last():
    text = 'declare -a PROMPT_COMMAND=([0]="__ghostty_hook" [1]="__nvsh_hook")'
    check = doctor_checks.check_hook_first_in_prompt_command(text)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert 'PROMPT_COMMAND=(__nvsh_hook "${PROMPT_COMMAND[@]/__nvsh_hook}")' in check["remediation"]


#: What ``declare -p PROMPT_COMMAND`` prints once bash-preexec has installed
#: itself on top of an already-hooked shell (measured on a DGX Spark under
#: Ghostty with fig/amazon-q also loaded).
_BASH_PREEXEC_PROMPT_COMMAND = (
    "declare -a PROMPT_COMMAND=("
    "[0]=$'__bp_precmd_invoke_cmd\\n__nvsh_hook\\n:' "
    '[1]="__ghostty_hook" [2]="__bp_interactive_mode")'
)


def test_hook_first_in_prompt_command_accepts_the_bash_preexec_layout():
    check = doctor_checks.check_hook_first_in_prompt_command(_BASH_PREEXEC_PROMPT_COMMAND)
    assert check["passed"] is True
    assert "bash-preexec" in check["message"]
    assert "__bp_precmd_invoke_cmd" in check["message"]


def test_hook_first_in_prompt_command_message_names_the_plain_layout():
    text = 'declare -a PROMPT_COMMAND=([0]="__nvsh_hook" [1]="__ghostty_hook")'
    check = doctor_checks.check_hook_first_in_prompt_command(text)
    assert check["passed"] is True
    assert "first" in check["message"]
    assert "bash-preexec" not in check["message"]


def test_hook_first_in_prompt_command_fails_when_the_hook_appears_twice():
    text = (
        "declare -a PROMPT_COMMAND=("
        "[0]=$'__bp_precmd_invoke_cmd\\n__nvsh_hook' "
        '[1]="__nvsh_hook")'
    )
    check = doctor_checks.check_hook_first_in_prompt_command(text)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "appears 2 times" in check["message"]


def test_hook_first_in_prompt_command_fails_when_preexec_element_lacks_the_hook():
    text = "declare -a PROMPT_COMMAND=([0]=$'__bp_precmd_invoke_cmd\\n:' [1]=\"__nvsh_hook\")"
    check = doctor_checks.check_hook_first_in_prompt_command(text)
    assert check["passed"] is False
    assert check["severity"] == "error"


def test_hook_first_in_prompt_command_handles_string_form():
    text = 'declare -- PROMPT_COMMAND="__nvsh_hook"'
    check = doctor_checks.check_hook_first_in_prompt_command(text)
    assert check["passed"] is True


# --- bindings_present ---------------------------------------------------------

_REAL_BIND_P = r"""
"\C-x\C-g": __nvsh_ctrl_g
"\C-g": "\C-x\C-g\C-j"
"\C-m": "\C-x\C-n\C-j"
"\C-x\C-n": __nvsh_enter
"""

#: What a bash that only exports ``bind -p`` hands over (d4a). Captured from
#: ``bash --norc --noprofile -i`` on a pty with ``nvsh/shell/readline.bash``
#: sourced: the nvsh bindings are all there in the shell, but ``bind -p``
#: lists neither ``bind -x`` functions nor macros, so NONE of the three
#: entries appear -- and ``\C-m`` has vanished from ``bind -p`` entirely
#: because it now holds a macro.
_BIND_P_ONLY_PAYLOAD = r"""
"\C-a": beginning-of-line
"\C-e": end-of-line
"\C-x\C-g": abort
"\C-x\C-r": re-read-init-file
"\C-x\C-u": undo
"""

#: What a bash that exports all three dumps hands over (``bind -p``, then the
#: marker, then ``bind -s`` and ``bind -X``). Captured from the same pty run:
#: note ``bind -X`` quotes the shell command, which the old substring check
#: did not allow for.
_FULL_BIND_PAYLOAD = (
    _BIND_P_ONLY_PAYLOAD
    + doctor_checks.BIND_SECTION_MARKER
    + "\n"
    + '"\\C-m": "\\C-x\\C-n\\C-j"\n'
    + '"\\C-x\\C-g": "__nvsh_ctrl_g"\n"\\C-g": "\\C-x\\C-g\\C-j"\n'
    + '"\\C-x\\C-n": "__nvsh_enter"\n'
)


def test_bindings_present_absent_is_info():
    check = doctor_checks.check_bindings_present(None, "emacs")
    assert check["passed"] is False
    assert check["severity"] == "info"


def test_bindings_present_passes_with_real_bind_p_output():
    check = doctor_checks.check_bindings_present(_REAL_BIND_P, "emacs")
    assert check["passed"] is True


def test_bindings_present_passes_with_the_real_three_dump_payload():
    """d4a: ``bind -X`` quotes the function name; the check must accept it."""
    check = doctor_checks.check_bindings_present(_FULL_BIND_PAYLOAD, "emacs")
    assert check["passed"] is True, check
    assert check["severity"] == "info"


def test_bindings_present_bind_p_only_payload_cannot_verify_and_is_info():
    """d4a: a bind -p-only export proves nothing either way -- never a FAIL."""
    check = doctor_checks.check_bindings_present(_BIND_P_ONLY_PAYLOAD, "emacs")
    assert check["passed"] is False
    assert check["severity"] == "info", check
    assert "bind -x" in check["message"]
    assert "nvsh setup" in check["remediation"]


def test_bindings_present_fails_when_entries_missing():
    text = _BIND_P_ONLY_PAYLOAD + doctor_checks.BIND_SECTION_MARKER + "\n"
    check = doctor_checks.check_bindings_present(text, "vi-insert")
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "vi-insert" in check["message"]


def test_bindings_present_fails_when_only_some_entries_are_bound():
    text = _BIND_P_ONLY_PAYLOAD + '"\\C-x\\C-n": "__nvsh_enter"\n'
    check = doctor_checks.check_bindings_present(text, "emacs")
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "__nvsh_ctrl_g" in check["message"]


# --- capture_active ------------------------------------------------------------


def test_capture_active_missing_env_outside_hook_is_info():
    # No NVSH_HOOK_VERSION either: this is a plain, un-hooked shell, so the
    # absence of capture is expected, not a problem — info, not warning.
    check = doctor_checks.check_capture_active({})
    assert check["passed"] is False
    assert check["severity"] == "info"


def test_capture_active_missing_log_inside_hook_is_warning():
    # NVSH_HOOK_VERSION present but NVSH_LOG absent: hooked shell that
    # should be capturing but isn't — that is a real warning.
    check = doctor_checks.check_capture_active({"NVSH_HOOK_VERSION": "0.9.2"})
    assert check["passed"] is False
    assert check["severity"] == "warning"


def test_capture_active_script_mode_0600_passes(tmp_path):
    log = tmp_path / "123.log"
    log.write_text("", encoding="utf-8")
    log.chmod(0o600)
    check = doctor_checks.check_capture_active({"NVSH_WRAPPED": "1", "NVSH_LOG": str(log)})
    assert check["passed"] is True
    assert "script:" in check["message"]
    assert str(log) in check["message"]


def test_capture_active_tmux_mode_0600_passes(tmp_path):
    log = tmp_path / "123.log"
    log.write_text("", encoding="utf-8")
    log.chmod(0o600)
    check = doctor_checks.check_capture_active({"TMUX": "/tmp/x,1,0", "NVSH_LOG": str(log)})
    assert check["passed"] is True
    assert "tmux:" in check["message"]


def test_capture_active_wrong_mode_warns(tmp_path):
    log = tmp_path / "123.log"
    log.write_text("", encoding="utf-8")
    log.chmod(0o644)
    check = doctor_checks.check_capture_active({"NVSH_WRAPPED": "1", "NVSH_LOG": str(log)})
    assert check["passed"] is False
    assert check["severity"] == "warning"


# --- daemon_status --------------------------------------------------------------


def test_daemon_status_is_always_info():
    check_running = doctor_checks.check_daemon_status(
        {}, is_running=lambda env: True, socket_path=lambda env: Path("/tmp/nvsh/daemon.sock")
    )
    check_not_running = doctor_checks.check_daemon_status(
        {}, is_running=lambda env: False, socket_path=lambda env: Path("/tmp/nvsh/daemon.sock")
    )
    assert check_running["severity"] == "info"
    assert check_not_running["severity"] == "info"
    assert check_running["passed"] is True
    assert check_not_running["passed"] is True
    assert "/tmp/nvsh/daemon.sock" in check_running["message"]


# --- agent_turn_not_hung (task t19) ----------------------------------------------


def test_agent_turn_not_hung_passes_with_no_active_turn():
    check = doctor_checks.check_agent_turn_not_hung(None, threshold=300.0)
    assert check["passed"] is True
    assert check["severity"] == "info"


def test_agent_turn_not_hung_passes_when_owner_alive_and_within_threshold():
    check = doctor_checks.check_agent_turn_not_hung(
        {"shell": "123", "elapsed": 5.0},
        threshold=300.0,
        pid_gone=lambda shell: False,
    )
    assert check["passed"] is True
    assert check["severity"] == "info"


def test_agent_turn_not_hung_fails_when_owner_pid_is_dead():
    check = doctor_checks.check_agent_turn_not_hung(
        {"shell": "999999", "elapsed": 1.0},
        threshold=300.0,
        pid_gone=lambda shell: True,
    )
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "999999" in check["message"]
    assert check["remediation"] == "nvsh doctor --apply"


def test_agent_turn_not_hung_fails_when_elapsed_exceeds_threshold():
    check = doctor_checks.check_agent_turn_not_hung(
        {"shell": "123", "elapsed": 999.0},
        threshold=300.0,
        pid_gone=lambda shell: False,
    )
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "exceeds" in check["message"]


def test_agent_turn_not_hung_included_in_collect_checks_via_probes():
    """collect_checks wires the check to Probes.active_turn, not a hard-coded call."""
    own_pid = str(os.getpid())
    checks = doctor_checks.collect_checks(
        env={},
        current_version="1.2.3",
        config=Config(),
        config_error=None,
        platform=Platform(kind="generic", values=()),
        which=lambda name: None,
        run=lambda argv, timeout: (1, "", ""),
        probes=doctor_checks.Probes(active_turn=lambda env: {"shell": own_pid, "elapsed": 1.0}),
    )
    check = _check(checks, "agent_turn_not_hung")
    # own_pid is this test process's own pid -- always alive -- so the check
    # falls through to the elapsed-vs-threshold comparison and passes.
    assert check["passed"] is True
    assert own_pid in check["message"]


# --- terminfo_present -----------------------------------------------------------


def test_terminfo_present_passes_when_infocmp_succeeds():
    check = doctor_checks.check_terminfo_present(
        {"TERM": "xterm-256color"}, run=lambda argv, timeout: (0, "", "")
    )
    assert check["passed"] is True


def test_terminfo_present_xterm_ghostty_missing_gives_exact_tic_remediation(tmp_path):
    # Isolate from this dev box's real /usr/share/terminfo (which may
    # genuinely have xterm-ghostty installed) by pointing the search dirs
    # at an empty tree, matching "a host without xterm-ghostty terminfo".
    empty = tmp_path / "terminfo"
    empty.mkdir()
    check = doctor_checks.check_terminfo_present(
        {
            "TERM": "xterm-ghostty",
            "HOSTNAME": "thor",
            "TERMINFO_DIRS": str(empty),
            "HOME": str(tmp_path),
        },
        run=lambda argv, timeout: (1, "", "infocmp: couldn't open terminfo file"),
    )
    assert check["passed"] is False
    assert check["remediation"] == "infocmp -x xterm-ghostty | ssh thor -- tic -x -"


def test_terminfo_present_finds_entry_via_search_dirs(tmp_path):
    terminfo = tmp_path / "share" / "terminfo"
    (terminfo / "x").mkdir(parents=True)
    (terminfo / "x" / "xterm-custom").write_text("x", encoding="utf-8")
    check = doctor_checks.check_terminfo_present(
        {"TERM": "xterm-custom", "TERMINFO_DIRS": str(terminfo)},
        run=lambda argv, timeout: (1, "", "no infocmp"),
    )
    assert check["passed"] is True


# --- collect_checks -------------------------------------------------------------


def test_collect_checks_returns_every_new_check_id(tmp_path):
    checks = doctor_checks.collect_checks(
        env={},
        current_version="1.2.3",
        config=Config(),
        config_error=None,
        platform=Platform(kind="generic", values=()),
        which=lambda name: None,
        run=lambda argv, timeout: (1, "", ""),
        home=tmp_path,
    )
    ids = {c["id"] for c in checks}
    assert ids == {
        "platform_detected",
        "agent_configured",
        "agent_reachable",
        "default_target_not_demo",
        "agent_allowlist",
        "hook_sourced",
        "hook_first_in_prompt_command",
        "bindings_present",
        "capture_active",
        "daemon_status",
        "agent_turn_not_hung",
        "terminfo_present",
    }
    for check in checks:
        assert set(check) == {"id", "passed", "severity", "message", "remediation"}
        assert check["severity"] in ("error", "warning", "info")


# --- agent_reachable: where the bearer came from (deviation d10) -------------


def _bearer_key_file(path, value: str = "doctor-bearer-value", mode: int = 0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def test_agent_reachable_names_api_key_file_as_the_bearer_source(tmp_path):
    server, thread, base_url = _serve(200)
    try:
        key_file = _bearer_key_file(tmp_path / "keys" / "api_key")
        cfg = Config(
            agent_provider="openai-compat",
            agents={"openai-compat": {"base_url": base_url, "api_key_file": str(key_file)}},
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert check["message"] == (
            "endpoint reachable (base_url from [agents.openai-compat], " "bearer from api_key_file)"
        )
        _assert_no_endpoint_in_check(check, base_url)
        text = _all_check_text(check)
        assert "doctor-bearer-value" not in text
        assert str(tmp_path) not in text
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_names_the_default_key_file_as_the_bearer_source(tmp_path):
    from nvsh.config import default_key_file

    server, thread, base_url = _serve(200)
    try:
        _bearer_key_file(default_key_file())
        cfg = Config(
            agent_provider="openai-compat", agents={"openai-compat": {"base_url": base_url}}
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert check["message"] == (
            "endpoint reachable (base_url from [agents.openai-compat], "
            "bearer from the default key file)"
        )
        _assert_no_endpoint_in_check(check, base_url)
        assert str(tmp_path) not in _all_check_text(check)
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_says_no_bearer_configured_when_there_is_none():
    server, thread, base_url = _serve(401)
    try:
        cfg = Config(
            agent_provider="openai-compat", agents={"openai-compat": {"base_url": base_url}}
        )
        check = doctor_checks.check_agent_reachable(cfg)
        assert "no bearer configured" in check["message"]
        # The remediation says where to put a key, in placeholder form only.
        assert "api_key" in check["remediation"]
        _assert_no_endpoint_in_check(check, base_url)
    finally:
        server.shutdown()
        thread.join()


def test_agent_reachable_reports_a_refused_key_file_without_the_key(tmp_path):
    server, thread, base_url = _serve(401)
    try:
        key_file = _bearer_key_file(tmp_path / "keys" / "api_key", mode=0o644)
        cfg = Config(
            agent_provider="openai-compat",
            agents={"openai-compat": {"base_url": base_url, "api_key_file": str(key_file)}},
        )
        check = doctor_checks.check_agent_reachable(cfg)
        text = _all_check_text(check)
        assert "0644" in text
        assert "doctor-bearer-value" not in text
        assert str(tmp_path) not in text
    finally:
        server.shutdown()
        thread.join()


# ---------------------------------------------------------------------------
# agent_reachable: per-harness CLI dispatch (task t17) -----------------------
#
# claude/codex/qwen/qwen-p/agy/kiro are driven as plain subprocess CLIs
# instead of the pi-rpc/openai-compat-http probes above. Every fake ``run``
# here is a plain Python callable -- no real binary is ever spawned and no
# network or model call happens, per this task's own instruction.
# ---------------------------------------------------------------------------


def _fake_run(responses: dict[tuple[str, ...], tuple[int, str, str]]):
    """A CliRunner fake keyed by the exact argv tuple; unlisted argv errors loudly."""

    def run(argv: list[str], timeout: float) -> tuple[int, str, str]:
        key = tuple(argv)
        if key not in responses:
            raise AssertionError(f"unexpected CLI invocation in test: {argv!r}")
        return responses[key]

    return run


def test_agent_reachable_cli_harness_missing_binary():
    cfg = Config(agent_provider="claude")
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: None)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "claude-missing" in check["message"]
    assert "nvsh agent install claude" in check["remediation"]


def test_agent_reachable_claude_version_ok_reports_reachable():
    cfg = Config(agent_provider="claude")
    run = _fake_run({("claude", "--version"): (0, _fixture_text("claude-version.txt"), "")})
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/claude", run=run)
    assert check["passed"] is True
    assert check["severity"] == "info"
    assert "2.1.270" in check["message"]
    assert "not verified" in check["message"]  # claude has no verified auth probe (t17 honesty)


def test_agent_reachable_codex_version_ok():
    cfg = Config(agent_provider="codex")
    run = _fake_run({("codex", "--version"): (0, _fixture_text("codex-version.txt"), "")})
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/codex", run=run)
    assert check["passed"] is True
    assert "0.147.0" in check["message"]


def test_agent_reachable_qwen_version_ok():
    cfg = Config(agent_provider="qwen")
    run = _fake_run({("qwen", "--version"): (0, _fixture_text("qwen-version.txt"), "")})
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/qwen", run=run)
    assert check["passed"] is True
    assert "0.23.3" in check["message"]


def test_agent_reachable_qwen_p_shares_the_qwen_binary_and_range():
    cfg = Config(agent_provider="qwen-p")
    run = _fake_run({("qwen", "--version"): (0, _fixture_text("qwen-version.txt"), "")})
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/qwen", run=run)
    assert check["passed"] is True
    assert "0.23.3" in check["message"]


def test_agent_reachable_agy_version_ok():
    cfg = Config(agent_provider="agy")
    run = _fake_run({("agy", "--version"): (0, _fixture_text("agy-version.txt"), "")})
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/agy", run=run)
    assert check["passed"] is True
    assert "1.2.2" in check["message"]


def test_agent_reachable_below_supported_range_warns():
    cfg = Config(agent_provider="agy")
    run = _fake_run({("agy", "--version"): (0, "0.9.0\n", "")})
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/agy", run=run)
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "0.9.0" in check["message"]
    assert "supported range" in check["message"]
    assert "install a supported agy version" in check["remediation"]


def test_agent_reachable_unparseable_version_is_a_warning():
    cfg = Config(agent_provider="claude")
    run = _fake_run({("claude", "--version"): (1, "", "unexpected output")})
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/claude", run=run)
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "could not determine claude version" in check["message"]


def test_agent_reachable_cli_hang_is_a_failed_check_with_remediation():
    def run(argv, timeout):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    cfg = Config(agent_provider="codex")
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/codex", run=run)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "codex-hung" in check["message"]
    assert "hanging" in check["remediation"]
    assert "codex --version" in check["remediation"]


def test_agent_reachable_cli_oserror_is_a_failed_check():
    def run(argv, timeout):
        raise OSError("no such file or directory")

    cfg = Config(agent_provider="claude")
    check = doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/claude", run=run)
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "claude-unreachable" in check["message"]


# --- agent_reachable: kiro's verified 'kiro-cli whoami' auth probe ----------


def test_agent_reachable_kiro_unauthenticated_fails_with_login_remediation():
    cfg = Config(agent_provider="kiro")
    run = _fake_run(
        {
            ("kiro-cli", "--version"): (0, _fixture_text("kiro-version.txt"), ""),
            ("kiro-cli", "whoami"): (0, _fixture_text("kiro-whoami-unauthenticated.txt"), ""),
        }
    )
    check = doctor_checks.check_agent_reachable(
        cfg, which=lambda name: "/usr/bin/kiro-cli", run=run
    )
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "not authenticated" in check["message"]
    assert check["remediation"] == "kiro-cli login"


def test_agent_reachable_kiro_authenticated_passes():
    cfg = Config(agent_provider="kiro")
    run = _fake_run(
        {
            ("kiro-cli", "--version"): (0, _fixture_text("kiro-version.txt"), ""),
            ("kiro-cli", "whoami"): (0, _fixture_text("kiro-whoami-authenticated.txt"), ""),
        }
    )
    check = doctor_checks.check_agent_reachable(
        cfg, which=lambda name: "/usr/bin/kiro-cli", run=run
    )
    assert check["passed"] is True
    assert check["severity"] == "info"
    # kiro has a verified probe, so (unlike claude/codex/qwen/agy) the message
    # does not carry the "not verified" auth disclaimer.
    assert "not verified" not in check["message"]


def test_agent_reachable_kiro_auth_probe_hang_is_a_failed_check():
    def run(argv, timeout):
        if argv == ["kiro-cli", "--version"]:
            return 0, _fixture_text("kiro-version.txt"), ""
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    cfg = Config(agent_provider="kiro")
    check = doctor_checks.check_agent_reachable(
        cfg, which=lambda name: "/usr/bin/kiro-cli", run=run
    )
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "kiro-hung" in check["message"]


def test_agent_reachable_cli_harness_never_calls_run_more_than_documented():
    """No network access and no model call: the fake run() only ever answers
    --version / whoami-shaped argv, so anything else raises inside _fake_run."""
    cfg = Config(agent_provider="claude")
    run = _fake_run({("claude", "--version"): (0, _fixture_text("claude-version.txt"), "")})
    doctor_checks.check_agent_reachable(cfg, which=lambda name: "/usr/bin/claude", run=run)
    # No assertion needed beyond "did not raise": _fake_run raises
    # AssertionError itself on any unexpected argv (e.g. a prompt/model call).


def test_agent_reachable_unknown_cli_provider_still_falls_back_to_the_generic_message():
    cfg = Config(agent_provider="not-a-real-adapter")
    check = doctor_checks.check_agent_reachable(cfg)
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert "has no reachability probe" in check["message"]


# ---------------------------------------------------------------------------
# agent_allowlist (task t17) --------------------------------------------------
# ---------------------------------------------------------------------------


def test_agent_allowlist_reports_info_when_nothing_is_found(tmp_path):
    check = doctor_checks.check_agent_allowlist(home=tmp_path)
    assert check["id"] == "agent_allowlist"
    assert check["passed"] is True
    assert check["severity"] == "info"


def test_agent_allowlist_warns_and_names_the_claude_settings_file(tmp_path):
    dest = tmp_path / ".claude" / "settings.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(
        (FIXTURES / "claude-settings-with-allow.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    check = doctor_checks.check_agent_allowlist(home=tmp_path)
    assert check["passed"] is False
    assert check["severity"] == "warning"
    assert str(dest) in check["message"]


def test_agent_allowlist_warns_and_names_the_agy_settings_file(tmp_path):
    dest = tmp_path / ".gemini" / "antigravity-cli" / "settings.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(
        (FIXTURES / "agy-settings-with-allow.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    check = doctor_checks.check_agent_allowlist(home=tmp_path)
    assert check["passed"] is False
    assert str(dest) in check["message"]


def test_agent_allowlist_warns_and_names_the_codex_config_file(tmp_path):
    dest = tmp_path / ".codex" / "config.toml"
    dest.parent.mkdir(parents=True)
    dest.write_text(
        (FIXTURES / "codex-config-with-approval-policy.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    check = doctor_checks.check_agent_allowlist(home=tmp_path)
    assert check["passed"] is False
    assert str(dest) in check["message"]


def test_agent_allowlist_reports_multiple_files_together(tmp_path):
    claude_dest = tmp_path / ".claude" / "settings.json"
    claude_dest.parent.mkdir(parents=True)
    claude_dest.write_text(
        (FIXTURES / "claude-settings-with-allow.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    codex_dest = tmp_path / ".codex" / "config.toml"
    codex_dest.parent.mkdir(parents=True)
    codex_dest.write_text(
        (FIXTURES / "codex-config-with-approval-policy.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    check = doctor_checks.check_agent_allowlist(home=tmp_path)
    assert check["passed"] is False
    assert str(claude_dest) in check["message"]
    assert str(codex_dest) in check["message"]


def test_agent_allowlist_ignores_an_empty_allow_list(tmp_path):
    dest = tmp_path / ".claude" / "settings.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps({"permissions": {"allow": []}}), encoding="utf-8")
    check = doctor_checks.check_agent_allowlist(home=tmp_path)
    assert check["passed"] is True


def test_agent_allowlist_ignores_malformed_json(tmp_path):
    dest = tmp_path / ".claude" / "settings.json"
    dest.parent.mkdir(parents=True)
    dest.write_text("{not json", encoding="utf-8")
    check = doctor_checks.check_agent_allowlist(home=tmp_path)
    assert check["passed"] is True


def test_agent_allowlist_never_writes_to_any_file(tmp_path):
    dest = tmp_path / ".claude" / "settings.json"
    dest.parent.mkdir(parents=True)
    original = (FIXTURES / "claude-settings-with-allow.json").read_text(encoding="utf-8")
    dest.write_text(original, encoding="utf-8")
    mtime_before = dest.stat().st_mtime_ns

    doctor_checks.check_agent_allowlist(home=tmp_path)

    assert dest.read_text(encoding="utf-8") == original
    assert dest.stat().st_mtime_ns == mtime_before


def test_agent_allowlist_is_included_in_collect_checks(tmp_path):
    checks = doctor_checks.collect_checks(
        env={},
        current_version="1.2.3",
        config=Config(),
        config_error=None,
        platform=Platform(kind="generic", values=()),
        which=lambda name: None,
        home=tmp_path,
    )
    check = _check(checks, "agent_allowlist")
    assert check["passed"] is True
