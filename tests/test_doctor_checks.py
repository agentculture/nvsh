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
import socket
import threading
from pathlib import Path
from urllib.parse import urlsplit

from nvsh import doctor_checks
from nvsh.config import Config
from nvsh.platform import Platform, Value


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


def test_agent_configured_fails_when_config_toml_failed_to_load():
    check = doctor_checks.check_agent_configured(None, "malformed TOML")
    assert check["passed"] is False
    assert check["severity"] == "error"
    assert "malformed TOML" in check["message"]


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
        assert check["message"] == "endpoint reachable (base_url from [agents.openai-compat])"
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
"\C-g": __nvsh_ctrl_g
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
    + '"\\C-g": "__nvsh_ctrl_g"\n'
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


def test_collect_checks_returns_every_new_check_id():
    checks = doctor_checks.collect_checks(
        env={},
        current_version="1.2.3",
        config=Config(),
        config_error=None,
        platform=Platform(kind="generic", values=()),
        which=lambda name: None,
        run=lambda argv, timeout: (1, "", ""),
    )
    ids = {c["id"] for c in checks}
    assert ids == {
        "platform_detected",
        "agent_configured",
        "agent_reachable",
        "hook_sourced",
        "hook_first_in_prompt_command",
        "bindings_present",
        "capture_active",
        "daemon_status",
        "terminfo_present",
    }
    for check in checks:
        assert set(check) == {"id", "passed", "severity", "message", "remediation"}
        assert check["severity"] in ("error", "warning", "info")
