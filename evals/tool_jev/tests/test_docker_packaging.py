"""The autonomous driver's packaging (issue #64, deviation d2).

Deviation d2: a docker compose service with a pinned python:3.12-slim + uv
image, host networking for the localhost lobes gateway, the private data
mounted read-only and the run directory read-write, restart unless-stopped,
bounded logs, deepeval telemetry off, and keys passed through from grant by
NAME only -- no key value and no key file anywhere in the repository.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # nosec B404 -- runs only `docker compose config`, fixed argv
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
DOCKER = REPO / "evals" / "docker"
KEY_NAMES = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "NGC_API_KEY",
    "LOBES_GATEWAY_API_KEY",
    "NVSH_EVALS_ALERT_WEBHOOK",
}


def _service():
    compose = yaml.safe_load((DOCKER / "compose.yaml").read_text())
    return compose["services"]["drive"]


def test_image_is_pinned_and_runs_the_driver():
    text = (DOCKER / "Dockerfile").read_text()
    assert re.search(r"^FROM python:3\.12-slim@sha256:[0-9a-f]{64}$", text, re.M)
    assert re.search(r"^COPY --from=ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+ ", text, re.M)
    assert "uv sync --frozen --no-dev --group evals" in text
    assert "DEEPEVAL_TELEMETRY_OPT_OUT=1" in text and "DEEPEVAL_DISABLE_DOTENV=1" in text
    assert '"-m", "evals.tool_jev"' in text
    assert 'CMD ["drive"]' in text
    # No private data or secrets are ever copied into the image.
    for line in text.splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            assert not any(word in line for word in ("private", "lfm-train", ".env", "manifest"))


def test_service_survives_restarts_and_reaches_the_local_gateway():
    service = _service()
    assert service["restart"] == "unless-stopped"
    assert service["network_mode"] == "host"
    assert service["command"][0] == "drive"
    assert "--start" in service["command"] and "--idle-when-done" in service["command"]
    assert service["logging"]["options"]["max-size"]


def test_keys_are_passed_by_name_only():
    environment = _service()["environment"]
    for name in KEY_NAMES:
        assert name in environment
        assert environment[name] is None, f"{name} must carry no value in the compose file"
    assert not (DOCKER / ".env").exists()
    assert "env_file" not in _service()


def test_private_data_is_read_only_and_the_run_dir_is_writable():
    mounts = {volume["target"]: volume for volume in _service()["volumes"]}
    assert mounts["/private"]["read_only"] is True
    assert not mounts["/run-dir"].get("read_only", False)
    # Host paths come from the operator's environment, never a literal path.
    for volume in mounts.values():
        assert volume["source"].startswith("${NVSH_EVALS_")


def test_build_context_excludes_private_and_vcs_state():
    ignored = (REPO / ".dockerignore").read_text().split()
    for entry in (".git", ".devague", ".deepeval", ".venv"):
        assert entry in ignored


def test_the_drive_verb_exists_in_the_cli():
    from evals.tool_jev import __main__ as cli

    source = Path(cli.__file__).read_text()
    assert '"drive"' in source


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_file_is_valid(tmp_path):
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "NVSH_EVALS_PRIVATE_ROOT": str(tmp_path / "private"),
        "NVSH_EVALS_RUN_DIR": str(tmp_path / "run"),
    }
    result = subprocess.run(  # nosec B603 B607 -- fixed argv, no model content
        ["docker", "compose", "-f", str(DOCKER / "compose.yaml"), "config", "--quiet"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    if result.returncode != 0 and "Cannot connect" in result.stderr:
        pytest.skip("docker daemon unavailable")
    assert result.returncode == 0, result.stderr
