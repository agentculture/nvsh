"""``import nvsh.cli`` must stay stdlib-only.

CLAUDE.md: "The runtime package has no third-party dependencies
(``dependencies = []``)... a login shell has to start fast and must not
break when a venv or wheel breaks." This test enforces that at import time,
not just in ``pyproject.toml``: it runs ``python -X importtime -c "import
nvsh.cli"`` in a real subprocess and asserts every module it loads is either
stdlib, part of ``nvsh`` itself, or a well-known interpreter/venv shim.
"""

from __future__ import annotations

import subprocess
import sys

#: Modules an interpreter or virtualenv activation may legitimately inject
#: before/around the import, even though they are not in
#: ``sys.stdlib_module_names`` and are not part of ``nvsh``.
_INTERPRETER_SHIMS = {
    "_virtualenv",
    "sitecustomize",
    "usercustomize",
    "_distutils_hack",
}


def _importtime_module_names(output: str) -> list[str]:
    """Parse ``python -X importtime`` stderr output into top-level module names."""
    names = []
    for line in output.splitlines():
        if not line.startswith("import time:"):
            continue
        # "import time:     self us |   cumulative us |   module.name"
        _, _, rest = line.partition(":")
        fields = rest.split("|")
        assert len(fields) == 3, f"unexpected importtime line shape: {line!r}"
        module = fields[2].strip()
        if module and module != "imported package":
            names.append(module)
    return names


def test_import_nvsh_cli_loads_only_stdlib_and_nvsh() -> None:
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import nvsh.cli"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"import failed: {proc.stderr}"

    modules = _importtime_module_names(proc.stderr)
    assert modules, "no importtime lines parsed — is -X importtime output format unchanged?"

    stdlib = set(sys.stdlib_module_names)
    offenders = []
    for name in modules:
        top = name.split(".")[0]
        if top in stdlib:
            continue
        if name == "nvsh" or name.startswith("nvsh."):
            continue
        if top in _INTERPRETER_SHIMS:
            continue
        offenders.append(name)

    assert not offenders, (
        "import nvsh.cli pulled in non-stdlib, non-nvsh module(s): " f"{sorted(set(offenders))}"
    )
