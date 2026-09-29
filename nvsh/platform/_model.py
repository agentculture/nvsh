"""Dataclasses shared by platform detection: ``Value``, ``Platform`` and
``DeviceCli``."""

from __future__ import annotations

from dataclasses import dataclass, field

#: Where a Value's content came from.
FILE = "file"
SUBPROCESS = "subprocess"
PATH = "path"

_METHODS = frozenset({FILE, SUBPROCESS, PATH})

#: Where a device CLI (``spark``/``thor``/``orin``) was found: nvsh's own env
#: bin directory (``Path(sys.executable).parent``, where an extra such as
#: ``nvsh[orin]`` installs it) or ``PATH``. Own env is checked first.
NVSH_ENV = "nvsh-env"
ON_PATH = "path"

#: The ``<cli>_cli`` value's ``source`` for each origin, and when absent.
_DEVICE_CLI_SOURCES = {NVSH_ENV: "{cli} (nvsh env)", ON_PATH: "{cli} (PATH)"}
_DEVICE_CLI_ABSENT_SOURCE = "{cli} (nvsh env, PATH)"


def device_cli_source(cli: str, origin: str | None) -> str:
    """The ``source`` string of a ``<cli>_cli`` value found at *origin*
    (``None`` = found nowhere)."""
    template = _DEVICE_CLI_SOURCES[origin] if origin else _DEVICE_CLI_ABSENT_SOURCE
    return template.format(cli=cli)


@dataclass(frozen=True)
class Value:
    """One detected (or checked-but-absent) fact about the platform.

    ``present`` is False when the source was checked and nothing usable was
    found there (file missing, binary not on PATH, content unparseable).
    ``text`` is None in that case; ``source`` and ``method`` are always
    filled in, so the caller can see what was checked even when nothing was
    found. The field is ``text``, not ``value`` (a field may not repeat its
    own class's name); the serialized key stays ``"value"``.
    """

    name: str
    text: str | None
    source: str
    method: str
    present: bool

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"unknown Value.method {self.method!r} for {self.name!r}")
        if self.present and self.text is None:
            raise ValueError(f"Value {self.name!r} is present but value is None")
        if not self.present and self.text is not None:
            raise ValueError(f"Value {self.name!r} is absent but value is not None")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.text,
            "source": self.source,
            "method": self.method,
            "present": self.present,
        }


@dataclass(frozen=True)
class DeviceCli:
    """A device CLI detection found: its name, resolved path, origin and version.

    ``origin`` is :data:`NVSH_ENV` or :data:`ON_PATH`; ``version`` is the
    last token of ``<cli> --version`` (e.g. ``"0.5.0"``) or ``None`` when that
    call failed or printed nothing parseable.
    """

    name: str
    path: str
    origin: str
    version: str | None


@dataclass(frozen=True)
class Platform:
    """The detected machine: a ``kind`` plus every value checked for it."""

    kind: str
    values: tuple[Value, ...] = field(default_factory=tuple)

    def get(self, name: str) -> Value | None:
        for value in self.values:
            if value.name == name:
                return value
        return None

    def board(self) -> str | None:
        """``"thor"`` or ``"orin"`` from ``/proc/device-tree/model``; ``None``
        when the model is unreadable or names neither (callers then fall back
        to PATH order)."""
        value = self.get("jetson_board")
        return value.text if value is not None and value.present else None

    def device_cli(self, cli: str) -> DeviceCli | None:
        """The device CLI *cli* (``spark``/``thor``/``orin``) if detected."""
        value = self.get(f"{cli}_cli")
        if value is None or not value.present or value.text is None:
            return None
        origin = NVSH_ENV if value.source == device_cli_source(cli, NVSH_ENV) else ON_PATH
        version = self.get(f"{cli}_cli_version")
        return DeviceCli(
            name=cli,
            path=value.text,
            origin=origin,
            version=version.text if version is not None and version.present else None,
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "values": [value.to_dict() for value in self.values],
        }

    def render_block(self) -> str:
        """Compact text block: every value with its source, absent or not.

        Used by the (future) failure panel so an operator or agent sees
        exactly what nvsh detected and where each fact came from, in one
        glance.
        """
        lines = [f"platform: {self.kind}"]
        for value in self.values:
            shown = value.text if value.present else "absent"
            lines.append(f"  {value.name}: {shown}  [{value.method}: {value.source}]")
        return "\n".join(lines)
