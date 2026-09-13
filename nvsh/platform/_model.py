"""Dataclasses shared by platform detection: ``Value`` and ``Platform``."""

from __future__ import annotations

from dataclasses import dataclass, field

#: Where a Value's content came from.
FILE = "file"
SUBPROCESS = "subprocess"
PATH = "path"

_METHODS = frozenset({FILE, SUBPROCESS, PATH})


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
class Platform:
    """The detected machine: a ``kind`` plus every value checked for it."""

    kind: str
    values: tuple[Value, ...] = field(default_factory=tuple)

    def get(self, name: str) -> Value | None:
        for value in self.values:
            if value.name == name:
                return value
        return None

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
