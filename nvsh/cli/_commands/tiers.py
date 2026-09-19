"""``nvsh tiers`` — inspect, export and prefetch the local response tiers.

Task t20. Sits alongside :mod:`nvsh.tiers.records` (the append-only JSONL
measurement log) and :mod:`nvsh.tiers.fetch` (pinned engine/weights/image
prefetch), giving an operator a CLI view over both without importing either
module at ``nvsh.cli`` startup — :mod:`nvsh.tiers` and its submodules are
imported lazily, inside each handler, per the stdlib-only import-time
contract in ``CLAUDE.md``.

* ``nvsh tiers stats``   — per-tier counts, latency p50/p95, an
  escalation/decline-reason histogram and operator approve/decline rates,
  aggregated by :func:`nvsh.tiers.stats.compute_stats` from
  ``TierRecords.read_all()``.
* ``nvsh tiers export <file>`` — writes a redacted bundle (records
  re-redacted on the way out, plus the nvsh version and a platform summary)
  to a local path only. Refuses anything that looks like a URL or a remote
  scp-style target, creates the file mode ``0600``, and never opens a
  socket.
* ``nvsh tiers prefetch`` — shows what :func:`nvsh.tiers.fetch.plan_prefetch`
  says would be fetched, with sizes, and asks before downloading anything:
  ``--yes`` downloads without asking; off a terminal or under ``--json``
  without ``--yes`` it refuses outright (a CliError naming ``--yes``, never
  a silent download); on an interactive terminal it prompts per item. With nothing missing it just reports.

Sub-subparser layout (``tiers <verb>``) matches ``nvsh agent`` — a later
``nvsh tiers bench`` verb (task t22) is one more ``noun_sub.add_parser`` call
away.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from nvsh.cli._errors import EXIT_USER_ERROR, CliError
from nvsh.cli._output import emit_result

#: Help text every ``--json`` flag in this verb group shares.
_JSON_HELP = "Emit structured JSON."


def _is_interactive() -> bool:
    """Whether there is a terminal to ask the operator on.

    Mirrors ``nvsh/cli/_commands/agent.py``'s ``_is_interactive``: a closed,
    custom or otherwise unavailable ``stdin`` must decline rather than
    raise.
    """
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _records_cap_bytes(cfg) -> int:
    """``[tiers] records_cap_mb`` (MiB) converted to bytes for ``TierRecords``."""
    records_cap_mb = cfg.tiers.get("records_cap_mb", 8)
    return int(records_cap_mb) * 1024 * 1024


def _open_records():
    """Construct a :class:`nvsh.tiers.records.TierRecords` over the configured path/cap."""
    from nvsh import config as nvsh_config
    from nvsh.tiers.records import TierRecords, default_records_path

    cfg = nvsh_config.load()
    return TierRecords(path=default_records_path(), cap_bytes=_records_cap_bytes(cfg))


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def _stats_text(stats: dict) -> str:
    lines = [f"total records: {stats['total']} (dropped: {stats['dropped']})", ""]
    lines.append("per-tier:")
    if not stats["tiers"]:
        lines.append("  (none)")
    for tier, row in stats["tiers"].items():
        lines.append(
            f"  {tier}: count={row['count']} "
            f"p50={row['latency_p50_ms']:.1f}ms p95={row['latency_p95_ms']:.1f}ms"
        )
    lines.append("")
    lines.append("escalation/decline reasons:")
    if not stats["escalation_reasons"]:
        lines.append("  (none)")
    for reason, count in stats["escalation_reasons"].items():
        lines.append(f"  {reason}: {count}")
    lines.append("")
    decisions = stats["operator_decisions"]
    lines.append(
        "operator decisions: "
        f"approved={decisions['approved']} declined={decisions['declined']} "
        f"approve_rate={decisions['approve_rate']:.2f} decline_rate={decisions['decline_rate']:.2f}"
    )
    return "\n".join(lines)


def cmd_tiers_stats(args: argparse.Namespace) -> int:
    from nvsh.tiers.stats import compute_stats

    records = _open_records()
    entries = records.read_all()
    stats = compute_stats(entries, dropped=records.dropped)

    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(stats, json_mode=True)
    else:
        emit_result(_stats_text(stats), json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _refuse_if_remote(raw: str) -> None:
    """Refuse an export target that looks like a URL or an scp-style remote.

    Anything with ``scheme://`` is a URL. Anything with a colon before the
    first ``/`` (or with no ``/`` at all) reads as ``host:path`` — an scp
    target, not a local path. A bare relative or absolute local path never
    contains a colon before its first slash, so this never rejects one.
    """
    if "://" in raw:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"export target looks like a URL, not a local path: {raw!r}",
            remediation="pass a local file path, e.g. 'nvsh tiers export ./tiers-bundle.json'",
        )
    colon = raw.find(":")
    slash = raw.find("/")
    if colon != -1 and (slash == -1 or colon < slash):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"export target looks like a remote host:path, not a local path: {raw!r}",
            remediation="pass a local file path, e.g. 'nvsh tiers export ./tiers-bundle.json'",
        )


def _redact_value(value: object):
    from nvsh.redact import redact

    if isinstance(value, str):
        return redact(value.encode("utf-8", "replace")).decode("utf-8", "replace")
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


def _platform_summary() -> dict:
    from nvsh import platform as platform_mod

    platform = platform_mod.detect()
    return {"kind": platform.kind}


def _build_export_bundle(entries: list[dict], dropped: int) -> dict:
    from nvsh import __version__

    return {
        "nvsh_version": __version__,
        "platform": _platform_summary(),
        "dropped": dropped,
        "records": [_redact_value(entry) for entry in entries],
    }


def _write_bundle(path, bundle: dict, *, force: bool) -> None:
    import json
    import os

    # O_NOFOLLOW: --force must never write through a symlink someone planted
    # at the destination. fchmod on the open descriptor (not chmod on the
    # path) tightens a pre-existing, wider-mode file without a path race.
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_TRUNC if force else os.O_EXCL)
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{path} already exists",
            remediation="pass --force to overwrite it",
        ) from exc
    except OSError as exc:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"cannot write {path}: {exc.strerror or exc}",
            remediation="pass a writable local file path in an existing directory",
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(bundle, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def cmd_tiers_export(args: argparse.Namespace) -> int:
    from pathlib import Path

    _refuse_if_remote(args.file)
    records = _open_records()
    entries = records.read_all()
    bundle = _build_export_bundle(entries, records.dropped)
    _write_bundle(Path(args.file), bundle, force=bool(getattr(args, "force", False)))

    result = {"path": str(args.file), "records": len(entries)}
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(f"exported {len(entries)} record(s) to {args.file}", json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------


def _yes_confirm(_item) -> bool:
    return True


def _interactive_confirm(item) -> bool:
    try:
        answer = input(f"download {item.name} ({item.size_bytes} bytes)? [y/N] ")
    except (EOFError, OSError):
        return False
    return answer.strip().lower() in ("y", "yes")


def _prefetch_rows(items) -> list[dict]:
    return [dataclasses.asdict(item) for item in items]


def _prefetch_text(rows: list[dict], problems: list[dict]) -> str:
    lines = [
        (
            "still missing:"
            if any(not r["present"] for r in rows)
            else "all pinned tier files are present:"
        )
    ]
    for row in rows:
        status = "present" if row["present"] else "missing"
        lines.append(f"  [{row['kind']}] {row['name']}: {row['size_bytes']} bytes ({status})")
    if problems:
        lines.append("")
        lines.append("problems:")
        for problem in problems:
            lines.append(f"  {problem['item']}: {problem['code']} — {problem['message']}")
    return "\n".join(lines)


def cmd_tiers_prefetch(args: argparse.Namespace) -> int:
    from nvsh.tiers.fetch import plan_prefetch
    from nvsh.tiers.fetch import prefetch as run_prefetch

    items = plan_prefetch()
    rows = _prefetch_rows(items)
    missing = [item for item in items if not item.present]
    json_mode = bool(getattr(args, "json", False))
    yes = bool(getattr(args, "yes", False))

    if missing and not yes and (json_mode or not _is_interactive()):
        total = sum(item.size_bytes for item in missing)
        names = ", ".join(f"{item.name} ({item.size_bytes} bytes)" for item in missing)
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"prefetch would download {len(missing)} item(s), {total} bytes: {names}; "
                "confirmation required"
            ),
            remediation="pass --yes to download without prompting",
        )

    problems = []
    if missing:
        confirm = _yes_confirm if yes else _interactive_confirm
        problems = [dataclasses.asdict(p) for p in run_prefetch(items, confirm=confirm)]
        # Plan again so the report shows what is on disk now, not what was
        # missing before the download.
        rows = _prefetch_rows(plan_prefetch())

    if json_mode:
        emit_result({"items": rows, "problems": problems}, json_mode=True)
    else:
        emit_result(_prefetch_text(rows, problems), json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def _no_verb(args: argparse.Namespace) -> int:
    return cmd_tiers_stats(args)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tiers",
        help="Inspect, export and prefetch the local response tiers (see 'nvsh explain tiers').",
    )
    p.add_argument("--json", action="store_true", help=_JSON_HELP)
    p.set_defaults(func=_no_verb, json=False)
    noun_sub = p.add_subparsers(dest="tiers_command", parser_class=type(p))

    stats = noun_sub.add_parser(
        "stats", help="Per-tier counts, latency percentiles and decision rates."
    )
    stats.add_argument("--json", action="store_true", help=_JSON_HELP)
    stats.set_defaults(func=cmd_tiers_stats)

    export = noun_sub.add_parser("export", help="Write a redacted records bundle to a local file.")
    export.add_argument("file", help="Local destination path (never a URL or remote target).")
    export.add_argument(
        "--force", action="store_true", help="Overwrite the destination if it already exists."
    )
    export.add_argument("--json", action="store_true", help=_JSON_HELP)
    export.set_defaults(func=cmd_tiers_export)

    prefetch = noun_sub.add_parser(
        "prefetch", help="Show what would be fetched, with sizes, and optionally fetch it."
    )
    prefetch.add_argument("--yes", action="store_true", help="Download without prompting.")
    prefetch.add_argument("--json", action="store_true", help=_JSON_HELP)
    prefetch.set_defaults(func=cmd_tiers_prefetch)
    # A later `nvsh tiers bench` verb (task t22) adds one more
    # `noun_sub.add_parser(...)` call here.
