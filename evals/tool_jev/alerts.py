"""Progress, spend and finish alerts for the unattended gate driver (issue #64).

The driver (:mod:`evals.tool_jev.drive`) calls :func:`notify` after every
step with the run's :func:`evals.tool_jev.run.status` document. When an
alert webhook is configured it posts one short message per new event:

- **progress**: each provider crossing another 10% of its calls answered
  (done + invalid over every registered call; the denominator grows as the
  Track A loop registers later rounds, so a milestone is only ever sent the
  first time it is crossed);
- **spend**: total spend crossing another whole dollar ($1, $2, ...), with
  each provider's spend so far;
- **stops**: a provider or model stop the first time it appears (money,
  budget cap, truncation, rejected request, uncertain attempts, failed
  batches), and the run's end: complete, or stopped to ask the operator.

Only counts, dollars, provider and model names and stop reasons are sent,
never case text, prompts or answers. The webhook URL is a secret: it is
read at call time from the environment variable named by
:data:`ENV_WEBHOOK` (inject it with ``grant run --inject``), never from the
manifest or any committed file. Every milestone already sent is recorded
in ``<run_dir>/alerts.json`` (atomic write), so a driver restart never
sends it again. A failed post is logged by the caller and retried at the
next step; it never stops or slows the run.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

#: Environment variable holding the Discord-compatible webhook URL.
ENV_WEBHOOK = "NVSH_EVALS_ALERT_WEBHOOK"
ALERTS_FILE = "alerts.json"
PROGRESS_STEP = 10  # percent
SPEND_STEP = 1.0  # US dollars
#: Discord rejects a message longer than 2000 characters.
MAX_MESSAGE = 1900

Poster = Callable[[str, str], None]


def post_webhook(url: str, text: str) -> None:
    """POST ``{"content": text}`` to a Discord-style webhook (stdlib only)."""
    if not url.startswith("https://"):
        raise ValueError("alert webhook must be an https URL")
    body = json.dumps({"content": text[:MAX_MESSAGE]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "nvsh-evals-gate"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310 -- https only
        response.read()


def _load(run_dir: Path) -> dict:
    path = run_dir / ALERTS_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {}
    doc.setdefault("progress", {})
    doc.setdefault("spend_dollars", 0)
    doc.setdefault("stops", [])
    doc.setdefault("finished", None)
    return doc


def _save(run_dir: Path, doc: Mapping[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=ALERTS_FILE + ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, run_dir / ALERTS_FILE)


def _percent(row: Mapping[str, Any]) -> int:
    answered = int(row.get("done", 0)) + int(row.get("invalid", 0))
    total = answered + int(row.get("pending", 0)) + int(row.get("submitted", 0))
    return int(100 * answered / total) if total else 0


def events(doc: Mapping[str, Any], sent: Mapping[str, Any]) -> tuple[list[str], dict]:
    """The new alert lines for status *doc*, and the updated *sent* record."""
    record = json.loads(json.dumps(sent))
    lines: list[str] = []
    run = f"run {doc.get('run_id', '?')}"
    providers = doc.get("providers", {})
    for kind, row in sorted(providers.items()):
        percent = _percent(row)
        milestone = (percent // PROGRESS_STEP) * PROGRESS_STEP
        if milestone > int(record["progress"].get(kind, 0)):
            record["progress"][kind] = milestone
            answered = int(row.get("done", 0)) + int(row.get("invalid", 0))
            lines.append(
                f"{run}: {kind} {milestone}% ({answered} answered, "
                f"{int(row.get('pending', 0)) + int(row.get('submitted', 0))} to go, "
                f"{int(row.get('invalid', 0))} invalid)"
            )
    total = sum(float(row.get("spend_usd", 0.0)) for row in providers.values())
    dollars = int(total // SPEND_STEP)
    if dollars > int(record["spend_dollars"]):
        record["spend_dollars"] = dollars
        parts = ", ".join(
            f"{kind} ${float(row.get('spend_usd', 0.0)):.2f}/${float(row.get('usd_cap', 0.0)):.0f}"
            for kind, row in sorted(providers.items())
            if float(row.get("spend_usd", 0.0)) or float(row.get("usd_cap", 0.0))
        )
        lines.append(f"{run}: spend passed ${dollars} (total ${total:.2f}; {parts})")
    stops = doc.get("stops", {})
    for scope in ("providers", "models"):
        for name, stop in sorted(stops.get(scope, {}).items()):
            reason = str(stop.get("reason") or stop.get("kind") or "stopped")
            marker = f"{scope}:{name}:{reason}"
            if marker in record["stops"]:
                continue
            record["stops"].append(marker)
            lines.append(f"{run}: STOP {name}: {reason}")
    status = doc.get("status")
    if status in ("complete", "ask") and record["finished"] != status:
        record["finished"] = status
        if status == "complete":
            lines.append(f"{run}: COMPLETE (total spend ${total:.2f}); result.json and report.md")
        else:
            lines.append(f"{run}: STOPPED TO ASK the operator; see drive.log")
    return lines, record


def notify(
    run_dir: Path,
    doc: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    poster: Poster = post_webhook,
    finished: str | None = None,
) -> list[str]:
    """Post every new alert for *doc*; the lines posted (none when no webhook is set).

    *finished* ("complete" or "ask") overrides the status document's own
    status for the driver's last step. A post that fails raises, and the
    milestones it carried stay unsent, so the next step retries them.
    """
    url = (env if env is not None else os.environ).get(ENV_WEBHOOK, "")
    if not url:
        return []
    run_dir = Path(run_dir)
    sent = _load(run_dir)
    if finished is not None:
        doc = {**doc, "status": finished}
    lines, record = events(doc, sent)
    if not lines:
        return []
    poster(url, "\n".join(lines))
    _save(run_dir, record)
    return lines
