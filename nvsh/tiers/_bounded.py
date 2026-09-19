"""Cutting text to a bound *before* it is handed to :func:`nvsh.redact.redact`.

Both Tier 2 (:mod:`nvsh.tiers.lfm`) and the router (:mod:`nvsh.tiers.router`)
bound a text before redacting it, because ``redact()`` is quadratic on long
unbroken input (40k characters measured at 0.8 s, 160k at 12.7 s) and a model
handing over a megabyte must not stall the request for minutes.

Cutting first is only safe for rules that match a *local* shape: a token
straddling the cut still lies wholly inside the slack the callers add, so the
redactor sees it whole. It is **not** safe for the one rule in
:mod:`nvsh.redact` that needs a whole multi-line structure -- the private-key
block, which is matched from its ``BEGIN`` marker to its ``END`` marker. A cut
that falls inside such a block leaves the redactor a ``BEGIN`` marker with no
``END`` (or, for a tail cut, an ``END`` with no ``BEGIN``); the rule then does
not fire at all and the key material on the kept side of the cut survives into
the local model's context and, on escalation, into the full agent's.

:func:`bounded_cut` closes that hole without giving up the bound: after the
cut it looks for a structure the cut *opened* (or, keeping the tail, one it
*closed*) and replaces that whole dangling run with a short fixed placeholder,
so nothing of the key body reaches the redactor -- or anything after it.

Stdlib only; no third-party dependency (``dependencies = []`` stays empty).
"""

from __future__ import annotations

import re

#: What a private-key block cut in half is replaced by. A fixed string, like
#: every marker in :mod:`nvsh.redact`, so the result stays idempotent and no
#: part of the matched content leaks into it.
CUT_MARKER = "<CUT:private_key_block>"

# Deliberately wider than nvsh.redact's own BEGIN/END alternation: this helper
# must not miss a variant the redactor would have caught (and erring wide only
# ever drops more, never less).
_KEY_BEGIN = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
_KEY_END = re.compile(r"-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----")


def _seal_opened(text: str) -> str:
    """Drop a ``BEGIN`` block the head cut left unclosed, and all that follows."""
    opened = None
    for match in _KEY_BEGIN.finditer(text):
        opened = match
    if opened is None or _KEY_END.search(text, opened.end()):
        return text
    return text[: opened.start()] + CUT_MARKER


def _seal_closed(text: str) -> str:
    """Drop an ``END`` the tail cut left unopened, and all that precedes it."""
    closed = _KEY_END.search(text)
    if closed is None:
        return text
    opened = _KEY_BEGIN.search(text)
    if opened is not None and opened.start() < closed.start():
        return text
    return CUT_MARKER + text[closed.end() :]


def bounded_cut(raw: str, window: int, *, tail: bool = False) -> str:
    """The first (or last) *window* characters of *raw*, safe to redact.

    ``tail=True`` keeps the end of the text rather than its start, which is
    what a failed command's output needs: the error is at the bottom. A
    private-key block the cut split is replaced by :data:`CUT_MARKER` before
    the text is returned, so the redactor never has to reassemble it and no
    key body survives on the kept side.
    """
    if len(raw) <= window:
        return raw
    if tail:
        return _seal_closed(raw[-window:])
    return _seal_opened(raw[:window])
