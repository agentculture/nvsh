"""Discord alerts for the unattended driver (issue #64, deviation d5).

Progress every 10% per provider, spend every whole dollar, each stop once,
and the run's end; milestones survive a restart; a failed post is retried;
nothing but counts, dollars, names and reasons is ever sent. No network.
"""

from __future__ import annotations

import pytest

from evals.tool_jev import alerts

URL = "https://discord.example/api/webhooks/1/x"


def _doc(done, pending, spend=0.0, invalid=0, stops=None, status="waiting"):
    return {
        "run_id": "r1",
        "status": status,
        "providers": {
            "openrouter": {
                "done": done,
                "submitted": 0,
                "pending": pending,
                "invalid": invalid,
                "spend_usd": spend,
                "usd_cap": 14.0,
            }
        },
        "stops": stops or {"providers": {}, "models": {}},
    }


class Poster:
    def __init__(self, fail=False):
        self.posts: list[tuple[str, str]] = []
        self.fail = fail

    def __call__(self, url, text):
        if self.fail:
            raise OSError("webhook down")
        self.posts.append((url, text))


def test_no_webhook_means_no_alert(tmp_path):
    poster = Poster()
    assert alerts.notify(tmp_path, _doc(50, 50), env={}, poster=poster) == []
    assert poster.posts == [] and not (tmp_path / alerts.ALERTS_FILE).exists()


def test_progress_every_ten_percent_once_and_across_a_restart(tmp_path):
    env = {alerts.ENV_WEBHOOK: URL}
    poster = Poster()
    assert alerts.notify(tmp_path, _doc(5, 95), env=env, poster=poster) == []
    lines = alerts.notify(tmp_path, _doc(23, 77), env=env, poster=poster)
    assert lines == ["run r1: openrouter 20% (23 answered, 77 to go, 0 invalid)"]
    # The same state again (or a restarted driver) sends nothing new.
    assert alerts.notify(tmp_path, _doc(24, 76), env=env, poster=poster) == []
    # A later round registers more calls: the share drops, no milestone repeats.
    assert alerts.notify(tmp_path, _doc(24, 200), env=env, poster=poster) == []
    assert alerts.notify(tmp_path, _doc(100, 0), env=env, poster=poster) == [
        "run r1: openrouter 100% (100 answered, 0 to go, 0 invalid)"
    ]
    assert [url for url, _ in poster.posts] == [URL, URL]


def test_spend_every_whole_dollar(tmp_path):
    env = {alerts.ENV_WEBHOOK: URL}
    poster = Poster()
    alerts.notify(tmp_path, _doc(1, 99, spend=0.99), env=env, poster=poster)
    lines = alerts.notify(tmp_path, _doc(1, 99, spend=2.40), env=env, poster=poster)
    assert lines == ["run r1: spend passed $2 (total $2.40; openrouter $2.40/$14)"]
    assert alerts.notify(tmp_path, _doc(1, 99, spend=2.90), env=env, poster=poster) == []


def test_stops_and_the_end_are_sent_once(tmp_path):
    env = {alerts.ENV_WEBHOOK: URL}
    poster = Poster()
    stops = {
        "providers": {"anthropic": {"kind": "money", "reason": "insufficient_credit"}},
        "models": {"openai/gpt-6-sol": {"kind": "truncation", "reason": "truncation"}},
    }
    lines = alerts.notify(tmp_path, _doc(0, 10, stops=stops), env=env, poster=poster)
    assert "run r1: STOP anthropic: insufficient_credit" in lines
    assert "run r1: STOP openai/gpt-6-sol: truncation" in lines
    assert alerts.notify(tmp_path, _doc(0, 10, stops=stops), env=env, poster=poster) == []
    done = alerts.notify(
        tmp_path, _doc(10, 0, spend=0.5, stops=stops), env=env, poster=poster, finished="complete"
    )
    assert done[-1].startswith("run r1: COMPLETE (total spend $0.50)")
    asked = alerts.events(_doc(0, 1, status="ask"), alerts._load(tmp_path / "other"))[0]
    assert asked == ["run r1: STOPPED TO ASK the operator; see drive.log"]


def test_a_failed_post_is_retried_next_step(tmp_path):
    env = {alerts.ENV_WEBHOOK: URL}
    with pytest.raises(OSError):
        alerts.notify(tmp_path, _doc(50, 50), env=env, poster=Poster(fail=True))
    poster = Poster()
    assert alerts.notify(tmp_path, _doc(50, 50), env=env, poster=poster)
    assert len(poster.posts) == 1


def test_only_counts_and_names_are_sent(tmp_path):
    doc = _doc(40, 60, spend=1.2)
    doc["hosts"] = {"openrouter.ai": ["secret case text should never be read"]}
    doc["messages"] = ["a message that may quote a case id"]
    lines, _ = alerts.events(doc, alerts._load(tmp_path))
    text = "\n".join(lines)
    assert "case" not in text and "hosts" not in text


def test_post_webhook_refuses_a_non_https_url():
    with pytest.raises(ValueError):
        alerts.post_webhook("http://example.invalid/hook", "x")
