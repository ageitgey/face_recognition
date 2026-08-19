"""Source-level guards for the desktop half (`face2ai/desktop/plugin.js`).

The Hermes desktop app loads this file as plain ESM with no build step, so there is no place to unit-test
it from here (it imports `@hermes/plugin-sdk` and `react`). These are deliberately *source* assertions,
not behaviour: they exist because the checks that would look like real gates are not.

In particular `node --check plugin.js` does **not** reject JSX in this file: measured on Node v24.16.0,
appending `const probe = <div className="x"/>` to a copy still exits 0, because the module-syntax detection
triggered by the `import` statements swallows the syntax error. Copying it to `*.mjs` first does fail
(exit 1), and so does the regex below.
"""

from __future__ import annotations

import re
from pathlib import Path

PLUGIN_JS = Path(__file__).resolve().parents[1] / "face2ai" / "desktop" / "plugin.js"
JSX = re.compile(r"</|/>|<[A-Za-z][A-Za-z0-9]*[ />]")


def test_desktop_plugin_is_plain_esm_without_jsx():
    text = PLUGIN_JS.read_text(encoding="utf-8")
    assert "jsx-runtime" in text, "the pane builds elements through jsx()/jsxs(), never JSX syntax"
    hit = JSX.search(text)  # the message is only built when the assert fails, i.e. when `hit` is a match
    assert hit is None, f"JSX syntax in a file the host loads unbuilt: {text[max(0, hit.start() - 40):hit.end() + 40]!r}"


def test_desktop_plugin_forgets_expression_history_with_face2ai():
    """Both directions of "forget": Face2AI's `timeline_cleared` (the user's explicit reset) and the
    disposer (plugin disabled in Settings) must leave no mood/action history in module scope — a live
    `/presence` answer carries none, so nothing would overwrite it on the next enable."""
    text = PLUGIN_JS.read_text(encoding="utf-8")
    cleared = re.search(r"event === 'timeline_cleared'\)? \{(.+?)\} else", text, re.S)
    assert cleared, "the desktop mirror ignores Face2AI's timeline_cleared frame"
    for line in ("moods = []", "actions = []", "timeline = EMPTY_TIMELINE"):
        assert line in cleared.group(1), f"timeline_cleared must drop {line}"
    disposer = text[text.index("return () => {", text.index("const stopSocket")):]
    for line in ("moods = []", "actions = []", "timeline = EMPTY_TIMELINE", "history = []"):
        assert line in disposer, f"the disposer must drop {line}"


def test_desktop_pane_asks_face2ai_for_one_persons_timeline_on_its_own_cadence():
    """The timeline is up to 2000 samples over a tunnel and feeds a 240 px line: it must not ride the
    4 s presence poll, and it must be narrowed server-side to the person the sparkline claims to show."""
    text = PLUGIN_JS.read_text(encoding="utf-8")
    assert "identity_id=${encodeURIComponent(who)}" in text, "the proxy must filter, not the pane"
    interval = re.search(r"TIMELINE_MIN_INTERVAL_MS = (\d+)", text)
    poll = re.search(r"POLL_MS = (\d+)", text)
    assert interval and poll and int(interval.group(1)) >= 5 * int(poll.group(1))


def test_desktop_pane_makes_no_freshness_claim():
    """Freshness left the wire (`Presence` has `observed_at`, no `stale`), so a `p.stale` branch here
    would be a line that can never render — the defect this file exists to catch in unbuilt JS."""
    text = PLUGIN_JS.read_text(encoding="utf-8")
    # A property, not the English word: the timeline cache legitimately calls itself stale in prose.
    hit = re.search(r"\.stale\b|\bstale\s*:", text)
    assert hit is None, f"the desktop half must not read or invent a freshness flag: {text[max(0, hit.start() - 60):hit.end() + 20]!r}"
    assert "frischen Frames" not in text
