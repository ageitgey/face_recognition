from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face2ai.presence import ACTION_WORDS, MOOD_WORDS, PresenceStore, SseFrame, action_sentence, clock, context_line, describe, mood_sentence, parse_sse  # noqa: E402

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def frames(*specs):
    return [SseFrame(event=e, data=d, id=i) for e, d, i in specs]


def test_parse_sse_frames():
    out = list(parse_sse(["id: 3", "event: presence", 'data: {"to_state": "KNOWN"}', "", ": ping", "event: heartbeat", 'data: {"presence": {"state": "KNOWN"}}', ""]))
    assert [f.event for f in out] == ["presence", "heartbeat"]
    assert out[0].id == "3" and out[0].data["to_state"] == "KNOWN"


def test_store_hello_transition_and_describe_de_and_en():
    store = PresenceStore()
    assert "nicht verbunden" in describe(store, now=T0)
    store.apply(SseFrame("hello", {"presence": {"state": "NO_FACE", "since": T0.isoformat()}, "last_sequence": 5, "engine_available": True}), now=T0)
    assert store.connected and store.hello_sequence == 5
    assert "niemand steht davor" in describe(store, now=T0 + timedelta(seconds=3))
    t = store.apply(SseFrame("presence", {"sequence": 6, "at": T0.isoformat(), "from_state": "NO_FACE", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ben", "faces": 1}), now=T0)
    assert t is not None and t.display_name == "Ben"
    de = describe(store, now=T0 + timedelta(seconds=40))
    assert de.startswith("Ben steht vor der Kamera seit etwa 40 s")
    en = describe(store, now=T0 + timedelta(seconds=40), language="en")
    assert en.startswith("Ben is in front of the camera for about 40 s")
    assert "keine Gewissheit" in de and "not certainty" in en


def test_replayed_transitions_update_state_but_do_not_react():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "NO_SIGNAL"}, "last_sequence": 10}), now=T0)
    replayed = store.apply(SseFrame("presence", {"sequence": 9, "from_state": "NO_SIGNAL", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ben", "faces": 1, "at": T0.isoformat()}), now=T0)
    assert replayed is None and store.current.state == "KNOWN"
    live = store.apply(SseFrame("presence", {"sequence": 11, "from_state": "KNOWN", "to_state": "UNKNOWN", "faces": 1, "at": T0.isoformat()}), now=T0)
    assert live is not None and live.to_state == "UNKNOWN"


def test_unknown_multiple_and_engine_down_wording_without_a_freshness_claim():
    """Freshness is not on the wire and this store must not claim it from a payload key.

    The plugin's freshness owner is `context_line(max_age_seconds=…)`, which withholds the whole
    line from its own clock (`last_frame_at`) — one rule, one place. A `stale` key from an older
    backend is ignored rather than turned into a sentence nothing keeps true.
    """
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "UNKNOWN", "faces": 1, "stale": True}, "engine_available": False}), now=T0)
    text = describe(store, now=T0)
    assert "kennt sie nicht" in text and "Rate keinen Namen" in text
    assert "frischen Frames" not in text
    assert "stale" not in store.snapshot()["presence"]
    assert "nicht verfügbar" in text
    store.apply(SseFrame("heartbeat", {"presence": {"state": "MULTIPLE_FACES", "faces": 3}}), now=T0)
    assert "3 Personen" in describe(store, now=T0)


def test_recently_seen_lists_only_other_known_people():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "NO_FACE"}, "last_sequence": 0}), now=T0)
    store.apply(SseFrame("presence", {"sequence": 1, "at": T0.isoformat(), "from_state": "NO_FACE", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ada", "faces": 1}), now=T0)
    store.apply(SseFrame("presence", {"sequence": 2, "at": (T0 + timedelta(seconds=30)).isoformat(), "from_state": "KNOWN", "to_state": "NO_FACE", "faces": 0}), now=T0)
    store.apply(SseFrame("presence", {"sequence": 3, "at": (T0 + timedelta(seconds=60)).isoformat(), "from_state": "NO_FACE", "to_state": "KNOWN", "identity_id": "b", "display_name": "Ben", "faces": 1}), now=T0)
    text = describe(store, now=T0 + timedelta(seconds=61))
    assert text.startswith("Ben steht vor der Kamera")
    assert "Zuletzt gesehen: Ada" in text and "no face" not in text.lower()


def test_context_line_is_withheld_when_frames_are_old():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a"}}), now=T0)
    assert context_line(store, now=T0 + timedelta(seconds=5)).startswith("[face2ai] Ben steht vor der Kamera")
    assert context_line(store, now=T0 + timedelta(seconds=45)) is None  # older than 30 s without a heartbeat
    store.apply(SseFrame("heartbeat", {"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a"}}), now=T0 + timedelta(seconds=44))
    assert context_line(store, now=T0 + timedelta(seconds=45)) is not None
    store.mark_lost("boom")
    assert "nicht verbunden" in context_line(store, now=T0 + timedelta(seconds=45))


def test_snapshot_is_json_friendly():
    import json

    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "NO_FACE", "since": T0.isoformat()}}), now=T0)
    store.apply(SseFrame("store", {"kind": "enrolled", "identity_count": 2}), now=T0)
    snap = store.snapshot()
    json.dumps(snap)
    assert snap["identity_count"] == 2 and snap["presence"]["since"] == T0.isoformat()


def test_describe_includes_hedged_mood_and_mood_frames_update_and_clear():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}, "last_sequence": 0}), now=T0)
    assert store.current.mood == "Happiness" and store.current.valence == 0.6 and store.current.arousal == 0.1
    de = describe(store, now=T0)
    assert "Ben wirkt fröhlich" in de and "Valenz +0.6" in de and "Erregung +0.1" in de and "keine Tatsache" in de
    assert "ist fröhlich" not in de and "erkannt" not in de.split("wirkt")[1]
    en = describe(store, now=T0, language="en")
    assert "Ben looks happy" in en and "valence +0.6" in en and "arousal +0.1" in en and "not a fact" in en
    assert "is happy" not in en
    snap = store.snapshot()
    assert snap["presence"]["mood"] == "Happiness" and snap["presence"]["valence"] == 0.6 and snap["presence"]["arousal"] == 0.1
    # a mood frame updates the hint without being a presence transition
    assert store.apply(SseFrame("mood", {"sequence": 1, "at": T0.isoformat(), "identity_id": "a", "display_name": "Ben", "from_mood": "Happiness", "to_mood": "Sadness", "valence": -0.4, "arousal": -0.2}), now=T0) is None
    assert store.current.state == "KNOWN" and store.current.mood == "Sadness" and store.current.valence == -0.4
    assert "Ben wirkt traurig" in describe(store, now=T0) and len(store.history) == 0
    assert store.apply(SseFrame("mood", {"sequence": 2, "at": T0.isoformat(), "identity_id": "a", "from_mood": "Sadness", "to_mood": "Boredom"}), now=T0) is None
    assert "wirkt boredom" in describe(store, now=T0)  # unknown label: still hedged, never raises
    store.apply(SseFrame("mood", {"sequence": 3, "at": T0.isoformat(), "identity_id": "a", "from_mood": "Boredom", "to_mood": None}), now=T0)
    assert store.current.mood is None and store.current.valence is None and store.current.arousal is None
    assert "wirkt" not in describe(store, now=T0)


def test_presence_transition_starts_a_fresh_presence_without_mood_and_unknown_uses_generic_subject():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}, "last_sequence": 0}), now=T0)
    store.apply(SseFrame("presence", {"sequence": 1, "at": T0.isoformat(), "from_state": "KNOWN", "to_state": "UNKNOWN", "faces": 1}), now=T0)
    assert store.current.mood is None and "wirkt" not in describe(store, now=T0)
    store.apply(SseFrame("heartbeat", {"presence": {"state": "UNKNOWN", "faces": 1, "mood": "Surprise", "valence": 0.2, "arousal": 0.7}}), now=T0)
    assert "Die Person wirkt überrascht" in describe(store, now=T0)
    assert "The person looks surprised" in describe(store, now=T0, language="en")


def test_mood_word_tables_cover_exactly_the_eight_wire_labels():
    """Drift guard: both languages must translate exactly the 8 EmotiEffLib labels Face2AI puts on the wire
    (domain.models.EMOTIONS) — an unknown label would fall back to the raw English token in a German sentence."""
    expected = {"Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"}
    assert set(MOOD_WORDS["de"]["labels"]) == set(MOOD_WORDS["en"]["labels"]) == expected
    for words in MOOD_WORDS.values():
        assert all(isinstance(v, str) and v for v in words["labels"].values())


def test_store_keeps_mood_and_action_history_and_snapshot_carries_them():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ben"}}), now=T0)
    assert store.apply(SseFrame("mood", {"sequence": 1, "at": "2026-08-18T12:00:00Z", "identity_id": "a", "display_name": "Ben", "from_mood": None, "to_mood": "Happiness", "valence": 0.6, "arousal": 0.1}), now=T0) is None
    assert store.apply(SseFrame("action", {"sequence": 2, "at": "2026-08-18T12:00:03Z", "identity_id": "a", "display_name": "Ben", "action": "smile", "onset_at": "2026-08-18T12:00:01Z", "apex_at": "2026-08-18T12:00:02Z", "offset_at": "2026-08-18T12:00:03Z", "peak": 0.9, "duration_ms": 2000, "frames": 4}), now=T0) is None
    snap = store.snapshot()
    assert snap["moods"][-1]["to_mood"] == "Happiness" and snap["actions"][-1]["action"] == "smile"
    assert set(snap) == {"connected", "engine_available", "identity_count", "last_frame_at", "last_error", "presence", "history", "moods", "actions"}
    # raw wire dicts only (nothing invented, nothing dropped), JSON friendly, action never touches presence/history
    assert snap["actions"][-1] == {"sequence": 2, "at": "2026-08-18T12:00:03Z", "identity_id": "a", "display_name": "Ben", "action": "smile", "onset_at": "2026-08-18T12:00:01Z", "apex_at": "2026-08-18T12:00:02Z", "offset_at": "2026-08-18T12:00:03Z", "peak": 0.9, "duration_ms": 2000, "frames": 4}
    json.dumps(snap)
    assert store.current.mood == "Happiness" and len(store.history) == 0
    # a mood end is history too (frozen valence/arousal live in the frame), and the snapshot is bounded: last 20 / last 10
    store.apply(SseFrame("mood", {"sequence": 3, "at": "2026-08-18T12:00:09Z", "identity_id": "a", "from_mood": "Happiness", "to_mood": None}), now=T0)
    assert store.snapshot()["moods"][-1]["to_mood"] is None and store.current.mood is None
    for i in range(60):
        store.apply(SseFrame("mood", {"sequence": 10 + i, "at": T0.isoformat(), "from_mood": None, "to_mood": "Neutral"}), now=T0)
        store.apply(SseFrame("action", {"sequence": 100 + i, "at": T0.isoformat(), "action": "frown", "duration_ms": 600 + i}), now=T0)
    snap = store.snapshot()
    assert len(store.moods) == 50 and len(store.actions) == 30
    assert len(snap["moods"]) == 20 and len(snap["actions"]) == 10 and snap["actions"][-1]["duration_ms"] == 659


def test_action_sentence_is_hedged():
    assert action_sentence({"action": "smile", "duration_ms": 900}, language="de") == "kurzes Lächeln (0.9 s)"
    assert action_sentence({"action": "brow_raise", "duration_ms": 2300}, language="en") == "brow raise (2.3 s)"
    assert action_sentence({"action": "wink", "duration_ms": 900}, language="de") == "kurzes wink (0.9 s)"
    assert action_sentence({"action": "smile", "duration_ms": 6000}, language="de") == "anhaltendes Lächeln (6.0 s)"
    assert action_sentence({"action": "smile", "duration_ms": 6000}, language="en") == "held smile (6.0 s)"
    assert action_sentence({"action": "lip_press", "duration_ms": 1000}, language="en") == "brief lip press (1.0 s)"
    # The qualifier follows the *printed* number: 4999 ms prints "5.0 s", so it reads "anhaltendes" —
    # printing "Augen weit (5.0 s)" next to "anhaltendes Augen weit (5.0 s)" would be two phrasings for
    # one visible value (this line pinned that pair before; the value below comes from the implementation).
    assert action_sentence({"action": "eyes_wide", "duration_ms": 4999}, language="de") == "anhaltendes Augen weit (5.0 s)"
    assert action_sentence({"action": "eyes_wide", "duration_ms": 4949}, language="de") == "Augen weit (4.9 s)"
    assert action_sentence({"action": "smile", "duration_ms": 1049}, language="en") == "brief smile (1.0 s)"
    assert action_sentence({"action": "smile", "duration_ms": 1051}, language="en") == "smile (1.1 s)"
    assert action_sentence({"action": "smile", "duration_ms": "soon"}, language="de") == "Lächeln"  # non-numeric duration → no parentheses
    assert action_sentence({"action": "smile", "duration_ms": float("nan")}, language="de") == "Lächeln"  # never "(nan s)"
    assert action_sentence({"action": "smile", "duration_ms": float("inf")}, language="de") == "Lächeln"  # never "(inf s)"
    assert action_sentence({"action": "smile"}, language="en") == "smile"
    assert action_sentence({}, language="de") == "" and action_sentence(None, language="en") == ""  # never raises
    assert action_sentence({"action": "eye_squint", "duration_ms": 900}, language="fr") == "brief eye squint (0.9 s)"  # unknown language → en
    assert action_sentence({"action": "smile", "duration_ms": 900}, language=None) == "kurzes Lächeln (0.9 s)"  # "never raises" includes None
    assert mood_sentence("Happiness", 0.6, 0.1, language=None).startswith("Die Person wirkt fröhlich")
    assert mood_sentence("Happiness", float("nan"), float("inf")) == "Die Person wirkt fröhlich – nur ein Hinweis aus dem Gesichtsausdruck, keine Tatsache."


def test_action_word_tables_cover_exactly_the_eight_wire_actions():
    """Drift guard: both languages must translate exactly the 8 action labels Face2AI puts on the wire
    (domain.models.ACTIONS)."""
    expected = {"smile", "frown", "brow_raise", "brow_furrow", "eye_squint", "eyes_wide", "nose_wrinkle", "lip_press"}
    assert set(ACTION_WORDS["de"]["labels"]) == set(ACTION_WORDS["en"]["labels"]) == expected
    for words in ACTION_WORDS.values():
        assert all(isinstance(v, str) and v for v in words["labels"].values())


def test_describe_does_not_speak_actions_into_context():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ben"}}), now=T0)
    store.apply(SseFrame("action", {"sequence": 1, "at": T0.isoformat(), "identity_id": "a", "display_name": "Ben", "action": "smile", "duration_ms": 900}), now=T0)
    text = describe(store, now=T0)
    assert text.startswith("Ben steht vor der Kamera") and "Lächeln" not in text and "smile" not in text
    assert "smile" not in describe(store, now=T0, language="en")


def test_snapshot_entries_are_copies_the_caller_cannot_write_back_into_the_store():
    """snapshot() goes to the Hermes tool layer and into the plugin state file; a consumer that
    redacts or normalises an entry in place must not rewrite this store's ring buffer."""
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ben"}}), now=T0)
    store.apply(SseFrame("mood", {"sequence": 1, "at": T0.isoformat(), "from_mood": None, "to_mood": "Happiness"}), now=T0)
    store.apply(SseFrame("action", {"sequence": 2, "at": T0.isoformat(), "action": "smile", "duration_ms": 900}), now=T0)
    snap = store.snapshot()
    assert snap["actions"][-1] is not list(store.actions)[-1] and snap["moods"][-1] is not list(store.moods)[-1]
    snap["actions"][-1]["action"] = "MUTATED"
    snap["moods"][-1]["to_mood"] = "MUTATED"
    assert list(store.actions)[-1]["action"] == "smile" and list(store.moods)[-1]["to_mood"] == "Happiness"
    assert store.snapshot()["actions"][-1]["action"] == "smile"


def test_timeline_cleared_makes_the_mirror_forget_with_face2ai():
    """`POST /api/presence/reset` empties Face2AI's affect history; the `timeline_cleared` frame is how
    that forget reaches this mirror. A NO_SIGNAL transition is not the signal (an ordinary expiry
    publishes one and keeps the history), and the forget is not a presence change."""
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ben"}, "last_sequence": 0}), now=T0)
    store.apply(SseFrame("mood", {"sequence": 1, "at": T0.isoformat(), "from_mood": None, "to_mood": "Happiness"}), now=T0)
    store.apply(SseFrame("action", {"sequence": 2, "at": T0.isoformat(), "action": "smile", "duration_ms": 900}), now=T0)
    assert store.snapshot()["moods"] and store.snapshot()["actions"]
    assert store.apply(SseFrame("timeline_cleared", {"sequence": 3, "at": T0.isoformat()}), now=T0) is None
    snap = store.snapshot()
    assert (snap["moods"], snap["actions"]) == ([], [])
    assert len(store.moods) == 0 and len(store.actions) == 0  # the ring buffers, not only the snapshot
    assert snap["presence"]["state"] == "KNOWN" and snap["connected"] is True  # forgetting is not a presence change
    assert len(store.history) == 0


def test_clock_prints_local_wall_time_and_tolerates_junk():
    """Face2AI timestamps are UTC and describe() prints local times: one reply must not mix two clocks."""
    assert clock(T0.isoformat()) == T0.astimezone().strftime("%H:%M:%S")
    assert clock("2026-08-18T12:00:03Z") == datetime(2026, 8, 18, 12, 0, 3, tzinfo=timezone.utc).astimezone().strftime("%H:%M:%S")
    assert clock(None) == "" and clock("") == "" and clock("nope") == "" and clock(12) == ""
