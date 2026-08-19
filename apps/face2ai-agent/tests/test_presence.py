from __future__ import annotations

from datetime import datetime, timedelta, timezone

from face2ai_agent.presence import MOOD_WORDS, PresenceMemory, StoreChange, Transition, parse_sse

T0 = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def test_parse_sse_handles_ids_events_and_multiline_data():
    lines = [
        "id: 3",
        "event: presence",
        'data: {"to_state": "KNOWN",',
        'data:  "display_name": "Ada"}',
        "",
        ": comment",
        "event: heartbeat",
        'data: {"presence": {"state": "KNOWN"}}',
        "",
        "data: not json",
        "",
    ]
    frames = list(parse_sse(lines))
    assert [f.event for f in frames] == ["presence", "heartbeat", "message"]
    assert frames[0].id == "3" and frames[0].data == {"to_state": "KNOWN", "display_name": "Ada"}
    assert frames[1].id is None
    assert frames[2].data == {"raw": "not json"}


def test_memory_hello_transitions_and_describe():
    memory = PresenceMemory()
    assert "not connected" in memory.describe()
    memory.apply_hello({"presence": {"state": "NO_FACE", "since": T0.isoformat()}, "greeting_cooldown_seconds": 9, "engine_available": True})
    assert memory.connected and memory.greeting_cooldown_seconds == 9
    assert "nobody is in front" in memory.describe(T0 + timedelta(seconds=2))

    memory.apply_transition(Transition.from_payload({"at": T0.isoformat(), "from_state": "NO_FACE", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ada", "faces": 1}))
    text = memory.describe(T0 + timedelta(seconds=30))
    assert text.startswith("Ada is in front of the camera")
    assert "about 30 seconds" in text

    memory.apply_transition(Transition.from_payload({"at": (T0 + timedelta(seconds=40)).isoformat(), "from_state": "KNOWN", "to_state": "UNKNOWN", "faces": 1}))
    memory.apply_transition(Transition.from_payload({"at": (T0 + timedelta(seconds=41)).isoformat(), "from_state": "UNKNOWN", "to_state": "NO_FACE", "faces": 0}))
    text = memory.describe(T0 + timedelta(seconds=41))
    assert "nobody is in front" in text
    assert "Recently seen: Ada" in text  # history is kept ...
    assert "no face" not in text and "unknown at" not in text  # ... but only people are listed

    memory.apply_store_change(StoreChange.from_payload({"kind": "enrolled", "display_name": "Bo", "identity_count": 2}))
    assert "2 people are enrolled" in memory.describe(T0 + timedelta(seconds=42))


def test_memory_multiple_and_engine_down_and_no_freshness_claim():
    """Freshness is not on the wire and the agent must not claim it.

    A `stale` key from an older backend is ignored, not turned into an honesty line the agent
    cannot back up — it owns no clock for presence (only `Presence.since` elapsed text).
    """
    memory = PresenceMemory()
    memory.apply_hello({"presence": {"state": "MULTIPLE_FACES", "faces": 3, "stale": True}, "engine_available": False})
    text = memory.describe()
    assert "3 people" in text and "cannot attribute identity" in text
    assert "No fresh frames" not in text
    assert "engine reports itself unavailable" in text
    assert not hasattr(memory.current, "stale")
    memory.apply_heartbeat({"presence": {"state": "NO_SIGNAL"}})
    assert "camera is currently off" in memory.describe()


def test_unknown_states_are_normalized():
    memory = PresenceMemory()
    memory.apply_hello({"presence": {"state": "WEIRD"}})
    assert memory.current.state == "NO_SIGNAL"


def test_situation_key_ignores_elapsed_time_but_tracks_state_changes():
    memory = PresenceMemory()
    memory.apply_hello({"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ada", "since": T0.isoformat()}, "last_sequence": 7})
    key = memory.situation_key()
    assert memory.describe(T0 + timedelta(seconds=1)) != memory.describe(T0 + timedelta(seconds=90))  # elapsed text differs
    assert memory.situation_key() == key  # ... but the situation key is stable
    memory.apply_heartbeat({"presence": {"state": "KNOWN", "identity_id": "b", "display_name": "Bo"}})
    assert memory.situation_key() != key
    assert memory.hello_sequence == 7
    assert memory.is_replayed({"sequence": 7}) and memory.is_replayed({"sequence": 3})
    assert not memory.is_replayed({"sequence": 8}) and not memory.is_replayed({})


async def test_presence_loop_flags_replayed_events(monkeypatch):
    from face2ai_agent import presence as mod

    frames = [
        mod.SseFrame("hello", {"presence": {"state": "NO_FACE"}, "last_sequence": 2}),
        mod.SseFrame("presence", {"sequence": 1, "from_state": "NO_SIGNAL", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ada", "faces": 1}, "1"),
        mod.SseFrame("store", {"sequence": 2, "kind": "enrolled", "display_name": "Bo", "identity_count": 2}, "2"),
        mod.SseFrame("presence", {"sequence": 3, "from_state": "KNOWN", "to_state": "UNKNOWN", "faces": 1}, "3"),
        mod.SseFrame("heartbeat", {"presence": {"state": "UNKNOWN"}}),
    ]

    class FakeClient:
        async def frames(self):
            for f in frames:
                yield f

    seen = []

    async def on_event(kind, payload, replayed):
        seen.append((kind, replayed))

    memory = mod.PresenceMemory()
    await mod.run_presence_loop(FakeClient(), memory, on_event)
    assert seen == [("hello", False), ("presence", True), ("store", True), ("presence", False), ("heartbeat", False)]
    assert memory.current.state == "UNKNOWN"
    assert len(memory.history) == 2 and memory.identity_count == 2


def test_describe_includes_hedged_mood():
    memory = PresenceMemory()
    memory.apply_hello({"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}})
    assert memory.current.mood == "Happiness" and memory.current.valence == 0.6
    de = memory.describe(T0, language="de")
    assert "wirkt fröhlich" in de and "Valenz +0.6" in de and "Erregung +0.1" in de
    assert "ist fröhlich" not in de and "erkannt" not in de.split("wirkt")[1]
    en = memory.describe(T0)  # the situation report itself is English; default clause language follows
    assert "looks happy" in en and "valence +0.6" in en and "arousal +0.1" in en
    assert "is happy" not in en
    assert "keine Tatsache" in de and "not a fact" in en


def test_mood_event_updates_current_mood_and_null_ends_it_without_touching_situation():
    memory = PresenceMemory()
    memory.apply_hello({"presence": {"state": "UNKNOWN", "faces": 1}, "last_sequence": 0})
    key = memory.situation_key()
    memory.apply_mood({"sequence": 1, "identity_id": None, "from_mood": None, "to_mood": "Sadness", "valence": -0.4, "arousal": -0.2})
    assert memory.current.mood == "Sadness" and memory.current.valence == -0.4 and memory.current.arousal == -0.2
    assert memory.situation_key() == key  # mood never triggers an instruction refresh or a greeting
    assert "wirkt traurig" in memory.describe(T0, language="de") and "Valenz -0.4" in memory.describe(T0, language="de")
    memory.apply_mood({"sequence": 2, "from_mood": "Sadness", "to_mood": "Boredom"})  # unknown label: hedged, never raises
    assert "wirkt boredom" in memory.describe(T0, language="de") and "Valenz" not in memory.describe(T0, language="de")
    memory.apply_mood({"sequence": 3, "from_mood": "Boredom", "to_mood": None})
    assert memory.current.mood is None and memory.current.valence is None and memory.current.arousal is None
    assert "wirkt" not in memory.describe(T0, language="de") and "looks" not in memory.describe(T0)


def test_presence_transition_starts_a_fresh_presence_without_mood():
    memory = PresenceMemory()
    memory.apply_hello({"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}})
    memory.apply_transition(Transition.from_payload({"at": T0.isoformat(), "from_state": "KNOWN", "to_state": "NO_FACE", "faces": 0}))
    assert memory.current.mood is None and memory.current.valence is None
    assert "wirkt" not in memory.describe(T0, language="de")
    memory.apply_heartbeat({"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a", "mood": "Surprise", "valence": 0.2, "arousal": 0.7}})
    assert memory.current.mood == "Surprise"  # heartbeat snapshots carry the mood too


async def test_presence_loop_dispatches_mood_events():
    from face2ai_agent import presence as mod

    frames = [
        mod.SseFrame("hello", {"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ada"}, "last_sequence": 1}),
        mod.SseFrame("mood", {"sequence": 1, "identity_id": "a", "display_name": "Ada", "from_mood": None, "to_mood": "Neutral", "valence": 0.0, "arousal": 0.0}, "1"),
        mod.SseFrame("mood", {"sequence": 2, "identity_id": "a", "display_name": "Ada", "from_mood": "Neutral", "to_mood": "Happiness", "valence": 0.6, "arousal": 0.1}, "2"),
    ]

    class FakeClient:
        async def frames(self):
            for f in frames:
                yield f

    seen = []

    async def on_event(kind, payload, replayed):
        seen.append((kind, replayed))

    memory = mod.PresenceMemory()
    await mod.run_presence_loop(FakeClient(), memory, on_event)
    assert seen == [("hello", False), ("mood", True), ("mood", False)]
    assert memory.current.mood == "Happiness" and memory.current.state == "KNOWN"
    assert len(memory.history) == 0  # a mood is not a presence transition


def test_mood_word_tables_cover_exactly_the_eight_wire_labels():
    """Drift guard: both languages must translate exactly the 8 EmotiEffLib labels Face2AI puts on the wire
    (domain.models.EMOTIONS) — an unknown label would fall back to the raw English token in a German sentence."""
    expected = {"Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"}
    assert set(MOOD_WORDS["de"]["labels"]) == set(MOOD_WORDS["en"]["labels"]) == expected
    for words in MOOD_WORDS.values():
        assert all(isinstance(v, str) and v for v in words["labels"].values())


async def test_presence_loop_ignores_action_frames_by_design():
    """Stage 2 SSE ``action`` (a completed facial action: onset/apex/offset) and ``timeline_cleared``
    (the user's explicit forget, for mirrors that keep expression history — the agent keeps none) are for
    panes and history only. The voice loop must stay inert: no dispatch, no exception, memory untouched —
    live and replayed alike — and the frames after them are still handled."""
    from face2ai_agent import presence as mod

    action = {"identity_id": "a", "display_name": "Ada", "action": "smile", "onset_at": T0.isoformat(),
              "apex_at": (T0 + timedelta(milliseconds=600)).isoformat(), "offset_at": (T0 + timedelta(milliseconds=1200)).isoformat(),
              "peak": 0.71, "duration_ms": 1200, "frames": 2}
    frames = [
        mod.SseFrame("hello", {"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ada", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}, "last_sequence": 1}),
        mod.SseFrame("action", {"sequence": 1, "at": T0.isoformat(), **action}, "1"),  # replayed (sequence <= hello.last_sequence)
        mod.SseFrame("action", {"sequence": 2, "at": T0.isoformat(), **action}, "2"),  # live
        mod.SseFrame("timeline_cleared", {"sequence": 3, "at": T0.isoformat()}, "3"),  # the forget signal: nothing to forget here
        mod.SseFrame("heartbeat",{"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ada", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}}),
    ]

    class FakeClient:
        async def frames(self):
            for f in frames:
                yield f

    seen = []

    async def on_event(kind, payload, replayed):
        seen.append((kind, replayed))

    memory = mod.PresenceMemory()
    await mod.run_presence_loop(FakeClient(), memory, on_event)
    assert seen == [("hello", False), ("heartbeat", False)]  # neither action frame nor the clear is dispatched
    assert memory.current == mod.Presence.from_payload(frames[0].data["presence"])  # memory unchanged by actions
    assert memory.current.mood == "Happiness" and memory.current.state == "KNOWN"
    assert len(memory.history) == 0 and len(memory.store_changes) == 0 and memory.identity_count is None
    assert "smile" not in memory.describe(T0) and "smile" not in memory.describe(T0, language="de")
