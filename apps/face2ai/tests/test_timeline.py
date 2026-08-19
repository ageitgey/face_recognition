from datetime import datetime, timedelta, timezone

from face2ai_app.domain.models import ActionEvent, AffectSample, MoodTransition
from face2ai_app.services.timeline import AffectHistory

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def sample(i, identity="a", valence=0.5):
    return AffectSample(at=T0 + timedelta(seconds=i), identity_id=identity, valence=valence)


def test_samples_are_bounded_by_age_and_count():
    h = AffectHistory(max_seconds=60, max_samples=1000)
    for i in range(100):
        h.record_sample(sample(i))
    snap = h.snapshot(now=T0 + timedelta(seconds=99))
    assert snap.seconds == 60 and len(snap.samples) == 61 and snap.samples[0].at == T0 + timedelta(seconds=39)
    small = AffectHistory(max_seconds=60, max_samples=10)
    for i in range(100):
        small.record_sample(sample(i))
    assert len(small.snapshot(now=T0 + timedelta(seconds=99)).samples) == 10


def test_snapshot_filters_window_and_identity():
    h = AffectHistory(max_seconds=600)
    h.record_sample(sample(0, "a")); h.record_sample(sample(1, "b")); h.record_sample(sample(500, "a"))
    snap = h.snapshot(seconds=60, identity_id="a", now=T0 + timedelta(seconds=520))
    assert [s.at for s in snap.samples] == [T0 + timedelta(seconds=500)]
    assert h.snapshot(seconds=60, now=T0 + timedelta(seconds=520)).samples[-1].identity_id == "a"


def test_moods_and_actions_are_kept_and_cleared():
    h = AffectHistory(max_moods=2, max_actions=2)
    for i in range(3):
        h.record_mood(MoodTransition(at=T0 + timedelta(seconds=i), from_mood=None, to_mood="Happiness"))
        h.record_action(ActionEvent(at=T0 + timedelta(seconds=i), action="smile", onset_at=T0, apex_at=T0, offset_at=T0, peak=0.5, duration_ms=1, frames=1))
    snap = h.snapshot(now=T0 + timedelta(seconds=3))
    assert len(snap.moods) == 2 and len(snap.actions) == 2 and snap.moods[0].at == T0 + timedelta(seconds=1)
    h.record_sample(sample(3, "a"))
    # clear() reports what it dropped (2 moods + 2 actions + 1 sample), so the route can tell a real
    # forget from a no-op and only announce the former to mirrors.
    assert h.clear() == 5
    empty = h.snapshot()
    assert (empty.samples, empty.moods, empty.actions) == ([], [], [])
    assert h.clear() == 0  # nothing left to forget


def test_a_mood_end_after_the_last_sample_stays_visible():
    """A toggle-off/expiry/reset records a mood end after the last frame; anchoring the window on
    samples alone used to hide it (measured 2026-08-18: buffer had it, snapshot did not)."""
    h = AffectHistory(max_seconds=600)
    h.record_sample(AffectSample(at=T0, valence=0.5))
    h.record_mood(MoodTransition(at=T0, from_mood=None, to_mood="Happiness"))
    h.record_mood(MoodTransition(at=T0 + timedelta(milliseconds=2), from_mood="Happiness", to_mood=None))
    h.record_action(ActionEvent(at=T0 + timedelta(milliseconds=3), action="smile", onset_at=T0,
                                apex_at=T0, offset_at=T0 + timedelta(milliseconds=3), peak=0.9,
                                duration_ms=3, frames=2))
    snap = h.snapshot()
    assert [(m.from_mood, m.to_mood) for m in snap.moods] == [(None, "Happiness"), ("Happiness", None)]
    assert [a.action for a in snap.actions] == ["smile"]
    assert len(snap.samples) == 1
