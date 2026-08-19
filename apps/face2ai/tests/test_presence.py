from __future__ import annotations

from datetime import datetime, timedelta, timezone

from face2ai_app.domain.models import (
    FaceBox,
    FaceObservation,
    PresenceState,
    RecognitionEvent,
    RecognitionState,
)
from face2ai_app.services.presence import PresenceTracker

T0 = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
BOX = FaceBox(top=1, right=2, bottom=3, left=0)


def no_face() -> RecognitionEvent:
    return RecognitionEvent(state=RecognitionState.NO_FACE)


def unknown() -> RecognitionEvent:
    return RecognitionEvent(state=RecognitionState.UNKNOWN, faces=[FaceObservation(box=BOX)], can_enroll=True)


def known(identity_id: str, name: str, distance: float = 0.3) -> RecognitionEvent:
    return RecognitionEvent(
        state=RecognitionState.KNOWN,
        faces=[FaceObservation(box=BOX, matched=True, identity_id=identity_id, display_name=name, match_distance=distance)],
    )


def multiple() -> RecognitionEvent:
    return RecognitionEvent(state=RecognitionState.MULTIPLE_FACES, faces=[FaceObservation(box=BOX), FaceObservation(box=BOX)])


def test_first_observation_after_no_signal_commits_immediately():
    tracker = PresenceTracker(stable_ticks=2)
    transition = tracker.observe(no_face(), T0)
    assert transition is not None
    assert (transition.from_state, transition.to_state) == (PresenceState.NO_SIGNAL, PresenceState.NO_FACE)
    assert tracker.snapshot().state is PresenceState.NO_FACE


def test_single_frame_flicker_is_filtered_and_stable_change_is_emitted():
    tracker = PresenceTracker(stable_ticks=2)
    tracker.observe(no_face(), T0)
    assert tracker.observe(unknown(), T0 + timedelta(seconds=1)) is None  # one frame is not enough
    assert tracker.observe(no_face(), T0 + timedelta(seconds=2)) is None  # back to current: nothing
    assert tracker.observe(unknown(), T0 + timedelta(seconds=3)) is None
    transition = tracker.observe(unknown(), T0 + timedelta(seconds=4))
    assert transition is not None
    assert transition.to_state is PresenceState.UNKNOWN
    assert transition.faces == 1
    snap = tracker.snapshot()
    assert snap.state is PresenceState.UNKNOWN
    assert snap.since == T0 + timedelta(seconds=4)


def test_known_identity_switch_is_a_transition_but_distance_updates_are_not():
    tracker = PresenceTracker(stable_ticks=2)
    tracker.observe(no_face(), T0)
    tracker.observe(known("a", "Ada", 0.31), T0)
    first = tracker.observe(known("a", "Ada", 0.35), T0)
    assert first is not None and first.display_name == "Ada" and first.identity_id == "a"
    assert not hasattr(first, "match_distance")  # distances never leave the recognition response
    assert tracker.observe(known("a", "Ada", 0.40), T0) is None
    assert tracker.observe(known("a", "Ada", 0.20), T0) is None
    tracker.observe(known("b", "Bo", 0.3), T0)
    switch = tracker.observe(known("b", "Bo", 0.3), T0)
    assert switch is not None
    assert (switch.from_state, switch.to_state) == (PresenceState.KNOWN, PresenceState.KNOWN)
    assert switch.identity_id == "b" and switch.display_name == "Bo"


def test_multiple_faces_and_reset_to_no_signal():
    tracker = PresenceTracker(stable_ticks=1)
    tracker.observe(no_face(), T0)
    transition = tracker.observe(multiple(), T0)
    assert transition is not None and transition.to_state is PresenceState.MULTIPLE_FACES and transition.faces == 2
    reset = tracker.reset(T0)
    assert reset is not None and reset.to_state is PresenceState.NO_SIGNAL
    assert tracker.reset(T0) is None  # idempotent
    assert tracker.snapshot().state is PresenceState.NO_SIGNAL
    assert tracker.snapshot().identity_id is None


def test_expire_turns_a_presence_without_frames_into_no_signal_and_snapshot_only_reports():
    """There is exactly one server-side freshness rule and it is ``expire()``.

    ``snapshot()`` is read-only: it never ages a presence and carries no freshness flag. Consumers
    that want one compute it from ``observed_at`` against their own budget.
    """
    tracker = PresenceTracker(stable_ticks=1, stale_after=timedelta(seconds=5))
    tracker.observe(no_face(), T0)
    tracker.observe(unknown(), T0 + timedelta(seconds=1))
    assert tracker.snapshot().observed_at == T0 + timedelta(seconds=1)
    assert tracker.expire(T0 + timedelta(seconds=3)) is None
    old = tracker.snapshot()
    assert old.state is PresenceState.UNKNOWN  # snapshot never ages the presence itself ...
    assert old.observed_at == T0 + timedelta(seconds=1)  # ... it just reports when the last frame was
    expired = tracker.expire(T0 + timedelta(seconds=7))
    assert expired is not None
    assert (expired.from_state, expired.to_state) == (PresenceState.UNKNOWN, PresenceState.NO_SIGNAL)
    assert tracker.snapshot().state is PresenceState.NO_SIGNAL
    assert tracker.expire(T0 + timedelta(seconds=8)) is None  # idempotent


def test_returning_person_after_a_frame_gap_is_a_fresh_arrival():
    """Tab hidden / paused / closed without Stop: frames stop, then the same person is seen again."""
    tracker = PresenceTracker(stable_ticks=2, stale_after=timedelta(seconds=5))
    tracker.observe(no_face(), T0)
    tracker.observe(known("a", "Ada"), T0 + timedelta(seconds=1))
    assert tracker.observe(known("a", "Ada"), T0 + timedelta(seconds=2)) is not None  # KNOWN Ada
    assert tracker.observe(known("a", "Ada"), T0 + timedelta(seconds=3)) is None
    # 30 s without frames, then Ada is back: must be announced again, from NO_SIGNAL, immediately.
    back = tracker.observe(known("a", "Ada"), T0 + timedelta(seconds=33))
    assert back is not None
    assert (back.from_state, back.to_state) == (PresenceState.NO_SIGNAL, PresenceState.KNOWN)
    assert back.display_name == "Ada"
    assert tracker.snapshot().since == T0 + timedelta(seconds=33)


def test_set_mood_shows_in_snapshot_and_is_cleared_by_transitions():
    tracker = PresenceTracker(stable_ticks=1, stale_after=timedelta(seconds=5))
    tracker.observe(known("a", "Ada"), T0)
    assert tracker.snapshot().mood is None
    tracker.set_mood("Happiness", 0.45, 0.075)
    snap = tracker.snapshot()
    assert (snap.mood, snap.valence, snap.arousal) == ("Happiness", 0.45, 0.075)
    assert snap.state is PresenceState.KNOWN and snap.display_name == "Ada"  # rest untouched
    tracker.set_mood(None, None, None)
    assert tracker.snapshot().mood is None
    # A committed presence transition belongs to a new presence: mood starts empty again.
    tracker.set_mood("Happiness", 0.45, 0.075)
    left = tracker.observe(no_face(), T0 + timedelta(seconds=1))
    assert left is not None and left.to_state is PresenceState.NO_FACE
    snap = tracker.snapshot()
    assert (snap.mood, snap.valence, snap.arousal) == (None, None, None)


def test_expire_and_reset_clear_mood():
    tracker = PresenceTracker(stable_ticks=1, stale_after=timedelta(seconds=5))
    tracker.observe(known("a", "Ada"), T0)
    tracker.set_mood("Sadness", -0.5, -0.2)
    assert tracker.snapshot().mood == "Sadness"
    expired = tracker.expire(T0 + timedelta(seconds=7))
    assert expired is not None and expired.to_state is PresenceState.NO_SIGNAL
    snap = tracker.snapshot()
    assert (snap.state, snap.mood, snap.valence, snap.arousal) == (PresenceState.NO_SIGNAL, None, None, None)
    tracker.observe(known("a", "Ada"), T0 + timedelta(seconds=8))
    tracker.set_mood("Sadness", -0.5, -0.2)
    assert tracker.reset(T0 + timedelta(seconds=9)) is not None
    assert tracker.snapshot().mood is None


def test_wire_models_carry_no_biometrics():
    from face2ai_app.domain.models import Presence, PresenceTransition

    # No `stale`: freshness is the consumer's business, computed from `observed_at` against its own
    # budget. A server-side flag would be a second threshold beside `expire()`'s, and unreachable.
    assert set(Presence.model_fields) == {
        "state", "identity_id", "display_name", "faces", "since", "observed_at", "mood", "valence", "arousal",
    }
    # No mood on transitions: they start a fresh presence; the ended mood travels on the ``mood`` event.
    assert set(PresenceTransition.model_fields) == {"at", "from_state", "to_state", "identity_id", "display_name", "faces"}
