from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from face2ai_app.domain.models import (
    ACTIONS,
    EMOTIONS,
    ActionEvent,
    AffectSample,
    DetectedFace,
    Expression,
    FaceBox,
    FaceObservation,
    IdentityRecord,
    Presence,
    PresenceTransition,
    TimelineSnapshot,
)


def test_expression_shape_and_bounds():
    e = Expression(dominant="Happiness", scores={"Happiness": 0.9, "Neutral": 0.1}, valence=0.7, arousal=0.2,
                   blendshapes={"mouthSmileLeft": 0.95}, yaw=3.0, pitch=-2.0, roll=0.5)
    assert e.dominant == "Happiness" and set(e.scores) <= set(EMOTIONS)
    with pytest.raises(ValidationError):
        Expression(dominant="Happiness", scores={"Bogus": 1.0})
    with pytest.raises(ValidationError):
        Expression(dominant="Happiness", scores={"Happiness": 1.0}, valence=2.0)


def test_face_observation_expression_is_optional_and_presence_carries_mood():
    box = FaceBox(top=1, right=2, bottom=3, left=0)
    assert FaceObservation(box=box).expression is None
    p = Presence(state="KNOWN", mood="Happiness", valence=0.5, arousal=0.1)
    assert p.mood == "Happiness"
    t = PresenceTransition(at="2026-08-18T12:00:00Z", from_state="NO_FACE", to_state="KNOWN")
    assert "mood" not in t.model_dump()  # a transition starts a fresh presence; the ended mood rides the mood event
    # wire-contract pin: these labels reach the voice agent and the Hermes plugin
    assert EMOTIONS == ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")


def test_scores_and_presence_valence_are_bounded():
    with pytest.raises(ValidationError):
        Expression(dominant="Happiness", scores={"Happiness": 7.5})
    with pytest.raises(ValidationError):
        Expression(dominant="Happiness", blendshapes={"mouthSmileLeft": -0.1})
    with pytest.raises(ValidationError):
        Presence(state="KNOWN", valence=42.0)
    with pytest.raises(ValidationError):
        Presence(state="KNOWN", arousal=-1.5)


def test_action_event_is_wire_safe_and_validated():
    t = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
    e = ActionEvent(at=t, action="smile", onset_at=t, apex_at=t, offset_at=t, peak=0.9, duration_ms=900, frames=2)
    assert set(e.model_dump()) == {"at", "identity_id", "display_name", "action", "onset_at", "apex_at", "offset_at",
                                   "peak", "duration_ms", "frames"}
    with pytest.raises(ValidationError):
        ActionEvent(at=t, action="wink", onset_at=t, apex_at=t, offset_at=t, peak=0.9, duration_ms=1, frames=1)
    with pytest.raises(ValidationError):
        ActionEvent(at=t, action="smile", onset_at=t, apex_at=t, offset_at=t, peak=1.5, duration_ms=1, frames=1)
    # wire-contract pin: these labels reach the browser, the voice agent and the Hermes plugin
    assert ACTIONS == ("smile", "frown", "brow_raise", "brow_furrow", "eye_squint", "eyes_wide", "nose_wrinkle", "lip_press")


def test_timeline_snapshot_shape():
    snap = TimelineSnapshot(seconds=60, samples=[AffectSample(at=datetime.now(timezone.utc), valence=0.2)], moods=[], actions=[])
    assert snap.model_dump()["samples"][0]["mood"] is None


def test_stored_encoding_shape_is_enforced():
    """``IdentityRecord.encodings`` is 128 floats per encoding, like ``DetectedFace.encoding``.

    A hand-edited ``identities.json`` with a shorter vector used to load happily and then make every
    ``/api/recognize`` raise ``ValueError`` inside ``math.dist`` — an HTTP 500, contradicting the
    documented "corrupt store -> 503" contract.
    """
    now = datetime.now(timezone.utc)
    ok = IdentityRecord(id="i", display_name="Ada", encodings=[[0.1] * 128], created_at=now)
    assert len(ok.encodings[0]) == 128
    for bogus in ([[0.1, 0.2, 0.3]], [[0.1] * 127], [[0.1] * 129], [[0.1] * 128, [0.2] * 3]):
        with pytest.raises(ValidationError):
            IdentityRecord(id="i", display_name="Mallory", encodings=bogus, created_at=now)
    with pytest.raises(ValidationError):
        DetectedFace(box=FaceBox(top=1, right=2, bottom=3, left=0), encoding=[0.1, 0.2, 0.3])
