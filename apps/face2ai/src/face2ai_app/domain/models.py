from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

ENCODING_LENGTH = 128  # dlib's face embedding; the only shape this application ever handles

# One owner for the encoding shape: a *detected* face and a *stored* one are the same 128 floats.
# They were not — `IdentityRecord.encodings` was a bare `list[list[float]]`, so a hand-edited
# identities.json loaded happily and then made every recognize raise inside `math.dist`.
Encoding = Annotated[list[float], Field(min_length=ENCODING_LENGTH, max_length=ENCODING_LENGTH)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]


class RecognitionState(StrEnum):
    NO_FACE = "NO_FACE"
    UNKNOWN = "UNKNOWN"
    LEARNING = "LEARNING"
    KNOWN = "KNOWN"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    ERROR = "ERROR"


class FaceBox(BaseModel):
    top: int
    right: int
    bottom: int
    left: int


class DetectedFace(BaseModel):
    box: FaceBox
    encoding: Encoding


EMOTIONS = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")


class Expression(BaseModel):
    """Best-effort facial expression for one face — a mood hint, never a fact, never authentication.

    Wire-safe by construction: labels, scores in 0..1, valence/arousal in -1..1, named blendshape
    intensities and head pose angles. No landmarks, no pixels, no embeddings.
    """

    dominant: str
    scores: dict[str, UnitScore] = Field(default_factory=dict)
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)
    blendshapes: dict[str, UnitScore] = Field(default_factory=dict)  # only entries >= 0.2, rounded to 2 decimals
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None

    @field_validator("scores")
    @classmethod
    def _known_labels(cls, value: dict[str, UnitScore]) -> dict[str, UnitScore]:
        unknown = set(value) - set(EMOTIONS)
        if unknown:
            raise ValueError(f"unknown emotion labels: {sorted(unknown)}")
        return value

    @field_validator("dominant")
    @classmethod
    def _dominant_known(cls, value: str) -> str:
        if value not in EMOTIONS:
            raise ValueError(f"unknown dominant emotion: {value}")
        return value


class FaceObservation(BaseModel):
    box: FaceBox
    matched: bool = False
    identity_id: str | None = None
    display_name: str | None = None
    match_distance: float | None = None
    expression: Expression | None = None


class RecognitionEvent(BaseModel):
    state: RecognitionState
    faces: list[FaceObservation] = Field(default_factory=list)
    can_enroll: bool = False
    message: str | None = None


class IdentityRecord(BaseModel):
    id: str
    display_name: str
    encodings: list[Encoding]
    created_at: datetime
    purpose: str = "local Face2AI recognition"

    @classmethod
    def new(cls, display_name: str, encoding: Encoding) -> "IdentityRecord":
        return cls(
            id=str(uuid4()),
            display_name=display_name.strip(),
            encodings=[encoding],
            created_at=datetime.now(timezone.utc),
        )


class IdentitySummary(BaseModel):
    id: str
    display_name: str
    created_at: datetime
    encoding_count: int


class SystemStatus(BaseModel):
    app: str = "Face2AI"
    version: str
    engine_available: bool
    engine_reason: str | None = None
    identity_count: int
    greeting_cooldown_seconds: int
    agent_connected: bool = False
    event_subscribers: int = 0
    expression_available: bool = False  # the opt-in expression engine (MediaPipe extra + model asset) is loaded
    expression_reason: str | None = None  # why it is not available, when it is not
    expression_enabled: bool = False  # runtime toggle: expressions are attached to recognize responses


class PresenceState(StrEnum):
    """Stable, debounced view of who is in front of the camera (derived from RecognitionEvents)."""

    NO_SIGNAL = "NO_SIGNAL"
    NO_FACE = "NO_FACE"
    UNKNOWN = "UNKNOWN"
    KNOWN = "KNOWN"
    MULTIPLE_FACES = "MULTIPLE_FACES"


class Presence(BaseModel):
    """Wire contract for agents / Party Mirror: states, names, counts, timestamps — nothing biometric.

    ``valence``/``arousal`` are the *live* smoothed affect (Stage 2) — may be present without a
    ``mood``; ``mood`` keeps its hysteresis.

    No freshness flag: ``observed_at`` is when the last frame was seen, and the server's only
    freshness rule is ``PresenceTracker.expire()`` (``FACE2AI_PRESENCE_STALE_SECONDS``), which turns
    a presence without frames into ``NO_SIGNAL``. A consumer that wants a freshness line owns its own
    budget and compares it against ``observed_at``; a second server-side threshold would either
    duplicate ``expire()``'s or, if equal to it, be unreachable.
    """

    state: PresenceState = PresenceState.NO_SIGNAL
    identity_id: str | None = None
    display_name: str | None = None
    faces: int = 0
    since: datetime | None = None
    observed_at: datetime | None = None
    mood: str | None = None
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)


class PresenceTransition(BaseModel):
    """Emitted once per stable presence change; consumers (agents, Party Mirror) subscribe to these.

    Carries no mood: a transition starts a fresh, mood-less presence; the mood that ended with the
    previous presence is announced as ``from_mood`` on the ``mood`` event that follows.
    """

    at: datetime
    from_state: PresenceState
    to_state: PresenceState
    identity_id: str | None = None
    display_name: str | None = None
    faces: int = 0


class MoodTransition(BaseModel):
    """Emitted once per stable mood change (SSE ``mood``). A mood is a hint ("wirkt …"), never a fact.

    Wire-safe: label, names, timestamp and two rounded scalars — no per-frame scores, no blendshapes,
    no pixels. ``to_mood`` None means the mood ended (person left, expression stopped being readable).
    """

    at: datetime
    identity_id: str | None = None
    display_name: str | None = None
    from_mood: str | None
    to_mood: str | None
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)


ACTIONS = ("smile", "frown", "brow_raise", "brow_furrow", "eye_squint", "eyes_wide", "nose_wrinkle", "lip_press")


class ActionEvent(BaseModel):
    """One completed facial action (SSE ``action``): onset → apex → offset from blendshape intensities.

    Timing is quantized to the frame rate (~0.6 s at the browser's loop), so these are expression
    *dynamics*, not micro-expressions; a hint, never a fact. Wire-safe: label, names, timestamps,
    one peak intensity — no per-frame series, no landmarks.
    """

    at: datetime  # == offset_at: when the action became known
    identity_id: str | None = None
    display_name: str | None = None
    action: str
    onset_at: datetime
    apex_at: datetime
    offset_at: datetime
    peak: UnitScore
    duration_ms: int = Field(ge=0)
    frames: int = Field(ge=1)

    @field_validator("action")
    @classmethod
    def _known_action(cls, value: str) -> str:
        if value not in ACTIONS:
            raise ValueError(f"unknown action: {value}")
        return value


class AffectSample(BaseModel):
    """One point of the in-memory affect history: live smoothed valence/arousal + the mood at that time."""

    at: datetime
    identity_id: str | None = None
    display_name: str | None = None
    mood: str | None = None
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)


class TimelineSnapshot(BaseModel):
    """``GET /api/expression/timeline``: bounded, in-memory, never persisted; cleared on presence reset."""

    seconds: int
    samples: list[AffectSample] = Field(default_factory=list)
    moods: list[MoodTransition] = Field(default_factory=list)
    actions: list[ActionEvent] = Field(default_factory=list)


class StoreEventKind(StrEnum):
    ENROLLED = "enrolled"
    DELETED = "deleted"
    ERASED = "erased"


class StoreEvent(BaseModel):
    at: datetime
    kind: StoreEventKind
    identity_id: str | None = None
    display_name: str | None = None
    identity_count: int
