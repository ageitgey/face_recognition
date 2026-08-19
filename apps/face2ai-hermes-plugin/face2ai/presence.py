"""Face2AI presence: SSE parsing, a thread-safe store and the text Hermes gets to see.

Pure Python (stdlib only) so it unit-tests without Hermes and can be imported by both the
gateway plugin (writer) and the dashboard API (reader). Nothing here handles frames or face
encodings — Face2AI's event stream never carries them; a mood is a hedged hint ("wirkt …"), never a fact,
and a facial action ("kurzes Lächeln (0.9 s)") is expression *dynamics* at ~0.6 s resolution, never a
micro-expression and never a fact either.
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

STATES = ("NO_SIGNAL", "NO_FACE", "UNKNOWN", "KNOWN", "MULTIPLE_FACES")

# Hedged wording for Face2AI's mood labels (apps/face2ai domain/models.py EMOTIONS). Same literal table
# as the Face2AI UI (static/js/model.js) and the voice agent (face2ai_agent/presence.py): a mood is how
# a face *appears* ("wirkt …" / "looks …"), never how someone *is* — never a fact, never a gate for anything.
MOOD_WORDS: dict[str, dict[str, Any]] = {
    "de": {
        "prefix": "wirkt ",
        "labels": {"Happiness": "fröhlich", "Sadness": "traurig", "Anger": "verärgert", "Fear": "ängstlich",
                   "Surprise": "überrascht", "Disgust": "angewidert", "Contempt": "abschätzig", "Neutral": "neutral"},
        "valence": "Valenz", "arousal": "Erregung",
        "hedge": "nur ein Hinweis aus dem Gesichtsausdruck, keine Tatsache",
        "generic_subject": "Die Person",
    },
    "en": {
        "prefix": "looks ",
        "labels": {"Happiness": "happy", "Sadness": "sad", "Anger": "angry", "Fear": "fearful",
                   "Surprise": "surprised", "Disgust": "disgusted", "Contempt": "contemptuous", "Neutral": "neutral"},
        "valence": "valence", "arousal": "arousal",
        "hedge": "only a hint from facial expression, not a fact",
        "generic_subject": "The person",
    },
}


def _number(value: Any) -> float | None:
    """A wire number, or None — NaN/inf included: they would print as "(nan s)" in a user-visible phrase."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mood_sentence(mood: str | None, valence: float | None, arousal: float | None, *, language: str = "de", subject: str | None = None) -> str:
    """One hedged sentence for a mood hint, e.g. "Ben wirkt fröhlich (Valenz +0.6, Erregung +0.1) – nur ein
    Hinweis aus dem Gesichtsausdruck, keine Tatsache." Empty string when there is no mood. Unknown labels
    are hedged too (lower-cased) and never raise."""
    if not isinstance(mood, str) or not mood:
        return ""
    words = MOOD_WORDS["de"] if str(language or "de").lower().startswith("de") else MOOD_WORDS["en"]
    label = words["labels"].get(mood, mood.lower())
    # _number() again, not `is not None`: a NaN/inf axis would print as "Valenz +nan" in a user-visible sentence.
    numbers = [f"{words[key]} {round(number, 1) + 0.0:+.1f}" for key, value in (("valence", valence), ("arousal", arousal)) if (number := _number(value)) is not None]
    detail = f" ({', '.join(numbers)})" if numbers else ""
    dash = "–" if words is MOOD_WORDS["de"] else "—"
    return f"{subject or words['generic_subject']} {words['prefix']}{label}{detail} {dash} {words['hedge']}."


# Hedged wording for Face2AI's facial action labels (apps/face2ai domain/models.py ACTIONS). Same literal table as
# the Face2AI UI (static/js/model.js). Timing is quantized to the browser loop (~0.6 s): "brief" ≤ 1 s, "held" ≥ 5 s —
# expression dynamics, never micro-expressions; a hint about what a face *did*, never what someone *felt*.
ACTION_WORDS: dict[str, dict[str, Any]] = {
    "de": {
        "labels": {"smile": "Lächeln", "frown": "Mundwinkel runter", "brow_raise": "Brauen hoch", "brow_furrow": "Stirnrunzeln",
                   "eye_squint": "Augen zusammengekniffen", "eyes_wide": "Augen weit", "nose_wrinkle": "Nasenrümpfen", "lip_press": "Lippen gepresst"},
        "brief": "kurzes ", "held": "anhaltendes ",
    },
    "en": {
        "labels": {"smile": "smile", "frown": "frown", "brow_raise": "brow raise", "brow_furrow": "brow furrow",
                   "eye_squint": "eye squint", "eyes_wide": "eyes wide", "nose_wrinkle": "nose wrinkle", "lip_press": "lip press"},
        "brief": "brief ", "held": "held ",
    },
}
ACTION_BRIEF_MS = 1000  # ≤ → "brief"/"kurzes"
ACTION_HELD_MS = 5000  # ≥ → "held"/"anhaltendes"


def action_sentence(data: Any, *, language: str = "de") -> str:
    """One hedged phrase for a completed facial action, e.g. "kurzes Lächeln (0.9 s)" / "brow raise (2.3 s)" /
    "anhaltendes Lächeln (6.0 s)". Empty string when there is no action label. Unknown labels are lower-cased
    (``_`` → space) and still qualified; a non-numeric ``duration_ms`` drops the parentheses; never raises."""
    if not isinstance(data, dict):
        return ""
    action = data.get("action")
    if not isinstance(action, str) or not action:
        return ""
    words = ACTION_WORDS["de"] if str(language or "de").lower().startswith("de") else ACTION_WORDS["en"]
    label = words["labels"].get(action, action.lower().replace("_", " "))
    duration = _number(data.get("duration_ms"))
    qualifier, detail = "", ""
    if duration is not None:
        shown = f"{duration / 1000:.1f}"  # the number the reader sees …
        shown_ms = float(shown) * 1000  # … and the qualifier follows it, so 4999 ms is not "Lächeln (5.0 s)"
        if shown_ms <= ACTION_BRIEF_MS:
            qualifier = words["brief"]
        elif shown_ms >= ACTION_HELD_MS:
            qualifier = words["held"]
        detail = f" ({shown} s)"
    return f"{qualifier}{label}{detail}"


@dataclass(frozen=True)
class SseFrame:
    event: str
    data: dict[str, Any]
    id: str | None = None


def parse_sse(lines: Iterable[str]) -> Iterator[SseFrame]:
    """Minimal SSE parser: one frame per blank-line-terminated block."""
    event, data_lines, frame_id = "message", [], None
    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data_lines:
                try:
                    data = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    data = {"raw": "\n".join(data_lines)}
                yield SseFrame(event=event, data=data, id=frame_id)
            event, data_lines, frame_id = "message", [], None
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if key == "event":
            event = value
        elif key == "data":
            data_lines.append(value)
        elif key == "id":
            frame_id = value


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def clock(value: Any) -> str:
    """An ISO wire timestamp as a local wall-clock "12:00:03" — "" when there is none.

    Face2AI timestamps are UTC; ``describe()`` already prints local times, so every other user-visible
    time has to agree with it instead of slicing the raw UTC string.
    """
    dt = _parse_time(value)
    return dt.astimezone().strftime("%H:%M:%S") if dt else ""


@dataclass
class Presence:
    state: str = "NO_SIGNAL"
    identity_id: str | None = None
    display_name: str | None = None
    faces: int = 0
    since: datetime | None = None
    mood: str | None = None  # best-effort hint ("wirkt …"), never a fact; None = nothing to say
    valence: float | None = None  # -1..1
    arousal: float | None = None  # -1..1

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Presence":
        state = payload.get("state") if payload.get("state") in STATES else "NO_SIGNAL"
        mood = payload.get("mood")
        return cls(
            state=state,
            identity_id=payload.get("identity_id"),
            display_name=payload.get("display_name"),
            faces=int(payload.get("faces") or 0),
            since=_parse_time(payload.get("since")),
            mood=mood if isinstance(mood, str) and mood else None,
            valence=_number(payload.get("valence")),
            arousal=_number(payload.get("arousal")),
        )

    def with_mood(self, data: dict[str, Any]) -> "Presence":
        """Apply a ``mood`` event payload (``to_mood`` None = the mood ended); state/identity untouched."""
        to_mood = data.get("to_mood")
        if not isinstance(to_mood, str) or not to_mood:
            return replace(self, mood=None, valence=None, arousal=None)
        return replace(self, mood=to_mood, valence=_number(data.get("valence")), arousal=_number(data.get("arousal")))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["since"] = _iso(self.since)
        return d


@dataclass
class Transition:
    at: datetime
    from_state: str
    to_state: str
    identity_id: str | None
    display_name: str | None
    faces: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Transition":
        return cls(
            at=_parse_time(payload.get("at")) or datetime.now(timezone.utc),
            from_state=str(payload.get("from_state", "NO_SIGNAL")),
            to_state=str(payload.get("to_state", "NO_SIGNAL")),
            identity_id=payload.get("identity_id"),
            display_name=payload.get("display_name"),
            faces=int(payload.get("faces") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["at"] = _iso(self.at)
        return d


class PresenceStore:
    """Latest presence + recent transitions, safe to read from hook callbacks on any thread."""

    def __init__(self, *, history: int = 30) -> None:
        self._lock = threading.Lock()
        self.current = Presence()
        self.connected = False
        self.engine_available: bool | None = None
        self.last_frame_at: datetime | None = None
        self.hello_sequence = 0
        self.history: deque[Transition] = deque(maxlen=history)
        # Expression history as it came off the wire (raw dicts: labels, names, timestamps, rounded floats).
        # Bounded, in memory, mirrors Face2AI's own ring buffers — a mood end (``to_mood`` None) is history too.
        self.moods: deque[dict[str, Any]] = deque(maxlen=50)
        self.actions: deque[dict[str, Any]] = deque(maxlen=30)
        self.identity_count: int | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------ writers

    def apply(self, frame: SseFrame, now: datetime | None = None) -> Transition | None:
        """Apply one SSE frame; returns the transition when a `presence` frame arrived (for reactions)."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            self.last_frame_at = now
            if frame.event == "hello":
                self.connected = True
                self.last_error = None
                self.current = Presence.from_payload(frame.data.get("presence") or {})
                self.engine_available = frame.data.get("engine_available")
                seq = frame.data.get("last_sequence")
                self.hello_sequence = seq if isinstance(seq, int) else 0
                return None
            if frame.event == "heartbeat":
                payload = frame.data.get("presence") or {}
                if payload:
                    self.current = Presence.from_payload(payload)
                return None
            if frame.event == "presence":
                transition = Transition.from_payload(frame.data)
                self.history.append(transition)
                self.current = Presence(  # a fresh presence carries no mood
                    state=transition.to_state,
                    identity_id=transition.identity_id,
                    display_name=transition.display_name,
                    faces=transition.faces,
                    since=transition.at,
                )
                seq = frame.data.get("sequence")
                replayed = isinstance(seq, int) and seq <= self.hello_sequence
                return None if replayed else transition
            if frame.event == "store":
                count = frame.data.get("identity_count")
                if isinstance(count, int):
                    self.identity_count = count
                return None
            if frame.event == "mood":  # hint changed/ended: not a transition, never a reaction
                self.current = self.current.with_mood(frame.data)
                self.moods.append(dict(frame.data))
                return None
            if frame.event == "action":  # completed facial action: history only — never touches presence, never a reaction
                self.actions.append(dict(frame.data))
                return None
            if frame.event == "timeline_cleared":
                # Face2AI forgot its affect history on an explicit `POST /api/presence/reset`. That is a
                # privacy control, so the mirror forgets with it — a NO_SIGNAL transition alone would not
                # do (an ordinary presence expiry publishes one and keeps the history on both sides).
                self.moods.clear()
                self.actions.clear()
                return None
            return None

    def mark_lost(self, error: str) -> None:
        with self._lock:
            self.connected = False
            self.last_error = error

    # ------------------------------------------------------------------ readers

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self.connected,
                "engine_available": self.engine_available,
                "identity_count": self.identity_count,
                "last_frame_at": _iso(self.last_frame_at),
                "last_error": self.last_error,
                "presence": self.current.to_dict(),
                "history": [t.to_dict() for t in list(self.history)[-10:]],
                # dict(…) per entry, not just a new list: a caller that edits a snapshot entry in place
                # (host-side redaction, normalisation) must not rewrite the store's ring buffer.
                "moods": [dict(m) for m in list(self.moods)[-20:]],
                "actions": [dict(a) for a in list(self.actions)[-10:]],
            }

    def age_seconds(self, now: datetime | None = None) -> float | None:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if self.last_frame_at is None:
                return None
            return max(0.0, (now - self.last_frame_at).total_seconds())


def describe(store: PresenceStore, *, now: datetime | None = None, language: str = "de") -> str:
    """One or two plain sentences for the model — honest about what is known and what is not.

    Freshness is not one of them: ``context_line`` withholds the whole line when the last frame is
    older than its budget, which is the single place this plugin judges how fresh presence is.
    """
    now = now or datetime.now(timezone.utc)
    snap = store.snapshot()
    de = language.lower().startswith("de")
    if not snap["connected"]:
        return ("Face2AI (Kamera-Präsenz) ist gerade nicht verbunden; du kannst nicht sehen, wer da ist."
                if de else "Face2AI (camera presence) is not connected; you cannot see who is here.")
    p = snap["presence"]
    since = _parse_time(p.get("since"))
    seconds = int((now - since).total_seconds()) if since else None
    for_text = (f" seit etwa {seconds} s" if de else f" for about {seconds} s") if seconds is not None and seconds >= 0 else ""
    state = p["state"]
    if state == "NO_SIGNAL":
        text = "Die Kamera ist aus (kein Bildsignal)." if de else "The camera is off (no vision signal)."
    elif state == "NO_FACE":
        text = f"Die Kamera läuft, aber niemand steht davor{for_text}." if de else f"The camera is on but nobody is in front of it{for_text}."
    elif state == "UNKNOWN":
        text = (f"Eine Person steht vor der Kamera{for_text}, Face2AI kennt sie nicht (nicht enrollt). Rate keinen Namen."
                if de else f"One person is in front of the camera{for_text}; Face2AI does not recognize them (not enrolled). Do not guess a name.")
    elif state == "KNOWN":
        name = p.get("display_name") or ("eine bekannte Person" if de else "a known person")
        text = (f"{name} steht vor der Kamera{for_text} (von Face2AI erkannt, beste Übereinstimmung, keine Gewissheit)."
                if de else f"{name} is in front of the camera{for_text} (recognized by Face2AI — best match, not certainty).")
    else:
        text = (f"{p.get('faces', 0)} Personen stehen vor der Kamera{for_text}; bei mehreren Gesichtern ordnet Face2AI keine Identität zu."
                if de else f"{p.get('faces', 0)} people are in front of the camera{for_text}; Face2AI does not attribute identity with several faces.")
    # No freshness sentence here: this plugin's freshness owner is ``context_line(max_age_seconds=…)``,
    # which withholds the whole line from its own clock (``last_frame_at``) rather than annotating it.
    if snap.get("engine_available") is False:
        text += " Die Erkennungs-Engine meldet sich als nicht verfügbar." if de else " The recognition engine reports itself unavailable."
    mood = mood_sentence(p.get("mood"), p.get("valence"), p.get("arousal"), language=language, subject=p.get("display_name") if state == "KNOWN" else None)
    if mood:
        text += " " + mood
    recent = [t for t in snap["history"] if t["to_state"] == "KNOWN" and t["display_name"] and t["identity_id"] != p.get("identity_id")][-4:]
    if recent:
        names = ", ".join(f"{t['display_name']} ({(_parse_time(t['at']) or now).astimezone().strftime('%H:%M')})" for t in recent)
        text += (f" Zuletzt gesehen: {names}." if de else f" Recently seen: {names}.")
    return text


def context_line(store: PresenceStore, *, now: datetime | None = None, language: str = "de", max_age_seconds: int = 30) -> str | None:
    """The line injected into a turn, or None when the information is not trustworthy enough."""
    age = store.age_seconds(now)
    if age is None or age > max_age_seconds:
        return None
    return f"[face2ai] {describe(store, now=now, language=language)}"
