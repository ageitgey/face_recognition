"""Consume Face2AI presence/store/mood events (Server-Sent Events) and keep a small memory of
who is in front of the camera. This module never sees frames or encodings — only states,
display names, timestamps and a hedged mood hint, exactly what the Face2AI event stream publishes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("face2ai_agent.presence")

PRESENCE_STATES = ("NO_SIGNAL", "NO_FACE", "UNKNOWN", "KNOWN", "MULTIPLE_FACES")

# Hedged wording for Face2AI's mood labels (domain/models.py EMOTIONS). Same literal table as the
# Face2AI UI (static/js/model.js) and the Hermes plugin: a mood is how a face *appears* ("wirkt …" /
# "looks …"), never how someone *is* — never a finding, never a fact, never a gate for anything.
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
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def mood_sentence(mood: str | None, valence: float | None, arousal: float | None, *, language: str = "en", subject: str | None = None) -> str:
    """One hedged sentence for a mood hint, e.g. "Ben wirkt fröhlich (Valenz +0.6, Erregung +0.1) – nur ein
    Hinweis aus dem Gesichtsausdruck, keine Tatsache." Empty string when there is no mood. Unknown labels
    are hedged too (lower-cased) and never raise."""
    if not isinstance(mood, str) or not mood:
        return ""
    words = MOOD_WORDS["de"] if language.lower().startswith("de") else MOOD_WORDS["en"]
    label = words["labels"].get(mood, mood.lower())
    numbers = [f"{words[key]} {round(value, 1) + 0.0:+.1f}" for key, value in (("valence", valence), ("arousal", arousal)) if value is not None]
    detail = f" ({', '.join(numbers)})" if numbers else ""
    dash = "–" if words is MOOD_WORDS["de"] else "—"
    return f"{subject or words['generic_subject']} {words['prefix']}{label}{detail} {dash} {words['hedge']}."


@dataclass(frozen=True, slots=True)
class SseFrame:
    event: str
    data: dict[str, Any]
    id: str | None = None


def parse_sse(lines: Iterable[str]) -> Iterable[SseFrame]:
    """Minimal SSE parser: yields one frame per blank-line-terminated block."""
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


@dataclass(frozen=True, slots=True)
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
        state = payload.get("state") if payload.get("state") in PRESENCE_STATES else "NO_SIGNAL"
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


@dataclass(frozen=True, slots=True)
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

    def as_presence(self) -> Presence:
        return Presence(
            state=self.to_state,
            identity_id=self.identity_id,
            display_name=self.display_name,
            faces=self.faces,
            since=self.at,
        )


@dataclass(frozen=True, slots=True)
class StoreChange:
    at: datetime
    kind: str  # enrolled | deleted | erased
    identity_id: str | None
    display_name: str | None
    identity_count: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StoreChange":
        return cls(
            at=_parse_time(payload.get("at")) or datetime.now(timezone.utc),
            kind=str(payload.get("kind", "")),
            identity_id=payload.get("identity_id"),
            display_name=payload.get("display_name"),
            identity_count=int(payload.get("identity_count") or 0),
        )


@dataclass
class PresenceMemory:
    """What the agent knows about presence: current state, recent transitions, store changes."""

    current: Presence = field(default_factory=Presence)
    connected: bool = False
    engine_available: bool | None = None
    greeting_cooldown_seconds: int = 15
    history: deque[Transition] = field(default_factory=lambda: deque(maxlen=20))
    store_changes: deque[StoreChange] = field(default_factory=lambda: deque(maxlen=10))
    identity_count: int | None = None
    hello_sequence: int = 0  # events with sequence <= this were replayed after (re)connect, not live

    def apply_hello(self, data: dict[str, Any]) -> None:
        self.connected = True
        self.current = Presence.from_payload(data.get("presence") or {})
        self.engine_available = data.get("engine_available")
        seq = data.get("last_sequence")
        self.hello_sequence = seq if isinstance(seq, int) else 0
        cooldown = data.get("greeting_cooldown_seconds")
        if isinstance(cooldown, int) and cooldown >= 0:
            self.greeting_cooldown_seconds = cooldown

    def situation_key(self) -> tuple:
        """Changes only when the prompt-relevant situation changes (not every heartbeat)."""
        p = self.current
        return (self.connected, p.state, p.identity_id, self.engine_available, self.identity_count)

    def is_replayed(self, frame_data: dict[str, Any]) -> bool:
        seq = frame_data.get("sequence")
        return isinstance(seq, int) and seq <= self.hello_sequence

    def apply_heartbeat(self, data: dict[str, Any]) -> None:
        payload = data.get("presence") or {}
        if payload:
            self.current = Presence.from_payload(payload)

    def apply_transition(self, transition: Transition) -> None:
        self.history.append(transition)
        self.current = transition.as_presence()  # a fresh presence carries no mood

    def apply_mood(self, data: dict[str, Any]) -> None:
        """SSE ``mood``: update the hint on the current presence — no transition, no greeting, no refresh."""
        self.current = self.current.with_mood(data)

    def apply_store_change(self, change: StoreChange) -> None:
        self.store_changes.append(change)
        self.identity_count = change.identity_count

    def describe(self, now: datetime | None = None, *, language: str = "en") -> str:
        """Plain-language situation report for the LLM system prompt / tools.

        The report itself is English; ``language`` only picks the wording of the hedged mood hint
        ("wirkt fröhlich" / "looks happy") so the model can reuse it verbatim in the spoken language.
        """
        now = now or datetime.now(timezone.utc)
        if not self.connected:
            return "The Face2AI camera service is not connected; you cannot see who is here."
        p = self.current
        if p.state == "NO_SIGNAL":
            head = "The camera is currently off (no vision signal)."
        elif p.state == "NO_FACE":
            head = "The camera is on but nobody is in front of it right now."
        elif p.state == "UNKNOWN":
            head = "One person is in front of the camera, but Face2AI does not recognize them (not enrolled)."
        elif p.state == "KNOWN":
            head = f"{p.display_name or 'A known person'} is in front of the camera (recognized by Face2AI)."
        else:
            head = f"{p.faces} people are in front of the camera; Face2AI cannot attribute identity while several faces are visible."
        if p.since is not None:
            seconds = max(0, int((now - p.since).total_seconds()))
            head += f" This has been the case for about {seconds} seconds."
        # No freshness sentence: Face2AI expires a presence without frames to NO_SIGNAL itself, and
        # this memory owns no clock over the SSE stream to make a second claim from.
        if self.engine_available is False:
            head += " The recognition engine reports itself unavailable."
        mood = mood_sentence(p.mood, p.valence, p.arousal, language=language, subject=p.display_name if p.state == "KNOWN" else None)
        if mood:
            head += " " + mood
        recent = [
            t for t in list(self.history)[-8:]
            if t.to_state == "KNOWN" and t.display_name and t.identity_id != p.identity_id
        ][-5:]
        if recent:
            parts = [f"{t.display_name} at {t.at.astimezone().strftime('%H:%M:%S')}" for t in recent]
            head += " Recently seen: " + "; ".join(parts) + "."
        if self.identity_count is not None:
            head += f" {self.identity_count} people are enrolled locally."
        return head


class PresenceClient:
    """Long-lived SSE subscription to ``GET /api/events?role=agent`` with reconnect + resume."""

    def __init__(self, base_url: str, *, role: str = "agent", reconnect_delay: float = 2.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._role = role
        self._reconnect_delay = reconnect_delay
        self._last_id: str | None = None
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def frames(self) -> AsyncIterator[SseFrame]:
        """Yield frames forever (until stop()), reconnecting with Last-Event-ID on failure."""
        while not self._stop.is_set():
            headers = {"Accept": "text/event-stream"}
            if self._last_id:
                headers["Last-Event-ID"] = self._last_id
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
                    async with client.stream(
                        "GET", f"{self._base_url}/api/events", params={"role": self._role}, headers=headers
                    ) as response:
                        response.raise_for_status()
                        async for frame in _aiter_frames(response.aiter_lines()):
                            if frame.id:
                                self._last_id = frame.id
                            yield frame
                            if self._stop.is_set():
                                return
            except (httpx.HTTPError, OSError) as exc:
                logger.warning("presence stream lost (%s); reconnecting in %.1fs", exc, self._reconnect_delay)
                yield SseFrame(event="lost", data={"error": str(exc)})
            if self._stop.is_set():
                return
            await asyncio.sleep(self._reconnect_delay)


async def _aiter_frames(lines: AsyncIterator[str]) -> AsyncIterator[SseFrame]:
    buffer: list[str] = []
    async for line in lines:
        buffer.append(line)
        if line.rstrip("\r") == "":
            for frame in parse_sse(buffer):
                yield frame
            buffer = []


Handler = Callable[[str, Any, bool], Awaitable[None]]


async def run_presence_loop(client: PresenceClient, memory: PresenceMemory, on_event: Handler | None = None) -> None:
    """Feed frames into memory and notify ``on_event(kind, payload, replayed)``.

    ``replayed`` is True for events the server re-sent after a (re)connect (sequence <= hello's
    last_sequence): they update memory/history but should not be spoken as if they just happened.
    """
    async for frame in client.frames():
        if frame.event == "hello":
            memory.apply_hello(frame.data)
            if on_event:
                await on_event("hello", memory.current, False)
        elif frame.event == "presence":
            transition = Transition.from_payload(frame.data)
            memory.apply_transition(transition)
            if on_event:
                await on_event("presence", transition, memory.is_replayed(frame.data))
        elif frame.event == "store":
            change = StoreChange.from_payload(frame.data)
            memory.apply_store_change(change)
            if on_event:
                await on_event("store", change, memory.is_replayed(frame.data))
        elif frame.event == "heartbeat":
            memory.apply_heartbeat(frame.data)
            if on_event:
                await on_event("heartbeat", memory.current, False)
        elif frame.event == "mood":
            memory.apply_mood(frame.data)
            if on_event:
                await on_event("mood", frame.data, memory.is_replayed(frame.data))
        elif frame.event == "lost":
            memory.connected = False
            if on_event:
                await on_event("lost", frame.data, False)
