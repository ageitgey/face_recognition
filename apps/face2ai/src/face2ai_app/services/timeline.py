from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timedelta, timezone

from face2ai_app.domain.models import ActionEvent, AffectSample, MoodTransition, TimelineSnapshot


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AffectHistory:
    """Bounded in-memory affect timeline: live valence/arousal samples, mood changes, facial actions.

    Three ring buffers (``collections.deque(maxlen=…)``) hold the last ``max_samples`` ``AffectSample``s,
    the last ``max_moods`` ``MoodTransition``s and the last ``max_actions`` ``ActionEvent``s. Samples are
    additionally bounded by age: ``record_sample()`` drops leading samples older than ``max_seconds``
    relative to the newest sample, so the buffer never holds more than one window even at a slow
    frame rate. ``snapshot()`` answers ``GET /api/expression/timeline`` with everything inside
    ``[now - seconds, now]`` (inclusive on both ends), optionally narrowed to one identity.

    This history lives **in memory only**: nothing is ever persisted, and it is cleared on
    ``POST /api/presence/reset`` and on restart (``clear()``). It stores what the wire already
    carries — timestamps, labels, names, rounded scalars — never frames, landmarks or per-frame
    scores. Like everything expression-related it is a hint, never a fact, and gates nothing.
    """

    def __init__(
        self,
        max_seconds: int = 600,
        max_samples: int = 2000,
        max_moods: int = 50,
        max_actions: int = 100,
    ) -> None:
        if max_seconds < 1:
            raise ValueError("max_seconds must be >= 1")
        if min(max_samples, max_moods, max_actions) < 1:
            raise ValueError("max_samples, max_moods and max_actions must be >= 1")
        self._max_seconds = max_seconds
        self._max_age = timedelta(seconds=max_seconds)
        self._lock = threading.Lock()
        self._samples: deque[AffectSample] = deque(maxlen=max_samples)
        self._moods: deque[MoodTransition] = deque(maxlen=max_moods)
        self._actions: deque[ActionEvent] = deque(maxlen=max_actions)

    @property
    def max_seconds(self) -> int:
        return self._max_seconds

    def record_sample(self, sample: AffectSample) -> None:
        """Append one affect sample and drop leading samples older than ``max_seconds`` before it."""
        with self._lock:
            newest = max(sample.at, self._samples[-1].at) if self._samples else sample.at
            self._samples.append(sample)
            oldest_allowed = newest - self._max_age  # prune against the newest seen; an out-of-order sample never un-prunes
            while self._samples and self._samples[0].at < oldest_allowed:
                self._samples.popleft()

    def record_mood(self, transition: MoodTransition) -> None:
        """Remember one committed mood change (bounded by ``max_moods``)."""
        with self._lock:
            self._moods.append(transition)

    def record_action(self, event: ActionEvent) -> None:
        """Remember one completed facial action (bounded by ``max_actions``)."""
        with self._lock:
            self._actions.append(event)

    def snapshot(
        self,
        seconds: int | None = None,
        identity_id: str | None = None,
        now: datetime | None = None,
    ) -> TimelineSnapshot:
        """Everything recorded within ``[now - seconds, now]``, oldest first.

        ``seconds`` defaults to ``max_seconds``; ``now`` defaults to the newest entry across all three
        buffers (so a paused camera still shows its last window) or, without any entry, to the current
        UTC time. Anchoring on samples alone would hide every mood end and action that lands after the
        last frame — exactly what a toggle-off, an expiry or a reset produces.
        With ``identity_id`` only samples, moods and actions of that identity are returned.
        """
        window = self._max_seconds if seconds is None else seconds
        with self._lock:
            if now is None:
                newest = [buffer[-1].at for buffer in (self._samples, self._moods, self._actions) if buffer]
                now = max(newest) if newest else _now()
            start = now - timedelta(seconds=window)

            def keep(entry: AffectSample | MoodTransition | ActionEvent) -> bool:
                if not start <= entry.at <= now:
                    return False
                return identity_id is None or entry.identity_id == identity_id

            return TimelineSnapshot(
                seconds=window,
                samples=[s for s in self._samples if keep(s)],
                moods=[m for m in self._moods if keep(m)],
                actions=[a for a in self._actions if keep(a)],
            )

    def clear(self) -> int:
        """Forget everything (presence reset, restart) — the timeline never outlives the process.

        Returns how many entries were dropped, so the caller can tell a real "forget" from a no-op
        (mirrors on other machines only learn about a clear when there was something to clear).
        """
        with self._lock:
            dropped = len(self._samples) + len(self._moods) + len(self._actions)
            self._samples.clear()
            self._moods.clear()
            self._actions.clear()
            return dropped
