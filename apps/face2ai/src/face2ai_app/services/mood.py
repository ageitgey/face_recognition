from __future__ import annotations

import threading
from datetime import datetime, timezone

from face2ai_app.domain.models import EMOTIONS, Expression, MoodTransition


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MoodTracker:
    """Turns per-frame expression scores into a stable mood with hysteresis.

    Per frame the emotion ``scores`` are smoothed with an EMA (``alpha``; labels missing from a
    frame count as 0.0; ``expression.dominant`` is ignored — the EMA over scores decides). The
    argmax of the EMA is the *candidate*; it becomes the mood once it has been the candidate for
    ``stable_ticks`` consecutive frames and its EMA is at least ``min_score``. Only a candidate
    that differs from the current mood produces a transition, so a committed mood survives brief
    flicker until another label wins the same race.

    The tracker follows the presence it decorates: a change of ``presence_key`` (state or
    identity changed) resets the smoothing immediately and ends the current mood; ``stable_ticks``
    consecutive frames without an expression end it as well. Either produces one
    ``mood -> None`` transition, and only if a mood was set. Frames without an expression (``None``
    or empty ``scores``) never touch the EMA or the streak; they only count towards that reset.

    Valence/arousal are smoothed alongside: ``current()`` and the transition carry them *frozen at
    commit time* for the mood event (one change per mood, not per frame); ``affect()`` returns the
    live EMA for ``Presence`` (Stage 2).
    """

    def __init__(self, stable_ticks: int = 3, min_score: float = 0.5, alpha: float = 0.5) -> None:
        if stable_ticks < 1:
            raise ValueError("stable_ticks must be >= 1")
        if not 0.0 < min_score <= 1.0:
            raise ValueError("min_score must be in (0, 1]")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._stable_ticks = stable_ticks
        self._min_score = min_score
        self._alpha = alpha
        self._lock = threading.Lock()
        # Whose mood this is (mirrors the presence being decorated).
        self._presence_key: str | None = None
        self._identity_id: str | None = None
        self._display_name: str | None = None
        # Committed mood; valence/arousal frozen at commit time (the mood event's values).
        self._mood: str | None = None
        self._valence: float | None = None
        self._arousal: float | None = None
        # Smoothing state, reset by _clear_smoothing().
        self._ema: dict[str, float] = {}
        self._ema_valence: float | None = None
        self._ema_arousal: float | None = None
        self._candidate: str | None = None
        self._candidate_count: int = 0
        self._missing_count: int = 0
        self._clear_smoothing()

    def _clear_smoothing(self) -> None:
        self._ema = dict.fromkeys(EMOTIONS, 0.0)
        self._ema_valence = None
        self._ema_arousal = None
        self._candidate = None
        self._candidate_count = 0
        self._missing_count = 0

    def _clear_mood(self, now: datetime) -> MoodTransition | None:
        """Drop smoothing and mood; announce the end of the mood if one was set."""
        self._clear_smoothing()
        if self._mood is None:
            return None
        transition = MoodTransition(
            at=now,
            identity_id=self._identity_id,
            display_name=self._display_name,
            from_mood=self._mood,
            to_mood=None,
        )
        self._mood = self._valence = self._arousal = None
        return transition

    def _smooth(self, previous: float | None, value: float) -> float:
        """One EMA step; an estimate that has not started yet (``None``) counts as 0.0."""
        return self._alpha * value + (1.0 - self._alpha) * (previous if previous is not None else 0.0)

    def observe(
        self,
        presence_key: str,
        expression: Expression | None,
        now: datetime | None = None,
        *,
        identity_id: str | None = None,
        display_name: str | None = None,
    ) -> MoodTransition | None:
        """Record one frame's expression for the given presence; return a transition on mood change."""
        now = now or _now()
        with self._lock:
            ended = None
            if presence_key != self._presence_key:
                ended = self._clear_mood(now)
                self._presence_key = presence_key
            self._identity_id = identity_id
            self._display_name = display_name

            if expression is None or not expression.scores:
                self._missing_count += 1
                if self._missing_count >= self._stable_ticks and ended is None:
                    ended = self._clear_mood(now)
                return ended
            self._missing_count = 0

            for label in EMOTIONS:
                self._ema[label] = self._smooth(self._ema[label], expression.scores.get(label, 0.0))
            if expression.valence is not None:
                self._ema_valence = self._smooth(self._ema_valence, expression.valence)
            if expression.arousal is not None:
                self._ema_arousal = self._smooth(self._ema_arousal, expression.arousal)

            candidate = max(EMOTIONS, key=self._ema.__getitem__)
            if candidate == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = candidate
                self._candidate_count = 1

            if ended is not None:
                return ended  # a presence change is announced first; the new mood needs its own streak
            if (
                candidate == self._mood
                or self._candidate_count < self._stable_ticks
                or self._ema[candidate] < self._min_score
            ):
                return None
            return self._commit(candidate, now)

    def _commit(self, mood: str, now: datetime) -> MoodTransition:
        transition = MoodTransition(
            at=now,
            identity_id=self._identity_id,
            display_name=self._display_name,
            from_mood=self._mood,
            to_mood=mood,
            valence=None if self._ema_valence is None else round(self._ema_valence, 3),
            arousal=None if self._ema_arousal is None else round(self._ema_arousal, 3),
        )
        self._mood, self._valence, self._arousal = mood, transition.valence, transition.arousal
        return transition

    def current(self) -> tuple[str | None, float | None, float | None]:
        """``(mood, valence, arousal)`` as committed — the mood label ``Presence`` carries and the values
        frozen on the ``mood`` event; ``Presence.valence/arousal`` come from ``affect()`` (Stage 2)."""
        with self._lock:
            return self._mood, self._valence, self._arousal

    def affect(self) -> tuple[float | None, float | None]:
        """Live smoothed ``(valence, arousal)`` (rounded to 3), None until the first readable frame — what
        ``Presence`` carries since Stage 2. ``current()`` keeps the commit-frozen values for the mood event."""
        with self._lock:
            v = None if self._ema_valence is None else round(self._ema_valence, 3)
            a = None if self._ema_arousal is None else round(self._ema_arousal, 3)
            return v, a

    def reset(self, now: datetime | None = None) -> MoodTransition | None:
        """Forget everything (camera stopped, presence expired or reset).

        Returns the ``mood -> None`` transition if a mood was set (so the wire learns the mood
        ended, symmetric with a frame-driven end), None otherwise; the smoothing is cleared either way.
        """
        with self._lock:
            ended = self._clear_mood(now or _now())
            self._presence_key = None
            self._identity_id = self._display_name = None
            return ended
