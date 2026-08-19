"""Facial action dynamics from blendshape intensities: onset -> apex -> offset per action group.

Pure and per-presence like ``MoodTracker``. Each action is the mean of a small blendshape group;
a hysteresis state machine (``on_threshold`` to start, ``off_threshold`` to end) turns the per-frame
series into one ``ActionEvent`` per completed action. Timing is quantized to the frame rate — at the
browser's loop (~1.7 fps) that is ~0.6 s, so these are expression *dynamics*, not micro-expressions;
the event says how long and how strong, never why. Speech articulators (jawOpen, mouthFunnel,
mouthPucker, mouthClose) and blinks are deliberately not action groups.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from face2ai_app.domain.models import ACTIONS, ActionEvent, Expression

ACTION_GROUPS: dict[str, tuple[str, ...]] = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "frown": ("mouthFrownLeft", "mouthFrownRight"),
    "brow_raise": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "brow_furrow": ("browDownLeft", "browDownRight"),
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "eyes_wide": ("eyeWideLeft", "eyeWideRight"),
    "nose_wrinkle": ("noseSneerLeft", "noseSneerRight"),
    "lip_press": ("mouthPressLeft", "mouthPressRight"),
}
assert tuple(ACTION_GROUPS) == ACTIONS  # the wire vocabulary (domain) and the groups must agree


def _now() -> datetime:
    return datetime.now(timezone.utc)


def action_intensity(blendshapes: Mapping[str, float], action: str) -> float:
    """Mean intensity of the action's blendshape group; blendshapes missing from a frame count as 0.

    ``Expression.blendshapes`` is compacted (only entries >= 0.2 survive), so "missing" is the normal
    way a low intensity arrives — an empty dict is a readable, all-zero (neutral) frame.
    """
    group = ACTION_GROUPS[action]
    return sum(float(blendshapes.get(name, 0.0)) for name in group) / len(group)


@dataclass
class _Active:
    onset_at: datetime
    apex_at: datetime
    peak: float
    frames: int


class ActionTracker:
    """Hysteresis state machine per action group over ``Expression.blendshapes``.

    Per frame every group's intensity (``action_intensity``) is compared against two thresholds: an
    inactive action starts when it reaches ``on_threshold``; an active one keeps going (frames counted,
    peak/apex updated) while it stays at or above ``off_threshold`` and completes on the first frame
    below it. Completion yields an ``ActionEvent`` (onset/apex/offset timestamps, peak, duration,
    frames) — but only if the action lasted at least ``min_frames`` frames; shorter spikes are
    swallowed as noise. Several actions can be active at once (smile + eye squint).

    Like ``MoodTracker`` the tracker follows the presence it decorates: a change of ``presence_key``
    (state or identity changed) drops whatever was active *without* an event — the offset is unknown
    and is never guessed. An unreadable frame (``expression is None``) drops active actions the same
    way. Timestamps are those of the observed frames, so onset/offset resolution equals the frame
    interval (~0.6 s from the browser loop); onset and offset each carry one frame of uncertainty,
    so ``duration_ms`` (offset - onset, clamped at 0 should the wall clock step back) is accurate
    to about +/- 1 frame. On equal peaks the first apex wins.
    """

    def __init__(self, on_threshold: float = 0.35, off_threshold: float = 0.2, min_frames: int = 2) -> None:
        if not 0.0 < off_threshold < on_threshold <= 1.0:
            raise ValueError("need 0 < off_threshold < on_threshold <= 1")
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        self._on = on_threshold
        self._off = off_threshold
        self._min_frames = min_frames
        self._lock = threading.Lock()
        # Whose actions these are (mirrors the presence being decorated).
        self._presence_key: str | None = None
        self._active: dict[str, _Active] = {}

    def observe(
        self,
        presence_key: str,
        expression: Expression | None,
        now: datetime | None = None,
        *,
        identity_id: str | None = None,
        display_name: str | None = None,
    ) -> list[ActionEvent]:
        """Record one frame's expression for the given presence; return the actions completed by it."""
        now = now or _now()
        with self._lock:
            if presence_key != self._presence_key:
                self._active.clear()  # unknown offset for whatever was active: dropped, never guessed
                self._presence_key = presence_key
            if expression is None:
                self._active.clear()  # unreadable frame: dropped, not completed
                return []
            done: list[ActionEvent] = []
            for action in ACTIONS:
                value = action_intensity(expression.blendshapes, action)
                active = self._active.get(action)
                if active is None:
                    if value >= self._on:
                        self._active[action] = _Active(onset_at=now, apex_at=now, peak=value, frames=1)
                    continue
                if value >= self._off:
                    active.frames += 1
                    if value > active.peak:
                        active.peak, active.apex_at = value, now
                    continue
                del self._active[action]
                if active.frames >= self._min_frames:
                    done.append(
                        ActionEvent(
                            at=now,
                            identity_id=identity_id,
                            display_name=display_name,
                            action=action,
                            onset_at=active.onset_at,
                            apex_at=active.apex_at,
                            offset_at=now,
                            peak=round(min(active.peak, 1.0), 3),
                            duration_ms=max(0, int((now - active.onset_at).total_seconds() * 1000)),
                            frames=active.frames,
                        )
                    )
            return done

    def reset(self) -> None:
        """Forget everything (camera stopped, presence expired or reset).

        Active actions are dropped, not completed — their offset is unknown, so no event is produced.
        """
        with self._lock:
            self._active.clear()
            self._presence_key = None
