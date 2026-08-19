from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# Decoded-pixel budget per frame, enforced by every adapter *before* the frame is decoded (a 0.19 MiB
# PNG can declare gigabytes of pixels). The browser's own snapshot is 640 px wide, so 4 MP is ~16x
# generous. See apps/face2ai/tests/test_port_conformance.py for the obligation this number serves.
DEFAULT_MAX_FRAME_PIXELS = 4_000_000


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var; unset/empty -> default, anything but true/false spellings fails visibly."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return default
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean (true/false), got {raw!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    match_tolerance: float = 0.6
    max_frame_bytes: int = 5 * 1024 * 1024
    max_frame_pixels: int = DEFAULT_MAX_FRAME_PIXELS
    data_dir: Path = Path.home() / ".face2ai"
    greeting_cooldown_seconds: int = 15
    presence_stable_ticks: int = 2
    presence_stale_seconds: float = 5.0
    events_heartbeat_seconds: float = 15.0
    events_buffer_size: int = 200
    expression_enabled: bool = False
    expression_models_dir: Path = Path.home() / ".face2ai" / "models"  # == default data_dir / "models"
    mood_stable_ticks: int = 3
    mood_min_score: float = 0.5
    action_on_threshold: float = 0.35  # blendshape group mean that starts a facial action
    action_off_threshold: float = 0.2  # ... and ends it (hysteresis; == compact_blendshapes floor)
    action_min_frames: int = 2  # frames an action must persist before it is reported
    timeline_seconds: int = 600  # in-memory affect history window (never persisted)

    def __post_init__(self) -> None:
        """Range-check every numeric knob. The ranges are the ones README.md prints — one owner.

        ``tests/test_config.py::test_every_numeric_setting_is_validated`` fails when a numeric field
        is added to this dataclass without a line here, which is the only thing that stops the
        per-commit habit that left four knobs (``port``, ``match_tolerance``, ``max_frame_bytes``,
        ``greeting_cooldown_seconds``) unchecked while eleven were checked (measured at `3857adc`:
        15 numeric fields, 11 range-checked, 4 not).
        """
        if not 1 <= self.port <= 65535:
            raise ValueError("FACE2AI_PORT must be in 1..65535")
        if not 0 < self.match_tolerance <= 2:
            # The knob that decides who you are: <= 0 recognises nobody, a large value everybody.
            raise ValueError("FACE2AI_MATCH_TOLERANCE must be in (0, 2]")
        if self.max_frame_bytes < 1:
            raise ValueError("FACE2AI_MAX_FRAME_BYTES must be >= 1")
        if self.greeting_cooldown_seconds < 0:
            raise ValueError("FACE2AI_GREETING_COOLDOWN_SECONDS must be >= 0")
        if self.max_frame_pixels < 1:
            raise ValueError("FACE2AI_MAX_FRAME_PIXELS must be >= 1")
        if self.presence_stale_seconds <= 0:
            raise ValueError("FACE2AI_PRESENCE_STALE_SECONDS must be > 0")
        if self.events_heartbeat_seconds <= 0:
            raise ValueError("FACE2AI_EVENTS_HEARTBEAT_SECONDS must be > 0")
        if self.presence_stable_ticks < 1:
            raise ValueError("FACE2AI_PRESENCE_STABLE_TICKS must be >= 1")
        if self.events_buffer_size < 1:
            raise ValueError("FACE2AI_EVENTS_BUFFER_SIZE must be >= 1")
        if self.mood_stable_ticks < 1:
            raise ValueError("FACE2AI_MOOD_STABLE_TICKS must be >= 1")
        if not 0 < self.mood_min_score <= 1:
            raise ValueError("FACE2AI_MOOD_MIN_SCORE must be in (0, 1]")
        if not 0 < self.action_off_threshold < self.action_on_threshold <= 1:
            raise ValueError("FACE2AI_ACTION_OFF_THRESHOLD must be > 0 and < FACE2AI_ACTION_ON_THRESHOLD <= 1")
        if self.action_min_frames < 1:
            raise ValueError("FACE2AI_ACTION_MIN_FRAMES must be >= 1")
        if self.timeline_seconds < 10:
            raise ValueError("FACE2AI_TIMELINE_SECONDS must be >= 10")

    @property
    def identity_store_path(self) -> Path:
        return self.data_dir / "identities.json"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("FACE2AI_DATA_DIR", str(Path.home() / ".face2ai"))).expanduser()
        return cls(
            host=os.getenv("FACE2AI_HOST", "127.0.0.1"),
            port=int(os.getenv("FACE2AI_PORT", "8765")),
            match_tolerance=float(os.getenv("FACE2AI_MATCH_TOLERANCE", "0.6")),
            max_frame_bytes=int(os.getenv("FACE2AI_MAX_FRAME_BYTES", str(5 * 1024 * 1024))),
            max_frame_pixels=int(os.getenv("FACE2AI_MAX_FRAME_PIXELS", str(DEFAULT_MAX_FRAME_PIXELS))),
            data_dir=data_dir,
            greeting_cooldown_seconds=int(os.getenv("FACE2AI_GREETING_COOLDOWN_SECONDS", "15")),
            presence_stable_ticks=int(os.getenv("FACE2AI_PRESENCE_STABLE_TICKS", "2")),
            presence_stale_seconds=float(os.getenv("FACE2AI_PRESENCE_STALE_SECONDS", "5")),
            events_heartbeat_seconds=float(os.getenv("FACE2AI_EVENTS_HEARTBEAT_SECONDS", "15")),
            events_buffer_size=int(os.getenv("FACE2AI_EVENTS_BUFFER_SIZE", "200")),
            expression_enabled=_env_bool("FACE2AI_EXPRESSION_ENABLED", False),
            expression_models_dir=Path(
                os.getenv("FACE2AI_EXPRESSION_MODELS_DIR", str(data_dir / "models"))
            ).expanduser(),
            mood_stable_ticks=int(os.getenv("FACE2AI_MOOD_STABLE_TICKS", "3")),
            mood_min_score=float(os.getenv("FACE2AI_MOOD_MIN_SCORE", "0.5")),
            action_on_threshold=float(os.getenv("FACE2AI_ACTION_ON_THRESHOLD", "0.35")),
            action_off_threshold=float(os.getenv("FACE2AI_ACTION_OFF_THRESHOLD", "0.2")),
            action_min_frames=int(os.getenv("FACE2AI_ACTION_MIN_FRAMES", "2")),
            timeline_seconds=int(os.getenv("FACE2AI_TIMELINE_SECONDS", "600")),
        )
