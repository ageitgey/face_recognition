from __future__ import annotations

from face2ai_app.domain.models import Expression, FaceBox


class NullExpressionEngine:
    """Stands in when the expression extra is not installed or the feature is off: never crashes, never analyzes."""

    def __init__(self, reason: str = "expression engine disabled") -> None:
        self._reason = reason

    @property
    def available(self) -> bool:
        return False

    @property
    def availability_reason(self) -> str | None:
        return self._reason

    def analyze(self, image_bytes: bytes, boxes: list[FaceBox]) -> list[Expression | None]:
        return [None for _ in boxes]
