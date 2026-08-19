from __future__ import annotations

from typing import Protocol

from face2ai_app.domain.models import Expression, FaceBox


class ExpressionEngine(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def availability_reason(self) -> str | None: ...

    def analyze(self, image_bytes: bytes, boxes: list[FaceBox]) -> list[Expression | None]:
        """One Expression (or None) per box, same order. Boxes are in the pixel space of image_bytes."""
        ...
