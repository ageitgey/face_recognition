from __future__ import annotations

import threading

from face2ai_app.adapters.frame_decode import decode_frame
from face2ai_app.config import DEFAULT_MAX_FRAME_PIXELS
from face2ai_app.domain.errors import InvalidFrame, RecognitionUnavailable
from face2ai_app.domain.models import DetectedFace, FaceBox

# ``face_recognition`` keeps its dlib detector, shape predictor and encoder in module-level
# objects: process-global mutable state that every engine instance and every thread shares. The
# API runs detect() in a 40-worker threadpool, so two browser tabs are enough to call into them
# concurrently. Measured on this machine, unlocked, 4 threads x 24 calls on a cold interpreter:
# SIGSEGV in 7 of 10 runs. The lock is therefore module-level, not per-instance — a second engine
# object would otherwise reintroduce exactly the race it is meant to prevent.
_DLIB_LOCK = threading.Lock()


class FaceRecognitionEngine:
    """Recognition through ``face_recognition`` (dlib), honouring the port contract in
    ``face2ai_app.ports.recognition``. See ``apps/face2ai/tests/test_port_conformance.py``."""

    def __init__(self, max_frame_pixels: int = DEFAULT_MAX_FRAME_PIXELS) -> None:
        self._max_frame_pixels = max_frame_pixels
        self._module = None
        self._reason: str | None = None
        try:
            import face_recognition  # type: ignore
            self._module = face_recognition
        except Exception as exc:
            self._reason = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._module is not None

    @property
    def availability_reason(self) -> str | None:
        return self._reason

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        if self._module is None:
            raise RecognitionUnavailable(self._reason or "face_recognition engine unavailable")
        # Bounds and EXIF are the shared decoder's business; InvalidFrame passes straight through.
        image = decode_frame(image_bytes, self._max_frame_pixels)
        try:
            with _DLIB_LOCK:  # held across both calls: they share the same global dlib objects
                boxes = self._module.face_locations(image, model="hog")
                encodings = self._module.face_encodings(image, boxes, model="small")
        except Exception as exc:
            raise InvalidFrame(f"unable to analyze frame: {exc}") from exc
        return [
            DetectedFace(
                box=FaceBox(top=top, right=right, bottom=bottom, left=left),
                encoding=[float(value) for value in encoding],
            )
            for (top, right, bottom, left), encoding in zip(boxes, encodings, strict=True)
        ]
