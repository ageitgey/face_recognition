from __future__ import annotations

from typing import Protocol

from face2ai_app.domain.models import DetectedFace


class RecognitionEngine(Protocol):
    """What the application may assume about any recognition adapter.

    The four obligations below are documentation; the enforcement is
    ``apps/face2ai/tests/test_port_conformance.py``, which runs them against *every* adapter
    present in the environment, real ones included. Prose enforces nothing — that is the whole
    reason this contract is executable.

    1. **Re-entrancy.** ``detect`` may be called from several threads of one process at once, and
       concurrent calls must agree with the serial result. The API runs it in a threadpool, so two
       browser tabs already exercise this. An adapter over process-global native state (dlib) owes
       the caller a lock; the caller must not need to know which kind it got.
    2. **Input bounds before decoding.** A frame over the configured pixel budget
       (``Settings.max_frame_pixels``) is rejected with ``InvalidFrame`` *before* it is decoded —
       a small compressed file can declare an enormous image, so checking afterwards is no check.
    3. **EXIF-upright coordinates.** Boxes and encodings are produced in the EXIF-upright space, the
       space the browser draws its overlay in. A frame stored rotated with an orientation tag is
       analysed the way a human sees it.
    4. **A closed error set.** Only ``InvalidFrame`` and ``RecognitionUnavailable`` escape ``detect``.
       Whatever the adapter's library raises is translated; nothing else reaches the API layer,
       which maps exactly these two.

    ``face2ai_app.adapters.frame_decode.decode_frame`` is the one owner of obligations 2 and 3 for
    adapters that decode pixels themselves.
    """

    @property
    def available(self) -> bool:
        """False (never an exception) when the engine cannot run — a missing extra, a broken native
        library. ``detect`` then raises ``RecognitionUnavailable``."""

    @property
    def availability_reason(self) -> str | None:
        """Why the engine is unavailable, for ``/readyz`` and ``/api/status``; None while available."""

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        """Faces in one frame, honouring the four obligations above."""
