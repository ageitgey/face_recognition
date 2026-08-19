from __future__ import annotations

import logging
import math

from face2ai_app.domain.errors import EnrollmentRejected, RecognitionUnavailable
from face2ai_app.domain.models import (
    DetectedFace,
    FaceObservation,
    IdentityRecord,
    IdentitySummary,
    RecognitionEvent,
    RecognitionState,
)
from face2ai_app.ports.expression import ExpressionEngine
from face2ai_app.ports.identity_store import IdentityStore
from face2ai_app.ports.recognition import RecognitionEngine

logger = logging.getLogger(__name__)


class IdentityService:
    def __init__(
        self,
        engine: RecognitionEngine,
        store: IdentityStore,
        tolerance: float,
        expression: ExpressionEngine | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.tolerance = tolerance
        # Expression is an opt-in *hint* attached to observations; it never takes part in matching.
        self.expression = expression
        self.expression_enabled = False  # runtime toggle (POST /api/expression); main.py seeds it from settings
        self._expression_warned = False

    @property
    def expression_available(self) -> bool:
        """An expression engine is configured and loaded (the opt-in can be switched on)."""
        return self.expression is not None and bool(self.expression.available)

    @property
    def expression_reason(self) -> str | None:
        """Why expressions are unavailable; ``None`` while the engine is available."""
        if self.expression is None:
            return "not configured"
        if self.expression.available:
            return None
        return self.expression.availability_reason or "expression engine unavailable"

    def _nearest(
        self, candidate: list[float], identities: list[IdentityRecord]
    ) -> tuple[IdentityRecord | None, float | None]:
        nearest_identity: IdentityRecord | None = None
        nearest_distance: float | None = None
        for identity in identities:
            for known in identity.encodings:
                distance = math.dist(known, candidate)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_identity = identity
                    nearest_distance = distance
        if nearest_distance is None or nearest_distance > self.tolerance:
            return None, nearest_distance
        return nearest_identity, nearest_distance

    def recognize(self, image_bytes: bytes) -> RecognitionEvent:
        if not self.engine.available:
            raise RecognitionUnavailable(
                self.engine.availability_reason or "recognition engine unavailable"
            )
        detected = self.engine.detect(image_bytes)
        if not detected:
            return RecognitionEvent(state=RecognitionState.NO_FACE, can_enroll=False)

        identities = self.store.list()
        observations: list[FaceObservation] = []
        for face in detected:
            identity, distance = self._nearest(face.encoding, identities)
            observations.append(
                FaceObservation(
                    box=face.box,
                    matched=identity is not None,
                    identity_id=identity.id if identity else None,
                    display_name=identity.display_name if identity else None,
                    match_distance=distance,
                )
            )
        self._attach_expressions(image_bytes, detected, observations)

        if len(observations) > 1:
            return RecognitionEvent(
                state=RecognitionState.MULTIPLE_FACES,
                faces=observations,
                can_enroll=False,
                message="Enrollment requires exactly one visible face.",
            )
        observation = observations[0]
        return RecognitionEvent(
            state=RecognitionState.KNOWN if observation.matched else RecognitionState.UNKNOWN,
            faces=observations,
            can_enroll=not observation.matched,
        )

    def _attach_expressions(
        self, image_bytes: bytes, detected: list[DetectedFace], observations: list[FaceObservation]
    ) -> None:
        """Best effort: a failing expression engine leaves ``expression`` None and never breaks recognition."""
        engine = self.expression
        if engine is None or not (self.expression_enabled and engine.available):
            return
        try:
            expressions = engine.analyze(image_bytes, [face.box for face in detected])
        except Exception as exc:
            logger.log(
                logging.DEBUG if self._expression_warned else logging.WARNING,
                "expression analysis failed, recognition continues without it: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._expression_warned = True
            return
        # non-strict on purpose: a short/long list from a misbehaving engine must never raise here
        for observation, expression in zip(observations, expressions):
            observation.expression = expression

    def enroll(self, image_bytes: bytes, display_name: str, consent: bool) -> IdentitySummary:
        clean_name = display_name.strip()
        if not consent:
            raise EnrollmentRejected(
                "explicit local biometric enrollment consent is required"
            )
        if not clean_name or len(clean_name) > 80:
            raise EnrollmentRejected("display name must contain 1-80 characters")
        if not self.engine.available:
            raise RecognitionUnavailable(
                self.engine.availability_reason or "recognition engine unavailable"
            )
        detected = self.engine.detect(image_bytes)
        if len(detected) != 1:
            raise EnrollmentRejected("enrollment requires exactly one visible face")
        record = self.store.add(IdentityRecord.new(clean_name, detected[0].encoding))
        return IdentitySummary(
            id=record.id,
            display_name=record.display_name,
            created_at=record.created_at,
            encoding_count=len(record.encodings),
        )

    def summaries(self) -> list[IdentitySummary]:
        return [
            IdentitySummary(
                id=record.id,
                display_name=record.display_name,
                created_at=record.created_at,
                encoding_count=len(record.encodings),
            )
            for record in self.store.list()
        ]
