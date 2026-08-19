from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from face2ai_app.adapters.json_identity_store import JsonIdentityStore
from face2ai_app.config import Settings
from face2ai_app.domain.models import DetectedFace, Expression, FaceBox
from face2ai_app.main import create_app


class FakeEngine:
    available = True
    availability_reason = None

    def __init__(self) -> None:
        self.faces: list[DetectedFace] = []

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        return self.faces


class FakeExpressionEngine:
    """Available expression engine returning scripted expressions (padded/truncated to the box count).

    ``script`` holds per-call results consumed first-in-first-out (frame 1, frame 2, ...); once it is
    exhausted (or when empty) every call answers with ``expressions``.
    """

    available = True
    availability_reason = None

    def __init__(self) -> None:
        self.expressions: list[Expression | None] = []
        self.script: list[list[Expression | None]] = []
        self.raise_error = False
        self.calls = 0

    def analyze(self, image_bytes: bytes, boxes: list[FaceBox]) -> list[Expression | None]:
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("fake expression failure")
        scripted = list((self.script.pop(0) if self.script else self.expressions)[: len(boxes)])
        return scripted + [None] * (len(boxes) - len(scripted))


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def fake_expression() -> FakeExpressionEngine:
    return FakeExpressionEngine()


@pytest.fixture
def client(tmp_path: Path, fake_engine: FakeEngine, fake_expression: FakeExpressionEngine) -> TestClient:
    settings = Settings(data_dir=tmp_path, greeting_cooldown_seconds=7)
    store = JsonIdentityStore(settings.identity_store_path)
    return TestClient(create_app(settings=settings, engine=fake_engine, store=store, expression=fake_expression))


@pytest.fixture
def face() -> DetectedFace:
    return DetectedFace(
        box=FaceBox(top=10, right=110, bottom=120, left=20),
        encoding=[0.1] * 128,
    )
