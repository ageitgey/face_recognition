"""Expression is an opt-in hint attached to RecognitionEvent faces — never a fact, never breaking recognition."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from face2ai_app.adapters.json_identity_store import JsonIdentityStore
from face2ai_app.adapters.null_expression import NullExpressionEngine
from face2ai_app.config import Settings
from face2ai_app.domain.models import Expression
from face2ai_app.main import create_app

HEADERS = {"content-type": "image/jpeg"}
HAPPY = Expression(dominant="Happiness", scores={"Happiness": 0.9}, valence=0.6, arousal=0.1)


def _recognize(client: TestClient) -> dict:
    response = client.post("/api/recognize", content=b"frame", headers=HEADERS)
    assert response.status_code == 200
    return response.json()


def test_recognize_attaches_expression_only_when_enabled(client, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.expressions = [HAPPY]
    off = _recognize(client)
    assert off["faces"][0]["expression"] is None  # opt-in default off
    assert fake_expression.calls == 0  # off means the engine is not even consulted
    assert client.post("/api/expression", json={"enabled": True}).json() == {"enabled": True, "available": True}
    on = _recognize(client)
    assert on["faces"][0]["expression"]["dominant"] == "Happiness"
    assert on["faces"][0]["expression"]["valence"] == 0.6
    assert fake_expression.calls == 1
    status = client.get("/api/status").json()
    assert status["expression_available"] is True and status["expression_enabled"] is True
    assert status["expression_reason"] is None


def test_expression_toggle_off_again_drops_expressions(client, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.expressions = [HAPPY]
    client.post("/api/expression", json={"enabled": True})
    assert _recognize(client)["faces"][0]["expression"] is not None
    assert client.post("/api/expression", json={"enabled": False}).json() == {"enabled": False, "available": True}
    assert _recognize(client)["faces"][0]["expression"] is None
    assert client.get("/api/status").json()["expression_enabled"] is False


def test_expression_toggle_refused_when_engine_unavailable(client):
    client.app.state.identity_service.expression = NullExpressionEngine("not installed")
    r = client.post("/api/expression", json={"enabled": True})
    assert r.status_code == 409 and "not installed" in r.json()["detail"]
    assert client.get("/api/status").json()["expression_enabled"] is False
    # turning it off is always allowed, and reports availability truthfully
    assert client.post("/api/expression", json={"enabled": False}).json() == {"enabled": False, "available": False}


def test_expression_toggle_validates_body(client):
    assert client.post("/api/expression", json={}).status_code == 422
    assert client.post("/api/expression", json={"enabled": "maybe"}).status_code == 422


def test_expression_failure_never_breaks_recognition(client, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.raise_error = True
    client.post("/api/expression", json={"enabled": True})
    payload = _recognize(client)
    assert payload["state"] == "UNKNOWN" and payload["can_enroll"] is True
    assert payload["faces"][0]["expression"] is None
    assert fake_expression.calls == 1


def test_expression_per_face_for_multiple_faces(client, fake_engine, fake_expression, face):
    fake_engine.faces = [face, face]
    fake_expression.expressions = [HAPPY]  # engine only knows the first face's mood
    client.post("/api/expression", json={"enabled": True})
    payload = _recognize(client)
    assert payload["state"] == "MULTIPLE_FACES"
    assert [f["expression"] and f["expression"]["dominant"] for f in payload["faces"]] == ["Happiness", None]


def test_status_reports_unavailable_expression_engine(tmp_path: Path, fake_engine):
    settings = Settings(data_dir=tmp_path, expression_enabled=True)  # env opt-in cannot enable an absent engine
    store = JsonIdentityStore(settings.identity_store_path)
    client = TestClient(
        create_app(settings=settings, engine=fake_engine, store=store, expression=NullExpressionEngine("not installed"))
    )
    status = client.get("/api/status").json()
    assert status["expression_available"] is False
    assert status["expression_reason"] == "not installed"
    assert status["expression_enabled"] is False


def test_startup_log_for_unavailable_expression_engine(tmp_path: Path, fake_engine, caplog):
    """Opt-in off: an absent engine is INFO. Opt-in on: exactly one WARNING, nothing else about it."""
    import logging

    store = JsonIdentityStore(Settings(data_dir=tmp_path).identity_store_path)
    with caplog.at_level(logging.INFO, logger="face2ai_app.main"):
        create_app(settings=Settings(data_dir=tmp_path), engine=fake_engine, store=store, expression=NullExpressionEngine("not installed"))
    expression_lines = [r for r in caplog.records if "expression" in r.getMessage()]
    assert [r.levelno for r in expression_lines] == [logging.INFO]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="face2ai_app.main"):
        create_app(
            settings=Settings(data_dir=tmp_path, expression_enabled=True),
            engine=fake_engine, store=store, expression=NullExpressionEngine("not installed"),
        )
    expression_lines = [r for r in caplog.records if "expression" in r.getMessage()]
    assert [r.levelno for r in expression_lines] == [logging.WARNING]
    assert "not installed" in expression_lines[0].getMessage()


def test_env_opt_in_enables_available_engine(tmp_path: Path, fake_engine, fake_expression, face):
    settings = Settings(data_dir=tmp_path, expression_enabled=True)
    store = JsonIdentityStore(settings.identity_store_path)
    client = TestClient(create_app(settings=settings, engine=fake_engine, store=store, expression=fake_expression))
    assert client.get("/api/status").json()["expression_enabled"] is True
    fake_engine.faces = [face]
    fake_expression.expressions = [HAPPY]
    assert _recognize(client)["faces"][0]["expression"]["dominant"] == "Happiness"


def test_timeline_query_validation(client):
    assert client.get("/api/expression/timeline?seconds=1").status_code == 422
    assert client.get("/api/expression/timeline?seconds=3601").status_code == 422
    assert client.get("/api/expression/timeline?identity_id=" + "x" * 81).status_code == 422
    assert client.get("/api/expression/timeline").json() == {"seconds": 600, "samples": [], "moods": [], "actions": []}
    assert client.get("/api/expression/timeline?seconds=10&identity_id=abc").json()["seconds"] == 10


def test_empty_identity_filter_is_not_a_filter(client, fake_engine, fake_expression, face):
    """``?identity_id=`` means "everyone", not ``identity_id == ""`` — which would match nobody."""
    fake_engine.faces = [face]
    fake_expression.expressions = [HAPPY]
    client.post("/api/expression", json={"enabled": True})
    _recognize(client)
    unfiltered = client.get("/api/expression/timeline").json()
    assert len(unfiltered["samples"]) == 1  # an UNKNOWN person: the sample's identity_id is None
    assert client.get("/api/expression/timeline?identity_id=").json() == unfiltered
    assert client.get("/api/expression/timeline?identity_id=nobody").json()["samples"] == []  # a real filter still filters


def test_expression_dynamics_failure_never_breaks_recognition(client, fake_engine, fake_expression, face, caplog):
    """Actions/history are hints: a failure there is warned once (then debug) and recognize still answers."""
    import logging

    class BrokenActions:
        def observe(self, *args, **kwargs):
            raise RuntimeError("clock went backwards")

        def reset(self) -> None:
            pass

    fake_engine.faces = [face]
    fake_expression.expressions = [HAPPY]
    client.post("/api/expression", json={"enabled": True})
    client.app.state.actions = BrokenActions()
    with caplog.at_level(logging.DEBUG, logger="face2ai_app.api.routes"):
        for _ in range(2):
            payload = _recognize(client)
            assert payload["state"] == "UNKNOWN" and payload["faces"][0]["expression"]["dominant"] == "Happiness"
    lines = [r for r in caplog.records if "expression dynamics failed" in r.getMessage()]
    assert [r.levelno for r in lines] == [logging.WARNING, logging.DEBUG]
    assert "clock went backwards" in lines[0].getMessage()
    # the presence still carries the live affect and the sample was recorded before the failure
    assert client.get("/api/presence").json()["valence"] == pytest.approx(0.45)
    assert len(client.get("/api/expression/timeline").json()["samples"]) == 2
