from __future__ import annotations


def image_headers():
    return {"content-type": "image/jpeg"}


def test_health_and_status_expose_runtime_configuration(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine_available"] is True
    assert payload["greeting_cooldown_seconds"] == 7


def test_no_face_cannot_enroll(client, fake_engine):
    fake_engine.faces = []
    response = client.post("/api/recognize", content=b"frame", headers=image_headers())
    assert response.status_code == 200
    assert response.json()["state"] == "NO_FACE"
    assert response.json()["can_enroll"] is False


def test_unknown_then_enroll_then_known(client, fake_engine, face):
    fake_engine.faces = [face]
    unknown = client.post("/api/recognize", content=b"frame", headers=image_headers())
    assert unknown.status_code == 200
    assert unknown.json()["state"] == "UNKNOWN"
    assert unknown.json()["can_enroll"] is True

    enrolled = client.post(
        "/api/enroll?display_name=PersonA&consent=true",
        content=b"frame",
        headers=image_headers(),
    )
    assert enrolled.status_code == 201
    assert enrolled.json()["display_name"] == "PersonA"

    known = client.post("/api/recognize", content=b"frame", headers=image_headers())
    assert known.status_code == 200
    assert known.json()["state"] == "KNOWN"
    assert known.json()["can_enroll"] is False
    assert known.json()["faces"][0]["display_name"] == "PersonA"
    assert known.json()["faces"][0]["match_distance"] == 0.0


def test_multiple_faces_block_recognition_enrollment_and_api_enrollment(
    client, fake_engine, face
):
    fake_engine.faces = [face, face]
    recognized = client.post("/api/recognize", content=b"frame", headers=image_headers())
    assert recognized.status_code == 200
    assert recognized.json()["state"] == "MULTIPLE_FACES"
    assert recognized.json()["can_enroll"] is False

    rejected = client.post(
        "/api/enroll?display_name=PersonB&consent=true",
        content=b"frame",
        headers=image_headers(),
    )
    assert rejected.status_code == 422


def test_recognition_reads_identity_store_once_per_operation(
    client, fake_engine, face, monkeypatch
):
    fake_engine.faces = [face, face]
    store = client.app.state.identity_service.store
    original_list = store.list
    calls = 0

    def counted_list():
        nonlocal calls
        calls += 1
        return original_list()

    monkeypatch.setattr(store, "list", counted_list)
    response = client.post("/api/recognize", content=b"frame", headers=image_headers())

    assert response.status_code == 200
    assert response.json()["state"] == "MULTIPLE_FACES"
    assert calls == 1


def test_enrollment_requires_explicit_consent(client, fake_engine, face):
    fake_engine.faces = [face]
    response = client.post(
        "/api/enroll?display_name=PersonB&consent=false",
        content=b"frame",
        headers=image_headers(),
    )
    assert response.status_code == 422


def test_corrupted_identity_store_is_reported_without_killing_app(client):
    path = client.app.state.settings.identity_store_path
    path.write_text("{broken", encoding="utf-8")

    response = client.get("/api/status")
    assert response.status_code == 503
    assert "identity store contains invalid JSON" in response.json()["detail"]
    assert client.get("/healthz").status_code == 200


def test_image_content_type_is_enforced(client):
    response = client.post(
        "/api/recognize",
        content=b"frame",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 415


def test_wrong_length_encoding_in_the_store_is_a_503_not_a_crash(client, fake_engine, face):
    """The documented contract is "corrupt store -> 503"; a 3-float encoding used to reach math.dist."""
    fake_engine.faces = [face]
    client.app.state.settings.identity_store_path.write_text(
        '[{"id": "1", "display_name": "Mallory", "encodings": [[0.1, 0.2, 0.3]],'
        ' "created_at": "2026-08-19T00:00:00Z"}]',
        encoding="utf-8",
    )

    response = client.post("/api/recognize", content=b"frame", headers=image_headers())

    assert response.status_code == 503
    assert "invalid identity records" in response.json()["detail"]
