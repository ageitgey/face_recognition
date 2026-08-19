"""The dashboard API half (`/api/plugins/face2ai/*`) — needs fastapi + httpx, which the plain plugin test
environment (and CI) does not have: skipped there, runs locally with
`uv run --no-project --with pytest==9.0.2 --with fastapi --with httpx pytest apps/face2ai-hermes-plugin/tests`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "face2ai" / "dashboard"))

import plugin_api  # noqa: E402


def _app():
    app = fastapi.FastAPI()
    app.include_router(plugin_api.router)
    return app


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


def _fake_client(calls, payload=None, status=200, exc=None):
    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            calls.append(("init", kw))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            calls.append(("get", url, dict(params or {})))
            if exc:
                raise exc
            return _Response(payload, status)

    return FakeAsyncClient


def test_timeline_proxies_face2ai_with_clamped_seconds_and_identity(monkeypatch):
    calls = []
    upstream = {"seconds": 3600, "samples": [{"at": "2026-08-18T12:00:00Z", "identity_id": "a", "display_name": "Ben", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}], "moods": [], "actions": [{"action": "smile", "duration_ms": 900}]}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(calls, upstream))
    monkeypatch.setattr(plugin_api, "_events_url", lambda: "http://face2ai.test:8765")
    client = TestClient(_app())
    body = client.get("/timeline", params={"seconds": 99999, "identity_id": "a"}).json()
    assert body == upstream and "error" not in body
    assert calls[-1] == ("get", "http://face2ai.test:8765/api/expression/timeline", {"seconds": 3600, "identity_id": "a"})
    assert calls[-2][1]["timeout"] == 3.0
    client.get("/timeline", params={"seconds": 5})
    assert calls[-1][2] == {"seconds": 10}  # clamped up, no identity_id key when not given
    client.get("/timeline")
    assert calls[-1][2] == {"seconds": 600}


def test_timeline_error_keeps_the_shape(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(calls, exc=httpx.ConnectError("refused")))
    monkeypatch.setattr(plugin_api, "_events_url", lambda: "http://face2ai.test:8765")
    client = TestClient(_app())
    body = client.get("/timeline", params={"seconds": 120}).json()
    assert body == {"error": "refused", "seconds": 120, "samples": [], "moods": [], "actions": []}


def test_whitespace_identity_is_not_a_filter(monkeypatch):
    """`?identity_id=%20` is nobody: forwarding " " makes Face2AI (`identity_id or None`) filter everything away."""
    calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(calls, {"seconds": 600, "samples": [], "moods": [], "actions": []}))
    monkeypatch.setattr(plugin_api, "_events_url", lambda: "http://face2ai.test:8765")
    client = TestClient(_app())
    client.get("/timeline", params={"identity_id": "   "})
    assert calls[-1][2] == {"seconds": 600}
    client.get("/timeline", params={"identity_id": " a "})
    assert calls[-1][2] == {"seconds": 600, "identity_id": "a"}


def test_timeline_non_2xx_and_unexpected_payload_keep_the_shape(monkeypatch):
    """Both failure branches the pane must survive: an HTTP error status and a body that is not a dict."""
    calls = []
    monkeypatch.setattr(plugin_api, "_events_url", lambda: "http://face2ai.test:8765")
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(calls, {"detail": "nope"}, status=503))
    client = TestClient(_app())
    body = client.get("/timeline", params={"seconds": 60}).json()
    assert (body["seconds"], body["samples"], body["moods"], body["actions"]) == (60, [], [], [])
    assert body["error"], "the pane is told why it has no data"

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(calls, ["not", "a", "dict"]))
    body = client.get("/timeline", params={"seconds": 60}).json()
    assert body == {"error": "unexpected timeline payload", "seconds": 60, "samples": [], "moods": [], "actions": []}


def test_live_presence_says_connected_like_both_fallback_branches(monkeypatch):
    """The desktop chip reads `latest.connected` and `refresh()` replaces `latest` wholesale
    (`plugin.js`: `latest = await pluginCtx.rest('/presence')`), so a live answer without the key
    is falsy — "Face2AI nicht verbunden" after every *successful* poll, until the next SSE frame.
    All three branches must answer the same question."""
    calls = []
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(calls, {"state": "KNOWN", "display_name": "Ada", "faces": 1}))
    monkeypatch.setattr(plugin_api, "_events_url", lambda: "http://face2ai.test:8765")
    monkeypatch.setattr(plugin_api, "_snapshot_from_state", lambda: None)
    body = TestClient(_app()).get("/presence").json()
    assert body["source"] == "live"
    assert body["presence"]["display_name"] == "Ada"
    assert body["connected"] is True

    # ... and the failure branch still says the opposite, so the key discriminates.
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client(calls, exc=httpx.ConnectError("refused")))
    down = TestClient(_app()).get("/presence").json()
    assert (down["source"], down["connected"]) == ("none", False)
