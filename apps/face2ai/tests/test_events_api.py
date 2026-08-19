"""Presence + Server-Sent Events API.

The SSE stream is infinite, and Starlette's TestClient buffers a response until the app
returns, so these tests run the FastAPI app under a real uvicorn server on a loopback port
and read the stream with httpx — the same path the voice agent uses.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import uvicorn

from face2ai_app.adapters.json_identity_store import JsonIdentityStore
from face2ai_app.config import Settings
from face2ai_app.domain.models import Expression
from face2ai_app.main import create_app

HEADERS = {"content-type": "image/jpeg"}
HAPPY = Expression(dominant="Happiness", scores={"Happiness": 0.9}, valence=0.6, arousal=0.1)
# MoodTracker EMA (alpha 0.5) starts at 0: two HAPPY frames -> score 0.675 (>= min_score 0.5, commits on
# the second frame), valence 0.3 -> 0.45, arousal 0.05 -> 0.075. The *mood event* freezes both at commit
# time; the presence carries the LIVE smoothed values (Stage 2), which equal these only on the very frame
# the mood commits — a later frame moves the presence while the mood event keeps its frozen pair.
EMA_VALENCE_2 = 0.45
EMA_AROUSAL_2 = 0.075


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class LiveServer:
    base_url: str
    client: httpx.Client

    def sse(
        self,
        path: str,
        wanted: int,
        headers: dict | None = None,
        timeout: float = 5.0,
        *,
        skip_heartbeats: bool = False,
    ) -> list[dict]:
        """Read ``wanted`` SSE frames as [{'id':..., 'event':..., 'data':{...}}]."""
        frames: list[dict] = []
        current: dict = {}
        with self.client.stream("GET", path, headers=headers or {}, timeout=timeout) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            for line in response.iter_lines():
                if line == "":
                    if current and not (skip_heartbeats and current.get("event") == "heartbeat"):
                        frames.append(current)
                    current = {}
                    if len(frames) >= wanted:
                        break
                    continue
                key, _, value = line.partition(": ")
                current[key] = json.loads(value) if key == "data" else value
        return frames[:wanted]


@contextmanager
def _live_server(settings: Settings, fake_engine, fake_expression) -> Iterator[LiveServer]:
    """Run the app under a real uvicorn on a free loopback port for the duration of the block.

    A test whose ``Settings`` the shared ``live`` fixture cannot express (a short stale window, say)
    builds its own server with this instead of weakening its assertions.
    """
    app = create_app(
        settings=settings, engine=fake_engine, store=JsonIdentityStore(settings.identity_store_path), expression=fake_expression
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, name="test-uvicorn", daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not server.started:
        assert time.time() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            yield LiveServer(base_url=base_url, client=client)
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def live(tmp_path: Path, fake_engine, fake_expression) -> Iterator[LiveServer]:
    settings = Settings(
        data_dir=tmp_path,
        presence_stable_ticks=1,
        mood_stable_ticks=1,
        action_min_frames=1,  # one frame may complete an action: a dropped one is then provably dropped
        events_heartbeat_seconds=0.05,
        greeting_cooldown_seconds=7,
    )
    with _live_server(settings, fake_engine, fake_expression) as server:
        yield server


def test_presence_endpoint_follows_recognition_and_reset(live, fake_engine, face):
    assert live.client.get("/api/presence").json()["state"] == "NO_SIGNAL"
    fake_engine.faces = [face]
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    presence = live.client.get("/api/presence").json()
    assert presence["state"] == "UNKNOWN"
    assert presence["faces"] == 1
    # Freshness is not a server-side flag: `observed_at` is the fact, `expire()` is the only rule.
    assert "stale" not in presence
    assert presence["observed_at"] is not None
    assert live.client.post("/api/presence/reset").json()["state"] == "NO_SIGNAL"


def test_events_stream_starts_with_hello_and_replays_buffered_events(live, fake_engine, face):
    fake_engine.faces = [face]
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)  # NO_SIGNAL -> UNKNOWN (seq 1)
    enrolled = live.client.post("/api/enroll?display_name=Ada&consent=true", content=b"frame", headers=HEADERS)
    assert enrolled.status_code == 201  # seq 2

    frames = live.sse("/api/events?after=0", wanted=3)
    assert frames[0]["event"] == "hello"
    assert frames[0]["data"]["presence"]["state"] == "UNKNOWN"
    assert frames[0]["data"]["greeting_cooldown_seconds"] == 7
    assert frames[0]["data"]["last_sequence"] == 2
    assert frames[0]["data"]["engine_available"] is True

    assert frames[1]["event"] == "presence" and frames[1]["id"] == "1"
    assert frames[1]["data"]["from_state"] == "NO_SIGNAL"
    assert frames[1]["data"]["to_state"] == "UNKNOWN"

    assert frames[2]["event"] == "store" and frames[2]["id"] == "2"
    assert frames[2]["data"]["kind"] == "enrolled"
    assert frames[2]["data"]["display_name"] == "Ada"
    assert frames[2]["data"]["identity_count"] == 1


def test_live_events_arrive_while_subscribed(live, fake_engine, face):
    fake_engine.faces = [face]
    received: list[dict] = []

    def subscriber() -> None:
        received.extend(live.sse("/api/events", wanted=2, skip_heartbeats=True))

    thread = threading.Thread(target=subscriber, daemon=True)
    thread.start()
    deadline = time.time() + 3
    while live.client.get("/api/status").json()["event_subscribers"] < 1:  # deterministic handshake
        assert time.time() < deadline, "subscriber never registered"
        time.sleep(0.02)
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert [f["event"] for f in received] == ["hello", "presence"]
    assert received[1]["data"]["to_state"] == "UNKNOWN"


def test_heartbeat_carries_presence_snapshot(live, fake_engine, face):
    fake_engine.faces = [face]
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    frames = live.sse("/api/events", wanted=2)
    assert frames[0]["event"] == "hello"
    assert frames[1]["event"] == "heartbeat"
    assert frames[1]["data"]["presence"]["state"] == "UNKNOWN"


def test_last_event_id_header_resumes(live, fake_engine, face):
    fake_engine.faces = [face]
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)  # seq 1
    live.client.post("/api/presence/reset")  # seq 2
    frames = live.sse("/api/events", wanted=2, headers={"Last-Event-ID": "1"})
    assert [f["event"] for f in frames] == ["hello", "presence"]
    assert frames[1]["id"] == "2" and frames[1]["data"]["to_state"] == "NO_SIGNAL"


def test_store_events_for_delete_and_erase(live, fake_engine, face):
    fake_engine.faces = [face]
    record = live.client.post("/api/enroll?display_name=Ada&consent=true", content=b"frame", headers=HEADERS).json()
    assert live.client.delete(f"/api/identities/{record['id']}").status_code == 204
    assert live.client.delete("/api/identities").json() == {"deleted": 0}
    frames = live.sse("/api/events?after=0", wanted=4)
    kinds = [(f["event"], f["data"].get("kind")) for f in frames[1:]]
    assert kinds == [("store", "enrolled"), ("store", "deleted"), ("store", "erased")]
    assert frames[2]["data"]["display_name"] == "Ada"
    assert frames[3]["data"]["identity_count"] == 0


def test_status_reports_agent_subscription_while_stream_is_open(live):
    assert live.client.get("/api/status").json()["agent_connected"] is False
    with live.client.stream("GET", "/api/events?role=agent") as response:
        lines = response.iter_lines()  # keep the iterator alive: dropping it closes the stream
        assert next(lines) == "event: hello"  # subscription is registered once hello arrives
        status = live.client.get("/api/status").json()
        assert status["agent_connected"] is True
        assert status["event_subscribers"] == 1
    deadline = time.time() + 3
    while live.client.get("/api/status").json()["agent_connected"] and time.time() < deadline:
        time.sleep(0.05)
    assert live.client.get("/api/status").json()["agent_connected"] is False


BROWSER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128.0 Safari/537.36"


@pytest.mark.parametrize("site", ["cross-site", "same-site", "none"])
def test_a_page_from_elsewhere_cannot_subscribe_as_the_agent(live, site):
    """Greeting ownership is handed over by a query parameter, so without this check any page that
    reaches the port can claim it: the browser shell then goes deliberately silent and a background
    tab disables the product's headline behaviour. This port is reverse-tunnelled to a VPS
    (project CLAUDE.md), so "loopback, therefore safe" is the wrong frame. A browser always sends
    Sec-Fetch-Site; only ``same-origin`` may take the agent role."""
    assert live.client.get("/api/status").json()["agent_connected"] is False
    headers = {
        "Origin": "https://evil.example",
        "Referer": "https://evil.example/",
        "Sec-Fetch-Site": site,
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "User-Agent": BROWSER_AGENT,
        "Accept": "text/event-stream",
    }
    with live.client.stream("GET", "/api/events?role=agent", headers=headers) as response:
        assert response.status_code == 403
        response.read()
        assert "same-origin" in response.json()["detail"]
        assert site in response.json()["detail"]
    status = live.client.get("/api/status").json()
    assert status["agent_connected"] is False
    assert status["event_subscribers"] == 0


def test_the_real_agent_sends_no_sec_fetch_site_and_is_unaffected(live):
    """httpx (voice agent, Hermes plugin) never sends the header — the check must not touch them."""
    with live.client.stream("GET", "/api/events?role=agent", headers={"Accept": "text/event-stream"}) as response:
        assert response.status_code == 200
        assert "sec-fetch-site" not in {k.lower() for k in response.request.headers}
        lines = response.iter_lines()  # keep the iterator alive: dropping it closes the stream
        assert next(lines) == "event: hello"
        status = live.client.get("/api/status").json()
        assert status["agent_connected"] is True
        assert status["event_subscribers"] == 1


@pytest.mark.parametrize("role", ["browser", "agent"])
def test_a_same_origin_browser_subscription_is_accepted(live, role):
    """The shipped page subscribes with role=browser; a same-origin agent role stays allowed too."""
    headers = {
        "Origin": live.base_url,
        "Referer": f"{live.base_url}/",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "User-Agent": BROWSER_AGENT,
        "Accept": "text/event-stream",
    }
    with live.client.stream("GET", f"/api/events?role={role}", headers=headers) as response:
        assert response.status_code == 200
        lines = response.iter_lines()
        assert next(lines) == "event: hello"
        status = live.client.get("/api/status").json()
        assert status["event_subscribers"] == 1
        assert status["agent_connected"] is (role == "agent")


def test_events_stream_carries_only_the_documented_keys(live, fake_engine, face):
    """ADR-002 §2: states, names, counts, timestamps — never frames, boxes, encodings or distances."""
    fake_engine.faces = [face]
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    live.client.post("/api/enroll?display_name=Ada&consent=true", content=b"frame", headers=HEADERS)
    frames = live.sse("/api/events?after=0", wanted=3)
    hello, presence, store = frames
    presence_keys = {
        "state", "identity_id", "display_name", "faces", "since", "observed_at", "mood", "valence", "arousal",
    }
    assert set(hello["data"]["presence"]) == presence_keys
    assert set(presence["data"]) == {"sequence", "at", "from_state", "to_state", "identity_id", "display_name", "faces"}
    assert set(store["data"]) == {"sequence", "at", "kind", "identity_id", "display_name", "identity_count"}
    text = json.dumps([f["data"] for f in frames])
    for forbidden in ("encoding", "box", "match_distance", "top", "left"):
        assert forbidden not in text, forbidden


def test_recognize_response_signals_agent_ownership_per_frame(live, fake_engine, face):
    fake_engine.faces = [face]
    assert live.client.post("/api/recognize", content=b"frame", headers=HEADERS).headers["x-face2ai-agent"] == "0"
    with live.client.stream("GET", "/api/events?role=agent") as response:
        lines = response.iter_lines()
        assert next(lines) == "event: hello"
        assert live.client.post("/api/recognize", content=b"frame", headers=HEADERS).headers["x-face2ai-agent"] == "1"


# ------------------------------------------------------------------------------- mood


def _enable_expression(live: LiveServer, fake_expression, expressions: list) -> None:
    fake_expression.expressions = expressions
    assert live.client.post("/api/expression", json={"enabled": True}).status_code == 200


def _recognize(live: LiveServer, times: int = 1) -> None:
    for _ in range(times):
        assert live.client.post("/api/recognize", content=b"frame", headers=HEADERS).status_code == 200


def test_mood_events_and_presence_mood(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    _enable_expression(live, fake_expression, [HAPPY])
    _recognize(live, times=2)  # presence commits on frame 1 (stable ticks 1), mood on frame 2 (EMA >= 0.5)
    presence = live.client.get("/api/presence").json()
    assert presence["state"] == "UNKNOWN" and presence["mood"] == "Happiness"
    assert presence["valence"] == pytest.approx(EMA_VALENCE_2) and presence["arousal"] == pytest.approx(EMA_AROUSAL_2)

    frames = live.sse("/api/events?after=0", wanted=3, skip_heartbeats=True)
    assert [f["event"] for f in frames] == ["hello", "presence", "mood"]
    assert frames[0]["data"]["presence"]["mood"] == "Happiness"  # hello snapshot carries the mood fields
    assert "mood" not in frames[1]["data"]  # a presence transition starts a fresh, mood-less presence
    mood = frames[2]["data"]
    assert frames[2]["id"] == "2" and mood["sequence"] == 2
    assert (mood["from_mood"], mood["to_mood"]) == (None, "Happiness")
    assert mood["valence"] == pytest.approx(EMA_VALENCE_2) and mood["arousal"] == pytest.approx(EMA_AROUSAL_2)
    assert (mood["identity_id"], mood["display_name"]) == (None, None)  # UNKNOWN person
    assert set(mood) == {"sequence", "at", "identity_id", "display_name", "from_mood", "to_mood", "valence", "arousal"}
    text = json.dumps(mood)  # ADR-002 §2 sweep over a mood frame: label + two rounded scalars, never per-frame data
    for forbidden in ("encoding", "box", "match_distance", "top", "left", "scores", "blendshapes", "dominant"):
        assert forbidden not in text, forbidden


def test_mood_carries_identity_and_ends_when_the_person_leaves(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    record = live.client.post("/api/enroll?display_name=Ada&consent=true", content=b"frame", headers=HEADERS).json()
    _enable_expression(live, fake_expression, [HAPPY])
    _recognize(live, times=2)  # seq: 1 store, 2 presence NO_SIGNAL->KNOWN, 3 mood ->Happiness
    presence = live.client.get("/api/presence").json()
    assert (presence["state"], presence["display_name"], presence["mood"]) == ("KNOWN", "Ada", "Happiness")
    fake_engine.faces = []
    _recognize(live)  # seq 4 presence KNOWN->NO_FACE, seq 5 mood Happiness->None
    presence = live.client.get("/api/presence").json()
    assert (presence["state"], presence["mood"], presence["valence"], presence["arousal"]) == ("NO_FACE", None, None, None)
    frames = live.sse("/api/events?after=1", wanted=5, skip_heartbeats=True)
    assert [f["event"] for f in frames] == ["hello", "presence", "mood", "presence", "mood"]
    started, ended = frames[2]["data"], frames[4]["data"]
    assert (started["identity_id"], started["display_name"], started["to_mood"]) == (record["id"], "Ada", "Happiness")
    assert (ended["identity_id"], ended["display_name"]) == (record["id"], "Ada")
    assert (ended["from_mood"], ended["to_mood"], ended["valence"], ended["arousal"]) == ("Happiness", None, None, None)


def test_multiple_faces_never_produce_a_mood(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face, face]
    _enable_expression(live, fake_expression, [HAPPY, HAPPY])
    _recognize(live, times=4)
    presence = live.client.get("/api/presence").json()
    assert presence["state"] == "MULTIPLE_FACES" and presence["faces"] == 2
    assert (presence["mood"], presence["valence"], presence["arousal"]) == (None, None, None)
    frames = live.sse("/api/events?after=0", wanted=3)  # heartbeats included: proves nothing else was buffered
    assert [f["event"] for f in frames] == ["hello", "presence", "heartbeat"]


def test_expression_toggle_off_ends_the_mood_immediately(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    _enable_expression(live, fake_expression, [HAPPY])
    _recognize(live, times=2)  # seq 1 presence, seq 2 mood ->Happiness
    assert live.client.get("/api/presence").json()["mood"] == "Happiness"
    assert live.client.post("/api/expression", json={"enabled": False}).json()["enabled"] is False  # seq 3 mood ->None
    presence = live.client.get("/api/presence").json()
    assert (presence["state"], presence["mood"], presence["valence"], presence["arousal"]) == ("UNKNOWN", None, None, None)
    frames = live.sse("/api/events?after=0", wanted=4)  # heartbeats included: nothing else was buffered
    assert [f["event"] for f in frames] == ["hello", "presence", "mood", "mood"]
    ended = frames[3]["data"]
    assert (ended["from_mood"], ended["to_mood"], ended["valence"], ended["arousal"]) == ("Happiness", None, None, None)
    live.client.post("/api/expression", json={"enabled": False})  # idempotent: no mood, no event
    _recognize(live)  # presence unchanged, expressions off: nothing published
    frames = live.sse("/api/events?after=3", wanted=2)
    assert [f["event"] for f in frames] == ["hello", "heartbeat"]


def test_presence_reset_clears_mood_and_forgets_the_smoothing(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    _enable_expression(live, fake_expression, [HAPPY])
    _recognize(live, times=2)  # seq 1 presence NO_SIGNAL->UNKNOWN, seq 2 mood ->Happiness
    assert live.client.get("/api/presence").json()["mood"] == "Happiness"
    reset = live.client.post("/api/presence/reset").json()  # seq 3 presence UNKNOWN->NO_SIGNAL, seq 4 mood ->None, seq 5 timeline_cleared
    assert (reset["state"], reset["mood"], reset["valence"], reset["arousal"]) == ("NO_SIGNAL", None, None, None)
    # The mood tracker was reset too: one frame after the reset cannot re-commit (EMA restarts at 0) ...
    _recognize(live)  # seq 6 presence NO_SIGNAL->UNKNOWN
    presence = live.client.get("/api/presence").json()
    assert (presence["state"], presence["mood"]) == ("UNKNOWN", None)
    # ... the second one can.
    _recognize(live)  # seq 7 mood ->Happiness
    assert live.client.get("/api/presence").json()["mood"] == "Happiness"
    # A reset with no mood set publishes the presence transition (plus the clear) only.
    live.client.post("/api/presence/reset")  # seq 8 presence UNKNOWN->NO_SIGNAL, seq 9 mood ->None, seq 10 timeline_cleared
    _recognize(live)  # seq 11 presence NO_SIGNAL->UNKNOWN (no mood yet, but one affect sample)
    live.client.post("/api/presence/reset")  # seq 12 presence UNKNOWN->NO_SIGNAL, no mood event, seq 13 timeline_cleared
    live.client.post("/api/presence/reset")  # already NO_SIGNAL and nothing left to forget: nothing published
    frames = live.sse("/api/events?after=0", wanted=14)  # heartbeats included: proves nothing else was buffered
    kinds = [(f["event"], f["data"].get("to_state", f["data"].get("to_mood"))) for f in frames[1:]]
    assert kinds == [
        ("presence", "UNKNOWN"), ("mood", "Happiness"),
        ("presence", "NO_SIGNAL"), ("mood", None), ("timeline_cleared", None),  # exactly one mood end per reset, right after the presence
        ("presence", "UNKNOWN"), ("mood", "Happiness"),
        ("presence", "NO_SIGNAL"), ("mood", None), ("timeline_cleared", None),
        ("presence", "UNKNOWN"),
        ("presence", "NO_SIGNAL"), ("timeline_cleared", None),  # no mood was set: no mood event
    ]
    ended = frames[4]["data"]
    assert (ended["from_mood"], ended["to_mood"], ended["valence"], ended["arousal"]) == ("Happiness", None, None, None)


def test_stale_presence_expires_to_no_signal_and_is_published(tmp_path: Path, fake_engine, fake_expression, face):
    settings = Settings(
        data_dir=tmp_path, presence_stable_ticks=1, mood_stable_ticks=1, presence_stale_seconds=0.2, events_heartbeat_seconds=0.05
    )
    with _live_server(settings, fake_engine, fake_expression) as live:
        client = live.client
        fake_engine.faces = [face]
        _enable_expression(live, fake_expression, [HAPPY])
        _recognize(live, times=2)  # UNKNOWN (seq 1), mood Happiness (seq 2)
        assert client.get("/api/presence").json()["mood"] == "Happiness"
        time.sleep(0.4)  # longer than stale_seconds: no frames
        frames = live.sse("/api/events?after=0", wanted=5, skip_heartbeats=True)
        assert [f["event"] for f in frames] == ["hello", "presence", "mood", "presence", "mood"]
        hello_presence = frames[0]["data"]["presence"]
        assert (hello_presence["state"], hello_presence["mood"]) == ("NO_SIGNAL", None)  # expired before hello
        assert frames[1]["data"]["to_state"] == "UNKNOWN"
        assert frames[3]["data"]["from_state"] == "UNKNOWN" and frames[3]["data"]["to_state"] == "NO_SIGNAL"
        # Expiry ends the mood on the wire too, right after the presence it belonged to.
        ended = frames[4]["data"]
        assert (ended["from_mood"], ended["to_mood"]) == ("Happiness", None) and ended["at"] == frames[3]["data"]["at"]
        _recognize(live)  # back: fresh arrival, and the mood tracker was reset with the expiry ...
        presence = client.get("/api/presence").json()
        assert (presence["state"], presence["mood"]) == ("UNKNOWN", None)
        _recognize(live)  # ... so the returning person builds a fresh mood instead of being stuck without one
        assert client.get("/api/presence").json()["mood"] == "Happiness"


# ------------------------------------------------------------- actions + timeline (Stage 2)


def smiling(v: float) -> Expression:
    """HAPPY plus a smile blendshape group at intensity ``v`` (0.0 = readable, all-zero, ends the smile)."""
    return Expression(
        dominant="Happiness", scores={"Happiness": 0.9}, valence=0.6, arousal=0.1,
        blendshapes={"mouthSmileLeft": v, "mouthSmileRight": v},
    )


def test_action_events_and_timeline(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.script = [[smiling(0.9)], [smiling(0.9)], [smiling(0.0)]]
    _enable_expression(live, fake_expression, [])
    _recognize(live, times=3)  # seq 1 presence, seq 2 mood ->Happiness (frame 2), seq 3 action smile (offset on frame 3)
    frames = live.sse("/api/events?after=0", wanted=4, skip_heartbeats=True)
    assert [f["event"] for f in frames] == ["hello", "presence", "mood", "action"]
    action = frames[3]["data"]
    assert action["action"] == "smile" and action["frames"] == 2 and action["peak"] == 0.9
    assert action["onset_at"] <= action["apex_at"] <= action["offset_at"] == action["at"]
    assert action["duration_ms"] >= 0
    assert set(action) == {
        "sequence", "at", "identity_id", "display_name", "action", "onset_at", "apex_at", "offset_at", "peak", "duration_ms", "frames",
    }
    text = json.dumps(action)  # ADR-002 §2 sweep over an action frame: label, timestamps, one peak — never per-frame data
    for forbidden in ("encoding", "box", "match_distance", "top", "left", "scores", "blendshapes", "dominant"):
        assert forbidden not in text, forbidden
    tl = live.client.get("/api/expression/timeline?seconds=60").json()
    assert len(tl["samples"]) == 3 and tl["samples"][-1]["mood"] == "Happiness"
    assert len(tl["moods"]) == 1 and tl["moods"][0]["to_mood"] == "Happiness"
    assert len(tl["actions"]) == 1 and tl["actions"][0]["action"] == "smile"
    assert tl["seconds"] == 60
    presence = live.client.get("/api/presence").json()
    assert presence["mood"] == "Happiness" and presence["valence"] == pytest.approx(0.525)  # live EMA after 3 frames of 0.6
    assert presence["arousal"] == pytest.approx(0.088)  # 0.05, 0.075, 0.0875 -> rounded to 3


def test_presence_valence_is_live_even_without_mood(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    weak = Expression(dominant="Neutral", scores={"Neutral": 0.4, "Happiness": 0.3}, valence=0.2, arousal=0.0)
    _enable_expression(live, fake_expression, [weak])
    _recognize(live)
    p = live.client.get("/api/presence").json()
    assert p["mood"] is None and p["valence"] == pytest.approx(0.1) and p["arousal"] == pytest.approx(0.0)
    tl = live.client.get("/api/expression/timeline").json()
    assert len(tl["samples"]) == 1 and tl["samples"][0]["mood"] is None and tl["samples"][0]["valence"] == pytest.approx(0.1)


def test_presence_reset_clears_the_timeline_and_active_actions(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    _enable_expression(live, fake_expression, [smiling(0.9)])
    _recognize(live)  # smile active (onset), one sample recorded
    assert len(live.client.get("/api/expression/timeline").json()["samples"]) == 1
    live.client.post("/api/presence/reset")
    assert live.client.get("/api/expression/timeline").json() == {"seconds": 600, "samples": [], "moods": [], "actions": []}
    fake_expression.expressions = [smiling(0.0)]
    _recognize(live)
    assert live.client.get("/api/expression/timeline").json()["actions"] == []  # the smile's offset is unknown -> no event


def test_reset_announces_the_forget_on_the_wire_only_when_something_was_forgotten(live, fake_engine, fake_expression, face):
    """A mirror (Hermes plugin, pane) keeps its own copy of the mood/action history, so the user's explicit
    "forget" has to be visible on the wire. A NO_SIGNAL transition is not that signal: an ordinary presence
    expiry publishes one too and deliberately keeps the timeline."""
    fake_engine.faces = [face]
    _enable_expression(live, fake_expression, [smiling(0.9)])
    _recognize(live)  # seq 1 presence; one affect sample recorded
    live.client.post("/api/presence/reset")  # seq 2 presence NO_SIGNAL, seq 3 timeline_cleared
    live.client.post("/api/presence/reset")  # nothing left: no transition, no clear announcement
    frames = live.sse("/api/events?after=0", wanted=4, skip_heartbeats=True)
    assert [f["event"] for f in frames] == ["hello", "presence", "presence", "timeline_cleared"]
    cleared = frames[3]
    assert set(cleared["data"]) == {"sequence", "at"}  # a timestamp and nothing else: no names, no labels
    assert cleared["data"]["at"].endswith("+00:00") and cleared["id"] == "3"


def test_toggle_off_drops_active_actions(live, fake_engine, fake_expression, face):
    """The smile that was running when expressions went off must not be completed by a later frame.

    Toggling back on before the neutral frame is what makes this discriminating: the frame *is*
    readable (so it would close a still-active smile, ``action_min_frames`` being 1), and it does
    not carry the expression-off ``None`` that would drop the action for a second reason.
    """
    fake_engine.faces = [face]
    _enable_expression(live, fake_expression, [smiling(0.9)])
    _recognize(live)  # seq 1 presence; smile active (onset, 1 frame)
    assert live.client.post("/api/expression", json={"enabled": False}).json()["enabled"] is False
    assert live.client.post("/api/expression", json={"enabled": True}).json()["enabled"] is True
    fake_expression.expressions = [smiling(0.0)]
    _recognize(live)  # readable neutral frame: it would close a smile that was still active
    frames = live.sse("/api/events?after=0", wanted=3)  # heartbeats included: proves nothing else was buffered
    assert [f["event"] for f in frames] == ["hello", "presence", "heartbeat"]  # no `action`: dropped, not completed
    assert live.client.get("/api/expression/timeline").json()["actions"] == []


def test_toggle_off_clears_a_live_affect_without_a_committed_mood(live, fake_engine, fake_expression, face):
    """The live valence exists long before a mood commits, so the toggle clears the presence unconditionally."""
    fake_engine.faces = [face]
    weak = Expression(dominant="Neutral", scores={"Neutral": 0.4, "Happiness": 0.3}, valence=0.2, arousal=0.0)
    _enable_expression(live, fake_expression, [weak])
    _recognize(live)  # seq 1 presence; EMA 0.2 < min_score: live affect, no mood
    presence = live.client.get("/api/presence").json()
    assert presence["mood"] is None and presence["valence"] == pytest.approx(0.1)
    assert live.client.post("/api/expression", json={"enabled": False}).json()["enabled"] is False
    presence = live.client.get("/api/presence").json()
    assert (presence["state"], presence["mood"], presence["valence"], presence["arousal"]) == ("UNKNOWN", None, None, None)
    frames = live.sse("/api/events?after=0", wanted=3)  # no mood was committed, so nothing ends on the wire
    assert [f["event"] for f in frames] == ["hello", "presence", "heartbeat"]


def _timeline_moods(live: LiveServer) -> list[tuple[str | None, str | None]]:
    return [(m["from_mood"], m["to_mood"]) for m in live.client.get("/api/expression/timeline").json()["moods"]]


def test_toggle_off_mood_end_is_visible_in_the_timeline(live, fake_engine, fake_expression, face):
    """The wire and the timeline have to agree: a mood the toggle ends is *closed* in the history too.

    That end is recorded after the last affect sample — the case a sample-anchored window used to hide,
    leaving a mirror replaying the timeline stuck on a mood the user already switched off.
    """
    fake_engine.faces = [face]
    _enable_expression(live, fake_expression, [HAPPY])
    _recognize(live, times=2)  # mood_stable_ticks is 1, but the EMA (alpha 0.5, from 0) needs two frames
    assert _timeline_moods(live) == [(None, "Happiness")]
    assert live.client.post("/api/expression", json={"enabled": False}).json()["enabled"] is False
    assert _timeline_moods(live) == [(None, "Happiness"), ("Happiness", None)]
    ended = live.client.get("/api/expression/timeline").json()["moods"][-1]
    assert ended["from_mood"] == "Happiness" and ended["to_mood"] is None


def test_expired_presence_mood_end_is_visible_in_the_timeline(tmp_path: Path, fake_engine, fake_expression, face):
    """Same for the lazy expiry: the person stopped sending frames, so the end has no frame of its own.

    The shared ``live`` fixture cannot express the short stale window, hence a second server.
    """
    settings = Settings(
        data_dir=tmp_path, presence_stable_ticks=1, mood_stable_ticks=1, presence_stale_seconds=0.2, events_heartbeat_seconds=0.05
    )
    with _live_server(settings, fake_engine, fake_expression) as live:
        fake_engine.faces = [face]
        _enable_expression(live, fake_expression, [HAPPY])
        _recognize(live, times=2)  # UNKNOWN (seq 1), mood Happiness (seq 2)
        assert _timeline_moods(live) == [(None, "Happiness")]
        time.sleep(0.4)  # longer than stale_seconds: no frames
        assert live.client.get("/api/presence").json()["state"] == "NO_SIGNAL"  # the read is what expires it
        assert _timeline_moods(live) == [(None, "Happiness"), ("Happiness", None)]
        ended = live.client.get("/api/expression/timeline").json()["moods"][-1]
        assert ended["from_mood"] == "Happiness" and ended["to_mood"] is None
