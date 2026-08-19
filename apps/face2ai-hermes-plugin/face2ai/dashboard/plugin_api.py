"""Backend routes for the Face2AI plugin, mounted at /api/plugins/face2ai/ by the Hermes
dashboard / `hermes serve` process (the one the desktop app talks to).

- GET  /presence   live presence (proxied from Face2AI; falls back to the gateway's persisted snapshot)
- GET  /history    recent transitions as recorded by the gateway-side consumer
- GET  /timeline   Face2AI's in-memory affect history (valence/arousal samples, mood changes, facial
                   actions of the last `seconds`, optional `identity_id`) proxied for the pane sparkline
- GET  /health     plugin + Face2AI reachability
- WS   /events     live twin for the desktop pane: relays Face2AI's SSE frames as JSON
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger("hermes.plugins.face2ai.api")
router = APIRouter()

PLUGIN_ID = "face2ai"
DEFAULT_EVENTS_URL = "http://127.0.0.1:8765"
TIMELINE_DEFAULT_SECONDS = 600
TIMELINE_MIN_SECONDS, TIMELINE_MAX_SECONDS = 10, 3600  # Face2AI's own bounds for GET /api/expression/timeline


def _settings() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        return ((cfg.get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID, {}).get("settings") or {}
    except Exception:
        return {}


def _events_url() -> str:
    return str(_settings().get("events_url") or DEFAULT_EVENTS_URL).rstrip("/")


def _state_path() -> Path | None:
    try:
        from hermes_cli.plugins import PluginState

        return PluginState(PLUGIN_ID).path
    except Exception:
        return None


def _snapshot_from_state() -> dict[str, Any] | None:
    path = _state_path()
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    snap = data.get("snapshot") if isinstance(data, dict) else None
    return snap if isinstance(snap, dict) else None


@router.get("/presence")
async def presence() -> dict[str, Any]:
    """Live presence straight from Face2AI; snapshot fallback when the tunnel/app is down."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{_events_url()}/api/presence")
            response.raise_for_status()
            # `connected` is what the desktop chip reads, and `refresh()` replaces its `latest`
            # wholesale — omitting it here reads as "nicht verbunden" after every *successful* poll.
            return {"source": "live", "connected": True, "presence": response.json(), "events_url": _events_url()}
    except Exception as exc:
        snap = _snapshot_from_state()
        if snap:
            return {"source": "snapshot", "error": str(exc), **snap}
        return {"source": "none", "error": str(exc), "presence": {"state": "NO_SIGNAL"}, "connected": False}


@router.get("/history")
async def history() -> dict[str, Any]:
    snap = _snapshot_from_state() or {}
    return {"history": snap.get("history", []), "connected": snap.get("connected", False), "last_frame_at": snap.get("last_frame_at")}


@router.get("/timeline")
async def timeline(seconds: int = Query(default=TIMELINE_DEFAULT_SECONDS), identity_id: str | None = Query(default=None, max_length=80)) -> dict[str, Any]:
    """Proxy of Face2AI's ``GET /api/expression/timeline`` — bounded, in memory on the Face2AI side, cleared on
    presence reset/restart; hints, never facts. An out-of-range ``seconds`` is clamped into Face2AI's range instead
    of failing so the pane always gets a well-formed answer (a non-integer is still rejected by FastAPI with 422
    before this runs); on any error the shape stays the same with empty lists."""
    seconds = max(TIMELINE_MIN_SECONDS, min(TIMELINE_MAX_SECONDS, int(seconds)))
    params: dict[str, Any] = {"seconds": seconds}
    identity_id = (identity_id or "").strip()  # " " is not a person: Face2AI would filter everything away
    if identity_id:
        params["identity_id"] = identity_id
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{_events_url()}/api/expression/timeline", params=params)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("unexpected timeline payload")
            return {"seconds": body.get("seconds", seconds), "samples": body.get("samples") or [], "moods": body.get("moods") or [], "actions": body.get("actions") or []}
    except Exception as exc:
        return {"error": str(exc), "seconds": seconds, "samples": [], "moods": [], "actions": []}


@router.get("/health")
async def health() -> dict[str, Any]:
    ok = False
    detail = None
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{_events_url()}/api/status")
            ok = response.status_code == 200
            detail = response.json() if ok else response.text[:200]
    except Exception as exc:
        detail = str(exc)
    snap = _snapshot_from_state() or {}
    return {"face2ai_reachable": ok, "face2ai_status": detail, "gateway_consumer_connected": snap.get("connected", False), "events_url": _events_url()}


@router.websocket("/events")
async def events(ws: WebSocket) -> None:
    """Relay Face2AI's SSE frames to the desktop plugin as JSON messages."""
    await ws.accept()
    try:
        import httpx
    except ImportError:
        await ws.send_json({"event": "error", "data": {"error": "httpx missing"}})
        await ws.close()
        return
    url = f"{_events_url()}/api/events"
    try:
        while True:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
                    async with client.stream("GET", url, params={"role": "desktop"}, headers={"Accept": "text/event-stream"}) as response:
                        response.raise_for_status()
                        event, data_lines = "message", []
                        async for line in response.aiter_lines():
                            line = line.rstrip("\r")
                            if line == "":
                                if data_lines:
                                    try:
                                        payload = json.loads("\n".join(data_lines))
                                    except ValueError:
                                        payload = {"raw": "\n".join(data_lines)}
                                    await ws.send_json({"event": event, "data": payload})
                                event, data_lines = "message", []
                                continue
                            if line.startswith(":"):
                                continue
                            key, _, value = line.partition(":")
                            value = value[1:] if value.startswith(" ") else value
                            if key == "event":
                                event = value
                            elif key == "data":
                                data_lines.append(value)
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                await ws.send_json({"event": "lost", "data": {"error": str(exc)}})
                await asyncio.sleep(3.0)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.debug("face2ai events websocket ended: %s", exc)
