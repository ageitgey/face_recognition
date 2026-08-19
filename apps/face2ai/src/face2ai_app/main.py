from __future__ import annotations

import logging
import os
import socket
from datetime import timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from face2ai_app.adapters.face_recognition_engine import FaceRecognitionEngine
from face2ai_app.adapters.json_identity_store import JsonIdentityStore
from face2ai_app.adapters.mediapipe_expression import MediaPipeExpressionEngine
from face2ai_app.api.routes import router
from face2ai_app.config import Settings
from face2ai_app.domain.errors import IdentityStoreCorrupted, IdentityStoreUnavailable
from face2ai_app.services.actions import ActionTracker
from face2ai_app.services.events import IdentityEventBroker
from face2ai_app.services.identity_service import IdentityService
from face2ai_app.services.mood import MoodTracker
from face2ai_app.services.presence import PresenceTracker
from face2ai_app.services.timeline import AffectHistory

STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)


def _log_engine(name: str, available: bool, reason: str | None, *, level: int = logging.WARNING) -> None:
    """One startup line per engine; ``level`` applies to the unavailable case."""
    if available:
        logger.info("%s engine available", name)
    else:
        logger.log(level, "%s engine unavailable: %s", name, reason or "no reason given")



class RevalidatedStaticFiles(StaticFiles):
    """Static assets that browsers must revalidate on every load (ETag/304 still applies).

    Without an explicit Cache-Control the browser caches heuristically from Last-Modified, so a
    redeployed ES-module graph can pair a fresh app.js with a stale model.js and fail on import.
    """

    async def get_response(self, path: str, scope: Scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response

def create_app(*, settings: Settings | None = None, engine=None, store=None, expression=None) -> FastAPI:
    app_settings = settings or Settings.from_env()
    recognition_engine = engine or FaceRecognitionEngine(max_frame_pixels=app_settings.max_frame_pixels)
    identity_store = store or JsonIdentityStore(app_settings.identity_store_path)
    # Always constructed: it is unavailable-not-crashing when the extra or the model asset is missing.
    expression_engine = expression or MediaPipeExpressionEngine(
        app_settings.expression_models_dir, max_frame_pixels=app_settings.max_frame_pixels
    )
    _log_engine("recognition", recognition_engine.available, recognition_engine.availability_reason)
    # Expression is opt-in: unavailable while nobody asked is information; unavailable while
    # FACE2AI_EXPRESSION_ENABLED is set is the one warning (the opt-in then stays off).
    _log_engine(
        "expression",
        expression_engine.available,
        expression_engine.availability_reason,
        level=logging.WARNING if app_settings.expression_enabled else logging.INFO,
    )

    app = FastAPI(title="Face2AI", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    app.state.settings = app_settings
    service = IdentityService(
        engine=recognition_engine,
        store=identity_store,
        tolerance=app_settings.match_tolerance,
        expression=expression_engine,
    )
    # Opt-in: env can pre-enable, but only an available engine; POST /api/expression toggles at runtime.
    service.expression_enabled = bool(app_settings.expression_enabled and expression_engine.available)
    app.state.identity_service = service
    # Presence + events: derived from RecognitionEvents, consumed by agents / Party Mirror.
    # They never touch matching and never carry frames or encodings.
    app.state.presence = PresenceTracker(
        stable_ticks=app_settings.presence_stable_ticks,
        stale_after=timedelta(seconds=app_settings.presence_stale_seconds),
    )
    # Mood decorates the presence (a hint, never a fact); fed only while expression is enabled.
    app.state.mood = MoodTracker(stable_ticks=app_settings.mood_stable_ticks, min_score=app_settings.mood_min_score)
    # Stage 2 (ADR-004): facial action dynamics + a bounded in-memory affect history. Hints, never
    # facts, never gates; nothing persisted — cleared on presence reset and gone with the process.
    app.state.actions = ActionTracker(
        on_threshold=app_settings.action_on_threshold,
        off_threshold=app_settings.action_off_threshold,
        min_frames=app_settings.action_min_frames,
    )
    app.state.history = AffectHistory(max_seconds=app_settings.timeline_seconds)
    app.state.events = IdentityEventBroker(buffer_size=app_settings.events_buffer_size)

    # The store's closed error set, mapped once for every route that touches it: unusable data and
    # an unreachable store are both "the service cannot answer right now", never a bad request.
    @app.exception_handler(IdentityStoreCorrupted)
    async def identity_store_corrupted_handler(
        request: Request, exc: IdentityStoreCorrupted
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(IdentityStoreUnavailable)
    async def identity_store_unavailable_handler(
        request: Request, exc: IdentityStoreUnavailable
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.include_router(router)
    app.mount("/assets", RevalidatedStaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def _broker_of(app: object, *, depth: int = 8) -> IdentityEventBroker | None:
    """The app's broker, reached through uvicorn's wrappers (``ProxyHeadersMiddleware``)."""
    for _ in range(depth):
        if app is None:
            return None
        events = getattr(getattr(app, "state", None), "events", None)
        if isinstance(events, IdentityEventBroker):
            return events
        app = getattr(app, "app", None)
    return None


class Face2AIServer(uvicorn.Server):
    """The process owns its shutdown: it ends its own streams before waiting for them.

    ``uvicorn.Server.shutdown`` awaits ``_wait_tasks_to_complete()`` *before* ``lifespan.shutdown()``
    (uvicorn 0.48 ``server.py``), so a FastAPI lifespan hook can never release the SSE streams the
    wait is blocked on. With one browser tab attached that made the process survive SIGTERM x3 and
    SIGINT x2 — only SIGKILL worked. Closing the broker here, the first shutdown seam that runs,
    exits in ~0.2 s; ``timeout_graceful_shutdown`` stays a backstop, never the mechanism.
    """

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        broker = _broker_of(self.config.loaded_app)
        if broker is not None:
            broker.close()
        else:  # pragma: no cover - only reachable if the app stops carrying its broker
            logger.warning("no event broker found on the loaded app; SSE streams may delay shutdown")
        await super().shutdown(sockets=sockets)


def run(*, timeout_graceful_shutdown: int = 10) -> None:
    """Serve the app. ``timeout_graceful_shutdown`` is the backstop; ``Face2AIServer`` is the mechanism.

    The app is built by uvicorn from the ``create_app`` factory: no module-level instance, so
    importing this module (the test suite does) neither costs seconds nor binds a real engine.
    """
    settings = Settings.from_env()
    logging.basicConfig(
        level=os.getenv("FACE2AI_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = uvicorn.Config(
        create_app,
        factory=True,
        host=settings.host,
        port=settings.port,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
    )
    Face2AIServer(config).run()
