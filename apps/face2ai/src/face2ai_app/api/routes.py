from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from face2ai_app import __version__
from face2ai_app.domain.errors import EnrollmentRejected, InvalidFrame, RecognitionUnavailable
from face2ai_app.domain.models import (
    AffectSample,
    Expression,
    IdentitySummary,
    Presence,
    PresenceState,
    PresenceTransition,
    RecognitionEvent,
    StoreEvent,
    StoreEventKind,
    SystemStatus,
    TimelineSnapshot,
)
from face2ai_app.services.actions import ActionTracker
from face2ai_app.services.events import IdentityEvent, IdentityEventBroker
from face2ai_app.services.mood import MoodTracker
from face2ai_app.services.presence import PresenceTracker
from face2ai_app.services.timeline import AffectHistory

logger = logging.getLogger(__name__)

router = APIRouter()

EVENT_ROLE_AGENT = "agent"
AGENT_HEADER = "X-Face2AI-Agent"  # "1"/"0" on recognize responses: fresh greeting-ownership signal for the browser
SAME_ORIGIN = "same-origin"  # the only Sec-Fetch-Site value allowed to claim EVENT_ROLE_AGENT


def _refuse_cross_origin_agent(role: str, sec_fetch_site: str | None) -> None:
    """Only a same-origin caller may take greeting ownership via ``?role=agent``.

    Subscribing as the agent makes the browser shell go deliberately silent (it leaves the spoken
    greeting to the agent), so without this check any page the user happens to have open can reach
    the port and switch the product's headline behaviour off. The port is reverse-tunnelled to a
    VPS, so "loopback, therefore safe" is the wrong frame.

    The discriminator is ``Sec-Fetch-Site``: browsers attach it to every request and a page cannot
    forge it, while the real consumers (voice agent, Hermes plugin) drive httpx and never send it.
    Absent header => not a browser => allowed. 403 rather than 401: the request is understood and
    refused on origin, and no credential the caller could add would change that.
    """
    if role != EVENT_ROLE_AGENT or sec_fetch_site is None:
        return
    if sec_fetch_site.strip().lower() != SAME_ORIGIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"role={EVENT_ROLE_AGENT} needs a same-origin request; this one carries "
                f"Sec-Fetch-Site: {sec_fetch_site}. Subscribe with another role to read the stream."
            ),
        )


def _service(request: Request):
    return request.app.state.identity_service


def _settings(request: Request):
    return request.app.state.settings


def _presence(request: Request) -> PresenceTracker:
    return request.app.state.presence


def _mood(request: Request) -> MoodTracker:
    return request.app.state.mood


def _events(request: Request) -> IdentityEventBroker:
    return request.app.state.events


def _actions(request: Request) -> ActionTracker:
    return request.app.state.actions


def _history(request: Request) -> AffectHistory:
    return request.app.state.history


async def _image_body(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "")
    if not (content_type.startswith("image/jpeg") or content_type.startswith("image/png")):
        raise HTTPException(status_code=415, detail="content-type must be image/jpeg or image/png")
    body = await request.body()
    if len(body) > _settings(request).max_frame_bytes:
        raise HTTPException(status_code=413, detail="image payload exceeds configured limit")
    if not body:
        raise HTTPException(status_code=400, detail="empty image payload")
    return body


def _publish_store_event(
    request: Request, kind: StoreEventKind, *, identity_id: str | None, display_name: str | None, identity_count: int
) -> None:
    _events(request).publish(
        "store",
        StoreEvent(
            at=datetime.now(timezone.utc),
            kind=kind,
            identity_id=identity_id,
            display_name=display_name,
            identity_count=identity_count,
        ),
    )


def _publish_presence_transition(request: Request, transition: PresenceTransition | None) -> None:
    """Publish a committed presence transition (no-op for None).

    Crossing NO_SIGNAL — expiry, reset, or an arrival after a frame gap — also forgets the mood:
    it belonged to a presence that is gone, and no further frame will announce its end. If a mood
    was set, its ``mood -> None`` end is published right after the presence event, so the wire is
    symmetric with a frame-driven end. (The presence itself starts fresh, without mood, on every
    committed transition, so nothing has to be cleared on the tracker.) Active facial actions are
    dropped the same way — their offset is unknown, so no ``action`` event is guessed.
    """
    if transition is None:
        return
    mood_ended = None
    if PresenceState.NO_SIGNAL in (transition.from_state, transition.to_state):
        mood_ended = _mood(request).reset(transition.at)
        _actions(request).reset()
    if mood_ended is not None:
        _history(request).record_mood(mood_ended)  # the timeline sees every mood end, not only frame-driven ones
    _events(request).publish("presence", transition)
    if mood_ended is not None:
        _events(request).publish("mood", mood_ended)


def _primary_expression(presence: Presence, event: RecognitionEvent) -> Expression | None:
    """The expression the mood follows: the single face's, only while one person is stably present."""
    if presence.state in (PresenceState.KNOWN, PresenceState.UNKNOWN) and len(event.faces) == 1:
        return event.faces[0].expression
    return None


def _observe_expression(request: Request, event: RecognitionEvent, now: datetime) -> None:
    """One frame -> mood (hysteresis label), live affect on the presence, action events, history samples.

    Everything runs under the stable presence's key. Frames without a usable expression (expression
    off, several faces, no face) count as missing for the mood, drop active actions and add no sample.
    The presence carries the *live* smoothed valence/arousal (Stage 2); the ``mood`` event keeps the
    values frozen at commit. Actions and history are hints only — a failure there must never break
    ``/api/recognize`` (warned once, then debug).
    """
    tracker = _presence(request)
    presence = tracker.snapshot()
    key = f"{presence.state}:{presence.identity_id or ''}"
    who = {"identity_id": presence.identity_id, "display_name": presence.display_name}
    # toggle-off race: a frame that left the browser before the toggle must not restart affect/actions
    expression = None if not _service(request).expression_enabled else _primary_expression(presence, event)
    mood, history, events = _mood(request), _history(request), _events(request)

    transition = mood.observe(key, expression, now, **who)
    if transition is not None:
        events.publish("mood", transition)
        history.record_mood(transition)
    current_mood, _, _ = mood.current()
    valence, arousal = mood.affect()
    tracker.set_mood(current_mood, valence, arousal)  # Stage 2: presence carries the live affect
    try:
        if expression is not None and (valence is not None or arousal is not None):
            history.record_sample(AffectSample(at=now, mood=current_mood, valence=valence, arousal=arousal, **who))
        for action in _actions(request).observe(key, expression, now, **who):
            events.publish("action", action)
            history.record_action(action)
    except Exception as exc:  # hints must never take recognition down (e.g. a model validation error)
        state = request.app.state
        first = not getattr(state, "expression_dynamics_warned", False)
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "expression dynamics failed, recognition continues without them: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=first,  # the one WARNING carries the traceback; the DEBUG repeats stay one-liners
        )
        state.expression_dynamics_warned = True


@router.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readiness(request: Request, response: Response) -> dict[str, str | bool | None]:
    service = _service(request)
    ready = bool(service.engine.available)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "reason": service.engine.availability_reason}


@router.get("/api/status", response_model=SystemStatus)
def system_status(request: Request) -> SystemStatus:
    service = _service(request)
    settings = _settings(request)
    events = _events(request)
    return SystemStatus(
        version=__version__,
        engine_available=service.engine.available,
        engine_reason=service.engine.availability_reason,
        identity_count=len(service.store.list()),
        greeting_cooldown_seconds=settings.greeting_cooldown_seconds,
        agent_connected=events.connected(EVENT_ROLE_AGENT),
        event_subscribers=events.subscriber_count,
        expression_available=service.expression_available,
        expression_reason=service.expression_reason,
        expression_enabled=bool(service.expression_enabled),
    )


class ExpressionToggle(BaseModel):
    enabled: bool


@router.post("/api/expression")
async def set_expression(request: Request, body: ExpressionToggle) -> dict[str, bool]:
    """Runtime opt-in: attach best-effort expression hints to recognize responses. Off by default.

    Enabling needs a loaded engine (409 otherwise); disabling is always allowed.
    """
    service = _service(request)
    if body.enabled and not service.expression_available:
        raise HTTPException(status_code=409, detail=f"expression engine unavailable: {service.expression_reason}")
    service.expression_enabled = body.enabled
    if not body.enabled:
        # No further frame will carry an expression, so the current mood ends now rather than after
        # stable_ticks missing frames. Presence itself is unchanged, so its mood *and* live affect
        # fields are cleared — unconditionally: a valence can be live without a committed mood.
        mood_ended = _mood(request).reset()
        _actions(request).reset()  # active actions are dropped, not completed: their offset is unknown
        _presence(request).set_mood(None, None, None)
        if mood_ended is not None:
            _history(request).record_mood(mood_ended)
            _events(request).publish("mood", mood_ended)
    return {"enabled": service.expression_enabled, "available": service.expression_available}


@router.post("/api/recognize", response_model=RecognitionEvent)
async def recognize(request: Request, response: Response) -> RecognitionEvent:
    body = await _image_body(request)
    try:
        # dlib work runs off the event loop so SSE subscribers keep receiving heartbeats/events.
        event = await run_in_threadpool(_service(request).recognize, body)
    except RecognitionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidFrame as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    now = datetime.now(timezone.utc)
    _publish_presence_transition(request, _presence(request).observe(event, now))
    # no await between these two: set_mood must land on the presence just observed
    _observe_expression(request, event, now)
    response.headers[AGENT_HEADER] = "1" if _events(request).connected(EVENT_ROLE_AGENT) else "0"
    return event


@router.post("/api/enroll", response_model=IdentitySummary, status_code=201)
async def enroll(
    request: Request,
    display_name: str = Query(min_length=1, max_length=80),
    consent: bool = Query(default=False),
) -> IdentitySummary:
    body = await _image_body(request)
    service = _service(request)
    try:
        summary = await run_in_threadpool(service.enroll, body, display_name=display_name, consent=consent)
    except RecognitionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (EnrollmentRejected, InvalidFrame) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    count = await run_in_threadpool(lambda: len(service.store.list()))
    _publish_store_event(request, StoreEventKind.ENROLLED, identity_id=summary.id, display_name=summary.display_name, identity_count=count)
    return summary


@router.get("/api/identities", response_model=list[IdentitySummary])
def identities(request: Request) -> list[IdentitySummary]:
    return _service(request).summaries()


@router.delete("/api/identities/{identity_id}", status_code=204)
async def delete_identity(identity_id: str, request: Request) -> Response:
    store = _service(request).store

    def _delete() -> tuple[bool, str | None, int]:
        display_name = next((r.display_name for r in store.list() if r.id == identity_id), None)
        deleted = store.delete(identity_id)
        return deleted, display_name, len(store.list()) if deleted else 0

    deleted, display_name, count = await run_in_threadpool(_delete)
    if not deleted:
        raise HTTPException(status_code=404, detail="identity not found")
    _publish_store_event(request, StoreEventKind.DELETED, identity_id=identity_id, display_name=display_name, identity_count=count)
    return Response(status_code=204)


@router.delete("/api/identities")
async def delete_all_identities(request: Request) -> dict[str, int]:
    deleted = await run_in_threadpool(_service(request).store.clear)
    _publish_store_event(request, StoreEventKind.ERASED, identity_id=None, display_name=None, identity_count=0)
    return {"deleted": deleted}


# ----------------------------------------------------------------- presence + events


@router.get("/api/presence", response_model=Presence)
async def presence(request: Request) -> Presence:
    _publish_presence_transition(request, _presence(request).expire())
    return _presence(request).snapshot()


@router.post("/api/presence/reset", response_model=Presence)
async def reset_presence(request: Request) -> Presence:
    """Browser stopped/paused the camera or the page is going away: presence returns to NO_SIGNAL
    and everything derived from it is forgotten — the mood, active facial actions and the whole
    in-memory affect history (``/api/expression/timeline`` is empty afterwards). Publishes the
    presence transition and, if a mood was set, its end; when presence already was NO_SIGNAL the
    presence and mood are unchanged and nothing is published (crossing NO_SIGNAL already reset the
    mood tracker), but the history is still cleared — and a non-empty clear is announced as
    ``timeline_cleared`` so mirrors (Hermes plugin, panes) forget with us."""
    _publish_presence_transition(request, _presence(request).reset())
    _actions(request).reset()
    if _history(request).clear():
        # The forget is a privacy control, so it has to be visible on the wire: a NO_SIGNAL transition
        # alone is not the signal — an ordinary presence expiry publishes one too and keeps the history.
        _events(request).publish("timeline_cleared", {"at": datetime.now(timezone.utc).isoformat()})
    return _presence(request).snapshot()


@router.get("/api/expression/timeline", response_model=TimelineSnapshot)
def expression_timeline(
    request: Request,
    seconds: int = Query(default=600, ge=10, le=3600),
    identity_id: str | None = Query(default=None, max_length=80),
) -> TimelineSnapshot:
    """In-memory affect history (live valence/arousal samples, mood changes, completed facial actions)
    of the last ``seconds`` — bounded, never persisted, cleared on ``POST /api/presence/reset`` and
    restart. Hints, never facts. ``identity_id`` narrows it to one known person (empty = no filter)."""
    return _history(request).snapshot(seconds=seconds, identity_id=identity_id or None)


def _sse(event: str, data: dict[str, Any], event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


def _sse_event(event: IdentityEvent) -> str:
    return _sse(event.kind, {"sequence": event.sequence, **event.payload}, event.sequence)


@router.get("/api/events")
async def events(
    request: Request,
    role: str = Query(default="client", min_length=1, max_length=32),
    after: int | None = Query(default=None, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    sec_fetch_site: str | None = Header(default=None, alias="Sec-Fetch-Site"),
) -> StreamingResponse:
    """Server-Sent Events: `hello`, then `presence` / `mood` / `action` / `store` events, `heartbeat` while idle.

    `action` is a completed facial action (onset/apex/offset timestamps, peak, duration_ms, frames) —
    expression dynamics at frame-rate resolution; a hint, never a fact, and nothing may gate on it.

    Consumers such as the voice agent subscribe with ``?role=agent``; the browser then knows an
    agent is present (``/api/status.agent_connected``) and leaves greetings to it. Reconnects may
    pass ``Last-Event-ID`` (or ``?after=``) to replay buffered events.

    ``?role=agent`` is refused with 403 for any request whose ``Sec-Fetch-Site`` is not
    ``same-origin``, so a page from elsewhere cannot take greeting ownership; every other role is
    unaffected.
    """
    # Refuse before the StreamingResponse exists: once it is returned the 200 is already committed
    # and ``stream()`` — which registers the subscription — runs with no way back.
    _refuse_cross_origin_agent(role, sec_fetch_site)
    broker = _events(request)
    tracker = _presence(request)
    settings = _settings(request)
    resume_from = after
    if resume_from is None and last_event_id and last_event_id.isdigit():
        resume_from = int(last_event_id)

    async def stream() -> AsyncIterator[str]:
        subscription = broker.subscribe(role)
        last_sent = subscription.since_sequence
        try:
            _publish_presence_transition(request, tracker.expire())
            yield _sse(
                "hello",
                {
                    "presence": tracker.snapshot().model_dump(mode="json"),
                    "last_sequence": subscription.since_sequence,
                    "greeting_cooldown_seconds": settings.greeting_cooldown_seconds,
                    "engine_available": bool(_service(request).engine.available),
                },
            )
            if resume_from is not None:
                # Replay only up to the sequence seen at subscribe time; everything after that
                # arrives through the queue, so nothing is delivered twice.
                for missed in broker.replay(resume_from, up_to=subscription.since_sequence):
                    yield _sse_event(missed)
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(
                        subscription.queue.get(), timeout=settings.events_heartbeat_seconds
                    )
                except asyncio.TimeoutError:
                    # wait_for cancelling a woken get() can drop the item, so the sentinel alone is
                    # not enough: the closed broker is the second, authoritative end condition.
                    if broker.closed:
                        return
                    expired = tracker.expire()
                    if expired is not None:
                        _publish_presence_transition(request, expired)  # delivered through the queue on the next loop
                        continue
                    yield _sse("heartbeat", {"presence": tracker.snapshot().model_dump(mode="json")})
                    continue
                if event is None:  # broker closed: the process is shutting down
                    return
                if event.sequence <= last_sent:
                    continue
                last_sent = event.sequence
                yield _sse_event(event)
        finally:
            broker.unsubscribe(subscription)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
