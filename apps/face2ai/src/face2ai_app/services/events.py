from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class IdentityEvent:
    """One published event: monotonically increasing ``sequence`` allows resume via Last-Event-ID.

    The payload carries its own ``at`` timestamp (PresenceTransition / StoreEvent), so the broker
    adds nothing but the sequence.
    """

    sequence: int
    kind: str
    payload: dict[str, Any]


@dataclass(eq=False, slots=True)
class Subscription:
    role: str
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[IdentityEvent | None]  # ``None`` is the sentinel: the broker closed, end the stream
    since_sequence: int  # last sequence that existed when the subscription was registered
    dropped: int = 0  # events discarded because the consumer fell behind (client resumes via replay)


class IdentityEventBroker:
    """In-process fan-out of identity/presence/store events to SSE subscribers.

    Publishing may happen from worker threads (sync route handlers), consuming happens on the
    event loop, so hand-off goes through ``loop.call_soon_threadsafe`` — scheduled inside the same
    critical section that assigns the sequence, so delivery order equals sequence order regardless
    of the publishing thread. Per-subscriber queues are bounded: a stalled consumer loses the
    oldest events (and can replay them by sequence) instead of growing memory without limit.
    Nothing here knows about faces, frames or matching: consumers see states, names and counts only.
    """

    def __init__(self, *, buffer_size: int = 200) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._buffer_size = max(1, buffer_size)
        self._buffer: deque[IdentityEvent] = deque(maxlen=self._buffer_size)
        self._subscriptions: set[Subscription] = set()
        self._closed = False

    # ------------------------------------------------------------------ publish

    def publish(self, kind: str, payload: BaseModel | dict[str, Any]) -> IdentityEvent | None:
        """Fan one event out to every subscriber; ``None`` once the broker is closed (shutdown)."""
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
        with self._lock:
            if self._closed:
                return None
            self._sequence += 1
            event = IdentityEvent(sequence=self._sequence, kind=kind, payload=data)
            self._buffer.append(event)
            for sub in list(self._subscriptions):
                try:
                    sub.loop.call_soon_threadsafe(self._enqueue, sub, event)
                except RuntimeError:
                    # Loop closed: subscriber is gone; it will be removed on unsubscribe.
                    pass
        return event

    @staticmethod
    def _enqueue(sub: Subscription, event: IdentityEvent | None) -> None:
        if sub.queue.full():
            try:
                sub.queue.get_nowait()  # drop the oldest; the client can replay by sequence
                sub.dropped += 1
            except asyncio.QueueEmpty:
                pass
        sub.queue.put_nowait(event)

    # ---------------------------------------------------------------- subscribe

    def subscribe(self, role: str = "client") -> Subscription:
        """Register a subscriber; ``since_sequence`` and the subscription are taken under one lock,
        so replaying ``(after, since_sequence]`` and then draining the queue yields every event once."""
        loop = asyncio.get_running_loop()
        with self._lock:
            sub = Subscription(
                role=role,
                loop=loop,
                queue=asyncio.Queue(maxsize=self._buffer_size),
                since_sequence=self._sequence,
            )
            if self._closed:
                # A request arriving during shutdown must not re-pin the process on a parked getter.
                sub.queue.put_nowait(None)
            self._subscriptions.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subscriptions.discard(sub)

    # -------------------------------------------------------------------- close

    def close(self) -> None:
        """End every subscription by handing each queue the ``None`` sentinel. Idempotent.

        Called from ``Face2AIServer.shutdown``: uvicorn waits for in-flight tasks *before* running
        the lifespan shutdown, so a stream parked on ``queue.get()`` has to be woken here or the
        process cannot exit (measured: SIGTERM x3 and SIGINT x2 left it running, only SIGKILL
        worked). The hand-off is ``call_soon_threadsafe`` exactly as in ``publish`` — a direct
        ``put_nowait`` from another thread does not wake a getter parked on the loop — and happens
        under the same lock, so no publish can be scheduled after the sentinel.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for sub in list(self._subscriptions):
                try:
                    sub.loop.call_soon_threadsafe(self._enqueue, sub, None)
                except RuntimeError:
                    # Loop closed: that subscriber's stream is already gone.
                    pass

    def replay(self, after_sequence: int, up_to: int | None = None) -> list[IdentityEvent]:
        with self._lock:
            return [
                event
                for event in self._buffer
                if event.sequence > after_sequence and (up_to is None or event.sequence <= up_to)
            ]

    # -------------------------------------------------------------------- state

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    def connected(self, role: str) -> bool:
        with self._lock:
            return any(sub.role == role for sub in self._subscriptions)
