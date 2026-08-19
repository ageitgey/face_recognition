"""The process owns its shutdown.

The browser opens an SSE connection on page load, so "one client attached" is the normal case.
Before this suite existed, SIGTERM x3 and SIGINT x2 left the server running and only SIGKILL
worked: the stream coroutine parks on ``queue.get()`` forever, and uvicorn's ``Server.shutdown()``
awaits ``_wait_tasks_to_complete()`` *before* ``lifespan.shutdown()``, so no lifespan hook can
unblock the wait it is meant to end. The owner of the rule is therefore ``Face2AIServer``.

The process test runs the real entry point (``main.run``) in a subprocess with a 30 s graceful
backstop: only a real broker close can make it exit inside the 2 s budget.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from face2ai_app.services.events import IdentityEventBroker

RUNNER = "from face2ai_app.main import run; run(timeout_graceful_shutdown=30)"
START_TIMEOUT = 60.0  # cold import of the recognition/expression adapters is seconds, not milliseconds
SHUTDOWN_BUDGET = 2.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _attach_sse(port: int, deadline: float) -> socket.socket:
    """Attach one raw SSE client and return the socket once ``event: hello`` has arrived."""
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        except OSError:
            time.sleep(0.1)
            continue
        sock.sendall(
            f"GET /api/events?role=client HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\nAccept: text/event-stream\r\n\r\n".encode()
        )
        sock.settimeout(10.0)
        buffer = b""
        try:
            while b"event: hello" not in buffer:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
        except socket.timeout:
            pass
        if b"event: hello" in buffer:
            return sock
        sock.close()
    raise AssertionError("server never answered /api/events with a hello frame")


def test_sigterm_stops_the_process_with_an_sse_client_attached(tmp_path: Path) -> None:
    port = _free_port()
    log = tmp_path / "server.log"
    env = {
        **os.environ,
        "FACE2AI_HOST": "127.0.0.1",
        "FACE2AI_PORT": str(port),
        "FACE2AI_DATA_DIR": str(tmp_path),
    }
    with log.open("wb") as sink:
        process = subprocess.Popen([sys.executable, "-c", RUNNER], env=env, stdout=sink, stderr=sink)
    sock = None
    try:
        sock = _attach_sse(port, time.monotonic() + START_TIMEOUT)
        process.send_signal(signal.SIGTERM)
        started = time.monotonic()
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "process survived SIGTERM for 5 s with one SSE client attached; "
                f"server log:\n{log.read_text(errors='replace')[-2000:]}"
            )
        elapsed = time.monotonic() - started
        # uvicorn re-raises the captured signal, so -SIGTERM is as correct an exit as 0.
        assert returncode in (0, -signal.SIGTERM), f"unexpected exit {returncode}"
        assert elapsed < SHUTDOWN_BUDGET, f"shutdown took {elapsed:.2f} s (backstop, not a real close)"
    finally:
        if sock is not None:
            sock.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_close_wakes_a_parked_subscriber() -> None:
    """A subscriber parked on ``queue.get()`` receives the sentinel: that is what ends the stream."""

    async def scenario() -> object:
        broker = IdentityEventBroker()
        subscription = broker.subscribe()
        getter = asyncio.ensure_future(subscription.queue.get())
        await asyncio.sleep(0)  # park the getter before closing
        assert not getter.done()
        broker.close()
        return await asyncio.wait_for(getter, timeout=2.0)

    assert asyncio.run(scenario()) is None


def test_subscribe_after_close_hands_out_a_closed_subscription() -> None:
    """A request arriving during shutdown must not re-pin the process."""

    async def scenario() -> object:
        broker = IdentityEventBroker()
        broker.close()
        subscription = broker.subscribe()
        return await asyncio.wait_for(subscription.queue.get(), timeout=2.0)

    assert asyncio.run(scenario()) is None


def test_publish_after_close_is_a_no_op() -> None:
    async def scenario() -> tuple[int, object]:
        broker = IdentityEventBroker()
        subscription = broker.subscribe()
        broker.close()
        assert await asyncio.wait_for(subscription.queue.get(), timeout=2.0) is None
        broker.publish("presence", {"state": "KNOWN"})
        return broker.last_sequence, broker.replay(0)

    sequence, replayed = asyncio.run(scenario())
    assert sequence == 0
    assert replayed == []
