"""Hermes plugin: Face2AI presence as a first-class input.

- A supervised background task subscribes to Face2AI's SSE stream (``GET /api/events?role=hermes``)
  and keeps the latest presence in memory + in the plugin's durable state (read by the dashboard
  API in the ``hermes serve``/dashboard process).
- ``pre_llm_call`` injects one ``[face2ai] …`` line into every turn (all platforms, or only the
  configured ones) so Hermes always knows who is in front of the camera.
- Tool ``presence_now`` and slash command ``/presence`` for explicit questions.
- Optional: announce arrivals proactively (``announce_arrivals`` + ``allow_gateway_injection``).

Wire contract (Face2AI apps/face2ai, ADR-002): states, display names, counts, timestamps, plus a
hedged mood hint (label + valence/arousal — "wirkt …", never a fact, never a gate for anything) and
completed facial actions (label + onset/apex/offset timestamps + one peak — "kurzes Lächeln (0.9 s)",
expression dynamics at ~0.6 s resolution, kept as history for the pane/`/presence`, never spoken into
the LLM context).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from .presence import PresenceStore, SseFrame, action_sentence, clock, context_line, describe, parse_sse

logger = logging.getLogger("hermes.plugins.face2ai")

PLUGIN_ID = "face2ai"
DEFAULT_EVENTS_URL = "http://127.0.0.1:8765"
STATE_KEY = "snapshot"

_store = PresenceStore()
_settings: dict[str, Any] = {}
_last_session_key: dict[str, str] = {}  # platform -> most recent gateway session key (for announcements)
_last_announce: dict[str, float] = {}  # identity_id -> monotonic time
_ctx: Any = None
_stop = threading.Event()


# ------------------------------------------------------------------ settings

def _setting(key: str, default: Any) -> Any:
    value = _settings.get(key)
    return default if value is None else value


def _load_settings(ctx: Any) -> None:
    for key, default in (
        ("events_url", DEFAULT_EVENTS_URL),
        ("inject_context", True),
        ("context_max_age_seconds", 30),
        ("platforms", []),
        ("announce_arrivals", False),
        ("announce_cooldown_seconds", 120),
        ("language", "de"),
    ):
        try:
            _settings[key] = ctx.get_config(key, default)
        except Exception:
            _settings[key] = default
    _settings["events_url"] = str(_settings["events_url"]).rstrip("/")


# ------------------------------------------------------------------ SSE consumer

async def _consume_events() -> None:
    """Follow the Face2AI stream forever; reconnect with backoff; persist snapshots for the API side."""
    try:
        import httpx
    except ImportError:
        logger.error("face2ai plugin needs httpx in the Hermes environment (pip install httpx)")
        return
    url = f"{_setting('events_url', DEFAULT_EVENTS_URL)}/api/events"
    last_id: str | None = None
    backoff = 2.0
    while not _stop.is_set():
        headers = {"Accept": "text/event-stream"}
        if last_id:
            headers["Last-Event-ID"] = last_id
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
                async with client.stream("GET", url, params={"role": "hermes"}, headers=headers) as response:
                    response.raise_for_status()
                    backoff = 2.0
                    buffer: list[str] = []
                    async for line in response.aiter_lines():
                        buffer.append(line)
                        if line.rstrip("\r") != "":
                            continue
                        for frame in parse_sse(buffer):
                            if frame.id:
                                last_id = frame.id
                            _handle_frame(frame)
                        buffer = []
                        if _stop.is_set():
                            return
        except (httpx.HTTPError, OSError) as exc:
            _store.mark_lost(str(exc))
            _persist()
            logger.info("face2ai stream unavailable (%s); retry in %.0fs", exc, backoff)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never let the supervisor die on a parse bug
            _store.mark_lost(f"{type(exc).__name__}: {exc}")
            logger.exception("face2ai consumer error")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


def _handle_frame(frame: SseFrame) -> None:
    transition = _store.apply(frame)
    if frame.event in ("hello", "presence", "store", "heartbeat", "mood", "action", "timeline_cleared"):
        _persist()
    if transition is not None:
        _emit_change(transition)
        if transition.to_state == "KNOWN" and transition.identity_id:
            _maybe_announce(transition)


def _persist() -> None:
    if _ctx is None:
        return
    try:
        _ctx.state.set(STATE_KEY, _store.snapshot())
    except Exception as exc:
        logger.debug("face2ai state persist failed: %s", exc)


def _emit_change(transition: Any) -> None:
    if _ctx is None:
        return
    try:
        _ctx.emit("presence_changed", transition.to_dict())
    except Exception:
        pass


def _maybe_announce(transition: Any) -> None:
    """Proactive turn on arrival — only when configured and a gateway session is known."""
    if not _setting("announce_arrivals", False) or _ctx is None:
        return
    now = time.monotonic()
    last = _last_announce.get(transition.identity_id)
    cooldown = float(_setting("announce_cooldown_seconds", 120))
    if last is not None and now - last < cooldown:
        return
    session_key = next(iter(_last_session_key.values()), None)
    if not session_key:
        logger.debug("face2ai: arrival of %s not announced (no gateway session key yet)", transition.display_name)
        return
    lang = str(_setting("language", "de"))
    text = (
        f"[face2ai] {transition.display_name} ist gerade vor die Kamera getreten (von Face2AI erkannt). Begrüße kurz mit Namen."
        if lang.startswith("de")
        else f"[face2ai] {transition.display_name} just stepped in front of the camera (recognized by Face2AI). Greet briefly by name."
    )
    try:
        if _ctx.inject_message(text, session_key=session_key):
            _last_announce[transition.identity_id] = now
    except Exception as exc:
        logger.debug("face2ai inject_message failed: %s", exc)


def _start_consumer(ctx: Any) -> None:
    coro = _consume_events()
    try:
        ctx.spawn_task(coro, name="face2ai-presence")
        logger.info("face2ai: presence consumer started (event loop)")
    except RuntimeError:
        coro.close()  # never awaited on this path; close it so Python does not warn at gateway start
        # No running loop (CLI mode): run the consumer on a private loop in a daemon thread.
        def _runner() -> None:
            asyncio.run(_consume_events())

        threading.Thread(target=_runner, name="face2ai-presence", daemon=True).start()
        logger.info("face2ai: presence consumer started (thread)")


# ------------------------------------------------------------------ hooks / tools / commands

def _on_pre_llm_call(platform: str = "", **_: Any) -> dict[str, str] | None:
    if not _setting("inject_context", True):
        return None
    allowed = [str(p) for p in (_setting("platforms", []) or [])]
    if allowed and platform not in allowed:
        return None
    line = context_line(
        _store,
        language=str(_setting("language", "de")),
        max_age_seconds=int(_setting("context_max_age_seconds", 30)),
    )
    return {"context": line} if line else None


def _on_pre_gateway_dispatch(event: Any = None, session_store: Any = None, **_: Any) -> None:
    """Remember the latest session key per platform so arrivals can be announced into it."""
    try:
        source = getattr(event, "source", None)
        if source is None or session_store is None:
            return None
        key = session_store._generate_session_key(source)  # noqa: SLF001 — no public builder takes a source
        platform = str(getattr(source, "platform", "") or "gateway")
        if key:
            _last_session_key[platform] = key
    except Exception:
        return None
    return None


def _tool_presence_now(args: dict, **_: Any) -> str:
    snap = _store.snapshot()
    snap["summary"] = describe(_store, language=str(_setting("language", "de")))
    return json.dumps(snap, ensure_ascii=False)


def _cmd_presence(raw_args: str = "") -> str:
    lang = str(_setting("language", "de"))
    text = describe(_store, language=lang)
    snap = _store.snapshot()
    if snap["history"]:
        # clock(): local wall time, like describe()'s "Zuletzt gesehen" — one reply must not mix two clocks.
        lines = [f"- {clock(t['at'])}  {t['from_state']} → {t['to_state']}" + (f" ({t['display_name']})" if t.get("display_name") else "") for t in snap["history"][-6:]]
        text += "\n" + ("Letzte Übergänge:" if lang.startswith("de") else "Recent transitions:") + "\n" + "\n".join(lines)
    shown = [sentence + (f" ({clock(a.get('at'))})" if clock(a.get("at")) else "") for a, sentence in ((a, action_sentence(a, language=lang)) for a in snap["actions"][-3:]) if sentence]
    if shown:  # facial actions: history for the human asking, deliberately not part of the injected context
        text += "\n" + ("Zuletzt gezeigt: " if lang.startswith("de") else "Recently shown: ") + ", ".join(shown)
    return text


PRESENCE_TOOL_SCHEMA = {
    "name": "presence_now",
    "description": (
        "Who is in front of the local camera right now according to Face2AI (states NO_SIGNAL, NO_FACE, "
        "UNKNOWN, KNOWN with display_name, MULTIPLE_FACES), plus recent transitions and, if any, a hedged mood "
        "hint (mood/valence/arousal — 'wirkt …', a best-effort guess from facial expression, never a fact) and the "
        "last few mood changes / completed facial actions (e.g. a brief smile, ~0.6 s timing resolution — expression "
        "dynamics, not micro-expressions, never facts). Best-effort recognition, never certainty and never authentication."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SYSTEM_SECTION = (
    "Face2AI (local face recognition on the user's computer) may prefix a turn with a line starting with "
    "'[face2ai]' describing who is in front of the camera: KNOWN <name> (best-effort match), UNKNOWN (someone "
    "not enrolled — never guess a name), MULTIPLE_FACES, NO_FACE, or camera off. Treat it as helpful context, "
    "not as identity verification; do not read the line back verbatim; use the tool presence_now or /presence "
    "when asked explicitly. The line may add a mood hint ('wirkt fröhlich' / 'looks happy' …): that is a best-effort "
    "guess from facial expression, never a fact — never state them as facts, do not psychoanalyse, mention them at "
    "most in passing and with reservation, never probe because of them, and never let them change how you treat someone. "
    "The tool presence_now may also list completed facial actions (a brief smile, raised brows …): those are movements "
    "of a face measured at ~0.6 s resolution — expression dynamics, never micro-expressions, never evidence of what "
    "someone feels, means or hides. The same reservation applies, and they are never a reason to interpret anyone."
)


def register(ctx: Any) -> None:
    global _ctx
    _ctx = ctx
    _load_settings(ctx)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_tool(
        name="presence_now",
        toolset="face2ai",
        schema=PRESENCE_TOOL_SCHEMA,
        handler=_tool_presence_now,
        description="Who is in front of the camera right now (Face2AI)",
        emoji="👤",
    )
    ctx.register_command("presence", _cmd_presence, description="Who is in front of the camera (Face2AI)")
    try:
        ctx.register_system_prompt_section("face2ai", SYSTEM_SECTION)
    except Exception as exc:  # older hosts
        logger.debug("system prompt section not registered: %s", exc)
    _start_consumer(ctx)
    logger.info("face2ai plugin registered (events_url=%s)", _setting("events_url", DEFAULT_EVENTS_URL))
