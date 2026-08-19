"""The Face2AI voice agent: a LiveKit Agent whose knowledge of "who is here" comes from the
Face2AI presence stream. It never touches face matching; it consumes events and talks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from livekit.agents import Agent, AgentSession, APIConnectOptions, JobContext, JobProcess, RunContext, function_tool
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import silero

from .config import AgentConfig, ResolvedProviders, resolve_providers
from .policy import GreetingPolicy, Prompt, build_instructions
from .presence import PresenceClient, PresenceMemory, run_presence_loop
from .providers import build_llm, build_stt, build_tts, build_turn_detection

logger = logging.getLogger("face2ai_agent")


class Face2AIAgent(Agent):
    def __init__(self, config: AgentConfig, memory: PresenceMemory) -> None:
        self._config = config
        self._memory = memory
        self._situation_key = memory.situation_key()
        super().__init__(instructions=build_instructions(config, memory))

    async def refresh_instructions(self, *, force: bool = False) -> bool:
        """Rebuild the system prompt when the prompt-relevant situation changed. Returns True if updated.

        Every update inserts a config-update item into the chat context, so this is keyed on
        state/identity/engine/store-count — not on the elapsed-seconds text or heartbeats.
        """
        key = self._memory.situation_key()
        if not force and key == self._situation_key:
            return False
        self._situation_key = key
        await self.update_instructions(build_instructions(self._config, self._memory))
        return True

    @function_tool()
    async def who_is_here(self, context: RunContext) -> str:
        """Report who is currently in front of the camera according to Face2AI (best-effort recognition, not certainty),
        including the current hedged mood hint ("wirkt …" / "looks …" — a guess from facial expression, never a fact) if any."""
        return self._memory.describe(language=self._config.language)

    @function_tool()
    async def list_known_people(self, context: RunContext) -> str:
        """List the display names of people enrolled in Face2AI on this device."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._config.face2ai_url}/api/identities")
                response.raise_for_status()
                rows = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return f"I could not reach the Face2AI store: {exc}"
        if not rows:
            return "Nobody is enrolled yet."
        names = ", ".join(sorted(str(row.get("display_name", "?")) for row in rows))
        return f"{len(rows)} people are enrolled: {names}."

    @function_tool()
    async def face2ai_status(self, context: RunContext) -> str:
        """Check whether the Face2AI camera service and its recognition engine are up."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._config.face2ai_url}/api/status")
                response.raise_for_status()
                status = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return f"Face2AI is not reachable at {self._config.face2ai_url}: {exc}"
        engine = "available" if status.get("engine_available") else f"unavailable ({status.get('engine_reason')})"
        return f"Face2AI is up; recognition engine {engine}; {status.get('identity_count', 0)} people enrolled."


class Speaker:
    """Turns policy prompts into speech: a natural LLM reply, or the fixed sentence when the LLM fails.

    ``generate_reply`` returns immediately; the LLM/TTS outcome is only known when the returned
    SpeechHandle finishes, so the fallback is decided in a background task instead of blocking
    the presence loop.
    """

    def __init__(self, session: AgentSession, config: AgentConfig) -> None:
        self._session = session
        self._config = config
        self._tasks: set[asyncio.Task[None]] = set()

    def speak(self, prompt: Prompt) -> None:
        if self._config.greeting_style == "say":
            self._session.say(prompt.fallback_text)
            return
        task = asyncio.create_task(self._speak_with_fallback(prompt), name=f"speak-{prompt.kind}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _speak_with_fallback(self, prompt: Prompt) -> None:
        try:
            handle = self._session.generate_reply(instructions=prompt.instructions)
            await handle
            spoke = any(getattr(item, "type", "") == "message" for item in handle.chat_items)
            failed = handle.exception() is not None
        except Exception as exc:  # session not running / LLM refused before dispatch
            logger.warning("generate_reply failed before dispatch (%s)", exc)
            spoke, failed = False, True
        if failed or (not spoke and not handle.interrupted):
            logger.warning("LLM greeting failed for %s; speaking fixed sentence", prompt.kind)
            try:
                self._session.say(prompt.fallback_text)
            except Exception as exc:
                logger.error("fallback greeting failed too: %s", exc)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


def llm_connect_options(resolved: ResolvedProviders) -> APIConnectOptions:
    """LiveKit applies conn_options.timeout per request (it overrides the client's own timeout).
    A full Hermes turn can take 10-30 s; bare models answer within seconds."""
    timeout = 90.0 if resolved.llm == "hermes" else 30.0
    return APIConnectOptions(max_retry=2, retry_interval=2.0, timeout=timeout)


def build_session(config: AgentConfig, resolved: ResolvedProviders, *, vad: Any = None) -> AgentSession:
    return AgentSession(
        stt=build_stt(config, resolved),
        llm=build_llm(config, resolved),
        tts=build_tts(config, resolved),
        vad=vad or silero.VAD.load(),
        turn_detection=build_turn_detection(),
        conn_options=SessionConnectOptions(llm_conn_options=llm_connect_options(resolved)),
    )


async def entrypoint(ctx: JobContext) -> None:
    config = AgentConfig.from_env()
    # Provider probes are blocking HTTP calls: keep them off the job's event loop.
    resolved = await asyncio.to_thread(resolve_providers, config)
    logger.info("providers: llm=%s(%s) stt=%s(%s) tts=%s(%s)", resolved.llm, resolved.llm_model, resolved.stt, resolved.stt_model, resolved.tts, resolved.tts_model)
    for note in resolved.notes:
        logger.info("note: %s", note)

    memory = PresenceMemory()
    policy = GreetingPolicy(config)
    agent = Face2AIAgent(config, memory)
    session = build_session(config, resolved, vad=ctx.proc.userdata.get("vad"))
    speaker = Speaker(session, config)

    await ctx.connect()
    await session.start(agent=agent, room=ctx.room)

    presence_client = PresenceClient(config.face2ai_url)

    async def on_event(kind: str, payload: Any, replayed: bool) -> None:
        try:
            if kind == "hello":
                policy.cooldown_seconds = memory.greeting_cooldown_seconds
                logger.info("presence stream connected; %s", memory.describe())
            elif kind == "lost":
                logger.warning("presence stream lost: %s", payload)
            await agent.refresh_instructions()
            if replayed:
                logger.info("replayed %s event applied silently", kind)
                return
            prompt = None
            if kind == "presence":
                prompt = policy.on_transition(payload)
            elif kind == "store":
                prompt = policy.on_store_change(payload)
            if prompt is not None:
                logger.info("speaking: %s", prompt.kind)
                speaker.speak(prompt)
        except Exception:
            logger.exception("presence event handling failed (%s)", kind)

    presence_task = asyncio.create_task(run_presence_loop(presence_client, memory, on_event), name="face2ai-presence")

    def _presence_done(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        if task.exception() is not None:
            logger.error("presence loop died", exc_info=task.exception())

    presence_task.add_done_callback(_presence_done)

    async def shutdown() -> None:
        presence_client.stop()
        presence_task.cancel()
        await asyncio.gather(presence_task, return_exceptions=True)
        await speaker.aclose()

    ctx.add_shutdown_callback(shutdown)
