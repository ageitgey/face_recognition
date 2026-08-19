"""Entry point.

    face2ai-agent check            # verify config, Face2AI, presence stream, LLM (and Kokoro TTS)
    face2ai-agent smoke [text]     # text-only conversation turn against the live Face2AI presence
    face2ai-agent console          # talk to the agent via this Mac's microphone/speaker (LiveKit console mode)
    face2ai-agent download-files   # fetch local VAD / turn-detector models once
    face2ai-agent dev|start        # LiveKit worker modes (need a LiveKit server)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time

from dotenv import load_dotenv

from .config import AgentConfig, ConfigError, resolve_providers


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def check(*, paid_probes: bool = False) -> int:
    import httpx

    from .presence import PresenceClient
    from .providers import build_llm, build_tts

    config = AgentConfig.from_env()
    ok = True
    print(f"face2ai url      : {config.face2ai_url}")
    print(f"language         : {config.language}")
    print(f"keys present     : {json.dumps(config.keys_present)}")
    try:
        resolved = resolve_providers(config)
    except ConfigError as exc:
        print(f"CONFIG ERROR     : {exc}")
        return 2
    print(f"llm              : {resolved.llm} model={resolved.llm_model} fallbacks={list(resolved.llm_fallbacks)} {resolved.llm_base_url}")
    print(f"stt              : {resolved.stt} model={resolved.stt_model} {resolved.stt_base_url}")
    print(f"tts              : {resolved.tts} model={resolved.tts_model} voice={resolved.tts_voice} {resolved.tts_base_url}")
    for note in resolved.notes:
        print(f"note             : {note}")

    # Face2AI backend
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            status = (await client.get(f"{config.face2ai_url}/api/status")).json()
        print(f"face2ai status   : engine_available={status.get('engine_available')} identities={status.get('identity_count')} agent_connected={status.get('agent_connected')}")
    except Exception as exc:
        print(f"face2ai status   : FAIL ({exc}) — start it with `uv run --project apps/face2ai face2ai`")
        ok = False

    # Presence stream
    if ok:
        presence = PresenceClient(config.face2ai_url, role="probe")  # a probe must not look like the agent
        try:
            frame = await asyncio.wait_for(presence.frames().__anext__(), timeout=5.0)
            presence.stop()
            if frame.event != "hello":
                raise RuntimeError(f"expected hello frame, got {frame.event}: {frame.data}")
            print(f"presence stream  : {frame.event} presence={frame.data.get('presence', {}).get('state')} cooldown={frame.data.get('greeting_cooldown_seconds')}s")
        except Exception as exc:
            print(f"presence stream  : FAIL ({exc})")
            ok = False

    # LLM round trip
    try:
        from livekit.agents import llm as agent_llm

        llm = build_llm(config, resolved)
        chat_ctx = agent_llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="Reply with exactly the single word OK.")
        started = time.perf_counter()
        text = ""
        from .agent import llm_connect_options

        async with llm.chat(chat_ctx=chat_ctx, conn_options=llm_connect_options(resolved)) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
        await llm.aclose()
        print(f"llm round trip   : {text.strip()[:40]!r} in {time.perf_counter() - started:.2f}s")
        if not text.strip():
            ok = False
    except Exception as exc:
        print(f"llm round trip   : FAIL ({exc})")
        ok = False

    # TTS (free local server always; cloud only with --paid), then STT on the synthesized audio.
    probe_text = "Hallo, hier spricht Face2AI. Erkennst du mich?" if config.language.startswith("de") else "Hello, this is Face2AI. Can you hear me?"
    audio_frames = []
    if resolved.tts == "local" or paid_probes:
        try:
            tts = build_tts(config, resolved)
            started = time.perf_counter()
            async with tts.synthesize(probe_text) as stream:
                async for event in stream:
                    audio_frames.append(event.frame)
            await tts.aclose()
            seconds = sum(f.samples_per_channel for f in audio_frames) / (audio_frames[0].sample_rate if audio_frames else 1)
            print(f"tts round trip   : {len(audio_frames)} frames = {seconds:.1f}s audio in {time.perf_counter() - started:.2f}s")
            if not audio_frames:
                ok = False
        except Exception as exc:
            print(f"tts round trip   : FAIL ({exc})")
            ok = False
    else:
        print("tts round trip   : skipped (cloud TTS; run `check --paid` to probe)")
    if audio_frames and (resolved.stt == "local" or paid_probes):
        try:
            from .providers import build_stt

            stt = build_stt(config, resolved)
            started = time.perf_counter()
            event = await stt.recognize(audio_frames, language=config.language)
            await stt.aclose()
            heard = event.alternatives[0].text if event.alternatives else ""
            print(f"stt round trip   : {heard!r} in {time.perf_counter() - started:.2f}s")
            if not heard.strip():
                ok = False
        except Exception as exc:
            print(f"stt round trip   : FAIL ({exc})")
            ok = False
    else:
        print("stt round trip   : skipped (needs local TTS audio or --paid)")
    print("RESULT           :", "OK" if ok else "PROBLEMS")
    return 0 if ok else 1


async def smoke(user_text: str) -> int:
    """Text-only turn: connect presence, build the real agent + LLM, ask one question, print the reply."""
    from livekit.agents import AgentSession

    from .agent import Face2AIAgent
    from .policy import GreetingPolicy, build_instructions
    from .presence import PresenceClient, PresenceMemory, run_presence_loop
    from .providers import build_llm

    config = AgentConfig.from_env()
    resolved = resolve_providers(config, need_audio=False)
    memory = PresenceMemory()
    client = PresenceClient(config.face2ai_url, role="probe")  # not "agent": must not take greeting ownership
    policy = GreetingPolicy(config)
    prompts = []

    async def on_event(kind, payload, replayed):
        if replayed:
            return
        if kind == "presence":
            p = policy.on_transition(payload)
        elif kind == "store":
            p = policy.on_store_change(payload)
        else:
            p = None
        if p:
            prompts.append(p)

    task = asyncio.create_task(run_presence_loop(client, memory, on_event))
    await asyncio.sleep(1.5)
    client.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    print("presence         :", memory.describe())
    print("greetings due    :", [p.kind for p in prompts] or "none in the last 1.5 s")

    from .agent import llm_connect_options
    from livekit.agents.voice.agent_session import SessionConnectOptions

    agent = Face2AIAgent(config, memory)
    session = AgentSession(llm=build_llm(config, resolved), conn_options=SessionConnectOptions(llm_conn_options=llm_connect_options(resolved)))
    await session.start(agent)
    print("instructions     :", build_instructions(config, memory).splitlines()[-1])
    result = await session.run(user_input=user_text)
    for event in result.events:
        item = getattr(event, "item", None)
        if item is None:
            continue
        kind = getattr(item, "type", type(item).__name__)
        if kind == "function_call":
            print(f"tool call        : {item.name}({item.arguments})")
        elif kind == "function_call_output":
            print(f"tool result      : {str(item.output)[:200]}")
        elif kind == "message":
            print(f"agent reply      : {item.text_content}")
    await session.aclose()
    return 0


def main() -> None:
    load_dotenv()
    _configure_logging()
    argv = sys.argv[1:]
    if argv[:1] == ["check"]:
        raise SystemExit(asyncio.run(check(paid_probes="--paid" in argv)))
    if argv[:1] == ["smoke"]:
        text = " ".join(argv[1:]) or "Wer ist gerade vor der Kamera?"
        raise SystemExit(asyncio.run(smoke(text)))
    from livekit.agents import WorkerOptions, cli

    from .agent import entrypoint, prewarm

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


if __name__ == "__main__":
    main()
