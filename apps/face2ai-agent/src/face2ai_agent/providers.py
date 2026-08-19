"""Build LiveKit STT / LLM / TTS plugin instances from the resolved provider choice.

Kept separate from config so config stays importable without the (heavy) LiveKit plugins.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import httpx

# LiveKit plugins register themselves on import and must be imported on the main thread,
# i.e. at module import time — never lazily inside a job. Optional ones degrade gracefully.
from livekit.plugins import openai, silero  # noqa: F401  (silero registers the VAD plugin)

try:
    from livekit.plugins import groq as _groq
except Exception:  # plugin not installed (extra "groq")
    _groq = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from livekit.plugins.turn_detector.multilingual import MultilingualModel as _MultilingualModel
except Exception:
    _MultilingualModel = None

from .config import AgentConfig, ResolvedProviders


def _key_kwargs(explicit: str, env_name: str | None = None) -> dict[str, str]:
    """The openai plugin treats ``api_key=None`` as "given but empty" and raises; only pass a key we have."""
    key = explicit or (os.environ.get(env_name, "") if env_name else "")
    return {"api_key": key} if key else {}


def build_llm(config: AgentConfig, resolved: ResolvedProviders) -> Any:
    if resolved.llm == "hermes":
        # Hermes gateway API server (OpenAI-compatible). The session key scopes Hermes' long-term
        # memory to this voice channel; the system prompt (presence) is layered on top of his core.
        import openai as openai_sdk

        key = config.llm_api_key or config.hermes_api_key or os.environ.get("HERMES_API_SERVER_KEY", "")
        client = openai_sdk.AsyncClient(
            base_url=resolved.llm_base_url,
            api_key=key,
            default_headers={"X-Hermes-Session-Key": config.hermes_session_key},
            timeout=120.0,
        )
        # A full Hermes turn (memory digest, tools, a 550B model) can take 10-30 s: give it room.
        return openai.LLM(model=resolved.llm_model, client=client, api_key=key, timeout=httpx.Timeout(90.0, connect=10.0))
    if resolved.llm == "openrouter":
        return openai.LLM.with_openrouter(
            model=resolved.llm_model,
            fallback_models=list(resolved.llm_fallbacks) or None,
            app_name="Face2AI",
            **_key_kwargs(config.llm_api_key, "OPENROUTER_API_KEY"),
        )
    if resolved.llm == "groq":
        return openai.LLM(
            model=resolved.llm_model,
            base_url="https://api.groq.com/openai/v1",
            **_key_kwargs(config.llm_api_key, "GROQ_API_KEY"),
        )
    if resolved.llm == "openai":
        return openai.LLM(model=resolved.llm_model, **_key_kwargs(config.llm_api_key, "OPENAI_API_KEY"))
    if resolved.llm == "ollama":
        return openai.LLM.with_ollama(model=resolved.llm_model, base_url=resolved.llm_base_url)
    return openai.LLM(model=resolved.llm_model, base_url=resolved.llm_base_url, api_key=config.llm_api_key or "not-needed")


def build_stt(config: AgentConfig, resolved: ResolvedProviders) -> Any:
    language = config.language
    if resolved.stt == "groq":
        if _groq is None:
            raise RuntimeError("Groq STT selected but livekit-plugins-groq is not installed (uv sync --extra groq)")
        return _groq.STT(model=resolved.stt_model, language=language, **_key_kwargs(config.stt_api_key, "GROQ_API_KEY"))
    if resolved.stt == "openai":
        return openai.STT(model=resolved.stt_model, language=language, **_key_kwargs(config.stt_api_key, "OPENAI_API_KEY"))
    # Local OpenAI-compatible Whisper server (e.g. speaches / faster-whisper-server): REST, not realtime.
    return openai.STT(
        model=resolved.stt_model,
        language=language,
        base_url=resolved.stt_base_url,
        api_key=config.stt_api_key or "not-needed",
        use_realtime=False,
    )


def build_tts(config: AgentConfig, resolved: ResolvedProviders) -> Any:
    if resolved.tts == "local":
        # Kokoro-FastAPI or speaches: plain OpenAI-compatible /v1/audio/speech (no SSE dialect)
        from .local_tts import LocalSpeechTTS

        return LocalSpeechTTS(
            base_url=resolved.tts_base_url,
            model=resolved.tts_model,
            voice=resolved.tts_voice,
            api_key=config.tts_api_key or "not-needed",
            response_format="wav",
        )
    return openai.TTS(model=resolved.tts_model, voice=resolved.tts_voice, **_key_kwargs(config.tts_api_key, "OPENAI_API_KEY"))


def build_turn_detection() -> Any:
    """Local end-of-turn model when available (no cloud), else VAD-based turn taking."""
    if _MultilingualModel is None:
        return "vad"
    try:
        return _MultilingualModel()
    except Exception:
        return "vad"
