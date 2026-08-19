"""Configuration for the Face2AI voice agent.

Everything is environment-driven so providers can be swapped without code changes.
The default stack is the cheapest one: LLM via OpenRouter free models, STT via a local
OpenAI-compatible Whisper server (speaches) or Groq Whisper, TTS via a local Kokoro/Piper server
(speaches, Kokoro-FastAPI) or OpenAI. `auto` prefers local/free and explains itself in `check`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_OPENROUTER_FALLBACKS = ("google/gemma-4-31b-it:free", "nvidia/nemotron-3-super-120b-a12b:free")
DEFAULT_LOCAL_TTS_URLS = ("http://127.0.0.1:8880/v1", "http://127.0.0.1:8000/v1")  # kokoro-fastapi, speaches
DEFAULT_LOCAL_STT_URLS = ("http://127.0.0.1:8000/v1",)  # speaches / faster-whisper-server

LLM_PROVIDERS = ("auto", "hermes", "openrouter", "groq", "openai", "ollama", "custom")
DEFAULT_HERMES_URL = "http://127.0.0.1:8642/v1"  # Hermes gateway API server (SSH-tunnelled from the VPS)
DEFAULT_HERMES_SESSION_KEY = "face2ai-voice"
STT_PROVIDERS = ("auto", "groq", "openai", "local")
TTS_PROVIDERS = ("auto", "local", "kokoro", "openai")  # kokoro = alias for local


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _flag(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AgentConfig:
    face2ai_url: str = "http://127.0.0.1:8765"
    language: str = "de"
    agent_name: str = "Face2AI"
    persona: str = ""

    llm_provider: str = "auto"
    llm_model: str = ""
    llm_fallback_models: tuple[str, ...] = ()
    llm_base_url: str = ""
    llm_api_key: str = ""

    stt_provider: str = "auto"
    stt_model: str = ""
    stt_base_url: str = ""
    stt_api_key: str = ""

    tts_provider: str = "auto"
    tts_model: str = ""
    tts_voice: str = ""
    tts_base_url: str = ""
    tts_api_key: str = ""

    greet_known: bool = True
    greet_unknown: bool = True
    announce_multiple: bool = True
    unknown_greeting_cooldown_seconds: int = 120
    regreet_after_seconds: int = 90  # same known person again only after being away this long
    greeting_style: str = "llm"  # llm = natural reply via LLM, say = fixed sentence via TTS

    hermes_url: str = DEFAULT_HERMES_URL
    hermes_api_key: str = ""
    hermes_session_key: str = DEFAULT_HERMES_SESSION_KEY

    keys_present: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        fallbacks = tuple(m.strip() for m in _env("FACE2AI_AGENT_LLM_FALLBACKS").split(",") if m.strip())
        return cls(
            face2ai_url=_env("FACE2AI_URL", "http://127.0.0.1:8765").rstrip("/"),
            language=_env("FACE2AI_AGENT_LANGUAGE", "de"),
            agent_name=_env("FACE2AI_AGENT_NAME", "Face2AI"),
            persona=_env("FACE2AI_AGENT_PERSONA"),
            llm_provider=_env("FACE2AI_AGENT_LLM", "auto").lower(),
            llm_model=_env("FACE2AI_AGENT_LLM_MODEL"),
            llm_fallback_models=fallbacks,
            llm_base_url=_env("FACE2AI_AGENT_LLM_BASE_URL"),
            llm_api_key=_env("FACE2AI_AGENT_LLM_API_KEY"),
            stt_provider=_env("FACE2AI_AGENT_STT", "auto").lower(),
            stt_model=_env("FACE2AI_AGENT_STT_MODEL"),
            stt_base_url=_env("FACE2AI_AGENT_STT_BASE_URL"),
            stt_api_key=_env("FACE2AI_AGENT_STT_API_KEY"),
            tts_provider=_env("FACE2AI_AGENT_TTS", "auto").lower(),
            tts_model=_env("FACE2AI_AGENT_TTS_MODEL"),
            tts_voice=_env("FACE2AI_AGENT_TTS_VOICE"),
            tts_base_url=_env("FACE2AI_AGENT_TTS_BASE_URL"),
            tts_api_key=_env("FACE2AI_AGENT_TTS_API_KEY"),
            greet_known=_flag("FACE2AI_AGENT_GREET_KNOWN", True),
            greet_unknown=_flag("FACE2AI_AGENT_GREET_UNKNOWN", True),
            announce_multiple=_flag("FACE2AI_AGENT_ANNOUNCE_MULTIPLE", True),
            unknown_greeting_cooldown_seconds=int(_env("FACE2AI_AGENT_UNKNOWN_COOLDOWN_SECONDS", "120")),
            regreet_after_seconds=int(_env("FACE2AI_AGENT_REGREET_AFTER_SECONDS", "90")),
            greeting_style=_env("FACE2AI_AGENT_GREETING_STYLE", "llm").lower(),
            hermes_url=_env("HERMES_API_SERVER_URL", DEFAULT_HERMES_URL).rstrip("/"),
            hermes_api_key=_env("HERMES_API_SERVER_KEY"),
            hermes_session_key=_env("HERMES_SESSION_KEY", DEFAULT_HERMES_SESSION_KEY),
            keys_present={
                "HERMES_API_SERVER_KEY": bool(_env("HERMES_API_SERVER_KEY")),
                "OPENROUTER_API_KEY": bool(_env("OPENROUTER_API_KEY")),
                "GROQ_API_KEY": bool(_env("GROQ_API_KEY")),
                "OPENAI_API_KEY": bool(_env("OPENAI_API_KEY")),
            },
        )


@dataclass(frozen=True, slots=True)
class ResolvedProviders:
    """Concrete provider choice after `auto` resolution; printed by `check`, used by the factory."""

    llm: str
    llm_model: str
    llm_fallbacks: tuple[str, ...]
    llm_base_url: str
    stt: str
    stt_model: str
    stt_base_url: str
    tts: str
    tts_model: str
    tts_voice: str
    tts_base_url: str
    notes: tuple[str, ...] = ()


class ConfigError(RuntimeError):
    pass


def resolve_providers(
    config: AgentConfig,
    *,
    need_audio: bool = True,
    probe: "Callable[[str], list[str] | None] | None" = None,
) -> ResolvedProviders:
    """Turn `auto` into concrete providers using which keys/servers are available.

    Order of preference is cost: OpenRouter (free models) > Groq > OpenAI for the LLM;
    local server > Groq > OpenAI for STT; local server (Kokoro-FastAPI / speaches) > OpenAI for TTS.
    Anything explicit in the config wins; missing prerequisites raise ConfigError with the fix in
    the message. ``probe(base_url)`` returns the model ids of a local OpenAI-compatible server or
    None when unreachable (injected in tests).
    """
    keys = config.keys_present
    notes: list[str] = []

    # ---- LLM
    llm = config.llm_provider
    if llm not in LLM_PROVIDERS:
        raise ConfigError(f"FACE2AI_AGENT_LLM must be one of {LLM_PROVIDERS}, got {llm!r}")
    if llm == "auto":
        if config.llm_base_url:
            llm = "custom"
        elif keys.get("HERMES_API_SERVER_KEY"):
            llm = "hermes"  # your own agent (memory, tools, persona) beats a bare model when it is reachable
        elif keys.get("OPENROUTER_API_KEY"):
            llm = "openrouter"
        elif keys.get("GROQ_API_KEY"):
            llm = "groq"
        elif keys.get("OPENAI_API_KEY"):
            llm = "openai"
        else:
            raise ConfigError(
                "No LLM available: set OPENROUTER_API_KEY (free models), GROQ_API_KEY, OPENAI_API_KEY, "
                "or FACE2AI_AGENT_LLM=ollama / custom with FACE2AI_AGENT_LLM_BASE_URL"
            )
    if llm == "hermes" and not (keys.get("HERMES_API_SERVER_KEY") or config.llm_api_key):
        raise ConfigError("FACE2AI_AGENT_LLM=hermes needs HERMES_API_SERVER_KEY (the gateway's API_SERVER_KEY) and the :8642 tunnel")
    if llm == "openrouter" and not (keys.get("OPENROUTER_API_KEY") or config.llm_api_key):
        raise ConfigError("FACE2AI_AGENT_LLM=openrouter needs OPENROUTER_API_KEY")
    if llm == "groq" and not (keys.get("GROQ_API_KEY") or config.llm_api_key):
        raise ConfigError("FACE2AI_AGENT_LLM=groq needs GROQ_API_KEY")
    if llm == "openai" and not (keys.get("OPENAI_API_KEY") or config.llm_api_key):
        raise ConfigError("FACE2AI_AGENT_LLM=openai needs OPENAI_API_KEY")
    if llm == "custom" and not config.llm_base_url:
        raise ConfigError("FACE2AI_AGENT_LLM=custom needs FACE2AI_AGENT_LLM_BASE_URL")
    llm_model = config.llm_model or {
        "hermes": "hermes-agent",
        "openrouter": DEFAULT_OPENROUTER_MODEL,
        "groq": "openai/gpt-oss-20b",
        "openai": "gpt-4.1-mini",
        "ollama": "llama3.2",
        "custom": "",
    }[llm]
    llm_fallbacks = config.llm_fallback_models or (DEFAULT_OPENROUTER_FALLBACKS if llm == "openrouter" else ())
    llm_base_url = config.llm_base_url or {"ollama": "http://127.0.0.1:11434/v1", "hermes": config.hermes_url}.get(llm, "")
    if llm == "openrouter" and llm_model.endswith(":free"):
        notes.append("OpenRouter free model: rate-limited; fallbacks configured")
    if llm == "hermes":
        notes.append(f"Hermes agent as brain via {llm_base_url} (session key {config.hermes_session_key!r}); slower than a bare model, but with memory + tools")

    if not need_audio:
        return ResolvedProviders(
            llm=llm, llm_model=llm_model, llm_fallbacks=llm_fallbacks, llm_base_url=llm_base_url,
            stt="none", stt_model="", stt_base_url="", tts="none", tts_model="", tts_voice="", tts_base_url="",
            notes=tuple(notes),
        )

    # ---- STT
    stt = config.stt_provider
    if stt not in STT_PROVIDERS:
        raise ConfigError(f"FACE2AI_AGENT_STT must be one of {STT_PROVIDERS}, got {stt!r}")
    probe = probe or probe_openai_compatible
    local_stt_url, local_stt_models = "", None
    if stt in ("auto", "local"):
        for url in ([config.stt_base_url] if config.stt_base_url else DEFAULT_LOCAL_STT_URLS):
            models = probe(url)
            if models is not None:
                local_stt_url, local_stt_models = url, models
                break
    if stt == "auto":
        if config.stt_base_url or local_stt_url:
            stt = "local"
        elif keys.get("GROQ_API_KEY"):
            stt = "groq"
        elif keys.get("OPENAI_API_KEY"):
            stt = "openai"
        else:
            raise ConfigError(
                "No STT available: run a local OpenAI-compatible Whisper server (speaches) at "
                f"{DEFAULT_LOCAL_STT_URLS[0]} or set FACE2AI_AGENT_STT_BASE_URL, or set GROQ_API_KEY / OPENAI_API_KEY"
            )
    if stt == "groq" and not (keys.get("GROQ_API_KEY") or config.stt_api_key):
        raise ConfigError("FACE2AI_AGENT_STT=groq needs GROQ_API_KEY")
    if stt == "openai" and not (keys.get("OPENAI_API_KEY") or config.stt_api_key):
        raise ConfigError("FACE2AI_AGENT_STT=openai needs OPENAI_API_KEY")
    if stt == "local":
        stt_base_url = config.stt_base_url or local_stt_url or DEFAULT_LOCAL_STT_URLS[0]
        stt_model = config.stt_model or pick_local_stt_model(local_stt_models or [])
    else:
        stt_base_url = ""
        stt_model = config.stt_model or {"groq": "whisper-large-v3-turbo", "openai": "gpt-4o-mini-transcribe"}[stt]

    # ---- TTS
    tts = config.tts_provider
    if tts not in TTS_PROVIDERS:
        raise ConfigError(f"FACE2AI_AGENT_TTS must be one of {TTS_PROVIDERS}, got {tts!r}")
    if tts == "kokoro":
        tts = "local"
    local_tts_url, local_tts_models = "", None
    if tts in ("auto", "local"):
        candidates = [config.tts_base_url] if config.tts_base_url else list(DEFAULT_LOCAL_TTS_URLS)
        for url in candidates:
            models = probe(url)
            if models is not None:
                local_tts_url, local_tts_models = url, models
                break
    if tts == "auto":
        if config.tts_base_url or local_tts_url:
            tts = "local"
        elif keys.get("OPENAI_API_KEY"):
            tts = "openai"
            notes.append(f"No local TTS server at {' / '.join(DEFAULT_LOCAL_TTS_URLS)}; using OpenAI TTS (paid)")
        else:
            raise ConfigError(
                "No TTS available: start Kokoro-FastAPI (:8880) or speaches (:8000), set FACE2AI_AGENT_TTS_BASE_URL, "
                "or set OPENAI_API_KEY"
            )
    if tts == "openai" and not (keys.get("OPENAI_API_KEY") or config.tts_api_key):
        raise ConfigError("FACE2AI_AGENT_TTS=openai needs OPENAI_API_KEY")
    if tts == "local":
        tts_base_url = config.tts_base_url or local_tts_url or DEFAULT_LOCAL_TTS_URLS[0]
        picked_model, picked_voice, note = pick_local_tts(local_tts_models or [], config.language)
        tts_model = config.tts_model or picked_model
        tts_voice = config.tts_voice or picked_voice
        if note:
            notes.append(note)
    else:
        tts_base_url = ""
        tts_model = config.tts_model or "gpt-4o-mini-tts"
        tts_voice = config.tts_voice or "ash"

    return ResolvedProviders(
        llm=llm, llm_model=llm_model, llm_fallbacks=llm_fallbacks, llm_base_url=llm_base_url,
        stt=stt, stt_model=stt_model, stt_base_url=stt_base_url,
        tts=tts, tts_model=tts_model, tts_voice=tts_voice, tts_base_url=tts_base_url,
        notes=tuple(notes),
    )


def probe_openai_compatible(base_url: str, timeout: float = 1.5) -> list[str] | None:
    """Model ids served by an OpenAI-compatible server at ``base_url`` (``GET /models``).

    None unless the server answers 2xx with a JSON model list — anything else on that port
    (another app, an error page) must not be mistaken for a speech server.
    """
    try:
        import httpx

        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        if not 200 <= response.status_code < 300:
            return None
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return None
        return [str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id")]
    except Exception:
        return None


def pick_local_stt_model(models: list[str]) -> str:
    """Prefer a small/turbo faster-whisper model that is already downloaded on the local server."""
    whisper = [m for m in models if "whisper" in m.lower()]
    for needle in ("turbo", "small", "base", "medium", "large"):
        for m in whisper:
            if needle in m.lower():
                return m
    return whisper[0] if whisper else "Systran/faster-whisper-small"


def pick_local_tts(models: list[str], language: str) -> tuple[str, str, str]:
    """(model, voice, note) for a local OpenAI-compatible TTS server.

    Kokoro has no German voices, so for `de` a Piper German voice is preferred when the server
    (e.g. speaches) offers one; otherwise Kokoro is used with a note. Kokoro-FastAPI reports
    its model simply as ``kokoro``; speaches reports Hugging Face ids.
    """
    lang = (language or "en").lower()[:2]
    lowered = [(m, m.lower()) for m in models]
    if lang == "de":
        for m, low in lowered:
            if "piper" in low and ("de_de" in low or "/de-" in low or "-de-" in low or "de_" in low):
                voice = m.split("/")[-1]
                return m, voice, f"German Piper voice found: {m}"
    for m, low in lowered:
        if "kokoro" in low:
            voice = "af_heart"
            note = "" if lang == "en" else f"Kokoro has no {lang} voices; German output will sound accented — set FACE2AI_AGENT_TTS_MODEL/VOICE to a Piper de voice or use OpenAI TTS"
            return m, voice, note
    return "kokoro", "af_heart", ("" if lang == "en" else "Local TTS model unknown; assuming Kokoro (no German voices)")
