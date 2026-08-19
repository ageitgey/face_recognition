from __future__ import annotations

import pytest

from face2ai_agent.config import AgentConfig, ConfigError, resolve_providers


def cfg(**overrides) -> AgentConfig:
    base = dict(keys_present={"OPENROUTER_API_KEY": False, "GROQ_API_KEY": False, "OPENAI_API_KEY": False})
    base.update(overrides)
    return AgentConfig(**base)


def servers(**by_url):
    """Fake probe: {url_prefix: model_ids}; anything else is unreachable."""

    def probe(url: str):
        for prefix, models in by_url.items():
            if url.startswith(prefix):
                return models
        return None

    return probe


NOTHING = servers()
KOKORO = servers(**{"http://127.0.0.1:8880": ["kokoro"]})
SPEACHES = servers(**{"http://127.0.0.1:8000": ["Systran/faster-whisper-small", "Systran/faster-whisper-large-v3-turbo", "speaches-ai/Kokoro-82M-v1.0-ONNX", "speaches-ai/piper-de_DE-thorsten-medium"]})


def test_auto_prefers_cheapest_stack_when_all_keys_exist():
    resolved = resolve_providers(
        cfg(keys_present={"OPENROUTER_API_KEY": True, "GROQ_API_KEY": True, "OPENAI_API_KEY": True}),
        probe=KOKORO,
    )
    assert (resolved.llm, resolved.stt, resolved.tts) == ("openrouter", "groq", "local")
    assert resolved.llm_model.endswith(":free")
    assert resolved.llm_fallbacks  # free models are rate-limited, so fallbacks are configured
    assert resolved.tts_base_url.startswith("http://127.0.0.1:8880")
    assert resolved.tts_model == "kokoro"
    assert any("no de voices" in n for n in resolved.notes)  # default language is German


def test_speaches_serves_local_stt_and_german_piper_tts():
    resolved = resolve_providers(cfg(language="de", keys_present={"OPENROUTER_API_KEY": True, "GROQ_API_KEY": True, "OPENAI_API_KEY": True}), probe=SPEACHES)
    assert resolved.stt == "local" and resolved.stt_model == "Systran/faster-whisper-large-v3-turbo"
    assert resolved.stt_base_url == "http://127.0.0.1:8000/v1"
    assert resolved.tts == "local" and resolved.tts_model == "speaches-ai/piper-de_DE-thorsten-medium"
    assert resolved.tts_voice == "piper-de_DE-thorsten-medium"
    english = resolve_providers(cfg(language="en", keys_present={"OPENROUTER_API_KEY": True, "GROQ_API_KEY": False, "OPENAI_API_KEY": False}), probe=SPEACHES)
    assert english.tts_model == "speaches-ai/Kokoro-82M-v1.0-ONNX" and english.tts_voice == "af_heart"


def test_auto_falls_back_to_groq_then_openai():
    resolved = resolve_providers(cfg(keys_present={"OPENROUTER_API_KEY": False, "GROQ_API_KEY": True, "OPENAI_API_KEY": True}), probe=NOTHING)
    assert (resolved.llm, resolved.stt, resolved.tts) == ("groq", "groq", "openai")
    assert any("No local TTS server" in n for n in resolved.notes)
    resolved = resolve_providers(cfg(keys_present={"OPENROUTER_API_KEY": False, "GROQ_API_KEY": False, "OPENAI_API_KEY": True}), probe=NOTHING)
    assert (resolved.llm, resolved.stt, resolved.tts) == ("openai", "openai", "openai")


def test_missing_prerequisites_raise_with_guidance():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        resolve_providers(cfg(), probe=KOKORO)
    with pytest.raises(ConfigError, match="needs OPENROUTER_API_KEY"):
        resolve_providers(cfg(llm_provider="openrouter"), probe=KOKORO)
    with pytest.raises(ConfigError, match="No TTS available"):
        resolve_providers(cfg(keys_present={"OPENROUTER_API_KEY": True, "GROQ_API_KEY": True, "OPENAI_API_KEY": False}), probe=NOTHING)
    with pytest.raises(ConfigError, match="No STT available"):
        resolve_providers(cfg(keys_present={"OPENROUTER_API_KEY": True, "GROQ_API_KEY": False, "OPENAI_API_KEY": False}), probe=KOKORO)
    with pytest.raises(ConfigError, match="FACE2AI_AGENT_LLM must be one of"):
        resolve_providers(cfg(llm_provider="banana"), probe=KOKORO)


def test_local_servers_and_explicit_choices_win():
    resolved = resolve_providers(
        cfg(llm_provider="ollama", stt_base_url="http://127.0.0.1:9000/v1", tts_provider="kokoro", tts_voice="af_bella"),
        probe=NOTHING,  # explicit local choices are used even when nothing answers right now
    )
    assert resolved.llm == "ollama" and resolved.llm_base_url.startswith("http://127.0.0.1:11434")
    assert resolved.stt == "local" and resolved.stt_base_url == "http://127.0.0.1:9000/v1"
    assert resolved.stt_model == "Systran/faster-whisper-small"
    assert resolved.tts == "local" and resolved.tts_voice == "af_bella" and resolved.tts_model == "kokoro"


def test_need_audio_false_only_resolves_the_llm():
    resolved = resolve_providers(cfg(keys_present={"OPENROUTER_API_KEY": False, "GROQ_API_KEY": True, "OPENAI_API_KEY": False}), need_audio=False)
    assert resolved.llm == "groq" and resolved.stt == "none" and resolved.tts == "none"


def test_from_env_reads_flags_and_lists(monkeypatch):
    monkeypatch.setenv("FACE2AI_URL", "http://10.0.0.5:8765/")
    monkeypatch.setenv("FACE2AI_AGENT_LLM_FALLBACKS", "a/b:free, c/d")
    monkeypatch.setenv("FACE2AI_AGENT_GREET_UNKNOWN", "off")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    config = AgentConfig.from_env()
    assert config.face2ai_url == "http://10.0.0.5:8765"
    assert config.llm_fallback_models == ("a/b:free", "c/d")
    assert config.greet_unknown is False
    assert config.keys_present["OPENROUTER_API_KEY"] is True


def test_hermes_is_preferred_when_its_key_is_present_and_explicit_choice_needs_it():
    keys = {"HERMES_API_SERVER_KEY": True, "OPENROUTER_API_KEY": True, "GROQ_API_KEY": True, "OPENAI_API_KEY": False}
    resolved = resolve_providers(cfg(keys_present=keys), probe=SPEACHES)
    assert resolved.llm == "hermes" and resolved.llm_model == "hermes-agent"
    assert resolved.llm_base_url == "http://127.0.0.1:8642/v1"
    assert any("Hermes agent as brain" in n for n in resolved.notes)
    explicit = resolve_providers(cfg(llm_provider="openrouter", keys_present=keys), probe=SPEACHES)
    assert explicit.llm == "openrouter"
    with pytest.raises(ConfigError, match="HERMES_API_SERVER_KEY"):
        resolve_providers(cfg(llm_provider="hermes"), probe=SPEACHES)
