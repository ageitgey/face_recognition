from __future__ import annotations

import pytest

from face2ai_agent.agent import Face2AIAgent
from face2ai_agent.config import AgentConfig, ResolvedProviders
from face2ai_agent.presence import PresenceMemory
from face2ai_agent.providers import build_llm, build_stt, build_tts


async def test_refresh_instructions_is_awaited_and_only_updates_on_situation_change():
    memory = PresenceMemory()
    agent = Face2AIAgent(AgentConfig(language="de"), memory)
    assert "not connected" in agent.instructions
    memory.apply_hello({"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ada"}, "engine_available": True})
    assert await agent.refresh_instructions() is True
    assert "Ada is in front of the camera" in agent.instructions
    assert await agent.refresh_instructions() is False  # same situation: no chat-context churn
    memory.apply_heartbeat({"presence": {"state": "NO_FACE"}})
    assert await agent.refresh_instructions() is True
    assert "nobody is in front" in agent.instructions


def _resolved(llm: str, stt: str, tts: str) -> ResolvedProviders:
    return ResolvedProviders(
        llm=llm, llm_model="m", llm_fallbacks=(), llm_base_url="",
        stt=stt, stt_model="whisper-1", stt_base_url="",
        tts=tts, tts_model="gpt-4o-mini-tts", tts_voice="ash", tts_base_url="",
    )


def test_openai_tier_builds_with_key_from_environment_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = AgentConfig(keys_present={"OPENAI_API_KEY": True})
    resolved = _resolved("openai", "openai", "openai")
    assert build_llm(config, resolved) is not None
    assert build_stt(config, resolved) is not None
    assert build_tts(config, resolved) is not None


def test_openrouter_and_groq_read_their_env_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    config = AgentConfig()
    assert build_llm(config, _resolved("openrouter", "groq", "local")) is not None
    assert build_llm(config, _resolved("groq", "groq", "local")) is not None
    assert build_stt(config, _resolved("groq", "groq", "local")) is not None


def test_missing_key_fails_loudly_at_build_time(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Exception):
        build_llm(AgentConfig(), _resolved("openai", "openai", "openai"))
