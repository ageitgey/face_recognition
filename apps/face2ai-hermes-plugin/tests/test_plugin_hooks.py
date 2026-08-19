"""The plugin module without Hermes: register() against a fake ctx, hook/tool/command outputs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import face2ai as plugin  # noqa: E402
from face2ai.presence import SseFrame  # noqa: E402


class FakeState:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        json.dumps(value)  # must be JSON-serialisable
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)


class FakeCtx:
    def __init__(self, settings=None):
        self.settings = settings or {}
        self.hooks, self.tools, self.commands, self.sections, self.tasks, self.emitted, self.injected = {}, {}, {}, {}, [], [], []
        self.state = FakeState()

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, cb):
        self.hooks[name] = cb

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = (schema, handler)

    def register_command(self, name, handler, description=""):
        self.commands[name] = handler

    def register_system_prompt_section(self, sid, content, **kw):
        self.sections[sid] = content

    def spawn_task(self, coro, name=None):
        coro.close()  # do not actually connect anywhere in tests
        self.tasks.append(name)

    def emit(self, event, payload):
        self.emitted.append((event, payload))

    def inject_message(self, content, role="user", *, session_key=None):
        self.injected.append((content, session_key))
        return True


def setup_function(_):
    plugin._store = plugin.PresenceStore()
    plugin._settings.clear()
    plugin._last_session_key.clear()
    plugin._last_announce.clear()


def test_register_wires_everything():
    ctx = FakeCtx({"events_url": "http://127.0.0.1:8765/"})
    plugin.register(ctx)
    assert set(ctx.hooks) == {"pre_llm_call", "pre_gateway_dispatch"}
    assert "presence_now" in ctx.tools and "presence" in ctx.commands
    assert "face2ai" in ctx.sections and "never guess a name" in ctx.sections["face2ai"]
    assert ctx.tasks == ["face2ai-presence"]
    assert plugin._settings["events_url"] == "http://127.0.0.1:8765"


def test_pre_llm_call_injects_context_only_when_fresh_and_allowed():
    ctx = FakeCtx({"platforms": ["api_server"]})
    plugin.register(ctx)
    now = datetime.now(timezone.utc)
    assert ctx.hooks["pre_llm_call"](platform="api_server") is None  # nothing received yet
    plugin._handle_frame(SseFrame("hello", {"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a"}, "last_sequence": 0}))
    result = ctx.hooks["pre_llm_call"](platform="api_server", session_id="s", user_message="hi")
    assert result["context"].startswith("[face2ai] Ben steht vor der Kamera")
    assert ctx.hooks["pre_llm_call"](platform="telegram") is None  # filtered by platforms
    assert ctx.state.get("snapshot")["presence"]["display_name"] == "Ben"  # persisted for the API side
    plugin._settings["inject_context"] = False
    assert ctx.hooks["pre_llm_call"](platform="api_server") is None


def test_tool_and_command_report_presence():
    ctx = FakeCtx()
    plugin.register(ctx)
    plugin._handle_frame(SseFrame("hello", {"presence": {"state": "UNKNOWN", "faces": 1}, "last_sequence": 0}))
    plugin._handle_frame(SseFrame("presence", {"sequence": 1, "at": datetime.now(timezone.utc).isoformat(), "from_state": "UNKNOWN", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ben", "faces": 1}))
    tool = json.loads(ctx.tools["presence_now"][1]({}))
    assert tool["presence"]["display_name"] == "Ben" and "Ben steht vor der Kamera" in tool["summary"]
    text = ctx.commands["presence"]("")
    assert "Ben steht vor der Kamera" in text and "UNKNOWN → KNOWN (Ben)" in text
    assert ctx.emitted and ctx.emitted[-1][0] == "presence_changed"


def test_announcements_need_setting_and_session_key_and_respect_cooldown():
    ctx = FakeCtx({"announce_arrivals": True, "announce_cooldown_seconds": 120})
    plugin.register(ctx)
    plugin._handle_frame(SseFrame("hello", {"presence": {"state": "NO_FACE"}, "last_sequence": 0}))
    arrival = SseFrame("presence", {"sequence": 1, "at": datetime.now(timezone.utc).isoformat(), "from_state": "NO_FACE", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ben", "faces": 1})
    plugin._handle_frame(arrival)
    assert ctx.injected == []  # no gateway session known yet
    plugin._last_session_key["desktop"] = "agent:main:desktop:dm:1"
    plugin._handle_frame(SseFrame("presence", {"sequence": 2, "at": datetime.now(timezone.utc).isoformat(), "from_state": "KNOWN", "to_state": "NO_FACE", "faces": 0}))
    plugin._handle_frame(SseFrame("presence", {"sequence": 3, "at": datetime.now(timezone.utc).isoformat(), "from_state": "NO_FACE", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ben", "faces": 1}))
    assert len(ctx.injected) == 1 and "Ben" in ctx.injected[0][0] and ctx.injected[0][1] == "agent:main:desktop:dm:1"
    plugin._handle_frame(SseFrame("presence", {"sequence": 4, "at": datetime.now(timezone.utc).isoformat(), "from_state": "KNOWN", "to_state": "NO_FACE", "faces": 0}))
    plugin._handle_frame(SseFrame("presence", {"sequence": 5, "at": datetime.now(timezone.utc).isoformat(), "from_state": "NO_FACE", "to_state": "KNOWN", "identity_id": "a", "display_name": "Ben", "faces": 1}))
    assert len(ctx.injected) == 1  # cooldown


def test_mood_frames_are_persisted_and_the_system_section_hedges_them():
    ctx = FakeCtx()
    plugin.register(ctx)
    plugin._handle_frame(SseFrame("hello", {"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a"}, "last_sequence": 0}))
    plugin._handle_frame(SseFrame("mood", {"sequence": 1, "at": datetime.now(timezone.utc).isoformat(), "identity_id": "a", "display_name": "Ben", "from_mood": None, "to_mood": "Happiness", "valence": 0.6, "arousal": 0.1}))
    assert ctx.state.get("snapshot")["presence"]["mood"] == "Happiness"  # dashboard API sees it
    assert not [e for e in ctx.emitted if e[0] == "presence_changed"] and ctx.injected == []  # a mood is never a transition or an announcement
    assert "Ben wirkt fröhlich" in ctx.hooks["pre_llm_call"](platform="desktop")["context"]
    assert "'wirkt" in ctx.sections["face2ai"] and "never state them as facts" in ctx.sections["face2ai"] and "psychoanaly" in ctx.sections["face2ai"]
    assert "never a fact" in ctx.tools["presence_now"][0]["description"]


def test_action_frames_are_persisted_but_never_a_transition_or_announcement_or_context():
    ctx = FakeCtx({"announce_arrivals": True})
    plugin.register(ctx)
    plugin._last_session_key["desktop"] = "agent:main:desktop:dm:1"
    plugin._handle_frame(SseFrame("hello", {"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a"}, "last_sequence": 0}))
    now = datetime.now(timezone.utc).isoformat()
    plugin._handle_frame(SseFrame("action", {"sequence": 1, "at": now, "identity_id": "a", "display_name": "Ben", "action": "smile", "onset_at": now, "apex_at": now, "offset_at": now, "peak": 0.9, "duration_ms": 900, "frames": 2}))
    snap = ctx.state.get("snapshot")
    assert snap["actions"] and snap["actions"][-1]["action"] == "smile" and snap["moods"] == []  # dashboard API sees it
    assert not [e for e in ctx.emitted if e[0] == "presence_changed"] and ctx.injected == []  # an action is never a transition or an announcement
    context = ctx.hooks["pre_llm_call"](platform="desktop")["context"]
    assert context.startswith("[face2ai] Ben steht vor der Kamera") and "Lächeln" not in context and "smile" not in context  # actions are not spoken into context
    text = ctx.commands["presence"]("")
    assert "Zuletzt gezeigt: kurzes Lächeln" in text  # but /presence lists them
    local = datetime.fromisoformat(now).astimezone().strftime("%H:%M:%S")
    assert f"kurzes Lächeln (0.9 s) ({local})" in text  # local wall time, like describe()'s "Zuletzt gesehen" — one clock per reply
    tool = json.loads(ctx.tools["presence_now"][1]({}))
    assert tool["actions"][-1]["duration_ms"] == 900 and "smile" not in tool["summary"]
    # presence_now hands the model raw action dicts, so the system section has to hedge them too
    assert "expression dynamics" in ctx.sections["face2ai"] and "never micro-expressions" in ctx.sections["face2ai"]
    assert "never evidence of what" in ctx.sections["face2ai"]
    # An action without a timestamp must not render an empty "()" …
    plugin._handle_frame(SseFrame("action", {"sequence": 2, "action": "brow_raise", "duration_ms": 2300}))
    assert "Brauen hoch (2.3 s)," in ctx.commands["presence"]("") or ctx.commands["presence"]("").endswith("Brauen hoch (2.3 s)")
    # … and Face2AI's explicit forget empties the mirror, persisted for the dashboard process.
    plugin._handle_frame(SseFrame("timeline_cleared", {"sequence": 3, "at": now}))
    snap = ctx.state.get("snapshot")
    assert (snap["moods"], snap["actions"]) == ([], [])
    assert "Zuletzt gezeigt" not in ctx.commands["presence"]("")
    assert snap["presence"]["display_name"] == "Ben"  # forgetting the history is not a presence change
