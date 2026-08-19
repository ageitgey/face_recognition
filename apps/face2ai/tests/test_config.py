import inspect
import re
from dataclasses import fields
from pathlib import Path

import pytest

from face2ai_app.config import Settings

_EXPRESSION_ENV = (
    "FACE2AI_EXPRESSION_ENABLED",
    "FACE2AI_EXPRESSION_MODELS_DIR",
    "FACE2AI_MOOD_STABLE_TICKS",
    "FACE2AI_MOOD_MIN_SCORE",
    "FACE2AI_ACTION_ON_THRESHOLD",
    "FACE2AI_ACTION_OFF_THRESHOLD",
    "FACE2AI_ACTION_MIN_FRAMES",
    "FACE2AI_TIMELINE_SECONDS",
    "FACE2AI_DATA_DIR",
)


def _clear_env(monkeypatch):
    for name in _EXPRESSION_ENV:
        monkeypatch.delenv(name, raising=False)


def test_expression_settings_have_safe_defaults(monkeypatch):
    _clear_env(monkeypatch)
    settings = Settings.from_env()
    assert settings.expression_enabled is False  # opt-in
    assert settings.expression_models_dir == Path.home() / ".face2ai" / "models"
    assert settings.expression_models_dir == settings.data_dir / "models"
    assert settings.mood_stable_ticks == 3
    assert settings.mood_min_score == 0.5


def test_expression_settings_read_env(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FACE2AI_EXPRESSION_ENABLED", "true")
    monkeypatch.setenv("FACE2AI_EXPRESSION_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("FACE2AI_MOOD_STABLE_TICKS", "5")
    monkeypatch.setenv("FACE2AI_MOOD_MIN_SCORE", "0.7")
    settings = Settings.from_env()
    assert settings.expression_enabled is True
    assert settings.expression_models_dir == tmp_path
    assert settings.mood_stable_ticks == 5
    assert settings.mood_min_score == 0.7


def test_expression_models_dir_derives_from_data_dir(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FACE2AI_DATA_DIR", str(tmp_path))
    settings = Settings.from_env()
    assert settings.data_dir == tmp_path
    assert settings.expression_models_dir == tmp_path / "models"


@pytest.mark.parametrize("raw", ["ture", "2", "enabled", "ja"])
def test_expression_enabled_rejects_garbage(monkeypatch, raw):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FACE2AI_EXPRESSION_ENABLED", raw)
    with pytest.raises(ValueError, match="FACE2AI_EXPRESSION_ENABLED must be a boolean"):
        Settings.from_env()


@pytest.mark.parametrize(("raw", "expected"), [("1", True), (" YES ", True), ("On", True),
                                               ("0", False), ("no", False), ("OFF", False), ("", False)])
def test_expression_enabled_accepts_known_spellings(monkeypatch, raw, expected):
    _clear_env(monkeypatch)
    monkeypatch.setenv("FACE2AI_EXPRESSION_ENABLED", raw)
    assert Settings.from_env().expression_enabled is expected


def test_mood_validators_reject_out_of_range():
    with pytest.raises(ValueError, match="FACE2AI_MOOD_STABLE_TICKS"):
        Settings(mood_stable_ticks=0)
    with pytest.raises(ValueError, match="FACE2AI_MOOD_MIN_SCORE"):
        Settings(mood_min_score=0)
    with pytest.raises(ValueError, match="FACE2AI_MOOD_MIN_SCORE"):
        Settings(mood_min_score=1.5)
    assert Settings(mood_min_score=1.0).mood_min_score == 1.0


def test_action_and_timeline_settings(monkeypatch):
    _clear_env(monkeypatch)
    s = Settings.from_env()
    assert (s.action_on_threshold, s.action_off_threshold, s.action_min_frames, s.timeline_seconds) == (0.35, 0.2, 2, 600)
    monkeypatch.setenv("FACE2AI_ACTION_ON_THRESHOLD", "0.5")
    monkeypatch.setenv("FACE2AI_ACTION_MIN_FRAMES", "1")
    monkeypatch.setenv("FACE2AI_TIMELINE_SECONDS", "120")
    s = Settings.from_env()
    assert (s.action_on_threshold, s.action_min_frames, s.timeline_seconds) == (0.5, 1, 120)


@pytest.mark.parametrize("kwargs", [
    {"action_on_threshold": 0.2, "action_off_threshold": 0.3},  # off must be < on
    {"action_on_threshold": 1.5},
    {"action_min_frames": 0},
    {"timeline_seconds": 5},
])
def test_action_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


# --- Every knob is range-checked, and stays that way (plan §5) ---------------------------------
#
# `port`, `match_tolerance`, `max_frame_bytes` and `greeting_cooldown_seconds` carried a documented
# range in README.md that nothing enforced. `match_tolerance` is the knob that decides who you are.

@pytest.mark.parametrize(("kwargs", "match"), [
    ({"port": 0}, "FACE2AI_PORT"),
    ({"port": -1}, "FACE2AI_PORT"),
    ({"port": 70000}, "FACE2AI_PORT"),
    ({"match_tolerance": -1}, "FACE2AI_MATCH_TOLERANCE"),
    ({"match_tolerance": 0}, "FACE2AI_MATCH_TOLERANCE"),
    ({"match_tolerance": 2.5}, "FACE2AI_MATCH_TOLERANCE"),
    ({"max_frame_bytes": 0}, "FACE2AI_MAX_FRAME_BYTES"),
    ({"max_frame_bytes": -1}, "FACE2AI_MAX_FRAME_BYTES"),
    ({"greeting_cooldown_seconds": -5}, "FACE2AI_GREETING_COOLDOWN_SECONDS"),
])
def test_documented_ranges_are_enforced(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Settings(**kwargs)


def test_documented_range_edges_are_accepted():
    """The bounds README.md prints are inclusive where it says so — a validator must not narrow them."""
    assert Settings(port=1).port == 1
    assert Settings(port=65535).port == 65535
    assert Settings(match_tolerance=2.0).match_tolerance == 2.0
    assert Settings(max_frame_bytes=1).max_frame_bytes == 1
    assert Settings(greeting_cooldown_seconds=0).greeting_cooldown_seconds == 0


_NUMERIC_ANNOTATION = re.compile(r"\b(?:int|float)\b")


def test_every_numeric_setting_is_validated():
    """Each new knob must be range-checked in ``__post_init__`` — the habit that produced 4 unchecked knobs.

    This is the owner for the rule, not a check of one knob: it fails on the commit that adds a
    numeric setting without a bound, which is exactly how `port`, `match_tolerance`,
    `max_frame_bytes` and `greeting_cooldown_seconds` came to be unvalidated one commit at a time.

    Deliberately a *source* scan: it proves a knob is mentioned in ``__post_init__``, not that the
    bound is the right one. That is the cheapest owner that cannot be satisfied by construction —
    the range itself is owned by the per-knob tests above and by README.md.
    """
    numeric = {f.name for f in fields(Settings) if _NUMERIC_ANNOTATION.search(str(f.type))}
    assert "match_tolerance" in numeric, "field discovery broke; an empty set would pass vacuously"
    checked = set(re.findall(r"self\.(\w+)", inspect.getsource(Settings.__post_init__)))
    assert numeric <= checked, f"unvalidated numeric settings: {sorted(numeric - checked)}"


def test_every_setting_is_documented():
    """README.md's configuration table is the whole env surface — nothing may be added without a row."""
    config_py = Path(__file__).resolve().parents[1] / "src" / "face2ai_app" / "config.py"
    readme = Path(__file__).resolve().parents[1] / "README.md"
    names = set(re.findall(r"FACE2AI_[A-Z_]+", config_py.read_text(encoding="utf-8")))
    assert "FACE2AI_MATCH_TOLERANCE" in names, "env-name discovery broke; an empty set would pass vacuously"
    documented = readme.read_text(encoding="utf-8")
    assert sorted(n for n in names if f"`{n}`" not in documented) == []
