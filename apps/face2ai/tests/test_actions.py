from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from face2ai_app.domain.models import Expression
from face2ai_app.services.actions import ACTION_GROUPS, ActionTracker, action_intensity

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
TICK = timedelta(milliseconds=500)


def expr(**blend):
    return Expression(dominant="Neutral", scores={"Neutral": 1.0}, blendshapes=blend)


def smile(v):
    return expr(mouthSmileLeft=v, mouthSmileRight=v)


def test_action_intensity_is_the_group_mean_with_missing_as_zero():
    assert action_intensity({"mouthSmileLeft": 0.8}, "smile") == pytest.approx(0.4)
    assert action_intensity({}, "brow_raise") == 0.0
    assert set(ACTION_GROUPS) == {"smile", "frown", "brow_raise", "brow_furrow", "eye_squint", "eyes_wide", "nose_wrinkle", "lip_press"}
    assert "jawOpen" not in {b for group in ACTION_GROUPS.values() for b in group}  # speech artefact, never an action


def test_smile_onset_apex_offset():
    t = ActionTracker(on_threshold=0.35, off_threshold=0.2, min_frames=2)
    values = [0.0, 0.5, 0.8, 0.6, 0.1]
    out = []
    for i, v in enumerate(values):
        out.extend(t.observe("KNOWN:a", smile(v), T0 + i * TICK, identity_id="a", display_name="Ada"))
    assert len(out) == 1
    e = out[0]
    assert e.action == "smile" and e.identity_id == "a" and e.display_name == "Ada"
    assert e.onset_at == T0 + 1 * TICK and e.apex_at == T0 + 2 * TICK and e.offset_at == e.at == T0 + 4 * TICK
    assert e.peak == pytest.approx(0.8) and e.duration_ms == 1500 and e.frames == 3


def test_single_frame_spike_needs_min_frames_one():
    strict = ActionTracker(min_frames=2)
    loose = ActionTracker(min_frames=1)
    for tr in (strict, loose):
        tr.observe("K:a", smile(0.9), T0)
    assert strict.observe("K:a", smile(0.0), T0 + TICK) == []
    spike = loose.observe("K:a", smile(0.0), T0 + TICK)
    assert len(spike) == 1 and spike[0].frames == 1 and spike[0].duration_ms == 500


def test_hysteresis_holds_between_thresholds():
    t = ActionTracker(on_threshold=0.35, off_threshold=0.2, min_frames=1)
    t.observe("K:a", smile(0.5), T0)
    assert t.observe("K:a", smile(0.25), T0 + TICK) == []      # below on, above off: still active
    assert t.observe("K:a", smile(0.3), T0 + 2 * TICK) == []
    assert len(t.observe("K:a", smile(0.1), T0 + 3 * TICK)) == 1


def test_key_change_or_missing_frame_drops_the_active_action():
    t = ActionTracker(min_frames=1)
    t.observe("K:a", smile(0.9), T0)
    assert t.observe("K:b", smile(0.0), T0 + TICK) == []        # different presence: unknown offset → nothing
    t.observe("K:b", smile(0.9), T0 + 2 * TICK)
    assert t.observe("K:b", None, T0 + 3 * TICK) == []           # unreadable frame: dropped, not completed
    assert t.observe("K:b", smile(0.0), T0 + 4 * TICK) == []     # nothing was active any more


def test_two_actions_can_overlap():
    t = ActionTracker(min_frames=1)
    t.observe("K:a", expr(mouthSmileLeft=0.9, mouthSmileRight=0.9, eyeSquintLeft=0.6, eyeSquintRight=0.6), T0)
    out = t.observe("K:a", expr(), T0 + TICK)
    assert sorted(e.action for e in out) == ["eye_squint", "smile"]


def test_reset_forgets_active_actions():
    t = ActionTracker(min_frames=1)
    t.observe("K:a", smile(0.9), T0)
    t.reset()
    assert t.observe("K:a", smile(0.0), T0 + TICK) == []


@pytest.mark.parametrize("kwargs", [{"on_threshold": 0.2, "off_threshold": 0.3}, {"on_threshold": 1.5}, {"off_threshold": 0.0}, {"min_frames": 0}])
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        ActionTracker(**kwargs)


def test_clock_stepping_back_never_raises_and_clamps_duration():
    t = ActionTracker(min_frames=1)
    t.observe("K:a", smile(0.9), T0)
    out = t.observe("K:a", smile(0.0), T0 - TICK)   # wall clock stepped back (sleep / NTP): still one event, duration clamped
    assert len(out) == 1 and out[0].duration_ms == 0
