# Face2AI Expression Dynamics (Stage 2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** On top of Stage 1 (ADR-003: per-frame expression + hysteresis mood), Face2AI turns the blendshape time series into **facial action events** (onset → apex → offset per action, e.g. "brief smile, 0.9 s"), keeps a **short in-memory affect history per person** (valence/arousal samples, mood changes, actions), shows a **valence timeline in the browser tile**, and gives the **Hermes pane a per-person mood history + sparkline**. Everything stays local, in memory, opt-in, hedged.

**Architecture:** Two new pure services next to `MoodTracker`: `ActionTracker` (services/actions.py — hysteresis state machine per action group over `Expression.blendshapes`) and `AffectHistory` (services/timeline.py — bounded ring buffers of `AffectSample` / `MoodTransition` / `ActionEvent`, cleared on presence reset and restart). `Presence.valence/arousal` become the **live** smoothed affect (mood label keeps its hysteresis; `MoodTransition.valence/arousal` stay frozen at commit). New SSE event `action`, new `GET /api/expression/timeline`. The browser gets a tiny same-origin `EventSource` client (mood + action entries from server truth; the client-side `trackMood` debounce goes away) and a valence sparkline built from its own frames. The Hermes plugin stores mood/action history in its snapshot, proxies the timeline for the desktop pane, and draws an SVG sparkline. The voice agent only has to tolerate the new event.

**Tech Stack:** Python 3.12, FastAPI, pydantic; vanilla ES modules (no deps); Hermes plugin stdlib-only Python + plain ESM. No new dependencies. Same commands as Stage 1: run from the fork root `face_recognition/`, Python via `uv run --project apps/face2ai …` (bare `python3` is shimmed), tests `uv run --project apps/face2ai pytest apps/face2ai/tests -p no:cacheprovider` (117 today), JS `node --test 'apps/face2ai/tests/js/**/*.test.mjs'` (20), agent `uv run --project apps/face2ai-agent pytest apps/face2ai-agent/tests -p no:cacheprovider` (31), plugin `uv run --no-project --with pytest==9.0.2 pytest apps/face2ai-hermes-plugin/tests -p no:cacheprovider` (15). The RTK hook may swallow pytest summary lines — use `2>&1 | tail -1` and exit codes, never `-q` twice.

**Facts that shape the design (measured 2026-08-18):**
- The browser loop posts a frame every ~450 ms + ~150 ms round trip → **~1.7 fps**. Any timing we derive is quantized to ~0.6 s. Sub-second "micro-expressions" (CASME sense, < 500 ms) are **not resolvable** — we ship *expression dynamics*: onset/apex/offset with ~0.6 s resolution, labelled "brief" ≤ 1 s, "held" ≥ 5 s. Say so in ADR-004 and the UI.
- `Expression.blendshapes` is compacted: only entries ≥ 0.2 survive, rounded to 2 decimals (`compact_blendshapes`). Absent = 0.0 — compatible with an off-threshold of 0.2.
- Talking moves `jawOpen`, `mouthFunnel`, `mouthPucker`, `mouthClose` every frame; those are **not** actions here (speech artefacts). Blinks are noise at 1.7 fps and excluded.
- Presence key for expression work is `f"{state}:{identity_id or ''}"` — two different UNKNOWN persons share a key. Known limitation (documented), same as Stage 1 mood.
- Current code (HEAD `bc253dd`): `routes.py` has `_publish_presence_transition`, `_primary_expression`, `_observe_mood` (called after presence observe, no await in between), `set_expression` toggle resets mood on off; `MoodTracker` has `observe/current/reset/_smooth`, EMA state `_ema_valence/_ema_arousal`; `PresenceTracker.set_mood(mood, valence, arousal)`; `IdentityEventBroker.publish(kind, payload)`; SSE `_sse_event` adds `sequence`; browser `app.js` uses `trackMood` + `renderExpression/clearExpression/applyExpression`, `els.*` from `[id]` elements, `addEvent(title, detail)`, `MAX_EVENTS = 8`; `model.js` exports `describeExpression, axisPercent, formatAxis, trackMood`; plugin `PresenceStore.apply` handles `mood`, `snapshot()` → `{connected, engine_available, identity_count, last_frame_at, last_error, presence, history}`; `desktop/plugin.js` `onFrame` handles hello/heartbeat/presence/mood/lost, `Pane()` renders state + `moodLabel(p)` + transitions; `dashboard/plugin_api.py` has `/presence`, `/history`, `/health`, WS `/events`; agent `run_presence_loop` dispatches by `frame.event` (`hello/presence/store/heartbeat/mood/lost`).

**Rules that bind every task:** `domain/` imports no FastAPI/mediapipe; the wire carries labels, names, timestamps, rounded floats — never frames, landmarks, per-pixel data; actions/moods are **hints, never facts, never gate anything** (no greeting, no enrollment, no auth); nothing persisted by Face2AI (ring buffers in memory only, cleared on `POST /api/presence/reset` and restart); wording hedged (browser "looks …"/"brief smile", German "wirkt …"/"kurzes Lächeln"); upstream `face_recognition/` untouched; every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; TDD (failing test → run → implement → run → commit). Session hooks drop `.claude/homunculus/` dirs and a stray root `uv.lock` — delete, never commit.

---

### Task 0: Settings, domain models, live affect on `MoodTracker`

**Files:**
- Modify: `apps/face2ai/src/face2ai_app/config.py`
- Modify: `apps/face2ai/src/face2ai_app/domain/models.py`
- Modify: `apps/face2ai/src/face2ai_app/services/mood.py` (`affect()`)
- Test: `apps/face2ai/tests/test_config.py`, `apps/face2ai/tests/test_models.py`, `apps/face2ai/tests/test_mood.py`

**Step 1: Failing tests**
```python
# tests/test_config.py (append)
def test_action_and_timeline_settings(monkeypatch, tmp_path):
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
```
```python
# tests/test_models.py (append)
from face2ai_app.domain.models import ACTIONS, ActionEvent, AffectSample, TimelineSnapshot

def test_action_event_is_wire_safe_and_validated():
    t = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
    e = ActionEvent(at=t, action="smile", onset_at=t, apex_at=t, offset_at=t, peak=0.9, duration_ms=900, frames=2)
    assert set(e.model_dump()) == {"at", "identity_id", "display_name", "action", "onset_at", "apex_at", "offset_at", "peak", "duration_ms", "frames"}
    with pytest.raises(ValidationError):
        ActionEvent(at=t, action="wink", onset_at=t, apex_at=t, offset_at=t, peak=0.9, duration_ms=1, frames=1)
    with pytest.raises(ValidationError):
        ActionEvent(at=t, action="smile", onset_at=t, apex_at=t, offset_at=t, peak=1.5, duration_ms=1, frames=1)
    assert ACTIONS == ("smile", "frown", "brow_raise", "brow_furrow", "eye_squint", "eyes_wide", "nose_wrinkle", "lip_press")  # wire-contract pin

def test_timeline_snapshot_shape():
    snap = TimelineSnapshot(seconds=60, samples=[AffectSample(at=datetime.now(timezone.utc), valence=0.2)], moods=[], actions=[])
    assert snap.model_dump()["samples"][0]["mood"] is None
```
```python
# tests/test_mood.py (append)
def test_affect_is_live_while_current_is_frozen():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    for _ in range(2): tr = t.observe("KNOWN:a", happy(0.6), T0)
    assert tr is not None and t.current()[1] == tr.valence
    t.observe("KNOWN:a", happy(-0.2), T0)          # same mood, valence swings
    assert t.current()[1] == tr.valence            # frozen at commit
    assert t.affect() == (pytest.approx(0.35), pytest.approx(0.1))  # live EMA: 0.5*-0.2 + 0.5*0.9... compute from the implementation and pin
    assert t.affect() != (None, None)
    t.reset()
    assert t.affect() == (None, None)
```
(Compute the exact live values from the EMA before pinning: after frames 0.6, 0.6, −0.2 with α=0.5 zero-start: 0.3 → 0.45 → 0.125; arousal 0.1,0.1,0.1: 0.05 → 0.075 → 0.0875. Assert those with `pytest.approx`.)

**Step 2:** run → FAIL (`Settings` has no `action_on_threshold`; `ACTIONS` import error; `affect` missing).

**Step 3: Implement**
- `config.py`: fields `action_on_threshold: float = 0.35`, `action_off_threshold: float = 0.2`, `action_min_frames: int = 2`, `timeline_seconds: int = 600`; env `FACE2AI_ACTION_ON_THRESHOLD`, `FACE2AI_ACTION_OFF_THRESHOLD`, `FACE2AI_ACTION_MIN_FRAMES`, `FACE2AI_TIMELINE_SECONDS`; `__post_init__`: `0 < off < on <= 1` else `ValueError("FACE2AI_ACTION_OFF_THRESHOLD must be > 0 and < FACE2AI_ACTION_ON_THRESHOLD <= 1")`, `action_min_frames >= 1`, `timeline_seconds >= 10`.
- `domain/models.py` (after `MoodTransition`):
```python
ACTIONS = ("smile", "frown", "brow_raise", "brow_furrow", "eye_squint", "eyes_wide", "nose_wrinkle", "lip_press")


class ActionEvent(BaseModel):
    """One completed facial action (SSE ``action``): onset → apex → offset from blendshape intensities.

    Timing is quantized to the frame rate (~0.6 s at the browser's loop), so these are expression
    *dynamics*, not micro-expressions; a hint, never a fact. Wire-safe: label, names, timestamps,
    one peak intensity — no per-frame series, no landmarks.
    """

    at: datetime  # == offset_at: when the action became known
    identity_id: str | None = None
    display_name: str | None = None
    action: str
    onset_at: datetime
    apex_at: datetime
    offset_at: datetime
    peak: UnitScore
    duration_ms: int = Field(ge=0)
    frames: int = Field(ge=1)

    @field_validator("action")
    @classmethod
    def _known_action(cls, value: str) -> str:
        if value not in ACTIONS:
            raise ValueError(f"unknown action: {value}")
        return value


class AffectSample(BaseModel):
    """One point of the in-memory affect history: live smoothed valence/arousal + the mood at that time."""

    at: datetime
    identity_id: str | None = None
    display_name: str | None = None
    mood: str | None = None
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)


class TimelineSnapshot(BaseModel):
    """``GET /api/expression/timeline``: bounded, in-memory, never persisted; cleared on presence reset."""

    seconds: int
    samples: list[AffectSample] = Field(default_factory=list)
    moods: list[MoodTransition] = Field(default_factory=list)
    actions: list[ActionEvent] = Field(default_factory=list)
```
Update `Presence` docstring: "`valence`/`arousal` are the *live* smoothed affect (Stage 2) — may be present without a `mood`; `mood` keeps its hysteresis."
- `services/mood.py`: add
```python
    def affect(self) -> tuple[float | None, float | None]:
        """Live smoothed ``(valence, arousal)`` (rounded to 3), None until the first readable frame — what
        ``Presence`` carries since Stage 2. ``current()`` keeps the commit-frozen values for the mood event."""
        with self._lock:
            v = None if self._ema_valence is None else round(self._ema_valence, 3)
            a = None if self._ema_arousal is None else round(self._ema_arousal, 3)
            return v, a
```
and adjust the class docstring's last paragraph ("Valence/arousal … frozen" → "`current()` frozen at commit for the mood event; `affect()` live for `Presence`").

**Step 4:** run → all pass (117 + 5). `uv run --project apps/face2ai python -m compileall -q apps/face2ai/src`.
**Step 5:** `git add apps/face2ai/src/face2ai_app/config.py apps/face2ai/src/face2ai_app/domain/models.py apps/face2ai/src/face2ai_app/services/mood.py apps/face2ai/tests/test_config.py apps/face2ai/tests/test_models.py apps/face2ai/tests/test_mood.py && git commit -m "feat(face2ai): action/timeline settings, ActionEvent + AffectSample models, live affect"`

---

### Task 1: `ActionTracker` — onset/apex/offset per action group

**Files:**
- Create: `apps/face2ai/src/face2ai_app/services/actions.py`
- Test: `apps/face2ai/tests/test_actions.py`

**Step 1: Failing tests**
```python
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
```
**Step 2:** run → FAIL (module missing).
**Step 3: Implement** (`services/actions.py`, mirrors `MoodTracker` style: `threading.Lock`, `now` param, pure)
```python
"""Facial action dynamics from blendshape intensities: onset → apex → offset per action group.

Pure and per-presence like ``MoodTracker``. Each action is the mean of a small blendshape group;
a hysteresis state machine (``on_threshold`` to start, ``off_threshold`` to end) turns the per-frame
series into one ``ActionEvent`` per completed action. Timing is quantized to the frame rate; the
event says how long and how strong, never why. Speech articulators (jawOpen, mouthFunnel, …) and
blinks are deliberately not action groups.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from face2ai_app.domain.models import ACTIONS, ActionEvent, Expression

ACTION_GROUPS: dict[str, tuple[str, ...]] = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "frown": ("mouthFrownLeft", "mouthFrownRight"),
    "brow_raise": ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    "brow_furrow": ("browDownLeft", "browDownRight"),
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "eyes_wide": ("eyeWideLeft", "eyeWideRight"),
    "nose_wrinkle": ("noseSneerLeft", "noseSneerRight"),
    "lip_press": ("mouthPressLeft", "mouthPressRight"),
}
assert tuple(ACTION_GROUPS) == ACTIONS


def action_intensity(blendshapes: Mapping[str, float], action: str) -> float:
    """Mean intensity of the action's blendshape group; blendshapes missing from a frame count as 0."""
    group = ACTION_GROUPS[action]
    return sum(float(blendshapes.get(name, 0.0)) for name in group) / len(group)


@dataclass
class _Active:
    onset_at: datetime
    apex_at: datetime
    peak: float
    frames: int


class ActionTracker:
    def __init__(self, on_threshold: float = 0.35, off_threshold: float = 0.2, min_frames: int = 2) -> None:
        if not 0.0 < off_threshold < on_threshold <= 1.0:
            raise ValueError("need 0 < off_threshold < on_threshold <= 1")
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        self._on, self._off, self._min_frames = on_threshold, off_threshold, min_frames
        self._lock = threading.Lock()
        self._presence_key: str | None = None
        self._active: dict[str, _Active] = {}

    def observe(self, presence_key, expression, now=None, *, identity_id=None, display_name=None) -> list[ActionEvent]:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if presence_key != self._presence_key:
                self._active.clear()            # unknown offset for whatever was active: dropped, never guessed
                self._presence_key = presence_key
            if expression is None or not expression.blendshapes:
                self._active.clear()
                return []
            done: list[ActionEvent] = []
            for action in ACTIONS:
                value = action_intensity(expression.blendshapes, action)
                active = self._active.get(action)
                if active is None:
                    if value >= self._on:
                        self._active[action] = _Active(onset_at=now, apex_at=now, peak=value, frames=1)
                    continue
                if value >= self._off:
                    active.frames += 1
                    if value > active.peak:
                        active.peak, active.apex_at = value, now
                    continue
                del self._active[action]
                if active.frames >= self._min_frames:
                    done.append(ActionEvent(
                        at=now, identity_id=identity_id, display_name=display_name, action=action,
                        onset_at=active.onset_at, apex_at=active.apex_at, offset_at=now,
                        peak=round(min(active.peak, 1.0), 3),
                        duration_ms=int((now - active.onset_at).total_seconds() * 1000), frames=active.frames,
                    ))
            return done

    def reset(self) -> None:
        with self._lock:
            self._active.clear()
            self._presence_key = None
```
(Type hints on `observe` as in `MoodTracker`.)
**Step 4:** run → PASS. Full suite still green.
**Step 5:** commit `feat(face2ai): ActionTracker — onset/apex/offset per blendshape group`.

---

### Task 2: `AffectHistory` — bounded in-memory timeline

**Files:**
- Create: `apps/face2ai/src/face2ai_app/services/timeline.py`
- Test: `apps/face2ai/tests/test_timeline.py`

**Step 1: Failing tests**
```python
from datetime import datetime, timedelta, timezone

from face2ai_app.domain.models import ActionEvent, AffectSample, MoodTransition
from face2ai_app.services.timeline import AffectHistory

T0 = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def sample(i, identity="a", valence=0.5):
    return AffectSample(at=T0 + timedelta(seconds=i), identity_id=identity, valence=valence)


def test_samples_are_bounded_by_age_and_count():
    h = AffectHistory(max_seconds=60, max_samples=1000)
    for i in range(100):
        h.record_sample(sample(i))
    snap = h.snapshot(now=T0 + timedelta(seconds=99))
    assert snap.seconds == 60 and len(snap.samples) == 61 and snap.samples[0].at == T0 + timedelta(seconds=39)
    small = AffectHistory(max_seconds=60, max_samples=10)
    for i in range(100):
        small.record_sample(sample(i))
    assert len(small.snapshot(now=T0 + timedelta(seconds=99)).samples) == 10


def test_snapshot_filters_window_and_identity():
    h = AffectHistory(max_seconds=600)
    h.record_sample(sample(0, "a")); h.record_sample(sample(1, "b")); h.record_sample(sample(500, "a"))
    snap = h.snapshot(seconds=60, identity_id="a", now=T0 + timedelta(seconds=520))
    assert [s.at for s in snap.samples] == [T0 + timedelta(seconds=500)]
    assert h.snapshot(seconds=60, now=T0 + timedelta(seconds=520)).samples[-1].identity_id == "a"


def test_moods_and_actions_are_kept_and_cleared():
    h = AffectHistory(max_moods=2, max_actions=2)
    for i in range(3):
        h.record_mood(MoodTransition(at=T0 + timedelta(seconds=i), from_mood=None, to_mood="Happiness"))
        h.record_action(ActionEvent(at=T0 + timedelta(seconds=i), action="smile", onset_at=T0, apex_at=T0, offset_at=T0, peak=0.5, duration_ms=1, frames=1))
    snap = h.snapshot(now=T0 + timedelta(seconds=3))
    assert len(snap.moods) == 2 and len(snap.actions) == 2 and snap.moods[0].at == T0 + timedelta(seconds=1)
    h.clear()
    empty = h.snapshot()
    assert (empty.samples, empty.moods, empty.actions) == ([], [], [])
```
**Step 2:** FAIL. **Step 3:** implement with `collections.deque(maxlen=…)` for the three series, `threading.Lock`, `record_sample` drops leading samples older than `max_seconds` relative to the newest, `snapshot(seconds=None → max_seconds, identity_id=None, now=None → newest sample's `at` or utcnow)` returns `TimelineSnapshot(seconds=…, samples=[within window & identity], moods=[within window], actions=[within window])` — moods/actions also filtered by identity when given. `clear()` empties all three. Docstring: in-memory only, never persisted, cleared on presence reset and restart.
**Step 4:** PASS. **Step 5:** commit `feat(face2ai): AffectHistory in-memory timeline`.

---

### Task 3: Wire actions + timeline into routes, live affect on presence, `GET /api/expression/timeline`

**Files:**
- Modify: `apps/face2ai/src/face2ai_app/main.py` (`app.state.actions`, `app.state.history`)
- Modify: `apps/face2ai/src/face2ai_app/api/routes.py`
- Modify: `apps/face2ai/tests/conftest.py` (`FakeExpressionEngine.script`)
- Test: `apps/face2ai/tests/test_events_api.py`, `apps/face2ai/tests/test_expression_api.py`

**Step 1: Failing tests** (live-uvicorn fixture; its Settings gets `action_min_frames=1`)
```python
# conftest.py — FakeExpressionEngine: add
#   self.script: list[list[Expression | None]] = []   # per-call results, consumed first-in-first-out; falls back to self.expressions
# and in analyze(): scripted = list((self.script.pop(0) if self.script else self.expressions)[: len(boxes)])

def smiling(v):
    return Expression(dominant="Happiness", scores={"Happiness": 0.9}, valence=0.6, arousal=0.1, blendshapes={"mouthSmileLeft": v, "mouthSmileRight": v})

def test_action_events_and_timeline(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.script = [[smiling(0.9)], [smiling(0.9)], [smiling(0.0)]]
    live.client.post("/api/expression", json={"enabled": True})
    for _ in range(3):
        assert live.client.post("/api/recognize", content=b"frame", headers=HEADERS).status_code == 200
    frames = live.sse("/api/events?after=0", wanted=4, skip_heartbeats=True)
    assert [f["event"] for f in frames] == ["hello", "presence", "mood", "action"]
    action = frames[3]["data"]
    assert action["action"] == "smile" and action["frames"] == 2 and action["peak"] == 0.9
    assert set(action) == {"sequence", "at", "identity_id", "display_name", "action", "onset_at", "apex_at", "offset_at", "peak", "duration_ms", "frames"}
    for token in ("scores", "blendshapes", "encoding", "box"):
        assert token not in json.dumps(action)
    tl = live.client.get("/api/expression/timeline?seconds=60").json()
    assert len(tl["samples"]) == 3 and tl["samples"][-1]["mood"] == "Happiness" and len(tl["moods"]) == 1 and len(tl["actions"]) == 1
    presence = live.client.get("/api/presence").json()
    assert presence["mood"] == "Happiness" and presence["valence"] == pytest.approx(0.525)  # live EMA after 3 frames of 0.6

def test_presence_valence_is_live_even_without_mood(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.expressions = [Expression(dominant="Neutral", scores={"Neutral": 0.4, "Happiness": 0.3}, valence=0.2, arousal=0.0)]
    live.client.post("/api/expression", json={"enabled": True})
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    p = live.client.get("/api/presence").json()
    assert p["mood"] is None and p["valence"] == pytest.approx(0.1)

def test_presence_reset_clears_the_timeline_and_active_actions(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.expressions = [smiling(0.9)]
    live.client.post("/api/expression", json={"enabled": True})
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    live.client.post("/api/presence/reset")
    assert live.client.get("/api/expression/timeline").json()["samples"] == []
    fake_expression.expressions = [smiling(0.0)]
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    assert live.client.get("/api/expression/timeline").json()["actions"] == []   # the smile's offset is unknown → no event

def test_toggle_off_drops_active_actions(live, fake_engine, fake_expression, face):
    (enable → smiling(0.9) frame → toggle off → smiling(0.0) frame with expression off (engine not consulted) → SSE has no `action`)

def test_timeline_query_validation(client):
    assert client.get("/api/expression/timeline?seconds=1").status_code == 422
    assert client.get("/api/expression/timeline").json() == {"seconds": 600, "samples": [], "moods": [], "actions": []}
```
Also extend the existing forbidden-token sweep helper if one exists to cover `action` frames.

**Step 2:** FAIL. **Step 3: Implement**
- `main.py`: `app.state.actions = ActionTracker(app_settings.action_on_threshold, app_settings.action_off_threshold, app_settings.action_min_frames)`, `app.state.history = AffectHistory(max_seconds=app_settings.timeline_seconds)`.
- `routes.py`: accessors `_actions`, `_history`; replace `_observe_mood` with
```python
def _observe_expression(request: Request, event: RecognitionEvent, now: datetime) -> None:
    """One frame → mood (hysteresis label), live affect on the presence, action events, history samples.
    Frames without a usable expression count as missing for the mood, drop active actions and add no sample."""
    tracker = _presence(request)
    presence = tracker.snapshot(now)
    key = f"{presence.state}:{presence.identity_id or ''}"
    who = {"identity_id": presence.identity_id, "display_name": presence.display_name}
    expression = _primary_expression(presence, event)
    mood, history, events = _mood(request), _history(request), _events(request)

    transition = mood.observe(key, expression, now, **who)
    if transition is not None:
        events.publish("mood", transition)
        history.record_mood(transition)
    current_mood, _, _ = mood.current()
    valence, arousal = mood.affect()
    tracker.set_mood(current_mood, valence, arousal)          # Stage 2: presence carries the live affect
    if expression is not None and (valence is not None or arousal is not None):
        history.record_sample(AffectSample(at=now, mood=current_mood, valence=valence, arousal=arousal, **who))
    for action in _actions(request).observe(key, expression, now, **who):
        events.publish("action", action)
        history.record_action(action)
```
- `_publish_presence_transition`: on a NO_SIGNAL crossing also `_actions(request).reset()`.
- `reset_presence`: additionally `_actions(request).reset()` and `_history(request).clear()` (docstring: "forget everything, incl. the affect history").
- `set_expression` off-branch: `_actions(request).reset()`.
- New route:
```python
@router.get("/api/expression/timeline", response_model=TimelineSnapshot)
def expression_timeline(
    request: Request,
    seconds: int = Query(default=600, ge=10, le=3600),
    identity_id: str | None = Query(default=None, max_length=80),
) -> TimelineSnapshot:
    """In-memory affect history (samples, mood changes, actions) — bounded, never persisted."""
    return _history(request).snapshot(seconds=seconds, identity_id=identity_id)
```
- SSE docstring: add `action`; README SSE bullet: `action` — a completed facial action (onset/apex/offset, peak, duration); hint, never a fact.
**Step 4:** PASS (all). compileall. Real smoke on :8799 (this Mac has the extra): enable → POST `examples/obama.jpg` ×2, then `examples/biden.jpg` ×1 (same key `UNKNOWN:` — the smile's offset arrives with the non-smiling frame) → `curl -N -m 2 'localhost:8799/api/events?after=0'` shows an `action` frame `smile`; `GET /api/expression/timeline` has samples/actions. Kill only the PID you started.
**Step 5:** commit `feat(face2ai): action events, live affect on presence, in-memory timeline endpoint`.

---

### Task 4: Browser — SSE mood/action entries, valence sparkline in the tile

**Files:**
- Create: `apps/face2ai/src/face2ai_app/static/js/events.js`
- Modify: `apps/face2ai/src/face2ai_app/static/js/model.js` (`describeAction`, `formatDuration`, `pushSample`, `sparklinePoints`; **remove** `trackMood`)
- Modify: `apps/face2ai/src/face2ai_app/static/js/app.js`, `static/index.html`, `static/css/app.css`
- Test: `apps/face2ai/tests/js/model.test.mjs`, `apps/face2ai/tests/test_static.py`

**Step 1: Failing JS tests**
```js
test('describeAction is hedged, quantised and bilingual', () => {
  const brief = describeAction({ action: 'smile', duration_ms: 900, peak: 0.9 }, 'en');
  assert.equal(brief.label, 'brief smile (0.9 s)');
  assert.equal(describeAction({ action: 'brow_raise', duration_ms: 2300 }, 'en').label, 'brow raise (2.3 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 6000 }, 'en').label, 'held smile (6.0 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 900 }, 'de').label, 'kurzes Lächeln (0.9 s)');
  assert.equal(describeAction({ action: 'wink', duration_ms: 900 }, 'en').label, 'brief wink (0.9 s)'); // unknown label never throws
  assert.equal(describeAction(null, 'en'), null);
});
test('sparkline maps -1..1 samples into the box, newest right', () => {
  const s = pushSample([], 0, 1); pushSample(s, 1, 2); pushSample(s, -1, 3);
  assert.equal(sparklinePoints(s, 120, 24), '0,12 60,0 120,24');
  assert.equal(sparklinePoints([], 120, 24), '');
  assert.equal(pushSample(Array.from({ length: 120 }, (_, i) => ({ v: 0, at: i })), 0.5, 999, 120).length, 120); // bounded
});
```
Import `describeAction, formatDuration, pushSample, sparklinePoints` in the test; drop the `trackMood` tests.

**Step 2:** `node --test …` → FAIL. **Step 3: Implement**
- `model.js`: `ACTION_WORDS = { en: {smile:'smile', frown:'frown', brow_raise:'brow raise', brow_furrow:'brow furrow', eye_squint:'eye squint', eyes_wide:'eyes wide', nose_wrinkle:'nose wrinkle', lip_press:'lip press'}, de: {smile:'Lächeln', frown:'Mundwinkel runter', brow_raise:'Brauen hoch', brow_furrow:'Stirnrunzeln', eye_squint:'Augen zusammengekniffen', eyes_wide:'Augen weit', nose_wrinkle:'Nasenrümpfen', lip_press:'Lippen gepresst'} }`; qualifiers `≤ 1000 ms → 'brief '/'kurzes '`, `≥ 5000 → 'held '/'anhaltendes '`; `formatDuration(ms)` → `'0.9 s'` (one decimal); `describeAction(a, lang)` → `{ label, tone: 'muted', duration }` or null; unknown action → its raw name lower-cased with `_`→space, still qualified. `pushSample(samples, value, at, max = 120)` mutates+returns array of `{v, at}` bounded to `max`; `sparklinePoints(samples, w, h)` → `x = i/(n-1)*w` (n=1 → x 0), `y = (1 - clamp(v)) / 2 * h`, rounded to 1 decimal, joined by spaces. Remove `trackMood`.
- `events.js`:
```js
// Same-origin SSE client for the browser: only mood + action entries; presence/store/heartbeat are ignored
// (the page derives its own view from the frames it sends). EventSource reconnects by itself.
export function subscribeEvents({ onMood, onAction, onOpen, onError }, url = '/api/events?role=browser') {
  const source = new EventSource(url);
  const parse = (e) => { try { return JSON.parse(e.data); } catch { return null; } };
  source.addEventListener('mood', (e) => { const d = parse(e); if (d) onMood?.(d); });
  source.addEventListener('action', (e) => { const d = parse(e); if (d) onAction?.(d); });
  source.onopen = () => onOpen?.();
  source.onerror = () => onError?.();
  return () => source.close();
}
```
- `index.html` tile: after the bars add `<svg class="affect-spark" id="valenceSpark" viewBox="0 0 120 24" preserveAspectRatio="none" role="img" aria-label="Valence over the last minute" hidden><line class="affect-zero" x1="0" y1="12" x2="120" y2="12"/><polyline id="valenceLine" points=""/></svg>` (SVG counts as no external ref; keep it inline).
- `app.js`: `state.affect = []` (drop `state.mood`, `MOOD_STABLE_TICKS`, `trackMood` import); in `renderExpression`: when `described?.valence` is finite → `pushSample(state.affect, described.valence, Date.now())`, `els.valenceLine.setAttribute('points', sparklinePoints(state.affect, 120, 24))`, `els.valenceSpark.hidden = state.affect.length < 2`; `clearExpression` also `state.affect = []`, hides the svg. On module init: `subscribeEvents({ onMood: (t) => { if (t.to_mood) addEvent('Mood', \`${t.display_name || 'someone'} ${describeExpression({ dominant: t.to_mood, scores: {} }, EXPRESSION_LANG).label}.\`); }, onAction: (a) => { const d = describeAction(a, EXPRESSION_LANG); if (d) addEvent('Expression', \`${a.display_name || 'someone'}: ${d.label}\`); }, onError: () => { if (!state.eventsWarned) { state.eventsWarned = true; addEvent('Live events unavailable', 'Mood and expression entries pause until the stream reconnects.'); } } })`. Only log moods/actions while `state.expression.enabled` (server events are the truth; entries stay quiet when off).
- `css`: `.affect-spark { width: 100%; height: 24px; margin-top: 8px; } .affect-spark polyline { fill: none; stroke: var(--accent, #82f3d4); stroke-width: 1.2; vector-effect: non-scaling-stroke; } .affect-zero { stroke: rgba(190,219,234,.18); stroke-dasharray: 2 3; }` (reuse existing tokens; check `app.css` variables).
- `test_static.py`: index has `valenceSpark`; `events.js` is imported by app.js and in the import graph test; hedged wording rules extended: no `"is smiling"`/`"detected"` next to action wording; `trackMood` gone from the bundle.

**Step 4:** node tests + `node --check` + `test_static.py` PASS; full pytest. Browser smoke on :8799: open, enable, POST obama/obama/biden via curl (frames arrive from curl, the page's SSE shows `Expression: someone: brief smile (…)` in the event stream — verify with `document.querySelector` via the Chrome tool or by reading the DOM), tile sparkline appears once the page itself sends frames (needs a face; may be left to the user's camera test).
**Step 5:** commit `feat(face2ai): browser mood/action entries from SSE + valence sparkline`.

---

### Task 5: Hermes plugin — mood/action history, timeline proxy, pane sparkline

**Files:**
- Modify: `apps/face2ai-hermes-plugin/face2ai/presence.py`, `__init__.py`, `dashboard/plugin_api.py`, `desktop/plugin.js`, `README.md`
- Test: `apps/face2ai-hermes-plugin/tests/test_presence.py`, `tests/test_plugin_hooks.py`

**Step 1: Failing tests**
```python
def test_store_keeps_mood_and_action_history_and_snapshot_carries_them():
    store = PresenceStore()
    store.apply(SseFrame("hello", {"presence": {"state": "KNOWN", "identity_id": "a", "display_name": "Ben"}}))
    store.apply(SseFrame("mood", {"at": "2026-08-18T12:00:00Z", "identity_id": "a", "display_name": "Ben", "from_mood": None, "to_mood": "Happiness", "valence": 0.6, "arousal": 0.1}))
    store.apply(SseFrame("action", {"at": "2026-08-18T12:00:03Z", "identity_id": "a", "display_name": "Ben", "action": "smile", "onset_at": "2026-08-18T12:00:01Z", "apex_at": "2026-08-18T12:00:02Z", "offset_at": "2026-08-18T12:00:03Z", "peak": 0.9, "duration_ms": 2000, "frames": 4}))
    snap = store.snapshot()
    assert snap["moods"][-1]["to_mood"] == "Happiness" and snap["actions"][-1]["action"] == "smile"
    assert set(snap) == {"connected", "engine_available", "identity_count", "last_frame_at", "last_error", "presence", "history", "moods", "actions"}

def test_action_sentence_is_hedged():
    assert action_sentence({"action": "smile", "duration_ms": 900}, language="de") == "kurzes Lächeln (0.9 s)"
    assert action_sentence({"action": "brow_raise", "duration_ms": 2300}, language="en") == "brow raise (2.3 s)"
    assert action_sentence({"action": "wink", "duration_ms": 900}, language="de") == "kurzes wink (0.9 s)"

# test_plugin_hooks.py: `_handle_frame` on an `action` frame persists (ctx.state has "actions"), does not emit presence_changed, does not announce.
```
**Step 2:** FAIL. **Step 3:** implement: `PresenceStore.moods: deque(maxlen=50)`, `actions: deque(maxlen=30)` filled in `apply` for `mood`/`action` frames (return None); `snapshot()` adds `"moods": last 20`, `"actions": last 10` (raw dicts, only wire keys); `ACTION_WORDS` de/en table + `action_sentence(data, language)`; `describe()` unchanged (actions are not spoken into context — noise); `_cmd_presence` appends "Zuletzt gezeigt: kurzes Lächeln (12:00:03) …" (last 3 actions) when present. `__init__._handle_frame`: persist on `action`. `plugin_api.py`: `GET /timeline` → proxies `{events_url}/api/expression/timeline` with `seconds` (default 600, clamp 10..3600) and optional `identity_id`; on error `{"error": str, "seconds": seconds, "samples": [], "moods": [], "actions": []}`. `desktop/plugin.js`: `ACTION_LABELS` (de) + `actionLabel(a)`; `onFrame`: `action` → `actions = [...actions.slice(-9), data]`, `mood` with `to_mood` → also `moods = [...moods.slice(-19), data]`; `refresh()` additionally fetches `/timeline?seconds=600` → `timeline`; `Pane`: `Sparkline({samples})` (plain `svg`/`polyline`, width 240, height 28, zero line, samples filtered to `presence.identity_id` when set), "Stimmung zuletzt" (last 6 moods: time · name · `wirkt …`), "Ausdruck zuletzt" (last 5 actions: time · `kurzes Lächeln (0.9 s)`), tooltip on both blocks "Vermutung aus dem Gesichtsausdruck, keine Tatsache". README: history/timeline paragraph (in memory on the Face2AI side, snapshot mirrored to plugin state — current values + last few entries, no long-term storage).
**Step 4:** plugin tests PASS (15 → 18+), `node --check desktop/plugin.js`. **Measured 2026-08-18 (Node v24.16.0): `node --check plugin.js` does NOT reject JSX in this file** — the `import` statements trigger module-syntax detection, which swallows the error, so the "no JSX allowed" claim rested on a check that cannot fail. Use `cp face2ai/desktop/plugin.js "$TMPDIR/plugin.mjs" && node --check "$TMPDIR/plugin.mjs"` (exit 1 on injected JSX) and/or `tests/test_desktop_plugin.py`, which greps for JSX syntax. **Step 5:** commit `feat(plugin): mood/action history, timeline proxy, pane sparkline`.

---

### Task 6: Voice agent — tolerate `action` frames

**Files:** `apps/face2ai-agent/src/face2ai_agent/presence.py`, `apps/face2ai-agent/tests/test_presence.py`, `apps/face2ai-agent/README.md`
**Step 1:** test: `run_presence_loop` (or the frame dispatcher used in tests) receives an `action` frame → no dispatch, no exception, memory unchanged; a following `heartbeat` still works. **Step 2:** run (may already pass — then keep the test as a pin and say so). **Step 3:** if needed add an explicit `elif frame.event == "action": pass  # dynamics are for panes/history, not for the voice loop` with a debug log; README: one sentence "action events are ignored by the agent by design (no reactions to facial dynamics)". **Step 4/5:** 31 → 32 pass; commit `test(agent): action frames are ignored by design`.

---

### Task 7: Docs — ADR-004, README, AGENTS, VALIDATION, UI_DIRECTION, CLAUDE.md

**Files:** Create `apps/face2ai/docs/architecture/ADR-004-expression-dynamics.md`; modify `apps/face2ai/README.md`, `AGENTS.md`, `docs/boilerplate/VALIDATION.md`, `docs/UI_DIRECTION.md`, `docs/boilerplate/ARCHITECTURE.md`, `docs/boilerplate/architecture-decision.json` (`adr_paths`), `/Users/benjaminpoersch/face2ai/CLAUDE.md`, `apps/face2ai-hermes-plugin/README.md` (if not done in Task 5).

ADR-004: Status Accepted; Context (Stage 2 asks: micro-events, per-person history, timelines); Decision (`ActionTracker` groups + thresholds + min_frames, why jaw/blink excluded; `AffectHistory` in memory, bounded, cleared on reset/restart; live affect on `Presence`, frozen on `MoodTransition`; SSE `action`; `GET /api/expression/timeline`; browser SSE client + local sparkline; plugin history/sparkline; agent ignores actions); **Honesty**: ~1.7 fps → ~0.6 s resolution; "micro-expressions" out of scope and why; actions are dynamics of visible expression, dataset-biased models, talking artefacts; unknown persons share one key; Consequences (+/−: more wire events — bounded by hysteresis + min_frames; browser now holds an SSE connection); Review triggers (higher frame rate path, per-person keys for unknowns, temporal models). AGENTS.md: actions/timeline = hints, never gate; nothing persisted; wording rules extended ("brief smile", never "smiled because…"); consumers must not react to `action` events with behaviour. README: "Expression dynamics" section (event shape, timeline endpoint, settings). VALIDATION: gate steps — enable, smile briefly → `Expression: … brief smile (x s)` entry + `action` in SSE; hold a smile ≥ 5 s → "held smile"; sparkline moves; Hermes pane shows sparkline + lists; `POST /api/presence/reset` empties `/api/expression/timeline`. UI_DIRECTION: sparkline + entries described (calm; ≤ 8 entries; actions may be frequent → note). CLAUDE.md: SSE list `+ action`, endpoint, env vars, test counts. All gates green; commit `docs(face2ai): ADR-004 expression dynamics + gates`.

---

### Task 8: Live verification + push + PR comment

1. Restart the backend on :8765 by killing the `uv run --project apps/face2ai face2ai` parent + child PIDs (**never** `lsof -ti :8765 | xargs kill` — that kills the agent console and every SSE consumer too), `nohup uv run --project apps/face2ai face2ai > <scratchpad>/srv.log 2>&1 &`; `POST /api/expression {"enabled":true}`.
2. curl: `obama.jpg` ×2 → `biden.jpg` ×1 → `curl -N -m 2 'localhost:8765/api/events?after=0'` shows `mood` and `action` (`smile`); `GET '/api/expression/timeline?seconds=120'` shows samples/moods/actions; `/api/presence.valence` moves between frames.
3. Chrome (hard-reload once — assets are `no-cache` now, but pre-fix caches may linger): "Expression: on", event stream gets `Mood`/`Expression` entries from the curl-driven frames; with a face in front of the camera the tile's sparkline draws (user gate if no face is available).
4. Agent: `uv run face2ai-agent smoke "Wer ist da?"` still fine (agent ignores actions); Hermes: redeploy `apps/face2ai-hermes-plugin/deploy.sh`, then restart `hermes-gateway` explicitly and check `ActiveEnterTimestamp` (the script's remote heredoc once did not run); ask Hermes via `/v1/chat/completions` — unchanged hedged answer; desktop pane (user): sparkline + lists after ⌘K → Reload desktop plugins.
5. Gates: app pytest, JS, compileall, agent, plugin, CI-equivalent no-extra run; `git diff 4702e4d..HEAD --stat -- face_recognition/` = 0.
6. Push `feat/face2ai-ui-port` (or a fresh branch if PR #3 is merged by then), PR comment with counts + measured numbers (events per minute while talking/smiling; timeline sizes), memory note update (`face2ai-expression-engine.md`).

**Out of scope (Stage 3 candidates):** per-person keys for unknown faces (short-lived track ids), higher frame rate path (worker-side capture), temporal models over blendshape windows, exporting history (never by default).
