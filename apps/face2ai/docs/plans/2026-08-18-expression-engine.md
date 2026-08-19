# Face2AI Expression Engine (Stage 1) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Face2AI reads facial expression, valence/arousal and facial action signals (blendshapes, head pose) **locally** on the Mac, publishes a debounced "mood" next to presence, and every consumer (browser, voice agent, Hermes plugin) sees it — opt-in, no frames leave the device.

**Architecture:** New port `ExpressionEngine` beside `RecognitionEngine`; one adapter combining MediaPipe Face Landmarker (52 blendshapes + head pose, CPU delegate) and EmotiEffLib `enet_b0_8_va_mtl` (8 emotion classes + valence + arousal). `IdentityService.recognize()` attaches an `Expression` per detected face; a new `MoodTracker` (services/mood.py) turns per-frame scores into stable `mood` transitions published on the existing SSE stream; `Presence` carries `mood/valence/arousal`. Consumers only add wording. Everything behind an explicit runtime opt-in (`POST /api/expression`) and an install extra.

**Tech Stack:** Python 3.12, FastAPI, pydantic; `mediapipe==0.10.21` (**not 1.0.1** — its macOS wheel aborts in `DrishtiMetalHelper`; use `BaseOptions.Delegate.CPU`), `emotiefflib==1.1.1` + `onnxruntime`, numpy/Pillow (already deps). Model assets: `face_landmarker.task` (3.7 MB, `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`), EmotiEffLib downloads `enet_b0_8_va_mtl.onnx` on first use into its cache. Measured on the M1: landmarker ~9 ms/frame, emotion+VA ~15 ms/face → fits the 450 ms loop.

**Verified API facts (2026-08-18, scratch project):**
```python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
base = mp_python.BaseOptions(model_asset_path=path, delegate=mp_python.BaseOptions.Delegate.CPU)
opts = vision.FaceLandmarkerOptions(base_options=base, output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True, num_faces=4,
                                    running_mode=vision.RunningMode.IMAGE)
lm = vision.FaceLandmarker.create_from_options(opts)
res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8_array))
# res.face_landmarks[i] (478 NormalizedLandmark with .x .y in 0..1), res.face_blendshapes[i] (52 Category with .category_name/.score),
# res.facial_transformation_matrixes[i] (4x4)
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
rec = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_va_mtl", device="cpu")
labels, out = rec.predict_emotions(face_rgb_crop, logits=True)   # out shape (1, 10): 8 emotion logits + valence + arousal
rec.idx_to_emotion_class == {0:'Anger',1:'Contempt',2:'Disgust',3:'Fear',4:'Happiness',5:'Neutral',6:'Sadness',7:'Surprise'}
```

**Rules that bind every task:** `domain/` imports no FastAPI/mediapipe/emotiefflib; the wire (presence, SSE, agent, Hermes) carries labels/scores/timestamps only — never frames, landmarks or per-pixel data (blendshapes are 52 named floats: allowed, they are not biometric templates); expression is **best-effort mood, never a fact and never a lie detector** — wording "wirkt …"; opt-in default off; upstream `face_recognition/` untouched. Run all commands from the fork root `face_recognition/`. Python: `uv run --project apps/face2ai …` (bare `python3` is shimmed on this Mac).

---

### Task 0: Dependencies, model asset, settings

**Files:**
- Modify: `apps/face2ai/pyproject.toml` (optional extra `expression`)
- Create: `apps/face2ai/scripts/fetch-expression-models.sh`
- Modify: `apps/face2ai/src/face2ai_app/config.py`
- Test: `apps/face2ai/tests/test_config.py` (new)

**Step 1: Write the failing test**

```python
# apps/face2ai/tests/test_config.py
from pathlib import Path

from face2ai_app.config import Settings


def test_expression_settings_have_safe_defaults(monkeypatch):
    monkeypatch.delenv("FACE2AI_EXPRESSION_ENABLED", raising=False)
    monkeypatch.delenv("FACE2AI_EXPRESSION_MODELS_DIR", raising=False)
    settings = Settings.from_env()
    assert settings.expression_enabled is False  # opt-in
    assert settings.expression_models_dir == Path.home() / ".face2ai" / "models"
    assert settings.mood_stable_ticks == 3
    assert settings.mood_min_score == 0.5


def test_expression_settings_read_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE2AI_EXPRESSION_ENABLED", "true")
    monkeypatch.setenv("FACE2AI_EXPRESSION_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("FACE2AI_MOOD_STABLE_TICKS", "5")
    settings = Settings.from_env()
    assert settings.expression_enabled is True
    assert settings.expression_models_dir == tmp_path
    assert settings.mood_stable_ticks == 5
```

**Step 2: Run test to verify it fails**

Run: `uv run --project apps/face2ai pytest apps/face2ai/tests/test_config.py -q`
Expected: FAIL with `TypeError`/`AttributeError` (no `expression_enabled`)

**Step 3: Write minimal implementation**

`apps/face2ai/src/face2ai_app/config.py` — add fields + env parsing (keep existing `__post_init__` validation, add `mood_stable_ticks >= 1`, `0 < mood_min_score <= 1`):

```python
    expression_enabled: bool = False
    expression_models_dir: Path = Path.home() / ".face2ai" / "models"
    mood_stable_ticks: int = 3
    mood_min_score: float = 0.5
```
```python
            expression_enabled=os.getenv("FACE2AI_EXPRESSION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
            expression_models_dir=Path(os.getenv("FACE2AI_EXPRESSION_MODELS_DIR", str(Path.home() / ".face2ai" / "models"))).expanduser(),
            mood_stable_ticks=int(os.getenv("FACE2AI_MOOD_STABLE_TICKS", "3")),
            mood_min_score=float(os.getenv("FACE2AI_MOOD_MIN_SCORE", "0.5")),
```

`apps/face2ai/pyproject.toml`:
```toml
[project.optional-dependencies]
recognition = ["face-recognition>=1.3,<2", "setuptools<82"]
expression = ["mediapipe==0.10.21", "emotiefflib==1.1.1", "onnxruntime>=1.20,<2"]
```

`apps/face2ai/scripts/fetch-expression-models.sh`:
```bash
#!/usr/bin/env bash
# Fetch the MediaPipe Face Landmarker asset (3.7 MB) into $FACE2AI_EXPRESSION_MODELS_DIR (default ~/.face2ai/models).
# EmotiEffLib downloads its ONNX model on first use into its own cache.
set -euo pipefail
DIR="${FACE2AI_EXPRESSION_MODELS_DIR:-$HOME/.face2ai/models}"
mkdir -p "$DIR"
[ -f "$DIR/face_landmarker.task" ] || curl -sS -L -o "$DIR/face_landmarker.task" \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
ls -la "$DIR/face_landmarker.task"
```

**Step 4: Run test to verify it passes**

Run: `uv run --project apps/face2ai pytest apps/face2ai/tests/test_config.py -q`
Expected: `2 passed`

**Step 5: Commit**

```bash
git add apps/face2ai/pyproject.toml apps/face2ai/scripts/fetch-expression-models.sh apps/face2ai/src/face2ai_app/config.py apps/face2ai/tests/test_config.py
git commit -m "feat(face2ai): expression settings + optional extra + model fetch script"
```

---

### Task 1: Domain model `Expression` on `FaceObservation` and presence mood fields

**Files:**
- Modify: `apps/face2ai/src/face2ai_app/domain/models.py`
- Test: `apps/face2ai/tests/test_models.py` (new)

**Step 1: Write the failing test**

```python
# apps/face2ai/tests/test_models.py
import pytest
from pydantic import ValidationError

from face2ai_app.domain.models import EMOTIONS, Expression, FaceBox, FaceObservation, Presence, PresenceTransition


def test_expression_shape_and_bounds():
    e = Expression(dominant="Happiness", scores={"Happiness": 0.9, "Neutral": 0.1}, valence=0.7, arousal=0.2,
                   blendshapes={"mouthSmileLeft": 0.95}, yaw=3.0, pitch=-2.0, roll=0.5)
    assert e.dominant == "Happiness" and set(e.scores) <= set(EMOTIONS)
    with pytest.raises(ValidationError):
        Expression(dominant="Happiness", scores={"Bogus": 1.0})
    with pytest.raises(ValidationError):
        Expression(dominant="Happiness", scores={"Happiness": 1.0}, valence=2.0)


def test_face_observation_expression_is_optional_and_presence_carries_mood():
    box = FaceBox(top=1, right=2, bottom=3, left=0)
    assert FaceObservation(box=box).expression is None
    p = Presence(state="KNOWN", mood="Happiness", valence=0.5, arousal=0.1)
    assert p.mood == "Happiness"
    t = PresenceTransition(at="2026-08-18T12:00:00Z", from_state="NO_FACE", to_state="KNOWN")
    assert "mood" not in t.model_dump()  # transitions carry no mood; the ended mood rides the mood event
    assert EMOTIONS == ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
```

**Step 2: Run test to verify it fails**

Run: `uv run --project apps/face2ai pytest apps/face2ai/tests/test_models.py -q`
Expected: FAIL `ImportError: cannot import name 'EMOTIONS'`

**Step 3: Write minimal implementation** (add to `domain/models.py`, above `FaceObservation`)

```python
EMOTIONS = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")


class Expression(BaseModel):
    """Best-effort facial expression for one face — a mood hint, never a fact, never authentication.

    Wire-safe by construction: labels, scores in 0..1, valence/arousal in -1..1, named blendshape
    intensities and head pose angles. No landmarks, no pixels, no embeddings.
    """

    dominant: str
    scores: dict[str, float] = Field(default_factory=dict)
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    arousal: float | None = Field(default=None, ge=-1.0, le=1.0)
    blendshapes: dict[str, float] = Field(default_factory=dict)  # only entries >= 0.2, rounded to 2 decimals
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None

    @field_validator("scores")
    @classmethod
    def _known_labels(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - set(EMOTIONS)
        if unknown:
            raise ValueError(f"unknown emotion labels: {sorted(unknown)}")
        return value

    @field_validator("dominant")
    @classmethod
    def _dominant_known(cls, value: str) -> str:
        if value not in EMOTIONS:
            raise ValueError(f"unknown dominant emotion: {value}")
        return value
```
Add `expression: Expression | None = None` to `FaceObservation`; add `mood: str | None = None`, `valence: float | None = None`, `arousal: float | None = None` to `Presence` (not to `PresenceTransition`: a transition starts a fresh, mood-less presence; the ended mood rides the `mood` event); add `from pydantic import BaseModel, Field, field_validator`.

**Step 4: Run test to verify it passes** — `uv run --project apps/face2ai pytest apps/face2ai/tests/test_models.py apps/face2ai/tests/test_presence.py -q` → all pass (update `test_wire_models_carry_no_biometrics` expected key sets to include `mood`, `valence`, `arousal`).

**Step 5: Commit** — `git commit -am "feat(face2ai): Expression domain model + mood fields on presence"`

---

### Task 2: Port `ExpressionEngine` + `NullExpressionEngine`

**Files:**
- Create: `apps/face2ai/src/face2ai_app/ports/expression.py`
- Create: `apps/face2ai/src/face2ai_app/adapters/null_expression.py`
- Test: `apps/face2ai/tests/test_expression_port.py`

**Step 1: Failing test**
```python
from face2ai_app.adapters.null_expression import NullExpressionEngine
from face2ai_app.domain.models import FaceBox


def test_null_engine_is_unavailable_and_returns_none_per_box():
    engine = NullExpressionEngine("not installed")
    assert engine.available is False and engine.availability_reason == "not installed"
    assert engine.analyze(b"jpeg", [FaceBox(top=0, right=1, bottom=1, left=0)]) == [None]
```
**Step 2:** run → FAIL (module missing).
**Step 3:** implement
```python
# ports/expression.py
from typing import Protocol
from face2ai_app.domain.models import Expression, FaceBox

class ExpressionEngine(Protocol):
    @property
    def available(self) -> bool: ...
    @property
    def availability_reason(self) -> str | None: ...
    def analyze(self, image_bytes: bytes, boxes: list[FaceBox]) -> list[Expression | None]:
        """One Expression (or None) per box, same order. Boxes are in the pixel space of image_bytes."""
```
```python
# adapters/null_expression.py
class NullExpressionEngine:
    def __init__(self, reason: str = "expression engine disabled") -> None:
        self._reason = reason
    @property
    def available(self) -> bool: return False
    @property
    def availability_reason(self) -> str | None: return self._reason
    def analyze(self, image_bytes, boxes): return [None for _ in boxes]
```
**Step 4:** run → PASS. **Step 5:** commit `feat(face2ai): ExpressionEngine port + null adapter`.

---

### Task 3: MediaPipe + EmotiEffLib adapter (pure helpers first, then the engine)

**Files:**
- Create: `apps/face2ai/src/face2ai_app/adapters/mediapipe_expression.py`
- Test: `apps/face2ai/tests/test_expression_adapter.py`

Pure helpers (testable without models): `match_faces(boxes, landmark_bboxes) -> list[int|None]` (best IoU ≥ 0.2), `pose_from_matrix(m4x4) -> (yaw, pitch, roll) degrees`, `softmax_scores(logits8) -> dict`, `compact_blendshapes(pairs, threshold=0.2)`, `crop_with_margin(arr, box, margin=0.2)`.

**Step 1: Failing tests**
```python
import numpy as np
from face2ai_app.adapters.mediapipe_expression import compact_blendshapes, match_faces, pose_from_matrix, softmax_scores, crop_with_margin
from face2ai_app.domain.models import FaceBox

def test_softmax_scores_maps_logits_to_labels():
    scores = softmax_scores([0, 0, 0, 0, 5, 0, 0, 0])
    assert max(scores, key=scores.get) == "Happiness" and abs(sum(scores.values()) - 1) < 1e-6

def test_pose_from_identity_is_zero():
    yaw, pitch, roll = pose_from_matrix(np.eye(4))
    assert (round(yaw), round(pitch), round(roll)) == (0, 0, 0)

def test_match_faces_by_iou():
    boxes = [FaceBox(top=100, right=400, bottom=300, left=200)]
    assert match_faces(boxes, [(210, 110, 390, 290)]) == [0]     # (left, top, right, bottom)
    assert match_faces(boxes, [(0, 0, 10, 10)]) == [None]

def test_compact_blendshapes_filters_and_rounds():
    assert compact_blendshapes([("mouthSmileLeft", 0.951), ("browDownLeft", 0.05), ("_neutral", 0.3)]) == {"mouthSmileLeft": 0.95}

def test_crop_with_margin_clamps_to_image():
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_with_margin(arr, FaceBox(top=0, right=200, bottom=100, left=0), margin=0.2)
    assert crop.shape == (100, 200, 3)
```
**Step 2:** run → FAIL. **Step 3:** implement helpers + engine:

```python
class MediaPipeExpressionEngine:
    """Blendshapes + head pose from MediaPipe Face Landmarker, emotions + valence/arousal from EmotiEffLib.
    Lazy, CPU only; unavailable (never crashing) when the extra or the model asset is missing."""

    def __init__(self, models_dir: Path) -> None:
        self._reason = None; self._landmarker = None; self._recognizer = None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
        except Exception as exc:
            self._reason = f"expression extra not installed: {type(exc).__name__}: {exc}"; return
        asset = models_dir / "face_landmarker.task"
        if not asset.exists():
            self._reason = f"missing model asset {asset} (run scripts/fetch-expression-models.sh)"; return
        try:
            base = mp_python.BaseOptions(model_asset_path=str(asset), delegate=mp_python.BaseOptions.Delegate.CPU)
            self._landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base, output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                num_faces=4, running_mode=vision.RunningMode.IMAGE))
            self._recognizer = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_va_mtl", device="cpu")
            self._mp = mp
        except Exception as exc:
            self._reason = f"expression engine failed to initialize: {type(exc).__name__}: {exc}"

    available -> self._landmarker is not None; availability_reason -> self._reason

    def analyze(self, image_bytes, boxes):
        if not self.available or not boxes: return [None for _ in boxes]
        arr = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
        h, w = arr.shape[:2]
        lm_boxes = [(min(p.x for p in lms) * w, min(p.y for p in lms) * h, max(p.x for p in lms) * w, max(p.y for p in lms) * h) for lms in result.face_landmarks]
        matches = match_faces(boxes, lm_boxes)
        out = []
        for box, idx in zip(boxes, matches):
            crop = crop_with_margin(arr, box, margin=0.2)
            _, logits = self._recognizer.predict_emotions(crop, logits=True)     # (1, 10)
            row = np.asarray(logits)[0]
            scores = softmax_scores(row[:8].tolist()); valence = float(np.clip(row[8], -1, 1)); arousal = float(np.clip(row[9], -1, 1))
            blend, pose = {}, (None, None, None)
            if idx is not None:
                blend = compact_blendshapes([(c.category_name, c.score) for c in result.face_blendshapes[idx]])
                pose = pose_from_matrix(np.asarray(result.facial_transformation_matrixes[idx]))
            out.append(Expression(dominant=max(scores, key=scores.get), scores={k: round(v, 3) for k, v in scores.items()},
                                  valence=round(valence, 3), arousal=round(arousal, 3), blendshapes=blend,
                                  yaw=None if pose[0] is None else round(pose[0], 1), pitch=..., roll=...))
        return out
```
(`pose_from_matrix`: rotation R = m[:3,:3]; yaw = atan2(R[0,2], R[2,2]), pitch = asin(-R[1,2]), roll = atan2(R[1,0], R[1,1]) → degrees. `softmax_scores`: numerically stable softmax over EMOTIONS order = `idx_to_emotion_class` order verified above.)

**Step 4:** run helper tests → PASS. Real-engine smoke (manual, needs extra + asset): `uv sync --project apps/face2ai --group dev --extra recognition --extra expression && bash apps/face2ai/scripts/fetch-expression-models.sh && uv run --project apps/face2ai python -c "from pathlib import Path; from face2ai_app.adapters.mediapipe_expression import MediaPipeExpressionEngine as E; from face2ai_app.domain.models import FaceBox; e=E(Path.home()/'.face2ai/models'); print(e.available, e.availability_reason); print(e.analyze(open('examples/obama.jpg','rb').read(), [FaceBox(top=142,right=617,bottom=409,left=349)]))"` → expect `dominant='Happiness'`, valence > 0.5, `mouthSmileLeft` in blendshapes.
**Step 5:** commit `feat(face2ai): MediaPipe+EmotiEffLib expression adapter`.

---

### Task 4: `IdentityService` attaches expressions; runtime opt-in; status

**Files:**
- Modify: `apps/face2ai/src/face2ai_app/services/identity_service.py`
- Modify: `apps/face2ai/src/face2ai_app/main.py` (create engine from settings; `app.state.expression = ExpressionState(enabled=settings.expression_enabled)`)
- Modify: `apps/face2ai/src/face2ai_app/api/routes.py` (`POST /api/expression {"enabled": bool}`, `SystemStatus.expression_available/expression_reason/expression_enabled`)
- Modify: `apps/face2ai/src/face2ai_app/domain/models.py` (`SystemStatus` fields)
- Modify: `apps/face2ai/tests/conftest.py` (`FakeExpressionEngine`, `create_app(..., expression=...)`)
- Test: `apps/face2ai/tests/test_api.py` (add), `apps/face2ai/tests/test_expression_api.py` (new)

**Step 1: Failing test**
```python
def test_recognize_attaches_expression_only_when_enabled(client, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.expressions = [Expression(dominant="Happiness", scores={"Happiness": 0.9}, valence=0.6, arousal=0.1)]
    off = client.post("/api/recognize", content=b"frame", headers=image_headers()).json()
    assert off["faces"][0]["expression"] is None                     # opt-in default off
    assert client.post("/api/expression", json={"enabled": True}).json() == {"enabled": True, "available": True}
    on = client.post("/api/recognize", content=b"frame", headers=image_headers()).json()
    assert on["faces"][0]["expression"]["dominant"] == "Happiness"
    status = client.get("/api/status").json()
    assert status["expression_available"] is True and status["expression_enabled"] is True

def test_expression_toggle_refused_when_engine_unavailable(client):
    client.app.state.expression_engine = NullExpressionEngine("not installed")
    r = client.post("/api/expression", json={"enabled": True})
    assert r.status_code == 409 and "not installed" in r.json()["detail"]
```
**Step 2:** run → FAIL. **Step 3:** implement: `IdentityService.__init__(…, expression: ExpressionEngine | None = None)`; in `recognize()` after building observations: `if self.expression is not None and self.expression.available and self.expression_enabled: for obs, expr in zip(observations, self.expression.analyze(image_bytes, [f.box for f in detected])): obs.expression = expr` (wrap analyze in try/except → log + leave None; expression must never break recognition). Keep `expression_enabled` as a mutable attribute on the service (set by the route). Route:
```python
class ExpressionToggle(BaseModel):
    enabled: bool

@router.post("/api/expression")
async def set_expression(request: Request, body: ExpressionToggle) -> dict:
    service = _service(request)
    engine = service.expression
    if body.enabled and (engine is None or not engine.available):
        raise HTTPException(status_code=409, detail=f"expression engine unavailable: {getattr(engine, 'availability_reason', 'not configured')}")
    service.expression_enabled = body.enabled
    return {"enabled": service.expression_enabled, "available": bool(engine and engine.available)}
```
`SystemStatus`: `expression_available: bool = False`, `expression_reason: str | None = None`, `expression_enabled: bool = False`. `main.py`: `expression_engine = expression or (MediaPipeExpressionEngine(app_settings.expression_models_dir) if app_settings.expression_enabled or _extra_present() else NullExpressionEngine("set FACE2AI_EXPRESSION_ENABLED=true or toggle via POST /api/expression"))` — simpler: always construct `MediaPipeExpressionEngine` lazily (it is unavailable-not-crashing when the extra is missing) and start with `enabled = settings.expression_enabled and engine.available`.
**Step 4:** run → PASS (also existing 44). **Step 5:** commit `feat(face2ai): expression opt-in on recognize + status + toggle route`.

---

### Task 5: `MoodTracker` — stable mood from per-frame expressions

**Files:**
- Create: `apps/face2ai/src/face2ai_app/services/mood.py`
- Test: `apps/face2ai/tests/test_mood.py`

Semantics: input `(presence_key, expression | None, now)`; EMA (alpha 0.5) over `scores`; candidate = argmax of EMA; commit when candidate == previous candidate for `stable_ticks` frames AND EMA[candidate] >= `min_score`; commit only if different from current mood; on presence key change (identity/state changes, or None expression for `stable_ticks` frames) → reset to `None` (emit `mood -> None`). Output `MoodTransition(at, identity_id, display_name, from_mood, to_mood, valence, arousal)`; `current()` returns `(mood, valence, arousal)` for `Presence`.

**Step 1: Failing tests**
```python
from face2ai_app.services.mood import MoodTracker
from face2ai_app.domain.models import Expression

def happy(v=0.6): return Expression(dominant="Happiness", scores={"Happiness": 0.9, "Neutral": 0.1}, valence=v, arousal=0.1)
def sad(): return Expression(dominant="Sadness", scores={"Sadness": 0.8, "Neutral": 0.2}, valence=-0.5, arousal=-0.2)

def test_mood_commits_after_stable_ticks_and_ignores_flicker():
    t = MoodTracker(stable_ticks=3, min_score=0.5)
    assert t.observe("KNOWN:a", happy(), T0) is None
    assert t.observe("KNOWN:a", sad(), T0) is None          # flicker resets the streak
    assert t.observe("KNOWN:a", happy(), T0) is None
    assert t.observe("KNOWN:a", happy(), T0) is None
    tr = t.observe("KNOWN:a", happy(), T0)
    assert tr is not None and (tr.from_mood, tr.to_mood) == (None, "Happiness") and tr.valence > 0
    assert t.current() == ("Happiness", tr.valence, tr.arousal)

def test_mood_switch_and_reset_on_presence_change():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    for _ in range(2): t.observe("KNOWN:a", happy(), T0)
    for _ in range(1): assert t.observe("KNOWN:a", sad(), T0) is None
    tr = t.observe("KNOWN:a", sad(), T0)
    assert tr is not None and (tr.from_mood, tr.to_mood) == ("Happiness", "Sadness")
    reset = t.observe("NO_FACE:", None, T0)
    assert reset is not None and reset.to_mood is None and t.current() == (None, None, None)

def test_low_confidence_never_commits():
    t = MoodTracker(stable_ticks=2, min_score=0.5)
    weak = Expression(dominant="Neutral", scores={"Neutral": 0.4, "Happiness": 0.3, "Sadness": 0.3})
    assert all(t.observe("KNOWN:a", weak, T0) is None for _ in range(5))
```
**Step 2:** FAIL. **Step 3:** implement (pure, `threading.Lock`, mirrors `PresenceTracker` style). **Step 4:** PASS. **Step 5:** commit `feat(face2ai): MoodTracker with hysteresis`.

---

### Task 6: Wire mood into presence + SSE

**Files:**
- Modify: `apps/face2ai/src/face2ai_app/services/presence.py` (`PresenceTracker.set_mood(mood, valence, arousal)`; `snapshot()` includes them; `_to_no_signal`/`_commit` clear mood)
- Modify: `apps/face2ai/src/face2ai_app/main.py` (`app.state.mood = MoodTracker(settings.mood_stable_ticks, settings.mood_min_score)`)
- Modify: `apps/face2ai/src/face2ai_app/api/routes.py` (`recognize`: after presence observe → `mood_tracker.observe(key, primary_expression, now)` → publish `"mood"` event with `MoodTransition`; `Presence` snapshot carries mood)
- Modify: `apps/face2ai/src/face2ai_app/domain/models.py` (`MoodTransition` model)
- Test: `apps/face2ai/tests/test_events_api.py` (add `test_mood_events_and_presence_mood`)

**Step 1: Failing test** (live-uvicorn fixture, `presence_stable_ticks=1`, add `mood_stable_ticks=1` to the fixture Settings):
```python
def test_mood_events_and_presence_mood(live, fake_engine, fake_expression, face):
    fake_engine.faces = [face]
    fake_expression.expressions = [Expression(dominant="Happiness", scores={"Happiness": 0.9}, valence=0.6, arousal=0.1)]
    live.client.post("/api/expression", json={"enabled": True})
    live.client.post("/api/recognize", content=b"frame", headers=HEADERS)
    presence = live.client.get("/api/presence").json()
    assert presence["mood"] == "Happiness" and presence["valence"] == 0.6
    frames = live.sse("/api/events?after=0", wanted=3, skip_heartbeats=True)
    assert [f["event"] for f in frames] == ["hello", "presence", "mood"]
    assert frames[2]["data"]["to_mood"] == "Happiness"
    assert set(frames[2]["data"]) == {"sequence", "at", "identity_id", "display_name", "from_mood", "to_mood", "valence", "arousal"}
```
Also extend `test_events_stream_carries_only_the_documented_keys` expected sets with `mood, valence, arousal`.
**Step 2:** FAIL. **Step 3:** implement (mood key = `f"{presence.state}:{identity_id or ''}"`; expression = first face's expression when exactly one face). **Step 4:** PASS. **Step 5:** commit `feat(face2ai): mood on presence + SSE mood events`.

---

### Task 7: Browser — opt-in toggle, mood chip, valence/arousal, events

**Files:**
- Modify: `apps/face2ai/src/face2ai_app/static/js/model.js` (`describeExpression(expr, lang)` → `{label:"looks happy" (en) / "wirkt fröhlich" (de), tone, valence, arousal}`; bilingual wording tables for the 8 labels — the browser shell is English and passes `'en'`; the German hedged wording lives in the voice agent / Hermes plugin)
- Modify: `apps/face2ai/src/face2ai_app/static/js/api.js` (`setExpression(enabled)`)
- Modify: `apps/face2ai/src/face2ai_app/static/index.html` (metric tile "Expression · a hint, not a fact" + tiny "Valence"/"Arousal" bars; action button `expressionButton` "Expression: off/on" disabled when unavailable)
- Modify: `apps/face2ai/src/face2ai_app/static/js/app.js` (render expression per event; toggle handling; `status.expression_*` → button state; event-stream entries on mood transitions come from `transitionKey` including mood? — keep separate: log "Mood: <Name> looks happy." only when the stable label changes)
- Modify: `apps/face2ai/src/face2ai_app/static/css/app.css` (`.mood-bar` styles)
- Test: `apps/face2ai/tests/js/model.test.mjs` (add `describeExpression` cases), `apps/face2ai/tests/test_static.py` (add "expression never presented as certainty": no "erkannt"/"is happy"/"detected mood" wording; served app.js uses "looks …", model.js keeps both "wirkt "/"looks " tables)

**Step 1: Failing JS test**
```js
test('describeExpression speaks in hedged German and never claims certainty', () => {
  const d = describeExpression({ dominant: 'Happiness', scores: { Happiness: 0.9 }, valence: 0.6, arousal: 0.1 }, 'de');
  assert.equal(d.label, 'wirkt fröhlich');
  assert.equal(d.tone, 'ok');
  assert.equal(describeExpression(null, 'de'), null);
  assert.equal(describeExpression({ dominant: 'Neutral', scores: {} }, 'en').label, 'looks neutral');
});
```
**Step 2:** `node --test 'apps/face2ai/tests/js/**/*.test.mjs'` → FAIL. **Step 3:** implement model.js + UI wiring (button hits `POST /api/expression`, refreshes status; expression tile shows label + two 60 px bars mapping -1..1 → 0..100 %). **Step 4:** node tests + `node --check` + `test_static.py` PASS; browser smoke in Chrome (toggle on, obama.jpg via curl → tile shows "looks happy"; the browser is English — German hedged wording lives in the voice agent / Hermes plugin). **Step 5:** commit `feat(face2ai): expression opt-in + mood UI`.

---

### Task 8: Consumers — voice agent and Hermes plugin wording

**Files:**
- Modify: `apps/face2ai-agent/src/face2ai_agent/presence.py` (`Presence.mood/valence/arousal` from payload; `describe()` appends "wirkt fröhlich (Valenz +0.6)" when mood present; heartbeat/hello carry it)
- Modify: `apps/face2ai-agent/src/face2ai_agent/policy.py` (`build_instructions` rule: "mood is a hint, never state it as fact, do not psychoanalyse")
- Modify: `apps/face2ai-agent/tests/test_presence.py` (+2 tests)
- Modify: `apps/face2ai-hermes-plugin/face2ai/presence.py` (same fields + wording), `apps/face2ai-hermes-plugin/tests/test_presence.py` (+1 test)
- Both consumers also handle SSE event `mood` (update current mood without a presence transition).

**Step 1: Failing tests**
```python
def test_describe_includes_hedged_mood():
    memory = PresenceMemory()
    memory.apply_hello({"presence": {"state": "KNOWN", "display_name": "Ben", "identity_id": "a", "mood": "Happiness", "valence": 0.6, "arousal": 0.1}})
    text = memory.describe(T0)
    assert "wirkt fröhlich" in text and "0.6" in text
```
**Step 2:** FAIL. **Step 3:** implement (German map: Happiness→fröhlich, Sadness→traurig, Anger→verärgert, Fear→ängstlich, Surprise→überrascht, Disgust→angewidert, Contempt→abschätzig, Neutral→neutral). **Step 4:** PASS (agent 26+, plugin 12+). **Step 5:** commit `feat(agent,plugin): hedged mood wording from presence`.

---

### Task 9: Docs — ADR-003, README, AGENTS, VALIDATION, UI_DIRECTION, CI

**Files:**
- Create: `apps/face2ai/docs/architecture/ADR-003-expression-engine.md`
- Modify: `apps/face2ai/README.md`, `apps/face2ai/AGENTS.md`, `apps/face2ai/docs/boilerplate/VALIDATION.md`, `apps/face2ai/docs/UI_DIRECTION.md`, `.github/workflows/face2ai.yml`

ADR-003 content: context (mood/micro-expression request; imentiv evaluated — cloud, batch, stores frames → rejected for the live loop); decision (local MediaPipe+EmotiEffLib behind `ExpressionEngine`, opt-in, wire = labels/scores/blendshape floats/pose only, hedged wording, no persistence, per-session toggle); honesty section ("micro-expressions" in the strict CASME sense are out of scope; we deliver expression + intensity + dynamics); legal note (EU AI Act Art. 5 prohibits emotion recognition at workplace/education — Face2AI is a private local tool; keep opt-in and no storage); consequences; review triggers.
AGENTS.md rules to add: expression is a hint (wording "wirkt"), never a fact, never used to gate anything; `domain/` must not import mediapipe/emotiefflib; the wire may carry named blendshape floats but never landmarks/pixels.
VALIDATION gate: `POST /api/expression {"enabled":true}` → smile at the camera → tile "looks happy" within 3 ticks (browser is English; the voice agent / Hermes plugin answer in hedged German "wirkt …"); frown → changes; `mood` event in stream; toggle off → `expression: null` again.
CI: `app-shell` job installs `--extra expression`? No (heavy). Add a small job step "expression helpers" running `pytest apps/face2ai/tests/test_expression_adapter.py` (pure helpers, no models) — already covered by the main pytest run; just ensure imports are lazy so CI without the extra passes.

**Step:** write docs, run full gates (`pytest` 44+ new, `node --test`, `node --check`, `compileall`), commit `docs(face2ai): ADR-003 expression engine + gates`.

---

### Task 10: Live verification + push

1. `uv sync --project apps/face2ai --group dev --extra recognition --extra expression` (mediapipe pulls opencv-contrib + matplotlib; ~200 MB) and `bash apps/face2ai/scripts/fetch-expression-models.sh`.
2. Restart backend; `curl -X POST :8765/api/expression -d '{"enabled":true}' -H 'content-type: application/json'` → `{"enabled": true, "available": true}`; `curl … examples/obama.jpg /api/recognize` → `faces[0].expression.dominant == "Happiness"`, `blendshapes.mouthSmileLeft ≈ 0.95`; second frame → `/api/presence.mood == "Happiness"`; SSE shows `mood`.
3. Browser: toggle "Expression: on", tile "looks happy", event entry "Mood: … looks happy." (English shell, hedged). Voice agent `smoke "Wie wirke ich gerade?"` → hedged German answer ("wirkt …"). Hermes: `/presence` shows mood line.
4. Timing: log per-frame `analyze` ms (expect ≤ 40 ms for one face incl. decode) — if the loop exceeds ~200 ms on the M1, downscale the crop to 224 px before EmotiEffLib (it resizes internally anyway).
5. Commit, push, PR #2 comment with measured numbers.

**Out of scope (Stage 2):** temporal dynamics from blendshape time series (onset/apex/offset per action, "micro" event detection), valence timeline in the pane, per-person mood history, Hermes plugin pane sparkline.
