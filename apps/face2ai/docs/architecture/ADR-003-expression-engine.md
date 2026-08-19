# ADR-003 - Local expression hints (Stage 1) beside recognition, opt-in, hedged, never gating

Status: accepted for Stage 1 (2026-08-18)
Supersedes nothing; extends ADR-001 (local modular monolith) and ADR-002 (event consumers).

## Context

The user asked for "micro-expression / emotion recognition" next to identity: Face2AI should be
able to say how the person in front of the camera *appears*, and the voice agent (ADR-002) and the
Hermes plugin should be able to mention it.

Constraints that already bind the product: local-first, no frames leave the machine, no raw frame
persistence, expression must not become part of matching or enrollment (AGENTS.md), and nothing
the UI shows may pretend to be a finding.

**imentiv.ai was evaluated and rejected for the live loop**: it is a cloud batch API (upload
video/images, poll for a report) that stores the frames on the provider's side. That breaks the
local-first rule of ADR-001, cannot serve a 450 ms camera loop, and would put third-party retention
of face imagery into a tool whose promise is "nothing leaves the Mac".

## Decision

1. **A second port beside `RecognitionEngine`: `ExpressionEngine`** (`ports/expression.py`:
   `available`, `availability_reason`, `analyze(image_bytes, boxes) -> list[Expression | None]`).
   The one adapter is `adapters/mediapipe_expression.py`: MediaPipe Face Landmarker (CPU delegate,
   52 blendshapes + facial transformation matrix → yaw/pitch/roll) plus EmotiEffLib
   `enet_b0_8_va_mtl` (ONNX; 8 emotion classes Anger/Contempt/Disgust/Fear/Happiness/Neutral/
   Sadness/Surprise + valence/arousal). Landmark faces are matched to the recognition boxes by IoU;
   the emotion model runs on a margin-expanded crop of each box. `create_app` always constructs
   this adapter; when the `expression` extra or the model asset is missing it reports
   `available=False` with a reason and never crashes. `NullExpressionEngine` is the explicit
   stand-in for callers that want no engine (tests, embedding without the extra); callers that
   inject nothing get `MediaPipeExpressionEngine`, which reports itself unavailable when the extra
   or asset is missing. Failures inside `analyze` are logged once at warning, then debug —
   recognition continues without expressions.
2. **Opt-in, default off, per session.** `POST /api/expression {"enabled": bool}` toggles the
   feature at runtime (409 when enabling while the engine is unavailable; disabling is always
   accepted, 200); `FACE2AI_EXPRESSION_ENABLED=true` only pre-enables an *available* engine at
   startup. `/api/status` reports `expression_available` / `expression_reason` /
   `expression_enabled`; the browser toggle stays disabled until the status says the engine is
   available.
3. **Per-frame hint on the recognize response**: `IdentityService.recognize()` attaches an
   `Expression` to each `faces[]` observation *after* matching. Expression never takes part in
   matching, enrollment, presence, or the greeting decision.
4. **Stable mood on the wire via `MoodTracker`** (`services/mood.py`): EMA over the 8 scores
   (alpha 0.5, zero-start), candidate = argmax; committed when the candidate has led for
   `FACE2AI_MOOD_STABLE_TICKS` (3) consecutive frames and its EMA is ≥ `FACE2AI_MOOD_MIN_SCORE`
   (0.5) at that tick. Valence/arousal are frozen at commit time (one change per mood, not per
   frame). The mood follows the stable presence (state + identity key): a key change,
   `stable_ticks` frames without a usable expression, presence expiry/reset, or toggling the
   feature off end it. The result decorates `Presence.mood/valence/arousal` and is published as
   SSE event `mood` (`from_mood` → `to_mood`, `to_mood: null` = ended). A mood-end caused by a
   presence transition is always published *after* that presence event, so consumers see a
   symmetric wire.
5. **Wire discipline**: `Expression` carries labels, scores in 0..1, valence/arousal in -1..1, named
   blendshape floats (only ≥ 0.2, 2 decimals) and three pose angles. **No landmarks, no crops, no
   pixels, no embeddings** — the model forbids it by construction. `MoodTransition` is coarser
   still: label, names, timestamp, two rounded scalars. Face2AI itself persists nothing about
   expression; the Hermes plugin mirrors the *current* presence snapshot (state, name, coarse
   mood/valence/arousal) into its plugin state file for the dashboard process — overwritten, no
   history.
6. **Hedged wording everywhere.** The browser shell (English) says "looks happy" and labels the tile
   "a hint, not a fact"; the voice agent and the Hermes plugin (German) say "wirkt fröhlich (…) –
   nur ein Hinweis aus dem Gesichtsausdruck, keine Tatsache." Never "is happy", never "erkannt",
   never "detected" for a mood; `tests/test_static.py` enforces this for the served bundle. The
   browser's "Mood" log entries are a 3-frame debounce (`trackMood`) of the per-frame hints; the
   wire mood is the server's `MoodTracker`.

## Honesty about what this is

- **"Micro-expressions" in the strict sense (CASME: involuntary, < 500 ms) are out of scope.**
  Stage 1 delivers *expression* (an 8-class label with scores), *intensity* (valence, arousal,
  named blendshapes) and *head pose* per frame, and a debounced mood on top. Temporal dynamics
  (onset/apex/offset per action, "micro" event detection) are Stage 2 and would need a time series
  of blendshapes, not a per-frame API.
- The models are dataset-biased (AffectNet-style posed/annotated faces); results vary with
  lighting, angle, glasses, skin tone and culture. Per-frame flicker is expected; the hysteresis
  smooths it, it does not make it true.
- A label describes how a face *appears* to a model. It is not a feeling, not a diagnosis, not a
  lie detector, and never an input to any decision the app makes.

## Legal / ethics note

The EU AI Act, Art. 5(1)(f), prohibits AI systems that infer emotions of natural persons in the
areas of workplace and education institutions (medical/safety exceptions aside). Face2AI is a
private, local tool for the user's own machine and the user's own face; the feature is opt-in,
stores nothing, and sends nothing to third parties. It must stay that way, and it is not for use
on other people without their consent — the same posture ADR-001 takes for enrollment.

## Dependencies and platform facts (measured on the M1, 2026-08-18)

- Install extra: `uv sync --project apps/face2ai --group dev --extra expression` (with
  `--extra recognition` for the real loop). `mediapipe==0.10.21`, `emotiefflib==1.1.1`,
  `onnxruntime>=1.20,<2`; ≈ 0.75 GB installed (mediapipe pulls jax/jaxlib, opencv, matplotlib —
  the bulk is jaxlib 252 MB, cv2 120 MB, mediapipe 107 MB, onnxruntime 76 MB, scipy 70 MB,
  onnx 67 MB, matplotlib 28 MB).
- **mediapipe is pinned to 0.10.21**: 1.0.1's macOS wheel aborts in `DrishtiMetalHelper`; the
  adapter uses `BaseOptions.Delegate.CPU`.
- **numpy override**: mediapipe 0.10.21 declares `numpy<2`, so `[tool.uv] override-dependencies =
  ["numpy==2.3.5"]` in `pyproject.toml` keeps the project resolvable; the FaceLandmarker CPU
  delegate runs fine on numpy 2.3.5 empirically. Remove the override when mediapipe supports
  numpy 2 officially.
- Model assets: `scripts/fetch-expression-models.sh` downloads BOTH assets atomically and
  idempotently: `face_landmarker.task` (3.7 MB, float16) into `FACE2AI_EXPRESSION_MODELS_DIR`
  (default `FACE2AI_DATA_DIR/models`, i.e. `~/.face2ai/models`) and EmotiEffLib's
  `enet_b0_8_va_mtl.onnx` (16 MB, from the EmotiEffLib GitHub repo) into `~/.emotiefflib/` — the
  exact path emotiefflib 1.1.1 resolves (`os.path.expanduser("~")/.emotiefflib/<model>.onnx`; it
  has no local-path option). Startup does not touch the network: the adapter checks both files in
  `__init__` and reports `expression_available:false` with the missing path instead of letting
  emotiefflib run its own downloader.
- Timing, adapter alone (one face, full-frame decode + landmarker + EmotiEffLib,
  `examples/obama.jpg`, 910×1137): first measurement `analyze` ~146 ms cold, ~77 ms warm; a
  later warm run ~84 ms / ~27 ms — well inside the browser's 450 ms loop. Component figures
  from the scratch project: landmarker ~9 ms/frame, emotion + VA ~15 ms/face on a crop.
- Timing, live end to end (measured 2026-08-18 on the M1 against the running app: `POST
  /api/recognize` round trip via curl, dlib HOG + expression, one face):
  `examples/obama-480p.jpg` (853×480) ~130 ms without expression → ~150 ms with expression;
  `examples/obama.jpg` (910×1137) ~300 ms → ~330–380 ms. Expression adds ~30–50 ms per frame
  for one face.
- Env: `FACE2AI_EXPRESSION_ENABLED` (false), `FACE2AI_EXPRESSION_MODELS_DIR`,
  `FACE2AI_MOOD_STABLE_TICKS` (3), `FACE2AI_MOOD_MIN_SCORE` (0.5).

## Alternatives considered

- imentiv.ai (cloud batch, stores frames): rejected, see Context.
- Running the emotion model only (no landmarker): loses blendshapes/pose, which are the
  ingredients Stage 2 needs and the only per-action signal we can show honestly. Rejected.
- Making expression part of `RecognitionEngine.detect`: couples dlib and MediaPipe release
  cadence and blurs the "never gates matching" rule. Rejected — separate port.
- Live valence/arousal per frame on the wire: rejected for Stage 1 (noisy wire, agents would
  narrate flicker); values are frozen at mood commit. Review trigger below.

## Consequences

- (+) Every consumer gets a mood with two lines of wording; nothing biometric or pictorial is added
  to the wire; recognition, enrollment and greeting are untouched.
- (+) The default install (no extra) behaves exactly as before: `expression_available:false`,
  toggle disabled, `faces[].expression` null, no `mood` events. CI stays light.
- (−) An optional dependency set of ≈ 0.75 GB installed (mediapipe pulls jax/jaxlib, opencv,
  matplotlib) with a version pin and a numpy override that must be revisited; two model files
  that must be fetched by the script before the engine reports available (no startup download).
- Face2AI itself persists nothing about expression; the Hermes plugin mirrors the *current*
  presence snapshot (state, name, coarse mood/valence/arousal) into its plugin state file for the
  dashboard process — overwritten, no history.
- Shell and `/assets/*` are served with `Cache-Control: no-cache` (ETag revalidation) since
  2026-08-18 — a redeploy must never pair a fresh `app.js` with a heuristically cached `model.js`
  (this broke the shell once during verification).
- (−) A model that can be wrong in ways users may over-trust — mitigated by wording, tests on the
  wording, opt-in and the tile's "a hint, not a fact" label, but not eliminated.
- New surface: `ExpressionEngine` port + two adapters, `MoodTracker`, `POST /api/expression`,
  `SystemStatus.expression_*`, `FaceObservation.expression`, `Presence.mood/valence/arousal`,
  SSE `mood`; ~70 new tests (adapter helpers run without the extra).

## Review triggers

- Stage 2 temporal dynamics (blendshape time series, onset/apex/offset, per-person mood history,
  valence timeline in the pane) — needs its own decision on buffering per-frame data in memory.
- mediapipe officially supporting numpy 2 → drop the `override-dependencies` line and re-measure.
- An on-device GPU/Metal delegate that does not abort on macOS → re-measure and reconsider the CPU
  pin.
- Any use beyond the user's own face on the user's own machine → re-read the legal note first.
