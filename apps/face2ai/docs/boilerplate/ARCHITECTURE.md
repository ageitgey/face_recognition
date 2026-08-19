# Face2AI Boilerplate Architecture

```text
Browser camera + UI
      |
      | sampled JPEG over localhost
      v
FastAPI routes ---------------------------------------> PresenceTracker
      |                                                        |
      |                                     MoodTracker / ActionTracker / AffectHistory
      |                                     (hints only, ADR-003/004: SSE mood + action,
      |                                      GET /api/expression/timeline, in memory)
      v
IdentityService
   /       |        \
Recognition  IdentityStore  Expression
Port         Port           Port (opt-in, ADR-003)
  |           |               |
  v           v               v
face_recognition adapter   atomic JSON adapter   MediaPipe + EmotiEffLib adapter
  |                                                (CPU, hint only, never in matching)
  v
dlib / model package
```

## Boundary rules

- `domain/` has no FastAPI, `face_recognition`, mediapipe or emotiefflib imports.
- `ports/` defines replacement boundaries.
- `services/` owns matching and enrollment policy; `services/presence.py` turns per-frame RecognitionEvents into debounced presence transitions and `services/events.py` fans them (plus store, mood and action events) out to SSE subscribers; `services/mood.py` (`MoodTracker`) turns per-frame expression scores into a stable, hedged `Presence.mood` plus the live smoothed `Presence.valence/arousal`; `services/actions.py` (`ActionTracker`, ADR-004) turns the blendshape series into completed facial actions (hysteresis onset → apex → offset per action group) published as SSE `action`; `services/timeline.py` (`AffectHistory`, ADR-004) keeps those samples, mood changes and actions in three bounded in-memory ring buffers answered by `GET /api/expression/timeline` — cleared on `POST /api/presence/reset`, never persisted. All three decorate presence, none of them decides anything.
- `adapters/` owns third-party and persistence details.
- `api/` translates HTTP to application calls.
- `static/` owns camera, presentation, motion and browser speech: `api.js` (HTTP client), `camera.js` (capture + overlay drawing), `model.js` (pure view-model incl. box projection, action wording and sparkline points, node-tested), `events.js` (same-origin `EventSource` on `/api/events?role=browser`: mood + action entries come from the server, never re-derived), `effects.js` (motion), `app.js` (orchestration). Visual rules: `docs/UI_DIRECTION.md`.

## Extension points

Future features attach through new adapters or event consumers:

- Party Mirror: consume `RecognitionEvent`; generate comments through a future `InteractionAdapter`.
- Agent gateway: `GET /api/events` publishes identity transitions without exposing raw frames — the voice agent in `apps/face2ai-agent/` is the first consumer (ADR-002).
- Alternative recognition engine: implement `RecognitionEngine`.
- Alternative expression/affect model: implement `ExpressionEngine` (ADR-003); the wire stays labels/scores/blendshape floats/pose — never landmarks or pixels. Dynamics consumers (panes, timelines) subscribe to SSE `action` or read `GET /api/expression/timeline` (ADR-004) — they never gate behaviour on either.
- Alternative persistence: implement `IdentityStore`.

Do not put Party Mirror, agent, LLM, or cloud behavior into the recognition adapter, and do not let the expression adapter feed matching, enrollment or the greeting.
