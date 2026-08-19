# Face2AI Agent Instructions

## Goal

Preserve the shortest real product loop: `UNKNOWN -> explicit LEARN -> leave -> return -> KNOWN -> greeting`.

## Architecture rules

- Do not modify the upstream `face_recognition/` package for Face2AI product features unless a separate ADR explicitly requires it.
- `domain/` must not import FastAPI, browser code, storage adapters, `face_recognition`, `mediapipe` or `emotiefflib`.
- Third-party recognition belongs behind `RecognitionEngine`; third-party expression/affect models belong behind `ExpressionEngine` (`ports/expression.py`, ADR-003) — new adapters implement that port, they are never wired into matching.
- Persistence belongs behind `IdentityStore`.
- A port's obligations are written in its `Protocol` docstring and *enforced* by `tests/test_port_conformance.py`, which runs them against every adapter present in the environment — the `recognition-contract` CI job installs the real one so the suite cannot be green only against doubles. A new recognition adapter is finished when it passes that file: re-entrant `detect`, input bounds rejected before decoding, EXIF-upright coordinates, and only `InvalidFrame`/`RecognitionUnavailable` escaping. Anything that decodes pixels uses `adapters/frame_decode.decode_frame` rather than its own `Image.open` (the bound and the EXIF rule have one owner; two copies is how they drifted apart the first time).
- HTTP belongs in `api/`; browser behavior belongs in `static/` (ES modules, no build step; visual rules in `docs/UI_DIRECTION.md`).
- Browser view logic that needs no DOM lives in `static/js/model.js` and is covered by `tests/js/*.test.mjs` (`node --test`).
- New Party Mirror or agent behavior must consume identity/recognition events instead of being inserted into face matching. The event surface is `services/presence.py` (`PresenceTracker`, debounced transitions) + `services/events.py` (`IdentityEventBroker`) exposed as `GET /api/events` (SSE) and `GET /api/presence`; it carries states, names, counts and timestamps only — never frames, boxes or encodings (a test enforces this). It also carries no freshness flag: `Presence.observed_at` is the fact and `PresenceTracker.expire()` is the only server-side freshness rule. A consumer that wants a "no fresh frames" line owns its own budget; do not add a second threshold to the wire (one equal to `expire()`'s can never be `True`, a different one is a second owner for one rule).
- The Hermes plugin lives in `apps/face2ai-hermes-plugin/` (Python half + dashboard API for the Hermes host, desktop half for the Mac); it consumes the same event stream.
- The voice agent lives in `apps/face2ai-agent/` (LiveKit Agents, own uv project, ADR-002). It subscribes with `?role=agent`; while it is connected the browser leaves the spoken greeting to it (`SystemStatus.agent_connected`). Because that role is privileged, `GET /api/events?role=agent` is refused with 403 unless the request's `Sec-Fetch-Site` is `same-origin` or absent — httpx consumers never send the header, browsers always do, so no page from elsewhere can claim ownership and mute the greeting. Do not add a second privileged role without the same guard.
- Expression (ADR-003) is a **hint, never a fact, never a gate**: it may decorate observations (`faces[].expression`), presence (`Presence.mood/valence/arousal`) and the stream (SSE `mood`), but it must never influence matching, enrollment, presence transitions, the greeting or any other decision. It is opt-in (`POST /api/expression`, default off) and nothing about it is persisted.
- Expression *dynamics* (ADR-004) are hints under exactly the same rule: facial actions (`ActionTracker`, SSE `action`) and the affect timeline (`AffectHistory`, `GET /api/expression/timeline`) may be shown, logged and drawn, but they must never gate anything — no greeting, no enrollment, no auth, no state machine. **A consumer must not turn an `action` event into behaviour**: the voice agent ignores `action` frames by design (`run_presence_loop` has no branch for them; a test pins it) and the Hermes plugin keeps actions out of the LLM context. Adding a reaction to an action is a new ADR, not a patch.
- Nothing about expression dynamics is persisted: `AffectHistory` is three bounded in-memory ring buffers (samples/moods/actions), cleared by `POST /api/presence/reset` and gone on restart. No file, no database, no export — not even "just for debugging". The reset publishes SSE `timeline_cleared` when it dropped something: any consumer that mirrors the `mood`/`action` stream must forget on that frame, because the user's "forget" has to survive the hop to another process or machine. A `presence` → `NO_SIGNAL` frame is not the signal (an ordinary expiry publishes one and keeps the history).
- Expression wording is hedged everywhere: browser (English) "looks happy", agent and Hermes plugin (German) "wirkt fröhlich" — never "is happy", "ist fröhlich", "erkannt" or "detected" for a mood. Actions are described as movements with a duration — "brief smile (0.9 s)", "kurzes Lächeln (0.9 s)", "held smile" — never as a state of the person ("is smiling") and never with a cause ("smiled because …"). Timing is quantized to ~0.6 s, so the served bundle must never call these "micro-expressions". `tests/test_static.py` enforces all of this for the served bundle; keep those tests green and extend them when wording moves.
- The expression wire may carry labels, scores, valence/arousal, named blendshape floats and head-pose angles; actions and timeline entries add only labels, names, timestamps, one peak, a duration and a frame count. It must never carry landmarks, crops, pixels or embeddings (`domain.Expression` / `ActionEvent` / `AffectSample` allow nothing else — do not widen them).
- Mood and action entries in the browser event stream come from the server stream (`static/js/events.js`, an `EventSource` on `/api/events?role=browser`); the server's `MoodTracker` (`services/mood.py`) and `ActionTracker` (`services/actions.py`) are the single source of truth. The Stage 1 client-side `trackMood` debounce is gone and must stay gone — do not make the browser publish, re-derive or re-debounce the wire mood. Two *display* rules are allowed and are not re-derivations: a replayed entry (SSE reconnect + `Last-Event-ID`) is not logged with the current clock (`isFreshEntry`), and the same action label is shown at most every 5 s (`allowActionEntry`) so the 8-slot log keeps its errors, greetings and enrollments. The tile's valence sparkline is the one exception and is explicitly local: it plots this page's own frames, not the server history.
- `Presence.valence/arousal` are the **live** smoothed affect (`MoodTracker.affect()`, EMA over readable frames); only `Presence.mood` and `MoodTransition.valence/arousal` keep the hysteresis/commit-frozen semantics. Do not "fix" the presence values back to the mood's.
- Do not persist raw camera frames by default.
- Do not represent face distance as a confidence percentage.
- The browser consumes `RecognitionEvent.can_enroll` / `message` and `SystemStatus.engine_available`; it never re-derives recognition state or engine readiness client-side.
- Destructive actions confirm in a native `<dialog>`, never via `confirm()`/`alert()`.
- Do not use face recognition as authentication or authorization.

## Before changing code

1. State the user-visible outcome.
2. Identify the existing boundary that owns it.
3. Add/adjust acceptance tests first where practical.
4. Keep WIP to one vertical slice.
5. Do not add a database, cloud service, queue, React, Tauri, LLM, or new runtime unless an architecture trigger is documented.

## Required checks

```bash
PYTHONPATH=apps/face2ai/src pytest apps/face2ai/tests
python -m compileall -q apps/face2ai/src
for file in apps/face2ai/src/face2ai_app/static/js/*.js; do node --check "$file"; done
node --test 'apps/face2ai/tests/js/**/*.test.mjs'
```

Mock/injected recognition tests are not evidence that real camera recognition works. Real recognition requires the target-Mac gate in `docs/boilerplate/VALIDATION.md`.
