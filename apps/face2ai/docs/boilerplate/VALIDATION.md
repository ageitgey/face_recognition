# Validation

## Automated gates

Run from repository root:

```bash
PYTHONPATH=apps/face2ai/src pytest apps/face2ai/tests
python -m compileall -q apps/face2ai/src
for file in apps/face2ai/src/face2ai_app/static/js/*.js; do node --check "$file"; done
node --test 'apps/face2ai/tests/js/**/*.test.mjs'
```

## Runtime shell smoke

```bash
uv run --project apps/face2ai face2ai            # the shipped entry point (FACE2AI_HOST/FACE2AI_PORT)
curl -fsS http://127.0.0.1:8765/healthz
# Ctrl-C / SIGTERM: the process must be gone in well under a second even with a browser tab open.
```

There is no module-level `app`: the app is built by the `create_app` factory. Serving it directly
(`uvicorn --factory face2ai_app.main:create_app`) works for a look at the shell but is **not** the
entry point — it lacks `Face2AIServer`, so an attached SSE stream (every open browser tab has one)
keeps the process alive on SIGTERM until `timeout_graceful_shutdown`. Use the `face2ai` command.

`/healthz` proves the app process only. `/readyz` returns 503 until the actual face-recognition engine can import and initialize.

## Required target-Mac recognition smoke

This is mandatory before claiming the real product flow works:

1. `uv sync --project apps/face2ai --group dev --extra recognition` on the target Mac.
2. Start Face2AI.
3. Confirm `/readyz` returns `ready: true`.
4. Grant browser camera permission.
5. Verify `NO_FACE -> UNKNOWN` with one consenting adult participant.
6. Enroll the participant explicitly.
7. Leave frame, return, verify `KNOWN` and spoken greeting.
8. Verify delete-one and erase-all remove the identity.
9. Record versions, command outputs, match distances, failures, and latency observation.

Unit or injected-adapter tests do not satisfy this real-boundary gate.

## Voice agent gate (apps/face2ai-agent, ADR-002)

```bash
cd apps/face2ai-agent && uv sync --group dev --extra groq
uv run pytest                                   # unit tests, no network
uv run face2ai-agent check                      # live: Face2AI reachable, SSE hello, LLM round trip (+ local TTS)
uv run face2ai-agent smoke "Wer ist gerade vor der Kamera?"   # live text turn with tool calls
uv run face2ai-agent console                    # manual: speak; step in front of the camera as an enrolled person -> spoken greeting by name
```

Required manual gate: with the browser running and one enrolled adult, `console` must greet the
person by name when Face2AI recognizes them, must not greet again within the cooldown, and the
browser's event stream must show "Greeting delegated" instead of speaking itself.

## Expression gate (ADR-003, opt-in mood hints)

Unit tests inject a `FakeExpressionEngine`; the adapter's pure helpers run without the extra. None of that proves
the models read a real face. Mandatory on the target Mac before claiming the expression stage runs:

```bash
uv sync --project apps/face2ai --group dev --extra recognition --extra expression
bash apps/face2ai/scripts/fetch-expression-models.sh
uv run --project apps/face2ai face2ai
curl -s http://127.0.0.1:8765/api/status | grep -o '"expression_[a-z]*":[^,}]*'      # expression_available:true, expression_enabled:false
curl -s -X POST http://127.0.0.1:8765/api/expression -H 'content-type: application/json' -d '{"enabled":true}'   # {"enabled":true,"available":true}
curl -s -N 'http://127.0.0.1:8765/api/events?role=probe'                             # leave running: expect `event: mood` frames below
```

1. Toggle-on works from the browser as well ("Expression: on"; the button is disabled while `/api/status` says the engine is unavailable).
2. Smile at the camera → the expression tile reads "looks happy" within ~3 ticks (the tile is per frame; the "Mood" event-stream entry appears once the label has held for 3 frames).
3. Frown / look sad → the tile changes (e.g. "looks sad" or "looks neutral"); valence bar moves left.
4. `/api/events` shows `event: mood` with `to_mood` for the change, and `/api/presence` carries `mood`, `valence`, `arousal`.
5. Leave the frame → after the presence transition a `mood` event with `to_mood: null` follows it.
6. `POST /api/expression {"enabled": false}` → next `/api/recognize` has `faces[].expression: null`, a `mood` end event was published, tile shows "off".
7. Voice agent (browser + agent running, one enrolled adult): `uv run face2ai-agent smoke "Wie wirke ich gerade?"` → hedged German answer ("… wirkt …", "nur ein Hinweis"), never "ist fröhlich".
8. Hermes plugin: `/presence` in Hermes shows the mood line ("wirkt … – nur ein Hinweis aus dem Gesichtsausdruck, keine Tatsache").
9. Record: mediapipe/emotiefflib/onnxruntime versions, per-frame `analyze` timing (M1 reference: ~146 ms cold, ~77 ms warm for one face), which labels you could and could not provoke.

Nothing in this gate may be described as the person *being* happy/sad — record what the tile *said*.

## Expression dynamics gate (ADR-004, actions + timeline)

Unit tests drive `ActionTracker`/`AffectHistory` with synthetic blendshapes; that proves the state
machine, not that a real face produces readable actions. Mandatory on the target Mac before claiming
Stage 2 runs. Prerequisites: the ADR-003 gate above (extras installed, models fetched, app running,
`expression_available:true`). Quote every URL containing `?` in single quotes — zsh globs otherwise.

```bash
curl -s -X POST http://127.0.0.1:8765/api/expression -H 'content-type: application/json' -d '{"enabled":true}'   # {"enabled":true,"available":true}
curl -s -N 'http://127.0.0.1:8765/api/events?role=probe'      # leave running in a second shell: expect `event: mood` and `event: action`
```

1. **Brief smile.** With the browser camera on and one face in frame, smile for about a second and
   let it go. The event stream shows an entry `Expression: <name>: brief smile (0.9 s)` (any
   duration ≤ 1.0 s reads "brief"), and the probe shell shows an `event: action` frame whose data
   has `"action":"smile"`, `onset_at`/`apex_at`/`offset_at`, `peak`, `duration_ms`, `frames`.
   Record the raw frame.
2. **Held smile.** Smile and hold it for ≥ 5 s, then relax. The entry now reads
   `Expression: <name>: held smile (x s)` with `duration_ms` ≥ 5000 (roughly eight frames at this loop rate — record the actual `frames`).
   A smile that never ends (you leave frame, reset, or toggle off while smiling) must produce **no**
   event — the offset is unknown and is never guessed. Verify that too.
3. **Sparkline.** Watch the Expression tile while you change expression: the small line under the
   valence/arousal bars appears once two readings exist and moves with them (up = positive valence).
   It is drawn from this page's own frames; covering the camera or toggling expression off clears it.
4. **Timeline.**
   ```bash
   curl -s 'http://127.0.0.1:8765/api/expression/timeline?seconds=120' | head -c 400
   curl -s 'http://127.0.0.1:8765/api/expression/timeline?seconds=120&identity_id=<id>' | head -c 400
   ```
   The first shows non-empty `samples` plus the `moods` and `actions` you provoked; the second shows
   only that person's. `seconds` is bounded 10..3600 — `?seconds=5` must fail with 422.
5. **Reset empties it.**
   ```bash
   curl -s -X POST http://127.0.0.1:8765/api/presence/reset > /dev/null
   curl -s 'http://127.0.0.1:8765/api/expression/timeline?seconds=600'
   # {"seconds":600,"samples":[],"moods":[],"actions":[]}
   ```
   Nothing was persisted: a restart of the app must show the same empty snapshot. The reset also
   publishes one `timeline_cleared` frame (`{at}` only) — watch it with
   `curl -sN 'http://127.0.0.1:8765/api/events?role=browser'` in a second terminal — and the Hermes
   pane's "Stimmung zuletzt"/"Ausdruck zuletzt" must be empty afterwards without a reload.
6. **Hermes pane** (after `apps/face2ai-hermes-plugin/deploy.sh` and ⌘K → Reload desktop plugins):
   the Face2AI pane shows "Valenz · letzte 10 min" as a sparkline (with the "~0,6 s" resolution
   note), "Stimmung zuletzt" and "Ausdruck zuletzt" with hedged German entries ("kurzes Lächeln
   (0.9 s)"). Check the proxy directly as well, against the Hermes dashboard host/port from
   `hermes config` (do not guess it): `curl -s '<hermes-dashboard>/api/plugins/face2ai/timeline?seconds=600' | head -c 200`.
7. **Nothing gates on it.** While actions are firing, recognition, enrollment and the greeting must
   behave exactly as in the ADR-003 gate, and the voice agent must not mention or react to a single
   action (`uv run face2ai-agent smoke "Was war gerade?"` stays mood-level and hedged).
8. Record: which actions you could and could not provoke, the measured `duration_ms`/`frames` of a
   brief and a held smile, how many `action` frames a minute of normal talking produced, and the
   sizes of `samples`/`moods`/`actions` after that minute.

Timing here is quantized to ~0.6 s (the 450 ms loop plus round trip). Nothing in this gate may be
recorded as a micro-expression, as the person *being* anything, or as a cause ("smiled because …") —
record what the stream *said*.
