# Face2AI

Face2AI is the local product layer built on top of the repository's existing `face_recognition` engine.

## Product flow

`UNKNOWN -> explicit LEARN -> leave frame -> return -> KNOWN -> greeting`

The application is intentionally local-first. Camera frames are processed in memory and are not intentionally persisted. Stored identity records contain display names and face encodings only.

## Architecture

- browser: camera preview, overlays, enrollment with explicit consent, identity management, event stream and spoken greeting; restrained ReactBits-inspired effects implemented as dependency-free CSS/ES modules (see `docs/UI_DIRECTION.md`);
- API: FastAPI on localhost;
- recognition: adapter over the repository's existing `face_recognition` package;
- expression (opt-in, ADR-003): a second port `ExpressionEngine` beside recognition — MediaPipe Face Landmarker + EmotiEffLib, local, CPU; attaches a hedged per-face hint (`faces[].expression`), a hysteresis `Presence.mood` and (Stage 2, ADR-004) live `Presence.valence/arousal`, facial actions (SSE `action`) and a bounded in-memory affect timeline; never part of matching, enrollment or greeting — see [Expression hints](#expression-hints-opt-in);
- storage: atomic local JSON file;
- presence + events: `GET /api/presence`, `GET /api/events` (SSE: `hello`, `presence`, `mood`, `action`, `timeline_cleared`, `store`, `heartbeat`), `POST /api/presence/reset` — the contract for agents / Party Mirror; no biometrics on the wire;
  - `presence` — stable presence changed (`from_state` → `to_state`, names, face count); a transition starts a fresh, mood-less presence and carries no mood fields;
  - `?role=` names the consumer (`browser`, `agent`, `hermes`, `probe`, …). Only `role=agent` is privileged — it takes greeting ownership — so it is refused with **403** when the request carries a `Sec-Fetch-Site` header whose value is not `same-origin`: a browser always sends that header and cannot forge it, while the voice agent and the Hermes plugin drive httpx and never send it. Without the check any page the user has open could reach the port (which the Mac tunnels to a VPS), claim the agent role and silence the greeting. Every other role is unaffected;
  - freshness is **not** a field. `Presence` carries `observed_at` (when the last frame was seen) and the server's single freshness rule is expiry: after `FACE2AI_PRESENCE_STALE_SECONDS` without frames the presence becomes `NO_SIGNAL` and that transition is published. A consumer that wants a "no fresh frames" line owns the budget it compares `observed_at` against — the Hermes plugin does exactly that and withholds its whole context line (`context_max_age_seconds`, 30 s) rather than annotating it;
  - `mood` — mood hint began/changed/ended (`from_mood` → `to_mood`, rounded valence/arousal); a `mood` with `to_mood: null` always follows the presence event that ended it (person left, presence expired or was reset), and is also published when expressions are switched off (`POST /api/expression {"enabled": false}`) while a mood is set; semantics in [Expression hints](#expression-hints-opt-in);
  - `action` — a completed facial action (`action` ∈ smile/frown/brow_raise/brow_furrow/eye_squint/eyes_wide/nose_wrinkle/lip_press, `onset_at`/`apex_at`/`offset_at`, `peak`, `duration_ms`, `frames`, names) — expression *dynamics* at frame-rate resolution (~0.6 s from the browser loop), never micro-expressions; a hint, never a fact, and consumers must not react to it with behaviour; an action whose offset is never seen (person left, reset, toggle off) produces no event;
- no LLM, cloud service, database, or security authorization in the recognition app itself. The optional voice agent (`apps/face2ai-agent/`, ADR-002) is a separate process that consumes the events.

Architecture decisions: `docs/architecture/ADR-001-local-modular-monolith.md` (local modular monolith), `ADR-002-voice-agent.md` (event-consuming voice agent), `ADR-003-expression-engine.md` (opt-in local expression hints), `ADR-004-expression-dynamics.md` (facial actions, live affect, in-memory timelines).

## Development

Target recognition runtime: Python 3.11 on macOS. On 2026-08-17 the recognition extra also built and ran on macOS/arm64 with Python 3.12 (dlib 20.0.1) via `uv sync --extra recognition`.

From the repository root:

```bash
uv sync --project apps/face2ai --group dev --extra recognition
uv run --project apps/face2ai face2ai
```

Open `http://127.0.0.1:8765`.

`uv.lock` is committed and is the only exact pin of the native stack: it resolves `dlib` to **20.0.1**
with a sha256, while the fork's `setup.py` asks only for `dlib>=19.7`. Reproduce the environment with
`uv sync --project apps/face2ai --frozen --group dev --extra recognition`; `--frozen` fails instead of
silently re-resolving. (This reverses an earlier decision that the lock is never committed — see
`docs/architecture/ADR-005-macos-launcher.md`.)

The `face_recognition` dependency is resolved from the repository root through `tool.uv.sources`. The temporary `setuptools<82` compatibility pin exists because the current model package still relies on `pkg_resources`; it is explicit S0 compatibility debt, not a permanent design choice.

## Configuration

Every setting is an environment variable read once by `Settings.from_env()` (`src/face2ai_app/config.py`).
There is no config file. This table is the **whole** surface — 19 variables, checked against the source
with `grep -o 'FACE2AI_[A-Z_]*' apps/face2ai/src/face2ai_app/config.py | sort -u | wc -l` (19). A value that
does not parse as the stated type raises at startup, so a typo fails visibly instead of silently falling
back to the default.

| Variable | Default | Range | Enforced? |
|---|---|---|---|
| `FACE2AI_HOST` | `127.0.0.1` | any bind address. Loopback is deliberate: the app has no authentication and must not be reachable from the network | n/a |
| `FACE2AI_PORT` | `8765` | `1..65535` | yes |
| `FACE2AI_DATA_DIR` | `~/.face2ai` | a directory; `~` is expanded. Holds `identities.json`, and `face2ai.log`/`face2ai.pid` when started through `scripts/face2ai-service.sh` | n/a |
| `FACE2AI_MATCH_TOLERANCE` | `0.6` | `0 < t <= 2` — Euclidean distance between face encodings; larger is more permissive. Never shown as a confidence percentage | yes |
| `FACE2AI_MAX_FRAME_BYTES` | `5242880` (5 MiB) | `>= 1` — request-body limit for `POST /api/recognize` and `/api/enroll` | yes |
| `FACE2AI_MAX_FRAME_PIXELS` | `4000000` (4 MP) | `>= 1` — decoded-pixel budget per frame, checked against the image header *before* decoding (a small file can declare an enormous image). Over the budget is `InvalidFrame` → HTTP 422 | yes |
| `FACE2AI_GREETING_COOLDOWN_SECONDS` | `15` | `>= 0` — reported by `/api/status`; the browser enforces it | yes |
| `FACE2AI_PRESENCE_STABLE_TICKS` | `2` | `>= 1` — frames a presence must hold before it is published | yes |
| `FACE2AI_PRESENCE_STALE_SECONDS` | `5` | `> 0` — a presence older than this expires to `NO_SIGNAL` | yes |
| `FACE2AI_EVENTS_HEARTBEAT_SECONDS` | `15` | `> 0` — SSE `heartbeat` interval | yes |
| `FACE2AI_EVENTS_BUFFER_SIZE` | `200` | `>= 1` — replay buffer for `Last-Event-ID` | yes |
| `FACE2AI_EXPRESSION_ENABLED` | `false` | `1/true/yes/on` or `0/false/no/off`, case-insensitive; empty means default, anything else raises. Pre-enables only an *available* engine | yes |
| `FACE2AI_EXPRESSION_MODELS_DIR` | `$FACE2AI_DATA_DIR/models` (`~/.face2ai/models`) | a directory; `~` is expanded. `face_landmarker.task` is expected here | n/a |
| `FACE2AI_MOOD_STABLE_TICKS` | `3` | `>= 1` — consecutive frames a mood candidate must lead before it commits | yes |
| `FACE2AI_MOOD_MIN_SCORE` | `0.5` | `0 < s <= 1` — EMA score a mood needs at commit | yes |
| `FACE2AI_ACTION_ON_THRESHOLD` | `0.35` | `off < on <= 1` — blendshape group mean that starts a facial action | yes |
| `FACE2AI_ACTION_OFF_THRESHOLD` | `0.2` | `0 < off < on` — and ends it (hysteresis) | yes |
| `FACE2AI_ACTION_MIN_FRAMES` | `2` | `>= 1` — frames an action must persist before it is reported | yes |
| `FACE2AI_TIMELINE_SECONDS` | `600` | `>= 10` — in-memory affect history window (never persisted) | yes |

**Every range in this table is enforced, and two tests keep it that way.** Until Task 3 of
`docs/plans/2026-08-19-boundary-contracts.md` landed, four knobs were documentation only:
`Settings(port=70000)`, `match_tolerance=-1`, `max_frame_bytes=0` and `greeting_cooldown_seconds=-5`
were all accepted (measured at `3857adc`: 15 numeric fields, 11 range-checked in
`Settings.__post_init__`, 4 not — they had been added one commit at a time, each validating only its
own knobs). The owners that stop the next occurrence live in `tests/test_config.py`:

- `test_every_numeric_setting_is_validated` — every `int`/`float` field of `Settings` must be
  mentioned in `__post_init__`. Adding a numeric knob without a bound fails this test by name.
- `test_every_setting_is_documented` — every `FACE2AI_*` literal in `config.py` must appear in this
  table. Adding a knob without a row fails it. (Nine of the earlier eighteen were undocumented, and
  `FACE2AI_HOST`, `FACE2AI_PORT` and `FACE2AI_MATCH_TOLERANCE` were described only in a wrapper
  directory outside every repository.)

Neither test checks that a bound is the *right* one — that is what the per-knob cases next to them
are for. They check that a bound and a row exist at all, which is the failure mode this repository
actually had.

Read only by `scripts/face2ai-service.sh`, never by the application: `FACE2AI_START_TIMEOUT_SECONDS`
(default `60`, how long `start` waits for `/healthz`) and `FACE2AI_STOP_TIMEOUT_SECONDS` (default `10`, how
long `stop` waits after SIGTERM before SIGKILL).

## Running it as a background service

`scripts/face2ai-service.sh {start|stop|status}` owns the process from outside Python and is the unit the
macOS launcher (`docs/architecture/ADR-005-macos-launcher.md`) is a thin shell over:

```bash
apps/face2ai/scripts/face2ai-service.sh start    # refuses to start a second one; waits for /healthz
apps/face2ai/scripts/face2ai-service.sh status   # /healthz JSON on stdout, non-zero exit when down
apps/face2ai/scripts/face2ai-service.sh stop     # SIGTERM, then SIGKILL after FACE2AI_STOP_TIMEOUT_SECONDS
```

Log `~/.face2ai/face2ai.log` (appended, with a banner per start), pid `~/.face2ai/face2ai.pid`, both under
`$FACE2AI_DATA_DIR`. `start` is atomic — on timeout it prints the log tail, stops what it started and exits
non-zero. `stop` is idempotent and **never kills a process it did not start**: if `/healthz` answers while the
pid file records nothing alive, it refuses and exits non-zero, because that port may be held by a backend, a
voice agent and an SSH tunnel this script knows nothing about. The SIGKILL escalation is not optional today:
with one SSE subscriber attached the current process ignores SIGTERM (see §3 of the boundary-contracts plan).

## Expression hints (opt-in)

The expression engine (Stage 1: ADR-003, Stage 2: ADR-004) reads facial expression **locally** and only when asked. Face2AI itself persists nothing about expression — Stage 2's affect history lives in bounded in-memory ring buffers, is cleared by `POST /api/presence/reset` and is gone on restart (the Hermes plugin mirrors the *current* presence snapshot — state, name, coarse mood/valence/arousal — plus its own bounded mood/action list into its plugin state file for the dashboard process: overwritten, no long-term storage). Nothing leaves the machine, and none of it gates recognition, enrollment or the greeting. It is a hint about how a face *appears* to a model — the UI says "looks happy", "brief smile (0.9 s)", the German consumers say "wirkt fröhlich", "kurzes Lächeln (0.9 s)" — never a fact, never a lie detector. **"Micro-expressions" in the strict (involuntary, < 500 ms) sense are out of scope**: the loop runs at ~1.7 fps, so timing resolves to about 0.6 s. Stage 1 delivers expression + intensity + head pose per frame plus a hysteresis mood; Stage 2 adds expression *dynamics* (onset/apex/offset, peak, duration), live valence/arousal on the presence and a short affect timeline.

Install and fetch both model assets (≈ 0.75 GB installed — mediapipe pulls jax/jaxlib, opencv, matplotlib). The fetch script downloads BOTH files up front — the MediaPipe landmarker and EmotiEffLib's ONNX model into the exact cache path emotiefflib expects — so startup never touches the network; if either file is missing the engine reports `expression_available:false` with the path and this script in the reason instead of downloading anything:

```bash
uv sync --project apps/face2ai --group dev --extra recognition --extra expression
bash apps/face2ai/scripts/fetch-expression-models.sh      # face_landmarker.task (3.7 MB) → $FACE2AI_EXPRESSION_MODELS_DIR
                                                          # enet_b0_8_va_mtl.onnx (16 MB)  → ~/.emotiefflib/
```

`mediapipe` is pinned to 0.10.21 (1.0.1 aborts on macOS in `DrishtiMetalHelper`; the adapter uses the CPU delegate) and declares `numpy<2`, so `pyproject.toml` overrides numpy to 2.3.5 — verified working on the M1. Without the extra everything degrades cleanly: `expression_available:false` with a reason, toggle disabled, `faces[].expression` null, no `mood` events.

Environment: `FACE2AI_EXPRESSION_ENABLED`, `FACE2AI_EXPRESSION_MODELS_DIR`, `FACE2AI_MOOD_STABLE_TICKS`, `FACE2AI_MOOD_MIN_SCORE` and, for the Stage 2 dynamics (ADR-004), `FACE2AI_ACTION_ON_THRESHOLD`, `FACE2AI_ACTION_OFF_THRESHOLD`, `FACE2AI_ACTION_MIN_FRAMES`, `FACE2AI_TIMELINE_SECONDS`. Defaults and ranges live in one place: [Configuration](#configuration).

API surface:

- `POST /api/expression` with `{"enabled": true|false}` — per-session runtime toggle, off by default; 409 when enabling while the engine is unavailable, disabling is always accepted (200); switching off ends the current mood immediately (a `mood` event with `to_mood: null`). Response `{"enabled", "available"}`.
- `GET /api/status` — `expression_available`, `expression_reason`, `expression_enabled`.
- `POST /api/recognize` — while enabled each `faces[]` observation carries `expression`: `{dominant, scores{Anger,Contempt,Disgust,Fear,Happiness,Neutral,Sadness,Surprise}, valence, arousal, blendshapes{name: 0..1, only ≥ 0.2}, yaw, pitch, roll}` — or `null` when the engine could not read that face. No landmarks, crops or pixels are ever on the wire.
- `GET /api/presence` / SSE `hello`, `heartbeat` — `Presence.mood` (the hysteresis label, null until committed) and `valence`, `arousal` — since Stage 2 the *live* smoothed values (EMA over the readable frames, rounded to 3), present as soon as one frame was readable, null while nothing is; the `mood` event keeps the values frozen at commit.
- `GET /api/expression/timeline?seconds=600` — the bounded in-memory affect history `{seconds, samples[{at, identity_id, display_name, mood, valence, arousal}], moods[MoodTransition], actions[ActionEvent]}` of the last `seconds` (10..3600, default 600 = `FACE2AI_TIMELINE_SECONDS`), oldest first. Add `identity_id=<id>` to narrow it to one known person; omit it for everyone (an empty value is treated as omitted, not as "nobody"). Never persisted: cleared by `POST /api/presence/reset` and by a restart.
- SSE `mood` — emitted once per stable mood change (server-side `MoodTracker`: EMA over the scores, alpha 0.5; a label commits when the candidate has led for `FACE2AI_MOOD_STABLE_TICKS` consecutive frames and its EMA is ≥ `FACE2AI_MOOD_MIN_SCORE` at that tick; valence/arousal are frozen at commit). The mood follows the stable presence: a presence change, `stable_ticks` frames without a readable expression (several faces, no face, expression off), presence expiry/reset or the toggle-off end it with `to_mood: null`. The browser's "Mood" and "Expression" event-stream entries come from this stream (`static/js/events.js`, an `EventSource` on `/api/events?role=browser`) — the server is the single source of truth and the Stage 1 client-side debounce is gone; the tile's valence sparkline is the one local view (this page's own last ~120 readings).
- SSE `action` — one completed facial action per event (`ActionTracker`, ADR-004): a group of blendshapes starts an action at `FACE2AI_ACTION_ON_THRESHOLD` (0.35), keeps it while it stays at or above `FACE2AI_ACTION_OFF_THRESHOLD` (0.2) and completes it on the first frame below, but only after `FACE2AI_ACTION_MIN_FRAMES` (2) frames. Speech articulators (`jawOpen`, `mouthFunnel`, `mouthPucker`, `mouthClose`) and blinks are deliberately not action groups. An action whose offset is never seen produces no event, and nothing may react to one with behaviour.
- SSE `timeline_cleared` — `{at}` and nothing else, published by `POST /api/presence/reset` when the clear actually dropped something. Consumers that mirror the `mood`/`action` stream (Hermes plugin, panes) forget with it; a `presence` transition to `NO_SIGNAL` is *not* that signal, because an ordinary presence expiry publishes one too and keeps the history. Consumers that keep no history (the voice agent) ignore it.
- The browser's own log discipline (`static/js/app.js`, hints stay hints): a replayed entry — `EventSource` resumes with `Last-Event-ID` and the server replays up to 200 buffered events — is dropped instead of being stamped with the current time (`isFreshEntry`, 10 s), and the same action label is displayed at most every 5 s (`allowActionEntry`) so a talking face cannot flush the 8-slot stream. Neither re-derives anything: the server stays the source of truth.

Measured on the M1 (2026-08-18, one face): adapter alone (full-frame decode + landmarker + EmotiEffLib, `examples/obama.jpg`, 910×1137) `analyze` ~146 ms cold / ~77 ms warm on the first measurement, ~84 / ~27 ms on a later warm run; live `POST /api/recognize` round trip via curl (dlib HOG + expression) `examples/obama-480p.jpg` (853×480) ~130 ms → ~150 ms with expression, `examples/obama.jpg` ~300 ms → ~330–380 ms — expression adds ~30–50 ms per frame, inside the browser's 450 ms loop.

Shell and `/assets/*` are served with `Cache-Control: no-cache` (ETag revalidation) since 2026-08-18 — a redeploy must never pair a fresh `app.js` with a heuristically cached `model.js` (this broke the shell once during verification).

Legal note, honesty section, dependency facts and review triggers: `docs/architecture/ADR-003-expression-engine.md` (Stage 1) and `docs/architecture/ADR-004-expression-dynamics.md` (Stage 2: why "micro-expressions" are out of scope, why speech articulators are not actions, what the shared `UNKNOWN:` key means for a timeline). Manual gates: `docs/boilerplate/VALIDATION.md`.

## Verification

```bash
PYTHONPATH=apps/face2ai/src pytest apps/face2ai/tests
python -m compileall -q apps/face2ai/src
for file in apps/face2ai/src/face2ai_app/static/js/*.js; do node --check "$file"; done
node --test 'apps/face2ai/tests/js/**/*.test.mjs'
```

Real recognition is not proven by unit tests. Before calling the product flow runtime-verified, run the target-Mac camera smoke described in `docs/boilerplate/VALIDATION.md`.
