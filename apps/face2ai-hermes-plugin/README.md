# Face2AI ↔ Hermes plugin

Makes Face2AI's camera presence a first-class input for [hermes-agent](https://github.com/NousResearch/hermes-agent):
Hermes knows on every turn who is in front of the local camera, can be asked explicitly, and the
desktop app shows it live.

```
Mac                                              Hermes host (VPS)
Face2AI :8765 ──SSE /api/events──▶ (ssh -R 8765) ──▶ gateway plugin `face2ai`
   ▲                                                  ├─ pre_llm_call → "[face2ai] Ben steht vor der Kamera …" in every turn
   │ ctx.rest via dashboard :9119 (ssh -L)             ├─ tool presence_now, /presence, system-prompt section
Hermes desktop app (plugin desktop/plugin.js) ◀────── └─ dashboard/plugin_api.py → /api/plugins/face2ai/{presence,history,timeline,health,events}
```

One installable folder (`face2ai/`), two halves, both opt-in:

| Half | Where | Enable |
| --- | --- | --- |
| Python (`plugin.yaml`, `__init__.py`, `dashboard/plugin_api.py`) | Hermes host `~/.hermes/plugins/face2ai/` | `hermes plugins enable face2ai` (+ gateway/dashboard restart) |
| Desktop (`desktop/plugin.js`) | the Mac running the desktop app `~/.hermes/plugins/face2ai/` | Settings → Plugins → "Face2AI presence" |

`./deploy.sh` does both (rsync to `hermes-brain`, enable, restart, copy the desktop half).

`GET /api/plugins/face2ai/presence` answers `connected` on **all three** branches (live proxy, persisted
snapshot, nothing): the desktop half polls it every 4 s and replaces its `latest` wholesale, so a branch
that omits the key paints the chip "Face2AI nicht verbunden" after a *successful* poll, until the next SSE
frame corrects it. Pinned by `tests/test_plugin_api.py`.

## What Hermes gets

- **Every turn**: `pre_llm_call` returns `{"context": "[face2ai] …"}` — appended to the user message
  (Hermes' documented seam for live context; system-prompt sections are frozen per session). Withheld
  when the last frame is older than `context_max_age_seconds` (30 s) so Hermes never acts on stale presence.
- **On demand**: tool `presence_now` (JSON incl. recent transitions), slash command `/presence`.
- **Mood hint** (Face2AI expression stage 1): presence snapshots and SSE `mood` frames carry `mood`/`valence`/`arousal`.
  The store keeps them on the current presence (a `presence` transition starts a fresh, mood-less presence;
  `to_mood: null` ends the hint) and `describe()` appends one hedged sentence in the configured language —
  "Ben wirkt fröhlich (Valenz +0.6, Erregung +0.1) – nur ein Hinweis aus dem Gesichtsausdruck, keine Tatsache." /
  "Ben looks happy (valence +0.6, arousal +0.1) — only a hint from facial expression, not a fact." The `[face2ai]`
  line, `presence_now`, `/presence`, the persisted snapshot (dashboard API) and the desktop pane all carry it; a mood
  frame is never a transition, never an announcement and never gates anything. Face2AI itself persists nothing about
  expression; this plugin mirrors the *current* presence snapshot (state, name, coarse mood/valence/arousal) into its
  plugin state file for the dashboard process — overwritten, no history (the snapshot's `history` holds only the last
  10 presence transitions: states, names, timestamps — never a mood). The system-prompt section tells
  Hermes to treat it as a guess — never state it as a fact, never psychoanalyse or probe.
- **History + timeline** (Face2AI expression stage 2): Face2AI keeps a *bounded in-memory* affect history per
  session — live valence/arousal samples, mood changes and completed facial actions (SSE `action`:
  `smile | frown | brow_raise | brow_furrow | eye_squint | eyes_wide | nose_wrinkle | lip_press` with
  onset/apex/offset timestamps, one peak, `duration_ms`) — cleared on `POST /api/presence/reset` and on restart.
  The plugin mirrors it minimally: `PresenceStore` keeps the last 50 `mood` and 30 `action` frames as they came off
  the wire, and the persisted snapshot (plugin state file, dashboard API, `presence_now`) carries the current values
  plus the last 20 moods / 10 actions — no long-term storage, overwritten on every frame, and dropped entirely
  when Face2AI publishes `timeline_cleared` (its `POST /api/presence/reset`: the user's explicit forget reaches
  the mirror; a `presence` → `NO_SIGNAL` frame does not, since an ordinary expiry keeps the history on both
  sides). Disabling the desktop plugin drops its copy too. `/presence` appends
  "Zuletzt gezeigt: kurzes Lächeln (0.9 s) (12:00:03), …" (last 3 actions, local wall time like the rest of the
  reply); the **agent/LLM context does not include
  actions** (`describe()` / the `[face2ai]` line stay mood-only — actions would be noise there). Wording is hedged
  and quantised: `action_sentence()` → "kurzes Lächeln (0.9 s)" (≤ 1 s), "Brauen hoch (2.3 s)", "anhaltendes
  Lächeln (6.0 s)" (≥ 5 s) — timing resolution is the browser loop (~0.6 s), so these are expression *dynamics*,
  never micro-expressions. `GET /api/plugins/face2ai/timeline?seconds=600[&identity_id=…]` proxies Face2AI's
  `GET /api/expression/timeline` (10..3600 s, clamped; on error the same shape with empty lists) for the desktop
  pane. The pane fetches it on its own ≥ 20 s cadence — not on the 4 s presence poll: a 10 min window is up to
  2000 samples (~160 KB) over the tunnel for a 240 px line — and passes `identity_id`, so the filtering happens
  in Face2AI rather than in the pane. It draws a valence sparkline (last 10 min, one person: the one in front of
  the camera, or only the unattributed samples when nobody is recognized),
  "Stimmung zuletzt" (last 6 mood changes) and "Ausdruck zuletzt" (last 5 actions), each with the tooltip
  "Vermutung aus dem Gesichtsausdruck, keine Tatsache" and the hint "Auflösung ~0,6 s — Ausdrucksdynamik, keine
  Mikroexpressionen".
- **Static explanation** in the system prompt: what the `[face2ai]` line means, best-effort, never guess names, never authentication, mood hints are guesses ("wirkt …"), never facts.
- **Optional proactive**: `announce_arrivals: true` + `plugins.entries.face2ai.allow_gateway_injection: true` →
  on a fresh `→ KNOWN` transition the plugin injects a user-turn into the most recent gateway session
  (per-person cooldown). Off by default; the voice agent already greets.

Settings (`plugins.entries.face2ai.settings.*`): `events_url` (default `http://127.0.0.1:8765`),
`inject_context`, `context_max_age_seconds`, `platforms` (e.g. `[api_server, desktop]`),
`announce_arrivals`, `announce_cooldown_seconds`, `language` (`de`/`en`).

## Network

The Hermes host must reach Face2AI. On this setup the Mac's tunnel LaunchAgent
(`~/.hermes-bridge/tunnel-9119.sh`) also carries `-R 127.0.0.1:8765:127.0.0.1:8765`, so the VPS
sees Face2AI at `localhost:8765`; nothing is exposed publicly. Alternatively set `events_url` to the
Mac's Tailscale address.

## Privacy

Only what Face2AI publishes: states, display names, counts, timestamps, plus the hedged expression hints
(mood label + rounded valence/arousal, action label + onset/apex/offset + one peak). No frames, boxes,
landmarks, blendshape series, encodings or match distances leave the Mac (guarded by Face2AI's tests), and
nothing about expression is stored long-term on either side: the mirror is bounded (50/30 in memory, 20/10 in
the state file) and `POST /api/presence/reset` on the Mac empties it here too, via the `timeline_cleared` frame.

## Verify

```bash
uv run --no-project --with pytest==9.0.2 pytest tests             # 27 passed, 1 skipped — no Hermes needed (tests/test_plugin_api.py skips as one module without fastapi)
uv run --no-project --with pytest==9.0.2 --with fastapi --with httpx pytest tests   # 32 incl. the 5 dashboard-API tests (4 /timeline + /presence connected)
cp face2ai/desktop/plugin.js "$TMPDIR/plugin.mjs" && node --check "$TMPDIR/plugin.mjs"   # JSX gate: `node --check plugin.js` does NOT fail on JSX (module-syntax detection swallows it); the .mjs copy does, and tests/test_desktop_plugin.py greps for it
ssh hermes-brain 'bash -lc "hermes plugins list | grep face2ai; curl -s 127.0.0.1:8765/api/presence"'
curl -s -H "Authorization: Bearer <dashboard token>" http://127.0.0.1:9119/api/plugins/face2ai/health   # via the 9119 tunnel
# then ask Hermes (desktop / telegram / voice agent): "Wer steht gerade vor der Kamera?"
```
