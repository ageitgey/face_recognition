# Face2AI Voice Agent

A voice agent that talks with you and knows who is in front of the camera. It consumes the
Face2AI presence stream (`GET /api/events`) — states, display names, timestamps, a hedged mood hint — and never sees
frames or face encodings. Built on [LiveKit Agents](https://docs.livekit.io/agents/) with a
swappable STT → LLM → TTS pipeline; the default stack costs nothing per minute.

```
Browser (camera) ─▶ Face2AI backend ─▶ IdentityService ─▶ PresenceTracker ─▶ /api/events (SSE)
                                                                                    │
                                              face2ai-agent (this app) ◀────────────┘
                                              mic → STT → LLM(+tools) → TTS → speaker
```

Product loop: `UNKNOWN -> LEARN (consent) -> leave -> return -> KNOWN -> greeting`. While an
agent is subscribed the browser stays silent and the agent speaks the greeting; the same
server-side cooldown applies.

## Quick start (console mode: this Mac's microphone and speaker, no LiveKit server)

```bash
# 1. Face2AI backend must run (from repository root)
uv run --project apps/face2ai face2ai            # http://127.0.0.1:8765, open it and activate vision

# 2. Agent
cd apps/face2ai-agent
cp .env.example .env                              # put keys / URLs here (git-ignored)
uv sync --group dev --extra groq
uv run face2ai-agent download-files               # local VAD + turn-detector models, once
uv run face2ai-agent check                        # config, Face2AI, presence stream, LLM, local TTS
uv run face2ai-agent smoke "Wer ist gerade vor der Kamera?"   # text-only turn with tools
uv run face2ai-agent console                      # talk. Ctrl+C to stop.
```

`console` is LiveKit's local mode: audio in/out on this machine, no room, no cloud. `dev`/`start`
are the LiveKit worker modes for a real room (later: browser audio via a LiveKit server).
Run `console` from a real terminal (it reads keys from the TTY); the first start asks macOS for
microphone access.

## Providers (all via environment, see `.env.example`)

| Role | `auto` picks (cheapest first) | Notes |
| --- | --- | --- |
| LLM | `custom` when `FACE2AI_AGENT_LLM_BASE_URL` is set → OpenRouter (`OPENROUTER_API_KEY`, default `openai/gpt-oss-20b:free` + free fallbacks) → Groq (`GROQ_API_KEY`) → OpenAI; `ollama` only when chosen explicitly | OpenRouter free models are rate-limited; fallbacks are used automatically |
| STT | local OpenAI-compatible Whisper server (speaches at `:8000`, or `FACE2AI_AGENT_STT_BASE_URL`) → Groq Whisper (`whisper-large-v3-turbo`) → OpenAI | local = 0 €; Groq is cheap/free-tier |
| TTS | local server (Kokoro-FastAPI `:8880` or speaches `:8000`, or `FACE2AI_AGENT_TTS_BASE_URL`) → OpenAI TTS | **Kokoro has no German voices** — for `de` use a Piper German voice via speaches (auto-picked when present) or OpenAI |
| VAD / turn detection | Silero VAD + local multilingual turn detector | 0 €, downloaded by `download-files` |

Explicit choices always win: `FACE2AI_AGENT_LLM=openrouter|groq|openai|ollama|custom`,
`FACE2AI_AGENT_STT=local|groq|openai`, `FACE2AI_AGENT_TTS=local|openai` (+ `_MODEL`, `_VOICE`,
`_BASE_URL`, `_API_KEY`). `check` prints what was resolved and why.

### 0 € stack for German (verified 2026-08-17 on an M1)

```bash
# speaches: STT (faster-whisper) + TTS (Kokoro / Piper) in one OpenAI-compatible server on :8000
docker compose -f docker-compose.local-audio.yml up -d speaches      # Linux/Docker
#   or natively on macOS without Docker (needs `brew install espeak-ng` once):
scripts/run-speaches-macos.sh                                        # keeps running; Ctrl+C stops it
# models, once:
uvx speaches-cli model download Systran/faster-whisper-small         # STT (or ...-large-v3-turbo)
uvx speaches-cli model download speaches-ai/piper-de_DE-thorsten-medium   # German TTS voice
uvx speaches-cli registry ls --task text-to-speech | grep -i de_DE   # other German voices
```
With `OPENROUTER_API_KEY` set the agent then runs LLM free, STT local, TTS local; `auto` picks the
German Piper voice by itself. `check` shows `stt: local ...` / `tts: local ...` and does a real
TTS→STT round trip (measured: 3.5 s of German audio synthesized in 0.6 s, transcribed in 2 s).

## Hermes as the brain (your own agent instead of a bare model)

Hermes (`hermes-agent`) exposes an OpenAI-compatible API server (`gateway` platform `api_server`,
port 8642: `API_SERVER_ENABLED=true` + `API_SERVER_KEY` in Hermes' `.env`). Point the voice agent
at it and every turn is a full Hermes turn — his persona (SOUL), tools and gbrain memory — with the
live presence report layered on top as an ephemeral system prompt:

```bash
# in .env — with the key present, `auto` picks hermes
HERMES_API_SERVER_KEY=<API_SERVER_KEY of the gateway>
HERMES_API_SERVER_URL=http://127.0.0.1:8642/v1     # on this Mac: SSH tunnel (com.hermes.tunnel9119 also forwards 8642)
HERMES_SESSION_KEY=face2ai-voice                   # scopes Hermes' long-term memory to this channel
# FACE2AI_AGENT_LLM_MODEL=hermes-fast              # optional model_routes alias configured in Hermes (faster upstream model)
```

Trade-off: a Hermes turn takes ~10–15 s on the VPS (large context + memory hook), a bare
OpenRouter/Groq model ~1–2 s. `FACE2AI_AGENT_LLM=openrouter` switches back at any time.

## Behaviour

- **KNOWN** transition → greets by name (LLM-phrased; `FACE2AI_AGENT_GREETING_STYLE=say` for a fixed sentence). The same person is greeted again only after being away for `FACE2AI_AGENT_REGREET_AFTER_SECONDS` (90 s; a brief drop-out is not a new arrival) and never within the server cooldown (`/api/status.greeting_cooldown_seconds`).
- **UNKNOWN** → says hello, says it doesn't know them, mentions "Learn person" (rate-limited, `FACE2AI_AGENT_GREET_UNKNOWN=off` to disable).
- **MULTIPLE_FACES** → mentions it sees several people and cannot tell who is who.
- **enrolled** (store event) → welcomes the new person by name.
- Tools available to the LLM: `who_is_here`, `list_known_people`, `face2ai_status`.
- The system prompt is rebuilt whenever the situation changes (state, identity, engine, store size) — "Current situation (updated live)"; heartbeats alone do not churn the chat context.
- **Mood hint** (Face2AI expression stage 1): presence snapshots and SSE `mood` events carry `mood`/`valence`/`arousal`. The agent keeps them on the current presence and appends one hedged sentence to the situation report in the configured language — "Ben wirkt fröhlich (Valenz +0.6, Erregung +0.1) – nur ein Hinweis aus dem Gesichtsausdruck, keine Tatsache." / "Ben looks happy (valence +0.6, arousal +0.1) — only a hint from facial expression, not a fact." A presence transition starts a fresh, mood-less presence; `to_mood: null` ends the hint. Mood is deliberately **not** part of the situation key: it never rebuilds the prompt, never triggers a greeting and never gates anything — the LLM sees it whenever the prompt is rebuilt for another reason and on demand via `who_is_here`. The prompt rules tell the model to treat it as a guess ("du wirkst …"), never as a fact, never to psychoanalyse or probe.
- **`action` events** (Face2AI expression stage 2 — facial dynamics such as "brief smile", onset/apex/offset) are **ignored by the agent by design**: it never reacts to how someone's face moves — no dispatch, no instruction refresh, no greeting, no memory change; dynamics are for the browser tile and the Hermes pane/history only (pinned by `test_presence_loop_ignores_action_frames_by_design`).
- Events replayed after a reconnect update memory but are not spoken; transitions older than 20 s are never spoken.

Rules baked into the prompt: recognition is best-effort and never a login; never invent a name;
never claim to see or store faces; answer in the configured language, one or two spoken sentences.

## Verification

```bash
uv run pytest                                     # unit tests (config, presence, policy) — no network
uv run face2ai-agent check                        # live: Face2AI + SSE hello + LLM round trip (+ local TTS)
uv run face2ai-agent smoke "..."                  # live: text turn with tool calls against real presence
uv run face2ai-agent console                      # real: speak, get greeted when Face2AI recognizes you
```

The console run is the acceptance gate for the voice loop (see `docs/boilerplate/VALIDATION.md` in `apps/face2ai`).

## Privacy / cost boundaries

- Only display names, presence states, timestamps and a coarse mood hint (label + two rounded scalars) reach the agent; nothing biometric.
- Your **voice** goes to whichever STT/LLM/TTS you configure. Local providers keep it on the machine; cloud providers do not — that is documented as an accepted trade-off in `apps/face2ai/docs/architecture/ADR-002-voice-agent.md`.
- No LiveKit Cloud account is needed for console mode.
