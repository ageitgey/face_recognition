# ADR-002 - Voice agent as an event consumer beside the local recognition monolith

Status: accepted for S1 (2026-08-17)
Supersedes nothing; extends ADR-001.

## Context

Face2AI (ADR-001) is a local-only modular monolith: browser camera → FastAPI → face_recognition,
identities in a local JSON file, no LLM, no cloud. The next product step is a **voice agent that
talks with the person in front of the camera and knows who they are**. Speech-to-text, a language
model and text-to-speech are needed; a local face-recognition process must not become an
LLM-shaped process.

## Decision

1. **The agent is a separate process** (`apps/face2ai-agent/`, LiveKit Agents) that **consumes**
   Face2AI events. Face2AI publishes a debounced presence stream (`/api/events` SSE, `/api/presence`)
   derived from `RecognitionEvent`s by a `PresenceTracker` in `services/`; matching, enrollment and
   the identity store are untouched. Rule from AGENTS.md kept: agent behaviour consumes events, it is
   never inserted into face matching.
2. **The event stream carries no biometrics**: states, display names, counts, timestamps only
   (guarded by a test). The agent never receives frames or encodings.
3. **Speech providers are pluggable and environment-driven**; the default stack is the cheapest:
   LLM via OpenRouter free models (or Groq/OpenAI/Ollama), STT via a local Whisper server or Groq,
   TTS via a local Kokoro/Piper server or OpenAI. `auto` resolution prefers local/free.
4. **Cloud is allowed for speech and the LLM in S1, explicitly and only there.** This is the
   architecture trigger ADR-001 asked for: voice quality/latency needs models that do not run
   comfortably next to dlib on a laptop. The face pipeline stays local; only *what someone says*
   may leave the machine, and only to the providers the operator configured. Documented in the
   agent README and surfaced by `face2ai-agent check`.
5. **Greeting ownership**: while an agent is subscribed (`role=agent`), the browser suppresses its
   own speech greeting. The signal travels on every `/api/recognize` response (`X-Face2AI-Agent`
   header) so ownership flips within one frame, and in `/api/status`. The agent greets on stable
   presence transitions with a per-identity cooldown taken from the server; the browser only logs
   the hand-off. Probes (`check`, `smoke`) subscribe as `role=probe` and never take ownership.
   Ownership is a privilege, so the server guards it: `?role=agent` is refused with 403 when the
   request carries `Sec-Fetch-Site` ≠ `same-origin`. Browsers always send that header and cannot
   forge it; the agent and the plugin use httpx and never send it. Measured before the guard: a
   page on `https://evil.example` subscribed as the agent (`HTTP/1.1 200 OK`, `agent_connected`
   `False` → `True`) and the shell fell silent — and this port is reverse-tunnelled to a VPS.
   Presence expires server-side after `FACE2AI_PRESENCE_STALE_SECONDS` without frames, so a
   person who leaves while the tab is hidden and returns is a fresh arrival. That expiry is the
   *only* server-side freshness rule: the wire carries `Presence.observed_at`, never a freshness
   flag. A consumer that wants one owns the budget it compares against (the Hermes plugin withholds
   its whole context line above `context_max_age_seconds`); a flag on the wire would either
   duplicate the expiry threshold or, if equal to it, never be reachable — the presence would
   already be `NO_SIGNAL`.
6. **Console mode first**: the agent uses the machine's microphone/speaker via LiveKit console
   mode. Browser audio through a LiveKit room (server + JS client) is a later step and would be
   the first frontend dependency — a separate ADR when it comes.

## Alternatives considered

- ProsodyAI full-duplex plugin: speech-to-speech with voice-based speaker identity, but the client
  can only send audio (no instructions, no external LLM, no way to inject camera identity). Rejected
  for now; revisit as a second identity modality if the gateway exposes priming.
- Putting the LLM/STT/TTS inside the FastAPI app: violates ADR-001 (one local runtime for
  recognition) and couples release cadence of dlib and speech stacks. Rejected.
- Polling `/api/presence` instead of SSE: simpler client, but greeting latency and missed
  transitions; SSE with `Last-Event-ID` replay chosen.

## Consequences

- New backend surface: `PresenceTracker`, `IdentityEventBroker`, `/api/events`, `/api/presence`,
  `/api/presence/reset`, `SystemStatus.agent_connected`. Tests run the app under real uvicorn for
  the stream.
- New app `apps/face2ai-agent` with its own uv project, tests and CI job.
- Operators must know that voice audio goes to the configured providers; the 0 EUR local stack
  keeps it on the machine.
- Party Mirror can attach the same way (event consumer), as ARCHITECTURE.md already foresaw.
