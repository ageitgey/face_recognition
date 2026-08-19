# ADR-004 - Expression dynamics (Stage 2): facial actions, live affect, bounded in-memory timelines

Status: accepted (2026-08-18)
Extends ADR-003 (Stage 1 expression hints) and ADR-002 (event consumers); supersedes exactly one
clause of ADR-003 decision 6 — the browser's client-side `trackMood` debounce, which is gone: mood
and action entries now come from the server stream (decision 5 below).

## Context

Stage 1 (ADR-003) reads one frame at a time: an 8-class label with scores, valence/arousal, 52
blendshape intensities, head pose, plus a hysteresis mood on the wire. Three asks were left open
and are what Stage 2 answers:

- **facial "micro" events** — the user wants "a brief smile", not only "looks happy": *when* a
  movement started, how strong it got, how long it lasted;
- **per-person history** — the Hermes pane and the browser should be able to show what the last
  minutes looked like, not just the current frame;
- **timelines** — one place to ask for that history over HTTP.

The constraints from ADR-001/ADR-003 do not move: local-first, nothing leaves the machine, no raw
frames, expression never enters matching, enrollment, presence, greeting or any other decision, and
nothing the UI shows may pretend to be a finding. Stage 2 adds a *time dimension* to a signal that
is already hedged — which makes over-trust easier, so the honesty section below is part of the
decision, not an appendix.

## Decision

1. **`ActionTracker` (`services/actions.py`) turns the blendshape series into completed actions.**
   Eight action groups, each the arithmetic mean of a small blendshape group (blendshapes absent
   from a frame count as 0.0 — `Expression.blendshapes` is compacted at 0.2, so "absent" is the
   normal way a low intensity arrives):

   | action | blendshapes |
   | --- | --- |
   | `smile` | `mouthSmileLeft`, `mouthSmileRight` |
   | `frown` | `mouthFrownLeft`, `mouthFrownRight` |
   | `brow_raise` | `browInnerUp`, `browOuterUpLeft`, `browOuterUpRight` |
   | `brow_furrow` | `browDownLeft`, `browDownRight` |
   | `eye_squint` | `eyeSquintLeft`, `eyeSquintRight` |
   | `eyes_wide` | `eyeWideLeft`, `eyeWideRight` |
   | `nose_wrinkle` | `noseSneerLeft`, `noseSneerRight` |
   | `lip_press` | `mouthPressLeft`, `mouthPressRight` |

   A hysteresis state machine per group: an inactive action starts at `FACE2AI_ACTION_ON_THRESHOLD`
   (0.35), an active one keeps running while it stays at or above `FACE2AI_ACTION_OFF_THRESHOLD`
   (0.2 — deliberately the compaction floor, so "gone from the frame" and "below off" are the same
   event) and completes on the first frame below it. Completion emits one `ActionEvent` *only* if
   the action lasted at least `FACE2AI_ACTION_MIN_FRAMES` (2) frames; shorter spikes are swallowed
   as noise. Several actions can be active at once (smile + eye squint). On equal peaks the first
   apex wins. Like `MoodTracker` the tracker follows the stable presence key
   (`f"{state}:{identity_id or ''}"`); a key change, an unreadable frame, presence expiry/reset or
   the expression toggle going off **drop** whatever was active *without* an event — the offset is
   unknown and is never guessed. `domain.ACTIONS` is the wire vocabulary and the module asserts at
   import that the groups and that tuple agree.

   **Speech articulators are deliberately not action groups.** `jawOpen`, `mouthFunnel`,
   `mouthPucker` and `mouthClose` move on nearly every frame while someone talks; a tracker over
   them would emit a stream of "actions" that say only "this person has a mouth and is using it".
   Blinks are excluded for the opposite reason: at the loop's frame rate (below) a blink usually
   falls between two frames, so what we could report would be an artefact of sampling, not of the
   face. Both exclusions are a *readability* decision, not a technical limit — they are review
   triggers, not permanent truth.

2. **`AffectHistory` (`services/timeline.py`) is a bounded in-memory ring-buffer trio.**
   `collections.deque` with `maxlen`: 2000 `AffectSample`s, 50 `MoodTransition`s, 100
   `ActionEvent`s. Samples are additionally bounded by age — `record_sample()` drops leading
   samples older than `max_seconds` (`FACE2AI_TIMELINE_SECONDS`, 600) relative to the *newest
   sample seen*, so an out-of-order sample can never un-prune the buffer and one window is the
   most that is ever held. `snapshot(seconds, identity_id)` returns everything inside
   `[now - seconds, now]`, oldest first, optionally narrowed to one identity; `now` defaults to the
   newest sample's timestamp, so a paused camera still shows its last window.
   **Nothing is persisted.** The history lives in the process, is cleared by
   `POST /api/presence/reset` (`clear()`) and is gone on restart. It stores only what the wire
   already carries — timestamps, labels, names, rounded scalars. There is no export, no file, no
   database, and adding one is a new decision, not an implementation detail.

3. **`Presence.valence/arousal` are now the *live* smoothed affect; `MoodTransition` stays frozen
   at commit.** `MoodTracker.affect()` returns the running EMA (alpha 0.5, rounded to 3 decimals,
   `None` until the first readable frame); `MoodTracker.current()` keeps the values as they were
   when the label committed. So the presence snapshot moves with the face while the `mood` event
   still fires once per mood change with the values that justified it — the wire does not narrate
   flicker (the point ADR-003 deferred), but a pane can draw a line. Toggling expression off clears
   the presence mood *and* affect unconditionally: a valence can be live without a committed mood.

4. **Two new SSE events and one new endpoint.**
   `GET /api/events` gains `action` next to `hello` / `presence` / `mood` / `store` / `heartbeat`;
   the payload is the `ActionEvent` (`at` == `offset_at`, `identity_id`, `display_name`, `action`,
   `onset_at`, `apex_at`, `offset_at`, `peak`, `duration_ms`, `frames`) plus the broker's
   `sequence`. `GET /api/expression/timeline?seconds=600&identity_id=<id>` returns the
   `TimelineSnapshot` (`{seconds, samples, moods, actions}`); `seconds` is bounded 10..3600
   (default 600), `identity_id` is at most 80 characters and an empty value means "no filter", not
   "nobody". Both are hints: nothing in the app reads them back, and a failure inside the dynamics
   block is caught in `routes._observe_expression` — logged once at WARNING with a traceback, then
   at DEBUG — so `/api/recognize` never fails because a hint failed.
   The second event is `timeline_cleared` (`{at}` and nothing else): `POST /api/presence/reset`
   publishes it when the clear actually dropped something. Forgetting is a privacy control, and
   consumers keep their own mirrors of the mood/action stream, so it has to be *visible* — a
   NO_SIGNAL transition is not that signal, because an ordinary presence expiry publishes one too
   and deliberately keeps the history. Consumers that hold no expression history ignore it.

5. **The browser takes mood and action entries from the server and draws its own sparkline.**
   `static/js/events.js` is a 24-line same-origin `EventSource` client on
   `/api/events?role=browser` (never `role=agent`, which would make the page hand greetings to a
   voice agent); it listens to `mood` and `action` only and drops malformed payloads instead of
   throwing. The server's `MoodTracker`/`ActionTracker` are the single source of truth, so nothing
   is re-derived or debounced client-side — the Stage 1 `trackMood` helper is deleted and
   `tests/test_static.py` asserts it stays deleted. Two *display* rules stay in the page, because
   the eight-slot log is shared with errors, greetings and enrollments: a replayed entry (the server
   buffers 200 events and `EventSource` resumes with `Last-Event-ID`) is dropped by `isFreshEntry`
   instead of being logged with the current clock, and the same action label is shown at most every
   5 s (`allowActionEntry`) — neither re-derives anything. Entries are hedged and capped at 8 in the
   stream: "Ben looks happy.", "Ben: brief smile (0.9 s)". The tile's valence sparkline is
   *separate*: it is built from this page's own frames (`pushSample`, last 120 readings ≈ 70 s),
   not from the server history — the page draws what it saw, and the timeline endpoint stays the
   one place that answers for the session.

6. **The Hermes plugin keeps its own bounded mirror and a pane sparkline.** `PresenceStore` holds
   the last 50 `mood` and 30 `action` frames as they came off the wire; its snapshot carries the
   last 20 moods / 10 actions for the dashboard process. `GET /api/plugins/face2ai/timeline`
   proxies Face2AI's endpoint (seconds clamped into 10..3600 instead of failing; on any error the
   same shape with empty lists). A `timeline_cleared` frame empties that mirror, and so does the
   pane's disposer, so disabling and re-enabling the plugin never shows the previous session.
   The desktop pane draws "Valenz · letzte 10 min" as an SVG sparkline over the timeline samples —
   fetched on its own ≥ 20 s cadence rather than on the 4 s presence poll (a 10 min window is up to
   2000 samples over the tunnel) and narrowed server-side to the person currently in front of the
   camera (without one, only the unattributed samples: one line is always one person) —
   plus "Stimmung zuletzt" (last 6 mood changes) and "Ausdruck zuletzt" (last 5 actions), each with
   the hedge in the tooltip and a "~0,6 s" resolution note. The **LLM context stays mood-only**:
   `describe()` and the `[face2ai]` line never mention actions — a language model narrating every
   brow raise is exactly the over-reading this ADR is trying to avoid.

7. **The voice agent ignores `action` and `timeline_cleared` by design.** `run_presence_loop`
   dispatches `hello/presence/store/heartbeat/mood/lost`; neither frame matches a branch and both
   are dropped without touching memory or the greeting path — the agent keeps no expression history,
   so it has nothing to forget. This is a decision, not an omission, and
   `tests/test_presence.py::test_presence_loop_ignores_action_frames_by_design` pins it.

8. **Settings** (all validated in `Settings.__post_init__`, all optional):
   `FACE2AI_ACTION_ON_THRESHOLD` (0.35), `FACE2AI_ACTION_OFF_THRESHOLD` (0.2, must satisfy
   `0 < off < on <= 1`), `FACE2AI_ACTION_MIN_FRAMES` (2, >= 1), `FACE2AI_TIMELINE_SECONDS`
   (600, >= 10).

## Honesty about what this is

- **The browser posts a frame every 450 ms plus round trip — about 1.7 fps, measured 2026-08-18.
  Every timing here is therefore quantized to roughly 0.6 s**, and onset and offset each carry one
  frame of uncertainty, so `duration_ms` is accurate to about ±1 frame.
- **"Micro-expressions" in the strict CASME sense (involuntary, under 500 ms) are OUT OF SCOPE and
  we say so.** At 1.7 fps such an event usually falls between two frames; anything we labelled
  "micro" would be a sampling artefact. What we ship is **expression dynamics**: onset, apex,
  offset, peak intensity and duration of *visible* expression, with "brief" reserved for ≤ 1 s and
  "held" for ≥ 5 s rather than any sub-second vocabulary. `tests/test_static.py` fails the build if
  the string "micro-expression" appears in the served bundle.
- The underlying models are dataset-biased (AffectNet-style posed/annotated faces, MediaPipe
  blendshapes); results vary with lighting, angle, glasses, skin tone and culture. Hysteresis and
  `min_frames` remove flicker; they do not make the remaining events true.
- **Talking moves the mouth constantly.** Even with the speech articulators excluded, a talking
  face raises brows and squints; expect actions while someone speaks, and do not read a smile
  during speech as a reaction to anything.
- **Unknown persons share one presence key.** The key is `f"{state}:{identity_id or ''}"`, so every
  unknown face is `UNKNOWN:` — two different people in front of the camera one after the other
  share one action state and one strand of history, and their samples mix in the timeline. Known
  limitation, same as the Stage 1 mood; per-person keys for unknown faces are a review trigger.
- An action says a movement happened, how strong and how long. It never says why. "Brief smile
  (0.9 s)" is admissible; "smiled because …", "is happy", "reacted to …" are not — in the UI, in
  the agent, in the plugin, and in any future consumer.

## Legal / ethics note

This carries ADR-003's note forward, unchanged in substance, to the new signals. The EU AI Act,
Art. 5(1)(f), prohibits AI systems that infer emotions of natural persons in the areas of workplace
and education institutions (medical/safety exceptions aside). Facial actions and an affect timeline
are emotion-adjacent inference of exactly that kind, and a *timeline* is more sensitive than a
single frame because it invites reading a story into it. Face2AI is a private, local tool for the
user's own machine and the user's own face: the feature is opt-in and off by default, the history
is in memory only and never persisted or exported, and nothing is sent to third parties. It must
stay that way, and it is not for use on other people without their consent. This is a product
posture, not legal advice; a deployment beyond one person's own machine needs its own review.

## Consequences

- (+) A consumer can now say "brief smile (0.9 s)" and draw the last ten minutes of valence without
  ever touching a frame, a landmark or a pixel — the wire still carries only labels, names,
  timestamps and rounded scalars.
- (+) `Presence` finally moves between mood commits (live affect), which is what made a sparkline
  possible at all, while the `mood` event keeps its one-change-per-mood semantics.
- (+) Two new pure services with no I/O (`ActionTracker`, `AffectHistory`) — both unit-testable
  without a camera, both injected in `main.create_app`, neither reachable from `domain/`.
- (+) Nothing new is installed: Stage 2 adds no dependency, and without the `expression` extra the
  whole surface degrades to "no `action` events, empty timeline", exactly as before.
- (−) **The browser now holds an open SSE connection** for as long as the page is open (previously
  it only polled). `EventSource` reconnects by itself with `Last-Event-ID`; a failed attempt logs
  "Live events unavailable" once instead of per retry, and the replayed buffer that arrives with the
  reconnect is filtered by age before anything is logged. It is one more long-lived connection to the
  local process and one more thing that can be stale after a redeploy.
- (−) **Forgetting is now a wire message.** A consumer that mirrors the mood/action stream can only
  honour `POST /api/presence/reset` if it hears about it, so `timeline_cleared` exists. It is one
  more event kind to keep in sync across four consumers (browser, plugin gateway half, desktop pane,
  agent), and a consumer that ignores it keeps showing history the user asked to forget.
- (−) More events on the wire. They are bounded by construction — hysteresis (0.35 on / 0.2 off)
  plus `min_frames` = 2 mean an action must be visible in at least two consecutive frames and only completes on the first frame below the off threshold, so about 1.2 s pass between onset and the reported offset at the browser's rate —
  but a talkative, expressive person will produce noticeably more `action` frames than `mood`
  frames. Measure before adding a consumer that reacts to each one (nothing may anyway).
- (−) Memory now holds a short affect history (bounded at 2000 samples / 50 moods / 100 actions and
  one time window). Small, but it is state that did not exist before, and it is the thing a future
  "just persist it" request will point at. The answer is a new ADR, not a flag.
- (−) One more surface that can be over-trusted, now with a *shape* (a line going down looks like a
  mood worsening). Mitigated by hedged wording, the resolution note in both panes, tested
  vocabulary and opt-in — not eliminated.
- New surface: `services/actions.py`, `services/timeline.py`, `domain.ACTIONS/ActionEvent/
  AffectSample/TimelineSnapshot`, SSE `action` and `timeline_cleared`, `GET /api/expression/timeline`,
  four settings, `static/js/events.js` + `describeAction/formatDuration/pushSample/sparklinePoints/
  isFreshEntry/allowActionEntry` in `model.js`, the plugin's `/timeline` proxy and pane sparkline.

## Review triggers

- **A higher frame-rate path** (worker-side capture, a native/desktop capture loop, MediaPipe's own
  streaming mode): the ~0.6 s quantization is the single reason "micro-expression" is out of scope.
  If the loop gets materially faster, re-measure, then revisit the vocabulary, the `min_frames`
  default and the "brief"/"held" boundaries — and only then may the word "micro" be discussed again.
- **Per-person keys for unknown faces** (short-lived track ids): would end the shared `UNKNOWN:`
  strand and change what a timeline means; needs a decision on how long such an id may live and
  whether it may reach the wire.
- **Temporal models over blendshape windows** (a sequence model instead of a threshold state
  machine): would replace `ActionTracker`'s hysteresis and would need buffered per-frame data with
  a stated retention — the reason the current design keeps only a state machine, not a window.
- Any request to persist, export or sync the timeline — including "just for debugging".
- Any use beyond the user's own face on the user's own machine → re-read the legal note first.
