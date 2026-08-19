# One Owner Per Rule — Root-Cause Fix Plan (revision 2)

> **Revision note.** Revision 1 of this document was attacked by three independent auditors
> (craftsmanship, factual claims, thesis). They falsified its root cause, proved its headline fix
> inert, and corrected four of its numbers. Everything below is the corrected version; §11 lists what
> was wrong in r1 so the record is honest. Every number here was measured by a command; where a value
> is extrapolated it says so.

**Goal:** give every rule that must hold in more than one place a single owner that can fail — starting
with the boundary that aborts the process today.

**Evidence base:** a six-dimension review of the whole application, every finding attacked by an
independent verifier: 45 filed, 34 confirmed, 11 refuted, 0 unresolved. Plus three plan auditors who
re-measured the load-bearing claims. (The finding table is not yet in the repository — see §10.)

---

## 1. Root cause

**Every artifact in this repository is scoped to one feature — the ADRs, the commits, and above all
the tests. No rule that must hold in more than one place has an owner that can fail, so each rule
holds exactly where the commit that wrote it looked.**

Measured support:

- `Settings.__post_init__` did not exist at `6aec84d`. It appears in `709b22a` validating exactly the
  four knobs that commit introduced, and was extended the same way twice (`0c73a82`, `1b483cb`). At
  HEAD: **14 numeric fields, 10 validated, 4 not** — `port`, `match_tolerance`, `max_frame_bytes`,
  `greeting_cooldown_seconds`. `Settings(match_tolerance=-1)`, `max_frame_bytes=0`, `port=70000`,
  `greeting_cooldown_seconds=-5` are all accepted today.
- The same unbounded `Image.open` with no EXIF handling exists **twice** — `face_recognition_engine.py:34`
  and `mediapipe_expression.py:200` — written a day apart, each blind to the other.
- **The counter-case proves the mechanism.** The two cross-cutting rules that *did* get a shared
  executable owner — hedged expression wording (`test_static.py:132,227`) and "the wire carries no
  encodings, boxes or distances" (`test_events_api.py:230,275,413`) — hold across all three codebases
  (browser English, agent German, plugin German) and produced **zero** findings. Every cross-cutting
  rule without such a test (input bounds, closed error sets, shutdown, liveness, response shape)
  fails in every place it appears.

**Reinforcing cause: the test suite has no process and no real adapter.** 156 tests. The only
"live" fixture runs uvicorn in a daemon thread with `lifespan="off"` (`test_events_api.py:88-89`), and
uvicorn's `capture_signals` returns early off the main thread — so the shutdown path is *structurally
untestable* there. Every port is exercised through a double (`FakeEngine`, `FakeExpressionEngine`) that
satisfies its contract by construction. CI installs `pip install -e apps/face2ai` — neither
`[recognition]` nor `[expression]`. That is exactly the severity distribution we measured: every severe
finding needs a real process, a real adapter, or concurrency; the in-process, single-threaded,
fake-adapter parts are excellent.

**Therefore the fix is not 34 patches.** For each rule that must hold in more than one place: one
executable owner that runs in CI against the real thing.

`AGENTS.md` already says *"Identify the existing boundary that owns it."* That rule is written, binding
— and produced 34 findings. Which is the whole argument in one line: prose enforces nothing.

---

## 2. What this plan does not do

- It does not fix the 11 refuted findings (`/readyz` deliberately probes the engine not the store;
  `127.0.0.1` is the documented safe default; the conditional `timeline_cleared` publish is documented
  intended behaviour).
- It does not add gap-signalling to the broker (needs a new wire event plus three consumers).
- It does not touch the vendored fork's always-red CI, the Hermes announcement platform, the agent's
  reconnect backoff, or the overlay crop. See §9.

---

## 3. Task 1 — the process owns its shutdown (at the seam that actually runs first)

**Files:** `services/events.py`, `api/routes.py`, `main.py`, `docs/boilerplate/VALIDATION.md`;
tests `tests/test_lifecycle.py` (new).

**The correction that makes this work:** uvicorn's `Server.shutdown()` awaits `_wait_tasks_to_complete()`
**before** `await self.lifespan.shutdown()` (`uvicorn/server.py:271-301`). A FastAPI lifespan therefore
cannot unblock the wait it is meant to end. Measured on a faithful replica:

| variant | mechanism | result |
|---|---|---|
| A | lifespan → `broker.close()` → sentinel | **still alive 20 s after SIGTERM** |
| B | `timeout_graceful_shutdown=5` alone | exits after 5.2 s (by task cancellation) |
| C | `uvicorn.Server` subclass closing the broker in `shutdown()` | **exits after 0.2 s** |

**Step 1 — failing test** (`tests/test_lifecycle.py`): start the real `face2ai` entry point in a
subprocess on a spare port with `timeout_graceful_shutdown=30`, attach one raw SSE socket, wait for
`event: hello`, send SIGTERM, then
```python
assert proc.wait(timeout=5) in (0, -signal.SIGTERM)   # uvicorn re-raises the captured signal: -15 is correct
assert elapsed < 2.0                                   # with the defect this is 30 s; with only a timeout backstop, 10 s
```
Discriminating by construction: the 30 s backstop means only a real close passes.

**Step 2:** run → FAIL (today: SIGTERM ×3 and SIGINT ×2 leave it running; only SIGKILL works — reproduced).

**Step 3 — implement:**
- `IdentityEventBroker.close()`: set `closed`; for every subscription
  `sub.loop.call_soon_threadsafe(self._enqueue, sub, None)` — the same hand-off `publish()` uses, because
  a direct `put_nowait` from another thread does not wake a parked getter. `subscribe()` after `close()`
  returns a subscription whose queue already holds the sentinel (otherwise a request arriving during
  shutdown re-pins the process). `publish()` after `close()` is a no-op.
- `routes.py` stream loop: return on the sentinel, **and** in the `asyncio.TimeoutError` branch return when
  `broker.closed` — `wait_for` cancelling a woken `Queue.get()` can drop the item, and without this the
  test is flaky rather than failing.
- `main.py`: `class Face2AIServer(uvicorn.Server)` whose `async def shutdown(self, sockets=None)` closes the
  broker, then `await super().shutdown(sockets=sockets)`. `run()` builds `uvicorn.Config(create_app, factory=True,
  host=…, port=…, timeout_graceful_shutdown=10)` and serves it. Keep the timeout as a backstop, not as the
  mechanism.
- `main.py`: `logging.basicConfig(level=os.getenv("FACE2AI_LOG_LEVEL", "INFO"), …)` in `run()` so the app's
  INFO startup lines are visible at all (WARNING lines already surface via `logging.lastResort`).
- Delete the module-level `app = create_app()` (measured: `import face2ai_app.main` costs ~1.6 s warm /
  ~4.9 s cold and ~425–446 MB RSS, and binds a real engine into every importer including the test suite).
  **`docs/boilerplate/VALIDATION.md` and the project `CLAUDE.md` prescribe `uvicorn face2ai_app.main:app`** —
  update both to the factory form in this task.

**Step 5:** commit `fix(face2ai): the process owns its shutdown`.

---

## 4. Task 2 — the adapter boundary becomes an executable contract

**Files:** `ports/recognition.py`, `ports/identity_store.py`, `adapters/face_recognition_engine.py`,
`adapters/mediapipe_expression.py`, `adapters/json_identity_store.py`, `domain/errors.py`, `config.py`,
`.github/workflows/face2ai.yml`; tests `tests/test_port_conformance.py` (new).

**Step 1 — failing conformance suite**, parametrised over every recognition adapter present
(`FakeEngine` always, `FaceRecognitionEngine` when the extra is installed): re-entrancy (4 threads ×
24 calls on one frame → identical face count, encodings within 1e-9), oversized frame rejected with
`InvalidFrame` **before decoding**, EXIF-rotated photo still finds the face, and only
`InvalidFrame`/`RecognitionUnavailable` ever escape.

Write the re-entrancy case so an interpreter abort is a clean failure: run it in a subprocess and assert
the exit code, otherwise a crash takes the whole pytest session with it.

**Step 2:** run → FAIL for the real adapter. Measured today, 2 threads × 60 calls on one-face frames:
unlocked reports `[1, 78, 193, 456]` faces and aborts the process (`returncode 133`); **the fabricated
`DetectedFace` objects pass the 128-float validator and reach `IdentityService._nearest`, so they are
matched against the store** — this is a correctness and privacy failure, not only a crash. Locked: `[1]`,
0 exceptions, encoding distance `0.000000` from the serial baseline.

**Step 3 — implement:**
- `ports/recognition.py`: the Protocol docstring states the four obligations (re-entrancy, input bounds,
  EXIF-upright coordinate space, closed error set). The docstring is documentation; **the conformance
  suite is the enforcement.**
- `adapters/face_recognition_engine.py`: a **module-level** `threading.Lock` (dlib's singletons in
  `face_recognition` are process-global) held across `face_locations` + `face_encodings`; central decode —
  `Image.open` → check `.size` against `settings.max_frame_pixels` → reject → `ImageOps.exif_transpose` →
  `np.asarray(img.convert("RGB"))`. `Image.open` is lazy, so the size check happens before any pixel is
  decoded.
- **`adapters/mediapipe_expression.py` gets the same decode helper.** It has the identical unbounded
  `Image.open` with no EXIF handling, and a recognition-port suite would never touch it. Extract one shared
  `decode_frame(image_bytes, max_pixels)` and use it in both — one rule, one owner.
- `config.py`: `max_frame_pixels: int = 4_000_000` (`FACE2AI_MAX_FRAME_PIXELS`, validated). The browser's
  own snapshot is 640 px wide (`camera.js`), so 4 MP is generous.
- `ports/identity_store.py` closes its error set; `json_identity_store.py` maps `OSError` →
  `IdentityStoreUnavailable`; `routes.py` maps that to 503.
- **`.github/workflows/face2ai.yml`: a job that installs `--extra recognition` and runs the conformance
  suite.** Without it the suite runs against `FakeEngine` alone — a double that satisfies every obligation
  by construction — and the plan's one systemic artifact would be green exactly where it cannot fail.

**Step 5:** commit `fix(face2ai): recognition adapter honours an executable port contract`.

**Note on the trade-off, stated because the plan must not hide it:** the lock serialises recognition.
Measure and record the added latency for two tabs at the shipped 450 ms cadence. If p95 exceeds the loop
interval, that is a new (non-crashing) product problem and belongs in §9, not a reason to skip the lock.

---

## 5. Task 3 — validate every knob, and keep it that way

**Files:** `config.py`, `domain/models.py`; tests `tests/test_config.py`, `tests/test_models.py`,
`tests/test_identity_store.py`.

**Step 1 — failing tests:** all four unvalidated knobs reject nonsense (`port` outside 1..65535,
`match_tolerance` outside `0 < t <= 2`, `max_frame_bytes < 1`, `greeting_cooldown_seconds < 0`); an
`IdentityRecord` with a 3-float encoding raises `ValidationError`, and a store file containing one raises
`IdentityStoreCorrupted` instead of turning every later `/api/recognize` into an HTTP 500 through
`math.dist`.

**And the owner that prevents the next occurrence** — three lines, and the only thing in this plan that
stops the per-commit pattern measured in §1:
```python
def test_every_numeric_setting_is_validated():
    """Each new knob must be range-checked in __post_init__ — the habit that produced 4 unchecked knobs."""
    numeric = {f.name for f in fields(Settings) if f.type in ("int", "float")}
    checked = set(re.findall(r"self\.(\w+)", inspect.getsource(Settings.__post_init__)))
    assert numeric <= checked, f"unvalidated: {sorted(numeric - checked)}"
```

**Step 5:** commit `fix(face2ai): validate every numeric setting and the stored encoding shape`.

---

## 6. Task 4 — decide what `stale` means, then make one thing true

**Files:** `domain/models.py`, `services/presence.py`, `api/routes.py`, both consumers, docs; tests in all three.

`snapshot()` **already** computes `stale` correctly (`presence.py:148-158`) and has a passing unit test.
It can never be `True` on the wire because `_expired()` uses the *same* threshold and every reader calls
`expire()` first — so a presence old enough to be stale has already become `NO_SIGNAL`, whose `stale` is
`False` by construction. Prescribing "compute it in snapshot()" would produce a commit that changes nothing.

**Decision: remove `stale` from the wire.** Freshness is the consumer's business and it already has
`observed_at`; the one consumer that owns a clock (`context_line(max_age_seconds=30)`) is the one that
behaves correctly under every induced failure. Delete the field from `Presence`, from the agent
(`presence.py:113,128,263`), from the plugin (`presence.py:174,189,364`, `plugin.js:245`), and from the
documented wire contract. Consumers that want a freshness line compute it from `observed_at` against
their own budget.

**Also in this task** (one-word fix, confirmed end to end): the plugin's live `/presence` returns
`{source, presence, events_url}` while its two fallback branches return `connected`, and `plugin.js:57`
replaces `latest` wholesale — so the desktop chip reads "Face2AI nicht verbunden" after **every
successful** poll. Add `connected: True` to the live branch.

**Step 5:** commit `fix(face2ai,plugin): one truth about liveness`.

---

## 7. Task 5 — the browser pairs what must move together

**Files:** `static/js/app.js`, `static/js/model.js`, `static/js/events.js`; tests `tests/js/model.test.mjs`,
`tests/test_static.py`.

1. **Consent is sticky (high, privacy).** `els.consent.checked = false` and `displayName.value = ''` run only
   on the success path (`app.js:489-490`); `onEnrollDialogClosed` (`:505-508`) resets neither. Verified: after a
   failed enrollment closed with Escape, one click enrols a **different** person as
   `display_name=Alice&consent=true`. Fix: reset name, consent and the frozen frame on every close.
2. **The frame and its event can drift apart.** *(Corrected diagnosis — `submitEnrollment` already uses
   `state.enrollBlob` and already refuses when it is gone.)* `tick()` assigns `state.lastBlob = blob`
   **before** `await api.recognize(blob)` while `state.lastEvent` is assigned after, and a result arriving
   during enrollment is dropped as stale — so a LEARN click during an in-flight request freezes frame N+1
   with frame N's event. At ~160 ms per request inside a 450 ms loop that is roughly a third of clicks.
   Fix: publish the pair in one assignment — `handleRecognition(blob, event)` sets
   `state.latest = { blob, event }`, and `openEnrollDialog` freezes that object.
3. **The greeting cooldown is a one-identity slot** (`model.js:83-87`), so two enrolled people bypass it.
   Fix: a `Map(identityId -> lastGreetedAt)`, capped at 32 entries with least-recently-used eviction (stated
   so two implementers build the same thing). JS test: two identities alternating inside the cooldown.
4. **`events.js` can go permanently deaf.** Per the EventSource spec a non-2xx status or wrong content-type
   fails the connection and stays `CLOSED` — unlike a network drop, which reconnects. The file's comment
   ("EventSource reconnects by itself") is true only for the network case. *(Reasoned from the spec plus the
   absence of any re-subscribe path; needs one browser session to confirm.)* Fix: on `error` with
   `readyState === CLOSED`, re-subscribe with bounded backoff and show the state.

**Step 5:** commit `fix(face2ai): pair the frame with its event, reset consent, key the cooldown`.

---

## 8. Task 6 — close the two confirmed handover defects, and build what the user asked for

**Honest framing, per the audit:** the Mac launcher is the user's explicit request, not a consequence of the
root cause. Building it is the natural moment to close two handover findings that *are* consequences.

1. **`apps/face2ai/uv.lock`** — the sibling voice agent tracks its lock; the product does not, so no second
   party can reproduce this environment, and the native library behind the critical finding is pinned only as
   `dlib>=19.7` in the fork's `setup.py` while the lock pins `dlib 20.0.1` with a hash. **This overturns a
   recorded decision** (the project `CLAUDE.md` says the lock is never committed) — say so in the commit. To be
   an owner rather than a document, the app-shell CI job must move to `uv sync --frozen`.
2. **Document the env surface** — the README documents 9 of 18 `FACE2AI_*` variables; `HOST`, `PORT` and
   `MATCH_TOLERANCE` live only in a wrapper repo with zero commits, i.e. in no repository at all.
3. **`scripts/face2ai-service.sh {start|stop|status}`** — the testable unit. `start` refuses to double-start
   (probes `/healthz` first), writes a PID file and a log under `~/.face2ai/`; `stop` sends SIGTERM, waits, and
   escalates to SIGKILL after a bound (that escalation is what such a script does anyway — it does **not**
   depend on Task 1, and r1's "only works because Task 1" claim was false); `status` prints the health JSON.
4. **`scripts/build-macos-app.sh`** builds `Face2AI.app` into `~/Applications` as an AppleScript applet — the
   pattern the user's other launchers on this machine already use. `on run` → `face2ai-service.sh start` → poll
   `/healthz` → `open http://127.0.0.1:<port>`; `on quit` → `face2ai-service.sh stop`.
5. **ADR-005** records that this is not the "Tauri/desktop shell plus React and Python sidecar" ADR-001
   rejected: no new UI runtime, no bundled interpreter, no second implementation — an OS-level start/stop
   wrapper around the same local process.

**Verification is a manual checklist, and this plan says so plainly:** `open -a Face2AI` → `/healthz` answers →
the browser opens → Quit → the process is gone and the port is free. It must be executed, not asserted.

---

## 9. Deferred, with reasons

`sse-silent-event-loss` (needs a new wire event + three consumers) · `upstream-ci-always-red` (the vendored
fork's own workflow) · `announce-platform`, `stale-tool`, `sse-2`, `agent-retry` (Hermes-side; need a live
gateway this session must not restart) · `overlay-crops-analysed-field-of-view` (needs a camera session) ·
`store-corruption-blamed-on-api-and-engine` (belongs with the gap-signal work) · `required-checks-use-ambient-python`,
`py313-expression-unresolvable`, `deploy-reports-false-success` (doc/CI hygiene) · **a concurrency budget and a
load gate** — the lock turns a crash into a queue, and the next defect of this family is "five tabs make every
consumer's presence permanently stale" with no crash and no failing test.

## 10. Task 7 — the cross-origin agent hijack (promoted from "open question" to a task)

**Measured, not hypothesised:** the server has no origin check, so any page that reaches `127.0.0.1:8765` can
subscribe as `?role=agent`:
```
before: agent_connected = False
cross-origin SSE accepted: HTTP/1.1 200 OK   (Origin: https://evil.example, Sec-Fetch-Site: cross-site)
after : agent_connected = True   subscribers = 1
```
The browser shell then goes deliberately silent ("greeting left to the voice agent"), so any background tab can
disable the product's headline behaviour. `CLAUDE.md` documents that this port is reverse-tunnelled to a VPS, so
"loopback, therefore safe" is the wrong frame.

**Fix:** reject `role=agent` when the request carries a `Sec-Fetch-Site` header whose value is not `same-origin`.
Browsers always send it; the real voice agent and the Hermes plugin use httpx and never do, so they are
unaffected. Test both directions.

Still genuinely unverified, and labelled as such: **a dead camera track is invisible** — nothing subscribes to
`track.onended`, so a revoked camera leaves the loop posting the last frame forever. Needs a browser session.

## 11. What revision 1 got wrong

- **Root cause.** r1 claimed "never handed to a second party". Falsified: the critical defect needs two tabs
  (one party), the most-crossed boundary in the system (plugin ↔ desktop, every 4 s across an SSH tunnel) is
  broken, and `git log` shows validation was introduced per-commit for its own knobs — the most-documented stage
  produced zero validated knobs. The thesis also counted two objects inside one process as a "boundary", which
  makes it fit any defect anywhere.
- **The shutdown fix was inert** (lifespan runs after the wait) and its test asserted an unreachable
  `returncode == 0`, referenced a `live.app` attribute that does not exist, and targeted a fixture that runs with
  `lifespan="off"`.
- **Task 5 item 2 prescribed code that already exists**, leaving the real race untouched.
- **Task 4 prescribed a no-op** — `snapshot()` already computes `stale`.
- **"nine of ten knobs"** was wrong: 14 numeric fields, 10 validated, 4 not.
- **The conformance suite would never have run against the real adapter in CI.**
- **Committing the lockfile** overturns a written decision r1 did not acknowledge, and enforces nothing unless CI
  uses `--frozen`.
- **Numbers corrected:** import cost ~1.6 s warm / ~4.9 s cold, ~425–446 MB (not "4.9 s and 447 MB" flat); the
  decompression bomb is measured at 0.19 MiB → 44.8 s with 115 s extrapolated, not measured; the unlocked adapter
  fabricates encodings that reach the matcher, which r1 understated as "14 of 257 responses wrong".
- **The `.app` was presented as following from the root cause.** It does not; it is the user's request, and two
  of its five sub-items are genuine consequences.
