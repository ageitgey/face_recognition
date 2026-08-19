# Face2AI UI Direction

Status: PRODUCT_UI_BASELINE (ported into `src/face2ai_app/static/` on 2026-08-17)
Origin: `DYAI2025/Face2ai-testbase` prototype (2026-08-16), adapted to the real API contract, ADR-001 and `AGENTS.md`.

## Design intent

Face2AI should feel like a quiet local intelligence instrument: futuristic enough to create curiosity, restrained enough to remain credible as a real product. The interface must never manufacture identity, confidence, activity, or telemetry for visual effect.

## Interaction principle

The visual system reacts to real product state as reported by the API:

- camera inactive -> quiet invitation to activate vision;
- camera active -> live local video becomes the visual focus (mirrored, like a mirror);
- recognition active -> restrained scanner motion and face overlays drawn only from `/api/recognize` output;
- `UNKNOWN` -> amber identity state and contextual enrollment action (`can_enroll` from the server, never re-derived);
- `KNOWN` -> mint identity state, name reveal, spoken greeting honoring the server-provided cooldown;
- `MULTIPLE_FACES` -> no enrollment affordance, server `message` shown verbatim;
- engine unavailable (`engine_available:false` or HTTP 503) -> explicit offline state with the server's reason, never synthetic fallback data;
- API errors -> the server's `detail` text is surfaced (toast, form error, event stream), never collapsed into a generic state.
- expression (opt-in, ADR-003) -> an "Expression" metric tile that starts as `off` and stays honest: while exactly one face is in frame it shows the server's per-frame hint as a hedged English phrase ("looks happy", "looks neutral"), valence/arousal as two small bars, and the caption "a hint, not a fact"; with no face, several faces or an unreadable face it shows `—`, toggled off it shows `off`; the toggle button ("Expression: off/on", `aria-pressed`) is disabled with the server's `expression_reason` in its title until `/api/status` says the engine is available, and its label follows `/api/status`, never the click. Tone: mint for a pleasant hint, neutral for neutral/surprise, amber for the rest — never red, expression is not an error state.
- expression dynamics (ADR-004) -> under the two bars the tile carries a **valence sparkline**: a 120×24 line of this page's own last ~120 readings (≈ 70 s at the 450 ms loop), +1 at the top, -1 at the bottom, newest at the right, over a dashed zero line. It is hidden until two readings exist and it restarts whenever the reading does (camera off, paused, toggle). Same neutral light stroke as the bars — a trace of what was read, never a trend line with a verdict, no fill, no axis labels, no gradient, no animation. It plots the browser's own frames; the server's history (`GET /api/expression/timeline`) is a separate thing and the tile never claims to show it.

## Enrollment

Enrollment happens in a native `<dialog>` and requires a display name **and** an explicit consent checkbox. The consent flag travels to `/api/enroll?consent=true`; the server rejects enrollment without it. Copy states what is stored (a face encoding and the name), that no raw frame is stored, and that the record can be deleted at any time.

## Effect set (ReactBits-inspired, dependency-free)

Implemented in `static/js/effects.js` and `static/css/app.css`; all off under `prefers-reduced-motion` (observed live via `matchMedia`).

1. DotGrid-inspired background — low-opacity spatial texture, tiny pointer parallax only, non-interactive.
2. LightRays-inspired camera atmosphere — prominent only before the camera is active, fades behind live video.
3. DecryptedText-inspired status transitions — short machine-state labels and identity names only, 320–520 ms, **only when the text actually changes** (no re-scramble on polling ticks), static under reduced motion. Scrambled text is never a live region: assistive tech receives the final text once through visually-hidden `aria-live` announcers.
4. SpotlightCard-inspired side panels — subtle pointer-driven radial highlight, no tilt or layout movement, `pointer-events: none`.
5. Magnet-inspired primary controls — maximum ~4 px attraction, disabled for touch and reduced motion, only on high-value contextual actions.

## Explicitly rejected effects

- full-screen glitch;
- cursor trails;
- hyperspeed backgrounds;
- persistent chromatic aberration;
- large WebGL shader stack;
- fake animated counters;
- decorative face-recognition boxes without recognition output;
- blocking browser prompts (`alert`/`confirm`) — destructive actions confirm in a `<dialog>`.

## Layout language

- Camera stage owns most of the viewport.
- Identity rail overlaps the stage slightly to break strict dashboard symmetry.
- Panels use different corner geometry and small positional offsets instead of a perfect card grid.
- Event stream sits below the camera as a secondary evidence layer: only real camera, recognition, greeting, enrollment, store, mood and expression events, at most 8 entries. It is evidence, not a feed: nothing is invented, nothing animates for its own sake, and every entry can be traced to something the server said. "Mood" and "Expression" entries come from the server stream (`events.js`, an `EventSource` on `/api/events?role=browser`) — the server's hysteresis and onset/offset are the truth, and the browser never re-derives or re-debounces them. What it does apply is *display* discipline for a panel with eight slots: an entry replayed after a reconnect is not logged as if it just happened, and the same action label is shown at most every 5 s. Wording stays hedged and describes movement, not the person: "Ben looks happy.", "Ben: brief smile (0.9 s)", "Ben: held smile (6.0 s)" — never "is smiling", never a cause, never "micro-expression" (timing resolves to ~0.6 s). Actions can be frequent while someone talks or is animated; the 8-entry cap and the calm single-line styling are what keep that from turning the panel into a ticker — if it ever reads as busy, that is a signal to log less, not to make it prettier.
- Product controls are contextual; unavailable actions stay disabled instead of pretending to work.

## State palette

- Mint: active / known / local / successful state.
- Amber (`--warning`): unknown / attention required / enrollment opportunity.
- Violet: atmospheric depth only, not semantic state.
- Red: destructive action or real error only.
- Neutral gray: inactive / unavailable / waiting.

## Quality constraints

- No horizontal overflow at desktop or mobile breakpoints (verified at 390 / 768 / 1024 / 1440 px).
- `prefers-reduced-motion` supported for every effect.
- Keyboard focus remains visible (`:focus-visible` outline on all controls).
- Modals and the identity drawer are native `<dialog>` elements: focus trap, Escape, top-layer backdrop, hidden controls are not focusable (closed dialogs are not laid out; the camera-on hero is `inert`). Cancel is never a form's default button, so Enter in a text field submits the primary action.
- While the enrollment dialog is open the recognition loop pauses and the frame/event the user clicked Learn on is frozen; enrollment is refused unless that event was `UNKNOWN` with `can_enroll`.
- Hover effects never block pointer events.
- Camera frames are not persisted by the UI; the capture canvas is transient.
- No synthetic identity events are emitted by the UI.
- Match distance is shown as a distance (3 decimals), never as a confidence percentage.
- Expression is shown as a hedged appearance ("looks …") with the caption "a hint, not a fact"; never as "is …", never as "detected", never with a certainty percentage (`tests/test_static.py` enforces the wording).
- A facial action is shown as a movement with a duration ("brief smile (0.9 s)"), never as a state ("is smiling"), never with a cause, and never as a "micro-expression" — the loop resolves to about 0.6 s (`tests/test_static.py` fails the build on all three).
- Everything is dependency-free: no CDN, fonts, frameworks or build step (ADR-001).
