// Pure view-model helpers for the Face2AI shell. No DOM, no fetch — testable with `node --test`.
// The browser is a consumer of RecognitionEvent objects produced by the API; nothing here
// re-derives recognition state or invents identity data.

export const STATE_LABELS = Object.freeze({
  NO_FACE: { name: 'No identity', sub: 'No face detected', cls: '', message: 'Waiting for a person.' },
  UNKNOWN: { name: 'Unknown', sub: 'Not enrolled', cls: 'unknown', message: 'One unknown face detected. Enrollment is available.' },
  LEARNING: { name: 'Learning', sub: 'Enrollment in progress', cls: 'unknown', message: 'Storing a local face encoding.' },
  KNOWN: { name: 'Known identity', sub: 'Recognized', cls: 'known', message: 'Identity matched.' },
  MULTIPLE_FACES: { name: 'Multiple people', sub: 'Enrollment unavailable', cls: '', message: 'Multiple faces detected. Enrollment is blocked.' },
  ERROR: { name: 'Unavailable', sub: 'Recognition error', cls: 'error', message: 'Recognition error.' },
});

const FALLBACK = Object.freeze({ name: 'Unavailable', sub: 'Unknown state', cls: 'error', message: 'Unrecognized recognition state.' });

/**
 * Turn a RecognitionEvent (see domain/models.py) into display fields.
 * `can_enroll` and `message` come from the server and are never re-derived here.
 */
export function describeEvent(event = {}) {
  const state = typeof event.state === 'string' ? event.state : 'ERROR';
  const faces = Array.isArray(event.faces) ? event.faces : [];
  const primary = faces.length ? faces[0] : null;
  const labels = STATE_LABELS[state] || FALLBACK;
  const distance = Number.isFinite(primary?.match_distance) ? primary.match_distance : null;
  const name = state === 'KNOWN' && primary?.display_name ? primary.display_name : labels.name;
  const serverMessage = typeof event.message === 'string' && event.message ? event.message : null;
  const message = serverMessage || (state === 'KNOWN' && primary?.display_name ? `Identity matched: ${primary.display_name}.` : labels.message);
  return {
    state,
    label: state.replace(/_/g, ' '),
    name,
    sub: labels.sub,
    cls: labels.cls,
    faces: faces.length,
    primary,
    distance,
    canEnroll: event.can_enroll === true,
    message,
  };
}

/** Views for states the server never sends: camera off, and engine unavailable while the camera runs. */
export function offlineView(reason = null) {
  return {
    state: 'OFFLINE',
    label: 'OFFLINE',
    name: 'No identity',
    sub: reason ? `Engine unavailable · ${reason}` : 'Vision inactive',
    cls: reason ? 'error' : '',
    faces: 0,
    primary: null,
    distance: null,
    canEnroll: false,
    message: reason || 'Vision inactive.',
  };
}

/** Key that changes only when the visible state or the resolved identity changes. */
export function transitionKey(event = {}) {
  const primary = Array.isArray(event.faces) && event.faces.length ? event.faces[0] : null;
  return `${event.state || ''}:${primary?.identity_id || ''}`;
}

/**
 * Project a face box from the coordinate space of the frame that was sent to the API
 * ({top,right,bottom,left} in pixels of that JPEG) into a view that displays the frame
 * with `object-fit: cover` semantics (scale to fill, crop the overflow axis, centred).
 */
export function projectBox(box, frame, view) {
  const scale = Math.max(view.width / frame.width, view.height / frame.height);
  const offsetX = (view.width - frame.width * scale) / 2;
  const offsetY = (view.height - frame.height * scale) / 2;
  return {
    x: box.left * scale + offsetX,
    y: box.top * scale + offsetY,
    width: (box.right - box.left) * scale,
    height: (box.bottom - box.top) * scale,
  };
}

export const GREETING_MEMORY_MAX = 32;  // identities remembered for the cooldown; see shouldGreet

/**
 * Greeting policy: a newly seen identity is greeted at once, the same identity again only once its
 * own cooldown has passed. `greeted` is a `Map(identityId -> lastGreetedAt)` and is mutated here —
 * a single last-identity slot let two enrolled people bypass the cooldown completely, because each
 * arrival overwrote the other's timestamp and every swap read as "a new identity".
 *
 * Bounded to `max` entries so a long session cannot grow it without limit: the entry greeted longest
 * ago is evicted first (a greeting re-inserts its key, and a `Map` iterates in insertion order). The
 * cost of that bound is stated rather than hidden — an evicted identity is greeted once more the next
 * time it appears, which for 33 people in one room is the right trade against unbounded memory.
 */
export function shouldGreet(greeted, identityId, now, cooldownMs, max = GREETING_MEMORY_MAX) {
  if (!identityId) return false;
  const last = greeted.get(identityId);
  if (Number.isFinite(last) && now - last < cooldownMs) return false;
  greeted.delete(identityId);
  greeted.set(identityId, now);
  // Math.max(1, …): `keep` of 0 would make this loop delete the entry it just recorded.
  const keep = Number.isFinite(max) ? Math.max(1, Math.trunc(max)) : greeted.size;
  while (greeted.size > keep) greeted.delete(greeted.keys().next().value);
  return true;
}

/**
 * One-line state of the live event stream for the Context card, from what `subscribeEvents` reports.
 * It says what this page is doing — never that the server is broken, which the page cannot know.
 */
export function describeEventsStatus(info) {
  if (!info) return 'Live';
  if (info.unsupported) return 'Unavailable · this browser has no EventSource';
  if (!info.closed) return 'Reconnecting';  // the connection dropped; the platform retries it itself
  const seconds = Number.isFinite(info.retryInMs) ? Math.max(1, Math.round(info.retryInMs / 1000)) : null;
  return seconds === null ? 'Stream closed · retrying' : `Stream closed · retrying in ${seconds} s`;
}

/** Map a getUserMedia failure to the pill/context wording; only permission problems read as "blocked". */
export function describeCameraError(error) {
  const name = error?.name || '';
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') {
    return { pill: 'CAMERA BLOCKED', context: 'Permission blocked' };
  }
  return { pill: 'CAMERA UNAVAILABLE', context: name ? `Unavailable · ${name}` : 'Unavailable' };
}

// ---------- expression (Stage 1): a best-effort mood hint, never a fact ----------

/**
 * Wording tables for `Expression.dominant` (see domain/models.py EMOTIONS). Every label is hedged
 * ("wirkt …" / "looks …"): the engine reports how a face *appears*, never how someone *is* — and
 * nothing here may read as a finding, a diagnosis or a lie detector.
 */
const EXPRESSION_WORDS = Object.freeze({
  de: Object.freeze({
    prefix: 'wirkt ',
    labels: Object.freeze({ Happiness: 'fröhlich', Sadness: 'traurig', Anger: 'verärgert', Fear: 'ängstlich', Surprise: 'überrascht', Disgust: 'angewidert', Contempt: 'abschätzig', Neutral: 'neutral' }),
  }),
  en: Object.freeze({
    prefix: 'looks ',
    labels: Object.freeze({ Happiness: 'happy', Sadness: 'sad', Anger: 'angry', Fear: 'fearful', Surprise: 'surprised', Disgust: 'disgusted', Contempt: 'contemptuous', Neutral: 'neutral' }),
  }),
});

/** Tile tone per label: mint for a pleasant hint, neutral for neutral/surprise, amber for the rest; unknown labels stay neutral. */
const EXPRESSION_TONES = Object.freeze({ Happiness: 'ok', Neutral: 'muted', Surprise: 'muted', Sadness: 'warn', Anger: 'warn', Fear: 'warn', Disgust: 'warn', Contempt: 'warn' });

/**
 * Turn `DetectedFace.expression` (or null) into `{ label, tone, valence, arousal }` — or null when
 * there is nothing to say. Unknown labels are hedged too and never throw.
 */
export function describeExpression(expr, lang = 'de') {
  const dominant = typeof expr?.dominant === 'string' ? expr.dominant : '';
  if (!dominant) return null;
  // hasOwn, not `[lang] ||`: a lang of "__proto__" would otherwise resolve to Object.prototype and throw below.
  const words = Object.hasOwn(EXPRESSION_WORDS, lang) ? EXPRESSION_WORDS[lang] : EXPRESSION_WORDS.de;
  const word = Object.hasOwn(words.labels, dominant) ? words.labels[dominant] : dominant.toLowerCase();
  return {
    label: `${words.prefix}${word}`,
    tone: Object.hasOwn(EXPRESSION_TONES, dominant) ? EXPRESSION_TONES[dominant] : 'muted',
    valence: Number.isFinite(expr.valence) ? expr.valence : null,
    arousal: Number.isFinite(expr.arousal) ? expr.arousal : null,
  };
}

/** Bar geometry for valence/arousal: -1..1 → 0..100 (%), clamped; null when there is no number. */
export function axisPercent(value) {
  if (!Number.isFinite(value)) return null;
  return Math.round(Math.min(1, Math.max(-1, value)) * 50 + 50);
}

/** Signed two-decimal label for a valence/arousal axis ("+0.60", "-0.50"); -0.004 is "+0.00", never "-0.00"; null when there is no number. */
export function formatAxis(value) {
  if (!Number.isFinite(value)) return null;
  const v = Number(value.toFixed(2)) + 0; // + 0 turns -0 into 0 so the sign check below is honest
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
}

// ---------- expression dynamics (Stage 2): action entries + valence sparkline ----------

/**
 * Wording tables for `ActionEvent.action` (see domain/models.py ACTIONS). An action is a description of
 * a movement ("brief smile (0.9 s)"), never a claim about the person and never a finding. Timing is
 * quantized to the frame rate (~0.6 s), hence "brief" ≤ 1 s and "held" ≥ 5 s rather than any
 * sub-second vocabulary (which the loop cannot resolve). Both languages stay here so the view-model is bilingual; app.js picks
 * English (`EXPRESSION_LANG`).
 */
const ACTION_WORDS = Object.freeze({
  en: Object.freeze({
    brief: 'brief ',
    held: 'held ',
    labels: Object.freeze({ smile: 'smile', frown: 'frown', brow_raise: 'brow raise', brow_furrow: 'brow furrow', eye_squint: 'eye squint', eyes_wide: 'eyes wide', nose_wrinkle: 'nose wrinkle', lip_press: 'lip press' }),
  }),
  de: Object.freeze({
    brief: 'kurzes ',
    held: 'anhaltendes ',
    labels: Object.freeze({ smile: 'Lächeln', frown: 'Mundwinkel runter', brow_raise: 'Brauen hoch', brow_furrow: 'Stirnrunzeln', eye_squint: 'Augen zusammengekniffen', eyes_wide: 'Augen weit', nose_wrinkle: 'Nasenrümpfen', lip_press: 'Lippen gepresst' }),
  }),
});
const ACTION_BRIEF_MS = 1000;
const ACTION_HELD_MS = 5000;

/** "0.9 s" — one decimal; anything that is not a finite non-negative number reads as 0.0 s. */
export function formatDuration(ms) {
  const n = Number.isFinite(ms) && ms > 0 ? ms : 0;
  return `${(n / 1000).toFixed(1)} s`;
}

/**
 * Turn an SSE `action` payload into `{ label, tone, duration }` — or null when there is nothing to say.
 * Unknown actions are hedged the same way (raw name, `_` → space) and never throw; unknown languages fall back to German.
 */
export function describeAction(action, lang = 'de') {
  const name = typeof action?.action === 'string' ? action.action : '';
  if (!name) return null;
  // hasOwn, not `[lang] ||`: a lang of "__proto__" would otherwise resolve to Object.prototype and throw below.
  const words = Object.hasOwn(ACTION_WORDS, lang) ? ACTION_WORDS[lang] : ACTION_WORDS.de;
  const word = Object.hasOwn(words.labels, name) ? words.labels[name] : name.toLowerCase().replace(/_/g, ' ');
  const duration = formatDuration(action?.duration_ms);
  // The qualifier follows the *printed* value, so the text and the number never disagree:
  // 4999 ms prints "5.0 s" and therefore reads "held", not a bare "smile (5.0 s)".
  const shownMs = parseFloat(duration) * 1000;
  const qualifier = shownMs <= ACTION_BRIEF_MS ? words.brief : shownMs >= ACTION_HELD_MS ? words.held : '';
  return { label: `${qualifier}${word} (${duration})`, tone: 'muted', duration };
}

/**
 * Append a valence sample `{ v, at }` from the browser's own frames, bounded to `max` (oldest dropped).
 * Mutates and returns `samples`; a non-number is ignored. This is a view of what the page itself sent —
 * it never re-derives the server's affect history.
 */
export function pushSample(samples, value, at, max = 120) {
  if (!Number.isFinite(value)) return samples;
  samples.push({ v: value, at });
  // splice, not a shift loop: `while (length > max)` never terminates for a negative max.
  const keep = Number.isFinite(max) ? Math.max(0, Math.trunc(max)) : samples.length;
  if (samples.length > keep) samples.splice(0, samples.length - keep);
  return samples;
}

/** SVG polyline `points` for samples in a `w`×`h` box: -1..1 clamped, +1 at the top, newest at the right; '' when empty. */
export function sparklinePoints(samples, w, h) {
  if (!Array.isArray(samples) || !samples.length) return '';
  const n = samples.length;
  const round = (x) => String(Math.round(x * 10) / 10);
  return samples.map((s, i) => {
    const x = n === 1 ? 0 : (i / (n - 1)) * w;
    const v = Number.isFinite(s?.v) ? s.v : 0; // a junk sample sits on the zero line; `points` must never carry NaN
    const y = ((1 - Math.min(1, Math.max(-1, v))) / 2) * h;
    return `${round(x)},${round(y)}`;
  }).join(' ');
}

const ENTRY_MAX_AGE_MS = 10000;   // an SSE reconnect replays up to 200 buffered events (Last-Event-ID)
export const ACTION_LOG_MIN_MS = 5000;  // per-action display rate limit for the 8-slot event log

/**
 * Is this server entry news rather than replayed history? `EventSource` resumes with `Last-Event-ID`
 * after every drop (sleep, reload, Wi-Fi blip) and the server replays its buffer, so minutes-old
 * moods/actions would otherwise be logged with the current clock. Server and page run on the same
 * machine, so their wall clocks agree; an absent or unparsable `at` is never fresh, and an `at`
 * slightly in the future still is.
 */
export function isFreshEntry(entry, now = Date.now(), maxAgeMs = ENTRY_MAX_AGE_MS) {
  const at = typeof entry?.at === 'string' ? Date.parse(entry.at) : NaN; // a number would parse as a year
  return Number.isFinite(at) && now - at < maxAgeMs;
}

/**
 * Display rate limit for action entries: the same action label is logged again only after `minMs`.
 * `seen` is a `Map` (an action label can never reach `Object.prototype`) and is mutated. This does not
 * re-derive anything — the server stays the source of truth; it only keeps a talking face's brow and
 * lip actions from evicting the eight slots that hold errors, greetings and enrollments.
 */
export function allowActionEntry(seen, action, now, minMs = ACTION_LOG_MIN_MS) {
  const name = typeof action === 'string' && action ? action : '';
  if (!name) return false;
  const last = seen.get(name);
  if (Number.isFinite(last) && now - last < minMs) return false;
  seen.set(name, now);
  return true;
}
