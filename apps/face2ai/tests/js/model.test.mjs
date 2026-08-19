import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  ACTION_LOG_MIN_MS,
  GREETING_MEMORY_MAX,
  allowActionEntry,
  axisPercent,
  describeAction,
  describeCameraError,
  describeEvent,
  describeEventsStatus,
  describeExpression,
  formatAxis,
  formatDuration,
  isFreshEntry,
  offlineView,
  projectBox,
  pushSample,
  shouldGreet,
  sparklinePoints,
  transitionKey,
} from '../../src/face2ai_app/static/js/model.js';

const box = { top: 100, right: 400, bottom: 300, left: 200 };

test('projectBox maps a box from the sent frame into a cover-fitted view of the same aspect', () => {
  // Frame that was actually sent to the API (downscaled snapshot), view twice as large.
  const projected = projectBox(box, { width: 640, height: 360 }, { width: 1280, height: 720 });
  assert.deepEqual(projected, { x: 400, y: 200, width: 400, height: 400 });
});

test('projectBox uses the snapshot dimensions, not the camera resolution', () => {
  // Regression: boxes are returned in the coordinate space of the JPEG the client sent
  // (640 wide), never in the camera's native 1280 space.
  const fromFrame = projectBox(box, { width: 640, height: 360 }, { width: 640, height: 360 });
  const fromCamera = projectBox(box, { width: 1280, height: 720 }, { width: 640, height: 360 });
  assert.deepEqual(fromFrame, { x: 200, y: 100, width: 200, height: 200 });
  assert.notDeepEqual(fromFrame, fromCamera);
});

test('projectBox centres the overflow axis like object-fit: cover', () => {
  // 16:9 frame shown in a square view: scale by height, crop width, offset x by -(1280-720)/2.
  const projected = projectBox(box, { width: 640, height: 360 }, { width: 720, height: 720 });
  assert.equal(projected.height, 400);
  assert.equal(projected.width, 400);
  assert.equal(projected.y, 200);
  assert.equal(projected.x, 200 * 2 - 280);
});

test('describeEvent follows the server event instead of re-deriving state', () => {
  const known = describeEvent({
    state: 'KNOWN',
    can_enroll: false,
    message: null,
    faces: [{ box, matched: true, identity_id: 'id-1', display_name: 'Ada', match_distance: 0.31 }],
  });
  assert.equal(known.state, 'KNOWN');
  assert.equal(known.name, 'Ada');
  assert.equal(known.cls, 'known');
  assert.equal(known.canEnroll, false);
  assert.equal(known.faces, 1);
  assert.equal(known.distance, 0.31);
  assert.equal(known.primary.identity_id, 'id-1');
});

test('describeEvent surfaces can_enroll and server message verbatim', () => {
  const multiple = describeEvent({
    state: 'MULTIPLE_FACES',
    can_enroll: false,
    message: 'Enrollment requires exactly one visible face.',
    faces: [{ box, matched: false }, { box, matched: false }],
  });
  assert.equal(multiple.canEnroll, false);
  assert.equal(multiple.message, 'Enrollment requires exactly one visible face.');
  assert.equal(multiple.faces, 2);

  const unknown = describeEvent({ state: 'UNKNOWN', can_enroll: true, faces: [{ box, matched: false }] });
  assert.equal(unknown.canEnroll, true);
  assert.equal(unknown.cls, 'unknown');
  assert.equal(unknown.distance, null);
  assert.ok(unknown.message.length > 0);
});

test('describeEvent tolerates missing fields and unknown states without inventing data', () => {
  const empty = describeEvent({ state: 'NO_FACE' });
  assert.equal(empty.faces, 0);
  assert.equal(empty.canEnroll, false);
  assert.equal(empty.primary, null);
  const weird = describeEvent({ state: 'SOMETHING_NEW', faces: [] });
  assert.equal(weird.state, 'SOMETHING_NEW');
  assert.equal(weird.cls, 'error');
});

test('transitionKey changes only when state or identity changes', () => {
  const a = transitionKey({ state: 'KNOWN', faces: [{ identity_id: 'id-1' }] });
  const b = transitionKey({ state: 'KNOWN', faces: [{ identity_id: 'id-1', match_distance: 0.4 }] });
  const c = transitionKey({ state: 'KNOWN', faces: [{ identity_id: 'id-2' }] });
  const d = transitionKey({ state: 'UNKNOWN', faces: [{}] });
  assert.equal(a, b);
  assert.notEqual(a, c);
  assert.notEqual(a, d);
});

test('shouldGreet greets a new identity immediately and the same identity only after cooldown', () => {
  const cooldown = 15_000;
  const greeted = new Map();
  assert.equal(shouldGreet(greeted, 'id-1', 1_000, cooldown), true);
  assert.equal(shouldGreet(greeted, 'id-1', 5_000, cooldown), false);
  assert.equal(shouldGreet(greeted, 'id-1', 16_001, cooldown), true);
  assert.equal(shouldGreet(greeted, 'id-2', 16_100, cooldown), true);
  assert.equal(shouldGreet(new Map(), null, 99_000, cooldown), false); // no identity: nothing to greet
});

test('the cooldown is kept per identity, so two enrolled people cannot greet each other free', () => {
  // The one-slot version stored only the last identity, so Ada arriving reset Grace's cooldown and
  // back: two people alternating in front of the camera were greeted on every single swap.
  const cooldown = 15_000;
  const greeted = new Map();
  assert.equal(shouldGreet(greeted, 'ada', 1_000, cooldown), true);
  assert.equal(shouldGreet(greeted, 'grace', 1_400, cooldown), true);
  for (const [identity, at] of [['ada', 1_800], ['grace', 2_200], ['ada', 2_600], ['grace', 3_000]]) {
    assert.equal(shouldGreet(greeted, identity, at, cooldown), false, `${identity} @ ${at}`);
  }
  // Each of them is greeted again only once its own cooldown has passed.
  assert.equal(shouldGreet(greeted, 'ada', 16_000, cooldown), true);
  assert.equal(shouldGreet(greeted, 'grace', 16_000, cooldown), false); // greeted 400 ms later than ada
  assert.equal(shouldGreet(greeted, 'grace', 16_400, cooldown), true);
});

test('the greeting memory is bounded and drops the identity greeted longest ago', () => {
  const cooldown = 60_000;
  const greeted = new Map();
  for (let i = 0; i < GREETING_MEMORY_MAX; i += 1) {
    assert.equal(shouldGreet(greeted, `id-${i}`, 1_000 + i, cooldown), true);
  }
  assert.equal(greeted.size, GREETING_MEMORY_MAX);
  assert.equal(shouldGreet(greeted, 'id-0', 2_000, cooldown), false); // still inside its own cooldown

  assert.equal(shouldGreet(greeted, 'id-new', 2_100, cooldown), true);
  assert.equal(greeted.size, GREETING_MEMORY_MAX, 'the map must not grow past its cap');
  assert.equal(greeted.has('id-0'), false, 'the identity greeted longest ago is evicted first');
  assert.equal(greeted.has('id-1'), true);
  // The visible cost of the bound, stated rather than hidden: an evicted identity is greeted once more.
  assert.equal(shouldGreet(greeted, 'id-0', 2_200, cooldown), true);
});

test('describeEvent pins fallback messages and names known people in the KNOWN message', () => {
  const known = describeEvent({ state: 'KNOWN', faces: [{ box, matched: true, identity_id: 'id-1', display_name: 'Ada' }] });
  assert.equal(known.message, 'Identity matched: Ada.');
  assert.equal(known.label, 'KNOWN');
  const unknown = describeEvent({ state: 'UNKNOWN', can_enroll: true, faces: [{ box, matched: false }] });
  assert.equal(unknown.message, 'One unknown face detected. Enrollment is available.');
  assert.equal(describeEvent().state, 'ERROR');
  assert.equal(describeEvent({ state: 42 }).state, 'ERROR');
  assert.equal(describeEvent({ state: 'MULTIPLE_FACES' }).label, 'MULTIPLE FACES');
});

test('shouldGreet cooldown boundary is inclusive', () => {
  assert.equal(shouldGreet(new Map([['id-1', 1_000]]), 'id-1', 16_000, 15_000), true);
  assert.equal(shouldGreet(new Map([['id-1', 1_000]]), 'id-1', 15_999, 15_000), false);
});

test('describeEventsStatus says what the client is doing, never that the server is broken', () => {
  assert.equal(describeEventsStatus(null), 'Live');
  assert.equal(describeEventsStatus({ unsupported: true }), 'Unavailable · this browser has no EventSource');
  // A connection the browser left CLOSED never retries on its own — the shell says when it will try again.
  assert.equal(describeEventsStatus({ unsupported: false, closed: true, retryInMs: 4_000 }), 'Stream closed · retrying in 4 s');
  assert.equal(describeEventsStatus({ unsupported: false, closed: true, retryInMs: 400 }), 'Stream closed · retrying in 1 s');
  assert.equal(describeEventsStatus({ unsupported: false, closed: true, retryInMs: null }), 'Stream closed · retrying');
  // A dropped connection: the platform is already retrying, so the shell only reports it.
  assert.equal(describeEventsStatus({ unsupported: false, closed: false, retryInMs: null }), 'Reconnecting');
});

test('offlineView distinguishes camera-off from engine-unavailable and never claims a face', () => {
  const off = offlineView();
  assert.equal(off.state, 'OFFLINE');
  assert.equal(off.cls, '');
  assert.equal(off.faces, 0);
  assert.equal(off.canEnroll, false);
  const engineDown = offlineView('ModuleNotFoundError: dlib');
  assert.equal(engineDown.cls, 'error');
  assert.match(engineDown.sub, /ModuleNotFoundError: dlib/);
});

test('describeCameraError only calls permission problems "blocked"', () => {
  assert.equal(describeCameraError({ name: 'NotAllowedError' }).pill, 'CAMERA BLOCKED');
  assert.equal(describeCameraError({ name: 'NotReadableError' }).pill, 'CAMERA UNAVAILABLE');
  assert.match(describeCameraError({ name: 'NotFoundError' }).context, /NotFoundError/);
  assert.equal(describeCameraError(undefined).pill, 'CAMERA UNAVAILABLE');
});

// ---------- expression (Stage 1): a hint, never a fact ----------

test('describeExpression speaks in hedged German and never claims certainty', () => {
  const d = describeExpression({ dominant: 'Happiness', scores: { Happiness: 0.9 }, valence: 0.6, arousal: 0.1 }, 'de');
  assert.equal(d.label, 'wirkt fröhlich');
  assert.equal(d.tone, 'ok');
  assert.equal(describeExpression(null, 'de'), null);
  assert.equal(describeExpression({ dominant: 'Neutral', scores: {} }, 'en').label, 'looks neutral');
});

test('describeExpression maps every emotion label to hedged wording in both languages', () => {
  const de = { Happiness: 'wirkt fröhlich', Sadness: 'wirkt traurig', Anger: 'wirkt verärgert', Fear: 'wirkt ängstlich', Surprise: 'wirkt überrascht', Disgust: 'wirkt angewidert', Contempt: 'wirkt abschätzig', Neutral: 'wirkt neutral' };
  const en = { Happiness: 'looks happy', Sadness: 'looks sad', Anger: 'looks angry', Fear: 'looks fearful', Surprise: 'looks surprised', Disgust: 'looks disgusted', Contempt: 'looks contemptuous', Neutral: 'looks neutral' };
  for (const [dominant, label] of Object.entries(de)) assert.equal(describeExpression({ dominant }, 'de').label, label);
  for (const [dominant, label] of Object.entries(en)) assert.equal(describeExpression({ dominant }, 'en').label, label);
  // Every label is hedged; none reads as a finding.
  for (const lang of ['de', 'en']) {
    for (const dominant of Object.keys(de)) {
      const { label } = describeExpression({ dominant }, lang);
      assert.match(label, /^(wirkt|looks) /);
      assert.doesNotMatch(label, /\b(ist|is|erkannt|detected|recognized)\b/);
    }
  }
});

test('describeExpression tone: happiness ok, neutral/surprise muted, the rest warn', () => {
  assert.equal(describeExpression({ dominant: 'Happiness' }).tone, 'ok');
  assert.equal(describeExpression({ dominant: 'Neutral' }).tone, 'muted');
  assert.equal(describeExpression({ dominant: 'Surprise' }).tone, 'muted');
  for (const dominant of ['Sadness', 'Anger', 'Fear', 'Disgust', 'Contempt']) assert.equal(describeExpression({ dominant }).tone, 'warn');
});

test('describeExpression tolerates unknown labels, unknown languages and missing numbers without throwing', () => {
  const odd = describeExpression({ dominant: 'Boredom' }, 'de');
  assert.equal(odd.label, 'wirkt boredom');
  assert.equal(odd.tone, 'muted');
  assert.equal(describeExpression({ dominant: 'Happiness' }, 'fr').label, 'wirkt fröhlich'); // falls back to German
  assert.equal(describeExpression({ dominant: 'Happiness' }).label, 'wirkt fröhlich'); // default language
  assert.equal(describeExpression({}), null);
  assert.equal(describeExpression({ dominant: '' }), null);
  assert.equal(describeExpression({ dominant: 42 }), null);
  const bare = describeExpression({ dominant: 'Neutral' });
  assert.equal(bare.valence, null);
  assert.equal(bare.arousal, null);
  const full = describeExpression({ dominant: 'Neutral', valence: -0.25, arousal: 'x' });
  assert.equal(full.valence, -0.25);
  assert.equal(full.arousal, null);
});

test('formatAxis prints a signed two-decimal label and never a negative zero', () => {
  assert.equal(formatAxis(-0.004), '+0.00');
  assert.equal(formatAxis(0.6), '+0.60');
  assert.equal(formatAxis(-0.5), '-0.50');
  assert.equal(formatAxis(0), '+0.00');
  assert.equal(formatAxis(null), null);
  assert.equal(formatAxis(NaN), null);
});

test('axisPercent maps -1..1 onto 0..100 and clamps', () => {
  assert.equal(axisPercent(-1), 0);
  assert.equal(axisPercent(0), 50);
  assert.equal(axisPercent(1), 100);
  assert.equal(axisPercent(0.6), 80);
  assert.equal(axisPercent(2), 100);
  assert.equal(axisPercent(-3), 0);
  assert.equal(axisPercent(null), null);
  assert.equal(axisPercent(undefined), null);
  assert.equal(axisPercent(NaN), null);
});

// ---------- expression dynamics (Stage 2): action entries come from the server's SSE `action` events ----------

test('describeAction is hedged, quantised and bilingual', () => {
  const brief = describeAction({ action: 'smile', duration_ms: 900, peak: 0.9 }, 'en');
  assert.equal(brief.label, 'brief smile (0.9 s)');
  assert.equal(brief.tone, 'muted');
  assert.equal(brief.duration, '0.9 s');
  assert.equal(describeAction({ action: 'brow_raise', duration_ms: 2300 }, 'en').label, 'brow raise (2.3 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 6000 }, 'en').label, 'held smile (6.0 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 900 }, 'de').label, 'kurzes Lächeln (0.9 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 6000 }, 'de').label, 'anhaltendes Lächeln (6.0 s)');
  assert.equal(describeAction({ action: 'wink', duration_ms: 900 }, 'en').label, 'brief wink (0.9 s)'); // unknown label never throws
  assert.equal(describeAction({ action: 'lip_press', duration_ms: 1200 }, 'fr').label, 'Lippen gepresst (1.2 s)'); // unknown language falls back to German
  assert.equal(describeAction(null, 'en'), null);
  assert.equal(describeAction({}, 'en'), null);
  assert.equal(describeAction({ action: 42 }, 'en'), null);
});

test('describeAction never reads as a finding: every action label in both languages is free of "is"/"detected"', () => {
  for (const lang of ['de', 'en']) {
    for (const action of ['smile', 'frown', 'brow_raise', 'brow_furrow', 'eye_squint', 'eyes_wide', 'nose_wrinkle', 'lip_press']) {
      for (const duration_ms of [400, 2000, 7000]) {
        const { label } = describeAction({ action, duration_ms }, lang);
        assert.doesNotMatch(label, /\b(is|ist|erkannt|detected|recognized|smiling|frowning)\b/, `${lang} ${action}: ${label}`);
        assert.match(label, /\(\d+\.\d s\)$/, label);
      }
    }
  }
});

test('formatDuration prints one decimal in seconds and tolerates junk', () => {
  assert.equal(formatDuration(900), '0.9 s');
  assert.equal(formatDuration(6000), '6.0 s');
  assert.equal(formatDuration(0), '0.0 s');
  assert.equal(formatDuration(1250), '1.3 s');
  assert.equal(formatDuration(-5), '0.0 s');
  assert.equal(formatDuration(null), '0.0 s');
  assert.equal(formatDuration('x'), '0.0 s');
});

test('sparkline maps -1..1 samples into the box, newest right', () => {
  const s = pushSample([], 0, 1); pushSample(s, 1, 2); pushSample(s, -1, 3);
  assert.equal(sparklinePoints(s, 120, 24), '0,12 60,0 120,24');
  assert.equal(sparklinePoints([], 120, 24), '');
  assert.equal(sparklinePoints([{ v: 0.5, at: 1 }], 120, 24), '0,6'); // a single sample sits at the left edge, never NaN
  assert.equal(sparklinePoints([{ v: 3, at: 1 }, { v: -3, at: 2 }], 120, 24), '0,0 120,24'); // clamped
  assert.equal(pushSample(Array.from({ length: 120 }, (_, i) => ({ v: 0, at: i })), 0.5, 999, 120).length, 120); // bounded
  const bounded = pushSample(Array.from({ length: 3 }, (_, i) => ({ v: 0, at: i })), 0.5, 99, 3);
  assert.deepEqual(bounded, [{ v: 0, at: 1 }, { v: 0, at: 2 }, { v: 0.5, at: 99 }]); // drops the oldest, keeps the newest
  assert.equal(pushSample([], NaN, 1).length, 0); // a non-number is not a sample
});

test('describeAction and describeExpression survive a hostile language key', () => {
  // `ACTION_WORDS['__proto__']` is Object.prototype: truthy, but without `labels` — `[lang] ||` threw here.
  assert.equal(describeAction({ action: 'smile', duration_ms: 900 }, '__proto__').label, 'kurzes Lächeln (0.9 s)');
  assert.equal(describeExpression({ dominant: 'Happiness' }, '__proto__').label, 'wirkt fröhlich');
  assert.equal(describeAction({ action: 'smile', duration_ms: 900 }, 'constructor').label, 'kurzes Lächeln (0.9 s)');
});

test('the brief/held qualifier follows the printed duration, so text and number never disagree', () => {
  // 4999 ms prints "5.0 s": calling that a bare "smile (5.0 s)" while 5000 ms reads "held smile (5.0 s)"
  // would put two different phrasings on the same visible number.
  assert.equal(describeAction({ action: 'smile', duration_ms: 4999 }, 'en').label, 'held smile (5.0 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 4949 }, 'en').label, 'smile (4.9 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 1049 }, 'en').label, 'brief smile (1.0 s)');
  assert.equal(describeAction({ action: 'smile', duration_ms: 1051 }, 'en').label, 'smile (1.1 s)');
});

test('pushSample stays bounded for every max, including 0 and negatives', () => {
  // A `while (length > max) shift()` loop never terminates for a negative max — a regression here
  // hangs this test instead of failing it.
  assert.deepEqual(pushSample([{ v: 0, at: 1 }], 0.5, 2, -1), []);
  assert.deepEqual(pushSample([{ v: 0, at: 1 }], 0.5, 2, 0), []);
  assert.deepEqual(pushSample([{ v: 0, at: 1 }], 0.5, 2, 1), [{ v: 0.5, at: 2 }]);
});

test('sparklinePoints never emits NaN coordinates or throws on junk', () => {
  // setAttribute('points', …) swallows "0,NaN" silently and draws nothing.
  assert.equal(sparklinePoints([{ v: 'x' }, { v: 0 }], 120, 24), '0,12 120,12');
  assert.equal(sparklinePoints([null, { v: 1 }], 120, 24), '0,12 120,0');
  assert.equal(sparklinePoints(null, 120, 24), '');
  assert.equal(sparklinePoints(undefined, 120, 24), '');
  assert.doesNotMatch(sparklinePoints([{ v: NaN }, { v: 0.5 }], 120, 24), /NaN/);
});

test('isFreshEntry keeps replayed history out of the live event log', () => {
  const now = Date.parse('2026-08-18T12:00:00Z');
  assert.equal(isFreshEntry({ at: '2026-08-18T12:00:00Z' }, now), true);
  assert.equal(isFreshEntry({ at: '2026-08-18T11:59:51Z' }, now), true);   // 9 s old: still news
  assert.equal(isFreshEntry({ at: '2026-08-18T11:59:50Z' }, now), false);  // 10 s: the replay boundary
  assert.equal(isFreshEntry({ at: '2026-08-18T11:45:00Z' }, now), false);  // a reconnect replays minutes-old frames
  assert.equal(isFreshEntry({ at: '2026-08-18T12:00:01Z' }, now), true);   // server marginally ahead
  assert.equal(isFreshEntry({}, now), false);                              // no timestamp: never treated as news
  assert.equal(isFreshEntry({ at: 'not a date' }, now), false);
  assert.equal(isFreshEntry({ at: 12345 }, now), false); // Date.parse would read a number as a year 12345 → "fresh"
  assert.equal(isFreshEntry(null, now), false);
});

test('allowActionEntry rate-limits one action label without touching the others', () => {
  const seen = new Map();
  assert.equal(allowActionEntry(seen, 'smile', 1000), true);
  assert.equal(allowActionEntry(seen, 'smile', 1000 + ACTION_LOG_MIN_MS - 1), false); // a talking face would flush the log
  assert.equal(allowActionEntry(seen, 'brow_raise', 1000), true);                     // a different action is not blocked
  assert.equal(allowActionEntry(seen, 'smile', 1000 + ACTION_LOG_MIN_MS), true);
  assert.equal(allowActionEntry(seen, '', 9999), false);
  assert.equal(allowActionEntry(seen, undefined, 9999), false);
  assert.equal(allowActionEntry(seen, '__proto__', 1000), true);                      // a Map: no prototype to reach
  assert.equal(allowActionEntry(seen, '__proto__', 1001), false);
});
