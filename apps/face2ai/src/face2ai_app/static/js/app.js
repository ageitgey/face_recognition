// Face2AI browser shell. Consumes RecognitionEvent/SystemStatus from the local API and renders
// them; it never invents identity, certainty, or activity. Product loop kept intact:
// UNKNOWN -> explicit LEARN (with consent) -> leave -> return -> KNOWN -> greeting.
import * as api from './api.js';
import { CameraController } from './camera.js';
import { decryptText, installAtmosphere, installMagnets, installSpotlights } from './effects.js';
import { subscribeEvents } from './events.js';
import { allowActionEntry, axisPercent, describeAction, describeCameraError, describeEvent, describeEventsStatus, describeExpression, formatAxis, isFreshEntry, offlineView, pushSample, shouldGreet, sparklinePoints, transitionKey } from './model.js';

const RECOGNIZE_INTERVAL_MS = 450;
const RECOGNIZE_ERROR_BACKOFF_MS = 1500;
const STATUS_INTERVAL_MS = 5000;
const MAX_EVENTS = 8;          // mood + action entries arrive from the server stream too; nothing is re-derived here, but an action label is *displayed* at most every ACTION_LOG_MIN_MS so a talking face cannot flush the log
const AFFECT_SAMPLES = 120;    // valence samples kept for the tile sparkline (~70 s at the ~1.7 fps loop), from this page's own frames only
const EXPRESSION_LANG = 'en';  // wording table in model.js; the shell is English, so the tile says "looks …" (hedged, never a fact)
const TIME_FORMAT = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });

const els = Object.fromEntries([...document.querySelectorAll('[id]')].map((el) => [el.id, el]));
const camera = new CameraController(els.camera, els.overlay);

const state = {
  cameraOn: false,
  starting: false,
  paused: false,
  session: 0,          // incremented on every camera start/stop; in-flight results from older sessions are dropped
  inflight: false,
  timer: null,
  frame: 0,
  latest: null,        // { blob, event }: the frame that was sent and the event it produced, published together
  lastTransition: null,
  engine: { known: false, available: false },
  cooldownMs: 0,
  greeted: new Map(),  // identity id -> last greeting; one cooldown per person (model.js: shouldGreet)
  events: 0,
  lastErrorText: null,
  enrolling: false,     // true while the enroll dialog is open or the enroll request is in flight
  enrollFrozen: null,   // the { blob, event } pair frozen when the dialog opened
  agentConnected: false, // a voice agent subscribes to /api/events; it then owns the spoken greeting
  expression: { available: false, reason: null, enabled: false }, // opt-in mood hints (Stage 1), mirrored from /api/status
  affect: [],            // valence samples from this page's own frames → tile sparkline (Stage 2); mood/action log entries come from the server stream
  actionLogged: new Map(), // action label -> last time it was shown in the log (display rate limit, not a re-derivation)
  eventsWarned: false,   // one log entry per outage (retries flood the handler); cleared when the stream opens again
};

const IDLE_VIEW = describeEvent({ state: 'NO_FACE' });

// ---------- small UI helpers ----------

function nowTime() {
  return TIME_FORMAT.format(new Date());
}

/** Visible pill text is animated; assistive tech gets the final text once via the sr-only announcer. */
function setPill(pill, textEl, text, mode = '') {
  pill.classList.remove('online', 'warn', 'error');
  if (mode) pill.classList.add(mode);
  decryptText(textEl, text, { duration: 320 });
  els.statusAnnouncer.textContent = text;
}

let toastTimer = null;
function toast(message) {
  els.toast.textContent = message;
  els.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), 2800);
}

function addEvent(title, detail = '') {
  els.emptyEvent?.remove();
  const item = document.createElement('article');
  item.className = 'event';
  const time = document.createElement('div');
  time.className = 'event-time';
  time.textContent = nowTime();
  const heading = document.createElement('div');
  heading.className = 'event-title';
  heading.textContent = title;
  const body = document.createElement('div');
  body.className = 'event-detail';
  body.textContent = detail;
  item.append(time, heading, body);
  els.eventList.prepend(item);
  state.events += 1;
  els.eventCount.textContent = `${state.events} EVENT${state.events === 1 ? '' : 'S'}`;
  while (els.eventList.children.length > MAX_EVENTS) els.eventList.lastElementChild.remove();
}

function setRecognitionValue(text, muted) {
  els.recognitionValue.textContent = text;
  els.recognitionValue.classList.toggle('muted', muted);
}

/** Loop-level indicator: scan line + chip + context badge reflect whether recognition is running. */
function setLoopIndicator() {
  const live = state.cameraOn && !state.paused && state.engine.available && !state.enrolling;
  els.stage.classList.toggle('live', live);
  els.processingChip.textContent = !state.cameraOn ? 'IDLE' : state.paused ? 'PAUSED' : state.enrolling ? 'LEARNING' : live ? 'SCANNING' : 'ENGINE OFFLINE';
  els.contextHint.textContent = !state.cameraOn ? 'OFFLINE' : state.paused ? 'PAUSED' : live ? 'LIVE' : 'DEGRADED';
}

// ---------- identity card ----------

function renderIdentity(view) {
  decryptText(els.identityName, view.name, { duration: 520 });
  els.identityAnnouncer.textContent = `${view.name}. ${view.sub}.`;
  els.identityState.className = `identity-state ${view.cls}`.trim();
  decryptText(els.identityStateText, view.sub, { duration: 360 });
  els.identityMini.textContent = view.label;
  els.distanceValue.textContent = view.distance === null ? '—' : view.distance.toFixed(3);
  els.distanceValue.classList.toggle('muted', view.distance === null);
  els.facesValue.textContent = String(view.faces);
  els.faceCountChip.textContent = `${view.faces} FACE${view.faces === 1 ? '' : 'S'}`;
  els.learnButton.disabled = !(view.canEnroll && state.cameraOn && !state.paused && !state.enrolling);
}

// ---------- expression tile (opt-in mood hint, never a fact) ----------

function setExpressionTile(text, tone, described = null) {
  els.expressionValue.textContent = text;
  els.expressionValue.className = `metric-value ${tone}`.trim();
  const show = described !== null && (described.valence !== null || described.arousal !== null);
  els.moodBars.hidden = !show;
  if (!show) return;
  for (const [axis, fill, num] of [[described.valence, els.valenceFill, els.valenceValue], [described.arousal, els.arousalFill, els.arousalValue]]) {
    const pct = axisPercent(axis);
    fill.style.width = pct === null ? '0%' : `${pct}%`;
    num.textContent = formatAxis(axis) ?? '—';
  }
}

/** Per recognize result: one face → its hedged expression; several, none, or no expression → nothing to say. */
function renderExpression(event) {
  if (!state.expression.enabled) { setExpressionTile('off', 'muted'); return; }
  const faces = Array.isArray(event?.faces) ? event.faces : [];
  const described = faces.length === 1 ? describeExpression(faces[0].expression, EXPRESSION_LANG) : null;
  if (!described) {
    setExpressionTile('—', 'muted');
  } else {
    setExpressionTile(described.label, described.tone, described);
  }
  // Sparkline: this page's own valence readings, newest right; frames without a reading leave a gap in time, not a point.
  // Mood / action entries for the event stream are NOT derived here — they arrive from the server stream (see subscribeEvents below).
  if (Number.isFinite(described?.valence)) {
    pushSample(state.affect, described.valence, Date.now(), AFFECT_SAMPLES);
    renderSparkline();
  }
}

/** Tile sparkline from `state.affect`; hidden until there are at least two samples to draw a line between. */
function renderSparkline() {
  els.valenceLine.setAttribute('points', sparklinePoints(state.affect, 120, 24));
  els.valenceSpark.hidden = state.affect.length < 2;
}

/** Nothing being recognized right now (camera off, paused, error, toggle): the tile shows no reading and the sparkline restarts. */
function clearExpression() {
  state.affect = [];
  renderSparkline();
  setExpressionTile(state.expression.enabled ? '—' : 'off', 'muted');
}

/** /api/status → button + tile state (the server is the only source of truth; a click never sets it). */
function applyExpression(available, reason, enabled) {
  const wasEnabled = state.expression.enabled;
  state.expression = { available: available === true, reason: reason || null, enabled: enabled === true };
  els.expressionButton.disabled = !state.expression.available;
  els.expressionLabel.textContent = state.expression.enabled ? 'Expression: on' : 'Expression: off';
  els.expressionButton.setAttribute('aria-pressed', String(state.expression.enabled));
  els.expressionButton.title = state.expression.available
    ? (state.expression.enabled ? 'Turn expression hints off' : 'Turn expression hints on — a per-frame hint, nothing is persisted')
    : `Not available · ${state.expression.reason || 'expression engine unavailable'}`;
  if (state.expression.enabled !== wasEnabled) clearExpression(); // switched either way: the reading restarts
}

async function toggleExpression() {
  const enable = !state.expression.enabled;
  els.expressionButton.disabled = true;
  try {
    const result = await api.setExpression(enable);
    addEvent(result.enabled ? 'Expression on' : 'Expression off', result.enabled
      ? 'Expression hints are shown per frame ("looks …"). Nothing is persisted.'
      : 'No more expression hints; results carry identity only again.');
  } catch (error) {
    // 409 (engine unavailable) or any other failure: surface the server's detail, claim nothing.
    toast(`Expression not available: ${error.message}`);
    addEvent('Expression not available', error.message);
  } finally {
    await refreshStatus(); // button label/tile follow /api/status, not the click
  }
}

// ---------- engine / status ----------

function applyEngine(available, reason) {
  const changed = !state.engine.known || state.engine.available !== available;
  state.engine = { known: true, available };
  if (available) {
    setPill(els.enginePill, els.engineStatus, 'ENGINE READY', 'online');
    els.engineContext.textContent = 'Ready';
    if (!state.cameraOn) setRecognitionValue('Ready', false);
  } else {
    setPill(els.enginePill, els.engineStatus, 'ENGINE OFFLINE', 'warn');
    els.engineContext.textContent = reason ? `Offline · ${reason}` : 'Offline';
    setRecognitionValue('Offline', true);
    camera.clearOverlay();
    state.latest = null;
    if (state.cameraOn) renderIdentity(offlineView(reason || 'engine unavailable'));
    els.learnButton.disabled = true;
    clearExpression();
  }
  setLoopIndicator();
  if (changed) {
    addEvent(available ? 'Engine ready' : 'Engine offline', available ? 'Face recognition engine is loaded.' : (reason || 'Face recognition engine is not available on this runtime.'));
    if (!available) toast('Face recognition engine is not ready on this runtime.');
    if (available && state.cameraOn && !state.paused) {
      renderIdentity(IDLE_VIEW);
      schedule(0);
    }
  }
}

async function refreshStatus() {
  try {
    const status = await api.getStatus();
    state.cooldownMs = Math.max(0, Number(status.greeting_cooldown_seconds) || 0) * 1000;
    els.cooldownContext.textContent = `${Math.round(state.cooldownMs / 1000)} s`;
    els.identityCount.textContent = String(status.identity_count ?? '—');
    applyAgent(status.agent_connected === true);
    applyExpression(status.expression_available, status.expression_reason, status.expression_enabled);
    applyEngine(status.engine_available === true, status.engine_reason);
  } catch (error) {
    setPill(els.enginePill, els.engineStatus, 'API OFFLINE', 'error');
    els.engineContext.textContent = `API unreachable · ${error.message}`;
    setRecognitionValue('Offline', true);
    els.learnButton.disabled = true;
    els.expressionButton.disabled = true;
    if (state.engine.known && state.engine.available) addEvent('API unreachable', error.message);
    state.engine = { known: true, available: false };
    setLoopIndicator();
  }
}

function applyAgent(connected) {
  const changed = connected !== state.agentConnected;
  state.agentConnected = connected;
  els.agentContext.textContent = connected ? 'Connected · owns greetings' : 'Not connected';
  if (changed) addEvent(connected ? 'Voice agent connected' : 'Voice agent disconnected', connected ? 'Spoken greetings are left to the agent; the browser stays silent.' : 'Browser speech greeting is active again.');
}

// ---------- recognition loop ----------

function loopAllowed() {
  return state.cameraOn && !state.paused && state.engine.available && !state.enrolling && !document.hidden;
}

function schedule(delay = RECOGNIZE_INTERVAL_MS) {
  clearTimeout(state.timer);
  if (!loopAllowed()) return;
  state.timer = setTimeout(tick, delay);
}

async function tick() {
  if (state.inflight) { schedule(); return; }
  if (!loopAllowed()) return;
  const session = state.session;
  state.inflight = true;
  let delay = RECOGNIZE_INTERVAL_MS;
  try {
    const blob = await camera.snapshot();
    if (!blob || session !== state.session) return;
    state.frame += 1;
    els.frameInfo.textContent = `FRAME ${String(state.frame).padStart(4, '0')}`;
    // Nothing is published here: a frame whose event has not arrived yet must never become "the
    // latest frame", or a LEARN click landing inside this await freezes it with the previous event.
    const { event, agentConnected } = await api.recognize(blob);
    if (session !== state.session || state.paused || state.enrolling) return; // stale result: drop it
    applyAgent(agentConnected); // per-frame ownership signal, fresher than the 5 s status poll
    handleRecognition(blob, event);
  } catch (error) {
    if (session !== state.session || state.paused) return;
    delay = RECOGNIZE_ERROR_BACKOFF_MS;
    handleRecognitionError(error);
  } finally {
    state.inflight = false;
    if (session === state.session) schedule(delay);
  }
}

/** The frame and the event it produced are published in one assignment; they must never drift apart. */
function handleRecognition(blob, event) {
  state.latest = { blob, event };
  state.lastErrorText = null;
  const view = describeEvent(event);
  camera.drawFaces(Array.isArray(event.faces) ? event.faces : []);
  renderIdentity(view);
  renderExpression(event);
  setRecognitionValue('Live', false);

  const key = transitionKey(event);
  const transitioned = key !== state.lastTransition;
  if (transitioned) {
    state.lastTransition = key;
    addEvent(view.label, view.message);
  }
  if (view.state === 'KNOWN' && view.primary?.identity_id) {
    if (state.agentConnected) {
      // The agent owns spoken greetings; log the hand-off once per KNOWN transition, claim nothing about what it says.
      if (transitioned) addEvent('Greeting left to voice agent', `${view.name} recognized; the browser stays silent.`);
    } else {
      greet(view.primary);
    }
  }
}

function handleRecognitionError(error) {
  camera.clearOverlay();
  state.latest = null;
  clearExpression();
  if (error.status === 503) {
    // Engine unavailable: the server said so explicitly; applyEngine renders the offline view.
    applyEngine(false, error.message);
    return;
  }
  renderIdentity(describeEvent({ state: 'ERROR', message: error.message }));
  setRecognitionValue('Error', true);
  if (state.lastErrorText !== error.message) {
    state.lastErrorText = error.message;
    addEvent('Recognition error', error.message);
  }
}

function greet(face) {
  // One cooldown per identity: shouldGreet records the greeting itself (and bounds what it remembers).
  if (!shouldGreet(state.greeted, face.identity_id, Date.now(), state.cooldownMs)) return;
  const name = face.display_name || 'there';
  if ('speechSynthesis' in window) {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(`Hi ${name}.`));
    addEvent('Greeting spoken', `Hi ${name}.`);
  } else {
    addEvent('Greeting', `Hi ${name}. (Speech synthesis unavailable in this browser.)`);
  }
}

// ---------- camera control ----------

function setCameraButtons(busy) {
  els.activateButton.disabled = busy;
  els.visionToggle.disabled = busy;
}

async function startCamera() {
  if (state.cameraOn || state.starting) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Camera access is not supported in this browser.');
    return;
  }
  state.starting = true;
  setCameraButtons(true);
  try {
    await camera.start();
  } catch (error) {
    const wording = describeCameraError(error);
    setPill(els.cameraPill, els.cameraStatus, wording.pill, 'warn');
    els.cameraContext.textContent = wording.context;
    addEvent('Camera unavailable', `${error.name || 'Error'}: ${error.message}`);
    toast(`Camera unavailable: ${error.message}`);
    return;
  } finally {
    state.starting = false;
    setCameraButtons(false);
  }
  state.session += 1;
  state.cameraOn = true;
  state.paused = false;
  state.frame = 0;
  state.lastTransition = null;
  state.latest = null;
  clearExpression();
  els.stage.classList.add('camera-on');
  els.centerState.inert = true;
  els.visionToggle.title = 'Stop vision';
  els.visionToggle.setAttribute('aria-label', 'Stop vision');
  setPill(els.cameraPill, els.cameraStatus, 'VISION ONLINE', 'online');
  els.cameraContext.textContent = 'Live';
  els.sessionContext.textContent = nowTime();
  els.pauseButton.disabled = false;
  els.pauseLabel.textContent = 'Pause vision';
  addEvent('Vision activated', 'Local camera stream started. Frames stay on this device.');
  if (state.engine.available) {
    renderIdentity(IDLE_VIEW);
    setRecognitionValue('Live', false);
    schedule(150);
  } else {
    renderIdentity(offlineView('engine unavailable'));
    setRecognitionValue('Offline', true);
  }
  setLoopIndicator();
}

function stopCamera() {
  if (!state.cameraOn) return;
  state.session += 1;
  clearTimeout(state.timer);
  camera.stop();
  state.cameraOn = false;
  state.paused = false;
  state.latest = null;
  els.stage.classList.remove('camera-on');
  els.centerState.inert = false;
  els.visionToggle.title = 'Activate vision';
  els.visionToggle.setAttribute('aria-label', 'Activate vision');
  setPill(els.cameraPill, els.cameraStatus, 'VISION OFFLINE', '');
  els.cameraContext.textContent = 'Inactive';
  els.sessionContext.textContent = 'Not started';
  els.pauseButton.disabled = true;
  els.frameInfo.textContent = 'FRAME —';
  setRecognitionValue(state.engine.available ? 'Ready' : 'Offline', !state.engine.available);
  addEvent('Vision stopped', 'Camera stream closed. No frames are being processed.');
  renderIdentity(offlineView());
  clearExpression();
  setLoopIndicator();
  api.resetPresence().catch(() => { /* best-effort; the backend expires a presence without frames after a few seconds */ });
}

function togglePause() {
  if (!state.cameraOn) return;
  state.paused = !state.paused;
  if (state.paused) {
    clearTimeout(state.timer);
    camera.clearOverlay();
    state.latest = null;
    api.resetPresence().catch(() => {}); // no frames while paused: subscribers should see NO_SIGNAL now, not after expiry
    els.pauseLabel.textContent = 'Resume vision';
    setRecognitionValue('Paused', true);
    clearExpression();
    els.learnButton.disabled = true;
    addEvent('Recognition paused', 'Camera preview stays local and visible; recognition requests are stopped.');
  } else {
    els.pauseLabel.textContent = 'Pause vision';
    setRecognitionValue(state.engine.available ? 'Live' : 'Offline', !state.engine.available);
    addEvent('Recognition resumed', 'Identity processing resumed.');
    schedule(0);
  }
  setLoopIndicator();
}

// ---------- enrollment ----------

function openEnrollDialog() {
  if (!state.latest) return;
  // Freeze the frame *and the event it produced* as one pair; the loop pauses while the dialog is open.
  state.enrollFrozen = state.latest;
  state.enrolling = true;
  clearTimeout(state.timer);
  els.enrollError.hidden = true;
  els.enrollError.textContent = '';
  setLoopIndicator();
  els.enrollDialog.showModal();
  setTimeout(() => els.displayName.focus(), 40);
}

function showEnrollError(message) {
  els.enrollError.textContent = message;
  els.enrollError.hidden = false;
}

async function submitEnrollment(event) {
  event.preventDefault();
  const name = els.displayName.value.trim();
  const consent = els.consent.checked;
  const pair = state.enrollFrozen;
  const frozen = describeEvent(pair?.event || {});
  if (!name) { showEnrollError('Enter a display name.'); els.displayName.focus(); return; }
  if (!consent) { showEnrollError('Explicit consent is required before a face encoding is stored.'); els.consent.focus(); return; }
  if (!pair?.blob || frozen.state !== 'UNKNOWN' || !frozen.canEnroll) {
    showEnrollError('Enrollment needs exactly one unknown face. Close this dialog and try again.');
    return;
  }

  els.enrollSubmit.disabled = true;
  renderIdentity(describeEvent({ ...pair.event, state: 'LEARNING', can_enroll: false }));
  try {
    const record = await api.enroll(pair.blob, name, consent);
    els.enrollDialog.close('learned'); // the close handler clears name, consent and the frozen pair
    addEvent('Identity learned', `${record.display_name} stored locally. Step out of frame and return to verify recognition.`);
    toast(`${record.display_name} stored locally.`);
    state.lastTransition = null;
    els.identityCard.focus();
    await refreshStatus();
  } catch (error) {
    showEnrollError(error.message);
    addEvent('Enrollment rejected', error.message);
    renderIdentity(frozen);
  } finally {
    els.enrollSubmit.disabled = false;
  }
}

/**
 * Everything the dialog collected, cleared in one place. The consent box is consent to store a
 * biometric encoding: it must never be found already ticked, and the name must never be inherited —
 * a dialog dismissed with Escape after a failed attempt would otherwise enrol the next person under
 * the previous person's name with a consent nobody gave.
 */
function resetEnrollForm() {
  els.displayName.value = '';
  els.consent.checked = false;
  els.enrollError.hidden = true;
  els.enrollError.textContent = '';
  state.enrollFrozen = null;
}

/** Fires on every close path: submit, Cancel, Escape. */
function onEnrollDialogClosed() {
  state.enrolling = false;
  resetEnrollForm();
  setLoopIndicator();
  schedule(0);
}

// ---------- identity store ----------

function identityRow(item) {
  const row = document.createElement('div');
  row.className = 'identity-item';
  const text = document.createElement('div');
  const strong = document.createElement('strong');
  strong.textContent = item.display_name;
  const small = document.createElement('small');
  const created = item.created_at ? new Date(item.created_at).toLocaleString() : 'unknown time';
  small.textContent = `${item.encoding_count} encoding${item.encoding_count === 1 ? '' : 's'} · ${created}`;
  text.append(strong, small);
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.className = 'ghost-btn';
  remove.textContent = 'Delete';
  remove.setAttribute('aria-label', `Delete ${item.display_name}`);
  remove.addEventListener('click', async () => {
    remove.disabled = true;
    const index = [...els.identityList.children].indexOf(row);
    try {
      await api.deleteIdentity(item.id);
      addEvent('Identity deleted', `${item.display_name} removed from the local store.`);
      toast(`${item.display_name} deleted.`);
      state.lastTransition = null;
      state.greeted.delete(item.id); // a deleted person keeps no slot in the greeting memory
    } catch (error) {
      remove.disabled = false;
      toast(`Delete failed: ${error.message}`);
      if (error.status !== 404) return;
    }
    await Promise.all([renderIdentities(), refreshStatus()]);
    focusInDrawer(index);
  });
  row.append(text, remove);
  return row;
}

/** Keep keyboard focus inside the drawer after the list re-renders: same row index, else the close button. */
function focusInDrawer(index = 0) {
  const buttons = els.identityList.querySelectorAll('button');
  const target = buttons[Math.min(index, buttons.length - 1)] || els.identityDialog.querySelector('form button');
  target?.focus();
}

async function renderIdentities() {
  const root = els.identityList;
  try {
    const list = await api.listIdentities();
    root.replaceChildren();
    if (!list.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-list';
      empty.textContent = 'No identities are enrolled on this device.';
      root.append(empty);
      return;
    }
    root.append(...list.map(identityRow));
  } catch (error) {
    root.replaceChildren();
    const failed = document.createElement('div');
    failed.className = 'empty-list';
    failed.textContent = `Identity store unavailable: ${error.message}`;
    root.append(failed);
  }
}

async function openIdentities() {
  els.identityDialog.showModal();
  await renderIdentities();
}

async function submitErase(event) {
  event.preventDefault();
  els.eraseSubmit.disabled = true;
  try {
    const result = await api.eraseAll();
    els.eraseDialog.close('erased');
    addEvent('Identity store erased', `${result.deleted} identit${result.deleted === 1 ? 'y' : 'ies'} removed from this device.`);
    toast('Local identity store erased.');
    state.lastTransition = null;
    state.greeted.clear(); // nobody is enrolled any more: no cooldown may survive the erase
    if (els.identityDialog.open) {
      await renderIdentities();
      focusInDrawer(0);
    }
    await refreshStatus();
  } catch (error) {
    els.eraseDialog.close();
    toast(`Erase failed: ${error.message}`);
  } finally {
    els.eraseSubmit.disabled = false;
  }
}

// ---------- wiring ----------

els.activateButton.addEventListener('click', startCamera);
els.visionToggle.addEventListener('click', () => (state.cameraOn ? stopCamera() : startCamera()));
els.pauseButton.addEventListener('click', togglePause);
els.learnButton.addEventListener('click', openEnrollDialog);
els.enrollForm.addEventListener('submit', submitEnrollment);
els.enrollCancel.addEventListener('click', () => els.enrollDialog.close('cancel'));
els.enrollDialog.addEventListener('close', onEnrollDialogClosed);
els.identitiesButton.addEventListener('click', openIdentities);
els.openIdentitiesButton.addEventListener('click', openIdentities);
els.eraseButton.addEventListener('click', () => els.eraseDialog.showModal());
els.drawerEraseButton.addEventListener('click', () => els.eraseDialog.showModal());
els.eraseForm.addEventListener('submit', submitErase);
els.eraseCancel.addEventListener('click', () => els.eraseDialog.close('cancel'));
els.expressionButton.addEventListener('click', toggleExpression);

// Visibility hygiene: no recognition work while the tab is hidden; camera stays user-controlled.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearTimeout(state.timer);
  else schedule(0);
});
// Page going away with the camera on: presence subscribers should not wait for expiry.
window.addEventListener('pagehide', () => { if (state.cameraOn) api.resetPresenceBeacon(); });
window.addEventListener('resize', () => {
  if (state.latest && loopAllowed()) camera.drawFaces(state.latest.event.faces || []);
});

function tickClock() { els.clock.textContent = nowTime(); }
tickClock();
setInterval(tickClock, 1000);

installSpotlights();
installMagnets();
installAtmosphere(els.stage);
renderIdentity(offlineView());
setLoopIndicator();
await refreshStatus();
setInterval(refreshStatus, STATUS_INTERVAL_MS);

// Live event stream (same origin): mood and action entries come from the server — the MoodTracker's hysteresis
// and the ActionTracker's onset/apex/offset are the single source of truth, so nothing is re-derived or debounced
// here. Entries are logged only while expression is on (state mirrors /api/status); wording stays hedged
// ("Ben looks happy.", "Ben: brief smile (0.9 s)") — a hint, never a fact, and nothing in the shell gates on it.
subscribeEvents({
  onMood: (t) => {
    // A reconnect replays the server's buffer (Last-Event-ID): a minutes-old mood is history, not news,
    // and must not be stamped with the current time.
    if (!state.expression.enabled || !t?.to_mood || !isFreshEntry(t)) return; // to_mood null = the mood ended; nothing to say
    const described = describeExpression({ dominant: t.to_mood }, EXPRESSION_LANG);
    if (described) addEvent('Mood', `${t.display_name || 'someone'} ${described.label}.`);
  },
  onAction: (a) => {
    if (!state.expression.enabled || !isFreshEntry(a)) return;
    // Actions can complete every ~1.8 s per group and several groups run at once while someone talks:
    // without this display limit a few seconds of conversation would evict every error/greeting entry.
    if (!allowActionEntry(state.actionLogged, a.action, Date.now())) return;
    const described = describeAction(a, EXPRESSION_LANG);
    if (described) addEvent('Expression', `${a.display_name || 'someone'}: ${described.label}`);
  },
  onOpen: () => {
    els.eventsContext.textContent = describeEventsStatus(null);
    if (!state.eventsWarned) return;
    state.eventsWarned = false; // a later outage is worth logging again
    addEvent('Live events restored', 'The event stream is connected again; mood and expression entries resume.');
  },
  onError: (info) => {
    // Retries flood this handler, so the log gets one entry per outage while the Context row keeps
    // the current state. A closed connection is retried by events.js; a dropped one by the platform.
    els.eventsContext.textContent = describeEventsStatus(info);
    if (state.eventsWarned) return;
    state.eventsWarned = true;
    addEvent('Live events unavailable', info?.unsupported
      ? 'This browser has no EventSource, so mood and expression entries stay off.'
      : 'Mood and expression entries pause until the stream is back.');
  },
});
