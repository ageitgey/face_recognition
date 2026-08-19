// Face2AI presence — desktop half of the Hermes `face2ai` plugin.
// Plain ESM, no build step (JSX not allowed): renders a status-bar chip and a pane that show
// who is in front of the local camera, fed by the plugin's own backend namespace
// (`ctx.rest('/presence')` → /api/plugins/face2ai/presence, `ctx.socket('/events')` as accelerator).
// Nothing biometric ever reaches this UI — Face2AI's stream carries states, names, counts, timestamps,
// a hedged mood hint ("wirkt …" — a best-effort guess from facial expression, never a fact) and completed
// facial actions ("kurzes Lächeln (0.9 s)" — expression dynamics at ~0.6 s resolution, no micro-expressions).
// The pane adds a valence sparkline from Face2AI's in-memory timeline (`ctx.rest('/timeline?seconds=600')`,
// narrowed to the person in front of the camera and fetched on its own slow cadence, not on every poll),
// bounded and cleared on presence reset/restart — a `timeline_cleared` frame clears this mirror with it,
// and so does the disposer; nothing is stored long-term anywhere.
import { PALETTE_AREA, PANES_AREA, STATUSBAR_AREAS, haptic } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const POLL_MS = 4000
const TIMELINE_SECONDS = 600
// The timeline is the expensive call (up to 2000 samples ≈ 320 KB) and it feeds a 240 px sparkline, so it
// runs on its own slow cadence instead of every presence poll — and re-runs at once when the person changes.
const TIMELINE_MIN_INTERVAL_MS = 20000
const EMPTY_TIMELINE = { seconds: TIMELINE_SECONDS, samples: [], moods: [], actions: [] }
const IDLE_PRESENCE = { source: 'none', presence: { state: 'NO_SIGNAL' }, connected: false }
const listeners = new Set()
let latest = IDLE_PRESENCE
let history = []
let moods = [] // last mood changes (`to_mood` set) as they came off the wire, newest last
let actions = [] // last completed facial actions, newest last
let timeline = EMPTY_TIMELINE // Face2AI's in-memory affect history (samples for the sparkline)
let timelineAt = 0 // when the timeline was last fetched
let timelineIdentity = null // whose timeline it is (null = everyone)
let pluginCtx = null

function publish() {
  for (const fn of listeners) fn({ latest, history, moods, actions, timeline })
}

/** Face2AI's affect history for `identityId` (null = everyone); skipped unless it is stale, for someone else, or forced. */
async function refreshTimeline(identityId, { force = false, now = Date.now() } = {}) {
  if (!pluginCtx) return
  const who = identityId || null
  if (!force && who === timelineIdentity && now - timelineAt < TIMELINE_MIN_INTERVAL_MS) return
  timelineAt = now
  timelineIdentity = who
  // Filtering server-side keeps another person's samples off the wire entirely, not just out of the drawing.
  const query = `/timeline?seconds=${TIMELINE_SECONDS}${who ? `&identity_id=${encodeURIComponent(who)}` : ''}`
  try {
    const t = await pluginCtx.rest(query)
    timeline = t && Array.isArray(t.samples) ? t : EMPTY_TIMELINE
  } catch (error) {
    timeline = { ...EMPTY_TIMELINE, error: String(error && error.message ? error.message : error) }
  }
}

async function refresh({ forceTimeline = false } = {}) {
  if (!pluginCtx) return
  try {
    latest = await pluginCtx.rest('/presence')
    if (latest && Array.isArray(latest.history)) history = latest.history
    // Snapshot fallback (gateway state file) also carries the last few moods/actions — seed from it.
    if (latest && Array.isArray(latest.moods)) moods = latest.moods.filter((m) => m && m.to_mood).slice(-20)
    if (latest && Array.isArray(latest.actions)) actions = latest.actions.slice(-10)
  } catch (error) {
    latest = { source: 'error', error: String(error && error.message ? error.message : error), presence: { state: 'NO_SIGNAL' }, connected: false }
  }
  await refreshTimeline((latest && latest.presence && latest.presence.identity_id) || null, { force: forceTimeline })
  publish()
}

function onFrame(frame) {
  if (!frame || typeof frame !== 'object') return
  const { event, data } = frame
  if (event === 'hello' || event === 'heartbeat') {
    if (data && data.presence) latest = { ...latest, source: 'live', presence: data.presence, connected: true }
  } else if (event === 'presence' && data) {
    // A transition starts a fresh, mood-less presence (the mood that ended arrives as its own `mood` frame).
    latest = { ...latest, source: 'live', connected: true, presence: { state: data.to_state, identity_id: data.identity_id, display_name: data.display_name, faces: data.faces, since: data.at } }
    history = [...history.slice(-29), data]
  } else if (event === 'mood' && data) {
    // Hint began/changed/ended (`to_mood: null`); never a transition, never a reaction.
    latest = { ...latest, presence: { ...(latest.presence || {}), mood: data.to_mood ?? null, valence: data.valence ?? null, arousal: data.arousal ?? null } }
    if (data.to_mood) moods = [...moods.slice(-19), data]
  } else if (event === 'action' && data) {
    // A completed facial action: history only — never touches presence, never a reaction.
    actions = [...actions.slice(-9), data]
  } else if (event === 'timeline_cleared') {
    // The user reset presence in Face2AI, which forgot its affect history: this mirror forgets too.
    moods = []
    actions = []
    timeline = EMPTY_TIMELINE
    timelineAt = 0
  } else if (event === 'lost') {
    latest = { ...latest, connected: false, source: 'lost', error: data && data.error }
  }
  publish()
}

function usePresence() {
  const [state, setState] = useState({ latest, history, moods, actions, timeline })
  useEffect(() => {
    listeners.add(setState)
    return () => listeners.delete(setState)
  }, [])
  return state
}

const LABELS = {
  NO_SIGNAL: { de: 'Kamera aus', en: 'camera off', tone: 'muted' },
  NO_FACE: { de: 'niemand da', en: 'nobody', tone: 'muted' },
  UNKNOWN: { de: 'Unbekannt', en: 'unknown', tone: 'warn' },
  KNOWN: { de: '', en: '', tone: 'ok' },
  MULTIPLE_FACES: { de: 'mehrere', en: 'several', tone: 'warn' },
}
const TONE_CLASS = {
  ok: 'text-(--ui-accent, #82f3d4)',
  warn: 'text-(--ui-warning, #f4ce8a)',
  muted: 'text-(--ui-text-tertiary)',
}

// Hedged mood wording — same literal table as Face2AI's UI (model.js) and the gateway half (presence.py).
const MOOD_LABELS = { Happiness: 'fröhlich', Sadness: 'traurig', Anger: 'verärgert', Fear: 'ängstlich', Surprise: 'überrascht', Disgust: 'angewidert', Contempt: 'abschätzig', Neutral: 'neutral' }

function fmtAxis(value) {
  if (!Number.isFinite(value)) return null
  const rounded = Number(value.toFixed(1)) + 0 // + 0 turns -0 into 0 so tiny negatives print as +0.0
  return (rounded >= 0 ? '+' : '') + rounded.toFixed(1)
}

/** "wirkt fröhlich (Valenz +0.6, Erregung +0.1)" or '' when there is no mood; unknown labels stay hedged. */
function moodLabel(presence) {
  const p = presence || {}
  if (typeof p.mood !== 'string' || !p.mood) return ''
  const word = Object.hasOwn(MOOD_LABELS, p.mood) ? MOOD_LABELS[p.mood] : p.mood.toLowerCase()
  const parts = [['Valenz', fmtAxis(p.valence)], ['Erregung', fmtAxis(p.arousal)]].filter(([, v]) => v !== null).map(([k, v]) => `${k} ${v}`)
  return `wirkt ${word}${parts.length ? ` (${parts.join(', ')})` : ''}`
}

// Hedged facial-action wording — same literal table as Face2AI's UI (model.js) and the gateway half (presence.py).
// Timing is quantized to the browser loop (~0.6 s): "kurzes" ≤ 1 s, "anhaltendes" ≥ 5 s — dynamics, not micro-expressions.
const ACTION_LABELS = { smile: 'Lächeln', frown: 'Mundwinkel runter', brow_raise: 'Brauen hoch', brow_furrow: 'Stirnrunzeln', eye_squint: 'Augen zusammengekniffen', eyes_wide: 'Augen weit', nose_wrinkle: 'Nasenrümpfen', lip_press: 'Lippen gepresst' }

/** "kurzes Lächeln (0.9 s)" / "Brauen hoch (2.3 s)" / "anhaltendes Lächeln (6.0 s)"; '' without an action; unknown labels stay hedged. */
function actionLabel(a) {
  if (!a || typeof a.action !== 'string' || !a.action) return ''
  const word = Object.hasOwn(ACTION_LABELS, a.action) ? ACTION_LABELS[a.action] : a.action.toLowerCase().replaceAll('_', ' ')
  const ms = typeof a.duration_ms === 'number' && Number.isFinite(a.duration_ms) ? a.duration_ms : null
  if (ms === null) return word
  const shown = (ms / 1000).toFixed(1) // the number the reader sees …
  const shownMs = parseFloat(shown) * 1000 // … and the qualifier follows it: 4999 ms is not "Lächeln (5.0 s)"
  const qualifier = shownMs <= 1000 ? 'kurzes ' : shownMs >= 5000 ? 'anhaltendes ' : ''
  return `${qualifier}${word} (${shown} s)`
}

/** "wirkt fröhlich" for a mood transition (`to_mood`); '' when the mood ended. */
function moodWord(toMood) {
  if (typeof toMood !== 'string' || !toMood) return ''
  return `wirkt ${Object.hasOwn(MOOD_LABELS, toMood) ? MOOD_LABELS[toMood] : toMood.toLowerCase()}`
}

const SPARK_W = 240
const SPARK_H = 28

/** Polyline points for valence samples (−1..1 → bottom..top), oldest left / newest right; '' below 2 points. */
function sparklinePoints(samples, w = SPARK_W, h = SPARK_H) {
  const values = (samples || []).map((s) => (s && typeof s.valence === 'number' && Number.isFinite(s.valence) ? Math.max(-1, Math.min(1, s.valence)) : null)).filter((v) => v !== null)
  if (values.length < 2) return ''
  const n = values.length
  return values.map((v, i) => `${((i / (n - 1)) * w).toFixed(1)},${(((1 - v) / 2) * h).toFixed(1)}`).join(' ')
}

/** Plain SVG (no JSX): a valence line over Face2AI's timeline samples with a zero line; nothing without points. */
function Sparkline({ points }) {
  if (!points) return null
  return jsxs('svg', {
    width: '100%',
    height: SPARK_H,
    viewBox: `0 0 ${SPARK_W} ${SPARK_H}`,
    preserveAspectRatio: 'none',
    role: 'img',
    'aria-label': 'Valenz-Verlauf (Vermutung aus dem Gesichtsausdruck, keine Tatsache)',
    style: { display: 'block' },
    children: [
      jsx('line', { x1: 0, y1: SPARK_H / 2, x2: SPARK_W, y2: SPARK_H / 2, stroke: 'currentColor', strokeOpacity: 0.25, strokeWidth: 1, vectorEffect: 'non-scaling-stroke' }),
      jsx('polyline', { points, fill: 'none', stroke: 'var(--ui-accent, #82f3d4)', strokeWidth: 1.5, strokeLinejoin: 'round', strokeLinecap: 'round', vectorEffect: 'non-scaling-stroke' }),
    ],
  })
}

function labelFor(presence, lang = 'de') {
  const p = presence || {}
  const state = p.state || 'NO_SIGNAL'
  if (state === 'KNOWN') return { text: p.display_name || (lang === 'de' ? 'bekannt' : 'known'), tone: 'ok' }
  const entry = LABELS[state] || LABELS.NO_SIGNAL
  return { text: entry[lang] || entry.en, tone: entry.tone }
}

function fmtTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function Chip() {
  const { latest: current } = usePresence()
  const { text, tone } = labelFor(current.presence)
  const dot = current.connected ? '●' : '○'
  return jsx('button', {
    type: 'button',
    title: current.connected ? `Face2AI · ${current.source}` : `Face2AI nicht verbunden${current.error ? ': ' + current.error : ''}`,
    className: `px-1.5 text-[0.6875rem] ${TONE_CLASS[tone]}`,
    onClick: () => {
      haptic('tap')
      void refresh({ forceTimeline: true })
    },
    children: `${dot} 👤 ${text}`,
  })
}

const HEDGE_TITLE = 'Vermutung aus dem Gesichtsausdruck, keine Tatsache'
const RESOLUTION_HINT = 'Auflösung ~0,6 s — Ausdrucksdynamik, keine Mikroexpressionen'

function Pane() {
  const { latest: current, history: items, moods: moodItems, actions: actionItems, timeline: tl } = usePresence()
  const p = current.presence || {}
  const { text, tone } = labelFor(p)
  const rows = [...items].reverse().slice(0, 12)
  // One line = one person: with a known person only their samples, otherwise only the unattributed ones
  // (UNKNOWN/NO_FACE/MULTIPLE_FACES) — never several people's valence drawn as one curve.
  const samples = (tl && Array.isArray(tl.samples) ? tl.samples : []).filter((s) => s && (p.identity_id ? s.identity_id === p.identity_id : !s.identity_id))
  const sparkPoints = sparklinePoints(samples)
  const moodRows = moodItems.filter((m) => m && m.to_mood).slice(-6).reverse()
  const actionRows = actionItems.slice(-5).reverse()
  return jsxs('div', {
    className: 'flex h-full flex-col gap-3 overflow-y-auto p-3 text-sm',
    children: [
      jsxs('div', {
        className: 'flex items-baseline justify-between',
        children: [
          jsx('div', { className: 'text-[0.6875rem] uppercase tracking-wider text-(--ui-text-tertiary)', children: 'Face2AI · Kamera' }),
          jsx('div', { className: 'text-[0.6875rem] text-(--ui-text-tertiary)', children: current.connected ? current.source : 'offline' }),
        ],
      }),
      jsxs('div', {
        children: [
          jsx('div', { className: `text-2xl font-semibold ${TONE_CLASS[tone]}`, children: text }),
          jsx('div', { className: 'text-(--ui-text-tertiary)', children: `${p.state || 'NO_SIGNAL'} · ${p.faces || 0} ${p.faces === 1 ? 'Gesicht' : 'Gesichter'}${p.since ? ' · seit ' + fmtTime(p.since) : ''}` }),
          moodLabel(p) ? jsx('div', { className: 'text-[0.75rem] text-(--ui-text-tertiary)', title: HEDGE_TITLE, children: moodLabel(p) }) : null,
        ],
      }),
      current.error ? jsx('div', { className: 'text-[0.75rem] text-(--ui-warning, #f4ce8a)', children: String(current.error) }) : null,
      jsxs('div', {
        className: 'flex flex-col gap-1',
        title: HEDGE_TITLE,
        children: [
          jsx('div', { className: 'text-[0.6875rem] uppercase tracking-wider text-(--ui-text-tertiary)', children: 'Valenz · letzte 10 min' }),
          sparkPoints
            ? jsx(Sparkline, { points: sparkPoints })
            : jsx('div', { className: 'text-[0.75rem] text-(--ui-text-tertiary)', children: 'Noch kein Verlauf.' }),
          jsx('div', { className: 'text-[0.6875rem] text-(--ui-text-tertiary)', children: RESOLUTION_HINT }),
        ],
      }),
      jsxs('div', {
        className: 'flex flex-col gap-1',
        title: HEDGE_TITLE,
        children: [
          jsx('div', { className: 'text-[0.6875rem] uppercase tracking-wider text-(--ui-text-tertiary)', children: 'Stimmung zuletzt' }),
          moodRows.length === 0
            ? jsx('div', { className: 'text-(--ui-text-tertiary)', children: 'Noch keine Hinweise.' })
            : moodRows.map((m, i) => jsxs('div', {
                className: 'flex justify-between gap-2 text-[0.75rem]',
                children: [
                  jsx('span', { className: 'text-(--ui-text-tertiary)', children: fmtTime(m.at) }),
                  jsx('span', { children: `${m.display_name ? m.display_name + ' · ' : ''}${moodWord(m.to_mood)}` }),
                ],
              }, `${m.at || i}-${i}`)),
        ],
      }),
      jsxs('div', {
        className: 'flex flex-col gap-1',
        title: HEDGE_TITLE,
        children: [
          jsx('div', { className: 'text-[0.6875rem] uppercase tracking-wider text-(--ui-text-tertiary)', children: 'Ausdruck zuletzt' }),
          actionRows.length === 0
            ? jsx('div', { className: 'text-(--ui-text-tertiary)', children: 'Noch keine Ausdrücke.' })
            : actionRows.map((a, i) => jsxs('div', {
                className: 'flex justify-between gap-2 text-[0.75rem]',
                children: [
                  jsx('span', { className: 'text-(--ui-text-tertiary)', children: fmtTime(a.at) }),
                  jsx('span', { children: `${a.display_name ? a.display_name + ' · ' : ''}${actionLabel(a)}` }),
                ],
              }, `${a.at || i}-${i}`)),
        ],
      }),
      jsxs('div', {
        className: 'flex flex-col gap-1',
        children: [
          jsx('div', { className: 'text-[0.6875rem] uppercase tracking-wider text-(--ui-text-tertiary)', children: 'Übergänge' }),
          rows.length === 0
            ? jsx('div', { className: 'text-(--ui-text-tertiary)', children: 'Noch keine Ereignisse.' })
            : rows.map((t, i) => jsxs('div', {
                className: 'flex justify-between gap-2 text-[0.75rem]',
                children: [
                  jsx('span', { className: 'text-(--ui-text-tertiary)', children: fmtTime(t.at) }),
                  jsx('span', { children: `${t.from_state} → ${t.to_state}${t.display_name ? ' · ' + t.display_name : ''}` }),
                ],
              }, `${t.at || i}-${i}`)),
        ],
      }),
      jsxs('div', {
        className: 'mt-auto flex gap-2',
        children: [
          jsx('button', { type: 'button', className: 'rounded border px-2 py-1 text-[0.75rem]', onClick: () => void refresh({ forceTimeline: true }), children: 'Aktualisieren' }),
          jsx('button', { type: 'button', className: 'rounded border px-2 py-1 text-[0.75rem]', onClick: () => void pluginCtx?.os.openExternal('http://127.0.0.1:8765/'), children: 'Face2AI öffnen' }),
        ],
      }),
    ],
  })
}

export default {
  id: 'face2ai',
  name: 'Face2AI presence',
  defaultEnabled: false,
  register(ctx) {
    pluginCtx = ctx
    ctx.register({ id: 'chip', area: STATUSBAR_AREAS.right, order: 115, render: () => jsx(Chip, {}) })
    ctx.register({ id: 'pane', area: PANES_AREA, title: 'Face2AI', data: { placement: 'right', width: '280px' }, render: () => jsx(Pane, {}) })
    ctx.register({
      id: 'open-app',
      area: PALETTE_AREA,
      data: { id: 'face2ai.open', label: 'Face2AI: Kamera-App öffnen', keywords: ['face2ai', 'kamera', 'camera', 'presence'], run: () => void ctx.os.openExternal('http://127.0.0.1:8765/') },
    })
    ctx.register({
      id: 'refresh',
      area: PALETTE_AREA,
      data: { id: 'face2ai.refresh', label: 'Face2AI: Präsenz aktualisieren', keywords: ['face2ai', 'presence'], run: () => void refresh({ forceTimeline: true }) },
    })
    void refresh()
    const timer = setInterval(() => void refresh(), POLL_MS)
    const stopSocket = ctx.socket('/events', onFrame)
    // Disposer (hot reload, or the user disabling the plugin in Settings): stop polling and the live socket,
    // and drop the expression history with it — re-enabling must not show the previous session's moods and
    // actions (a live /presence answer carries none, so nothing would overwrite them).
    return () => {
      clearInterval(timer)
      stopSocket()
      pluginCtx = null
      latest = IDLE_PRESENCE
      history = []
      moods = []
      actions = []
      timeline = EMPTY_TIMELINE
      timelineAt = 0
      timelineIdentity = null
      publish()
    }
  },
}
