// Same-origin SSE client for the browser shell: it listens to `mood` and `action` only. Presence,
// store and heartbeat frames are ignored on purpose — the page derives its own view from the frames
// it sends. The server is the single source of truth for both entry kinds (mood hysteresis and action
// onset/apex/offset are applied there); nothing is re-derived or debounced here. `?role=browser` keeps
// this subscription apart from the voice agent's role, which would make the page hand greetings to it.
//
// Two different failures arrive on the same `error` handler and only `readyState` tells them apart.
// A dropped connection leaves the source CONNECTING and the platform reconnects it by itself (with
// `Last-Event-ID`). A non-2xx status or a wrong content-type instead *fails* the connection per the
// EventSource spec: `readyState` stays CLOSED and nothing ever retries — one 503 during a restart
// would leave this page deaf for the rest of the session. So we re-subscribe ourselves, with a
// bounded backoff, and report which of the two happened. (Reasoned from the spec and pinned against
// a fake EventSource in tests/js/events.test.mjs; not yet confirmed in a browser session.)

const CLOSED = 2;              // EventSource.CLOSED, read numerically so a test double needs no constants
const RETRY_BASE_MS = 1000;
const RETRY_MAX_MS = 30000;
const RETRY_MAX_ATTEMPTS = 20; // only to keep 2 ** attempt finite; the delay itself is capped at maxMs

/**
 * Subscribe to the local event stream. Returns a `close()` function that also cancels a pending retry.
 * Handlers get the parsed JSON payload; malformed data is dropped, never thrown. `onError` receives
 * `{ unsupported, closed, retryInMs }`: `closed` means the browser gave up and `retryInMs` is when
 * this client will try again.
 */
export function subscribeEvents(
  { onMood, onAction, onOpen, onError } = {},
  url = '/api/events?role=browser',
  { baseMs = RETRY_BASE_MS, maxMs = RETRY_MAX_MS } = {},
) {
  if (typeof EventSource !== 'function') {
    onError?.({ unsupported: true }); // nothing will reconnect here: the caller must not promise that it will
    return () => {};
  }
  const parse = (e) => { try { return JSON.parse(e.data); } catch { return null; } };
  let source = null;
  let timer = null;
  let attempt = 0;
  let disposed = false;

  const connect = () => {
    source = new EventSource(url);
    source.addEventListener('mood', (e) => { const d = parse(e); if (d) onMood?.(d); });
    source.addEventListener('action', (e) => { const d = parse(e); if (d) onAction?.(d); });
    source.onopen = () => { attempt = 0; onOpen?.(); };
    source.onerror = () => {
      const closed = source?.readyState === CLOSED;
      let retryInMs = null;
      if (closed && !disposed) {
        retryInMs = Math.min(maxMs, baseMs * 2 ** attempt);
        attempt = Math.min(attempt + 1, RETRY_MAX_ATTEMPTS);
        clearTimeout(timer);
        timer = setTimeout(() => { timer = null; if (!disposed) connect(); }, retryInMs);
      }
      onError?.({ unsupported: false, closed, retryInMs });
    };
  };

  connect();
  return () => {
    disposed = true;
    clearTimeout(timer);
    timer = null;
    source?.close();
  };
}
