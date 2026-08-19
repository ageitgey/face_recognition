// Thin client for the local Face2AI API (see api/routes.py). Frames are sent as raw
// image/jpeg bodies; enrollment parameters travel as query parameters. Errors carry the
// server's `detail` text and HTTP status so the UI can show the real reason.

export async function getStatus() {
  return jsonFetch('/api/status');
}

export async function listIdentities() {
  return jsonFetch('/api/identities');
}

export async function deleteIdentity(id) {
  return jsonFetch(`/api/identities/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export async function eraseAll() {
  return jsonFetch('/api/identities', { method: 'DELETE' });
}

/** Camera stopped/paused: tell the backend so presence subscribers (voice agent) see NO_SIGNAL. */
export async function resetPresence() {
  return jsonFetch('/api/presence/reset', { method: 'POST' });
}

/** Page is going away: fire-and-forget reset (sendBeacon survives unload; falls back to keepalive fetch). */
export function resetPresenceBeacon() {
  if (navigator.sendBeacon && navigator.sendBeacon('/api/presence/reset')) return;
  fetch('/api/presence/reset', { method: 'POST', keepalive: true }).catch(() => {});
}

/** Runtime opt-in for expression hints (Stage 1). Server answers { enabled, available }; enabling an
 *  unavailable engine is a 409 whose `detail` names the reason. The only thing the browser ever sends
 *  for this feature is this flag — no frames, no scores. */
export async function setExpression(enabled) {
  return jsonFetch('/api/expression', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ enabled: enabled === true }),
  });
}

/** Returns { event, agentConnected }: the backend flags on every frame whether a voice agent owns greetings. */
export async function recognize(blob) {
  const response = await fetch('/api/recognize', { method: 'POST', headers: { 'content-type': 'image/jpeg' }, body: blob });
  if (!response.ok) throw await apiError(response);
  return { event: await response.json(), agentConnected: response.headers.get('x-face2ai-agent') === '1' };
}

export async function enroll(blob, name, consent) {
  const qs = new URLSearchParams({ display_name: name, consent: consent ? 'true' : 'false' });
  return imageFetch(`/api/enroll?${qs}`, blob);
}

async function imageFetch(url, blob) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'image/jpeg' },
    body: blob,
  });
  if (!response.ok) throw await apiError(response);
  return response.json();
}

async function jsonFetch(url, init) {
  const response = await fetch(url, { cache: 'no-store', ...init });
  if (!response.ok) throw await apiError(response);
  return response.status === 204 ? null : response.json();
}

async function apiError(response) {
  let detail = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') detail = body.detail;
    else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
  } catch {
    // Non-JSON error body: keep the HTTP status text.
  }
  const error = new Error(detail);
  error.status = response.status;
  return error;
}
