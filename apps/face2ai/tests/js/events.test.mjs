// The browser's SSE client (static/js/events.js) against a fake EventSource: what it subscribes to,
// which frames reach the handlers, and what it reports when the browser has no EventSource at all.
import assert from 'node:assert/strict';
import { mock, test } from 'node:test';

import { subscribeEvents } from '../../src/face2ai_app/static/js/events.js';

const CONNECTING = 0;
const OPEN = 1;
const CLOSED = 2;

class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    this.closed = false;
    this.readyState = CONNECTING;
    FakeEventSource.last = this;
    FakeEventSource.instances.push(this);
  }

  static reset() {
    FakeEventSource.instances = [];
    FakeEventSource.last = null;
  }

  addEventListener(kind, handler) {
    this.listeners.set(kind, handler);
  }

  close() {
    this.closed = true;
    this.readyState = CLOSED;
  }

  /** Deliver one server frame the way the platform does: a MessageEvent with a JSON string. */
  emit(kind, data) {
    this.listeners.get(kind)?.({ data });
  }

  /** The stream is up. */
  open() {
    this.readyState = OPEN;
    this.onopen?.({});
  }

  /** A dropped connection: the platform is already retrying with Last-Event-ID. */
  failRetrying() {
    this.readyState = CONNECTING;
    this.onerror?.({});
  }

  /** A non-2xx status or a wrong content-type: per the spec the connection *fails* and stays closed. */
  failClosed() {
    this.readyState = CLOSED;
    this.onerror?.({});
  }
}
FakeEventSource.instances = [];

function withEventSource(impl, run) {
  const original = globalThis.EventSource;
  globalThis.EventSource = impl;
  FakeEventSource.reset();
  try {
    return run();
  } finally {
    globalThis.EventSource = original;
  }
}

test('subscribeEvents listens same-origin as the browser role and hands mood/action payloads through', () => {
  withEventSource(FakeEventSource, () => {
    const moods = [];
    const actions = [];
    const close = subscribeEvents({ onMood: (m) => moods.push(m), onAction: (a) => actions.push(a) });
    const source = FakeEventSource.last;

    assert.equal(source.url, '/api/events?role=browser');
    assert.doesNotMatch(source.url, /role=agent/); // role=agent would hand the spoken greeting to a voice agent
    assert.deepEqual([...source.listeners.keys()], ['mood', 'action']); // presence/store/heartbeat stay ignored

    source.emit('mood', JSON.stringify({ at: '2026-08-18T12:00:00Z', to_mood: 'Happiness' }));
    source.emit('action', JSON.stringify({ at: '2026-08-18T12:00:03Z', action: 'smile', duration_ms: 900 }));
    assert.deepEqual(moods, [{ at: '2026-08-18T12:00:00Z', to_mood: 'Happiness' }]);
    assert.deepEqual(actions, [{ at: '2026-08-18T12:00:03Z', action: 'smile', duration_ms: 900 }]);

    source.emit('mood', '{not json');  // dropped, never thrown
    source.emit('action', 'null');     // parses to null: nothing to say
    assert.equal(moods.length, 1);
    assert.equal(actions.length, 1);

    assert.equal(source.closed, false);
    close();
    assert.equal(source.closed, true);
  });
});

test('subscribeEvents reports a missing EventSource as unsupported, not as a reconnect', () => {
  withEventSource(undefined, () => {
    const errors = [];
    const close = subscribeEvents({ onError: (info) => errors.push(info) });
    assert.deepEqual(errors, [{ unsupported: true }]); // nothing will ever reconnect: the shell must not promise it
    assert.equal(typeof close, 'function');
    close(); // a no-op disposer, never a throw
  });
});

test('a dropped connection is left to the platform, which reconnects by itself', () => {
  withEventSource(FakeEventSource, () => {
    const errors = [];
    subscribeEvents({ onError: (info) => errors.push(info) });
    FakeEventSource.last.failRetrying();
    assert.deepEqual(errors, [{ unsupported: false, closed: false, retryInMs: null }]);
    assert.equal(FakeEventSource.instances.length, 1, 'no second subscription: the platform owns this retry');
  });
});

test('a connection the browser left CLOSED is re-subscribed with bounded backoff', () => {
  // Per the EventSource spec a non-2xx status or a wrong content-type fails the connection and leaves
  // readyState CLOSED for good. Without this the page goes permanently deaf after one 503.
  mock.timers.enable({ apis: ['setTimeout'] });
  try {
    withEventSource(FakeEventSource, () => {
      const errors = [];
      const close = subscribeEvents(
        { onError: (info) => errors.push(info) },
        '/api/events?role=browser',
        { baseMs: 1_000, maxMs: 4_000 },
      );
      assert.equal(FakeEventSource.instances.length, 1);

      FakeEventSource.last.failClosed();
      assert.deepEqual(errors, [{ unsupported: false, closed: true, retryInMs: 1_000 }]);
      mock.timers.tick(999);
      assert.equal(FakeEventSource.instances.length, 1, 'the retry is scheduled, not immediate');
      mock.timers.tick(1);
      assert.equal(FakeEventSource.instances.length, 2);
      assert.equal(FakeEventSource.last.url, '/api/events?role=browser');
      assert.deepEqual([...FakeEventSource.last.listeners.keys()], ['mood', 'action']);

      for (const expected of [2_000, 4_000, 4_000]) { // doubles, then stays at the cap
        FakeEventSource.last.failClosed();
        assert.equal(errors.at(-1).retryInMs, expected);
        mock.timers.tick(expected);
      }
      assert.equal(FakeEventSource.instances.length, 5);

      FakeEventSource.last.open(); // a stream that came back resets the backoff
      FakeEventSource.last.failClosed();
      assert.equal(errors.at(-1).retryInMs, 1_000);

      close();
      mock.timers.tick(60_000);
      assert.equal(FakeEventSource.instances.length, 5, 'a disposed subscription must not resurrect itself');
    });
  } finally {
    mock.timers.reset();
  }
});
