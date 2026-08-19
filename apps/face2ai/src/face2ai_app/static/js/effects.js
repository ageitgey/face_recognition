// ReactBits-inspired micro-effects, implemented dependency-free (see docs/UI_DIRECTION.md).
// Every effect is decorative: it never carries information, never blocks pointer events,
// and switches off under prefers-reduced-motion (observed live, not snapshotted once).

const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
const coarsePointerQuery = window.matchMedia('(pointer: coarse)');

export function motionAllowed() {
  return !reducedMotionQuery.matches;
}

/**
 * DecryptedText: short machine-state labels resolve from glyph noise into the final text.
 * Re-invoking with the same text is a no-op (no re-scramble on polling ticks); a new text
 * cancels the running animation on that element.
 */
const running = new WeakMap();
const GLYPHS = '01<>/{}[]+*=#?';

export function decryptText(el, text, { duration = 420 } = {}) {
  if (!el) return;
  const final = String(text);
  if (el.dataset.finalText === final) return;
  el.dataset.finalText = final;
  const active = running.get(el);
  if (active) cancelAnimationFrame(active);
  if (!motionAllowed() || !final.length) {
    running.delete(el);
    el.textContent = final;
    return;
  }
  const start = performance.now();
  const frame = (now) => {
    const progress = Math.min(1, (now - start) / duration);
    const reveal = Math.floor(final.length * progress);
    let out = '';
    for (let i = 0; i < final.length; i += 1) {
      const char = final[i];
      out += i < reveal || char === ' ' ? char : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
    }
    el.textContent = out;
    if (progress < 1) {
      running.set(el, requestAnimationFrame(frame));
    } else {
      running.delete(el);
      el.textContent = final;
    }
  };
  running.set(el, requestAnimationFrame(frame));
}

/** SpotlightCard: pointer-driven radial highlight via CSS custom properties. No tilt, no layout shift. */
export function installSpotlights(root = document) {
  const cards = root.querySelectorAll('.spotlight');
  cards.forEach((card) => {
    card.addEventListener('pointermove', (event) => {
      if (!motionAllowed()) return;
      const rect = card.getBoundingClientRect();
      card.style.setProperty('--sx', `${event.clientX - rect.left}px`);
      card.style.setProperty('--sy', `${event.clientY - rect.top}px`);
    }, { passive: true });
  });
  reducedMotionQuery.addEventListener('change', () => {
    if (motionAllowed()) return;
    cards.forEach((card) => { card.style.removeProperty('--sx'); card.style.removeProperty('--sy'); });
  });
}

/** Magnet: at most ~4px pointer attraction on high-value controls; off for touch and reduced motion. */
export function installMagnets(root = document, { maxOffset = 4 } = {}) {
  root.querySelectorAll('.magnet').forEach((button) => {
    button.addEventListener('pointermove', (event) => {
      if (!motionAllowed() || coarsePointerQuery.matches || event.pointerType === 'touch') return;
      const rect = button.getBoundingClientRect();
      const dx = ((event.clientX - rect.left) / rect.width - 0.5) * 2 * maxOffset;
      const dy = ((event.clientY - rect.top) / rect.height - 0.5) * 2 * maxOffset;
      button.style.transform = `translate(${dx.toFixed(1)}px, ${dy.toFixed(1)}px)`;
    }, { passive: true });
    button.addEventListener('pointerleave', () => { button.style.transform = ''; });
  });
  reducedMotionQuery.addEventListener('change', () => {
    if (!motionAllowed()) root.querySelectorAll('.magnet').forEach((b) => { b.style.transform = ''; });
  });
}

/** DotGrid parallax (whole page) + camera-stage highlight; tiny amplitude, pointer-only, motion-gated. */
export function installAtmosphere(stage) {
  const root = document.documentElement;
  window.addEventListener('pointermove', (event) => {
    if (!motionAllowed()) return;
    root.style.setProperty('--px', (event.clientX / window.innerWidth).toFixed(3));
    root.style.setProperty('--py', (event.clientY / window.innerHeight).toFixed(3));
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const mx = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100));
    const my = Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100));
    stage.style.setProperty('--mx', `${mx.toFixed(1)}%`);
    stage.style.setProperty('--my', `${my.toFixed(1)}%`);
  }, { passive: true });
  reducedMotionQuery.addEventListener('change', () => {
    if (motionAllowed()) return;
    root.style.removeProperty('--px');
    root.style.removeProperty('--py');
    stage?.style.removeProperty('--mx');
    stage?.style.removeProperty('--my');
  });
}
