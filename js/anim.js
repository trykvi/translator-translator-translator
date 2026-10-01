// Tiny animation clock with a global speed knob and cancellation.

export const clock = {
  speed: 1, // 0 = skip
};

export class Cancelled extends Error {
  constructor() { super('cancelled'); }
}

/** manim's "smooth" rate function. */
export const smooth = t => {
  const s = 1 / (1 + Math.exp(10 * 0.5)); // sigmoid(-5)
  const x = 1 / (1 + Math.exp(-10 * (t - 0.5)));
  return Math.min(1, Math.max(0, (x - s) / (1 - 2 * s)));
};
export const linear = t => t;
export const easeOut = t => 1 - (1 - t) ** 3;

export const lerp = (a, b, t) => a + (b - a) * t;
export const clamp01 = t => Math.min(1, Math.max(0, t));
/** Map global progress t to a sub-interval [a, b]. */
export const span = (t, a, b) => clamp01((t - a) / (b - a));

/** Run fn(easedProgress) every frame for `ms` (scaled by clock.speed). */
export function tween(ms, fn, { signal, ease = smooth } = {}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new Cancelled());
    if (!clock.speed || ms <= 0) { fn(1); return resolve(); }
    let elapsed = 0;
    let last = performance.now();
    const frame = now => {
      if (signal?.aborted) return reject(new Cancelled());
      elapsed += (now - last) * (clock.speed || Infinity);
      last = now;
      const t = Math.min(1, elapsed / ms);
      fn(ease(t));
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });
}

export function wait(ms, signal) {
  return tween(ms, () => {}, { signal, ease: linear });
}
