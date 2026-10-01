// The left panel: a 2D window into a 128-dimensional embedding space, drawn like a manim NumberPlane.

import { tween, lerp, span, easeOut } from './anim.js';
import { vec, similarities, topK, localBasis, project, fitSimilarity } from './linalg.js';

export const FONT_STACK = '"CMU Serif", "Latin Modern Roman", "Noto Serif CJK JP", "Hiragino Mincho ProN", "Yu Mincho", "Noto Serif CJK SC", "Songti SC", Georgia, serif';
export const MONO_STACK = '"CMU Typewriter", Menlo, Consolas, monospace';
export const COLOR = {
  bg: '#050607',
  text: '#f4f6f8',
  muted: '#a3adb9',
  ghost: '#6b7583',
  accent: '#58c4dd',
  accentRGB: '88,196,221',
};

const NEIGHBOURS = 14;

/**
 * A "frame" is a 2D view of one language's space, centred on a vector: its nearest words,
 * projected onto the two directions in which they spread out the most.
 */
export function makeFrame(lang, center, { self = -1, k = NEIGHBOURS } = {}) {
  const sims = similarities(lang, center);
  const nn = topK(sims, k, self >= 0 ? new Set([self]) : null);
  // PCA of the little cloud (the vector + its neighbours) around its own mean, so we see how the
  // neighbours spread out rather than just "they're all over there"
  const cloud = [center, ...nn.map(n => vec(lang, n.idx))];
  const mean = new Float32Array(center.length);
  for (const p of cloud) for (let d = 0; d < mean.length; d++) mean[d] += p[d] / cloud.length;
  const basis = localBasis(mean, cloud);
  const points = nn.map(n => ({
    id: `${lang.code}:${n.idx}`,
    idx: n.idx,
    word: lang.words[n.idx],
    sim: n.sim,
    pos: project(basis, center, vec(lang, n.idx)),
  }));
  return { lang, center, basis, sims, points, self };
}

/**
 * How frame F's points land in frame G after applying `mapVec` (a rotation matrix, or nothing).
 * The best-fitting 2D rotation is the "shadow" of the 128×128 matrix we animate.
 */
export function fitTransition(F, G, mapVec) {
  const src = F.points.map(p => p.pos);
  const mapped = F.points.map(p => mapVec(vec(F.lang, p.idx)));
  let dst = mapped.map(x => project(G.basis, G.center, x));
  let fit = fitSimilarity(src, dst);
  if (fit.reflected) {
    // Mirror G instead, so the animation is a proper rotation rather than a flip.
    G.basis = [G.basis[0], G.basis[1].map(x => -x)];
    G.points.forEach(p => { p.pos = [p.pos[0], -p.pos[1]]; });
    dst = dst.map(([x, y]) => [x, -y]);
    fit = fitSimilarity(src, dst);
  }
  return { ...fit, src, dst };
}

function similarityAt(fit, t, [x, y]) {
  const th = fit.theta * t;
  const s = lerp(1, fit.scale, t);
  const c = Math.cos(th) * s, sn = Math.sin(th) * s;
  return [c * x - sn * y + fit.tx * t, sn * x + c * y + fit.ty * t];
}

function mulAffine(a, b) {
  // a ∘ b, affine maps as {m: [a, b, c, d], t: [x, y]} acting as m·p + t
  const [a0, a1, a2, a3] = a.m, [b0, b1, b2, b3] = b.m;
  return {
    m: [a0 * b0 + a1 * b2, a0 * b1 + a1 * b3, a2 * b0 + a3 * b2, a2 * b1 + a3 * b3],
    t: [a0 * b.t[0] + a1 * b.t[1] + a.t[0], a2 * b.t[0] + a3 * b.t[1] + a.t[1]],
  };
}

function rotationAffine(fit, t) {
  // the grid only rotates and slides; letting it scale too would make it denser with every hop
  const th = fit.theta * t;
  const c = Math.cos(th), sn = Math.sin(th);
  return { m: [c, -sn, sn, c], t: [fit.tx * t, fit.ty * t] };
}

function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x));
  const f = x / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

export class SpaceStage {
  constructor(canvas, labelEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.labelEl = labelEl;
    this.items = new Map(); // id -> {word, x, y, alpha, kind, labelAlpha}
    this.lines = [];        // {x, y, len, alpha, label}
    this.query = { x: 0, y: 0, alpha: 0, r: 0, label: '' };
    this.grid = { aff: { m: [1, 0, 0, 1], t: [0, 0] }, alpha: 0.6, step: 0.5 };
    this.scale = 300;
    this.cx = 0;
    this.cy = 0;
    this.pulse = null;
    this.widths = new Map();
    this.dirty = true;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    const loop = () => {
      if (this.dirty || this.pulse) this.draw();
      this.dirty = false;
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = r.width;
    this.h = r.height;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty = true;
  }

  fontsChanged() {
    this.widths.clear();
    this.dirty = true;
  }

  setLabel(text) {
    this.labelEl.textContent = text;
  }

  /** Camera (centre + zoom) that fits the vector (at the origin) and the given points. */
  fitView(...pointSets) {
    let x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    for (const pts of pointSets) {
      for (const [x, y] of pts) {
        x0 = Math.min(x0, x); x1 = Math.max(x1, x);
        y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
    }
    const ex = Math.max(0.05, (x1 - x0) / 2), ey = Math.max(0.05, (y1 - y0) / 2);
    const sx = (this.w / 2 - Math.min(80, this.w * 0.15)) / ex;
    const sy = (this.h / 2 - 48) / ey;
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 - 8 / Math.min(sx, sy), scale: Math.max(30, Math.min(sx, sy)) };
  }

  setView(v) {
    this.cx = v.cx;
    this.cy = v.cy;
    this.scale = v.scale;
  }

  /** Forget everything and show a fresh frame (the "look up a word" step). */
  async enter(frame, queryLabel, { signal, ms = 1300 } = {}) {
    const old = [...this.items.values()];
    const oldQ = this.query.alpha;
    const oldLines = this.lines;
    const oldGrid = this.grid.alpha;
    if (old.length || oldQ) {
      await tween(260, t => {
        old.forEach(it => { it.fade = 1 - t; });
        this.query.alpha = oldQ * (1 - t);
        oldLines.forEach(l => { l.fade = 1 - t; });
        this.grid.alpha = oldGrid * (1 - t);
        this.dirty = true;
      }, { signal });
    }
    this.items.clear();
    this.lines = [];
    this.pulse = null;
    this.setView(this.fitView(frame.points.map(p => p.pos)));
    this.grid = { aff: { m: [1, 0, 0, 1], t: [0, 0] }, alpha: 0, step: niceStep(130 / this.scale) };
    this.query = { x: 0, y: 0, alpha: 0, r: 0, label: queryLabel };
    const pts = frame.points.map((p, i) => {
      const it = { word: p.word, x: p.pos[0], y: p.pos[1], alpha: 0, kind: 'word' };
      this.items.set(p.id, it);
      return it;
    });
    this.setLabel(spaceLabel(frame.lang));
    await tween(ms, t => {
      this.grid.alpha = span(t, 0, 0.35);
      this.query.alpha = span(t, 0.1, 0.35);
      this.query.r = easeOut(span(t, 0.1, 0.4));
      pts.forEach((it, i) => { it.alpha = span(t, 0.3 + 0.5 * i / pts.length, 0.5 + 0.5 * i / pts.length); });
      this.dirty = true;
    }, { signal });
  }

  /**
   * Move from frame F to frame G. mode 'rotate': F's words become ghosts sitting where the matrix
   * sent them. mode 'recenter': same language, the camera glides to a new word.
   */
  async transition(F, G, fit, { mode = 'rotate', signal, ms = 2200, queryLabel } = {}) {
    this.clearHighlights();
    const fIds = new Set(F.points.map(p => p.id));
    const gPos = new Map(G.points.map(p => [p.id, p.pos]));
    // Leftovers from earlier hops: ghosts that reappear in G come back to life, the rest fade out.
    const leftovers = [...this.items.entries()].filter(([id]) => !fIds.has(id));
    const dying = leftovers.filter(([id]) => !gPos.has(id)).map(([id, it]) => ({ id, it, a0: it.alpha }));
    const reviving = leftovers.filter(([id]) => gPos.has(id)).map(([id, it]) => ({ it, start: [it.x, it.y], end: gPos.get(id), a0: it.alpha }));
    const moving = F.points.map((p, i) => {
      const it = this.items.get(p.id) ?? { word: p.word, x: p.pos[0], y: p.pos[1], alpha: 1, kind: 'word' };
      this.items.set(p.id, it);
      const start = [it.x, it.y];
      const end = gPos.get(p.id) ?? fit.dst[i];
      const atEnd = similarityAt(fit, 1, start);
      return { id: p.id, it, start, res: [end[0] - atEnd[0], end[1] - atEnd[1]], stays: gPos.has(p.id), a0: it.alpha };
    });
    const arriving = G.points.filter(p => !this.items.has(p.id)).map((p, i) => {
      const it = { word: p.word, x: p.pos[0], y: p.pos[1], alpha: 0, kind: 'word' };
      this.items.set(p.id, it);
      return { it, i };
    });
    const aff0 = this.grid.aff;
    const v0 = { cx: this.cx, cy: this.cy, scale: this.scale };
    const v1 = this.fitView(G.points.map(p => p.pos), mode === 'rotate' ? fit.dst : []);
    const q0 = [this.query.x, this.query.y];

    await tween(ms, t => {
      const tm = span(t, 0.08, 0.85);
      dying.forEach(d => { d.it.alpha = d.a0 * (1 - span(t, 0, 0.3)); });
      reviving.forEach(r => {
        r.it.x = lerp(r.start[0], r.end[0], tm);
        r.it.y = lerp(r.start[1], r.end[1], tm);
        r.it.alpha = lerp(r.a0, 1, span(t, 0.5, 0.9));
        if (t > 0.6) r.it.kind = 'word';
      });
      for (const m of moving) {
        const p = similarityAt(fit, tm, m.start);
        m.it.x = p[0] + m.res[0] * tm;
        m.it.y = p[1] + m.res[1] * tm;
        if (!m.stays) {
          if (mode === 'rotate') {
            m.it.kind = t > 0.6 ? 'ghost' : 'word';
            m.it.alpha = lerp(m.a0, 0.55, span(t, 0.55, 0.85));
          } else {
            m.it.alpha = m.a0 * (1 - span(t, 0.2, 0.7));
          }
        }
      }
      arriving.forEach(({ it, i }) => { it.alpha = span(t, 0.6 + 0.25 * i / arriving.length, 0.8 + 0.2 * i / arriving.length); });
      this.grid.aff = mulAffine(rotationAffine(fit, tm), aff0);
      this.setView({ cx: lerp(v0.cx, v1.cx, tm), cy: lerp(v0.cy, v1.cy, tm), scale: lerp(v0.scale, v1.scale, tm) });
      this.query.x = q0[0] * (1 - tm);
      this.query.y = q0[1] * (1 - tm);
      if (queryLabel && t > 0.5) this.query.label = queryLabel;
      this.dirty = true;
    }, { signal });

    dying.forEach(d => this.items.delete(d.id));
    if (mode === 'recenter') moving.filter(m => !m.stays).forEach(m => this.items.delete(m.id));
    this.setLabel(spaceLabel(G.lang));
  }

  /** Draw lines from the vector to its nearest words, then crown the winner. */
  async highlight(frame, top, { signal, ms = 1700, final = true } = {}) {
    this.clearHighlights();
    const cands = top.slice(0, final ? 5 : 3).map(c => {
      const id = `${frame.lang.code}:${c.idx}`;
      let it = this.items.get(id);
      if (!it) {
        const pos = project(frame.basis, frame.center, vec(frame.lang, c.idx));
        it = { word: frame.lang.words[c.idx], x: pos[0], y: pos[1], alpha: 0, kind: 'word' };
        this.items.set(id, it);
      }
      return { it, sim: c.sim, a0: it.alpha };
    });
    const others = [...this.items.values()].filter(it => !cands.some(c => c.it === it) && it.kind !== 'ghost');
    this.lines = cands.map(c => ({ to: c.it, len: 0, alpha: 1, label: c.sim.toFixed(2) }));
    await tween(ms, t => {
      cands.forEach((c, i) => {
        const tt = span(t, 0.05 + i * 0.08, 0.45 + i * 0.08);
        this.lines[i].len = easeOut(tt);
        c.it.alpha = Math.max(c.a0, tt);
        c.it.kind = i === 0 && t > 0.7 ? 'winner' : 'cand';
      });
      others.forEach(it => { it.dim = span(t, 0.1, 0.5); });
      this.lines.forEach((l, i) => { l.alpha = i === 0 ? 1 : 1 - 0.6 * span(t, 0.7, 1); });
      this.dirty = true;
    }, { signal });
    this.pulse = { item: cands[0].it, t0: performance.now() };
  }

  clearHighlights() {
    this.lines = [];
    this.pulse = null;
    for (const it of this.items.values()) {
      it.dim = 0;
      if (it.kind === 'cand' || it.kind === 'winner') it.kind = 'word';
    }
    this.dirty = true;
  }

  // ------------------------------------------------------------------ drawing

  toScreen(x, y) {
    return [this.w / 2 + (x - this.cx) * this.scale, this.h / 2 - (y - this.cy) * this.scale];
  }

  draw() {
    const { ctx, w, h } = this;
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, w, h);
    if (this.grid.alpha > 0) this.drawGrid();

    // candidate lines
    const [qx, qy] = this.toScreen(this.query.x, this.query.y);
    ctx.lineCap = 'round';
    for (const l of this.lines) {
      const [tx, ty] = this.toScreen(l.to.x, l.to.y);
      const ex = qx + (tx - qx) * l.len, ey = qy + (ty - qy) * l.len;
      const a = l.alpha * (l.fade ?? 1);
      ctx.strokeStyle = l.to.kind === 'winner' ? `rgba(${COLOR.accentRGB},${a})` : `rgba(244,246,248,${0.55 * a})`;
      ctx.lineWidth = l.to.kind === 'winner' ? 2.2 : 1.2;
      ctx.beginPath();
      ctx.moveTo(qx, qy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      if (l.len > 0.95) {
        ctx.font = `12px ${MONO_STACK}`;
        ctx.fillStyle = l.to.kind === 'winner' ? `rgba(${COLOR.accentRGB},${a})` : `rgba(163,173,185,${a})`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const mx = (qx + tx) / 2, my = (qy + ty) / 2;
        ctx.fillText(l.label, mx, my - 9);
      }
    }

    // words
    const labels = [];
    for (const it of this.items.values()) {
      const a = it.alpha * (it.fade ?? 1) * (1 - 0.6 * (it.dim || 0));
      if (a <= 0.01) continue;
      const [x, y] = this.toScreen(it.x, it.y);
      const ghost = it.kind === 'ghost';
      const winner = it.kind === 'winner';
      const r = winner ? 5 : ghost ? 2.5 : 3.2;
      ctx.fillStyle = winner ? COLOR.accent : ghost ? COLOR.ghost : COLOR.text;
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      labels.push({
        text: it.word, x, y, r, alpha: a,
        size: winner ? 21 : it.kind === 'cand' ? 17 : ghost ? 14 : 16,
        italic: ghost,
        color: winner ? COLOR.accent : ghost ? COLOR.ghost : it.kind === 'cand' ? COLOR.text : '#d5dbe2',
        prio: winner ? 3 : it.kind === 'cand' ? 2 : ghost ? 0 : 1,
      });
    }

    // pulse ring on the winner, three times, then stop redrawing
    if (this.pulse && performance.now() - this.pulse.t0 > 3 * 1600) this.pulse = null;
    if (this.pulse) {
      const [x, y] = this.toScreen(this.pulse.item.x, this.pulse.item.y);
      const ph = ((performance.now() - this.pulse.t0) / 1600) % 1;
      ctx.strokeStyle = `rgba(${COLOR.accentRGB},${0.7 * (1 - ph)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 7 + ph * 22, 0, Math.PI * 2);
      ctx.stroke();
    }

    // the travelling vector
    const q = this.query;
    if (q.alpha > 0) {
      ctx.save();
      ctx.globalAlpha = q.alpha;
      ctx.shadowColor = COLOR.accent;
      ctx.shadowBlur = 22;
      ctx.fillStyle = COLOR.accent;
      ctx.beginPath();
      ctx.arc(qx, qy, 7 * q.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      if (q.label) labels.push({ text: q.label, x: qx, y: qy, r: 8, alpha: q.alpha, size: 20, italic: true, color: COLOR.accent, prio: 4, above: true });
    }
    this.drawLabels(labels);
  }

  drawGrid() {
    const { ctx, w, h } = this;
    const { m, t } = this.grid.aff;
    const det = m[0] * m[3] - m[1] * m[2];
    if (Math.abs(det) < 1e-6) return;
    const inv = [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det];
    // viewport corners -> grid-local coordinates
    const corners = [[0, 0], [w, 0], [0, h], [w, h]].map(([sx, sy]) => {
      const fx = (sx - w / 2) / this.scale + this.cx - t[0], fy = -(sy - h / 2) / this.scale + this.cy - t[1];
      return [inv[0] * fx + inv[1] * fy, inv[2] * fx + inv[3] * fy];
    });
    const xs = corners.map(c => c[0]), ys = corners.map(c => c[1]);
    const step = this.grid.step / 2;
    const x0 = Math.floor(Math.min(...xs) / step), x1 = Math.ceil(Math.max(...xs) / step);
    const y0 = Math.floor(Math.min(...ys) / step), y1 = Math.ceil(Math.max(...ys) / step);
    if (x1 - x0 > 400 || y1 - y0 > 400) return;
    const map = (gx, gy) => this.toScreen(m[0] * gx + m[1] * gy + t[0], m[2] * gx + m[3] * gy + t[1]);
    const A = this.grid.alpha;
    const line = (a, b, style, width) => {
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(...a);
      ctx.lineTo(...b);
      ctx.stroke();
    };
    for (const pass of [0, 1, 2]) {
      for (let i = x0; i <= x1; i++) {
        const kind = i === 0 ? 2 : i % 2 === 0 ? 1 : 0;
        if (kind !== pass) continue;
        const style = kind === 2 ? `rgba(244,246,248,${0.55 * A})` : `rgba(${COLOR.accentRGB},${(kind ? 0.42 : 0.13) * A})`;
        line(map(i * step, y0 * step), map(i * step, y1 * step), style, kind === 2 ? 1.6 : kind ? 1.1 : 0.8);
      }
      for (let j = y0; j <= y1; j++) {
        const kind = j === 0 ? 2 : j % 2 === 0 ? 1 : 0;
        if (kind !== pass) continue;
        const style = kind === 2 ? `rgba(244,246,248,${0.55 * A})` : `rgba(${COLOR.accentRGB},${(kind ? 0.42 : 0.13) * A})`;
        line(map(x0 * step, j * step), map(x1 * step, j * step), style, kind === 2 ? 1.6 : kind ? 1.1 : 0.8);
      }
    }
  }

  measure(text, font) {
    const key = font + '|' + text;
    let wd = this.widths.get(key);
    if (wd === undefined) {
      this.ctx.font = font;
      wd = this.ctx.measureText(text).width;
      this.widths.set(key, wd);
    }
    return wd;
  }

  drawLabels(labels) {
    const { ctx } = this;
    labels.sort((a, b) => b.prio - a.prio);
    const placed = [];
    const overlaps = b => placed.some(p => b.x < p.x + p.w + 3 && b.x + b.w + 3 > p.x && b.y < p.y + p.h && b.y + b.h > p.y);
    ctx.textBaseline = 'alphabetic';
    ctx.textAlign = 'left';
    for (const l of labels) {
      const font = `${l.italic ? 'italic ' : ''}${l.size}px ${FONT_STACK}`;
      const wd = this.measure(l.text, font);
      const hh = l.size * 0.95;
      const opts = [
        [l.x - wd / 2, l.y - l.r - 6 - hh],     // above
        [l.x + l.r + 6, l.y - hh / 2],          // right
        [l.x - l.r - 6 - wd, l.y - hh / 2],     // left
        [l.x - wd / 2, l.y + l.r + 5],          // below
      ];
      if (!l.above) opts.push(opts.shift());
      let box = null;
      for (const [x, y] of opts) {
        const b = { x, y, w: wd, h: hh };
        if (x < 4 || x + wd > this.w - 4 || y < 26 || y + hh > this.h - 4) continue;
        if (!overlaps(b)) { box = b; break; }
      }
      if (!box) {
        if (l.prio < 2) continue;
        box = { x: Math.min(Math.max(4, opts[0][0]), this.w - wd - 4), y: Math.max(26, opts[0][1]), w: wd, h: hh };
      }
      placed.push(box);
      ctx.font = font;
      ctx.globalAlpha = l.alpha;
      if (l.prio >= 2) {
        // dark halo so highlighted words stay readable over grid lines
        ctx.strokeStyle = COLOR.bg;
        ctx.lineWidth = 4;
        ctx.lineJoin = 'round';
        ctx.strokeText(l.text, box.x, box.y + hh * 0.8);
      }
      ctx.fillStyle = l.color;
      ctx.fillText(l.text, box.x, box.y + hh * 0.8);
      ctx.globalAlpha = 1;
    }
  }
}

function spaceLabel(lang) {
  return `${lang.name} space · ${lang.V.toLocaleString('en-US')} words · a 2D shadow of ℝ${superscript(lang.D)}`;
}

export function superscript(n) {
  return String(n).replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]);
}
