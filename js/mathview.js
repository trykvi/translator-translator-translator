// The right panel: the actual arithmetic. Vectors as columns of coloured cells, the matrix as a heatmap.

import { tween, span, easeOut } from './anim.js';
import { displayMatrixEntry } from './linalg.js';
import { FONT_STACK, MONO_STACK, COLOR } from './stage.js';

const POS = [88, 196, 221];  // blue for positive
const NEG = [252, 98, 85];   // red for negative (manim's RED)
const BG = [5, 6, 7];

function cellColor(v, range) {
  const t = Math.min(1, Math.abs(v) / range);
  const c = v >= 0 ? POS : NEG;
  return `rgb(${BG.map((b, i) => Math.round(b + (c[i] - b) * t)).join(',')})`;
}

export class MathPanel {
  constructor(canvas, labelEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.labelEl = labelEl;
    this.scene = { kind: 'idle' };
    this.heatmaps = new Map();
    this.dirty = true;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    const loop = () => {
      if (this.dirty) this.draw();
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

  heatmap(lang, dir) {
    const key = lang.code + dir;
    if (!this.heatmaps.has(key)) {
      const D = lang.D;
      const off = document.createElement('canvas');
      off.width = off.height = D;
      const g = off.getContext('2d');
      const img = g.createImageData(D, D);
      for (let i = 0; i < D; i++) {
        for (let j = 0; j < D; j++) {
          const v = displayMatrixEntry(lang, dir, i, j);
          const t = Math.min(1, Math.abs(v) / 0.22);
          const c = v >= 0 ? POS : NEG;
          const o = 4 * (i * D + j);
          for (let k = 0; k < 3; k++) img.data[o + k] = Math.round(BG[k] + (c[k] - BG[k]) * t);
          img.data[o + 3] = 255;
        }
      }
      g.putImageData(img, 0, 0);
      this.heatmaps.set(key, off);
    }
    return this.heatmaps.get(key);
  }

  idle(text) {
    this.scene = { kind: 'idle', text };
    this.labelEl.textContent = 'The arithmetic';
    this.dirty = true;
  }

  /** "katt" ↦ v: the word's 128 numbers appear. */
  async showVector(lang, word, v, { signal, ms = 1200 } = {}) {
    this.scene = { kind: 'vector', lang, word, v, p: 0 };
    this.labelEl.textContent = 'Step 1: look it up';
    await tween(ms, t => { this.scene.p = t; this.dirty = true; }, { signal });
  }

  /** W · v = v′, sweeping row by row like a very patient student. */
  async showMatmul(step, L, { signal, ms = 2000 } = {}) {
    const lang = L[step.matrix];
    this.scene = { kind: 'matmul', step, lang, from: L[step.from], to: L[step.to], img: this.heatmap(lang, step.dir), p: 0 };
    this.labelEl.textContent = `${lang.D} × ${lang.D} = ${(lang.D * lang.D).toLocaleString('en-US')} multiplications`;
    await tween(ms, t => { this.scene.p = t; this.dirty = true; }, { signal });
  }

  /** Cosine similarity against every word, as a histogram with the winners marked. */
  async showSearch(step, L, { signal, ms = 1500 } = {}) {
    const lang = L[step.lang];
    const bins = new Array(64).fill(0);
    const lo = -0.3, hi = 1.0;
    for (const s of step.sims) bins[Math.max(0, Math.min(63, Math.floor((s - lo) / (hi - lo) * 64)))]++;
    this.scene = { kind: 'search', step, lang, bins, lo, hi, p: 0 };
    this.labelEl.textContent = step.type === 'pass' ? 'Just passing through' : 'Nearest neighbour search';
    await tween(ms, t => { this.scene.p = t; this.dirty = true; }, { signal });
  }

  // ------------------------------------------------------------------ drawing

  draw() {
    const { ctx, w, h } = this;
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, w, h);
    const s = this.scene;
    if (s.kind === 'idle') this.drawIdle(s);
    else if (s.kind === 'vector') this.drawVector(s);
    else if (s.kind === 'matmul') this.drawMatmul(s);
    else if (s.kind === 'search') this.drawSearch(s);
  }

  text(str, x, y, { size = 16, color = COLOR.text, align = 'center', italic = false, mono = false, alpha = 1, baseline = 'alphabetic' } = {}) {
    const { ctx } = this;
    ctx.font = `${italic ? 'italic ' : ''}${size}px ${mono ? MONO_STACK : FONT_STACK}`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.globalAlpha = alpha;
    ctx.fillText(str, x, y);
    ctx.globalAlpha = 1;
    return ctx.measureText(str).width;
  }

  /** Draw e.g. v_{no} or W_{ja}^{⊤} with proper sub/superscripts. Returns width. */
  symbol(base, sub, sup, x, y, { size = 22, color = COLOR.text, align = 'center', alpha = 1, measureOnly = false } = {}) {
    const { ctx } = this;
    const fb = `italic ${size}px ${FONT_STACK}`, fs = `${Math.round(size * 0.6)}px ${FONT_STACK}`;
    ctx.font = fb;
    const wb = ctx.measureText(base).width;
    ctx.font = fs;
    const wsub = sub ? ctx.measureText(sub).width : 0;
    const wsup = sup ? ctx.measureText(sup).width : 0;
    const total = wb + Math.max(wsub, wsup) + 2;
    if (measureOnly) return total;
    let x0 = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = fb;
    ctx.fillText(base, x0, y);
    ctx.font = fs;
    if (sub) ctx.fillText(sub, x0 + wb + 1, y + size * 0.28);
    if (sup) ctx.fillText(sup, x0 + wb + 1, y - size * 0.42);
    ctx.globalAlpha = 1;
    return total;
  }

  brackets(x, y, wd, ht, alpha = 1) {
    const { ctx } = this;
    const k = Math.min(7, wd * 0.3);
    ctx.strokeStyle = `rgba(244,246,248,${alpha})`;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x + k, y); ctx.lineTo(x, y); ctx.lineTo(x, y + ht); ctx.lineTo(x + k, y + ht);
    ctx.moveTo(x + wd - k, y); ctx.lineTo(x + wd, y); ctx.lineTo(x + wd, y + ht); ctx.lineTo(x + wd - k, y + ht);
    ctx.stroke();
  }

  strip(v, x, y, wd, ht, upto = 1, range = 0.2) {
    const { ctx } = this;
    const n = v.length;
    const ch = ht / n;
    const m = Math.floor(n * upto);
    for (let i = 0; i < m; i++) {
      ctx.fillStyle = cellColor(v[i], range);
      ctx.fillRect(x, y + i * ch, wd, Math.ceil(ch));
    }
  }

  drawIdle(s) {
    const { w, h } = this;
    const y = h / 2;
    const size = Math.min(26, w / 13);
    // v_ja = W_ja^T W_no v_no, laid out by hand
    const parts = [['v', 'ja', ''], ['=', '', ''], ['W', 'ja', '⊤'], ['W', 'no', ''], ['v', 'no', '']];
    const gaps = 8;
    const widths = parts.map(([b, sb, sp]) => b === '=' ? size : this.symbol(b, sb, sp, 0, 0, { size, measureOnly: true }));
    let x = w / 2 - (widths.reduce((a, b) => a + b, 0) + gaps * (parts.length - 1)) / 2;
    parts.forEach(([b, sb, sp], i) => {
      if (b === '=') this.text('=', x + size / 2, y, { size });
      else this.symbol(b, sb, sp, x, y, { size, align: 'left', color: i === 0 ? COLOR.accent : COLOR.text });
      x += widths[i] + gaps;
    });
    this.text(s.text || 'Awaiting a word to multiply.', w / 2, y + 44, { size: 15, color: COLOR.muted, italic: true });
  }

  drawVector(s) {
    const { w, h } = this;
    const { v, word, lang, p } = s;
    const top = 48, ht = Math.max(80, h - top - 34);
    const sw = Math.max(12, Math.min(22, w * 0.05));
    const sx = Math.min(w * 0.42, w / 2 - 10);
    // left: the word
    this.text(`“${word}”`, sx - 34, top + ht / 2, { size: Math.min(28, w / 12), align: 'right', color: COLOR.accent, italic: true, baseline: 'middle' });
    this.text('↦', sx - 16, top + ht / 2, { size: 20, color: COLOR.muted, baseline: 'middle' });
    this.brackets(sx - 4, top - 4, sw + 8, ht + 8);
    this.strip(v, sx, top, sw, ht, easeOut(span(p, 0, 0.8)));
    // right: a few actual numbers
    const nx = sx + sw + 18;
    const shown = [0, 1, 2, 3, 4];
    const rowH = Math.min(22, ht / 9);
    shown.forEach((i, k) => {
      const a = span(p, 0.2 + k * 0.1, 0.4 + k * 0.1);
      const val = v[i] * Math.sqrt(v.length); // shown at a friendlier scale
      this.text(`${val >= 0 ? '+' : '−'}${Math.abs(val).toFixed(3)}`, nx, top + 10 + k * rowH, { size: 14, mono: true, align: 'left', alpha: a, color: val >= 0 ? COLOR.text : '#f6a59c' });
    });
    this.text('⋮', nx + 28, top + 10 + 5.6 * rowH, { size: 18, alpha: span(p, 0.7, 0.9) });
    this.text(`${v.length} numbers.`, nx, top + ht - 26, { size: 15, align: 'left', italic: true, color: COLOR.muted, alpha: span(p, 0.75, 1) });
    this.text('That is what it means now.', nx, top + ht - 6, { size: 15, align: 'left', italic: true, color: COLOR.muted, alpha: span(p, 0.8, 1) });
    this.symbol('v', lang.code, '', sx + sw / 2, top + ht + 28, { size: 18, color: COLOR.muted });
  }

  drawMatmul(s) {
    const { ctx, w, h } = this;
    const { step, img, p, lang } = s;
    const D = lang.D;
    const top = 50;
    const sw = Math.max(9, Math.min(16, w * 0.04));
    const avail = w - 2 * sw - 110;
    const S = Math.max(60, Math.min(h - top - 64, avail));
    const total = S + 2 * sw + 74;
    let x = (w - total) / 2;
    const y = top;
    const sweep = span(p, 0.12, 0.92);
    const row = Math.floor(sweep * D);

    // matrix
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = Math.min(1, span(p, 0, 0.12) * 1.5 + 0.2);
    ctx.drawImage(img, x, y, S, S);
    ctx.globalAlpha = 1;
    this.brackets(x - 5, y - 5, S + 10, S + 10);
    if (sweep > 0 && sweep < 1) {
      ctx.fillStyle = 'rgba(244,246,248,0.85)';
      ctx.fillRect(x, y + (row / D) * S, S, Math.max(1.5, S / D));
      ctx.strokeStyle = 'rgba(244,246,248,0.5)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 1, y + (row / D) * S - 1, S + 2, Math.max(1.5, S / D) + 2);
    }
    const matSub = step.dir === 'toHub' ? `${step.from}→en` : `en→${step.to}`;
    this.symbol('W', matSub, '', x + S / 2, y + S + 34, { size: 19 });
    x += S + 16;

    // input vector
    this.brackets(x - 4, y - 5, sw + 8, S + 10);
    this.strip(step.vin, x, y, sw, S);
    if (sweep > 0 && sweep < 1) {
      ctx.strokeStyle = `rgba(${COLOR.accentRGB},0.9)`;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - 1.5, y - 1.5, sw + 3, S + 3);
    }
    this.symbol('v', step.from, '', x + sw / 2, y + S + 34, { size: 19 });
    x += sw + 10;
    this.text('=', x + 13, y + S / 2, { size: 24, baseline: 'middle' });
    x += 30;

    // output vector, filling in
    this.brackets(x - 4, y - 5, sw + 8, S + 10);
    this.strip(step.vout, x, y, sw, S, sweep >= 1 ? 1 : row / D);
    if (sweep > 0 && sweep < 1) {
      ctx.fillStyle = 'rgba(244,246,248,0.9)';
      ctx.fillRect(x - 3, y + (row / D) * S, sw + 6, Math.max(1.5, S / D));
    }
    this.symbol('v', step.to, '', x + sw / 2, y + S + 34, { size: 19, color: COLOR.accent });

    const note = step.dir === 'toHub' ? 'each new number = one row · the old vector' : `just W(${step.to}→en) transposed: rotations undo themselves`;
    this.text(note, w / 2, h - 12, { size: 14, italic: true, color: COLOR.muted, alpha: span(p, 0.3, 0.6) });
  }

  drawSearch(s) {
    const { ctx, w, h } = this;
    const { step, lang, bins, lo, hi, p } = s;
    const left = 28, right = w - 20, top = 70, bottom = h - 46;
    const bw = (right - left) / bins.length;
    const max = Math.sqrt(Math.max(...bins));
    const grow = easeOut(span(p, 0, 0.5));
    this.text(`cos(v, w) for all ${lang.V.toLocaleString('en-US')} ${lang.name} words`, w / 2, 40, { size: 15, italic: true, color: COLOR.muted });
    bins.forEach((n, i) => {
      if (!n) return;
      const bh = (Math.sqrt(n) / max) * (bottom - top) * grow;
      ctx.fillStyle = `rgba(${COLOR.accentRGB},0.55)`;
      ctx.fillRect(left + i * bw + 0.5, bottom - bh, Math.max(1, bw - 1), bh);
    });
    // axis
    ctx.strokeStyle = 'rgba(244,246,248,0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, bottom + 0.5);
    ctx.lineTo(right, bottom + 0.5);
    ctx.stroke();
    const X = v => left + ((v - lo) / (hi - lo)) * (right - left);
    for (const t of [0, 0.5, 1]) {
      ctx.beginPath();
      ctx.moveTo(X(t), bottom);
      ctx.lineTo(X(t), bottom + 5);
      ctx.stroke();
      this.text(String(t), X(t), bottom + 20, { size: 13, mono: true, color: COLOR.muted });
    }
    // the winners, as pins
    const top3 = step.top.slice(0, 3);
    top3.forEach((c, i) => {
      const a = span(p, 0.45 + i * 0.12, 0.65 + i * 0.12);
      if (!a) return;
      const x = X(c.sim);
      const yTop = top - 10 + i * 24;
      ctx.strokeStyle = i === 0 ? `rgba(${COLOR.accentRGB},${a})` : `rgba(244,246,248,${0.5 * a})`;
      ctx.lineWidth = i === 0 ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(x, bottom);
      ctx.lineTo(x, yTop + 6);
      ctx.stroke();
      const label = `${lang.words[c.idx]} ${c.sim.toFixed(2)}`;
      this.text(label, Math.min(x, right - 4), yTop, { size: i === 0 ? 17 : 14, align: 'right', color: i === 0 ? COLOR.accent : COLOR.muted, alpha: a });
    });
    this.text(step.type === 'pass' ? 'Close, but this is Express. No stopping.' : 'Highest wins. That is the whole algorithm.', w / 2, h - 10, { size: 14, italic: true, color: COLOR.muted, alpha: span(p, 0.8, 1) });
  }
}

