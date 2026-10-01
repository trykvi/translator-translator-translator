// Turns "katt, no → ja" into a list of steps: look up, rotate, rotate, pick the nearest word.
// Pure logic, no DOM, so it can be tested in Node.

import { normalizeWord } from './data.js';
import { vec, toHub, fromHub, similarities, topK } from './linalg.js';

export const HUB = 'en';
const CJK = new Set(['ja', 'zh']);

export function lookup(lang, raw) {
  const i = lang.index.get(normalizeWord(lang.code, raw));
  return i === undefined ? -1 : i;
}

/**
 * Split input into words. Japanese and Chinese don't use spaces, so unknown runs are cut greedily
 * into the longest pieces the vocabulary knows.
 */
export function tokenize(lang, text) {
  const parts = text.normalize('NFKC').split(/[\s,.;:!?¡¿"“”«»()[\]{}、。！？，；：]+/u).filter(Boolean);
  const out = [];
  for (const part of parts) {
    if (lookup(lang, part) >= 0) { out.push(part); continue; }
    if (!CJK.has(lang.code)) {
      // l'amour, dell'acqua, it's: try the pieces around apostrophes
      const bits = part.split(/['’]/).filter(Boolean);
      out.push(...(bits.length > 1 && bits.some(b => lookup(lang, b) >= 0) ? bits.filter(b => lookup(lang, b) >= 0) : [part]));
      continue;
    }
    for (let i = 0; i < part.length;) {
      let len = Math.min(8, part.length - i);
      while (len > 1 && lookup(lang, part.slice(i, i + len)) < 0) len--;
      out.push(part.slice(i, i + len));
      i += len;
    }
  }
  return out;
}

function editDistance(a, b, cap) {
  if (Math.abs(a.length - b.length) > cap) return cap + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      best = Math.min(best, cur[j]);
    }
    if (best > cap) return cap + 1;
    prev = cur;
  }
  return prev[b.length];
}

const stripAccents = s => s.normalize('NFD').replace(/\p{M}/gu, '');

/** "Did you mean" candidates: same word without accents, then small typos, then prefixes. */
export function suggestions(lang, raw, n = 5) {
  const q = normalizeWord(lang.code, raw);
  if (!q) return [];
  const qa = stripAccents(q);
  const cap = q.length <= 3 ? 1 : 2;
  const scored = [];
  lang.words.forEach((w, i) => {
    const k = normalizeWord(lang.code, w);
    let score = null;
    if (stripAccents(k) === qa) score = 0;
    else if (Math.abs(k.length - q.length) <= cap) {
      const d = editDistance(qa, stripAccents(k), cap);
      if (d <= cap) score = d;
    }
    if (score === null && q.length >= 3 && k.startsWith(q)) score = 2.5;
    if (score !== null) scored.push({ w, score: score + i / 1e6 });
  });
  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, n).map(s => s.w);
}

/** Words starting with what has been typed so far, most frequent first. */
export function completions(lang, raw, n = 8) {
  const q = normalizeWord(lang.code, raw);
  if (!q) return [];
  const out = [];
  for (let i = 0; i < lang.words.length && out.length < n; i++) {
    if (normalizeWord(lang.code, lang.words[i]).startsWith(q)) out.push(lang.words[i]);
  }
  return out;
}

/**
 * Plan the journey of one word along `stops` (e.g. ['no', 'es', 'ja']).
 * mode 'express': the vector rides straight through and only becomes a word at the end.
 * mode 'telephone': at every stop it snaps to the nearest actual word and carries on from there.
 */
export function planWord(L, stops, mode, idx) {
  const src = L[stops[0]];
  let v = vec(src, idx);
  let cur = src.code;
  const steps = [{ type: 'lookup', lang: cur, idx, vec: v }];
  const chain = [{ lang: cur, idx, word: src.words[idx], sim: 1 }];

  for (let s = 1; s < stops.length; s++) {
    const next = stops[s];
    const last = s === stops.length - 1;
    if (cur !== next) {
      if (cur !== HUB) {
        const out = toHub(L[cur], v);
        steps.push({ type: 'rotate', from: cur, to: HUB, matrix: cur, dir: 'toHub', vin: v, vout: out, viaHub: next !== HUB });
        v = out;
      }
      if (next !== HUB) {
        const out = fromHub(L[next], v);
        steps.push({ type: 'rotate', from: HUB, to: next, matrix: next, dir: 'fromHub', vin: v, vout: out });
        v = out;
      }
    }
    const sims = similarities(L[next], v);
    const top = topK(sims, 6);
    const snap = mode === 'telephone' || last;
    steps.push({ type: snap ? 'snap' : 'pass', lang: next, vec: v, sims, top, last });
    chain.push({ lang: next, idx: top[0].idx, word: L[next].words[top[0].idx], sim: top[0].sim, passing: !snap });
    if (snap) v = vec(L[next], top[0].idx);
    cur = next;
  }

  const final = steps[steps.length - 1];
  return {
    steps,
    chain,
    lang: final.lang,
    word: chain[chain.length - 1].word,
    sim: final.top[0].sim,
    top: final.top.slice(0, 5).map(t => ({ ...t, word: L[final.lang].words[t.idx] })),
  };
}

/** All the languages a route touches (the hub included). */
export function languagesFor(stops) {
  return [...new Set([HUB, ...stops])];
}
