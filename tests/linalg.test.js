import { test } from 'node:test';
import assert from 'node:assert/strict';
import { topK, toHub, fromHub, localBasis, project, fitSimilarity, dot } from '../js/linalg.js';
import { planWord, tokenize, suggestions } from '../js/translate.js';
import { normalizeWord } from '../js/data.js';

const close = (a, b, eps = 1e-5) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

function rotation(D, seed = 1) {
  // random orthogonal matrix via Gram-Schmidt
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5;
  const rows = [];
  for (let i = 0; i < D; i++) {
    let r = Array.from({ length: D }, rnd);
    for (const p of rows) { const d = dot(r, p); r = r.map((x, k) => x - d * p[k]); }
    const n = Math.sqrt(dot(r, r));
    rows.push(r.map(x => x / n));
  }
  return Float32Array.from(rows.flat());
}

function fakeLang(code, words, vectors, W) {
  const D = vectors[0].length;
  const vecs = Float32Array.from(vectors.flatMap(v => { const n = Math.hypot(...v); return v.map(x => x / n); }));
  const index = new Map(words.map((w, i) => [normalizeWord(code, w), i]));
  return { code, name: code, words, vecs, W, D, V: words.length, index };
}

test('topK returns the k largest, best first', () => {
  const vals = Float32Array.from([0.1, 0.9, -1, 0.5, 0.95, 0.2]);
  assert.deepEqual(topK(vals, 3).map(t => t.idx), [4, 1, 3]);
  assert.deepEqual(topK(vals, 2, new Set([4])).map(t => t.idx), [1, 3]);
  assert.equal(topK(vals, 10).length, 6);
});

test('fromHub undoes toHub (orthogonal matrices are their own undo button)', () => {
  const D = 6;
  const lang = { W: rotation(D), D };
  const v = Float32Array.from([0.3, -0.1, 0.5, 0.2, -0.7, 0.1]);
  const back = fromHub(lang, toHub(lang, v));
  v.forEach((x, i) => close(back[i], x));
});

test('localBasis is orthonormal and fitSimilarity recovers a rotation', () => {
  const pts = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0.1, 0], [-1, -1, 0, 0.05]].map(p => Float32Array.from(p));
  const [a, b] = localBasis(new Float32Array(4), pts);
  close(dot(a, a), 1);
  close(dot(b, b), 1);
  close(dot(a, b), 0);
  const src = pts.map(p => project([a, b], new Float32Array(4), p));
  const th = 0.7;
  const dst = src.map(([x, y]) => [Math.cos(th) * x - Math.sin(th) * y + 0.2, Math.sin(th) * x + Math.cos(th) * y]);
  const fit = fitSimilarity(src, dst);
  close(fit.theta, th);
  close(fit.scale, 1);
  close(fit.tx, 0.2);
  assert.equal(fit.reflected, false);
  assert.equal(fitSimilarity(src, dst.map(([x, y]) => [x, -y])).reflected, true);
});

test('planWord: express detours cancel out, telephone snaps at every stop', () => {
  const D = 4;
  const base = [[1, 0.1, 0, 0], [0, 1, 0.2, 0], [0, 0, 1, 0.1], [0.3, 0, 0, 1]];
  const R1 = rotation(D, 3), R2 = rotation(D, 7);
  // a language whose space is the hub's space rotated by R: its vectors are hub·Rᵀ, so x·R lands back
  const rotated = R => base.map(v => Array.from({ length: D }, (_, i) => v.reduce((s, x, j) => s + x * R[i * D + j], 0)));
  const L = {
    en: fakeLang('en', ['cat', 'dog', 'beer', 'love'], base, null),
    no: fakeLang('no', ['katt', 'hund', 'øl', 'kjærlighet'], rotated(R1), R1),
    ja: fakeLang('ja', ['猫', '犬', 'ビール', '愛'], rotated(R2), R2),
  };
  const p = planWord(L, ['no', 'ja'], 'express', 2);
  assert.equal(p.word, 'ビール');
  close(p.sim, 1, 1e-4);
  assert.deepEqual(p.steps.map(s => s.type), ['lookup', 'rotate', 'rotate', 'snap']);

  const detour = planWord(L, ['no', 'ja', 'en'], 'express', 0);
  assert.deepEqual(detour.chain.map(c => c.word), ['katt', '猫', 'cat']);
  assert.equal(detour.chain[1].passing, true);

  const tel = planWord(L, ['en', 'no', 'ja'], 'telephone', 1);
  assert.deepEqual(tel.steps.map(s => s.type), ['lookup', 'rotate', 'snap', 'rotate', 'rotate', 'snap']);
  assert.equal(tel.word, '犬');

  const same = planWord(L, ['no', 'no'], 'express', 3);
  assert.equal(same.word, 'kjærlighet');
});

test('tokenize and suggestions', () => {
  const ja = fakeLang('ja', ['猫', '犬', 'ビール', '好き'], [[1, 0], [0, 1], [1, 1], [1, -1]], null);
  assert.deepEqual(tokenize(ja, '猫ビール犬'), ['猫', 'ビール', '犬']);
  const fr = fakeLang('fr', ['amour', 'chat'], [[1, 0], [0, 1]], null);
  assert.deepEqual(tokenize(fr, "l'amour, chat!"), ['amour', 'chat']);
  const no = fakeLang('no', ['kjærlighet', 'katt', 'pingvin'], [[1, 0], [0, 1], [1, 1]], null);
  assert.equal(suggestions(no, 'pingvinn')[0], 'pingvin');
  assert.equal(suggestions(no, 'KATT')[0], 'katt');
});

test('normalizeWord', () => {
  assert.equal(normalizeWord('de', '  Katze '), 'katze');
  assert.equal(normalizeWord('el', 'ΣΚΥΛΟΣ'), 'σκυλος'.replace(/σ$/, 'ς'));
  assert.equal(normalizeWord('ja', 'ｶﾀｶﾅ'), 'カタカナ');
});
