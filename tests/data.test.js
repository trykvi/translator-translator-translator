// Checks the shipped data files decode correctly and still translate the classics.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeLanguage } from '../js/data.js';
import { planWord, lookup } from '../js/translate.js';

const dir = new URL('../data/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', dir)));
const L = {};
for (const { code } of manifest.languages) {
  const words = JSON.parse(readFileSync(new URL(`${code}.json`, dir)));
  const b = readFileSync(new URL(`${code}.bin`, dir));
  L[code] = decodeLanguage(manifest, code, words, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

test('at least 8 languages, including the required ones', () => {
  const codes = manifest.languages.map(l => l.code);
  assert.ok(codes.length >= 8);
  for (const c of ['en', 'no', 'ja', 'es']) assert.ok(codes.includes(c), c);
});

test('every matrix is orthogonal', () => {
  for (const lang of Object.values(L)) {
    const { W, D } = lang;
    let worst = 0;
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < D; j++) {
        let s = 0;
        for (let k = 0; k < D; k++) s += W[i * D + k] * W[j * D + k];
        worst = Math.max(worst, Math.abs(s - (i === j ? 1 : 0)));
      }
    }
    assert.ok(worst < 1e-4, `${lang.code}: |WWᵀ - I| = ${worst}`);
  }
});

const classics = [
  ['en', 'no', 'dog', 'hund'], ['en', 'no', 'beer', 'øl'], ['no', 'en', 'katt', 'cat'],
  ['en', 'ja', 'cat', '猫'], ['en', 'ja', 'beer', 'ビール'], ['no', 'ja', 'hus', '家'],
  ['en', 'es', 'dog', 'perro'], ['es', 'en', 'agua', 'water'], ['en', 'de', 'king', 'König'],
  ['en', 'fr', 'cheese', 'fromage'], ['en', 'it', 'friend', 'amico'], ['no', 'es', 'øl', 'cerveza'],
  ['en', 'no', 'penguin', 'pingvin'], ['ja', 'en', '犬', 'dog'],
];

test('the classics still translate (mostly)', () => {
  const misses = [];
  for (const [a, b, word, want] of classics) {
    const i = lookup(L[a], word);
    assert.ok(i >= 0, `${word} missing from ${a}`);
    const got = planWord(L, [a, b], 'express', i).word;
    if (got !== want) misses.push(`${a}→${b} ${word}: got ${got}, wanted ${want}`);
  }
  // it's a joke translator, but it shouldn't be *that* much of a joke
  assert.ok(misses.length <= 2, misses.join('\n'));
});
