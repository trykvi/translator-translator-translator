// Loads data/manifest.json and the per-language word lists + embeddings (see scripts/build_data.py).

const BASE = new URL('../data/', import.meta.url);
let manifestPromise = null;
const languages = new Map();

async function fetchOk(name, as) {
  const res = await fetch(new URL(name, BASE));
  if (!res.ok) throw new Error(`Could not load data/${name} (HTTP ${res.status})`);
  return as === 'json' ? res.json() : res.arrayBuffer();
}

export function loadManifest() {
  manifestPromise ??= fetchOk('manifest.json', 'json');
  return manifestPromise;
}

/** Lowercase + Unicode-normalise so lookups forgive casing and full-width characters. */
export function normalizeWord(code, s) {
  s = s.normalize('NFKC').trim().toLowerCase();
  if (code === 'el') s = s.replace(/σ$/u, 'ς');
  return s;
}

export function isLoaded(code) {
  return languages.has(code) && languages.get(code).ready;
}

export function loadLanguage(code) {
  if (!languages.has(code)) {
    const p = load(code);
    const entry = { promise: p, ready: false };
    languages.set(code, entry);
    p.then(() => { entry.ready = true; }, () => languages.delete(code));
  }
  return languages.get(code).promise;
}

async function load(code) {
  const manifest = await loadManifest();
  const [words, buf] = await Promise.all([fetchOk(`${code}.json`, 'json'), fetchOk(`${code}.bin`)]);
  return decodeLanguage(manifest, code, words, buf);
}

/** Unpack one language: float32 scale[D] | float32 W[D*D] | int8 vectors[V*D] (see scripts/build_data.py). */
export function decodeLanguage(manifest, code, words, buf) {
  const info = manifest.languages.find(l => l.code === code);
  if (!info) throw new Error(`Unknown language "${code}"`);
  const D = manifest.dim;
  const V = words.length;
  if (buf.byteLength !== 4 * D + 4 * D * D + V * D) throw new Error(`data/${code}.bin has an unexpected size`);

  const scale = new Float32Array(buf, 0, D);
  const W = new Float32Array(buf.slice(4 * D, 4 * D + 4 * D * D));
  const q = new Int8Array(buf, 4 * D + 4 * D * D, V * D);

  const vecs = new Float32Array(V * D);
  for (let i = 0; i < V; i++) {
    const o = i * D;
    let n = 0;
    for (let d = 0; d < D; d++) {
      const x = q[o + d] * scale[d];
      vecs[o + d] = x;
      n += x * x;
    }
    n = 1 / Math.sqrt(n);
    for (let d = 0; d < D; d++) vecs[o + d] *= n;
  }

  const index = new Map();
  words.forEach((w, i) => {
    const k = normalizeWord(code, w);
    if (!index.has(k)) index.set(k, i);
  });

  return { code, name: info.name, native: info.native, info, words, vecs, W, D, V, index };
}
