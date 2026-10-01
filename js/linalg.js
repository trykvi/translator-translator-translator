// The entire "AI": dot products, two flavours of matrix-vector product, and a sort.

export function vec(lang, i) {
  return lang.vecs.subarray(i * lang.D, (i + 1) * lang.D);
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function normalize(a) {
  const n = norm(a) || 1;
  return a.map(x => x / n);
}

export function sub(a, b) {
  return a.map((x, i) => x - b[i]);
}

/** Rotate a vector from `lang`'s space into the hub (English) space: v·W. */
export function toHub(lang, v) {
  const { W, D } = lang;
  const out = new Float32Array(D);
  for (let j = 0; j < D; j++) {
    const vj = v[j];
    const row = j * D;
    for (let i = 0; i < D; i++) out[i] += vj * W[row + i];
  }
  return out;
}

/** Rotate a vector from the hub space into `lang`'s space: v·Wᵀ (orthogonal, so Wᵀ = W⁻¹). */
export function fromHub(lang, v) {
  const { W, D } = lang;
  const out = new Float32Array(D);
  for (let i = 0; i < D; i++) {
    let s = 0;
    const row = i * D;
    for (let j = 0; j < D; j++) s += W[row + j] * v[j];
    out[i] = s;
  }
  return out;
}

/** The matrix as drawn in the "column vector" convention: entry (i, j) multiplies input j into output i. */
export function displayMatrixEntry(lang, dir, i, j) {
  const { W, D } = lang;
  return dir === 'toHub' ? W[j * D + i] : W[i * D + j];
}

/** Cosine similarity of `v` to every word in the language (vectors are unit length). */
export function similarities(lang, v) {
  const { vecs, D, V } = lang;
  const n = norm(v) || 1;
  const out = new Float32Array(V);
  for (let i = 0; i < V; i++) {
    let s = 0;
    const o = i * D;
    for (let d = 0; d < D; d++) s += vecs[o + d] * v[d];
    out[i] = s / n;
  }
  return out;
}

/** Indices of the k largest values, best first. */
export function topK(values, k, skip = null) {
  const idx = [];
  const val = [];
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    if (val.length === k && x <= val[k - 1]) continue;
    if (skip && skip.has(i)) continue;
    // insertion sort into a list of at most k (when full, the last one falls off)
    let p = Math.min(val.length, k - 1);
    while (p > 0 && val[p - 1] < x) {
      val[p] = val[p - 1];
      idx[p] = idx[p - 1];
      p--;
    }
    val[p] = x;
    idx[p] = i;
  }
  return idx.map((i, r) => ({ idx: i, sim: val[r] }));
}

/**
 * Two orthonormal directions that best show the spread of `points` around `center`
 * (a tiny PCA, done on the Gram matrix with power iteration).
 */
export function localBasis(center, points) {
  const D = center.length;
  const Y = points.map(p => sub(p, center));
  const K = Y.length;
  const G = Y.map(a => Y.map(b => dot(a, b)));
  const basis = [];
  for (let c = 0; c < 2; c++) {
    let u = Array.from({ length: K }, (_, i) => 1 + ((i * 7 + c * 3) % 5) / 10);
    for (let it = 0; it < 60; it++) {
      const nu = G.map(row => dot(row, u));
      const n = norm(nu);
      if (!n) break;
      u = nu.map(x => x / n);
    }
    let b = new Float32Array(D);
    Y.forEach((y, i) => { for (let d = 0; d < D; d++) b[d] += u[i] * y[d]; });
    for (const prev of basis) {
      const p = dot(b, prev);
      for (let d = 0; d < D; d++) b[d] -= p * prev[d];
    }
    if (norm(b) < 1e-9) {
      b = new Float32Array(D);
      b[c] = 1;
    }
    basis.push(normalize(b));
    // deflate: remove this direction from the Gram matrix
    const proj = Y.map(y => dot(y, basis[c]));
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) G[i][j] -= proj[i] * proj[j];
  }
  return basis;
}

export function project(basis, center, x) {
  let a = 0, b = 0;
  for (let d = 0; d < x.length; d++) {
    const y = x[d] - center[d];
    a += basis[0][d] * y;
    b += basis[1][d] * y;
  }
  return [a, b];
}

/**
 * Least-squares rotation + uniform scale + shift taking 2D points `src` to `dst`
 * (2D Procrustes). Returns { theta, scale, tx, ty, reflected } where `reflected`
 * says a mirror image would have fitted better.
 */
export function fitSimilarity(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let a = 0, b = 0, ar = 0, br = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const px = src[i][0] - sx, py = src[i][1] - sy;
    const qx = dst[i][0] - dx, qy = dst[i][1] - dy;
    a += px * qx + py * qy;
    b += px * qy - py * qx;
    ar += px * qx - py * qy; // same, with src mirrored in the x axis
    br += -px * qy - py * qx;
    ss += px * px + py * py;
  }
  const theta = Math.atan2(b, a);
  const scale = ss ? Math.hypot(a, b) / ss : 1;
  const c = Math.cos(theta) * scale, s = Math.sin(theta) * scale;
  return {
    theta,
    scale,
    tx: dx - (c * sx - s * sy),
    ty: dy - (s * sx + c * sy),
    reflected: Math.hypot(ar, br) > Math.hypot(a, b) * 1.05,
  };
}
