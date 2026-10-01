#!/usr/bin/env python3
"""Build data/ for Translator Translator Translator.

For every language this produces a word list, a quantized embedding table and a
D×D translation matrix W that maps that language's embedding space into the
English one (English is the hub, because that is where the dictionaries are).

Pipeline
  1. Word vectors   spaCy model packages (downloaded from their GitHub releases)
  2. Vocabulary     the top-N words by wordfreq that own a vector
  3. Translations   Open Multilingual Wordnet: two words are a "translation" if
                    they share a synset (linked across languages via the ILI)
  4. Geometry       unit-normalise, centre, re-normalise, PCA down to D dims
  5. Matrices       orthogonal Procrustes on the translation pairs, then a few
                    rounds of CSLS mutual-nearest-neighbour self-learning
  6. Export         int8 vectors (per-dimension scale) + float32 matrices

So yes: the translator is learned from translations. That is the joke.

Usage:  pip install -r scripts/requirements.txt && python scripts/build_data.py
"""
import argparse
import collections
import datetime
import glob
import json
import os
import random
import struct
import subprocess
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, '.cache')

SPACY_RELEASES = 'https://github.com/explosion/spacy-models/releases/download'
SPACY_VERSION = '3.8.0'

# code: display name, native name, spaCy model, wordfreq code, wordnet lexicons
LANGS = {
    'en': ('English', 'English', 'en_core_web_lg', 'en', ['omw-en']),
    'no': ('Norwegian', 'Norsk', 'nb_core_news_lg', 'nb', ['omw-nb', 'omw-nn']),
    'ja': ('Japanese', '日本語', 'ja_core_news_lg', 'ja', ['omw-ja']),
    'es': ('Spanish', 'Español', 'es_core_news_lg', 'es', ['omw-es']),
    'de': ('German', 'Deutsch', 'de_core_news_lg', 'de', ['odenet']),
    'fr': ('French', 'Français', 'fr_core_news_lg', 'fr', ['omw-fr']),
    'it': ('Italian', 'Italiano', 'it_core_news_lg', 'it', ['omw-iwn', 'omw-it']),
    'pt': ('Portuguese', 'Português', 'pt_core_news_lg', 'pt', ['omw-pt']),
    'nl': ('Dutch', 'Nederlands', 'nl_core_news_lg', 'nl', ['omw-nl']),
    'pl': ('Polish', 'Polski', 'pl_core_news_lg', 'pl', ['omw-pl']),
    'el': ('Greek', 'Ελληνικά', 'el_core_news_lg', 'el', ['omw-el']),
    'zh': ('Chinese', '中文', 'zh_core_web_lg', 'zh', ['omw-cmn']),
}
HUB = 'en'

# Elision fragments and other one-letter debris from tokenisation
SINGLE_LETTER_JUNK = {'en': set('stdm'), 'fr': set('ldjnscmt'), 'it': {'l'}, 'pt': {'d'}, 'es': {'d'}}

# Words people will obviously try. Every language gets the nearest words to these (as mapped by its own
# matrix) added to its vocabulary, even if they are too rare to make the frequency cut.
FUN_WORDS = set("""
penguin giraffe elephant lion tiger bear wolf fox rabbit mouse rat horse cow pig sheep goat chicken duck
goose owl eagle parrot shark whale dolphin octopus squid crab lobster frog snake turtle lizard crocodile
monkey gorilla kangaroo koala panda hedgehog squirrel deer moose reindeer camel zebra hippo rhino bee ant
spider butterfly mosquito snail worm seal walrus otter beaver badger hamster llama sloth unicorn dragon
dinosaur mammoth cat dog kitten puppy
pizza banana apple orange lemon strawberry grape cherry pear peach pineapple watermelon tomato potato
carrot onion garlic cucumber mushroom cheese butter bread waffle pancake cake cookie chocolate candy sugar
salt pepper honey jam sausage bacon ham egg rice noodles pasta spaghetti soup salad sandwich burger
hamburger sushi taco coffee tea milk juice beer wine vodka whisky water fish salmon shrimp meat steak
popcorn porridge dessert
troll ghost vampire zombie wizard witch elf dwarf giant monster robot alien ninja pirate knight princess
king queen castle sword magic demon angel mermaid goblin
computer phone television car bicycle train airplane boat rocket umbrella chair table bed sofa lamp clock
book pencil guitar piano drum ball hat shoe sock glasses key door window toilet bathtub sauna spoon fork
knife cup bottle balloon kite toothbrush soap pillow blanket mirror
sun moon star planet cloud rain snow storm thunder lightning rainbow mountain volcano island ocean river
lake forest tree flower rose desert beach fjord glacier iceberg
love happiness sadness anger fear dream friendship chaos nonsense freedom luck
grandmother grandfather mother father baby teacher doctor president translator
""".split())

# Pairs to report pivot accuracy for (src → en → tgt, scored against Wordnet)
PIVOT_EVAL = [('no', 'ja'), ('no', 'es'), ('ja', 'es'), ('de', 'fr'), ('es', 'it'), ('zh', 'ja')]


# --------------------------------------------------------------------------- data sources

def spacy_vocab_dir(model):
    """Download a spaCy model wheel (once) and return the path of its vocab/ folder."""
    os.makedirs(CACHE, exist_ok=True)
    whl = os.path.join(CACHE, f'{model}-{SPACY_VERSION}.whl')
    out = os.path.join(CACHE, model)
    if not glob.glob(f'{out}/*/*/vocab'):
        if not os.path.exists(whl):
            url = f'{SPACY_RELEASES}/{model}-{SPACY_VERSION}/{model}-{SPACY_VERSION}-py3-none-any.whl'
            print(f'  downloading {url}', flush=True)
            subprocess.run(['curl', '-sSLf', '-o', whl, url], check=True)
        with zipfile.ZipFile(whl) as z:
            z.extractall(out, [n for n in z.namelist() if '/vocab/' in n])
    return glob.glob(f'{out}/*/*/vocab')[0]


def setup_wordnets():
    import wn
    wn.config.data_directory = os.path.join(CACHE, 'wn')
    have = {lx.id for lx in wn.lexicons()}
    if 'omw-en' not in have:
        wn.download('omw:1.4', progress_handler=None)
    if 'odenet' not in have:
        wn.download('odenet', progress_handler=None)
    return wn


def synset_lemmas(wn, lexicons):
    """ILI id -> lemmas, over all the given lexicons."""
    out = collections.defaultdict(set)
    for lx in lexicons:
        for ss in wn.synsets(lexicon=lx):
            ili = ss.ili if isinstance(ss.ili, str) else (ss.ili.id if ss.ili else None)
            if ili:
                out[ili].update(str(l) for l in ss.lemmas())
    return out


def load_vocab(code, size, de_nouns):
    """Top `size` words by frequency that own a vector. Returns (display words, 300-d vectors)."""
    from spacy.strings import hash_string
    from spacy.vectors import Vectors
    import wordfreq

    _, _, model, wf, _ = LANGS[code]
    vec = Vectors().from_disk(spacy_vocab_dir(model))
    data = np.asarray(vec.data, dtype=np.float32)
    words, rows, seen, claimed = [], [], set(), set()
    for w in wordfreq.top_n_list(wf, 120000):
        if not w.isalpha() or w in SINGLE_LETTER_JUNK.get(code, ()):
            continue
        if code == 'el' and w.endswith('σ'):
            w = w[:-1] + 'ς'  # str.casefold() turns final sigma into σ
        # Some models are pruned: rare words share a row with a frequent neighbour. A row belongs
        # to the most frequent word that maps to it, so claim rows in frequency order.
        found = [(c, vec.key2row.get(hash_string(c))) for c in (w, w.capitalize())]
        found = [(c, r) for c, r in found if r is not None and r not in claimed]
        if not found:
            continue
        if code == 'de':
            # German nouns live capitalised in the vectors: take the more frequent casing (lower row),
            # and on a tie (a pruned variant pointing at the same row) the capitalised noun
            found.sort(key=lambda cr: (cr[1], 0 if cr[0] in de_nouns else 1 if cr[0] == w else 2))
            display, row = found[0]
        else:
            display, row = w, found[0][1]
        if display.lower() in seen or not np.any(data[row]):
            continue
        seen.add(display.lower())
        claimed.add(row)
        words.append(display)
        rows.append(row)
        if len(words) == size:
            break
    return words, data[rows]


# --------------------------------------------------------------------------- geometry

def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


class Reducer:
    """unit-normalise, centre, re-normalise, PCA to d dims, re-normalise. Fitted on one vocabulary,
    applicable to any other words of the same language."""

    def __init__(self, raw, d):
        X = unit(raw)
        self.mean = X.mean(0)
        X = unit(X - self.mean)
        _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        self.P = Vt[:d].T

    def __call__(self, raw):
        return unit(unit(unit(raw) - self.mean) @ self.P)


def procrustes(A, B):
    """Orthogonal W minimising ||AW - B||."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def knn_mean_sim(X, Y, k=10, batch=2048):
    """For each row of X, the mean cosine to its k nearest rows of Y."""
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), batch):
        s = X[i:i + batch] @ Y.T
        out[i:i + batch] = np.partition(s, -k, axis=1)[:, -k:].mean(1)
    return out


def refine(Xs, Xt, W, seed, n=5000, iters=3):
    """Self-learning: add CSLS mutual nearest neighbours among frequent words to the seed pairs."""
    for _ in range(iters):
        M, T = Xs[:n] @ W, Xt[:n]
        sim = 2 * M @ T.T - knn_mean_sim(M, T)[:, None] - knn_mean_sim(T, M)[None, :]
        fwd, bwd = sim.argmax(1), sim.argmax(0)
        mutual = [(i, j) for i, j in enumerate(fwd) if bwd[j] == i]
        pairs = seed + mutual
        W = procrustes(Xs[[a for a, _ in pairs]], Xt[[b for _, b in pairs]])
    return W


def translation_pairs(src_words, src_ili, tgt_words, tgt_ili):
    """src index -> set of tgt indices that share a Wordnet synset."""
    si = {w.lower(): i for i, w in enumerate(src_words)}
    ti = {w.lower(): i for i, w in enumerate(tgt_words)}
    P = collections.defaultdict(set)
    for ili, lems in src_ili.items():
        tgt = {ti[t.lower()] for t in tgt_ili.get(ili, ()) if t.lower() in ti}
        if not tgt:
            continue
        for l in lems:
            if l.lower() in si:
                P[si[l.lower()]] |= tgt
    return P


def fit(Xs, Xt, P):
    seed = [(s, t) for s in P for t in P[s]]
    W = procrustes(Xs[[a for a, _ in seed]], Xt[[b for _, b in seed]])
    return refine(Xs, Xt, W, seed)


def precision_at_1(Q, Xt, gold):
    pred = (Q @ Xt.T).argmax(1)
    return float(np.mean([pred[i] in g for i, g in enumerate(gold)]))


# --------------------------------------------------------------------------- export

def quantize(X):
    scale = np.abs(X).max(0)  # per-dimension, PCA dims have very different spreads
    q = np.clip(np.round(X / scale * 127), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def write_lang(out, code, words, X, W):
    """Binary layout (little endian): float32 scale[D] | float32 W[D*D] (row-major, x·W) |
    int8 vectors[V*D]."""
    q, scale = quantize(X)
    with open(os.path.join(out, f'{code}.bin'), 'wb') as f:
        f.write(scale.astype('<f4').tobytes())
        f.write(W.astype('<f4').tobytes())
        f.write(q.tobytes())
    with open(os.path.join(out, f'{code}.json'), 'w', encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False, separators=(',', ':'))
    # what the browser will actually see
    return q.astype(np.float32) * scale / 127


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--langs', default=','.join(LANGS))
    ap.add_argument('--vocab', type=int, default=15000, help='most frequent words per language')
    ap.add_argument('--pool', type=int, default=60000, help='less frequent words the top-up may draw from')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--out', default=os.path.join(ROOT, 'data'))
    args = ap.parse_args()
    codes = [HUB] + [c for c in args.langs.split(',') if c != HUB]
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)

    wn = setup_wordnets()
    de_nouns = {str(l) for ss in wn.synsets(lexicon='odenet', pos='n') for l in ss.lemmas()}

    words, X, ili, pool = {}, {}, {}, {}
    for c in codes:
        print(f'[{c}] vocabulary + vectors', flush=True)
        all_words, raw = load_vocab(c, args.pool, de_nouns)
        reduce = Reducer(raw[:args.vocab], args.dim)
        words[c], X[c] = all_words[:args.vocab], reduce(raw[:args.vocab])
        pool[c] = (all_words[args.vocab:], reduce(raw[args.vocab:]))
        ili[c] = synset_lemmas(wn, LANGS[c][4])

    # ---- learn W[c]: c-space -> en-space, with a held-out evaluation first
    W, stats = {HUB: np.eye(args.dim, dtype=np.float32)}, {}
    for c in codes[1:]:
        P = translation_pairs(words[c], ili[c], words[HUB], ili[HUB])
        frequent = sorted(s for s in P if s < 5000)
        rng.shuffle(frequent)
        test = frequent[:max(100, len(frequent) // 8)]
        train = {s: P[s] for s in P if s not in set(test)}
        Wc = fit(X[c], X[HUB], train)
        p1 = precision_at_1(X[c][test] @ Wc, X[HUB], [P[s] for s in test])
        W[c] = fit(X[c], X[HUB], P)
        stats[c] = {'pairs': sum(len(v) for v in P.values()), 'p1': round(p1, 3), 'test': len(test)}
        print(f'[{c}] {stats[c]["pairs"]} translation pairs, held-out P@1 → en: {p1:.2f}', flush=True)

    # ---- vocabulary top-up: people *will* type "penguin". For each such word, let each language's
    # matrix pull its nearest words in from the less frequent pool.
    have = {w.lower() for w in words[HUB]}
    pw, pX = pool[HUB]
    extra = [i for i, w in enumerate(pw) if w.lower() in FUN_WORDS and w.lower() not in have]
    words[HUB] += [pw[i] for i in extra]
    X[HUB] = np.vstack([X[HUB], pX[extra]])
    fun = [i for i, w in enumerate(words[HUB]) if w.lower() in FUN_WORDS]
    print(f'[{HUB}] +{len(extra)} words ({len(fun)} of {len(FUN_WORDS)} fun words known)', flush=True)
    for c in codes[1:]:
        pw, pX = pool[c]
        q = X[HUB][fun] @ W[c].T
        cands = np.vstack([X[c], pX]) @ q.T  # (base + pool) × queries
        best = np.argsort(-cands, axis=0)[:3].ravel()
        extra = sorted({int(b) - len(words[c]) for b in best if b >= len(words[c])})
        words[c] += [pw[i] for i in extra]
        X[c] = np.vstack([X[c], pX[extra]])
        print(f'[{c}] +{len(extra)} words, e.g. {[pw[i] for i in extra[:12]]}', flush=True)

    # ---- export, then evaluate on exactly what the browser gets
    Xq = {}
    for c in codes:
        Xq[c] = write_lang(args.out, c, words[c], X[c], W[c])
    Xqn = {c: Xq[c] / np.linalg.norm(Xq[c], axis=1, keepdims=True) for c in codes}

    pivot = []
    for a, b in PIVOT_EVAL:
        if a not in codes or b not in codes:
            continue
        P = translation_pairs(words[a], ili[a], words[b], ili[b])
        src = [s for s in sorted(P) if s < 5000]
        Q = Xqn[a][src] @ W[a] @ W[b].T
        p1 = precision_at_1(Q, Xqn[b], [P[s] for s in src])
        pivot.append({'src': a, 'tgt': b, 'n': len(src), 'p1': round(p1, 3)})
        print(f'[pivot] {a}→en→{b}: P@1 {p1:.2f} (n={len(src)})', flush=True)

    manifest = {
        'built': datetime.date.today().isoformat(),
        'hub': HUB,
        'dim': args.dim,
        'layout': 'float32 scale[dim] | float32 W[dim*dim] row-major (y = x·W maps into hub space) | '
                  'int8 vectors[words*dim]',
        'languages': [{
            'code': c,
            'name': LANGS[c][0],
            'native': LANGS[c][1],
            'words': len(words[c]),
            'vectors': LANGS[c][2],
            'wordnets': LANGS[c][4],
            **stats.get(c, {}),
        } for c in codes],
        'pivot_eval': pivot,
    }
    with open(os.path.join(args.out, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print('done →', args.out)


if __name__ == '__main__':
    sys.exit(main())
