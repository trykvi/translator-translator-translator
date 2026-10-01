// All the jokes live here, so the rest of the code can keep a straight face.

const pick = list => list[Math.floor(Math.random() * list.length)];

export const lookupQuips = [
  'It is now 128 numbers. It has never been happier.',
  'Words are just vectors that haven’t been multiplied yet.',
  'Meaning: approximately ±0.1 per dimension.',
  'Step one of turning language into arithmetic. There is no step two of understanding it.',
  'We don’t know what dimension 37 means either.',
];

export function rotateQuip(step, L) {
  const from = L[step.from].name, to = L[step.to].name;
  if (step.dir === 'toHub') {
    return pick([
      `Rotating ${from} space into English space. Nobody asked ${from} space how it feels about this.`,
      `English is the hub, because that’s where the dictionaries were. Sorry, ${from}.`,
      `${(L[step.from].D ** 2).toLocaleString('en-US')} multiplications. Done before you finished reading this sentence.`,
      `A rotation in ${L[step.from].D} dimensions. The 2D picture is doing its best.`,
      `Every ${from} word moves at once. We only care about one of them.`,
    ]);
  }
  return pick([
    `Same matrix that takes ${to} to English, just transposed. Rotations are their own undo button.`,
    `Rotating English space into ${to} space. Mind the gap.`,
    `Landing in ${to} space. Please keep your vectors inside the vehicle.`,
    `The ${to} words were here all along. We just had to turn around.`,
  ]);
}

export function passQuip(word, lang) {
  return pick([
    `Nearest ${lang.name} word is “${word}”, but this is Express. We don’t stop.`,
    `Waving at “${word}” from the window.`,
    `“${word}” would have been the answer, if this were the destination. It isn’t.`,
  ]);
}

export function searchQuip(lang) {
  return pick([
    `Comparing against all ${lang.V.toLocaleString('en-US')} ${lang.name} words. Closest one wins.`,
    'Nearest-neighbour search: the “close enough” of algorithms.',
    'The vector almost never lands exactly on a word. So we pick the closest and act confident.',
  ]);
}

export function telephoneSnapQuip(word) {
  return pick([
    `From now on we’re translating “${word}”. No take-backs.`,
    `The original word is gone. Long live “${word}”.`,
    `Snapped to “${word}”. Whatever got lost, stays lost.`,
  ]);
}

export function confidence(sim) {
  if (sim >= 0.999) return 'Translated it into itself. Flawless.';
  if (sim >= 0.8) return 'Suspiciously confident.';
  if (sim >= 0.65) return 'Confident. Possibly even correct.';
  if (sim >= 0.5) return 'Fairly sure. Would bet a waffle on it.';
  if (sim >= 0.4) return 'An educated guess.';
  if (sim >= 0.3) return 'Vibes-based translation.';
  return 'Pure linear-algebraic hallucination.';
}

export function tourVerdict(first, last, sameLang) {
  if (!sameLang) return '';
  if (first.toLowerCase() === last.toLowerCase()) {
    return pick([
      `Survived the whole trip and came home as “${last}”. Statistically suspicious.`,
      `Went around the world and came back unchanged. Like a cat.`,
    ]);
  }
  return pick([
    `Left home as “${first}”, came back as “${last}”. Close enough?`,
    `“${first}” went on holiday and came back as “${last}”. It doesn’t want to talk about it.`,
    `This is why nobody plays telephone with matrices.`,
  ]);
}

export function sameLanguageQuip(lang) {
  return `Translated ${lang.name} into ${lang.name}. Our best work yet.`;
}

export function expressDetourNote(detours) {
  const names = detours.join(' and ');
  return `Fun fact: in Express mode the detour through ${names} cancelled out exactly (W·Wᵀ = I). The vector went all that way and nothing happened. Try Telephone mode.`;
}

export function notFound(word, lang) {
  return pick([
    `“${word}” isn’t one of the ${lang.V.toLocaleString('en-US')} ${lang.name} words we know. Our vocabulary is small, but our matrices are large.`,
    `Never heard of “${word}”. In fairness, we’ve never heard anything. We only multiply.`,
    `“${word}”? Not in our ${lang.name} vocabulary. Maybe it’s too rare, maybe it’s a typo, maybe it’s just made up.`,
  ]);
}

export const loadingQuips = [
  'Teaching matrices to read…',
  'Unpacking several thousand dimensions…',
  'Calibrating vibes…',
  'Downloading a language. All of it. Well, 15,000 words of it.',
];

export const modeHints = {
  express: 'Express: the vector rides straight through every stop and only becomes a word at the end. Most accurate. Least fun.',
  telephone: 'Telephone: at every stop the vector snaps to the nearest actual word, and the next leg translates that word. Errors accumulate. That’s the point.',
};

export { pick };
