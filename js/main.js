import { loadManifest, loadLanguage, isLoaded } from './data.js';
import { vec, toHub, fromHub } from './linalg.js';
import { clock, wait, Cancelled } from './anim.js';
import { SpaceStage, makeFrame, fitTransition } from './stage.js';
import { MathPanel } from './mathview.js';
import { HUB, lookup, tokenize, suggestions, completions, planWord, languagesFor } from './translate.js';
import * as Q from './quips.js';

const $ = sel => document.querySelector(sel);
const el = {
  src: $('#src'), tgt: $('#tgt'), swap: $('#swap'), form: $('#ask'), word: $('#word'), go: $('#go'),
  random: $('#random'), suggest: $('#suggest'), mode: $('#mode'), route: $('#route'), detour: $('#detour'),
  tour: $('#tour'), speed: $('#speed'), hint: $('#mode-hint'), step: $('#step'), quip: $('#quip'),
  progress: $('#progress'), result: $('#result'), stage: $('.stage'), stats: $('#stats tbody'), title: $('#title'),
};

const stage = new SpaceStage($('#space'), $('#space-label'));
const math = new MathPanel($('#math'), $('#math-label'));
math.idle();

const MAX_DETOURS = 12;
const state = { src: 'en', tgt: 'ja', detours: [], mode: 'express', manifest: null, run: null };
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode etc. */ } },
};

const escapeHTML = s => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const langName = code => state.manifest.languages.find(l => l.code === code)?.name ?? code;
const isCJK = code => code === 'ja' || code === 'zh';

// ------------------------------------------------------------------ captions & progress

function caption(html, quip = '') {
  el.step.innerHTML = html;
  el.quip.textContent = quip;
}
const w = s => `<span class="w">${escapeHTML(s)}</span>`;

function buildProgress(steps) {
  const nodes = [{ code: steps[0].lang, hub: false }];
  for (const s of steps) if (s.type === 'rotate') nodes.push({ code: s.to, hub: s.to === HUB && s.viaHub });
  el.progress.innerHTML = nodes.map(n =>
    `<li class="${n.hub ? 'hub' : ''}"><span class="node" title="${n.hub ? 'English: the hub' : langName(n.code)}">${n.code.toUpperCase()}</span></li>`).join('');
  return [...el.progress.children];
}

function setProgress(nodes, i) {
  nodes.forEach((n, k) => {
    n.classList.toggle('active', k === i);
    n.classList.toggle('done', k < i);
  });
}

// ------------------------------------------------------------------ the show

async function animate(plan, L, signal, { index, count }) {
  const nodes = buildProgress(plan.steps);
  let node = 0;
  let F = null;
  const prefix = count > 1 ? `Word ${index + 1} of ${count}: ` : '';

  for (const step of plan.steps) {
    const lang = L[step.lang];
    if (step.type === 'lookup') {
      const word = lang.words[step.idx];
      setProgress(nodes, 0);
      caption(`${prefix}Look up ${w(word)} in ${lang.name}`, Q.pick(Q.lookupQuips));
      F = makeFrame(lang, step.vec, { self: step.idx });
      await Promise.all([stage.enter(F, word, { signal }), math.showVector(lang, word, step.vec, { signal })]);
      await wait(500, signal);
    } else if (step.type === 'rotate') {
      setProgress(nodes, ++node);
      const m = L[step.matrix];
      const sym = step.dir === 'toHub' ? `W<sub>${step.from}→en</sub>` : `W<sub>en→${step.to}</sub>`;
      caption(`Multiply by <i>${sym}</i>: ${L[step.from].name} space → ${L[step.to].name} space`, Q.rotateQuip(step, L));
      const G = makeFrame(L[step.to], step.vout);
      const fit = fitTransition(F, G, x => (step.dir === 'toHub' ? toHub(m, x) : fromHub(m, x)));
      await Promise.all([stage.transition(F, G, fit, { signal }), math.showMatmul(step, L, { signal })]);
      F = G;
      await wait(350, signal);
    } else if (step.type === 'pass') {
      const word = lang.words[step.top[0].idx];
      caption(`Passing through ${lang.name} space`, Q.passQuip(word, lang));
      await Promise.all([
        stage.highlight(F, step.top, { signal, final: false, ms: 1100 }),
        math.showSearch(step, L, { signal, ms: 1100 }),
      ]);
      await wait(300, signal);
    } else if (step.type === 'snap') {
      const top = step.top[0];
      const word = lang.words[top.idx];
      caption(step.last ? `Nearest ${lang.name} word: ${w(word)}` : `Stop: ${lang.name}. Snap to the nearest word: ${w(word)}`,
        step.last ? Q.searchQuip(lang) : Q.telephoneSnapQuip(word));
      await Promise.all([stage.highlight(F, step.top, { signal }), math.showSearch(step, L, { signal })]);
      if (!step.last) {
        await wait(450, signal);
        const G = makeFrame(lang, vec(lang, top.idx), { self: top.idx });
        const fit = fitTransition(F, G, x => x);
        await stage.transition(F, G, fit, { mode: 'recenter', signal, ms: 1000, queryLabel: word });
        F = G;
        await wait(250, signal);
      }
    }
  }
  setProgress(nodes, nodes.length - 1);
}

async function translate() {
  const text = el.word.value.trim();
  hideSuggest();
  if (!text) {
    el.word.focus();
    caption('Type a word first.', 'We can multiply nothing by a matrix, but the result is underwhelming.');
    return;
  }
  state.run?.abort();
  const run = new AbortController();
  state.run = run;
  const stops = [state.src, ...state.detours, state.tgt];
  writeHash(text);

  try {
    const codes = languagesFor(stops);
    if (!codes.every(isLoaded)) {
      caption('Loading word vectors…', Q.pick(Q.loadingQuips));
      el.go.disabled = true;
    }
    const L = Object.fromEntries(await Promise.all(codes.map(async c => [c, await loadLanguage(c)])));
    el.go.disabled = false;
    if (run.signal.aborted) return;

    const tokens = tokenize(L[state.src], text).slice(0, 8);
    const idxs = tokens.map(t => lookup(L[state.src], t));
    if (idxs.every(i => i < 0)) {
      showNotFound(tokens.length === 1 ? tokens[0] : text, L[state.src]);
      return;
    }
    const plans = idxs.map(i => (i >= 0 ? planWord(L, stops, state.mode, i) : null));
    el.result.hidden = true;
    const box = el.stage.getBoundingClientRect();
    if (clock.speed && (box.top < 0 || box.bottom > window.innerHeight)) {
      el.stage.scrollIntoView({ behavior: 'smooth', block: box.height < window.innerHeight ? 'center' : 'start' });
    }
    for (let i = 0; i < plans.length; i++) {
      if (plans[i]) await animate(plans[i], L, run.signal, { index: i, count: plans.length });
    }
    showResult({ tokens, plans, L, stops, mode: state.mode });
    swapTitle(L);
  } catch (e) {
    el.go.disabled = false;
    if (e instanceof Cancelled) return;
    console.error(e);
    showError(e);
  } finally {
    if (state.run === run) state.run = null;
  }
}

// ------------------------------------------------------------------ results

function showResult({ tokens, plans, L, stops, mode }) {
  const src = L[stops[0]], tgt = L[stops[stops.length - 1]];
  const ok = plans.filter(Boolean);
  const joiner = isCJK(tgt.code) ? '' : ' ';
  const phrase = plans.length > 1;
  const answer = plans.map((p, i) => (p ? p.word : `[${tokens[i]}]`)).join(joiner);
  const meanSim = ok.reduce((a, p) => a + p.sim, 0) / ok.length;
  const first = ok[0];
  const sameLang = src.code === tgt.code;

  let html = `<div class="main">
    <div class="label">${escapeHTML(src.name)} → ${escapeHTML(tgt.name)}${stops.length > 2 ? ` · via ${stops.slice(1, -1).map(langName).join(', ')}` : ''} · ${mode === 'telephone' ? 'Telephone' : 'Express'}</div>
    <div class="answer" lang="${tgt.code}">${escapeHTML(answer)}</div>
    <div class="answer-sub">${phrase
      ? 'Word by word, as nature intended. Grammar sold separately.'
      : `<span class="src" lang="${src.code}">“${escapeHTML(src.words[first.steps[0].idx])}”</span> ≈ ${escapeHTML(first.word)} · cosine similarity ${first.sim.toFixed(2)}`}</div>
    <div class="meter">
      <div class="label">Confidence</div>
      <div class="meter-bar"><span style="width:0%"></span></div>
      <div><span class="meter-quip">${escapeHTML(sameLang && !phrase && mode === 'express' ? Q.sameLanguageQuip(src) : Q.confidence(meanSim))}</span> <span class="meter-num">(${meanSim.toFixed(2)})</span></div>
    </div>
  </div>`;

  if (phrase) {
    html += `<div><div class="label">Word by word</div><ol class="alts">${plans.map((p, i) => p
      ? `<li><span class="word">${escapeHTML(tokens[i])} → <span lang="${tgt.code}">${escapeHTML(p.word)}</span></span><span class="bar"><span style="width:${Math.max(0, p.sim) * 100}%"></span></span><span class="num">${p.sim.toFixed(2)}</span></li>`
      : `<li><span class="word">${escapeHTML(tokens[i])} → ?</span><span class="bar"></span><span class="num">—</span></li>`).join('')}</ol></div>`;
  } else {
    html += `<div><div class="label">Runners-up</div><ol class="alts" lang="${tgt.code}">${first.top.map(t =>
      `<li><span class="word">${escapeHTML(t.word)}</span><span class="bar"><span style="width:${Math.max(0, t.sim) * 100}%"></span></span><span class="num">${t.sim.toFixed(2)}</span></li>`).join('')}</ol></div>`;
  }

  if (!phrase && first.chain.length > 2) {
    html += `<div class="chain">${first.chain.map((c, i) =>
      `${i ? '<span class="sep">→</span>' : ''}<span class="hop${c.passing ? ' passing' : ''}" title="${c.passing ? 'passed through, not snapped' : ''}"><span class="lang">${c.lang.toUpperCase()}</span><span class="word" lang="${c.lang}">${escapeHTML(c.word)}</span>${i ? `<span class="sim">${c.sim.toFixed(2)}</span>` : ''}</span>`).join('')}</div>`;
  }

  let note = '';
  if (!phrase && stops.length > 2 && mode === 'express') note = Q.expressDetourNote([...new Set(stops.slice(1, -1).map(langName))]);
  else if (!phrase && stops.length > 2 && mode === 'telephone' && sameLang) note = Q.tourVerdict(first.chain[0].word, first.word, true);
  html += `<div class="result-actions">
    <button type="button" class="ghost small" data-act="back">Translate it back</button>
    <button type="button" class="ghost small" data-act="link">Copy link</button>
    ${note ? `<span class="note">${escapeHTML(note)}</span>` : ''}
  </div>`;

  el.result.innerHTML = html;
  el.result.hidden = false;
  requestAnimationFrame(() => {
    const bar = el.result.querySelector('.meter-bar span');
    if (bar) bar.style.width = `${Math.max(3, Math.min(1, meanSim) * 100)}%`;
  });
  el.result.querySelector('[data-act="back"]').onclick = () => {
    [state.src, state.tgt] = [state.tgt, state.src];
    state.detours.reverse();
    el.word.value = answer.replace(/\[[^\]]*\]/g, '').trim() || answer;
    syncControls();
    translate();
  };
  el.result.querySelector('[data-act="link"]').onclick = async e => {
    try {
      await navigator.clipboard.writeText(location.href);
      e.target.textContent = 'Copied!';
    } catch {
      e.target.textContent = 'Copy the address bar instead';
    }
  };
}

function showNotFound(word, lang) {
  stage.clearHighlights();
  caption(`Unknown word: ${w(word)}`, 'The matrices refuse to multiply what they cannot look up.');
  const sugg = suggestions(lang, word);
  el.result.innerHTML = `<div class="error">
    <div class="big">${escapeHTML(Q.notFound(word, lang))}</div>
    ${sugg.length ? `<div class="did-you-mean">Did you mean ${sugg.map(s => `<button type="button" lang="${lang.code}">${escapeHTML(s)}</button>`).join('')}</div>` : ''}
  </div>`;
  el.result.hidden = false;
  el.result.querySelectorAll('.did-you-mean button').forEach(b => {
    b.onclick = () => { el.word.value = b.textContent; translate(); };
  });
}

function showError(e) {
  caption('Something broke.', 'Not the linear algebra, probably.');
  const local = location.protocol === 'file:';
  el.result.innerHTML = `<div class="error"><div class="big">Couldn’t load the word vectors.</div>
    <p class="answer-sub">${local
      ? 'Browsers won’t let a page loaded straight from disk fetch its data files. Serve the folder instead, e.g. <code>python3 -m http.server</code>, and open <code>http://localhost:8000</code>.'
      : escapeHTML(String(e.message || e))}</p></div>`;
  el.result.hidden = false;
}

/** The middle "Translator" in the title becomes whatever the engine thinks "translator" is. */
function swapTitle(L) {
  const span = el.title.children[1];
  const en = L[HUB];
  const target = L[state.tgt] && state.tgt !== HUB ? L[state.tgt] : null;
  const i = en.index.get('translator');
  let word = 'Translator';
  if (target && i !== undefined) {
    word = planWord({ [HUB]: en, [target.code]: target }, [HUB, target.code], 'express', i).word;
    word = isCJK(target.code) ? word : word.charAt(0).toUpperCase() + word.slice(1);
  }
  if (span.textContent === word) return;
  span.classList.add('swapping');
  setTimeout(() => {
    span.textContent = word;
    span.lang = target ? target.code : 'en';
    span.title = target ? `“translator” in ${target.name}, according to this very translator` : '';
    span.classList.remove('swapping');
  }, 400);
}

// ------------------------------------------------------------------ controls

function syncControls() {
  el.src.value = state.src;
  el.tgt.value = state.tgt;
  el.mode.querySelectorAll('button').forEach(b => b.setAttribute('aria-checked', String(b.dataset.mode === state.mode)));
  el.hint.textContent = Q.modeHints[state.mode];
  const stops = [state.src, ...state.detours, state.tgt];
  el.route.innerHTML = stops.map((c, i) => {
    const end = i === 0 || i === stops.length - 1;
    return `${i ? '<span class="arrow">→</span>' : ''}<span class="stop${end ? ' end' : ''}" title="${langName(c)}">${c.toUpperCase()}${end ? '' : `<button type="button" aria-label="Remove ${langName(c)} detour" data-i="${i - 1}">×</button>`}</span>`;
  }).join('');
  el.route.querySelectorAll('button').forEach(b => {
    b.onclick = () => { state.detours.splice(Number(b.dataset.i), 1); syncControls(); };
  });
  el.detour.disabled = state.detours.length >= MAX_DETOURS;
  el.word.lang = state.src;
  prefetch();
}

function setSpeed(v) {
  clock.speed = v;
  el.speed.querySelectorAll('button').forEach(b => b.setAttribute('aria-checked', String(Number(b.dataset.speed) === v)));
  store.set('ttt-speed', String(v));
}

function prefetch() {
  for (const c of languagesFor([state.src, state.tgt])) loadLanguage(c).catch(() => {});
}

function writeHash(text) {
  const p = new URLSearchParams({ from: state.src, to: state.tgt });
  if (state.detours.length) p.set('via', state.detours.join(','));
  if (state.mode !== 'express') p.set('mode', state.mode);
  p.set('q', text);
  history.replaceState(null, '', `#${p}`);
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const codes = new Set(state.manifest.languages.map(l => l.code));
  if (codes.has(p.get('from'))) state.src = p.get('from');
  if (codes.has(p.get('to'))) state.tgt = p.get('to');
  state.detours = (p.get('via') || '').split(',').filter(c => codes.has(c)).slice(0, MAX_DETOURS);
  if (p.get('mode') === 'telephone') state.mode = 'telephone';
  if (p.get('q')) el.word.value = p.get('q');
  return !!p.get('q');
}

let suggestIndex = -1;
function hideSuggest() {
  el.suggest.hidden = true;
  suggestIndex = -1;
  el.word.removeAttribute('aria-activedescendant');
}

function renderSuggest() {
  if (!isLoaded(state.src)) return hideSuggest();
  loadLanguage(state.src).then(lang => {
    const items = completions(lang, el.word.value, 8);
    if (!items.length || (items.length === 1 && items[0] === el.word.value.trim())) return hideSuggest();
    el.suggest.innerHTML = items.map((s, i) => `<li id="sugg-${i}" role="option" aria-selected="false" lang="${lang.code}">${escapeHTML(s)}</li>`).join('');
    el.suggest.hidden = false;
    suggestIndex = -1;
    el.suggest.querySelectorAll('li').forEach(li => {
      li.onmousedown = e => {
        e.preventDefault();
        el.word.value = li.textContent;
        hideSuggest();
        translate();
      };
    });
  });
}

function moveSuggest(d) {
  const lis = [...el.suggest.querySelectorAll('li')];
  if (el.suggest.hidden || !lis.length) return false;
  // cycle through -1 (back to what was typed), 0 … n-1
  const n = lis.length;
  suggestIndex = ((suggestIndex + 1 + d + n + 1) % (n + 1)) - 1;
  lis.forEach((li, i) => li.setAttribute('aria-selected', String(i === suggestIndex)));
  if (suggestIndex >= 0) {
    el.word.setAttribute('aria-activedescendant', lis[suggestIndex].id);
    lis[suggestIndex].scrollIntoView({ block: 'nearest' });
  }
  return true;
}

function randomOther(exclude) {
  const pool = state.manifest.languages.map(l => l.code).filter(c => !exclude.includes(c));
  return pool[Math.floor(Math.random() * pool.length)];
}

function bind() {
  el.src.onchange = () => { state.src = el.src.value; syncControls(); };
  el.tgt.onchange = () => { state.tgt = el.tgt.value; syncControls(); };
  el.swap.onclick = () => {
    [state.src, state.tgt] = [state.tgt, state.src];
    state.detours.reverse();
    syncControls();
  };
  el.form.onsubmit = e => {
    e.preventDefault();
    const lis = el.suggest.querySelectorAll('li');
    if (!el.suggest.hidden && suggestIndex >= 0 && lis[suggestIndex]) el.word.value = lis[suggestIndex].textContent;
    translate();
  };
  el.word.oninput = renderSuggest;
  el.word.onkeydown = e => {
    if (e.key === 'ArrowDown' && moveSuggest(1)) e.preventDefault();
    else if (e.key === 'ArrowUp' && moveSuggest(-1)) e.preventDefault();
    else if (e.key === 'Escape') hideSuggest();
  };
  el.word.onblur = () => setTimeout(hideSuggest, 120);
  el.random.onclick = async () => {
    const lang = await loadLanguage(state.src);
    let word;
    for (let tries = 0; tries < 50; tries++) {
      word = lang.words[150 + Math.floor(Math.random() * Math.min(4000, lang.V - 150))];
      if (word.length >= (isCJK(lang.code) ? 2 : 4)) break;
    }
    el.word.value = word;
    translate();
  };
  el.mode.querySelectorAll('button').forEach(b => {
    b.onclick = () => { state.mode = b.dataset.mode; syncControls(); };
  });
  el.speed.querySelectorAll('button').forEach(b => {
    b.onclick = () => setSpeed(Number(b.dataset.speed));
  });
  el.detour.onclick = () => {
    const prev = state.detours.length ? state.detours[state.detours.length - 1] : state.src;
    state.detours.push(randomOther([prev, state.tgt]));
    syncControls();
  };
  el.tour.onclick = () => {
    const others = state.manifest.languages.map(l => l.code).filter(c => c !== state.src);
    for (let i = others.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [others[i], others[j]] = [others[j], others[i]];
    }
    state.detours = others;
    state.tgt = state.src;
    state.mode = 'telephone';
    if (clock.speed && clock.speed < 4) setSpeed(4);
    if (!el.word.value.trim()) el.word.value = { en: 'cat', no: 'katt', ja: '猫', es: 'gato', de: 'Katze', fr: 'chat' }[state.src] ?? '';
    syncControls();
    translate();
  };
}

async function init() {
  try {
    state.manifest = await loadManifest();
  } catch (e) {
    showError(e);
    return;
  }
  const opts = state.manifest.languages
    .map(l => `<option value="${l.code}">${l.native === l.name ? l.name : `${l.native} · ${l.name}`}</option>`).join('');
  el.src.innerHTML = opts;
  el.tgt.innerHTML = opts;

  el.stats.innerHTML = state.manifest.languages.map(l => `<tr>
    <td>${escapeHTML(l.name)} <span class="loading-note" lang="${l.code}">${escapeHTML(l.native === l.name ? '' : l.native)}</span></td>
    <td class="num">${l.words.toLocaleString('en-US')}</td>
    <td class="num">${l.code === HUB ? '— (it’s the hub)' : l.pairs.toLocaleString('en-US')}</td>
    <td class="num">${l.code === HUB ? '—' : `<span class="pbar"><span style="width:${l.p1 * 100}%"></span></span>${Math.round(l.p1 * 100)}%`}</td>
  </tr>`).join('');

  const saved = store.get('ttt-speed');
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  setSpeed(saved !== null && [0, 1, 2, 4].includes(Number(saved)) ? Number(saved) : reduced ? 0 : 1);

  const autorun = readHash();
  if (!el.word.value) el.word.placeholder = 'Type a word… try “cat”, “beer” or “penguin”';
  bind();
  syncControls();
  document.fonts?.load(`16px "CMU Serif"`).then(() => { stage.fontsChanged(); math.dirty = true; });
  if (autorun) translate();
}

init();
