/* RobotAgentBench — site interactions */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches || /[?&]static\b/.test(location.search);

  if (/[?&]hide=hero\b/.test(location.search)) document.querySelector('.hero').style.display = 'none'; // screenshot helper
  /* ---------------- NAV ---------------- */
  const nav = $('#nav');
  const onScroll = () => nav.classList.toggle('scrolled', scrollY > 30);
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  $('#navToggle').addEventListener('click', () => $('#navLinks').classList.toggle('open'));
  $$('#navLinks a').forEach(a => a.addEventListener('click', () => $('#navLinks').classList.remove('open')));
  const secIO = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    $$('#navLinks a').forEach(a => a.classList.toggle('is-active', a.getAttribute('href') === '#' + e.target.id));
  }), { rootMargin: '-45% 0px -50% 0px' });
  ['demo', 'paper', 'capabilities', 'task-preview', 'environment', 'leaderboard', 'findings', 'citation'].forEach(id => { const el = document.getElementById(id); el && secIO.observe(el); });

  // Placeholder links (data-todo) — keep the page from jumping to top.
  $$('a[data-todo]').forEach(a => a.addEventListener('click', e => { if (a.getAttribute('href') === '#') { e.preventDefault(); toast('Link coming soon'); } }));
  function toast(msg) {
    let t = $('#toast');
    if (!t) { t = document.createElement('div'); t.id = 'toast'; Object.assign(t.style, { position: 'fixed', left: '50%', bottom: '28px', transform: 'translateX(-50%)', background: '#1B1129', color: '#fff', padding: '12px 18px', borderRadius: '999px', font: '600 14px Inter,sans-serif', zIndex: 200, boxShadow: '0 12px 40px -12px rgba(0,0,0,.5)', transition: 'opacity .3s' }); document.body.appendChild(t); }
    t.textContent = msg; t.style.opacity = 1; clearTimeout(t._h); t._h = setTimeout(() => t.style.opacity = 0, 1600);
  }

  /* ---------------- DEMO ---------------- */
  // Plays with sound by default. Browsers block autoplay with sound until the visitor has interacted with the
  // page, so in that case it plays muted and turns the sound on at the visitor's first click, tap or key press.
  const demo = $('#demoVideo'), sound = $('#demoSound');
  let autoMuted = false, unmutedAt = 0;
  const syncSound = () => { sound.hidden = !demo.muted; sound.classList.toggle('demo-tool-cta', autoMuted && demo.muted); };
  demo.addEventListener('volumechange', () => { if (!demo.muted) autoMuted = false; syncSound(); });
  const unmute = () => { demo.muted = false; if (demo.volume === 0) demo.volume = 1; unmutedAt = performance.now(); demo.play().catch(() => {}); };
  sound.addEventListener('click', unmute);
  $('#demoRestart').addEventListener('click', () => { demo.currentTime = 0; demo.play().catch(() => {}); });
  // These events count as a user gesture, so unmuting here is allowed (pointerdown is not one on touch screens).
  const GESTURES = ['click', 'keydown', 'touchend'];
  const firstGesture = () => { GESTURES.forEach(t => document.removeEventListener(t, firstGesture, true)); if (autoMuted && demo.muted) unmute(); };
  // If that first click lands on the video, the player's own click-to-pause would stop it: keep it playing.
  demo.addEventListener('pause', () => { if (performance.now() - unmutedAt < 400 && !demo.ended && !pausedOffscreen) demo.play().catch(() => {}); });
  const startDemo = () => {
    demo.muted = false;
    demo.play().catch(() => {
      autoMuted = true; demo.muted = true; syncSound();
      demo.play().catch(() => {});
      GESTURES.forEach(t => document.addEventListener(t, firstGesture, true));
    });
  };
  // Pause the demo while it is scrolled out of view (so it stops buffering) and resume when it comes back.
  let pausedOffscreen = false, demoVisible = true;
  new IntersectionObserver(([e]) => {
    demoVisible = e.isIntersecting;
    if (!e.isIntersecting && !demo.paused) { pausedOffscreen = true; demo.pause(); }
    else if (e.isIntersecting && pausedOffscreen) { pausedOffscreen = false; if (demo.currentTime === 0) startDemo(); else demo.play().catch(() => {}); }
  }, { threshold: 0.15 }).observe(demo);
  // Hold on the title-card poster (14 s frame) briefly, then start from the beginning.
  if (!reduce) setTimeout(() => { if (demo.paused && demo.currentTime === 0) { if (demoVisible) startDemo(); else pausedOffscreen = true; } }, 1500);
  syncSound();


  /* ---------------- LAZY LOOP VIDEOS ---------------- */
  const vIO = new IntersectionObserver(es => es.forEach(e => {
    const v = e.target;
    if (e.isIntersecting) {
      if (!v.src && v.dataset.src) v.src = v.dataset.src;
      if (!reduce) v.play().catch(() => {});
    } else v.pause();
  }), { rootMargin: '120px 0px' });
  $$('video[data-src]:not([data-early])').forEach(v => vIO.observe(v));
  // Clips marked data-early start downloading ~1.5 screens before they scroll into view, so they are already playing on arrival.
  const earlyIO = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const v = e.target; earlyIO.unobserve(v);
    if (!v.src) { v.preload = 'auto'; v.src = v.dataset.src; }
    vIO.observe(v);
  }), { rootMargin: '150% 0px' });
  $$('video[data-early]').forEach(v => earlyIO.observe(v));
  const bg = $('.hero-bg-video'); if (reduce && bg) bg.pause();

  /* ---------------- CAPABILITY PLAYER (demo footage) ---------------- */
  (() => {
    const root = $('#capx'); if (!root) return;
    const items = $$('.cap-item', root), vids = $$('.cap-vid', root), tag = $('#capxTag');
    let cur = 0, inView = false;
    const load = v => { if (!v.src) v.src = v.dataset.capSrc; };
    function show(i, play = true) {
      cur = i;
      items.forEach((b, k) => { b.classList.toggle('is-active', k === i); $('.cap-prog i', b).style.width = '0'; });
      vids.forEach((v, k) => { v.classList.toggle('is-active', k === i); if (k !== i) v.pause(); });
      tag.textContent = `Capability test 0${i + 1} / 0${vids.length}`;
      const v = vids[i]; load(v); v.currentTime = 0;
      if (play && inView && !reduce) v.play().catch(() => {});
      load(vids[(i + 1) % vids.length]);
    }
    vids.forEach((v, k) => {
      v.addEventListener('timeupdate', () => { if (k === cur && v.duration) $('.cap-prog i', items[k]).style.width = (v.currentTime / v.duration * 100) + '%'; });
      v.addEventListener('ended', () => { if (k === cur) show((cur + 1) % vids.length); });
    });
    items.forEach((b, k) => b.addEventListener('click', () => show(k)));
    new IntersectionObserver(es => {
      inView = es[0].isIntersecting;
      const v = vids[cur]; load(v);
      if (inView && !reduce) v.play().catch(() => {}); else v.pause();
    }, { threshold: .35 }).observe(root);
    show(0, false);
  })();

  /* ---------------- STAT COUNTERS ---------------- */
  const cIO = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return; cIO.unobserve(e.target);
    const el = e.target, end = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0);
    if (reduce) return;
    const t0 = performance.now(), dur = 1300;
    const step = t => { const k = Math.min(1, (t - t0) / dur), v = end * (1 - Math.pow(1 - k, 3)); el.textContent = v.toFixed(dec); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }), { threshold: .6 });
  $$('[data-count]').forEach(el => cIO.observe(el));

  /* ---------------- TABS ---------------- */
  $$('[data-tabs]').forEach(group => {
    const panels = document.getElementById(group.dataset.tabs);
    $$('.tab', group).forEach(btn => btn.addEventListener('click', () => {
      $$('.tab', group).forEach(b => b.classList.toggle('is-active', b === btn));
      $$('.fig-card', panels).forEach(p => { p.classList.toggle('is-active', p.id === btn.dataset.target); p.classList.add('in'); });
    }));
  });

  /* ---------------- LIGHTBOX ---------------- */
  const lb = $('#lightbox'), lbIn = $('#lightboxInner');
  const openLB = html => { lbIn.innerHTML = ''; lbIn.append(html); lb.classList.add('open'); lb.setAttribute('aria-hidden', 'false'); document.body.style.overflow = 'hidden'; lbIn.scrollTop = 0; };
  const closeLB = () => { lb.classList.remove('open'); lb.setAttribute('aria-hidden', 'true'); document.body.style.overflow = ''; };
  $('#lbClose').addEventListener('click', closeLB);
  lb.addEventListener('click', e => { if (e.target === lb) closeLB(); });
  addEventListener('keydown', e => { if (e.key === 'Escape') closeLB(); });
  document.addEventListener('click', e => {
    const img = e.target.closest('img[data-zoom]'); if (!img) return;
    const big = new Image(); big.src = img.dataset.full || img.currentSrc || img.src;  // full-resolution file for zoom big.alt = img.alt; openLB(big);
  });

  /* ---------------- TASK DATA ---------------- */
  const TASKS = (window.RAB_TASKS || []).map((t, i) => ({ ...t, id: i }));
  const CAT = {
    'Short': { label: 'Short-horizon', c: 'var(--c-mani)' },
    'Long': { label: 'Long-horizon', c: '#3A4BD8' },
    'Task Discovery': { label: 'Task discovery', c: 'var(--c-disc)' },
    'Context Reasoning': { label: 'Context reasoning', c: 'var(--c-reas)' },
    'Failure Recovery': { label: 'Failure recovery', c: 'var(--c-reco)' },
    'Robustness': { label: 'Robustness', c: 'var(--c-robu)' },
    'Open-ended Scenarios': { label: 'Open-ended', c: 'var(--c-open)' },
  };
  const keyOf = t => t.cat === 'Manipulation' ? (t.sub === 'Long tasks' ? 'Long' : 'Short') : t.cat;
  TASKS.forEach(t => t.key = keyOf(t));
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function openTask(t) {
    const m = CAT[t.key];
    const wrap = document.createElement('div'); wrap.className = 'task-modal';
    const fields = Object.entries(t.fields).map(([k, v]) => `<div><h6>${esc(k)}</h6><p>${esc(v)}</p></div>`).join('');
    wrap.innerHTML = `<div class="task-modal-head"><span class="task-cat" style="--c:${m.c}">${esc(m.label)}</span><h3>${esc(t.title)}</h3></div>
      <div class="task-modal-fields">${fields}</div>
      ${t.img ? `<img src="${t.img}" alt="${esc(t.caption)}" width="${t.w}" height="${t.h}">` : ''}
      ${t.caption ? `<p class="task-modal-cap">${esc(t.caption)}</p>` : ''}`;
    openLB(wrap);
  }

  /* ---------------- SCENARIOS (chat demo) ---------------- */
  const ICONS = {
    dinner: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 3v8a3 3 0 0 0 6 0V3M7 3v18M17 3c-2 0-3 2.5-3 6s1 4 3 4v8"/></svg>',
    guests: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.8-3.6 3.4-5.5 6.5-5.5s5.7 1.9 6.5 5.5"/><circle cx="17" cy="9" r="2.6"/><path d="M16.5 14.5c2.6.2 4.4 2 5 5"/></svg>',
    leave: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="7" width="18" height="13" rx="2.5"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 12h18"/></svg>',
    tidy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 3 9.5 12M6 12h8l2 9H4z"/><path d="M8 16v5M12 16v5"/></svg>',
  };
  const SCEN = [
    { k: 'guests', name: 'Host guests', tasks: 8, title: 'Host guests',
      user: 'Alex is coming over shortly. Please get some refreshments ready and make the place ready for his visit.',
      memory: 'Alex usually has Sprite and Oreos.',
      robot: 'Let me look around first… Found 8 to-dos across the shelf, sofa and coffee table.',
      todo: ['Serve Sprite on wooden tray', 'Serve Oreos on wooden tray', 'Book to side table', 'Cap to coat rack', 'Remote to TV cabinet', 'Wallet into storage basket', 'Recycle used can', 'Dispose of used cup'] },
    { k: 'dinner', name: 'Prepare dinner', tasks: 10, title: 'Prepare dinner',
      user: 'Ryan is joining us for dinner. He’s been working out a lot lately. Please get everything ready for him.',
      memory: 'Ryan’s usual dinner includes a warmed beef meal, a fork and knife, and freshly prepared coffee.',
      robot: 'Got it. The meal needs heating before serving — 10 to-dos, in order.',
      todo: ['Prepare coffee', 'Place coffee on meal tray', 'Move beef meal to microwave', 'Heat meal; place on meal tray', 'Add fork to meal tray', 'Add knife to meal tray', 'Serve meal tray at coffee table', 'Remote to TV cabinet', 'Book to side table', 'Dispose of used cup'] },
    { k: 'leave', name: 'Prepare to leave', tasks: 8, title: 'Prepare to leave',
      user: 'Emma is going out to work away from home today, and it’s raining. Could you help her get ready?',
      memory: 'Emma needs her wallet and power bank when working away from home. The water bottle and glasses case are usually kept in the open bag.',
      robot: 'Packing the bag first, then tidying up — 8 to-dos.',
      todo: ['Pack wallet in open bag', 'Pack power bank in open bag', 'Pack water bottle in open bag', 'Pack glasses case in open bag', 'Remote to TV cabinet', 'Notebook into drawer; close it', 'Dispose of paper waste', 'Dispose of snack wrapper'] },
    { k: 'tidy', name: 'Tidy the room', tasks: 8, title: 'Tidy the room',
      user: 'I’m about to start a video call from the sofa. Could you tidy up the room first?',
      memory: 'Household memory specifies where each belonging is stored and how waste should be sorted.',
      robot: 'Scanning the sofa and coffee table… 8 items are out of place.',
      todo: ['Book to side table', 'Cap to coat rack', 'Remote to TV cabinet', 'Notebook into drawer; close it', 'Wallet into storage basket', 'Glasses case into open bag', 'Dispose of used cup', 'Recycle used can'] },
  ];
  // Scenario cards (each opens its full scenario map from the appendix)
  const scenCards = $('#scenCards');
  scenCards.innerHTML = SCEN.map(s => `<div class="scen-card">
      <span class="scen-card-top"><span class="ico">${ICONS[s.k]}</span><b>${s.name}</b></span>
      <q>${esc(s.user)}</q></div>`).join('');

  // Scenario video: lazy-load and autoplay (muted) while in view
  const scenVideo = $('#scenVideo');
  new IntersectionObserver(es => {
    if (es[0].isIntersecting) { if (!scenVideo.src) scenVideo.src = scenVideo.dataset.scenSrc; if (!reduce) scenVideo.play().catch(() => {}); }
    else scenVideo.pause();
  }, { threshold: .4 }).observe(scenVideo);

  /* ---------------- ARCHITECTURES ---------------- */
  const kbox = c => `<i class="k k-${c === '–' ? '0' : c}">${c}</i>`;
  const ARCH = [
    { code: ['–', 'L', 'C'], name: 'Policy + code verifier', d: 'A VLA/WAM policy acts; a coding-agent verifier executes action prefixes and adds local corrections.', avg: 23.3, oe: 3.7 },
    { code: ['C', 'L', '–'], name: 'Code planner + policy', d: 'A coding agent decomposes the request into subtask instructions and delegates execution to a VLA/WAM policy.', avg: 26.4, oe: 9.1 },
    { code: ['C', 'C', 'C'], name: 'Fully code-based', d: 'A coding agent plans, composes predefined control functions such as move_eef_to(), and verifies outcomes.', avg: 39.7, oe: 9.8, note: 'best: Antigravity CLI' },
    { code: ['C', 'H', 'C'], name: 'Hybrid execution', d: 'Code-based control and learned policies exposed as unified tools — e.g. code positions the gripper, the policy grasps.', avg: 48.5, oe: 15.4, best: true },
    { code: ['C', 'C', 'C'], rsi: true, name: 'Code-based + RSI', d: 'Harness-level recursive self-improvement: the agent creates, refines and reuses control tools across attempts.', avg: 42.3, oe: 10.8 },
  ];
  const archsEl = $('#archs'); if (archsEl) archsEl.innerHTML = ARCH.map(a => `<div class="arch${a.best ? ' best' : ''}">
    <div class="arch-code">${a.code.map(kbox).join('<span class="slash">/</span>')}${a.rsi ? '<span class="arch-rsi">+RSI</span>' : ''}</div>
    <h5>${a.name}</h5><p>${a.d}</p>
    <div class="arch-score"><span>Avg<br><b>${a.avg.toFixed(1)}</b></span><span style="text-align:right">Open-ended<br><b>${a.oe.toFixed(1)}</b></span></div></div>`).join('');

  /* ---------------- LEADERBOARD ---------------- */
  // Table 1 — task-level scores and open-ended scenario score.
  const T1 = [
    // standalone VLA
    { n: 'π0.5', t: 'VLA', s: [13.1, 1.2, 16.7, 20.3, 8.1, 11.9, 1.1] },
    { n: 'Xiaomi Robotics-1', t: 'VLA', s: [18.1, 0.5, 15.8, 5.8, 21.6, 12.4, 0.4] },
    { n: 'InternVLA-A1.5', t: 'VLA', s: [11.3, 0.7, 10.8, 23.2, 44.2, 18.0, 1.0] },
    { n: 'MolmoAct2', t: 'VLA', s: [4.3, 1.7, 5.0, 2.7, 2.4, 3.2, 1.9] },
    { n: 'LingBot-VLA2', t: 'VLA', s: [6.5, 0.3, 1.7, 6.6, 10.2, 5.1, 0.5] },
    { n: 'HY-VLA', t: 'VLA', s: [10.6, 1.0, 15.0, 3.3, 7.6, 7.5, 0.8] },
    { n: 'Galaxea G0.5', t: 'VLA', s: [11.0, 1.5, 25.0, 9.3, 8.1, 11.0, 1.3] },
    { n: 'X-VLA', t: 'VLA', s: [9.9, 0.4, 5.0, 6.8, 7.6, 5.9, 0.3] },
    // standalone WAM
    { n: 'τ0', sub: 'low-level', t: 'WAM', s: [7.7, 0.6, 1.7, 3.5, 7.8, 4.3, 0.5] },
    { n: 'OpenWAM-Alpha', t: 'WAM', s: [13.4, 1.2, 15.0, 4.6, 8.6, 8.6, 2.0] },
    { n: 'LingBot-VA', t: 'WAM', s: [5.0, 1.5, 3.3, 5.4, 4.2, 3.9, 0.4] },
    // robot agents
    { n: 'Codex', sub: 'GPT-6 Astra', t: 'agent', a: 'C/C/C', s: [32.4, 35.2, 38.3, 22.0, 27.1, 31.0, 8.4] },
    { n: 'Claude Code', sub: 'Opus 5.5', t: 'agent', a: 'C/C/C', s: [20.6, 24.5, 60.0, 44.2, 17.1, 33.3, 9.2] },
    { n: 'Antigravity CLI', sub: 'Gemini 4 Argon', t: 'agent', a: 'C/C/C', s: [29.2, 36.5, 80.0, 27.8, 25.0, 39.7, 9.8] },
    { n: 'Codex + InternVLA-A1.5', t: 'agent', a: '–/L/C', s: [18.9, 4.3, 23.3, 24.6, 45.4, 23.3, 3.7] },
    { n: 'Codex + InternVLA-A1.5', t: 'agent', a: 'C/L/–', s: [16.5, 4.0, 40.0, 26.8, 44.8, 26.4, 9.1] },
    { n: 'Codex + InternVLA-A1.5', t: 'agent', a: 'C/H/C', s: [42.5, 37.2, 73.3, 38.6, 50.8, 48.5, 15.4] },
    { n: 'Codex', sub: 'GPT-6 Astra · 10 RSI rounds', t: 'agent', a: 'C/C/C', rsi: true, s: [36.8, 33.3, 63.3, 31.4, 46.5, 42.3, 10.8] },
  ];
  // Table 2 — open-ended scenario breakdown + behavioural metrics (stagnation: lower is better).
  const T2 = [
    { n: 'InternVLA-A1.5', t: 'VLA', s: [0.3, 1.8, 1.2, 0.7, 2.3, 4.5, 0.6, 88.3] },
    { n: 'OpenWAM-Alpha', t: 'WAM', s: [0.9, 3.4, 2.1, 1.6, 3.1, 2.7, 0.2, 84.6] },
    { n: 'Codex', t: 'agent', a: 'C/C/C', s: [11.2, 3.6, 1.4, 17.4, 13.4, 15.2, 15.4, 33.7] },
    { n: 'Claude Code', t: 'agent', a: 'C/C/C', s: [6.4, 12.8, 8.1, 9.5, 12.8, 16.2, 18.8, 18.7] },
    { n: 'Antigravity CLI', t: 'agent', a: 'C/C/C', s: [6.0, 13.8, 13.4, 6.0, 15.8, 17.0, 37.5, 18.4] },
    { n: 'Codex', t: 'agent', a: 'C/C/C', rsi: true, s: [13.1, 6.4, 3.0, 20.7, 20.3, 19.6, 38.9, 12.3] },
    { n: 'Codex + InternVLA-A1.5', t: 'agent', a: 'C/H/C', s: [15.8, 14.3, 13.9, 17.6, 15.2, 25.1, 30.7, 12.1] },
  ];
  // Table 3 — sim vs real on matched subsets.
  const T3 = [
    { n: 'InternVLA-A1.5', t: 'VLA', sim: [11.8, 1.2, 12.5, 20.8, 42.7, 17.8, 1.4], real: [10.7, 1.7, 10.0, 15.0, 30.0, 13.5, 1.0] },
    { n: 'OpenWAM-Alpha', t: 'WAM', sim: [14.5, 0.8, 12.5, 5.5, 9.2, 8.5, 2.4], real: [10.0, 0.0, 15.0, 5.3, 10.5, 8.2, 0.0] },
    { n: 'Codex', t: 'agent', a: 'C/C/C', sim: [35.8, 38.4, 45.8, 24.1, 26.8, 34.2, 8.7], real: [33.0, 33.3, 44.2, 25.0, 24.0, 31.9, 6.0] },
  ];

  const COLS1 = ['Manipulation', 'Discovery', 'Reasoning', 'Recovery', 'Robustness', 'Average', 'Open-Ended'];
  const COLS2 = ['Guests', 'Dinner', 'Leaving', 'Tidying', 'Coverage', 'Progress', 'Recovery', 'Stagnation'];
  const tbl = $('#lbTable'), note = $('#lbNote'), filt = $('#lbFilter');
  let view = 'main', filter = 'all', sortCol = 5, sortDir = -1;

  const tagHTML = r => r.t === 'agent' ? `<span class="tag agent">Agent</span>` : `<span class="tag ${r.t}">${r.t}</span>`;
  const archHTML = r => r.a ? `<span class="arch-mini">${r.a.split('/').map(kbox).join('')}</span>${r.rsi ? '<span class="arch-rsi" style="padding:3px 5px;font-size:10px">RSI</span>' : ''}` : '';
  const nameHTML = r => `<span class="name"><b>${esc(r.n)}</b><span class="name-meta">${tagHTML(r)}${archHTML(r)}${r.sub ? `<small>${esc(r.sub)}</small>` : ''}</span></span>`;
  const passes = r => filter === 'all' || (filter === 'agent' ? r.t === 'agent' : r.t === filter);
  const medalsFor = (rows, idx) => {
    // top-3 per group (standalone vs agents), matching the paper's convention
    const m = new Map();
    ['policy', 'agent'].forEach(g => {
      rows.filter(r => (r.t === 'agent') === (g === 'agent')).slice().sort((a, b) => b.s[idx] - a.s[idx]).slice(0, 3).forEach((r, i) => m.set(r, i + 1));
    });
    return m;
  };

  function sortHead(label, i, cls = '') {
    const on = sortCol === i;
    return `<th class="sortable ${cls}${on ? ' sorted' : ''}" data-col="${i}">${label}<span class="arr">${on ? (sortDir < 0 ? '▼' : '▲') : '▼'}</span></th>`;
  }

  function renderMain() {
    const rows = T1.filter(passes);
    const maxes = COLS1.map((_, i) => Math.max(...T1.map(r => r.s[i])));
    const mAvg = medalsFor(T1, 5), mOE = medalsFor(T1, 6);
    rows.sort((a, b) => sortDir * (a.s[sortCol] - b.s[sortCol]));
    const head = `<thead>
      <tr><th class="grp" colspan="2"></th><th class="grp" colspan="6"><span>Task-level</span></th><th class="grp scen"><span>Scenario-level</span></th></tr>
      <tr><th class="l">#</th><th class="l">Model / Agent</th>${COLS1.map((c, i) => sortHead(c, i)).join('')}</tr></thead>`;
    const body = rows.map((r, ri) => `<tr>
      <td class="l rank">${ri + 1}</td><td class="l">${nameHTML(r)}</td>
      ${r.s.map((v, i) => {
        const key = i >= 5, top = v === maxes[i];
        const md = i === 5 ? mAvg.get(r) : i === 6 ? mOE.get(r) : 0;
        return `<td class="${key ? 'key' : ''}"><span class="cell${top ? ' top' : ''}">${v.toFixed(1)}${md ? `<i class="medal m${md}" title="#${md} in group"></i>` : ''}<span class="bar" style="width:${Math.max(4, v / maxes[i] * 100)}%"></span></span></td>`;
      }).join('')}</tr>`).join('');
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    note.innerHTML = 'Average is the mean of the five task-level scores. <i class="medal m1"></i> <i class="medal m2"></i> <i class="medal m3"></i> mark the top three in Average and Open-Ended, ranked separately for standalone policies and robot agents. Click a column to sort.';
  }

  function renderOpen() {
    const rows = T2.filter(passes);
    const lowerBetter = i => i === 7;
    const best = COLS2.map((_, i) => lowerBetter(i) ? Math.min(...T2.map(r => r.s[i])) : Math.max(...T2.map(r => r.s[i])));
    const maxes = COLS2.map((_, i) => Math.max(...T2.map(r => r.s[i])));
    const sc = Math.min(sortCol, 7);
    rows.sort((a, b) => sortDir * (a.s[sc] - b.s[sc]));
    const head = `<thead>
      <tr><th class="grp" colspan="2"></th><th class="grp scen" colspan="4"><span>Scenario progress ↑</span></th><th class="grp" colspan="4"><span>Behavioural metrics</span></th></tr>
      <tr><th class="l">#</th><th class="l">Model / Agent</th>${COLS2.map((c, i) => sortHead(c + (i === 7 ? ' ↓' : i >= 4 ? ' ↑' : ''), i)).join('')}</tr></thead>`;
    const body = rows.map((r, ri) => `<tr><td class="l rank">${ri + 1}</td><td class="l">${nameHTML(r)}</td>
      ${r.s.map((v, i) => `<td class="${v === best[i] ? 'key' : ''}"><span class="cell${v === best[i] ? ' top' : ''}">${v.toFixed(1)}<span class="bar" style="width:${Math.max(4, v / maxes[i] * 100)}%${i === 7 ? ';background:#F2C3CF' : ''}"></span></span></td>`).join('')}</tr>`).join('');
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    note.innerHTML = '<b>Coverage</b>: fraction of scenario tasks identified and initiated · <b>Progress</b>: mean normalized progress on initiated tasks · <b>Recovery</b>: fraction of execution failures recovered · <b>Stagnation</b>: fraction of time spent in repetitive or no-progress states (lower is better).';
  }

  function renderS2R() {
    const rows = T3.filter(passes);
    const head = `<thead>
      <tr><th class="grp" colspan="2"></th><th class="grp" colspan="6"><span>Task-level</span></th><th class="grp scen"><span>Scenario-level</span></th></tr>
      <tr><th class="l">Model / Agent</th><th class="l">Env</th>${COLS1.map(c => `<th>${c}</th>`).join('')}</tr></thead>`;
    const body = rows.map(r => {
      const simRow = `<tr><td class="l" rowspan="2">${nameHTML(r)}</td><td class="l"><span class="env-tag">Sim</span></td>${r.sim.map((v, i) => `<td class="${i >= 5 ? 'key' : ''}">${v.toFixed(1)}</td>`).join('')}</tr>`;
      const realRow = `<tr><td class="l"><span class="env-tag real">Real</span></td>${r.real.map((v, i) => {
        const d = v - r.sim[i];
        return `<td class="${i >= 5 ? 'key' : ''}">${v.toFixed(1)}<span class="delta ${d >= 0 ? 'up' : 'dn'}">${d >= 0 ? '▲' : '▼'}</span></td>`;
      }).join('')}</tr>`;
      return simRow + realRow;
    }).join('');
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    note.innerHTML = 'Matched evaluation subsets in simulation and in the physical home (AgiBot G1), with equal numbers of training demonstrations. <span class="delta up">▲</span> / <span class="delta dn">▼</span> show change relative to simulation. Absolute scores drop in the real world, but rankings are preserved.';
  }

  function renderLB() {
    filt.classList.toggle('disabled', view === 's2r');
    if (view === 'main') renderMain(); else if (view === 'open') renderOpen(); else renderS2R();
  }
  tbl.addEventListener('click', e => {
    const th = e.target.closest('th.sortable'); if (!th) return;
    const c = +th.dataset.col;
    if (sortCol === c) sortDir *= -1; else { sortCol = c; sortDir = (view === 'open' && c === 7) ? 1 : -1; }
    renderLB();
  });
  $('#lbView') && $('#lbView').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    view = b.dataset.view;
    $$('#lbView button').forEach(x => x.classList.toggle('is-active', x === b));
    if (view === 'main') { sortCol = 5; sortDir = -1; }
    if (view === 'open') { sortCol = 5; sortDir = -1; }
    if (view === 's2r') { filter = 'all'; $$('#lbFilter button').forEach(x => x.classList.toggle('is-active', x.dataset.f === 'all')); }
    renderLB();
  });
  filt.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    filter = b.dataset.f;
    $$('#lbFilter button').forEach(x => x.classList.toggle('is-active', x === b));
    renderLB();
  });
  renderLB();

  /* ---------------- CITATION ---------------- */
  $('#bibCopy').addEventListener('click', async () => {
    const txt = $('#bibText').textContent;
    try { await navigator.clipboard.writeText(txt); } catch { const r = document.createRange(); r.selectNodeContents($('#bibText')); getSelection().removeAllRanges(); getSelection().addRange(r); document.execCommand('copy'); }
    $('#bibCopy').textContent = 'Copied ✓'; setTimeout(() => $('#bibCopy').textContent = 'Copy', 1500);
  });

  /* ---------------- REVEAL ---------------- */
  const rIO = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); rIO.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
  if (!reduce) $$('.sec-head, .cap, .fig-card, .finding, .scenario-block, .env-card, .arch, .lb, .takeaways, .abstract, .pev-node').forEach(el => { el.classList.add('reveal'); rIO.observe(el); });
})();
