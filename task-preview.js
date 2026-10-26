(() => {
  'use strict';
  const data = window.RAB_TASK_PREVIEWS;
  const section = document.querySelector('#task-preview');
  if (!data || !section) return;
  const isDraft = new URLSearchParams(location.search).get('task-preview') === '1';
  if (!data.published && !isDraft) return;
  const tasks = data.tasks.filter(t => t.src && t.galleryEligible);
  if (!tasks.length) return;
  section.hidden = false;
  document.querySelectorAll('[data-task-preview-link]').forEach(a => { a.hidden = false; });
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const saveData = navigator.connection?.saveData;
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const duration = n => `${Math.floor(n / 60)}:${String(Math.round(n % 60)).padStart(2, '0')}`;
  const strips = section.querySelector('.tp-strips');
  section.querySelector('[data-tp-count]').textContent = `${tasks.length} task previews`;
  const rows = [];
  const visible = new Set();
  let sectionVisible = false, paused = reduced.matches, dialogOpen = false, lastTime = 0;
  const syncVideos = () => {
    let playing = 0;
    for (const video of section.querySelectorAll('video')) {
      const card = video.closest('.tp-card');
      if (!saveData && !reduced.matches && !document.hidden && sectionVisible && !dialogOpen && visible.has(card) && playing < 4) {
        playing++;
        if (!video.src) { video.src = video.dataset.src; video.load(); }
        video.play().catch(() => {});
      } else video.pause();
    }
  };
  const cardObserver = new IntersectionObserver(entries => {
    entries.forEach(e => e.isIntersecting && e.intersectionRatio > .2 ? visible.add(e.target) : visible.delete(e.target));
    syncVideos();
  }, { threshold: [0, .2, .5] });
  [0, 1].forEach(rowIndex => {
    const items = tasks.filter((_, i) => i % 2 === rowIndex);
    const row = document.createElement('div'); row.className = 'tp-row';
    row.setAttribute('aria-label', `Task previews, row ${rowIndex + 1}`);
    let hovering = false, interactingUntil = 0;
    const rowState = { row, hovering: () => hovering, interactingUntil: () => interactingUntil, speed: rowIndex ? 17 : 21, remainder: 0 };
    for (let copy = 0; copy < 2; copy++) {
      items.forEach(t => {
        const button = document.createElement('button'); button.type = 'button'; button.className = 'tp-card';
        button.setAttribute('aria-label', `Watch ${t.title}`);
        if (copy) { button.tabIndex = -1; button.setAttribute('aria-hidden', 'true'); }
        button.innerHTML = `<span class="tp-media"><img src="${esc(t.poster)}" alt="" loading="lazy"><video data-src="${esc(t.thumb || t.src)}" muted loop playsinline preload="none" aria-hidden="true" tabindex="-1"></video><span class="tp-play" aria-hidden="true">▶</span></span><span class="tp-card-info"><span class="tp-card-meta"><span class="tp-card-id">Task-Level Evaluation</span><span>${duration(t.duration)}</span></span><span class="tp-card-title">${esc(t.title)}</span></span>`;
        const video = button.querySelector('video'); video.muted = true;
        // Cards play a small 8 s clip (thumb, cut from the same point); the dialog plays the full video.
        if (!t.thumb) video.addEventListener('loadedmetadata', () => { video.currentTime = Math.min(t.duration * .16, 12); });
        video.addEventListener('playing', () => video.classList.add('tp-playing'));
        video.addEventListener('error', () => video.classList.remove('tp-playing'));
        button.addEventListener('click', () => showTask(tasks.indexOf(t), button));
        row.append(button); cardObserver.observe(button);
      });
    }
    row.addEventListener('mouseenter', () => { hovering = true; });
    row.addEventListener('mouseleave', () => { hovering = false; });
    row.addEventListener('focusin', () => { hovering = true; });
    row.addEventListener('focusout', e => { if (!row.contains(e.relatedTarget)) hovering = false; });
    row.addEventListener('pointerdown', () => { interactingUntil = performance.now() + 5000; });
    row.addEventListener('wheel', () => { interactingUntil = performance.now() + 4000; }, { passive: true });
    row.addEventListener('scroll', syncVideos, { passive: true });
    strips.append(row); rows.push(rowState);
  });
  new IntersectionObserver(entries => { sectionVisible = entries[0].isIntersecting; syncVideos(); }).observe(section);
  const pauseButton = section.querySelector('[data-tp-pause]');
  const updatePause = () => { pauseButton.setAttribute('aria-pressed', String(paused)); pauseButton.textContent = paused ? '▶ Resume scrolling' : 'Ⅱ Pause scrolling'; };
  pauseButton.addEventListener('click', () => { paused = !paused; updatePause(); }); updatePause();
  reduced.addEventListener('change', () => { paused = reduced.matches; updatePause(); syncVideos(); });
  section.querySelectorAll('[data-tp-scroll]').forEach(button => button.addEventListener('click', () => {
    const direction = Number(button.dataset.tpScroll);
    for (const state of rows) state.row.scrollBy({ left: direction * 356, behavior: reduced.matches ? 'instant' : 'smooth' });
    paused = true; updatePause();
  }));
  const animate = now => {
    const dt = lastTime ? Math.min((now - lastTime) / 1000, .06) : 0; lastTime = now;
    if (!paused && !dialogOpen && !document.hidden && sectionVisible) for (const state of rows) {
      if (state.hovering() || now < state.interactingUntil()) continue;
      const cards = state.row.querySelectorAll('.tp-card');
      const halfway = cards[cards.length / 2].offsetLeft - cards[0].offsetLeft;
      state.remainder += dt * state.speed;
      const pixels = Math.floor(state.remainder);
      state.remainder -= pixels;
      state.row.scrollLeft += pixels;
      if (state.row.scrollLeft >= halfway) state.row.scrollLeft -= halfway;
    }
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
  document.addEventListener('visibilitychange', syncVideos);

  const dialog = document.createElement('dialog'); dialog.className = 'tp-dialog';
  dialog.setAttribute('aria-labelledby', 'tpDialogTitle');
  dialog.innerHTML = '<div class="tp-dialog-head"><div><span class="tp-eyebrow" id="tpDialogIndex"></span><h3 id="tpDialogTitle"></h3></div><button class="tp-icon" type="button" aria-label="Close video">×</button></div><video controls playsinline preload="metadata"></video><div class="tp-dialog-bottom"><div><select class="tp-picker" aria-label="Choose a task"></select><p id="tpDialogRequest"></p></div><div class="tp-dialog-buttons"><button class="tp-text-btn" data-prev>← Previous</button><button class="tp-text-btn" data-next>Next →</button></div></div>';
  document.body.append(dialog);
  const player = dialog.querySelector('video'), picker = dialog.querySelector('select');
  let current = 0, opener = null;
  tasks.forEach((t, i) => { const option = document.createElement('option'); option.value = i; option.textContent = t.title; picker.append(option); });
  function showTask(index, button) {
    current = index; if (button) opener = button;
    const t = tasks[index];
    dialog.querySelector('#tpDialogTitle').textContent = t.title;
    dialog.querySelector('#tpDialogIndex').textContent = 'Task-Level Evaluation';
    dialog.querySelector('#tpDialogRequest').textContent = t.request;
    dialog.querySelector('[data-prev]').disabled = index === 0;
    dialog.querySelector('[data-next]').disabled = index === tasks.length - 1;
    picker.value = index;
    player.pause(); player.src = t.src; player.poster = t.poster; player.load();
    if (!dialog.open) dialog.showModal();
    dialogOpen = true; syncVideos(); player.play().catch(() => {});
  }
  picker.addEventListener('change', () => showTask(Number(picker.value)));
  dialog.querySelector('.tp-icon').addEventListener('click', () => dialog.close());
  dialog.querySelector('[data-prev]').addEventListener('click', () => { if (current > 0) showTask(current - 1); });
  dialog.querySelector('[data-next]').addEventListener('click', () => { if (current < tasks.length - 1) showTask(current + 1); });
  dialog.addEventListener('click', e => { if (e.target === dialog) { const r = dialog.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close(); } });
  dialog.addEventListener('close', () => { player.pause(); player.removeAttribute('src'); player.load(); dialogOpen = false; syncVideos(); if (opener?.isConnected) opener.focus({ preventScroll: true }); });
  section.querySelector('[data-tp-browse]').addEventListener('click', e => showTask(0, e.currentTarget));
})();
