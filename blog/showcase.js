/* Blog copies of the website's capability player and scenario video (see ../app.js). */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Five capability tests: one clip each, played in turn while the player is on screen.
  const root = $('#capx');
  if (root) {
    const items = $$('.cap-item', root), vids = $$('.cap-vid', root), tag = $('#capxTag');
    let cur = 0, inView = false;
    const load = v => { if (!v.src) v.src = v.dataset.capSrc; };
    const show = (i, play = true) => {
      cur = i;
      items.forEach((b, k) => { b.classList.toggle('is-active', k === i); b.setAttribute('aria-selected', String(k === i)); $('.cap-prog i', b).style.width = '0'; });
      vids.forEach((v, k) => { v.classList.toggle('is-active', k === i); if (k !== i) v.pause(); });
      tag.textContent = `Capability test 0${i + 1} / 0${vids.length}`;
      const v = vids[i]; load(v); v.currentTime = 0;
      if (play && inView && !reduce) v.play().catch(() => {});
      load(vids[(i + 1) % vids.length]);
    };
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
  }

  // Scenario run: load and play (muted) only while on screen.
  const scen = $('#scenVideo');
  if (scen) new IntersectionObserver(es => {
    if (es[0].isIntersecting) { if (!scen.src) scen.src = scen.dataset.scenSrc; if (!reduce) scen.play().catch(() => {}); }
    else scen.pause();
  }, { threshold: .4 }).observe(scen);
})();
