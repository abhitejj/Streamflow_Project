/* Page behaviour: nav, scroll reveals, hero hydrograph, ΔNSE heatmap, station explorer, copy buttons. */
(function () {
  'use strict';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var fmt = function (v) { v = parseFloat(v); return isNaN(v) ? '—' : (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3); };

  /* ---------- icons ---------- */
  if (window.lucide) lucide.createIcons({ attrs: { class: 'icon', 'aria-hidden': 'true' } });

  /* ---------- nav: scrolled state, progress, active link, mobile menu ---------- */
  var nav = $('.nav'), bar = $('.progress'), toTop = $('.to-top');
  var ticking = false;

  var links = $$('.nav-links a');
  var byId = {};
  links.forEach(function (a) {
    var id = (a.getAttribute('href') || '').replace(/^#/, '');
    if (id) byId[id] = a;
  });

  var sectionIds = ['introduction', 'dataset', 'methodology', 'results', 'error-analysis', 'conclusion', 'reproducibility'];
  var sections = sectionIds.map(function (id) { return document.getElementById(id); }).filter(Boolean);

  var navLock = false;
  var navLockTimer = null;

  function setActiveNav(id) {
    if (!id || !byId[id]) return;
    links.forEach(function (a) { a.classList.remove('active'); });
    byId[id].classList.add('active');
  }

  function updateNavSpy() {
    if (navLock) return;
    var y = window.scrollY;
    var docHeight = document.documentElement.scrollHeight;
    var winHeight = window.innerHeight;

    // 1. Initial / page-top state (above introduction section)
    var introEl = document.getElementById('introduction');
    var introTop = introEl ? introEl.getBoundingClientRect().top + y : 600;
    if (y < introTop - 200) {
      setActiveNav('introduction');
      return;
    }

    // 2. Near bottom of page -> highlight the last section
    if (winHeight + y >= docHeight - 80) {
      setActiveNav('reproducibility');
      return;
    }

    // 3. Scan sections with a target line at ~32% from the viewport top
    var target = y + winHeight * 0.32;
    var activeId = null;

    for (var i = 0; i < sections.length; i++) {
      var s = sections[i];
      var sTop = s.offsetTop;
      var sBot = sTop + s.offsetHeight;
      if (target >= sTop && target < sBot) {
        activeId = s.id;
        break;
      }
    }

    if (!activeId && sections.length) {
      for (var j = sections.length - 1; j >= 0; j--) {
        if (sections[j].offsetTop <= target) {
          activeId = sections[j].id;
          break;
        }
      }
    }

    if (activeId) {
      setActiveNav(activeId);
    }
  }

  function onScroll() {
    var y = window.scrollY, h = document.documentElement.scrollHeight - window.innerHeight;
    nav.classList.toggle('scrolled', y > 8);
    bar.style.transform = 'scaleX(' + (h > 0 ? Math.min(1, y / h) : 0) + ')';
    toTop.classList.toggle('show', y > 900);
    updateNavSpy();
    ticking = false;
  }
  window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(onScroll); } }, { passive: true });
  onScroll();
  toTop.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' }); });

  var toggle = $('.nav-toggle'), menu = $('.nav-links');
  toggle.addEventListener('click', function () {
    var open = menu.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });

  links.forEach(function (a) {
    a.addEventListener('click', function () {
      var id = (a.getAttribute('href') || '').replace(/^#/, '');
      if (byId[id]) {
        setActiveNav(id);
        navLock = true;
        clearTimeout(navLockTimer);
        navLockTimer = setTimeout(function () {
          navLock = false;
          updateNavSpy();
        }, 900);
      }
      menu.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });

  window.addEventListener('hashchange', function () {
    var h = window.location.hash.replace(/^#/, '');
    if (byId[h]) setActiveNav(h);
  });

  // Highlight Introduction on initial page load
  var initialHash = window.location.hash.replace(/^#/, '');
  if (initialHash && byId[initialHash]) {
    setActiveNav(initialHash);
  } else {
    setActiveNav('introduction');
  }

  /* ---------- reveal on scroll (+ count-up, timeline, finding bars) ---------- */
  function countUp(el) {
    var raw = el.getAttribute('data-count'), end = parseFloat(raw), dec = (raw.split('.')[1] || '').length;
    var pre = el.getAttribute('data-prefix') || '', suf = el.getAttribute('data-suffix') || '';
    var sep = el.hasAttribute('data-sep');
    var show = function (v) {
      var s = sep ? Math.abs(v).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec }) : Math.abs(v).toFixed(dec);
      el.textContent = pre + (end < 0 ? '−' : '') + s + suf;
    };
    if (reduce) return show(end);
    var t0 = null, dur = 1300;
    requestAnimationFrame(function step(t) {
      t0 = t0 || t;
      var p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      show(end * e);
      if (p < 1) requestAnimationFrame(step);
    });
  }
  var revObs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add('in');
      $$('[data-count]', e.target).forEach(countUp);
      revObs.unobserve(e.target);
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
  $$('[data-stagger]').forEach(function (g) {
    $$(':scope > [data-reveal]', g).forEach(function (c, i) { c.style.setProperty('--d', (i * 0.09) + 's'); });
  });
  $$('[data-reveal], .timeline, .finding, .stats').forEach(function (el) { revObs.observe(el); });

  /* ---------- hero: observed hydrograph drawn as a ribbon ---------- */
  (function hero() {
    var svg = $('#river-svg');
    if (!svg || !window.FIGDATA) return;
    var d = FIGDATA.hydro.kantamal['1w'];
    var ys = d.obs.map(function (v) { return v === null ? 0 : Math.sqrt(Math.max(0, v)); });
    var max = Math.max.apply(null, ys), W = 1000, H = 200, top = 34;
    var pts = ys.map(function (v, i) { return [(i / (ys.length - 1)) * W, H - 3 - (v / max) * (H - top - 3)]; });
    var line = 'M' + pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join('L');
    var path = $('#river-line');
    path.setAttribute('d', line);
    $('#river-fill').setAttribute('d', line + 'L' + W + ',' + H + 'L0,' + H + 'Z');
    path.style.setProperty('--len', path.getTotalLength());
    setTimeout(function () { $('.river').classList.add('drawn'); }, reduce ? 0 : 200);
  })();

  /* ---------- ΔNSE heatmap (M2 − M1) ---------- */
  var S = window.STATIONS || [];
  var HS = ['1w', '2w', '4w', '8w'];
  var cap = function (n) { return n.charAt(0) + n.slice(1).toLowerCase(); };
  function heatColor(v) {
    /* diverging: ochre (M1 better) ← neutral grey → teal (M2 better); clamp at ±0.4 */
    var t = Math.max(-1, Math.min(1, v / 0.4));
    var neutral = [236, 233, 226], pos = [11, 122, 99], neg = [194, 98, 27];
    var tgt = t >= 0 ? pos : neg, a = Math.pow(Math.abs(t), 0.75);
    var c = neutral.map(function (n, i) { return Math.round(n + (tgt[i] - n) * a); });
    return { bg: 'rgb(' + c.join(',') + ')', fg: a > 0.55 ? '#fff' : '#16202b' };
  }
  var heat = $('#heat');
  var selected = { id: S.length ? S[0].id : null, h: '1w' };
  if (heat) {
    var html = '<div></div>' + HS.map(function (h) { return '<div class="hh">' + h.replace('w', ' wk') + '</div>'; }).join('');
    var lastState = null;
    S.forEach(function (s) {
      if (s.state !== lastState) { html += '<div class="state-sep">' + s.state + '</div>'; lastState = s.state; }
      var nm = cap(s.name);
      html += '<div class="rh">' + nm + '<small>' + (/Holdout/.test(s.role) ? 'Protocol 2 holdout' : s.basin) + '</small></div>';
      HS.forEach(function (h) {
        var v = parseFloat(s.metrics[h].m2_nse) - parseFloat(s.metrics[h].m1_nse);
        var c = heatColor(v);
        html += '<button type="button" class="cell" data-id="' + s.id + '" data-h="' + h + '" style="background:' + c.bg + ';color:' + c.fg +
          '" aria-label="' + nm + ', ' + h + ' lead: M2 minus M1 NSE ' + fmt(v) + '">' + fmt(v) + '</button>';
      });
    });
    heat.innerHTML = html;
    heat.addEventListener('click', function (e) {
      var b = e.target.closest('.cell');
      if (!b) return;
      setSelection(b.getAttribute('data-id'), b.getAttribute('data-h'));
      if (window.innerWidth < 900) $('#explorer').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    });
  }

  /* ---------- station explorer ---------- */
  var sel = $('#station-select'), hSeg = $('#horizon-seg');
  if (sel) {
    S.forEach(function (s) {
      var o = document.createElement('option');
      o.value = s.id;
      o.textContent = cap(s.name) + ' — ' + s.state + ' · ' + s.basin;
      sel.appendChild(o);
    });
    sel.addEventListener('change', function () { setSelection(sel.value, selected.h); });
    $$('button', hSeg).forEach(function (b) {
      b.addEventListener('click', function () { setSelection(selected.id, b.getAttribute('data-h')); });
    });
  }
  function setSelection(id, h) {
    selected.id = id; selected.h = h;
    if (sel) sel.value = id;
    if (hSeg) $$('button', hSeg).forEach(function (b) { b.setAttribute('aria-checked', String(b.getAttribute('data-h') === h)); });
    if (heat) $$('.cell', heat).forEach(function (c) { c.classList.toggle('sel', c.getAttribute('data-id') === id && c.getAttribute('data-h') === h); });
    renderExplorer();
  }
  function renderExplorer() {
    var s = S.filter(function (x) { return x.id === selected.id; })[0];
    if (!s || !$('#ex-attrs')) return;
    var m = s.metrics[selected.h];
    $('#ex-attrs').innerHTML =
      '<div><dt>Sub-basin</dt><dd>' + s.basin + '</dd></div>' +
      '<div><dt>CAMELS ID</dt><dd>Gauge ' + s.camels_id + '</dd></div>' +
      '<div><dt>Area</dt><dd>' + s.area_km2.toLocaleString('en-IN') + ' km²</dd></div>' +
      '<div><dt>Aridity</dt><dd>' + s.aridity + '</dd></div>' +
      '<div style="grid-column:1/-1"><dt>Regime · role</dt><dd>' + s.regime + ' · ' + s.role + '</dd></div>';
    $('#kpi-m0').textContent = fmt(m.m0_nse);
    $('#kpi-m1').textContent = fmt(m.m1_nse);
    $('#kpi-m2').textContent = fmt(m.m2_nse);
    var dEl = $('#kpi-d');
    dEl.textContent = m.delta_nse_m2_m1 + ' Δ';
    dEl.className = 'kd ' + (parseFloat(m.delta_nse_m2_m1) >= 0 ? 'pos' : 'neg');
    $('#kpi-kge').textContent = fmt(m.m2_kge);
    $('#far-m1').textContent = m.far_m1 || '—';
    $('#far-m2').textContent = m.far_m2 || '—';
    $('#csi-m2').textContent = m.csi_m2 || '—';
    var mini = $('#station-mini');
    if (mini && window.Plotly && window.Charts) Charts.stationMini(mini, s);
  }
  if (S.length) setSelection(selected.id, selected.h);

  /* ---------- copy buttons ---------- */
  $$('.copy').forEach(function (b) {
    b.addEventListener('click', function () {
      var code = b.parentNode.querySelector('code').textContent;
      var label = b.querySelector('span');
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText(code).then(function () {
        b.classList.add('done'); label.textContent = 'Copied';
        setTimeout(function () { b.classList.remove('done'); label.textContent = 'Copy'; }, 1600);
      }, function () {});
    });
  });

  /* ---------- charts (mounted lazily) ---------- */
  if (window.Charts) Charts.mountAll(document);
})();
