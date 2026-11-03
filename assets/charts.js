/* Chart module for the ES 335 Hydro-ML site.
 * Every chart is mounted from a <div data-chart="name[:arg:arg]"> element and
 * reads only window.FIGDATA (generated) and window.STATIONS / window.MEDIANS.
 * Used by index.html and by every standalone page in visualizations/.
 */
(function () {
  'use strict';

  var C = {
    ink: '#16202b', muted: '#5b6470', faint: '#8b929b', grid: '#ebe7de', zero: '#b9b3a6',
    obs: '#16202b', m0: '#8d8a83', m1: '#3466c9', m2: '#0f9478',
    dry: '#d4881c', dryText: '#94590a'
  };
  var MODEL = {
    m0: { name: 'M0 Baseline', color: C.m0 },
    m1: { name: 'M1 Flow-only', color: C.m1 },
    m2: { name: 'M2 Hydro-met', color: C.m2 }
  };
  var FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
  var MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace';
  var H_LABEL = { '1w': '1 wk', '2w': '2 wk', '4w': '4 wk', '8w': '8 wk' };

  function num(v) { return v === null || v === undefined || v === '' ? null : parseFloat(v); }

  function axis(extra) {
    var a = {
      gridcolor: C.grid, gridwidth: 1, zeroline: false, showline: true, linecolor: '#d9d4c8', linewidth: 1,
      ticks: '', tickfont: { family: MONO, size: 11, color: C.faint }, title: { font: { size: 12, color: C.muted }, standoff: 10 },
      automargin: true
    };
    for (var k in extra) a[k] = extra[k];
    return a;
  }
  function baseLayout(extra) {
    var l = {
      font: { family: FONT, size: 12.5, color: C.muted },
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 56, r: 18, t: 12, b: 44 },
      hovermode: 'closest',
      hoverlabel: { bgcolor: '#ffffff', bordercolor: '#d9d4c8', font: { family: FONT, size: 12.5, color: C.ink }, align: 'left' },
      showlegend: true,
      legend: { orientation: 'h', x: 0, xanchor: 'left', y: 1.02, yanchor: 'bottom', font: { size: 12.5, color: C.ink }, bgcolor: 'rgba(0,0,0,0)', traceorder: 'normal' },
      xaxis: axis(), yaxis: axis()
    };
    for (var k in extra) l[k] = extra[k];
    return l;
  }
  var CONFIG = { responsive: true, displaylogo: false, displayModeBar: false, scrollZoom: false };

  function draw(el, data, layout) {
    if (el.__drawn) return Plotly.react(el, data, layout, CONFIG);
    el.__drawn = true;
    return Plotly.newPlot(el, data, layout, CONFIG);
  }

  /* Segmented controls rendered above a chart. Returns the live state object. */
  function controls(host, groups, onChange) {
    var bar = document.createElement('div');
    bar.className = 'chart-controls';
    var state = {};
    groups.forEach(function (g) {
      var lab = document.createElement('span');
      lab.className = 'seg-label';
      lab.textContent = g.label;
      var wrap = document.createElement('div');
      wrap.className = 'seg';
      wrap.setAttribute('role', 'radiogroup');
      wrap.setAttribute('aria-label', g.label);
      state[g.key] = g.value;
      g.options.forEach(function (o) {
        var b = document.createElement('button');
        b.type = 'button';
        b.textContent = o.label;
        b.setAttribute('role', 'radio');
        b.setAttribute('aria-checked', String(o.value === g.value));
        b.addEventListener('click', function () {
          if (state[g.key] === o.value) return;
          state[g.key] = o.value;
          wrap.querySelectorAll('button').forEach(function (x) { x.setAttribute('aria-checked', 'false'); });
          b.setAttribute('aria-checked', 'true');
          onChange(state);
        });
        wrap.appendChild(b);
      });
      var group = document.createElement('div');
      group.className = 'seg-group';
      group.appendChild(lab);
      group.appendChild(wrap);
      bar.appendChild(group);
    });
    host.parentNode.insertBefore(bar, host);
    return state;
  }

  /* ---------- Portfolio median skill vs lead horizon ---------- */
  function skill(el) {
    var M = window.MEDIANS, hs = ['1w', '2w', '4w', '8w'];
    function render(s) {
      var key = s.metric;
      var traces = ['m0', 'm1', 'm2'].map(function (m) {
        return {
          x: hs.map(function (h) { return H_LABEL[h]; }),
          y: hs.map(function (h) { return num(M[h][m + '_' + key]); }),
          name: MODEL[m].name, mode: 'lines+markers',
          line: { color: MODEL[m].color, width: m === 'm2' ? 3 : 2, dash: m === 'm0' ? 'dot' : 'solid' },
          marker: { size: m === 'm2' ? 10 : 8, color: MODEL[m].color, line: { color: '#fbfaf7', width: 2 } },
          hovertemplate: '<b>' + MODEL[m].name + '</b><br>%{x} lead · median ' + key.toUpperCase() + ' %{y:+.3f}<extra></extra>'
        };
      });
      draw(el, traces, baseLayout({
        yaxis: axis({ title: { text: 'Portfolio median ' + key.toUpperCase() }, range: [-0.45, 0.75], zeroline: true, zerolinecolor: C.zero, zerolinewidth: 1 }),
        xaxis: axis({ title: { text: 'Forecast lead time' }, type: 'category' }),
        shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: 1.5, x1: 3.5, y0: 0, y1: 1, fillcolor: 'rgba(212,136,28,0.07)', line: { width: 0 }, layer: 'below' }],
        annotations: [
          { xref: 'x', yref: 'paper', x: 0.5, y: 0.03, text: 'M0 = persistence', showarrow: false, font: { size: 11, color: C.faint }, yanchor: 'bottom' },
          { xref: 'x', yref: 'paper', x: 2.5, y: 0.03, text: 'M0 = weekly climatology', showarrow: false, font: { size: 11, color: C.dryText }, yanchor: 'bottom' }
        ]
      }));
    }
    render(controls(el, [{ key: 'metric', label: 'Metric', value: 'nse', options: [{ label: 'NSE', value: 'nse' }, { label: 'KGE', value: 'kge' }] }], render));
  }

  /* ---------- Hydrograph: observed vs M1 vs M2 ---------- */
  function hydro(el, args) {
    var F = window.FIGDATA.hydro;
    var fixed = args.length === 2;
    function render(s) {
      var d = F[s.station][s.lead];
      var traces = [
        { x: d.x, y: d.obs, name: 'Observed Q', mode: 'lines', line: { color: C.obs, width: 2 }, hovertemplate: 'Observed %{y:,.1f} m³/s<extra></extra>' },
        { x: d.x, y: d.m1, name: 'M1 Flow-only · NSE ' + d.nse_m1.toFixed(3), mode: 'lines', line: { color: C.m1, width: 1.6, dash: 'dash' }, hovertemplate: 'M1 %{y:,.1f} m³/s<extra></extra>' },
        { x: d.x, y: d.m2, name: 'M2 Hydro-met · NSE ' + d.nse_m2.toFixed(3), mode: 'lines', line: { color: C.m2, width: 2 }, hovertemplate: 'M2 %{y:,.1f} m³/s<extra></extra>' }
      ];
      var range = s.view === 'focus' ? ['2018-01-01', '2019-12-31'] : [d.x[0], d.x[d.x.length - 1]];
      draw(el, traces, baseLayout({
        hovermode: 'x unified',
        xaxis: axis({ range: range, type: 'date', showspikes: true, spikemode: 'across', spikethickness: 1, spikecolor: '#b9b3a6', spikedash: 'solid' }),
        yaxis: axis({ title: { text: 'Discharge (m³/s)' }, rangemode: 'tozero' }),
        shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: d.q95, y1: d.q95, line: { color: C.dry, width: 1.5, dash: 'dot' } }],
        annotations: [{ xref: 'paper', x: 0.01, xanchor: 'left', y: d.q95, yanchor: 'bottom', text: 'Q95 severe-drought threshold · ' + d.q95.toFixed(1) + ' m³/s', showarrow: false, font: { size: 11, color: C.dryText } }]
      }));
    }
    var groups = [];
    if (!fixed) {
      groups.push({ key: 'station', label: 'Station', value: 'karad', options: [{ label: 'Karad · Krishna', value: 'karad' }, { label: 'Kantamal · Mahanadi', value: 'kantamal' }] });
      groups.push({ key: 'lead', label: 'Lead', value: '1w', options: [{ label: '1 week', value: '1w' }, { label: '4 weeks', value: '4w' }] });
    }
    groups.push({ key: 'view', label: 'Window', value: 'focus', options: [{ label: '2018–19', value: 'focus' }, { label: '2016–2025', value: 'all' }] });
    var st = controls(el, groups, render);
    if (fixed) { st.station = args[0]; st.lead = args[1]; }
    render(st);
  }

  /* ---------- Precision–recall for severe drought ---------- */
  function pr(el) {
    var F = window.FIGDATA.pr;
    function render(s) {
      var d = F[s.lead];
      var tr = ['m1', 'm2'].map(function (m) {
        var c = d[m];
        return {
          x: c.recall, y: c.precision, customdata: c.cutoff.map(function (v, i) { return [v, c.far[i]]; }),
          name: MODEL[m].name + ' · PR-AUC ' + c.auc.toFixed(3), mode: 'lines',
          line: { color: MODEL[m].color, width: m === 'm2' ? 2.6 : 2 },
          hovertemplate: '<b>' + MODEL[m].name + '</b><br>Recall (POD) %{x:.2f}<br>Precision %{y:.2f}<br>FAR %{customdata[1]:.2f}<br>Cutoff %{customdata[0]:.1f} m³/s<extra></extra>'
        };
      });
      tr.push({ x: [0, 1], y: [d.noskill, d.noskill], name: 'No-skill · ' + d.noskill.toFixed(3), mode: 'lines', line: { color: C.faint, width: 1.2, dash: 'dash' }, hoverinfo: 'skip' });
      draw(el, tr, baseLayout({
        xaxis: axis({ title: { text: 'Recall · probability of detection' }, range: [0, 1.02] }),
        yaxis: axis({ title: { text: 'Precision · 1 − FAR' }, range: [0, 1.02] })
      }));
    }
    render(controls(el, [{ key: 'lead', label: 'Lead', value: '2w', options: [{ label: '2 weeks', value: '2w' }, { label: '4 weeks', value: '4w' }] }], render));
  }

  /* ---------- KGE components as three small multiples ---------- */
  function kge(el) {
    var F = window.FIGDATA.kge, hs = F.h.map(function (h) { return H_LABEL[h]; });
    var panels = [['r', 'Correlation r'], ['alpha', 'Variability α'], ['beta', 'Bias β']];
    var gap = 0.06, w = (1 - 2 * gap) / 3, traces = [];
    var lay = baseLayout({ margin: { l: 44, r: 10, t: 30, b: 70 }, annotations: [], shapes: [] });
    panels.forEach(function (p, i) {
      var xa = i ? 'x' + (i + 1) : 'x', ya = i ? 'y' + (i + 1) : 'y';
      ['m0', 'm1', 'm2'].forEach(function (m) {
        traces.push({
          x: hs, y: F[p[0]][m], xaxis: xa, yaxis: ya, name: MODEL[m].name, legendgroup: m, showlegend: i === 0,
          mode: 'lines+markers', line: { color: MODEL[m].color, width: m === 'm2' ? 2.6 : 1.8, dash: m === 'm0' ? 'dot' : 'solid' },
          marker: { size: 7, color: MODEL[m].color, line: { color: '#fbfaf7', width: 1.5 } },
          hovertemplate: '<b>' + MODEL[m].name + '</b><br>' + p[1] + ' at %{x}: %{y:.3f}<extra></extra>'
        });
      });
      lay[i ? 'xaxis' + (i + 1) : 'xaxis'] = axis({ domain: [i * (w + gap), i * (w + gap) + w], anchor: ya, type: 'category' });
      lay[i ? 'yaxis' + (i + 1) : 'yaxis'] = axis({ anchor: xa, range: [0, 1.15], dtick: 0.25, showticklabels: i === 0 });
      lay.shapes.push({ type: 'line', xref: xa + ' domain', yref: ya, x0: 0, x1: 1, y0: 1, y1: 1, line: { color: C.ink, width: 1, dash: 'dot' } });
      lay.annotations.push({ xref: xa + ' domain', yref: 'paper', x: 0, xanchor: 'left', y: 1.0, yanchor: 'bottom', text: '<b>' + p[1] + '</b>', showarrow: false, font: { size: 12.5, color: C.ink } });
      lay.annotations.push({ xref: xa + ' domain', yref: ya, x: 1, xanchor: 'right', y: 1, yanchor: 'bottom', text: 'ideal', showarrow: false, font: { size: 10.5, color: C.faint } });
    });
    lay.legend = Object.assign({}, lay.legend, { y: -0.14, yanchor: 'top' });
    draw(el, traces, lay);
  }

  /* ---------- Feature importance 1w vs 8w as a dumbbell ---------- */
  function fi(el) {
    var F = window.FIGDATA.fi;
    var idx = F.features.map(function (_, i) { return i; }).sort(function (a, b) { return F.h1[a] - F.h1[b]; });
    var f = idx.map(function (i) { return F.features[i]; });
    var a = idx.map(function (i) { return F.h1[i]; });
    var b = idx.map(function (i) { return F.h8[i]; });
    var lx = [], ly = [];
    f.forEach(function (name, i) { lx.push(a[i], b[i], null); ly.push(name, name, null); });
    draw(el, [
      { x: lx, y: ly, mode: 'lines', line: { color: '#ddd8cc', width: 3 }, hoverinfo: 'skip', showlegend: false },
      { x: a, y: f, name: '1-week lead', mode: 'markers', marker: { size: 11, color: '#fbfaf7', line: { color: C.ink, width: 2 } }, hovertemplate: '%{y}<br>1-week lead: %{x:.1f}%<extra></extra>' },
      { x: b, y: f, name: '8-week lead', mode: 'markers', marker: { size: 11, color: C.dry, line: { color: '#fbfaf7', width: 1.5 } }, hovertemplate: '%{y}<br>8-week lead: %{x:.1f}%<extra></extra>' }
    ], baseLayout({
      margin: { l: 10, r: 18, t: 12, b: 44 },
      xaxis: axis({ title: { text: 'Relative importance (%)' }, range: [-2, 72] }),
      yaxis: axis({ showgrid: false, showline: false, tickfont: { family: FONT, size: 12, color: C.ink } })
    }));
  }

  /* ---------- Zero-shot transfer: dot plot per held-out basin ---------- */
  function transfer(el) {
    var F = window.FIGDATA.transfer;
    var labelM = { m0: 'M0 Climatology', m1: 'M1 Flow-only GBT', m2: 'M2 HydroLSTM' };
    function render(s) {
      var d = F[s.lead];
      var rows = d.basins.map(function (b, i) { return b + ' · ' + d.states[i]; });
      var lx = [], ly = [];
      rows.forEach(function (r, i) {
        var v = [d.m0[i], d.m1[i], d.m2[i]];
        lx.push(Math.min.apply(null, v), Math.max.apply(null, v), null);
        ly.push(r, r, null);
      });
      var tr = [{ x: lx, y: ly, mode: 'lines', line: { color: '#e4dfd4', width: 6 }, hoverinfo: 'skip', showlegend: false }];
      ['m1', 'm2', 'm0'].forEach(function (m) {
        tr.push({
          x: d[m], y: rows, name: labelM[m], mode: 'markers',
          marker: { size: m === 'm2' ? 15 : (m === 'm0' ? 11 : 12), color: m === 'm0' ? 'rgba(251,250,247,0.9)' : MODEL[m].color, symbol: m === 'm0' ? 'diamond' : 'circle', line: { color: m === 'm0' ? '#5f5c56' : '#fbfaf7', width: 2 } },
          hovertemplate: '<b>' + labelM[m] + '</b><br>%{y}<br>NSE %{x:+.3f}<extra></extra>'
        });
      });
      draw(el, tr, baseLayout({
        margin: { l: 10, r: 18, t: 12, b: 44 },
        xaxis: axis({ title: { text: 'Zero-shot NSE' }, range: [-1.4, 0.65], zeroline: true, zerolinecolor: C.ink, zerolinewidth: 1 }),
        yaxis: axis({ showgrid: false, showline: false, autorange: 'reversed', tickfont: { family: FONT, size: 12.5, color: C.ink } })
      }));
    }
    render(controls(el, [{ key: 'lead', label: 'Lead', value: '4w', options: [{ label: '4 weeks', value: '4w' }, { label: '8 weeks', value: '8w' }] }], render));
  }

  /* ---------- Catchment portfolio: aridity vs area ---------- */
  function portfolio(el) {
    var S = window.STATIONS, states = ['Maharashtra', 'Odisha', 'Karnataka'], sym = ['circle', 'square', 'diamond'];
    var holdout = function (x) { return /Holdout/.test(x.role); };
    var tr = states.map(function (st, k) {
      var s = S.filter(function (x) { return x.state === st; });
      return {
        x: s.map(function (x) { return x.aridity; }), y: s.map(function (x) { return x.area_km2; }), name: st,
        text: s.map(function (x) { return x.name.charAt(0) + x.name.slice(1).toLowerCase(); }),
        customdata: s.map(function (x) { return [x.basin, x.role]; }),
        mode: 'markers+text', textposition: 'middle right', cliponaxis: false,
        textfont: { size: 11.5, color: C.muted, family: FONT },
        marker: {
          size: 13, symbol: sym[k],
          color: s.map(function (x) { return holdout(x) ? '#fbfaf7' : C.ink; }),
          line: { color: s.map(function (x) { return holdout(x) ? C.dry : '#fbfaf7'; }), width: s.map(function (x) { return holdout(x) ? 3 : 1.5; }) }
        },
        hovertemplate: '<b>%{text}</b> · ' + st + '<br>%{customdata[0]}<br>Area %{y:,.0f} km² · Aridity %{x:.3f}<br>%{customdata[1]}<extra></extra>'
      };
    });
    draw(el, tr, baseLayout({
      margin: { l: 60, r: 70, t: 12, b: 48 },
      xaxis: axis({ title: { text: 'Aridity index (P/PET) — wetter to the right' }, range: [0.42, 1.22] }),
      yaxis: axis({ title: { text: 'Catchment area (km², log scale)' }, type: 'log', range: [3.15, 4.95] })
    }));
  }

  /* ---------- Per-station NSE across horizons (station explorer) ---------- */
  function stationMini(el, station) {
    var hs = ['1w', '2w', '4w', '8w'];
    var tr = ['m0', 'm1', 'm2'].map(function (m) {
      return {
        x: hs.map(function (h) { return H_LABEL[h]; }), y: hs.map(function (h) { return num(station.metrics[h][m + '_nse']); }),
        name: MODEL[m].name, mode: 'lines+markers',
        line: { color: MODEL[m].color, width: m === 'm2' ? 2.6 : 1.8, dash: m === 'm0' ? 'dot' : 'solid' },
        marker: { size: 8, color: MODEL[m].color, line: { color: '#fbfaf7', width: 1.5 } },
        hovertemplate: '<b>' + MODEL[m].name + '</b> · %{x}: NSE %{y:+.3f}<extra></extra>'
      };
    });
    draw(el, tr, baseLayout({
      margin: { l: 46, r: 10, t: 8, b: 36 },
      xaxis: axis({ type: 'category' }),
      yaxis: axis({ title: { text: 'NSE' }, range: [-0.75, 0.75], zeroline: true, zerolinecolor: C.zero }),
      legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 11.5, color: C.ink } }
    }));
  }

  var REGISTRY = { skill: skill, hydro: hydro, pr: pr, kge: kge, fi: fi, transfer: transfer, portfolio: portfolio };

  function mount(el) {
    if (el.__mounted) return;
    el.__mounted = true;
    var parts = el.getAttribute('data-chart').split(':');
    var fn = REGISTRY[parts[0]];
    if (fn && window.Plotly) fn(el, parts.slice(1));
  }

  /* Charts mount lazily as they approach the viewport (eager on standalone pages). */
  function mountAll(root, opts) {
    var els = Array.prototype.slice.call((root || document).querySelectorAll('[data-chart]'));
    if ((opts && opts.eager) || !('IntersectionObserver' in window)) { els.forEach(mount); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { mount(e.target); io.unobserve(e.target); } });
    }, { rootMargin: '300px 0px' });
    els.forEach(function (el) { io.observe(el); });
  }

  window.Charts = { mountAll: mountAll, stationMini: stationMini, colors: C, models: MODEL };
})();
