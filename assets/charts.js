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
  var COMPARISON_MODEL = {
    M0: { name: 'M0', color: C.m0, dash: 'dot' },
    M1: { name: 'M1', color: C.m1 },
    M2: { name: 'M2', color: C.m2 },
    'M1-log': { name: 'M1-log', color: '#8552a5', dash: 'dash' },
    'M2-log': { name: 'M2-log', color: '#a75b18', dash: 'dash' }
  };
  var FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
  var MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace';
  var H_LABEL = { '1w': '1 wk', '2w': '2 wk', '4w': '4 wk', '8w': '8 wk' };
  var LEADS = Object.keys(H_LABEL).map(function (h) {
    return { value: h, label: h === '1w' ? '1 week' : h[0] + ' weeks' };
  });

  function num(v) { return v === null || v === undefined || v === '' ? null : parseFloat(v); }

  function scoreRange(values) {
    var scores = [0].concat(values.filter(Number.isFinite));
    var low = Math.min.apply(null, scores), high = Math.max.apply(null, scores);
    var pad = Math.max(0.05, (high - low) * 0.08);
    return [low - pad, high + pad];
  }

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

  function radioGroup(group, onChange) {
    var buttons = Array.from(group.querySelectorAll('button[role="radio"]'));
    group.addEventListener('click', function (event) {
      var button = event.target.closest('button[role="radio"]');
      if (button) onChange(button.value);
    });
    group.addEventListener('keydown', function (event) {
      var i = buttons.indexOf(event.target), next;
      if (i < 0) return;
      switch (event.key) {
        case 'ArrowRight': case 'ArrowDown': next = (i + 1) % buttons.length; break;
        case 'ArrowLeft': case 'ArrowUp': next = (i + buttons.length - 1) % buttons.length; break;
        case 'Home': next = 0; break;
        case 'End': next = buttons.length - 1; break;
        default: return;
      }
      event.preventDefault();
      buttons[next].focus();
      buttons[next].click();
    });
    return function (value) {
      buttons.forEach(function (b) {
        var checked = b.value === value;
        b.setAttribute('aria-checked', String(checked));
        b.tabIndex = checked ? 0 : -1;
      });
    };
  }

  function controls(host, groups, onChange) {
    var bar = document.createElement('div');
    bar.className = 'chart-controls';
    var state = {}, update = {};
    groups.forEach(function (g) {
      var lab = document.createElement('span');
      lab.className = 'seg-label';
      lab.textContent = g.label;
      state[g.key] = g.value;
      function change(value) {
        if (state[g.key] === value) return;
        state[g.key] = value;
        update[g.key](value);
        onChange(state, g.key);
      }
      var wrap = document.createElement(g.select ? 'select' : 'div');
      wrap.setAttribute('aria-label', g.label);
      if (g.select) {
        g.options.forEach(function (o) { wrap.add(new Option(o.label, o.value)); });
        wrap.addEventListener('change', function () { change(wrap.value); });
        update[g.key] = function (value) { wrap.value = value; };
      } else {
        wrap.className = 'seg';
        wrap.setAttribute('role', 'radiogroup');
        g.options.forEach(function (o) {
          var b = document.createElement('button');
          b.type = 'button';
          b.value = o.value;
          b.textContent = o.label;
          b.setAttribute('role', 'radio');
          wrap.appendChild(b);
        });
        update[g.key] = radioGroup(wrap, change);
      }
      update[g.key](g.value);
      var group = document.createElement('div');
      group.className = 'seg-group';
      group.appendChild(lab);
      group.appendChild(wrap);
      bar.appendChild(group);
    });
    host.parentNode.insertBefore(bar, host);
    return {
      state: state,
      set: function (values) {
        Object.keys(values).forEach(function (key) {
          if (!update[key]) return;
          state[key] = values[key];
          update[key](values[key]);
        });
      }
    };
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
      traces.push({
        x: hs.map(function (h) { return H_LABEL[h]; }),
        y: hs.map(function (h) { return num(M[h]['clim_' + key]); }),
        name: 'Weekly Climatology (All Leads)',
        mode: 'lines+markers',
        line: { color: C.dryText, width: 1.6, dash: 'dot' },
        marker: { size: 6, color: C.dryText, symbol: 'diamond' },
        hovertemplate: '<b>Weekly Climatology (All Leads)</b><br>%{x} lead · median ' + key.toUpperCase() + ' %{y:+.3f}<extra></extra>'
      });
      draw(el, traces, baseLayout({
        margin: { l: 56, r: 18, t: 28, b: 44 },
        yaxis: axis({ title: { text: 'Portfolio median ' + key.toUpperCase() }, range: [-0.45, 0.75], zeroline: true, zerolinecolor: C.zero, zerolinewidth: 1 }),
        xaxis: axis({ title: { text: 'Forecast lead time' }, type: 'category' }),
        shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: 1.5, x1: 3.5, y0: 0, y1: 1, fillcolor: 'rgba(212,136,28,0.07)', line: { width: 0 }, layer: 'below' }],
        annotations: [
          { xref: 'x', yref: 'paper', x: 0.5, y: 0.03, text: 'M0 = persistence', showarrow: false, font: { size: 11, color: C.faint }, yanchor: 'bottom' },
          { xref: 'x', yref: 'paper', x: 2.5, y: 0.03, text: 'M0 = weekly climatology', showarrow: false, font: { size: 11, color: C.dryText }, yanchor: 'bottom' }
        ]
      }));
    }
    render(controls(el, [{ key: 'metric', label: 'Metric', value: 'nse', options: [{ label: 'NSE', value: 'nse' }, { label: 'KGE', value: 'kge' }] }], render).state);
  }

  /* ---------- Historical forecast replay ---------- */
  var replayManifest = window.REPLAY_DATA;
  var chartAssetURL = document.currentScript.src;
  var replayCache = new Map();

  function fetchReplay(station, lead) {
    var entry = replayManifest && replayManifest.chunks[station] && replayManifest.chunks[station][lead];
    if (!entry) return Promise.reject(new Error('No replay file for this selection'));
    if (!replayCache.has(entry.file)) {
      var url = new URL(entry.file, chartAssetURL);
      url.searchParams.set('v', entry.sha256.slice(0, 12));
      replayCache.set(entry.file, fetch(url).then(function (response) {
        if (!response.ok) throw new Error('Replay file could not be loaded');
        return response.json();
      }).then(function (chunk) {
        if (chunk.station !== station || chunk.lead !== lead) throw new Error('Replay selection mismatch');
        return chunk;
      }).catch(function (error) {
        replayCache.delete(entry.file);
        throw error;
      }));
    }
    return replayCache.get(entry.file);
  }

  function hydro(el, args, shared) {
    var stations = window.STATIONS;
    var fixed = args.length === 2;
    var params = new URLSearchParams(window.location.search);
    var initial = shared ? shared.get() : {
      station: fixed ? args[0] : stations.some(function (s) { return s.id === params.get('station'); }) ? params.get('station') : stations[0].id,
      lead: fixed ? args[1] : Object.keys(H_LABEL).includes(params.get('lead')) ? params.get('lead') : '1w',
      objective: params.get('objective') === 'drought' ? 'drought' : 'flow',
      severity: ['Q95', 'Q90', 'Q80'].includes(params.get('severity')) ? params.get('severity') : 'Q95'
    };
    var columns = replayManifest && replayManifest.fields;
    var modelIds = Object.keys(COMPARISON_MODEL);
    var data, weeks, index, active, request = 0;
    var status = document.createElement('p');
    status.className = 'chart-selection replay-status';
    status.setAttribute('role', 'status');
    var panel = document.createElement('div');
    panel.className = 'replay-panel';
    panel.innerHTML =
      '<div class="replay-controls">' +
        '<label class="field">Target week<select aria-label="Target week"></select></label>' +
        '<button type="button" data-step="-1">Previous scored week</button>' +
        '<button type="button" data-step="1">Next scored week</button>' +
        '<button type="button" data-range="focus">Zoom to selected week</button>' +
        '<button type="button" data-range="full">Reset date range</button>' +
      '</div>' +
      '<p class="replay-week-status" role="status"></p>' +
      '<div class="table-scroll" tabindex="0" role="region" aria-label="Selected week forecasts">' +
        '<table class="replay-table" aria-label="Selected week forecasts"><thead><tr>' +
          '<th scope="col">Model</th><th scope="col" class="num">Forecast (m³/s)</th>' +
          '<th scope="col" class="replay-outcome">Drought outcome</th>' +
        '</tr></thead><tbody></tbody></table>' +
      '</div>';
    var retry = document.createElement('button');
    retry.type = 'button';
    retry.textContent = 'Retry loading replay';
    retry.className = 'replay-retry';
    var weekSelect = panel.querySelector('select');
    var weekStatus = panel.querySelector('.replay-week-status');
    var previous = panel.querySelector('[data-step="-1"]');
    var next = panel.querySelector('[data-step="1"]');
    var tbody = panel.querySelector('tbody');

    var groups = fixed ? [] : [
      { key: 'station', label: 'Hydrograph station', value: initial.station, select: true, options: stations.map(function (s) {
        return { label: s.name[0] + s.name.slice(1).toLowerCase(), value: s.id };
      }) },
      { key: 'lead', label: 'Hydrograph lead', value: initial.lead, options: LEADS }
    ];
    groups.push(
      { key: 'objective', label: 'Replay objective', value: initial.objective, options: [
        { value: 'flow', label: 'Flow accuracy' }, { value: 'drought', label: 'Drought warnings' }
      ] },
      { key: 'severity', label: 'Replay severity', value: initial.severity, options: [
        { value: 'Q95', label: 'Severe (Q95)' }, { value: 'Q90', label: 'Moderate (Q90)' }, { value: 'Q80', label: 'Mild (Q80)' }
      ] }
    );
    var ctrl = controls(el, groups, function (state, key) {
      if (shared) shared.change({ [key]: state[key] });
      else update(Object.assign({}, initial, state));
    });
    el.before(status, retry);
    el.after(panel);
    retry.addEventListener('click', function () { load(active); });

    function stationLabel() {
      var name = stations.find(function (s) { return s.id === active.station; }).name;
      return name[0] + name.slice(1).toLowerCase();
    }

    function leadLabel() {
      return LEADS.find(function (lead) { return lead.value === active.lead; }).label;
    }

    function adjacent(step) {
      for (var i = index + step; i >= 0 && i < weeks.length; i += step) if (weeks[i][2]) return i;
      return -1;
    }

    function shapes() {
      var date = weeks[index][1], threshold = data.thresholds[active.severity];
      return [
        { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: threshold, y1: threshold, line: { color: C.dry, width: 1.5, dash: 'dot' } },
        { type: 'line', xref: 'x', yref: 'paper', x0: date, x1: date, y0: 0, y1: 1, line: { color: C.dryText, width: 2, dash: 'dash' } }
      ];
    }

    function formatFlow(value) {
      return value.toLocaleString('en-IN', { maximumFractionDigits: 4 }) + ' m³/s';
    }

    function outcome(observed, predicted, threshold) {
      if (observed <= threshold) return predicted <= threshold ? 'Hit' : 'Miss';
      return predicted <= threshold ? 'False alarm' : 'Correct negative';
    }

    function renderWeek() {
      var row = weeks[index], scored = row[2], threshold = data.thresholds[active.severity];
      var drought = active.objective === 'drought';
      weekSelect.value = String(index);
      previous.disabled = adjacent(-1) < 0;
      next.disabled = adjacent(1) < 0;
      panel.classList.toggle('replay-drought', drought);
      weekStatus.textContent = 'Issued ' + row[0] + ' · Target week ' + row[1] + ' · ' +
        (scored ? 'Observed ' + formatFlow(row[4]) + ' · Scored in the full test block.' :
          row[3] ? 'Not scored. A recorded zero at issue or target was treated as unmeasured.' :
            'Not scored. A required observation or forecast was unavailable.') +
        ' Frozen ' + active.severity + ' threshold ≤ ' + formatFlow(threshold) + '.';
      tbody.innerHTML = modelIds.map(function (model) {
        var value = row[columns.indexOf(model)];
        return '<tr><th scope="row">' + model + '</th><td class="num">' + (scored ? formatFlow(value) : '—') +
          '</td><td class="replay-outcome">' + (scored && drought ? outcome(row[4], value, threshold) : '—') + '</td></tr>';
      }).join('');
    }

    function range(focus) {
      var first = focus ? Math.max(0, index - 12) : 0;
      var last = focus ? Math.min(weeks.length - 1, index + 12) : weeks.length - 1;
      var dates = [weeks[first][1], weeks[last][1]];
      if (!focus) return { 'xaxis.range': dates, 'yaxis.autorange': true };
      var high = weeks.slice(first, last + 1).reduce(function (max, row) {
        return row[2] ? Math.max.apply(null, [max].concat(row.slice(4))) : max;
      }, data.thresholds[active.severity]);
      return { 'xaxis.range': dates, 'yaxis.range': [0, Math.max(1, high * 1.08)], 'yaxis.autorange': false };
    }

    function selectWeek(value) {
      if (!weeks || !weeks[value]) return;
      index = value;
      renderWeek();
      var zoom = el._fullLayout && el._fullLayout.xaxis.range;
      var target = Date.parse(weeks[index][1]);
      var nextLayout = zoom && (target < Date.parse(zoom[0]) || target > Date.parse(zoom[1])) ? range(true) : {};
      nextLayout.shapes = shapes();
      Plotly.relayout(el, nextLayout);
    }

    weekSelect.addEventListener('change', function () { selectWeek(Number(weekSelect.value)); });
    previous.addEventListener('click', function () { selectWeek(adjacent(-1)); });
    next.addEventListener('click', function () { selectWeek(adjacent(1)); });
    panel.querySelectorAll('[data-range]').forEach(function (button) {
      button.addEventListener('click', function () { Plotly.relayout(el, range(button.dataset.range === 'focus')); });
    });

    function updateStatus() {
      var hasDrought = weeks.some(function (row) { return row[2] && row[4] <= data.thresholds[active.severity]; });
      status.textContent = stationLabel() + ' · ' + leadLabel() + ' ahead.' +
        (hasDrought ? '' : ' No scored drought weeks at ' + active.severity + '.') +
        ' Dates show the target week. Targets after 2025 are excluded from the replay.';
    }

    function show(chunk) {
      data = chunk;
      weeks = data.rows.filter(function (row) { return row[1] <= replayManifest.test_end; });
      var firstDrought = weeks.findIndex(function (row) { return row[2] && row[4] <= data.thresholds[active.severity]; });
      var firstScored = weeks.findIndex(function (row) { return row[2]; });
      index = firstDrought < 0 ? Math.max(0, firstScored) : firstDrought;
      weekSelect.innerHTML = weeks.map(function (row, i) {
        return '<option value="' + i + '">' + row[1] + (row[2] ? '' : ' · not scored') + '</option>';
      }).join('');
      updateStatus();
      el.hidden = false;
      el.setAttribute('aria-label', 'Observed and forecast discharge for ' + stationLabel() + ' at ' + leadLabel());
      var dates = weeks.map(function (row) { return row[1]; });
      var traces = ['observed'].concat(modelIds).map(function (name) {
        var model = COMPARISON_MODEL[name] || { name: 'Observed', color: C.obs };
        var col = columns.indexOf(name);
        return {
          x: dates, y: weeks.map(function (row) { return row[2] ? row[col] : null; }),
          name: model.name, mode: name === 'observed' ? 'lines+markers' : 'lines', connectgaps: false,
          marker: { size: 4, color: model.color },
          line: { color: model.color, width: name === 'observed' ? 2.4 : 1.7, dash: model.dash || 'solid' },
          hovertemplate: '<b>' + model.name + '</b> %{y:,.4f} m³/s<extra></extra>'
        };
      });
      var focus = range(true);
      return draw(el, traces, baseLayout({
        hovermode: 'x unified',
        xaxis: axis({ title: { text: 'Target week' }, type: 'date', range: focus['xaxis.range'] }),
        yaxis: axis({ title: { text: 'Discharge (m³/s)' }, range: focus['yaxis.range'] }),
        shapes: shapes()
      })).then(function () {
        if (data !== chunk) return;
        el.on('plotly_click', function (event) {
          var point = event.points[0];
          if (point) selectWeek(point.pointNumber);
        });
        renderWeek();
        panel.hidden = false;
      });
    }

    function load(selection) {
      active = selection;
      var token = ++request;
      data = null;
      weeks = null;
      panel.hidden = true;
      tbody.replaceChildren();
      weekStatus.textContent = '';
      weekSelect.replaceChildren();
      el.hidden = true;
      if (el.__drawn) Plotly.purge(el);
      el.__drawn = false;
      retry.hidden = true;
      status.classList.remove('chart-empty');
      status.textContent = 'Loading replay for ' + stationLabel() + ' at ' + leadLabel() + '…';
      fetchReplay(selection.station, selection.lead).then(function (chunk) {
        if (token === request) return show(chunk);
      }).catch(function () {
        if (token !== request) return;
        el.hidden = true;
        status.classList.add('chart-empty');
        status.textContent = 'Could not load replay for ' + stationLabel() + ' at ' + leadLabel() + '.';
        retry.hidden = false;
      });
    }

    function update(selection) {
      ctrl.set(selection);
      if (!active || active.station !== selection.station || active.lead !== selection.lead) {
        load(selection);
      } else if (active.objective !== selection.objective || active.severity !== selection.severity) {
        active = selection;
        if (data) {
          updateStatus();
          renderWeek();
          Plotly.relayout(el, { shapes: shapes() });
        }
      }
    }

    if (shared) shared.subscribe(update);
    else update(initial);
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
    render(controls(el, [{ key: 'lead', label: 'Pooled PR lead', value: '2w', options: [{ label: '2 weeks', value: '2w' }, { label: '4 weeks', value: '4w' }] }], render).state);
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
  function transfer(el, args, shared) {
    var F = window.FIGDATA.transfer;
    var currentLead;
    var range = scoreRange(Object.keys(F).reduce(function (values, lead) {
      var d = F[lead];
      return values.concat(d.m0, d.m1, d.m2);
    }, []));
    var labelM = { m0: 'M0 Baseline (own history)', m1: 'M1 Flow-only GBT (pooled)', m2: 'M2 HydroLSTM (pooled)' };
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
        xaxis: axis({ title: { text: 'Zero-shot NSE' }, range: range, zeroline: true, zerolinecolor: C.ink, zerolinewidth: 1 }),
        yaxis: axis({ showgrid: false, showline: false, autorange: 'reversed', tickfont: { family: FONT, size: 12.5, color: C.ink } })
      }));
    }
    var ctrl = controls(el, [{ key: 'lead', label: 'Transfer lead', value: shared ? shared.get().lead : '4w', options: LEADS }], function (state) {
      if (shared) shared.set(shared.get().station, state.lead);
      else render(state);
    });
    if (shared) shared.subscribe(function (selection) {
      if (currentLead === selection.lead) return;
      currentLead = selection.lead;
      ctrl.set(selection);
      render(ctrl.state);
    });
    else render(ctrl.state);
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
      xaxis: axis({ title: { text: 'Aridity index PET/P (1980–2020) — drier to the right' }, range: [0.6, 1.55] }),
      yaxis: axis({ title: { text: 'Catchment area (km², log scale)' }, type: 'log', range: [3.15, 4.95] })
    }));
  }

  /* ---------- Per-station scores across horizons (station explorer) ---------- */
  function stationMini(el, horizons, selection, selectLead) {
    var hs = ['1w', '2w', '4w', '8w'];
    var drought = selection.objective === 'drought';
    var metric = drought ? selection.severity + ' CSI' : 'NSE';
    var tr = Object.keys(COMPARISON_MODEL).map(function (m) {
      var model = COMPARISON_MODEL[m];
      return {
        x: hs.map(function (h) { return H_LABEL[h]; }),
        y: hs.map(function (h) { return drought ? horizons[h].drought[selection.severity].models[m].csi : horizons[h].flow[m].nse; }),
        name: model.name, mode: 'lines+markers',
        line: { color: model.color, width: m === 'M2' ? 2.6 : 1.8, dash: model.dash || 'solid' },
        marker: { size: hs.map(function (h) { return h === selection.lead ? 14 : 8; }), color: model.color, line: { color: '#fbfaf7', width: 1.5 } },
        hovertemplate: '<b>' + model.name + '</b> · %{x}: ' + metric + ' %{y:.3f}<extra></extra>'
      };
    });
    var plot = draw(el, tr, baseLayout({
      margin: { l: 46, r: 10, t: 8, b: 36 },
      xaxis: axis({ type: 'category' }),
      yaxis: axis({ title: { text: metric }, range: drought ? [-0.05, 1.05] : scoreRange(tr.reduce(function (values, trace) {
        return values.concat(trace.y);
      }, [])), zeroline: true, zerolinecolor: C.zero }),
      legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 11.5, color: C.ink } }
    }));
    if (!el.__selectLead) {
      el.__selectLead = true;
      plot.then(function () {
        el.on('plotly_click', function (event) {
          var point = event.points[0], lead = point && hs[point.pointNumber];
          if (lead) selectLead(lead);
        });
      });
    }
  }

  var REGISTRY = { skill: skill, hydro: hydro, pr: pr, kge: kge, fi: fi, transfer: transfer, portfolio: portfolio };

  function mount(el, shared) {
    if (el.__mounted) return;
    el.__mounted = true;
    var parts = el.getAttribute('data-chart').split(':');
    var fn = REGISTRY[parts[0]];
    if (fn && window.Plotly) fn(el, parts.slice(1), shared);
  }

  /* Charts mount lazily as they approach the viewport (eager on standalone pages). */
  function mountAll(root, opts) {
    var els = Array.prototype.slice.call((root || document).querySelectorAll('[data-chart]'));
    var shared = opts && opts.selection;
    if ((opts && opts.eager) || !('IntersectionObserver' in window)) { els.forEach(function (el) { mount(el, shared); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { mount(e.target, shared); io.unobserve(e.target); } });
    }, { rootMargin: '300px 0px' });
    els.forEach(function (el) { io.observe(el); });
  }

  window.Charts = { mountAll: mountAll, radioGroup: radioGroup, stationMini: stationMini, colors: C, models: MODEL, comparisonModels: COMPARISON_MODEL };
})();
