/* Local, dependency-free geographic catchment explorer. */
(function () {
  'use strict';
  var svg = document.getElementById('catchment-map-svg');
  var data = window.STATION_MAP_DATA;
  if (!svg || !data || !data.features.length) return;
  var NS = 'http://www.w3.org/2000/svg';
  var context = window.STATION_MAP_CONTEXT;
  var W = 760, H = 760, pad = 46, selected, preview = null, zoom = 1;
  var select = document.getElementById('catchment-station');
  var details = document.getElementById('catchment-selection');
  var colours = {Maharashtra: '#12856e', Odisha: '#3466c9', Karnataka: '#bb7414'};
  var regions = {}, pins = {}, stationLabels = {}, byId = {}, promoting = false;
  var features = data.features;
  function element(tag, attrs, parent, text) {
    var node = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (key) { node.setAttribute(key, attrs[key]); });
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function polygons(feature) {
    return feature.geometry.type === 'Polygon' ? [feature.geometry.coordinates] : feature.geometry.coordinates;
  }
  function mercator(lat) { return Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)) * 180 / Math.PI; }
  var points = [];
  features.forEach(function (feature) {
    byId[feature.id] = feature;
    polygons(feature).forEach(function (polygon) { polygon.forEach(function (ring) { points.push.apply(points, ring); }); });
    points.push(feature.point);
  });
  var regionalPoints = points.slice();
  if (context && context.country) {
    polygons({geometry: context.country}).forEach(function (polygon) {
      polygon.forEach(function (ring) { points.push.apply(points, ring); });
    });
  }
  var minX = Math.min.apply(null, points.map(function (p) { return p[0]; })) - .5;
  var maxX = Math.max.apply(null, points.map(function (p) { return p[0]; })) + .5;
  var minLat = Math.min.apply(null, points.map(function (p) { return p[1]; })) - .5;
  var maxLat = Math.max.apply(null, points.map(function (p) { return p[1]; })) + .5;
  var minY = mercator(minLat), maxY = mercator(maxLat);
  var scale = Math.min((W - pad * 2) / (maxX - minX), (H - pad * 2) / (maxY - minY));
  var offsetX = (W - (maxX - minX) * scale) / 2;
  var offsetY = (H - (maxY - minY) * scale) / 2;
  function project(p) { return [offsetX + (p[0] - minX) * scale, offsetY + (maxY - mercator(p[1])) * scale]; }
  var southWest = project([68, 6.5]), northEast = project([89.5, 24.2]);
  var defaultSize = Math.max(northEast[0] - southWest[0], southWest[1] - northEast[1]);
  var defaultCentre = [(southWest[0] + northEast[0]) / 2, northEast[1] + defaultSize / 2];
  var minimumZoom = W / defaultSize;
  var grid = element('g', {'aria-hidden': 'true'}, svg);
  for (var lon = Math.ceil(minX / 4) * 4; lon <= maxX; lon += 4) {
    var gx = project([lon, minLat])[0];
    element('line', {x1: gx, x2: gx, y1: 20, y2: H - 30, 'class': 'catchment-grid'}, grid);
    element('text', {x: gx, y: H - 10, 'text-anchor': 'middle', 'class': 'catchment-coordinate'}, grid, lon + '°E');
  }
  for (var lat = Math.ceil(minLat / 4) * 4; lat <= maxLat; lat += 4) {
    var gy = project([minX, lat])[1];
    element('line', {x1: 32, x2: W - 20, y1: gy, y2: gy, 'class': 'catchment-grid'}, grid);
    element('text', {x: 27, y: gy + 4, 'text-anchor': 'end', 'class': 'catchment-coordinate'}, grid, lat + '°N');
  }
  var contextLayer = element('g', {'aria-hidden': 'true', 'class': 'catchment-context'}, svg);
  var outlines = element('g', {}, svg);
  var markers = element('g', {}, svg);
  var labels = element('g', {'aria-hidden': 'true'}, svg);
  function pathData(feature) {
    return polygons(feature).map(function (polygon) {
      return polygon.map(function (ring) {
        return ring.map(function (p, i) {
          var xy = project(p);
          return (i ? 'L' : 'M') + xy[0].toFixed(2) + ',' + xy[1].toFixed(2);
        }).join('') + 'Z';
      }).join('');
    }).join('');
  }
  if (context && context.country) {
    element('path', {d: pathData({geometry: context.country}), 'class': 'catchment-india-land', 'fill-rule': 'evenodd'}, contextLayer);
    context.states.forEach(function (state) {
      element('path', {d: pathData(state), 'class': 'catchment-state-border', 'data-state': state.name, 'fill-rule': 'evenodd'}, contextLayer);
    });
    element('path', {d: pathData({geometry: context.country}), 'class': 'catchment-india-border', 'fill-rule': 'evenodd'}, contextLayer);
  }
  function highlight() {
    features.forEach(function (feature) {
      regions[feature.id].classList.toggle('is-selected', feature.id === selected);
      regions[feature.id].classList.toggle('is-preview', feature.id === preview);
      regions[feature.id].setAttribute('aria-pressed', String(feature.id === selected));
      pins[feature.id].classList.toggle('is-selected', feature.id === selected);
      pins[feature.id].classList.toggle('is-preview', feature.id === preview);
      pins[feature.id].setAttribute('aria-pressed', String(feature.id === selected));
      stationLabels[feature.id].classList.toggle('is-active', feature.id === selected || feature.id === preview);
      var screenScale = W / Math.max(1, svg.getBoundingClientRect().width) / zoom;
      stationLabels[feature.id].style.fontSize = ((feature.id === selected || feature.id === preview ? 14 : (svg.clientWidth < 450 ? 9 : 11)) * screenScale) + 'px';
    });
    var focusedPin = document.activeElement;
    promoting = true;
    [preview || selected].forEach(function (id) {
      if (id) {
        if (markers.lastElementChild !== pins[id]) markers.appendChild(pins[id]);
        if (labels.lastElementChild !== stationLabels[id]) labels.appendChild(stationLabels[id]);
      }
    });
    if (focusedPin && focusedPin.classList.contains('catchment-pin') && document.activeElement !== focusedPin) focusedPin.focus({preventScroll: true});
    promoting = false;
  }
  function setPreview(id) { preview = id; highlight(); }
  function choose(id) {
    selected = id; select.value = id; preview = null;
    var feature = byId[id];
    details.replaceChildren();
    var title = document.createElement('h4'); title.textContent = feature.name; details.appendChild(title);
    var state = document.createElement('p'); state.textContent = feature.state; details.appendChild(state);
    var dl = document.createElement('dl');
    [['Sub-basin / gauge context', feature.subbasin], ['CAMELS gauge', feature.gauge_id],
      ['Upstream area', feature.area_km2.toLocaleString('en-IN') + ' km²'],
      ['Gauge location', feature.point[1].toFixed(4) + '°N, ' + feature.point[0].toFixed(4) + '°E'],
      ['Benchmark role', feature.role]].forEach(function (item) {
      var row = document.createElement('div');
      var dt = document.createElement('dt'); dt.textContent = item[0];
      var dd = document.createElement('dd'); dd.textContent = item[1];
      row.append(dt, dd); dl.appendChild(row);
    });
    details.appendChild(dl);
    document.getElementById('catchment-explore').href = '?station=' + id + '&lead=1w&objective=flow&severity=Q95#explorer';
    highlight();
    if (zoom > minimumZoom + 1e-9) updateZoom(zoom);
  }
  features.slice().sort(function (a, b) { return b.area_km2 - a.area_km2; }).forEach(function (feature) {
    var path = element('path', {d: pathData(feature), 'fill-rule': 'evenodd', 'class': 'catchment-region',
      'data-station': feature.id, style: '--catchment-colour:' + colours[feature.state], tabindex: '0', role: 'button',
      'aria-label': feature.name + ', ' + feature.subbasin + ', select upstream catchment', 'aria-pressed': 'false'}, outlines);
    element('title', {}, path, feature.name + ' · ' + feature.subbasin);
    regions[feature.id] = path;
    path.addEventListener('pointerenter', function () { setPreview(feature.id); });
    path.addEventListener('pointerleave', function () { setPreview(null); });
    path.addEventListener('focus', function () { setPreview(feature.id); });
    path.addEventListener('blur', function () { setPreview(null); });
    path.addEventListener('click', function () { choose(feature.id); });
    path.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(feature.id); }
    });
    var xy = project(feature.point);
    var pin = element('g', {'class': 'catchment-pin', transform: 'translate(' + xy.join(' ') + ')',
      'data-station': feature.id, style: '--catchment-colour:' + colours[feature.state],
      tabindex: '0', role: 'button', 'aria-label': feature.name + ' gauge, select catchment', 'aria-pressed': 'false'}, markers);
    element('title', {}, pin, feature.name + ' gauge'); pins[feature.id] = pin;
    element('circle', {cx: 0, cy: -10, r: 20, 'class': 'catchment-pin-hit'}, pin);
    var artwork = element('g', {'class': 'catchment-pin-art'}, pin);
    element('path', {d: 'M0 0C-3-5-9-11-9-18A9 9 0 1 1 9-18C9-11 3-5 0 0Z', 'class': 'catchment-pin-shape'}, artwork);
    element('circle', {cx: 0, cy: -18, r: 3.1, 'class': 'catchment-pin-centre'}, artwork);
    pin.addEventListener('pointerenter', function () { setPreview(feature.id); });
    pin.addEventListener('pointerleave', function (event) {
      requestAnimationFrame(function () {
        if (!pin.contains(document.elementFromPoint(event.clientX, event.clientY))) setPreview(null);
      });
    });
    pin.addEventListener('focus', function () { if (!promoting) setPreview(feature.id); });
    pin.addEventListener('blur', function () { if (!promoting) setPreview(null); });
    pin.addEventListener('click', function () { choose(feature.id); });
    pin.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(feature.id); }
    });
    var offsets = {karad: [12, 10], warunji: [-12, -20], huvinhedigi: [12, 14], yadgir: [12, -20],
                   bamni: [-12, -20], ashti: [12, 10], kantamal: [-12, 16], gomlai: [-12, -22],
                   anandpur: [12, 0], kashinagar: [12, 12], kudige: [-12, 14], haralahalli: [12, -20]};
    var off = offsets[feature.id] || [9, -8];
    stationLabels[feature.id] = element('text', {x: xy[0] + off[0], y: xy[1] + off[1], 'text-anchor': off[0] < 0 ? 'end' : 'start',
      'class': 'catchment-label', 'data-station': feature.id, 'data-offset-x': off[0], 'data-offset-y': off[1],
      'vector-effect': 'non-scaling-stroke'}, labels, feature.name === 'KUDIGE' ? 'Kudige' : feature.name);
  });
  features.forEach(function (feature) {
    var option = document.createElement('option'); option.value = feature.id;
    option.textContent = feature.name + ' — ' + feature.state; select.appendChild(option);
  });
  select.addEventListener('change', function () { choose(select.value); });
  var zoomIn = document.getElementById('catchment-zoom-in');
  var zoomOut = document.getElementById('catchment-zoom-out');
  function updateZoom(next, centre) {
    zoom = Math.max(minimumZoom, Math.min(6, next));
    centre = centre || (Math.abs(zoom - minimumZoom) < 1e-9 ? defaultCentre : project(byId[selected].point));
    var width = W / zoom, height = H / zoom;
    var x = Math.max(0, Math.min(W - width, centre[0] - width / 2));
    var y = Math.max(0, Math.min(H - height, centre[1] - height / 2));
    if (Math.abs(zoom - minimumZoom) < 1e-9) {
      x = defaultCentre[0] - width / 2; y = defaultCentre[1] - height / 2;
    }
    svg.setAttribute('viewBox', [x, y, width, height].join(' '));
    updateSymbols();
    zoomIn.disabled = zoom >= 6; zoomOut.disabled = zoom <= minimumZoom + 1e-9;
  }
  function updateSymbols() {
    var screenScale = W / Math.max(1, svg.getBoundingClientRect().width) / zoom;
    features.forEach(function (feature) {
      var xy = project(feature.point), label = stationLabels[feature.id];
      pins[feature.id].setAttribute('transform', 'translate(' + xy.join(' ') + ') scale(' + (.7 * screenScale) + ')');
      label.setAttribute('x', xy[0] + Number(label.dataset.offsetX) * screenScale);
      label.setAttribute('y', xy[1] + Number(label.dataset.offsetY) * screenScale);
    });
    highlight();
  }
  zoomIn.addEventListener('click', function () { updateZoom(zoom * 1.5); });
  zoomOut.addEventListener('click', function () { updateZoom(zoom / 1.5); });
  document.getElementById('catchment-fit').addEventListener('click', function () {
    var projected = regionalPoints.map(project);
    var xs = projected.map(function (p) { return p[0]; });
    var ys = projected.map(function (p) { return p[1]; });
    var left = Math.min.apply(null, xs), right = Math.max.apply(null, xs);
    var top = Math.min.apply(null, ys), bottom = Math.max.apply(null, ys);
    updateZoom(Math.min(W / (right - left + 70), H / (bottom - top + 70)), [(left + right) / 2, (top + bottom) / 2]);
  });
  document.getElementById('catchment-reset').addEventListener('click', function () { updateZoom(minimumZoom, defaultCentre); });
  choose(features[0].id);
  updateZoom(minimumZoom, defaultCentre);
  if (window.ResizeObserver) new ResizeObserver(updateSymbols).observe(svg);
}());
