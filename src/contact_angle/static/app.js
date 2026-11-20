import {constrainView, zoomView, fitImageView, actualImageView} from './viewport.js';
/* Contact Angle Studio. Every editable coordinate is an original-image pixel. */
const SVG_NS = 'http://www.w3.org/2000/svg';
const COLORS = {left: '#d36c39', right: '#168c80', baseline: '#3c7cdb', contour: '#8662b0', both: '#96a6ba'};
const DEFAULT_OPTIONS = {mode: 'auto', algorithm: 'general', orientation: 'auto', baseline: null, baseline_points: [], baseline_source: 'automatic', baseline_method: null, contour_points: [], left_points: [], right_points: [], manual_method: 'spline', sensitivity: true, polarity: 'dark', roi: null, excluded_regions: []};
const GROUP_NAMES = {baseline: '表面', contour: '轮廓'};
const TOOL_HINTS = {pan: '', baseline: '在液滴两侧的固体表面边缘选点', contour: '从一侧接触区沿液滴轮廓到另一侧，至少 7 点', roi: '框选完整液滴及接触线，至少 24 × 24 px'};
const state = {images: [], caches: new Map(), currentId: null, selectionEpoch: 0, tool: 'pan', pointTab: 'baseline', selectedPoint: null, view: null, manualScale: null, drag: null, space: false, uploading: false, loadController: null, pollController: null, toastTimer: null, draftTimer: null, maxUpload: 20 * 1024 * 1024};
const $ = (id) => document.getElementById(id);
const clone = (value) => JSON.parse(JSON.stringify(value));
const active = () => state.caches.get(state.currentId);
const activeImage = () => state.images.find((image) => image.id === state.currentId);
const isCurrent = (id, epoch) => state.currentId === id && (epoch === undefined || state.selectionEpoch === epoch);
const round = (value, places = 2) => Number(Number(value).toFixed(places));
const format = (value, places = 2) => Number.isFinite(Number(value)) && value !== null ? Number(value).toFixed(places) : '—';
const point = (p) => ({x: Number(p.x), y: Number(p.y)});
export function stableJSON(value) { if (Array.isArray(value)) return '[' + value.map(stableJSON).join(',') + ']'; if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map((k) => JSON.stringify(k) + ':' + stableJSON(value[k])).join(',') + '}'; return JSON.stringify(value); }
const optionSignature = (options) => stableJSON(payload(options));
const pointsFor = (options, group) => group === 'baseline' ? options.baseline_points || [] : options.contour_points || [];
const median = (values) => { const a = [...values].sort((x,y)=>x-y); return a.length ? (a[Math.floor((a.length-1)/2)] + a[Math.floor(a.length/2)])/2 : 0; };
export function fittedSurfaceLine(points) {
  if (points.length < 2) return null;
  const slopes = [];
  for (let i=0;i<points.length;i++) for (let j=i+1;j<points.length;j++) if (Math.abs(points[j].x-points[i].x)>=5) slopes.push((points[j].y-points[i].y)/(points[j].x-points[i].x));
  if (!slopes.length) return null;
  let slope=median(slopes), intercept=median(points.map(p=>p.y-slope*p.x));
  for (let k=0;k<5;k++) {
    const residual=points.map(p=>p.y-slope*p.x-intercept), scale=Math.max(.25,1.4826*median(residual.map(Math.abs)));
    const weights=residual.map(r=>Math.min(1,1.5*scale/Math.max(Math.abs(r),1e-9))), total=weights.reduce((a,b)=>a+b,0);
    const x=points.reduce((a,p,i)=>a+weights[i]*p.x,0)/total, y=points.reduce((a,p,i)=>a+weights[i]*p.y,0)/total;
    const variance=points.reduce((a,p,i)=>a+weights[i]*(p.x-x)**2,0);
    if (variance<1e-8) return null;
    slope=points.reduce((a,p,i)=>a+weights[i]*(p.x-x)*(p.y-y),0)/variance; intercept=y-slope*x;
  }
  if (Math.abs(slope)>1) return null;
  const xs=points.map(p=>p.x), low=Math.min(...xs), high=Math.max(...xs);
  return [{x:low,y:intercept+slope*low},{x:high,y:intercept+slope*high}];
}
function syncBaseline(options) { options.baseline = fittedSurfaceLine(options.baseline_points || []); }
function setPoints(options, group, points) {
  if (group === 'baseline') { options.baseline_points=points; options.baseline_method=null; syncBaseline(options); }
  else { options.contour_points=points; options.manual_method='spline'; options.left_points=[]; options.right_points=[]; }
}
function loadedOptions(input = {}) {
  const options={...clone(DEFAULT_OPTIONS),...clone(input)};
  if (!options.baseline_points?.length && options.baseline?.length===2) options.baseline_points=options.baseline.map(point);
  if (!options.contour_points?.length && (options.left_points?.length || options.right_points?.length)) options.contour_points=[...options.left_points].sort((a,b)=>b.y-a.y).concat([...options.right_points].sort((a,b)=>a.y-b.y));
  return options;
}
function setText(id, value) { $(id).textContent = value; }
function visible(id, show) { $(id).toggleAttribute('hidden', !show); }
function element(tag, classes, text) { const e = document.createElement(tag); if (classes) e.className = classes; if (text !== undefined) e.textContent = text; return e; }
function svg(tag, attributes = {}) { const e = document.createElementNS(SVG_NS, tag); for (const [key, value] of Object.entries(attributes)) e.setAttribute(key, String(value)); return e; }
function freshCache() { return {options: clone(DEFAULT_OPTIONS), undo: [], redo: [], loaded: false, estimating: false, history: [], result: null, resultJobId: null, pendingJob: null, dirty: false, editVersion: 0, saveVersion: 0, saveChain: Promise.resolve(), saved: false, error: null}; }
function notify(message, error = false) { clearTimeout(state.toastTimer); setText('toast', message); $('toast').classList.toggle('error', error); visible('toast', true); state.toastTimer = setTimeout(() => visible('toast', false), error ? 7000 : 4200); }
function friendly(message) {
  const text = String(message || '操作未完成，请重试。');
  const translations = [
    [/Image exceeds|20 MiB|20 MiB upload/i, '图片超过大小限制，请使用 20 MB 以内的单张图片。'],
    [/12 million|12 million pixels/i, '图片超过 1200 万像素，请等比例缩小后重试。'],
    [/Cannot decode|invalid image|cannot identify image/i, '无法读取这张图片，请使用有效的 PNG、JPG、TIFF、BMP 或 WebP。'],
    [/Normalize image orientation/i, '图片包含旋转方向信息，请先保存为方向正常的图片，再导入打点。'],
    [/No reliable full side|No separated visible|No visible contour/i, '未识别到可靠的液滴轮廓。请核对基线、框选区域，或改用手动打点。'],
    [/does not intersect|does not meet|No converged|no intersection/i, '拟合曲线未能可靠到达基线。请检查基线和接触区打点。'],
    [/Too much of the contact|excessive extrapolation/i, '接触区遮挡或外推距离过大，请补充靠近接触区的可见边缘点。'],
    [/Failed to fetch|NetworkError|Load failed/i, '无法连接本机服务，请从桌面重新打开接触角工作台。']
  ];
  return translations.find(([pattern]) => pattern.test(text))?.[1] || text;
}
async function request(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = null; }
  if (!response.ok) {
    let detail = data?.detail || data?.error || data?.message;
    if (Array.isArray(detail)) detail = detail.map((e) => (e.loc || []).join('.') + ': ' + e.msg).join('；');
    const message = typeof detail === 'string' ? detail : detail?.message || ('请求失败（' + response.status + '）');
    const error = new Error(friendly(message)); error.code = detail?.code; throw error;
  }
  if (data === null) throw new Error('服务返回了无法读取的内容，请刷新后重试。');
  return data;
}
const jsonRequest = (method, body, signal) => ({method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal});
function payload(options) { const p = clone(options); p.orientation ||= 'auto'; if (p.baseline?.length !== 2) p.baseline = null; return p; }
function pushUndo(cache) {
  const before = JSON.stringify(cache.options);
  if (cache.undo.at(-1) !== before) cache.undo.push(before);
  if (cache.undo.length > 80) cache.undo.shift();
  cache.redo = [];
}
function invalidate(cache) {
  cache.result = null; cache.resultJobId = null; cache.error = null; cache.dirty = true; cache.saved = false; cache.editVersion += 1;
}
function edit(mutator, options = {}) {
  const cache = active(); if (!cache?.loaded) return;
  pushUndo(cache); mutator(cache.options); invalidate(cache);
  renderAll(); scheduleDraft();
  if (options.focusCanvas) $('image-canvas').focus({preventScroll: true});
}
function scheduleDraft() { clearTimeout(state.draftTimer); const id = state.currentId; state.draftTimer = setTimeout(() => saveDraft(id), 500); }
async function saveDraft(id = state.currentId) {
  const cache = state.caches.get(id); if (!cache || !cache.loaded) return;
  if (pointsFor(cache.options,'baseline').length === 1) { if (isCurrent(id)) renderDraftStatus(); return; }
  const version = ++cache.saveVersion, savedOptions = payload(cache.options);
  cache.saveChain = cache.saveChain.catch(() => {}).then(async () => {
    if (version !== cache.saveVersion) return;
    try {
      await request('/api/images/' + encodeURIComponent(id) + '/draft', jsonRequest('PUT', savedOptions));
      if (version === cache.saveVersion) { cache.saved = true; if (isCurrent(id)) renderDraftStatus(); }
    } catch (error) { if (version === cache.saveVersion && isCurrent(id)) { setText('draft-status', '草稿保存失败，可重试或导出坐标'); $('draft-status').classList.add('warning'); } }
  });
  return cache.saveChain;
}
function renderDraftStatus() {
  const cache = active();
  $('draft-status').classList.toggle('warning', Boolean((cache && pointsFor(cache.options,'baseline').length === 1) || (cache && !cache.loaded && cache.error)));
  setText('draft-status', !cache ? '' : !cache.loaded ? cache.error ? '草稿未读取，请刷新图片库重试' : '正在读取草稿…' : pointsFor(cache.options,'baseline').length === 1 ? '基线还需一个点' : cache.pendingJob ? '' : cache.saved ? '已保存' : '待保存');
  visible('draft-status', Boolean($('draft-status').textContent));
}
function undo(redo = false) {
  const cache = active(); if (!cache?.loaded) return;
  const from = redo ? cache.redo : cache.undo, to = redo ? cache.undo : cache.redo;
  if (!from.length) return;
  to.push(JSON.stringify(cache.options)); cache.options = JSON.parse(from.pop()); invalidate(cache);
  state.selectedPoint = null; renderAll(); scheduleDraft();
}
async function refreshImages(preferId) {
  const data = await request('/api/images');
  state.images = data.images || [];
  renderLibrary();
  if (preferId) await selectImage(preferId);
  else if (!activeImage() && state.images.length) await selectImage(state.images[0].id);
  else if (activeImage() && !active()?.loaded) await selectImage(state.currentId);
  else if (!state.images.length) { state.currentId = null; renderAll(); }
}
function renderLibrary() {
  const list = $('image-list'); list.replaceChildren(); setText('image-count', state.images.length);
  const query = $('image-search').value.trim().toLocaleLowerCase();
  const images = state.images.filter((image) => image.filename.toLocaleLowerCase().includes(query));
  if (!images.length) { list.append(element('p', 'library-empty', query ? '没有匹配图片' : '暂无图片')); return; }
  for (const image of images) {
    const cache = state.caches.get(image.id), card = element('button', 'image-card' + (image.id === state.currentId ? ' active' : ''));
    card.type = 'button'; card.setAttribute('aria-label', '打开图片 ' + image.filename); card.setAttribute('aria-pressed', String(image.id === state.currentId)); card.title = image.filename + '（' + image.width + ' × ' + image.height + ' px）';
    const thumbnail = element('img'); thumbnail.src = image.preview_url || '/api/images/' + encodeURIComponent(image.id) + '/preview.png'; thumbnail.alt = ''; thumbnail.loading = 'lazy'; card.append(thumbnail);
    card.append(element('span', 'image-card-name', image.filename));
    const last = cache?.pendingJob ? '拟合中' : cache?.result || cache?.history.some((j) => j.state === 'done') ? '已测量' : '';
    const meta = element('span', 'image-card-meta');
    meta.append(element('span', '', image.width + ' × ' + image.height));
    if (last) meta.append(element('span', 'image-state', last));
    card.append(meta);
    card.addEventListener('click', () => selectImage(image.id)); list.append(card);
  }
}
async function selectImage(id) {
  if (state.currentId === id && active()?.loaded) return;
  const oldId = state.currentId; clearTimeout(state.draftTimer); if (oldId) saveDraft(oldId);
  state.loadController?.abort(); state.pollController?.abort();
  const controller = new AbortController(); state.loadController = controller;
  state.currentId = id; state.selectionEpoch += 1; const epoch = state.selectionEpoch;
  state.selectedPoint = null; state.drag = null; state.view = null; state.tool = 'pan'; state.pointTab = 'baseline';
  if (!state.caches.has(id)) state.caches.set(id, freshCache());
  const cache = state.caches.get(id);
  if (!cache.loaded) cache.error = null;
  renderAll(); fitView();
  try {
    if (!cache.loaded) {
      const results = await Promise.allSettled([
        request('/api/images/' + encodeURIComponent(id) + '/draft', {signal: controller.signal}),
        request('/api/images/' + encodeURIComponent(id) + '/results', {signal: controller.signal})
      ]);
      if (!isCurrent(id, epoch)) return;
      if (results[1].status === 'fulfilled') cache.history = results[1].value.results || [];
      if (results[0].status !== 'fulfilled') {
        cache.loaded = false; cache.saved = false;
        cache.error = '无法读取已保存的打点草稿，已暂停编辑、自动识别和保存。请刷新图片库或重新选择本图重试。' + friendly(results[0].reason?.message);
        renderAll(); notify('草稿读取失败，原有打点没有被覆盖。请刷新图片库重试。', true); return;
      }
      const draft = results[0].value.options;
      if (draft !== null && (typeof draft !== 'object' || Array.isArray(draft))) throw new Error('草稿格式无法读取，已暂停自动保存。请重新选择本图重试。');
      if (draft && cache.editVersion === 0) {
        cache.options = loadedOptions(draft);
        cache.needsBaselineRefresh = draft.baseline_source === 'automatic' && (
          !draft.orientation || !draft.baseline_points?.length ||
          !['dense_substrate_v1', 'pendant_surface_v1'].includes(draft.baseline_method)
        );
      }
      cache.error = null;
      cache.loaded = true; cache.saved = cache.editVersion === 0;
      if (cache.editVersion > 0) scheduleDraft();
      const latest = cache.history.find((job) => job.state === 'done' && job.result && optionSignature(job.options) === optionSignature(cache.options));
      if (latest) { cache.result = latest.result; cache.resultJobId = latest.id; cache.dirty = false; }
    } else {
      refreshHistory(id, epoch).catch(() => {});
    }
    if (!isCurrent(id, epoch)) return;
    renderAll();
    if (cache.options.baseline_source === 'automatic' && (!cache.options.baseline || cache.needsBaselineRefresh) && !cache.pendingJob) { cache.needsBaselineRefresh=false; estimateBaseline(false); }
    if (cache.pendingJob?.id) pollJob(cache.pendingJob.id, id, cache.pendingJob.snapshot, epoch);
  } catch (error) { if (error.name !== 'AbortError' && isCurrent(id, epoch)) { cache.error = friendly(error.message); renderAll(); notify(cache.error, true); } }
}
async function uploadFiles(files) {
  if (!files.length || state.uploading) return;
  state.uploading = true; $('upload-button').disabled = true; $('demo-button').disabled = true;
  let imported = 0, firstId = null, failures = [];
  visible('upload-status', true);
  for (let i = 0; i < files.length; i += 1) {
    const file = files[i]; setText('upload-status', '正在导入 ' + (i + 1) + '/' + files.length + '：' + file.name);
    if (file.size > state.maxUpload) { failures.push(file.name + '：超过大小限制'); continue; }
    const body = new FormData(); body.append('image', file, file.name);
    try { const image = await request('/api/images', {method: 'POST', body}); firstId ||= image.id; imported += 1; }
    catch (error) { failures.push(file.name + '：' + error.message); }
  }
  state.uploading = false; $('upload-button').disabled = false; $('demo-button').disabled = false;
  $('file-input').value = '';
  visible('upload-status', failures.length > 0);
  if (failures.length) setText('upload-status', '已导入 ' + imported + ' 张。' + failures.slice(0, 3).join('；') + (failures.length > 3 ? '；另有 ' + (failures.length - 3) + ' 张失败。' : ''));
  if (imported) { await refreshImages(firstId); notify('已导入 ' + imported + ' 张图片'); }
  else notify(failures[0] || '没有导入图片', true);
}
async function openDemo() {
  $('demo-button').disabled = true; $('empty-demo').disabled = true;
  try { const data = await request('/api/demo'); const images = data.images || []; await refreshImages(images[0]?.id); }
  catch (error) { notify(error.message, true); }
  finally { $('demo-button').disabled = false; $('empty-demo').disabled = false; }
}
function surfaceKey(o) {
  return JSON.stringify([o.orientation, o.polarity, o.algorithm, o.baseline_source, o.baseline_points]);
}
function isPendant(cache) {
  const o = cache.options;
  if (o.orientation !== 'auto') return o.orientation === 'pendant';
  if (cache.result?.orientation) return cache.result.orientation === 'pendant';
  if (o.baseline_method) return o.baseline_method === 'pendant_surface_v1';
  return cache.direction?.polarity === o.polarity && cache.direction.orientation === 'pendant';
}
function rememberSurface(cache, data, key) {
  cache.preview = {key, curve: data.surface_curve || []};
  cache.direction = {polarity: cache.options.polarity, orientation: data.orientation ||
    (data.method === 'pendant_surface_v1' ? 'pendant' : 'sessile')};
}
function baselineURL(id, o) {
  const params = new URLSearchParams({polarity: o.polarity, algorithm: o.algorithm, orientation: o.orientation});
  return '/api/images/' + encodeURIComponent(id) + '/baseline?' + params;
}
function queueSurfacePreview(cache) {
  const o = cache.options;
  const manual = o.baseline_source === 'manual';
  if (!o.baseline || cache.result || cache.needsBaselineRefresh || o.orientation === 'sessile' ||
      (!manual && o.baseline_method !== 'pendant_surface_v1')) return;
  const id = state.currentId, key = surfaceKey(o);
  if (cache.preview?.key === key || cache.requestedPreview === key) return;
  clearTimeout(cache.previewTimer);
  cache.requestedPreview = key;
  cache.previewTimer = setTimeout(async () => {
    if (key !== surfaceKey(cache.options)) { if (cache.requestedPreview === key) cache.requestedPreview = null; return; }
    try {
      const data = manual ?
        await request('/api/images/' + encodeURIComponent(id) + '/baseline', jsonRequest('POST', payload(o))) :
        await request(baselineURL(id, o));
      if (key !== surfaceKey(cache.options)) return;
      rememberSurface(cache, data, key);
      if (isCurrent(id)) renderAll();
    } catch (error) {
      if (key === surfaceKey(cache.options) && isCurrent(id)) setText('baseline-state', error.message);
    } finally {
      if (cache.requestedPreview === key) cache.requestedPreview = null;
    }
  }, 120);
}
async function estimateBaseline(announce = true) {
  const id = state.currentId, cache = active(); if (!cache?.loaded) return;
  const requestId = Symbol(); cache.estimateId = requestId;
  const version = cache.editVersion, epoch = state.selectionEpoch; cache.estimating = true; $('estimate-baseline').disabled = true; setText('estimate-baseline','识别中…'); renderRun();
  try {
    const data = await request(baselineURL(id, cache.options));
    if (!isCurrent(id, epoch) || cache.editVersion !== version || cache.estimateId !== requestId) return;
    if (!Array.isArray(data.baseline) || data.baseline.length !== 2) throw new Error('未能识别基线，请在图上手动选择两个点。');
    pushUndo(cache); cache.options.baseline = data.baseline.map(point); cache.options.baseline_points=(data.baseline_points || data.baseline).map(point); cache.options.baseline_source = 'automatic'; cache.options.baseline_method=data.method || 'dense_substrate_v1';
    rememberSurface(cache, data, surfaceKey(cache.options));
    invalidate(cache);
    renderAll(); scheduleDraft();
  } catch (error) { if (isCurrent(id, epoch) && cache.estimateId === requestId) { setText('baseline-state', '请手动设置'); if (announce) notify(error.message, true); } }
  finally {
    if (cache.estimateId === requestId) {
      cache.estimating = false;
      if (isCurrent(id, epoch)) {
        $('estimate-baseline').disabled = false;
        setText('estimate-baseline','自动识别');
        renderRun(); renderStage();
      }
    }
  }
}
function setTool(tool) {
  if (!active()?.loaded) return;
  if (tool === 'contour' && active().options.mode !== 'manual') edit((o) => { o.mode='manual'; o.manual_method='spline'; });
  if (tool === 'baseline') {
    const cache = active(), measured = cache.result?.baseline_points;
    if (measured?.length && cache.options.baseline_source === 'automatic' &&
        stableJSON(measured) !== stableJSON(cache.options.baseline_points)) {
      edit(o => { setPoints(o, 'baseline', measured.map(point)); o.baseline_source = 'manual'; });
    }
  }
  state.tool = tool; state.selectedPoint = null;
  if (tool === 'baseline' || tool === 'contour') state.pointTab = tool;
  if (tool !== 'pan') $('overlay-toggle').checked = true;
  renderTools(); renderPoints(); renderStage(); $('image-canvas').focus({preventScroll: true});
}
function renderTools() {
  const cache = active();
  document.querySelectorAll('button[data-tool]').forEach((b) => { b.classList.toggle('active', b.dataset.tool === state.tool); b.setAttribute('aria-pressed', String(b.dataset.tool === state.tool)); b.disabled = !cache?.loaded; });
  $('image-canvas').dataset.tool = state.space ? 'pan' : state.tool;
  setText('tool-hint', TOOL_HINTS[state.tool]);
  visible('tool-hint', Boolean(TOOL_HINTS[state.tool]));
  $('undo-button').disabled = !cache?.undo.length; $('redo-button').disabled = !cache?.redo.length;
}
function renderAll() {
  const image = activeImage(), cache = active();
  if (cache?.options.mode === 'auto' && state.tool === 'contour') {
    state.tool = 'pan';
    if (state.selectedPoint?.group !== 'baseline') state.selectedPoint = null;
  }
  renderLibrary();
  visible('image-canvas', Boolean(image)); visible('empty-stage', !image);
  setText('current-filename', image?.filename || '接触角测量');
  setText('current-metadata', image ? image.width + ' × ' + image.height + ' px' : '');
  visible('current-metadata', Boolean(image)); visible('stage-status', Boolean(image));
  for (const id of ['estimate-baseline','draw-baseline','mode-auto','mode-manual','algorithm-select','orientation-select','polarity-select','sensitivity-toggle','import-coordinates','export-coordinates','zoom-in','zoom-out','fit-view','actual-view','clear-roi','clear-exclusions']) $(id).disabled = !cache?.loaded;
  $('refresh-history').disabled = !cache;
  $('estimate-baseline').disabled = !cache?.loaded || cache.estimating;
  setText('estimate-baseline',cache?.estimating?'识别中…':'自动识别');
  if (image) {
    const source = $('source-image'), href = image.preview_url || '/api/images/' + encodeURIComponent(image.id) + '/preview.png';
    if (source.getAttribute('href') !== href) source.setAttribute('href', href);
    source.setAttribute('width', image.width); source.setAttribute('height', image.height);
    $('photo-background').setAttribute('width', image.width); $('photo-background').setAttribute('height', image.height);
    const o = cache.options, manual = o.mode === 'manual';
    $('mode-auto').classList.toggle('active', !manual); $('mode-manual').classList.toggle('active', manual);
    $('mode-auto').setAttribute('aria-selected', String(!manual)); $('mode-manual').setAttribute('aria-selected', String(manual));
    visible('auto-settings', !manual); visible('manual-settings', manual); visible('algorithm-field', !isPendant(cache));
    $('orientation-select').value = o.orientation;
    const autoLabel = $('orientation-select').options[0];
    autoLabel.textContent = o.orientation === 'auto' && (cache.direction || o.baseline_method) ?
      (isPendant(cache) ? '自动识别 · 悬滴' : '自动识别 · 座滴') : '自动识别';
    $('algorithm-select').value = o.algorithm; $('polarity-select').value = o.polarity; $('sensitivity-toggle').checked = o.sensitivity;
    const hasBase = o.baseline?.length === 2;
    setText('baseline-state', hasBase ? o.baseline_source === 'automatic' ? '自动' : '手动' : '未完成');
    $('baseline-state').classList.toggle('ready', hasBase);
    visible('roi-summary', Boolean(o.roi));
    if (o.roi) setText('roi-description', '区域 ' + o.roi.width + ' × ' + o.roi.height + ' px');
    const exclusions = o.excluded_regions || [], applyingExclusions = o.mode === 'auto' && (o.algorithm === 'local' || isPendant(cache));
    visible('exclusion-summary', exclusions.length > 0);
    $('exclusion-summary').classList.toggle('warning', !applyingExclusions);
    setText('exclusion-description', applyingExclusions ? exclusions.length + ' 个忽略区域' : '忽略区域仅适用于自动局部拟合；手动测量请按选点绕开遮挡。');
  } else { setText('baseline-state', '未设置'); $('baseline-state').classList.remove('ready'); visible('roi-summary', false); visible('exclusion-summary', false); }
  renderTools(); renderPoints(); renderStage(); renderResults(); renderHistory(); renderRun(); renderDraftStatus();
  if (cache?.loaded) queueSurfacePreview(cache);
}
function renderPoints() {
  const cache = active(), options = cache?.options;
  for (const group of ['baseline','contour']) setText(group + '-count', options ? pointsFor(options, group).length : 0);
  document.querySelectorAll('[data-point-tab]').forEach((b) => { b.classList.toggle('active', b.dataset.pointTab === state.pointTab); b.setAttribute('aria-selected', String(b.dataset.pointTab === state.pointTab)); });
  const table = $('point-table-body'); table.replaceChildren();
  const points = options ? pointsFor(options, state.pointTab) : [], image = activeImage();
  visible('points-empty', !points.length); $('clear-points').disabled = !points.length || !cache?.loaded;
  setText('points-empty', state.pointTab === 'baseline' ? '在液滴两侧表面边缘选点。' : '请标记' + GROUP_NAMES[state.pointTab] + '边缘点。');
  points.forEach((p, i) => {
    const row = element('tr'); if (state.selectedPoint?.group === state.pointTab && state.selectedPoint?.index === i) row.classList.add('selected');
    const indexCell = element('td', '', String(i + 1)); row.append(indexCell);
    for (const axis of ['x','y']) {
      const cell = element('td'), input = element('input'); input.type = 'number'; input.step = '0.1'; input.min = '0'; input.max = String((axis === 'x' ? image.width : image.height) - 1); input.value = String(round(p[axis], 3)); input.disabled = !cache.loaded;
      input.setAttribute('aria-label', GROUP_NAMES[state.pointTab] + '第' + (i + 1) + '点' + axis.toUpperCase() + '坐标');
      const group = state.pointTab;
      input.addEventListener('focus', () => { state.selectedPoint = {group, index: i}; });
      input.addEventListener('change', () => {
        const number = Number(input.value);
        if (input.value.trim() === '' || !Number.isFinite(number) || number < 0 || number >= (axis === 'x' ? image.width : image.height)) { input.value = String(round(p[axis], 3)); notify('坐标需要位于原图范围内', true); return; }
        edit((o) => { const points=pointsFor(o,group); points[i][axis]=number; setPoints(o,group,points); if (group==='baseline') o.baseline_source='manual'; });
      });
      cell.append(input); row.append(cell);
    }
    const remove = element('button', '', '×'); remove.type = 'button'; remove.setAttribute('aria-label', '删除' + GROUP_NAMES[state.pointTab] + '第' + (i + 1) + '点');
    remove.disabled = !cache.loaded;
    const group = state.pointTab; remove.addEventListener('click', () => removePoint(group, i));
    const cell = element('td'); cell.append(remove); row.append(cell); table.append(row);
  });
}
function removePoint(group, index) {
  edit((o) => { const points = pointsFor(o, group); points.splice(index, 1); setPoints(o, group, points); if (group === 'baseline') o.baseline_source = 'manual'; });
  state.selectedPoint = null; renderPoints();
}
function screenToImage(clientX, clientY) {
  const canvas = $('image-canvas'), matrix = canvas.getScreenCTM();
  if (!matrix) return {x: 0, y: 0};
  const p = canvas.createSVGPoint(); p.x = clientX; p.y = clientY;
  const transformed = p.matrixTransform(matrix.inverse()); return {x: transformed.x, y: transformed.y};
}
function insideImage(p) { const im = activeImage(); return Boolean(im && p.x >= 0 && p.y >= 0 && p.x < im.width && p.y < im.height); }
function boundedPoint(p) { const im = activeImage(); return {x: round(Math.max(0, Math.min(im.width - 1, p.x)), 3), y: round(Math.max(0, Math.min(im.height - 1, p.y)), 3)}; }
function stageSize() { const canvas=$('image-canvas'), stage=$('image-stage'); return {width:Math.max(1,canvas.clientWidth || stage.clientWidth),height:Math.max(1,canvas.clientHeight || stage.clientHeight)}; }
function fitView() {
  const image=activeImage(); if (!image) return;
  state.manualScale = null;
  state.view=fitImageView(image,stageSize()); applyView(); renderStage();
}
function actualView() {
  const image=activeImage(); if (!image) return;
  state.manualScale = 1;
  state.view=actualImageView(image,stageSize()); applyView(); renderStage();
}
function applyView() {
  if (!state.view || !activeImage()) { setText('zoom-value','—'); return; }
  state.view=constrainView(state.view,activeImage(),stageSize());
  const v=state.view; $('image-canvas').setAttribute('viewBox',[v.x,v.y,v.w,v.h].join(' '));
  const scale=stageSize().width/v.w;
  setText('zoom-value',Math.round(scale*100)+'%');
  const availablePixels = Math.min(activeImage().width, 64 / scale);
  const magnitude = 10 ** Math.floor(Math.log10(availablePixels));
  const barPixels = [5, 2, 1].find(step => step * magnitude <= availablePixels) * magnitude;
  $('scale-line').style.width = (barPixels * scale) + 'px';
  setText('scale-value', barPixels + ' px');
  $('zoom-out').disabled=scale<=.250001; $('zoom-in').disabled=scale>=7.99999;
}
function zoom(factor, anchor) {
  if (!activeImage() || !state.view) return;
  const stage = stageSize();
  state.view=zoomView(state.view,factor,anchor,activeImage(),stage);
  state.manualScale = stage.width / state.view.w;
  applyView(); renderStage();
}
function pathData(points) { return points.filter((p) => Number.isFinite(Number(p.x)) && Number.isFinite(Number(p.y))).map((p,i) => (i ? 'L' : 'M') + Number(p.x).toFixed(3) + ',' + Number(p.y).toFixed(3)).join(' '); }
function exposedSurface(curve, contactX, left) {
  const next = curve.findIndex(p => p.x >= contactX);
  if (next === -1) return left ? curve : [];
  if (next === 0) return left ? [] : curve;
  const a = curve[next-1], b = curve[next];
  const t = (contactX - a.x) / (b.x - a.x);
  const edge = {x: contactX, y: a.y + t * (b.y - a.y)};
  return left ? [...curve.slice(0,next), edge] : [edge, ...curve.slice(next)];
}
export function angleArc(contact, baseline, side, angleDeg, radius, count = 36, geometry = {}) {
  if (!baseline || baseline.length !== 2) return [];
  const [a,b] = [...baseline].sort((p,q) => p.x - q.x), dx = b.x-a.x, dy = b.y-a.y, length = Math.hypot(dx,dy);
  if (!length) return [];
  const sign = side === 'left' ? 1 : -1;
  const base = geometry.surface_tangent || {x: sign * dx / length, y: sign * dy / length};
  const liquid = geometry.tangent_vector;
  const cross = liquid ? base.x * liquid.y - base.y * liquid.x : -sign;
  const start = Math.atan2(base.y, base.x), sweep = (cross >= 0 ? 1 : -1) * angleDeg * Math.PI / 180;
  return Array.from({length: count + 1}, (_, i) => {
    const t = start + sweep * i / count;
    return {x: contact.x + radius * Math.cos(t), y: contact.y + radius * Math.sin(t)};
  });
}
function renderStage() {
  const image = activeImage(), cache = active();
  $('geometry-layer').replaceChildren(); $('point-layer').replaceChildren(); $('roi-layer').replaceChildren();
  if (!image || !cache) return;
  if (!state.view) fitView();
  applyView();
  if (!$('overlay-toggle').checked) return;
  const unit = state.view.w / Math.max(1, $('image-canvas').clientWidth), o = cache.options, result = cache.result, baseline = result?.baseline || o.baseline;
  const geometry = $('geometry-layer'), pointsLayer = $('point-layer');
  if (o.roi) $('roi-layer').append(svg('rect', {x:o.roi.x,y:o.roi.y,width:o.roi.width,height:o.roi.height,class:'roi-path'}));
  if (o.mode === 'auto' && (o.algorithm === 'local' || isPendant(cache))) (o.excluded_regions || []).forEach((r, index) => {
    const box = svg('rect', {x:r.x,y:r.y,width:r.width,height:r.height,class:'exclusion-path'});
    const title = svg('title'); title.textContent = '忽略区域 ' + (index + 1); box.append(title); $('roi-layer').append(box);
  });
  const preview = cache.preview?.key === surfaceKey(o) ? cache.preview : null;
  const surface = result ? result.surface_curve : preview?.curve;
  if (surface?.length > 1) {
    geometry.append(svg('path', {d:pathData(surface),class:'baseline-path surface-curve'}));
    if (result) {
      for (const side of ['left','right']) {
        const wing = exposedSurface(surface, result[side].contact_point.x, side === 'left');
        geometry.append(svg('path',{d:pathData(wing),class:'baseline-path exposed-surface'}));
      }
    }
  } else if (baseline?.length === 2 && !cache.needsBaselineRefresh && !cache.estimating &&
      (result || o.orientation === 'sessile' || o.baseline_method === 'dense_substrate_v1' ||
       (cache.direction?.polarity === o.polarity && cache.direction.orientation === 'sessile'))) {
    const [a,b] = [...baseline].sort((p,q) => p.x-q.x), slope = (b.y-a.y)/(b.x-a.x || 1);
    geometry.append(svg('path', {d:pathData([{x:0,y:a.y-slope*a.x},{x:image.width,y:a.y+slope*(image.width-a.x)}]),class:'baseline-path'}));
  }
  if (result) {
    for (const item of result.geometry?.contours || []) geometry.append(svg('path',{d:pathData(item.points || []),class:'contour-path'}));
    for (const item of result.geometry?.fits || []) geometry.append(svg('path',{d:pathData(item.points || []),class:'fit-path',stroke:COLORS[item.side] || COLORS.both}));
    for (const side of ['left','right']) {
      const r = result[side]; if (!r?.contact_point) continue;
      const p = r.contact_point, tangent = r.tangent_vector;
      const vectorLength = tangent ? Math.hypot(tangent.x,tangent.y) : 0, length = Math.min(image.width,image.height)*.15;
      if (vectorLength) geometry.append(svg('path',{d:pathData([p,{x:p.x+tangent.x/vectorLength*length,y:p.y+tangent.y/vectorLength*length}]),class:'tangent-path',stroke:COLORS[side]}));
      const gap = Math.hypot(result.left.contact_point.x-result.right.contact_point.x,
                             result.left.contact_point.y-result.right.contact_point.y);
      const radius = Math.min(Math.min(image.width,image.height)*.055, gap*.36);
      if (r.surface_tangent) {
        const v = r.surface_tangent, length = Math.hypot(v.x, v.y);
        if (length) geometry.append(svg('path', {d:pathData([p, {x:p.x+v.x/length*radius*1.45, y:p.y+v.y/length*radius*1.45}]),
                                                class:'tangent-path',stroke:COLORS.baseline}));
      }
      geometry.append(svg('path',{d:pathData(angleArc(p,baseline,side,r.angle_deg,radius,36,r)),class:'angle-path',stroke:COLORS[side]}));
      geometry.append(svg('circle',{cx:p.x,cy:p.y,r:3.5*unit,fill:COLORS[side],stroke:'white','stroke-width':1.2*unit}));
      const middle = angleArc(p,baseline,side,r.angle_deg/2,radius*1.6,1,r).at(-1) || p;
      const left = side === 'left', x = left ? p.x-radius*.55 : p.x+radius*.55;
      const label = svg('text',{x:left?Math.max(48*unit,x):Math.min(image.width-48*unit,x),'text-anchor':left?'end':'start',y:Math.max(15*unit,Math.min(image.height-7*unit,middle.y)),fill:COLORS[side],'font-size':12*unit,class:'point-label'});
      label.style.strokeWidth = String(3*unit); label.textContent = format(r.angle_deg,1)+'°'; geometry.append(label);
    }
  }
  for (const group of ['baseline','contour']) {
    if (group !== 'baseline' && o.mode !== 'manual') continue;
    if (group === 'baseline' && o.baseline_source === 'automatic' && result?.baseline_points?.length &&
        stableJSON(result.baseline_points) !== stableJSON(o.baseline_points)) {
      for (const p of result.baseline_points) {
        pointsLayer.append(svg('circle',{cx:p.x,cy:p.y,r:1.8*unit,fill:COLORS.baseline}));
      }
      continue;
    }
    pointsFor(o,group).forEach((p,i) => {
      const selected = state.selectedPoint?.group === group && state.selectedPoint?.index === i;
      if (selected) pointsLayer.append(svg('circle',{cx:p.x,cy:p.y,r:9*unit,fill:COLORS[group],'fill-opacity':.14,stroke:'none'}));
      const handle = svg('circle',{cx:p.x,cy:p.y,r:(group==='baseline'?(state.tool==='baseline'?3.2:1.8):4.5)*unit,fill:COLORS[group],stroke:'white','stroke-width':1.4*unit,class:'point-handle',tabindex:'0',role:'button','data-point-group':group,'data-point-index':i,'aria-label':GROUP_NAMES[group]+'第'+(i+1)+'点，X '+format(p.x,1)+'，Y '+format(p.y,1)});
      pointsLayer.append(handle);
      if (!selected) return;
      const text = svg('text',{x:p.x+7*unit,y:p.y-7*unit,fill:COLORS[group],'font-size':10*unit,class:'point-label'}); text.style.strokeWidth = String(3*unit); text.textContent = (group==='baseline'?'B':'P')+(i+1); pointsLayer.append(text);
    });
  }
}
function pointerDown(event) {
  const image = activeImage(), cache = active(); if (!image || !cache?.loaded || ![0,1].includes(event.button)) return;
  event.preventDefault(); $('image-canvas').focus({preventScroll:true}); $('image-canvas').setPointerCapture(event.pointerId);
  const p = screenToImage(event.clientX,event.clientY), target = event.target.closest('[data-point-group]');
  if (state.space || event.button === 1 || (!target && state.tool === 'pan')) {
    state.drag = {kind:'pan',clientX:event.clientX,clientY:event.clientY,view:{...state.view}}; state.selectedPoint = null; renderPoints(); return;
  }
  if (target) { state.selectedPoint = {group:target.dataset.pointGroup,index:Number(target.dataset.pointIndex)}; state.pointTab = state.selectedPoint.group; state.drag = {kind:'point',...state.selectedPoint,before:JSON.stringify(cache.options),start:p,moved:false}; renderPoints(); renderStage(); return; }
  if (!insideImage(p)) return;
  if (state.tool === 'roi') { state.drag = {kind:'roi',start:boundedPoint(p),end:boundedPoint(p)}; return; }
  if (['baseline','contour'].includes(state.tool)) {
    const group = state.tool, current = pointsFor(cache.options,group);
    if (current.length >= (group==='baseline'?200:400)) { notify('已达到打点数量上限',true); return; }
    pushUndo(cache);
    const points = pointsFor(cache.options,group); points.push(boundedPoint(p)); setPoints(cache.options,group,points);
    if (group === 'baseline') cache.options.baseline_source = 'manual';
    invalidate(cache); state.selectedPoint = {group,index:points.length-1};
    state.drag = {kind:'new-point',group,index:points.length-1};
    renderAll();
  }
}
function pointerMove(event) {
  const image = activeImage(), cache = active(); if (!image || !cache) return;
  const p = screenToImage(event.clientX,event.clientY);
  setText('cursor-coordinates', insideImage(p) ? 'X '+format(p.x,1)+' · Y '+format(p.y,1)+' px' : '');
  if (!state.drag) return;
  const drag = state.drag;
  if (drag.kind === 'pan') {
    const rect = $('image-canvas').getBoundingClientRect(), u = drag.view.w/rect.width;
    state.view = {...drag.view,x:drag.view.x-(event.clientX-drag.clientX)*u,y:drag.view.y-(event.clientY-drag.clientY)*u};
    applyView();
    if (state.view.x !== drag.view.x || state.view.y !== drag.view.y) state.manualScale = 1 / u;
    return;
  }
  if (drag.kind === 'roi') {
    drag.end = boundedPoint(p); const x=Math.min(drag.start.x,drag.end.x),y=Math.min(drag.start.y,drag.end.y);
    $('drag-layer').replaceChildren(svg('rect',{x,y,width:Math.abs(drag.end.x-drag.start.x),height:Math.abs(drag.end.y-drag.start.y),class:'roi-path'})); return;
  }
  if (drag.kind === 'point' && !drag.moved) {
    if (Math.hypot(p.x-drag.start.x,p.y-drag.start.y)<.03) return;
    if (cache.undo.at(-1)!==drag.before) cache.undo.push(drag.before); cache.redo=[]; invalidate(cache); drag.moved=true;
  }
  const points = pointsFor(cache.options,drag.group); points[drag.index] = boundedPoint(p);
  setPoints(cache.options,drag.group,points);
  if (drag.group==='baseline') cache.options.baseline_source='manual';
  renderStage(); renderResults(); renderRun();
}
function pointerUp() {
  const drag=state.drag,cache=active(); if (!drag || !cache) return;
  state.drag=null; $('drag-layer').replaceChildren();
  if (drag.kind==='roi') {
    const a=drag.start,b=drag.end,x=Math.floor(Math.min(a.x,b.x)),y=Math.floor(Math.min(a.y,b.y)),width=Math.ceil(Math.abs(a.x-b.x)),height=Math.ceil(Math.abs(a.y-b.y));
    if (width>=24&&height>=24) { edit((o)=>{o.roi={x,y,width,height};}); state.tool='pan'; }
    else notify('区域至少需要 24 × 24 像素，请拖动选择更大的范围');
  }
  if (drag.kind!=='pan') { renderAll(); scheduleDraft(); }
}
function renderResults() {
  const cache=active(),result=cache?.result;
  const needsRefit = Boolean(cache?.dirty && cache.history.some(job => job.state === 'done'));
  const status=$('result-status'); status.className='pill '+(cache?.pendingJob?'running':!result?'neutral':result.status==='review'?'review':'ok');
  status.textContent=cache?.pendingJob?'拟合中':!result?needsRefit?'待重新拟合':'未测量':result.status==='review'?'需核查':'已完成';
  for (const side of ['left','right','mean']) {
    const value=result?(side==='mean'?result.mean_angle_deg:result[side]?.angle_deg):null;
    const strong=$(side+'-angle');strong.textContent=format(value);
    if (Number.isFinite(value)) strong.append(element('small','','°'));
  }
  setText('result-method',result?.method_label||'');
  for(const side of ['left','right']){
    const range=result?.sensitivity?.[side];
    setText(side+'-range',range&&range.min_deg!==null?'敏感性范围 '+format(range.min_deg,1)+'–'+format(range.max_deg,1)+'°':result?.sensitivity?'方法范围不足':'');
  }
  setText('angle-difference',result?'左右差 '+format(result.asymmetry_deg,2)+'°':'');
  setText('sensitivity-note',result?.sensitivity?result.sensitivity.accepted_runs+' 个候选通过检查':'');
  $('sensitivity-note').title = result?.sensitivity?.interpretation || '';
  visible('result-footer', Boolean(result));
  const warnings=(result?.warnings||[]).filter(message=>!['请核对基线和接触区；低拟合残差不能保证角度准确。','请核对所选局部轮廓及基线。'].includes(message));const notices=$('result-notices');notices.replaceChildren();warnings.forEach((message)=>notices.append(element('p','',friendly(message))));visible('result-notices',warnings.length>0);
  visible('stage-badge',needsRefit && !result && !cache.pendingJob);
  const canExport=Boolean(result&&!cache?.pendingJob);
  visible('downloads',canExport);$('download-png-top').disabled=!canExport;
  if(result){
    const base='/api/results/'+encodeURIComponent(cache.resultJobId)+'/';
    const downloads=result.downloads||{};
    $('download-json').href=downloads.json_url||base+'result.json';$('download-csv').href=downloads.csv_url||base+'results.csv';$('download-png').href=downloads.png_url||result.annotation_url||base+'annotation.png';$('download-zip').href=downloads.zip_url||base+'bundle.zip';
  }
}
function validateReady(cache) {
  const o=cache.options;
  if (!cache.loaded) return '尚未读取本图草稿，请刷新图片库或重新选择本图重试。';
  if(pointsFor(o,'baseline').length===1 || (pointsFor(o,'baseline').length>=2&&!o.baseline))return '表面点横向至少相隔 5 px，倾角需在 ±45° 内。';
  if(o.mode==='auto' && !isPendant(cache) && o.algorithm!=='local' && o.excluded_regions?.length) return '座滴的忽略区域适用于局部边缘拟合。请切换方法或清除忽略区域。';
  if(o.mode==='manual'){
    if(o.baseline?.length!==2)return '请先设置表面基线。';
    if(o.contour_points.length<7)return '沿液滴轮廓至少选 7 点，覆盖两侧接触区和顶端。';
  }
  return null;
}
function renderRun() {
  const cache=active(),button=$('run-button');
  button.textContent=cache?.pendingJob?'正在拟合…':cache?.estimating?'正在识别基线…':cache&&!cache.loaded?'正在读取草稿…':cache?.options.mode==='manual'?'拟合液滴轮廓':'开始自动拟合';
  button.disabled=!cache||!cache.loaded||cache.estimating||Boolean(cache.pendingJob);visible('stage-busy',Boolean(cache?.pendingJob));
  if(cache?.pendingJob)setText('busy-title',cache.pendingJob.state==='queued'?'等待测量':'正在拟合');
  visible('measurement-error',Boolean(cache?.error));if(cache?.error)setText('measurement-error',cache.error);
}
async function runMeasurement() {
  const image=activeImage(),cache=active();if(!image||!cache||cache.pendingJob)return;
  if(cache.options.mode==='manual')cache.options.manual_method='spline';
  const error=validateReady(cache);if(error){cache.error=error;renderRun();notify(error,true);return;}
  const id=image.id,options=payload(cache.options),snapshot=optionSignature(options);
  cache.error=null;cache.pendingJob={id:null,state:'queued',snapshot};renderRun();renderResults();
  try{
    saveDraft(id);
    const job=await request('/api/jobs',jsonRequest('POST',{image_id:id,options}));
    cache.pendingJob={id:job.id,state:job.state,snapshot};
    if(isCurrent(id)){renderRun();renderLibrary();pollJob(job.id,id,snapshot,state.selectionEpoch);}
  }catch(error){cache.pendingJob=null;cache.error=error.message;if(isCurrent(id)){renderRun();renderResults();notify(error.message,true);}}
}
async function delay(ms,signal){return new Promise((resolve,reject)=>{if(signal.aborted){reject(new DOMException('Aborted','AbortError'));return;}const abort=()=>{clearTimeout(t);reject(new DOMException('Aborted','AbortError'));};const t=setTimeout(()=>{signal.removeEventListener('abort',abort);resolve();},ms);signal.addEventListener('abort',abort,{once:true});});}
async function pollJob(jobId,imageId,snapshot,epoch){
  if(!isCurrent(imageId,epoch))return;
  state.pollController?.abort();const controller=new AbortController();state.pollController=controller;
  const cache=state.caches.get(imageId);if(!cache)return;
  try{
    while(!controller.signal.aborted){
      const job=await request('/api/jobs/'+encodeURIComponent(jobId),{signal:controller.signal});
      if(cache.pendingJob?.id!==jobId)return;
      cache.pendingJob.state=job.state;
      if(job.state==='done'||job.state==='error'){
        cache.pendingJob=null;cache.history=[job,...cache.history.filter((r)=>r.id!==job.id)].slice(0,100);
        if(job.state==='done'&&job.result){
          if(optionSignature(cache.options)===snapshot){cache.result=job.result;cache.resultJobId=job.id;cache.dirty=false;cache.error=null;}
          else if(isCurrent(imageId,epoch))notify('测量已保存；当前打点已改变，请重新拟合或打开历史结果。');
        }else cache.error=friendly(job.error?.message||'此次拟合未能完成，请检查基线或改用手动打点。');
        if(isCurrent(imageId,epoch))renderAll();
        return;
      }
      if(isCurrent(imageId,epoch))renderRun();
      await delay(650,controller.signal);
    }
  }catch(error){
    if(error.name==='AbortError')return;
    // A network failure is not a measurement failure. Keep the ID for resuming.
    if(isCurrent(imageId,epoch)){cache.error='读取进度失败：'+friendly(error.message)+' 可点击刷新测量记录重试。';renderRun();}
  }
}
async function refreshHistory(id=state.currentId,epoch=state.selectionEpoch) {
  if(!id)return;
  const data=await request('/api/images/'+encodeURIComponent(id)+'/results'),cache=state.caches.get(id);if(!cache)return;
  cache.history=data.results||[];
  if(isCurrent(id,epoch)){
    renderHistory();
    if(cache.pendingJob?.id)pollJob(cache.pendingJob.id,id,cache.pendingJob.snapshot,epoch);
    else{const latest=cache.history.find((j)=>j.state==='done'&&j.result&&optionSignature(j.options)===optionSignature(cache.options));if(latest&&!cache.result&&!cache.dirty){cache.result=latest.result;cache.resultJobId=latest.id;renderResults();renderStage();}}
  }
}
function renderHistory(){
  const container=$('measurement-history');container.replaceChildren();const cache=active();
  if(!cache?.history.length){container.append(element('p','muted','暂无记录'));return;}
  for(const job of cache.history){
    const failed=job.state==='error',button=element('button','history-item'+(failed?' failed':'')+(job.id===cache.resultJobId?' active':''));button.type='button';
    button.disabled = !cache.loaded;
    const title=element('span'),date=new Date(job.created_at);
    title.append(element('strong','',job.options?.mode==='manual'?'手动打点拟合':'自动拟合'),element('small','',Number.isNaN(date.valueOf())?'测量记录':date.toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'})));
    button.append(title,element('span','',failed?'未完成':job.result?format(job.result.mean_angle_deg,1)+'°':job.state==='running'?'拟合中':'等待中'));
    button.setAttribute('aria-label','打开'+(failed?'失败的':'')+'测量记录 '+(job.result?format(job.result.mean_angle_deg,2)+'度':''));
    button.addEventListener('click',()=>openHistory(job));container.append(button);
  }
}
function openHistory(job){
  const cache=active();if(!cache)return;
  if(job.options){pushUndo(cache);cache.options=loadedOptions(job.options);cache.editVersion+=1;cache.saved=false;}
  cache.result=job.result||null;cache.resultJobId=job.result?job.id:null;cache.dirty=false;cache.error=job.state==='error'?friendly(job.error?.message||'此次测量未完成'):null;
  state.selectedPoint=null;renderAll();scheduleDraft();if(job.state==='error')notify('已恢复该次打点，可修改后重试');
}
export function normalizeCoordinateInput(raw,width,height,current=DEFAULT_OPTIONS){
  const data=raw.options||raw;
  if(typeof data!=='object'||data===null||Array.isArray(data))throw new Error('坐标JSON应为对象。');
  const result={...clone(current)};
  function checkPoint(p){
    if(!p||typeof p!=='object')throw new Error('每个点必须包含 x 和 y。');
    const x=Number(p.x),y=Number(p.y);
    if(p.x===null||p.y===null||p.x===''||p.y===''||!Number.isFinite(x)||!Number.isFinite(y)||x<0||y<0||x>=width||y>=height)throw new Error('所有点坐标必须位于原图范围内。');
    return{x,y};
  }
  let touched=false;
  for(const key of ['baseline','baseline_points','contour_points','left_points','right_points']){
    if(!(key in data))continue;touched=true;
    if(key==='baseline'&&data[key]===null){result.baseline=null;if(!('baseline_points' in data))result.baseline_points=[];continue;}
    if(!Array.isArray(data[key])||data[key].length>(key==='contour_points'?400:200))throw new Error('点坐标需为数组，且不超过数量上限。');
    if(key==='baseline'&&![0,2].includes(data[key].length))throw new Error('基线需提供两个点。');
    if(key==='baseline_points'&&data[key].length===1)throw new Error('表面基线至少需要两个采样点。');
    result[key]=data[key].map(checkPoint);
    if(key==='baseline'&&!('baseline_points' in data))result.baseline_points=result[key].map(point);
    if(key==='baseline'){result.baseline=result.baseline.length?result.baseline:null;result.baseline_source='manual';}
  }
  if ('excluded_regions' in data) touched = true;
  if(!touched)throw new Error('请提供 baseline_points、contour_points 或 excluded_regions 字段。');
  for(const[key,allowed]of Object.entries({mode:['auto','manual'],algorithm:['general','fibrous','local'],orientation:['auto','pendant','sessile'],manual_method:['spline','quadratic','cubic','tangent'],polarity:['dark','light']})){
    if(key in data){if(!allowed.includes(data[key]))throw new Error('不支持的选项：'+key);result[key]=data[key];}
  }
  if(typeof data.sensitivity==='boolean')result.sensitivity=data.sensitivity;
  if((data.contour_points?.length||data.left_points?.length||data.right_points?.length)&&!('mode'in data))result.mode='manual';
  if(data.contour_points?.length) { result.manual_method='spline'; result.left_points=[]; result.right_points=[]; }
  if(('left_points' in data||'right_points' in data)&&!('contour_points' in data)) result.contour_points=[...result.left_points].sort((a,b)=>b.y-a.y).concat([...result.right_points].sort((a,b)=>a.y-b.y));
  if('baseline_points' in data || 'baseline' in data) {
    result.baseline_source = data.baseline_source === 'automatic' ? 'automatic' : 'manual';
    result.baseline_method = result.baseline_source === 'automatic' &&
      ['dense_substrate_v1','pendant_surface_v1'].includes(data.baseline_method) ? data.baseline_method : null;
    if (result.baseline_method !== 'pendant_surface_v1' || data.baseline?.length !== 2) syncBaseline(result);
  }
  if('roi'in data){
    if(data.roi===null)result.roi=null;
    else{const r=data.roi;for(const k of ['x','y','width','height'])if(!Number.isInteger(r[k]))throw new Error('分析区域需使用整数像素。');if(r.x<0||r.y<0||r.width<24||r.height<24||r.x+r.width>width||r.y+r.height>height)throw new Error('分析区域需至少24×24像素，且位于原图内。');result.roi=clone(r);}
  }
  if('excluded_regions' in data){
    if(!Array.isArray(data.excluded_regions)||data.excluded_regions.length>20)throw new Error('忽略区域应为数组，最多20个。');
    result.excluded_regions=data.excluded_regions.map((r)=>{
      if(!r||typeof r!=='object')throw new Error('每个忽略区域需要x、y、width、height。');
      for(const key of ['x','y','width','height'])if(!Number.isInteger(r[key]))throw new Error('忽略区域需使用整数像素。');
      if(r.x<0||r.y<0||r.width<1||r.height<1||r.x+r.width>width||r.y+r.height>height)throw new Error('忽略区域需要位于原图内，宽高至少1像素。');
      return{x:r.x,y:r.y,width:r.width,height:r.height};
    });
  }
  return result;
}
function importCoordinates(){
  const image=activeImage(),cache=active();if(!image||!cache)return;
  try{
    const data=JSON.parse($('coordinate-json').value);
    if(data.image&&((data.image.width&&data.image.width!==image.width)||(data.image.height&&data.image.height!==image.height)||(data.image.sha256&&data.image.sha256!==image.sha256)))throw new Error('该坐标文件对应另一张图片，请先打开匹配的原图。');
    const next=normalizeCoordinateInput(data,image.width,image.height,cache.options);
    edit((o)=>Object.assign(o,next));state.selectedPoint=null;$('overlay-toggle').checked=true;renderStage();notify('已应用坐标');
  }catch(error){notify(error instanceof SyntaxError?'JSON格式不正确，请检查逗号和引号。':error.message,true);}
}
function downloadObject(data,filename){
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=element('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function exportCoordinates(){
  const image=activeImage(),cache=active();if(!image||!cache)return;
  if(pointsFor(cache.options,'baseline').length===1){notify('请先完成两个基线点后再导出。',true);return;}
  downloadObject({format:'contact-angle-studio-coordinates-v1',image:{filename:image.filename,width:image.width,height:image.height,sha256:image.sha256},options:payload(cache.options)},image.filename.replace(/\.[^.]+$/,'')+'.coordinates.json');
}
function keyboard(event){
  const tag=event.target.tagName;if(['INPUT','TEXTAREA','SELECT'].includes(tag)||event.target.isContentEditable||$('help-dialog').open)return;
  if(!active()?.loaded)return;
  if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='z'){event.preventDefault();undo(event.shiftKey);return;}
  if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='y'){event.preventDefault();undo(true);return;}
  if(event.metaKey||event.ctrlKey)return;
  if(event.code==='Space'){event.preventDefault();state.space=true;renderTools();return;}
  const key=event.key.toLowerCase(),tools={b:'baseline',c:'contour',h:'pan'};
  if(tools[key]){event.preventDefault();setTool(tools[key]);return;}
  if(key==='0'){event.preventDefault();fitView();return;}
  if(['+','='].includes(key)){event.preventDefault();zoom(1.25);return;}
  if(key==='-'){event.preventDefault();zoom(.8);return;}
  const selected=state.selectedPoint,p=selected&&pointsFor(active().options,selected.group)[selected.index];
  if(p&&['Delete','Backspace'].includes(event.key)){event.preventDefault();removePoint(selected.group,selected.index);return;}
  if(p&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){
    event.preventDefault();const step=event.altKey ? 0.1 : event.shiftKey ? 10 : 1,delta={ArrowLeft:[-step,0],ArrowRight:[step,0],ArrowUp:[0,-step],ArrowDown:[0,step]}[event.key];
    edit((o)=>{const points=pointsFor(o,selected.group);points[selected.index]=boundedPoint({x:p.x+delta[0],y:p.y+delta[1]});setPoints(o,selected.group,points);if(selected.group==='baseline')o.baseline_source='manual';},{focusCanvas:true});
  }
}
function boot(){
  for(const id of ['upload-button','empty-upload'])$(id).addEventListener('click',()=>$('file-input').click());
  $('file-input').addEventListener('change',(event)=>uploadFiles([...event.target.files]));
  for(const id of ['demo-button','empty-demo'])$(id).addEventListener('click',openDemo);
  $('refresh-images').addEventListener('click',()=>refreshImages().catch((e)=>notify(e.message,true)));
  $('image-search').addEventListener('input',renderLibrary);
  document.querySelectorAll('button[data-tool]').forEach((button)=>button.addEventListener('click',()=>setTool(button.dataset.tool)));
  document.querySelectorAll('[data-point-tab]').forEach((button)=>button.addEventListener('click',()=>{state.pointTab=button.dataset.pointTab;renderPoints();}));
  $('mode-auto').addEventListener('click',()=>{if(active()?.options.mode!=='auto')edit((o)=>{o.mode='auto';});if(state.tool === 'contour')setTool('pan');});
  $('mode-manual').addEventListener('click',()=>{if(active()?.options.mode!=='manual')edit((o)=>{o.mode='manual';o.manual_method='spline';});setTool('contour');});
  $('algorithm-select').addEventListener('change',event=>edit(o=>{o.algorithm=event.target.value;}));
  for (const [id,key] of [['orientation-select','orientation'],['polarity-select','polarity']]) {
    $(id).addEventListener('change',event=>{
      const value = event.target.value, cache = active();
      cache.direction = null;
      edit(o => {
        o[key] = value;
        if (o.baseline_source === 'automatic') setPoints(o, 'baseline', []);
      });
      if (cache.options.baseline_source === 'automatic') estimateBaseline(false);
    });
  }
  $('sensitivity-toggle').addEventListener('change',(event)=>edit((o)=>{o.sensitivity=event.target.checked;}));
  $('estimate-baseline').addEventListener('click',()=>estimateBaseline(true));
  $('draw-baseline').addEventListener('click',()=>{edit(o=>{setPoints(o,'baseline',[]);o.baseline_source='manual';});setTool('baseline');});
  $('clear-roi').addEventListener('click',()=>edit((o)=>{o.roi=null;}));
  $('clear-exclusions').addEventListener('click',()=>edit((o)=>{o.excluded_regions=[];}));
  $('clear-points').addEventListener('click',()=>{edit((o)=>{setPoints(o,state.pointTab,[]);if(state.pointTab==='baseline')o.baseline_source='manual';});state.selectedPoint=null;renderPoints();});
  $('undo-button').addEventListener('click',()=>undo());$('redo-button').addEventListener('click',()=>undo(true));
  $('zoom-in').addEventListener('click',()=>zoom(1.25));$('zoom-out').addEventListener('click',()=>zoom(.8));$('fit-view').addEventListener('click',fitView);$('actual-view').addEventListener('click',actualView);
  $('overlay-toggle').addEventListener('change',()=>{if(!$('overlay-toggle').checked)setTool('pan');renderStage();});
  $('import-coordinates').addEventListener('click',importCoordinates);$('export-coordinates').addEventListener('click',exportCoordinates);
  $('run-button').addEventListener('click',runMeasurement);
  $('refresh-history').addEventListener('click',()=>refreshHistory().catch((e)=>notify(e.message,true)));
  $('download-png-top').addEventListener('click',()=>{if(active()?.result&&!active()?.pendingJob)$('download-png').click();});
  const canvas=$('image-canvas');
  canvas.addEventListener('pointerdown',pointerDown);canvas.addEventListener('pointermove',pointerMove);canvas.addEventListener('pointerup',pointerUp);canvas.addEventListener('pointercancel',pointerUp);
  canvas.addEventListener('pointerleave',()=>{if(!state.drag)setText('cursor-coordinates','');});
  let lastWheelZoom=0;
  canvas.addEventListener('wheel',(event)=>{if(!event.ctrlKey&&!event.metaKey)return;event.preventDefault();const now=performance.now();if(now-lastWheelZoom<70||!event.deltaY)return;lastWheelZoom=now;if(active())zoom(event.deltaY<0?1.1:1/1.1,screenToImage(event.clientX,event.clientY));},{passive:false});
  canvas.addEventListener('contextmenu',(event)=>event.preventDefault());
  canvas.addEventListener('focusin',(event)=>{const target=event.target.closest('[data-point-group]');if(target)state.selectedPoint={group:target.dataset.pointGroup,index:Number(target.dataset.pointIndex)};});
  window.addEventListener('keydown',keyboard);window.addEventListener('keyup',(e)=>{if(e.code==='Space'){state.space=false;renderTools();}});
  window.addEventListener('blur',()=>{state.space=false;pointerUp();renderTools();});
  new ResizeObserver(() => {
    if (!activeImage()) return;
    if (!state.view || state.manualScale === null) {
      fitView();
      return;
    }
    const previous = state.view, stage = stageSize();
    const w = stage.width / state.manualScale, h = stage.height / state.manualScale;
    state.view = {x: previous.x + (previous.w - w) / 2, y: previous.y + (previous.h - h) / 2, w, h};
    applyView();
    renderStage();
  }).observe($('image-stage'));
  let dragDepth=0;
  document.addEventListener('dragenter',(event)=>{if(event.dataTransfer?.types.includes('Files')){event.preventDefault();dragDepth++;visible('drop-overlay',true);}});
  document.addEventListener('dragover',(event)=>{if(event.dataTransfer?.types.includes('Files')){event.preventDefault();event.dataTransfer.dropEffect='copy';}});
  document.addEventListener('dragleave',(event)=>{if(event.dataTransfer?.types.includes('Files')){dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)visible('drop-overlay',false);}});
  document.addEventListener('drop',(event)=>{if(event.dataTransfer?.files.length){event.preventDefault();dragDepth=0;visible('drop-overlay',false);uploadFiles([...event.dataTransfer.files]);}});
  $('help-button').addEventListener('click',()=>$('help-dialog').showModal());$('close-help').addEventListener('click',()=>$('help-dialog').close());$('help-done').addEventListener('click',()=>$('help-dialog').close());
  $('help-dialog').addEventListener('click',(event)=>{if(event.target===$('help-dialog')){const r=$('help-dialog').getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)$('help-dialog').close();}});
  renderAll();
  request('/api/info').then((info)=>{setText('app-version','v'+info.version);if(info.limits?.max_upload_bytes)state.maxUpload=info.limits.max_upload_bytes;setText('upload-limit','每张最大 '+Math.round(state.maxUpload/1024/1024)+' MB');}).catch(()=>{notify('无法连接本机服务，请从桌面重新打开接触角工作台。',true);});
  refreshImages().catch((error)=>notify(error.message,true));
}
if(typeof document!=='undefined')boot();
