// DOM Elements
const $ = id => document.getElementById(id);

const uploadBtn = $('upload-btn');
const fetchBtn = $('fetch-btn');
const demoBtn = $('demo-btn');
const reportBtn = $('report-btn');
const statusText = $('status-text');
const backendStatus = $('backend-status');
const backendBadge = $('backend-badge');
const stationBadge = $('active-station-text');
const traceMeta = $('trace-meta');
const plotFrame = $('plot-frame');
const plotDiv = $('waveform-plot');
const emptyState = $('empty-state');
const fitPicksBtn = $('fit-picks-btn');

// Quality & Metrics
const valSnr = $('val-snr');
const valNoise = $('val-noise');
const valClip = $('val-clip');
const valQuality = $('val-quality');

// Detections
const valProb = $('val-prob');
const valMag = $('val-mag');
const valP = $('val-p');
const valS = $('val-s');
const valOrigin = $('val-origin');
const valSp = $('val-sp');
const similarList = $('similar-list');

// Progress
const progressContainer = $('progress-container');
const progressFill = $('progress-fill');
const progressPercent = $('progress-percent');
const progressStep = $('progress-step');

// Fetch drawer
const fetchModal = $('fetch-modal');
const fetchForm = $('fetch-form');
const fetchError = $('fetch-error');

// AI summary
const aiBody = $('ai-body');
const aiBtn = $('ai-btn');

let currentSignalId = null;
let currentEventId = null;
let currentWaveformData = null;
let currentStationLabel = 'waveform';
let aiPollTimer = null;

const isLocalFile = window.location.protocol === 'file:';
const API_BASE = isLocalFile ? 'http://127.0.0.1:8000' : window.location.origin;
const WS_BASE = isLocalFile ? 'ws://127.0.0.1:8000' : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

const PIPELINE_STEPS = ['fetching_waveform', 'quality_analysis', 'filtering', 'eqtransformer', 'completed'];

// ---------- Small UI helpers ----------

function setStatus(text) {
    statusText.textContent = text;
}

function escapeHtml(str) {
    return String(str ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function toast(title, message = '', tone = 'info', timeout = 5200) {
    const el = document.createElement('div');
    el.className = 'toast';
    el.dataset.tone = tone;
    el.innerHTML = `
        <svg class="icon"><use href="#${tone === 'error' ? 'i-alert' : 'i-check'}"/></svg>
        <div><strong>${escapeHtml(title)}</strong>${message ? `<p>${escapeHtml(message)}</p>` : ''}</div>`;
    $('toasts').appendChild(el);
    Motion.toastIn(el);
    setTimeout(() => Motion.toastOut(el), timeout);
}

// Drops a trailing zone ("Z" or "+00:00"); every time in the app is UTC
function stripZone(iso) {
    return String(iso).replace(/(Z|[+-]\d\d:\d\d)$/, '');
}

function formatUtc(iso) {
    if (!iso) return '—';
    const [date, time = ''] = stripZone(iso).split('T');
    return `${date} ${time.slice(0, 11)}`.trim();
}

function setLoading(on) {
    document.querySelector('.readout').classList.toggle('is-loading', on);
    document.querySelector('.inspector').classList.toggle('is-loading', on);
}

// Check backend connectivity
async function checkBackend() {
    const started = performance.now();
    try {
        const res = await fetch(`${API_BASE}/`, { cache: 'no-store' });
        if (!res.ok) throw new Error();
        backendStatus.textContent = 'Connected';
        $('backend-latency').textContent = `${Math.round(performance.now() - started)} ms`;
        backendBadge.dataset.state = 'online';
    } catch {
        backendStatus.textContent = 'Backend offline';
        $('backend-latency').textContent = '';
        backendBadge.dataset.state = 'offline';
    }
}

setInterval(checkBackend, 5000);
checkBackend();

// ---------- Plot ----------

WavePlot.init(plotDiv);

function showPlot() {
    if (plotFrame.dataset.state !== 'empty') return;
    Motion.leaveEmpty(plotFrame, emptyState, () => {
        plotFrame.dataset.state = 'ready';
        emptyState.hidden = true;
        WavePlot.resize();
    });
}

// Plot waveform traces with P/S arrival annotations
function plotWaveform(traces, picks = [], reveal = 'traces') {
    currentWaveformData = traces;
    return WavePlot.render(traces, picks, { reveal });
}

// Download and plot waveform data
async function loadAndPlotWaveform(signalId) {
    try {
        const res = await fetch(`${API_BASE}/waveform/${signalId}`);
        if (!res.ok) throw new Error('Failed to load waveform');
        const data = await res.json();
        const fs = (data.traces[0] && data.traces[0].sample_rate) || 100;
        const comps = data.traces.map(t => (t.component || t.channel.slice(-1))).join(' · ');
        traceMeta.textContent = `${data.network}.${data.station}  ·  ${comps}  ·  ${fs} Hz`;
        stationBadge.textContent = `${data.network}.${data.station}`;
        currentStationLabel = `${data.network}.${data.station}`;
        showPlot();
        await plotWaveform(data.traces);
    } catch (err) {
        console.error('Waveform load error:', err);
        toast('Could not load waveform', err.message, 'error');
    }
}

// ---------- Pipeline ----------

function resetResults() {
    currentEventId = null;
    reportBtn.disabled = true;
    fitPicksBtn.disabled = true;
    aiBtn.disabled = true;
    clearTimeout(aiPollTimer);
    valProb.textContent = '—';
    valMag.textContent = '—';
    valP.textContent = '—';
    valS.textContent = '—';
    $('conf-p').textContent = '—';
    $('conf-s').textContent = '—';
    valSp.textContent = '—';
    valOrigin.textContent = '—';
    ['meter-prob', 'meter-p', 'meter-s'].forEach(id => { $(id).style.width = '0%'; });
    ['pt-p', 'pt-s', 'pt-window'].forEach(id => { $(id).style.opacity = 0; });
    similarList.innerHTML = '<p class="panel-empty">Searching the vector index…</p>';
    aiBody.innerHTML = '<p class="panel-empty">A written summary and ground-motion forecast can be generated once an event is detected.</p>';
}

function setPipelineStep(step, isError = false) {
    const idx = PIPELINE_STEPS.indexOf(step);
    progressContainer.querySelectorAll('li').forEach((li, i) => {
        li.classList.toggle('is-done', idx >= 0 && (i < idx || (step === 'completed' && i === idx)));
        li.classList.toggle('is-active', i === idx && step !== 'completed' && !isError);
        li.classList.toggle('is-error', isError && i === idx);
    });
}

function setProgress(pct) {
    progressFill.style.width = `${pct}%`;
    progressPercent.textContent = `${pct}%`;
}

function startPipeline() {
    resetResults();
    setLoading(true);
    setPipelineStep('fetching_waveform');
    setProgress(10);
    progressStep.textContent = 'Connecting to inference server';
    if (progressContainer.hidden) Motion.fadeIn(progressContainer);
}

function finishPipeline() {
    setPipelineStep('completed');
    setProgress(100);
    Motion.fadeOut(progressContainer, 0.9);
}

function failPipeline(message) {
    setLoading(false);
    const active = progressContainer.querySelector('li.is-active');
    setPipelineStep(active ? active.dataset.step : 'fetching_waveform', true);
    progressStep.textContent = `Inference failed: ${message}`;
    toast('Inference failed', message, 'error');
    Motion.fadeOut(progressContainer, 4);
}

// WebSocket inference runner
function runWebSocketInference(signalId) {
    startPipeline();
    setStatus('Running detection pipeline');

    const ws = new WebSocket(`${WS_BASE}/ws/infer/${signalId}`);
    let settled = false;

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.progress) setProgress(msg.progress);
        if (msg.message) progressStep.textContent = msg.message;
        if (msg.step && msg.step !== 'error') setPipelineStep(msg.step);

        if (msg.step === 'completed' && msg.result) {
            settled = true;
            setStatus('Inference complete. Picks and embedding indexed.');
            finishPipeline();
            renderInferenceResults(msg.result);
        } else if (msg.step === 'error') {
            settled = true;
            setStatus(`Error: ${msg.message}`);
            failPipeline(msg.message);
        }
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        if (settled) return;
        settled = true;
        setStatus('WebSocket unavailable. Falling back to HTTP.');
        runHttpInference(signalId);
    };
}

// Fallback HTTP inference
async function runHttpInference(signalId) {
    try {
        progressStep.textContent = 'Running pipeline over HTTP';
        setPipelineStep('eqtransformer');
        setProgress(60);
        const res = await fetch(`${API_BASE}/infer/${signalId}`, { method: 'POST' });
        const data = await res.json();
        if (!data.result) throw new Error(data.detail || 'No result returned');
        setStatus('Inference complete.');
        finishPipeline();
        renderInferenceResults(data.result);
    } catch (err) {
        setStatus(`Inference error: ${err.message}`);
        failPipeline(err.message);
    }
}

// ---------- Results ----------

function renderPick(phase, pick, offset, span) {
    const key = phase.toLowerCase();
    const valEl = phase === 'P' ? valP : valS;
    const mark = $(`pt-${key}`);
    if (!pick) {
        valEl.textContent = 'Not picked';
        $(`conf-${key}`).textContent = '—';
        return;
    }
    valEl.textContent = pick.pick_time ? stripZone(pick.pick_time).split('T')[1].slice(0, 12) : '—';
    Motion.countUp($(`conf-${key}`), pick.confidence * 100, v => `${v.toFixed(0)}%`, { delay: 0.2 });
    Motion.meter($(`meter-${key}`), pick.confidence * 100, 0.2);
    if (offset !== null && span > 0) {
        mark.style.left = `${(offset / span) * 100}%`;
        mark.style.opacity = 1;
    }
}

// Render inference results to UI
function renderInferenceResults(res) {
    currentEventId = res.event_id;
    reportBtn.disabled = false;
    aiBtn.disabled = false;
    setLoading(false);

    // Detections
    Motion.countUp(valProb, res.detection_prob * 100, v => `${v.toFixed(1)}%`);
    Motion.meter($('meter-prob'), res.detection_prob * 100);
    if (res.magnitude_proxy !== undefined && res.magnitude_proxy !== null) {
        Motion.countUp(valMag, res.magnitude_proxy, v => `M ${v.toFixed(1)}`, { delay: 0.08 });
    } else {
        valMag.textContent = '—';
    }
    valOrigin.textContent = formatUtc(res.origin_time);

    // Picks
    const picks = res.picks || [];
    const pPick = picks.find(p => p.phase === 'P');
    const sPick = picks.find(p => p.phase === 'S');
    const traces = currentWaveformData || [];
    const span = traces.length ? Math.max(...traces.map(t => t.times[t.times.length - 1] || 0)) : 0;
    const pOff = pPick ? WavePlot.pickOffset(pPick, traces) : null;
    const sOff = sPick ? WavePlot.pickOffset(sPick, traces) : null;

    renderPick('P', pPick, pOff, span);
    renderPick('S', sPick, sOff, span);

    if (pOff !== null && sOff !== null && sOff > pOff) {
        valSp.textContent = `${(sOff - pOff).toFixed(2)} s`;
        const win = $('pt-window');
        win.style.left = `${(pOff / span) * 100}%`;
        win.style.width = `${((sOff - pOff) / span) * 100}%`;
        win.style.opacity = 1;
    }
    Motion.stagger(document.querySelectorAll('#pick-timeline > .pt-mark, #pick-timeline > .pt-window'), 0.3);

    // Re-render waveform with pick lines
    if (currentWaveformData) {
        plotWaveform(currentWaveformData, picks, 'picks').then(marks => {
            fitPicksBtn.disabled = !(marks.p || marks.s);
        });
    }

    renderSimilarEvents(res.similar_events || []);

    if (res.explanation) {
        aiBody.innerHTML = `<p class="narrative">${escapeHtml(res.explanation)}</p>`;
        Motion.stagger(aiBody.children);
    }
}

function renderSimilarEvents(similarEvents) {
    if (!similarEvents || similarEvents.length === 0) {
        similarList.innerHTML = '<p class="panel-empty">No close matches in the vector index.</p>';
        return;
    }

    similarList.innerHTML = similarEvents.map((item, i) => `
        <div class="similar-item">
            <span class="sim-rank">${String(i + 1).padStart(2, '0')}</span>
            <div>
                <span class="sim-id mono">${escapeHtml(item.similar_event_id.slice(0, 8))}</span>
                <span class="sim-time mono">${escapeHtml(/^\d{4}-/.test(item.time) ? formatUtc(item.time) : item.time)}</span>
            </div>
            <div>
                <span class="sim-score mono">${(item.score * 100).toFixed(1)}%</span>
                <span class="sim-bar"><span class="sim-fill" style="width: ${(item.score * 100).toFixed(0)}%;"></span></span>
            </div>
        </div>
    `).join('');
    Motion.stagger(similarList.children, 0.15);
}

function renderQuality(q) {
    Motion.countUp(valSnr, q.snr, v => v.toFixed(1));
    Motion.countUp(valQuality, q.quality_score * 100, v => `${v.toFixed(0)}%`, { delay: 0.06 });
    Motion.meter($('meter-quality'), q.quality_score * 100);
    const clean = q.noise_label === 'clean';
    const label = q.noise_label.replace(/_/g, ' ');
    valNoise.textContent = label.charAt(0).toUpperCase() + label.slice(1);
    valNoise.dataset.tone = clean ? 'ok' : 'bad';
    const clipped = q.is_clipped ?? q.clipping_flag ?? false;
    valClip.textContent = clipped ? 'Saturated' : 'None detected';
}

function handleSignalLoaded(data, label) {
    currentSignalId = data.signal_id;
    setStatus(`${label}. Running inference.`);
    renderQuality(data.quality);
    // Load waveform & run WebSocket pipeline
    loadAndPlotWaveform(currentSignalId);
    runWebSocketInference(currentSignalId);
}

function handleUploadSuccess(data) {
    handleSignalLoaded(data, `Uploaded ${data.filename}`);
}

// ---------- Sources ----------

async function uploadFileObject(file) {
    setStatus(`Uploading ${file.name}`);
    const formData = new FormData();
    formData.append('file', file, file.name);

    try {
        const response = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Upload failed (status ${response.status}): ${errText}`);
        }
        handleUploadSuccess(await response.json());
    } catch (err) {
        setStatus(`Error: ${err.message}`);
        toast('Upload failed', err.message, 'error');
    }
}

// Hidden HTML5 file input for standard web browser environments
const webFileInput = document.createElement('input');
webFileInput.type = 'file';
webFileInput.accept = '.mseed,.miniseed';
webFileInput.hidden = true;
document.body.appendChild(webFileInput);

webFileInput.addEventListener('change', async () => {
    const file = webFileInput.files[0];
    if (file) await uploadFileObject(file);
    webFileInput.value = '';
});

// 1. Upload Button Handler (Supports both Electron and Web Browsers)
uploadBtn.addEventListener('click', async () => {
    if (window.electronAPI && typeof window.electronAPI.openFile === 'function') {
        setStatus('Choosing a file');
        const filePath = await window.electronAPI.openFile();
        if (!filePath) {
            setStatus('File selection cancelled.');
            return;
        }

        setStatus(`Uploading ${filePath.split(/[\\/]/).pop()}`);
        const res = await window.electronAPI.uploadFile(filePath);

        if (!res.success) {
            setStatus(`Error: ${res.error}`);
            toast('Upload failed', res.error, 'error');
            return;
        }

        handleUploadSuccess(res.data);
    } else {
        webFileInput.click();
    }
});

// Drag and drop a .mseed file anywhere on the window
let dragDepth = 0;
window.addEventListener('dragenter', e => {
    if (!e.dataTransfer || !Array.from(e.dataTransfer.types).includes('Files')) return;
    dragDepth += 1;
    document.body.classList.add('is-dragging');
});
window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) document.body.classList.remove('is-dragging');
});
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove('is-dragging');
    const file = e.dataTransfer && e.dataTransfer.files[0];
    if (!file) return;
    if (!/\.(mseed|miniseed)$/i.test(file.name)) {
        toast('Unsupported file', `${file.name} is not a MiniSEED file (.mseed).`, 'error');
        return;
    }
    uploadFileObject(file);
});

// 2. Demo Waveform Button Handler
demoBtn.addEventListener('click', async () => {
    setStatus('Synthesising demo station stream');
    demoBtn.classList.add('is-busy');
    try {
        const res = await fetch(`${API_BASE}/fetch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                network: 'CI',
                station: 'PAS',
                channel: 'HHZ',
                starttime: new Date().toISOString().slice(0, 19),
                endtime: new Date(Date.now() + 60000).toISOString().slice(0, 19),
                source: 'scedc'
            })
        });
        if (!res.ok) throw new Error(`Backend returned ${res.status}`);
        handleSignalLoaded(await res.json(), 'Demo stream CI.PAS loaded');
    } catch (err) {
        setStatus(`Demo stream error: ${err.message}`);
        toast('Demo stream unavailable', `${err.message}. Check that the backend is running on 127.0.0.1:8000.`, 'error');
    } finally {
        demoBtn.classList.remove('is-busy');
    }
});

// 3. Fetch from SCEDC / EIDA drawer
function openFetch() {
    fetchError.hidden = true;
    Motion.openDrawer(fetchModal);
    setTimeout(() => $('fetch-network').focus(), 200);
}

function closeFetch() {
    Motion.closeDrawer(fetchModal);
}

fetchBtn.addEventListener('click', openFetch);
fetchModal.querySelectorAll('[data-close]').forEach(el => el.addEventListener('click', closeFetch));
window.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !fetchModal.hidden) closeFetch();
});

const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?$/;

fetchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
        source: fetchForm.querySelector('input[name="fetch-source"]:checked').value,
        network: $('fetch-network').value.trim().toUpperCase(),
        station: $('fetch-station').value.trim().toUpperCase(),
        channel: $('fetch-channel').value.trim().toUpperCase(),
        starttime: $('fetch-start').value.trim(),
        endtime: $('fetch-end').value.trim()
    };

    // Validate before sending
    const problems = [];
    ['fetch-network', 'fetch-station', 'fetch-channel', 'fetch-start', 'fetch-end'].forEach(id => $(id).removeAttribute('aria-invalid'));
    ['network', 'station', 'channel'].forEach(k => { if (!payload[k]) { problems.push(`${k} is required`); $(`fetch-${k}`).setAttribute('aria-invalid', 'true'); } });
    ['starttime', 'endtime'].forEach(k => {
        const id = k === 'starttime' ? 'fetch-start' : 'fetch-end';
        if (!ISO_RE.test(payload[k])) { problems.push(`${k === 'starttime' ? 'Start' : 'End'} time must look like 2021-01-01T00:00:00`); $(id).setAttribute('aria-invalid', 'true'); }
    });
    if (!problems.length && Date.parse(payload.endtime + 'Z') <= Date.parse(payload.starttime + 'Z')) {
        problems.push('End time must be after start time');
        $('fetch-end').setAttribute('aria-invalid', 'true');
    }
    if (problems.length) {
        fetchError.textContent = problems.join('. ') + '.';
        fetchError.hidden = false;
        return;
    }

    closeFetch();
    setStatus(`Fetching ${payload.network}.${payload.station} from ${payload.source.toUpperCase()}`);

    try {
        const res = await fetch(`${API_BASE}/fetch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('FDSN query failed');
        const data = await res.json();
        handleSignalLoaded(data, `Fetched ${data.network}.${data.station}`);
    } catch (err) {
        setStatus(`Fetch error: ${err.message}`);
        toast('Fetch failed', `${err.message}. Check the station codes and time window.`, 'error');
    }
});

// 4. Generate PDF Report Handler
reportBtn.addEventListener('click', async () => {
    if (!currentEventId) return;

    setStatus('Generating PDF report');
    reportBtn.classList.add('is-busy');
    try {
        const res = await fetch(`${API_BASE}/report/${currentEventId}`);
        if (!res.ok) throw new Error('Report generation failed');

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.hidden = true;
        a.href = url;
        a.download = `SeismoDetect_Report_${currentEventId.slice(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        setStatus('PDF report saved.');
        toast('Report exported', a.download);
    } catch (err) {
        setStatus(`PDF error: ${err.message}`);
        toast('Export failed', err.message, 'error');
    } finally {
        reportBtn.classList.remove('is-busy');
    }
});

// 5. AI summary (RAG report): queue, then poll until done
function renderAiReport(report) {
    const f = report.forecast || {};
    const pga = f.pga_g_median !== undefined ? `${f.pga_g_median} g` : '—';
    const parts = [];
    if (report.forecast) {
        parts.push(`
            <div class="forecast">
                <div><span>PGA median</span><strong>${escapeHtml(pga)}</strong></div>
                <div><span>MMI</span><strong>${escapeHtml(f.mmi ?? '—')}</strong></div>
                <div><span>Damage</span><strong>${escapeHtml(f.damage_band ?? '—')}</strong></div>
            </div>`);
    }
    // Narrative is light markdown: "## HEADING" lines followed by paragraphs
    (report.narrative || 'The report finished without narrative text.')
        .split(/\n{2,}|\n(?=#{1,3} )/)
        .map(block => block.trim())
        .filter(Boolean)
        .forEach(block => {
            const heading = block.match(/^#{1,3}\s+(.+?)(?:\n([\s\S]*))?$/);
            if (heading) {
                const title = heading[1].toLowerCase().replace(/^\w/, c => c.toUpperCase());
                parts.push(`<h4 class="ai-heading">${escapeHtml(title)}</h4>`);
                if (heading[2]) parts.push(`<p class="narrative">${escapeHtml(heading[2].trim())}</p>`);
            } else {
                parts.push(`<p class="narrative">${escapeHtml(block)}</p>`);
            }
        });
    aiBody.innerHTML = parts.join('');
    Motion.stagger(aiBody.children);
}

async function pollAiReport(eventId, attempt = 0) {
    if (eventId !== currentEventId) return;
    try {
        const res = await fetch(`${API_BASE}/api/events/${eventId}/report`);
        const report = await res.json();
        if (report.status === 'done') {
            renderAiReport(report);
            aiBtn.disabled = false;
            aiBtn.lastChild.textContent = ' Regenerate summary';
            return;
        }
        if (report.status === 'error') throw new Error(report.error_msg || 'Report pipeline failed');
        if (attempt > 90) throw new Error('Timed out waiting for the report');
        aiPollTimer = setTimeout(() => pollAiReport(eventId, attempt + 1), 2000);
    } catch (err) {
        aiBody.innerHTML = `<p class="panel-empty">Summary failed: ${escapeHtml(err.message)}</p>`;
        aiBtn.disabled = false;
        toast('Summary failed', err.message, 'error');
    }
}

aiBtn.addEventListener('click', async () => {
    if (!currentEventId) return;
    aiBtn.disabled = true;
    aiBody.innerHTML = '<div class="ai-status"><span class="spinner"></span>Retrieving context and writing summary</div>';
    try {
        const res = await fetch(`${API_BASE}/api/events/${currentEventId}/report`, { method: 'POST' });
        if (!res.ok) throw new Error(`Backend returned ${res.status}`);
        pollAiReport(currentEventId);
    } catch (err) {
        aiBody.innerHTML = `<p class="panel-empty">Summary failed: ${escapeHtml(err.message)}</p>`;
        aiBtn.disabled = false;
        toast('Summary failed', err.message, 'error');
    }
});

// ---------- Plot toolbar ----------

document.querySelectorAll('.comp-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
        const on = btn.getAttribute('aria-pressed') !== 'true';
        const active = document.querySelectorAll('.comp-toggle[aria-pressed="true"]').length;
        if (!on && active <= 1) return; // keep at least one lane
        btn.setAttribute('aria-pressed', String(on));
        WavePlot.setVisible(btn.dataset.comp, on);
    });
});

fitPicksBtn.addEventListener('click', () => WavePlot.fitPicks());
$('reset-zoom-btn').addEventListener('click', () => WavePlot.resetZoom());
$('export-png-btn').addEventListener('click', () => WavePlot.exportPng(`SeismoDetect_${currentStationLabel}`));

// Empty-state buttons forward to the main source actions
document.querySelectorAll('[data-proxy]').forEach(btn => {
    btn.addEventListener('click', () => $(btn.dataset.proxy).click());
});

// ---------- Theme ----------

const themeBtn = $('theme-btn');

function syncThemeChrome(theme) {
    const next = theme === 'dark' ? 'light' : 'dark';
    themeBtn.setAttribute('aria-label', `Switch to ${next} theme`);
    // Keep the native window buttons in step with the top bar
    if (window.electronAPI && typeof window.electronAPI.setTheme === 'function') {
        window.electronAPI.setTheme(theme);
    }
}

themeBtn.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    Motion.themeSwap(() => { document.documentElement.dataset.theme = theme; });
    try { localStorage.setItem('seismo-theme', theme); } catch (e) {}
    syncThemeChrome(theme);
});

document.addEventListener('themechange', () => WavePlot.refreshTheme());
syncThemeChrome(document.documentElement.dataset.theme);

// ---------- Boot ----------

Motion.emptyTrace($('empty-trace'));
Motion.initLenis($('inspector'), document.querySelector('.inspector-content'));
Motion.splash(() => Motion.intro());
