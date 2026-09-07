// DOM Elements
const uploadBtn = document.getElementById('upload-btn');
const fetchBtn = document.getElementById('fetch-btn');
const demoBtn = document.getElementById('demo-btn');
const reportBtn = document.getElementById('report-btn');
const statusText = document.getElementById('status-text');
const backendStatus = document.getElementById('backend-status');
const backendBadge = document.getElementById('backend-badge');
const stationBadge = document.getElementById('active-station-text');
const traceMeta = document.getElementById('trace-meta');

// Quality & Metrics
const valSnr = document.getElementById('val-snr');
const valNoise = document.getElementById('val-noise');
const valClip = document.getElementById('val-clip');
const valQuality = document.getElementById('val-quality');

// Detections
const valProb = document.getElementById('val-prob');
const valMag = document.getElementById('val-mag');
const valP = document.getElementById('val-p');
const valS = document.getElementById('val-s');
const valOrigin = document.getElementById('val-origin');
const similarList = document.getElementById('similar-list');

// Progress
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressPercent = document.getElementById('progress-percent');
const progressStep = document.getElementById('progress-step');

// Modal Elements
const fetchModal = document.getElementById('fetch-modal');
const modalCancelBtn = document.getElementById('modal-cancel-btn');
const modalSubmitBtn = document.getElementById('modal-submit-btn');

let currentSignalId = null;
let currentEventId = null;
let currentWaveformData = null;

const isLocalFile = window.location.protocol === 'file:';
const API_BASE = isLocalFile ? 'http://127.0.0.1:8000' : window.location.origin;
const WS_BASE = isLocalFile ? 'ws://127.0.0.1:8000' : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

// Check backend connectivity
async function checkBackend() {
    try {
        const res = await fetch(`${API_BASE}/`);
        if (res.ok) {
            backendStatus.innerText = "Backend Connected";
            backendBadge.className = "badge";
        } else {
            throw new Error();
        }
    } catch {
        backendStatus.innerText = "Backend Offline";
        backendBadge.className = "badge offline";
    }
}

setInterval(checkBackend, 5000);
checkBackend();

// Initialize Plotly with an empty state
function initEmptyPlot() {
    const plotDiv = document.getElementById('waveform-plot');
    const layout = {
        paper_bgcolor: '#0d1322',
        plot_bgcolor: '#0d1322',
        font: { color: '#94a3b8', family: '-apple-system, sans-serif' },
        margin: { t: 30, l: 50, r: 20, b: 40 },
        xaxis: { title: 'Time (seconds)', gridcolor: '#1e293b', zerolinecolor: '#334155' },
        yaxis: { title: 'Amplitude (Counts)', gridcolor: '#1e293b', zerolinecolor: '#334155' },
        showlegend: true,
        legend: { orientation: 'h', y: 1.1, x: 0.7 }
    };
    Plotly.newPlot(plotDiv, [], layout, { responsive: true });
}
initEmptyPlot();

// Plot waveform traces with accurate P/S arrival annotations
function plotWaveform(traces, picks = []) {
    currentWaveformData = traces;
    const plotDiv = document.getElementById('waveform-plot');
    const colors = { 'Z': '#3b82f6', 'N': '#10b981', 'E': '#f59e0b' };

    const plotlyData = traces.map(tr => {
        const comp = tr.component || tr.channel.slice(-1) || 'Z';
        return {
            x: tr.times,
            y: tr.data,
            type: 'scatter',
            mode: 'lines',
            name: `${tr.channel} (${comp})`,
            line: { color: colors[comp] || '#06b6d4', width: 1.2 }
        };
    });

    const shapes = [];
    const annotations = [];

    picks.forEach(p => {
        const isP = p.phase === 'P';
        const pickColor = isP ? '#3b82f6' : '#f59e0b';

        // Priority 1: use offset_s (seconds from stream start) — computed by backend from pick_time - starttime
        // Priority 2: sample_index / sample_rate
        // Priority 3: fallback not used (undefined position is better than a wrong hardcoded one)
        let tVal = null;
        if (p.offset_s !== undefined && p.offset_s !== null) {
            tVal = p.offset_s;
        } else if (p.sample_index !== undefined && p.sample_index !== null) {
            const fs = (traces[0] && traces[0].sample_rate) || 100.0;
            tVal = p.sample_index / fs;
        }

        if (tVal === null) return;  // Skip annotation if we have no valid time

        shapes.push({
            type: 'line',
            x0: tVal,
            x1: tVal,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: pickColor, width: 2, dash: 'dash' }
        });

        annotations.push({
            x: tVal,
            y: 0.97,
            yref: 'paper',
            text: `<b>${p.phase}-wave</b><br>${(p.confidence * 100).toFixed(1)}%`,
            showarrow: true,
            arrowhead: 2,
            arrowcolor: pickColor,
            font: { color: '#ffffff', size: 10 },
            bgcolor: pickColor,
            borderpad: 4,
            ax: isP ? -30 : 30,
            ay: -20
        });
    });

    // Determine the x-axis range from the actual trace time span
    const allTimes = traces.flatMap(tr => tr.times);
    const tMin = allTimes.length ? Math.min(...allTimes) : 0;
    const tMax = allTimes.length ? Math.max(...allTimes) : 60;

    const layout = {
        paper_bgcolor: '#0d1322',
        plot_bgcolor: '#0d1322',
        font: { color: '#94a3b8', family: '-apple-system, sans-serif' },
        margin: { t: 30, l: 60, r: 20, b: 40 },
        xaxis: {
            title: 'Time (seconds from stream start)',
            gridcolor: '#1e293b',
            zerolinecolor: '#334155',
            range: [tMin, tMax]
        },
        yaxis: { title: 'Amplitude (Counts)', gridcolor: '#1e293b', zerolinecolor: '#334155' },
        shapes: shapes,
        annotations: annotations,
        showlegend: true,
        legend: { orientation: 'h', y: 1.12, x: 0.6 }
    };

    Plotly.react(plotDiv, plotlyData, layout, { responsive: true });
}


// Download and plot waveform data
async function loadAndPlotWaveform(signalId) {
    try {
        const res = await fetch(`${API_BASE}/waveform/${signalId}`);
        if (!res.ok) throw new Error("Failed to load waveform");
        const data = await res.json();
        traceMeta.innerText = `${data.network}.${data.station} • ${data.traces.length} Components • 100 Hz`;
        stationBadge.innerText = `${data.network}.${data.station}`;
        plotWaveform(data.traces);
    } catch (err) {
        console.error("Waveform load error:", err);
    }
}

// WebSocket inference runner
function runWebSocketInference(signalId) {
    progressContainer.style.display = 'block';
    progressFill.style.width = '10%';
    progressPercent.innerText = '10%';
    progressStep.innerText = 'Connecting to inference server...';
    statusText.innerText = 'Running deep learning pipeline...';

    const ws = new WebSocket(`${WS_BASE}/ws/infer/${signalId}`);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.progress) {
            progressFill.style.width = `${msg.progress}%`;
            progressPercent.innerText = `${msg.progress}%`;
        }
        if (msg.message) {
            progressStep.innerText = msg.message;
        }

        if (msg.step === 'completed' && msg.result) {
            statusText.innerText = 'Inference complete! Picks and vectors indexed.';
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 1000);
            renderInferenceResults(msg.result);
        } else if (msg.step === 'error') {
            statusText.innerText = `Error: ${msg.message}`;
            progressStep.innerText = `Inference failed: ${msg.message}`;
        }
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        statusText.innerText = 'WebSocket connection error. Falling back to HTTP...';
        runHttpInference(signalId);
    };
}

// Fallback HTTP inference
async function runHttpInference(signalId) {
    try {
        const res = await fetch(`${API_BASE}/infer/${signalId}`, { method: 'POST' });
        const data = await res.json();
        if (data.result) {
            progressContainer.style.display = 'none';
            statusText.innerText = 'Inference complete!';
            renderInferenceResults(data.result);
        }
    } catch (err) {
        statusText.innerText = `Inference error: ${err.message}`;
    }
}

// Render inference results to UI
function renderInferenceResults(res) {
    currentEventId = res.event_id;
    reportBtn.disabled = false;

    // Detections
    valProb.innerText = `${(res.detection_prob * 100).toFixed(1)}%`;
    valMag.innerText = `M ${res.magnitude_proxy || '2.8'}`;
    valOrigin.innerText = res.origin_time || '--';

    // Picks
    const pPick = res.picks.find(p => p.phase === 'P');
    const sPick = res.picks.find(p => p.phase === 'S');

    if (pPick) {
        valP.innerText = `P: ${(pPick.confidence * 100).toFixed(0)}% (${pPick.pick_time.split('T')[1].slice(0, 8)})`;
    } else {
        valP.innerText = 'P: None';
    }

    if (sPick) {
        valS.innerText = `S: ${(sPick.confidence * 100).toFixed(0)}% (${sPick.pick_time.split('T')[1].slice(0, 8)})`;
    } else {
        valS.innerText = 'S: None';
    }

    // Re-render waveform with pick lines
    if (currentWaveformData) {
        plotWaveform(currentWaveformData, res.picks);
    }

    // Render Similar Events
    renderSimilarEvents(res.similar_events || []);
}

function renderSimilarEvents(similarEvents) {
    if (!similarEvents || similarEvents.length === 0) {
        similarList.innerHTML = '<div style="font-size: 12px; color: var(--text-muted); text-align: center;">No vector matches found.</div>';
        return;
    }

    similarList.innerHTML = similarEvents.map(item => `
        <div class="similar-item">
            <div>
                <div style="font-weight: 600; color: #f1f5f9;">Event ${item.similar_event_id.slice(0, 8)}...</div>
                <div style="font-size: 10px; color: #94a3b8;">${item.time}</div>
                <div class="sim-bar">
                    <div class="sim-fill" style="width: ${(item.score * 100).toFixed(0)}%;"></div>
                </div>
            </div>
            <div style="font-weight: 700; color: #06b6d4; font-size: 13px;">
                ${(item.score * 100).toFixed(1)}%
            </div>
        </div>
    `).join('');
}

function handleUploadSuccess(data) {
    currentSignalId = data.signal_id;
    statusText.innerText = `Uploaded ${data.filename}. Initializing inference...`;

    // Populate quality metrics
    valSnr.innerText = `${data.quality.snr} dB`;
    valNoise.innerText = data.quality.noise_label.toUpperCase();
    valNoise.className = `pill ${data.quality.noise_label === 'clean' ? 'clean' : 'noise'}`;
    valClip.innerText = data.quality.is_clipped ? 'Yes (Saturated)' : 'No (Clean)';
    valQuality.innerText = `${(data.quality.quality_score * 100).toFixed(0)}%`;

    // Load waveform & run WebSocket pipeline
    loadAndPlotWaveform(currentSignalId);
    runWebSocketInference(currentSignalId);
}

// Hidden HTML5 file input for standard web browser environments
const webFileInput = document.createElement('input');
webFileInput.type = 'file';
webFileInput.accept = '.mseed';
webFileInput.style.display = 'none';
document.body.appendChild(webFileInput);

webFileInput.addEventListener('change', async () => {
    const file = webFileInput.files[0];
    if (!file) return;

    statusText.innerText = `Uploading ${file.name} to MinIO...`;
    const formData = new FormData();
    formData.append('file', file, file.name);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Upload failed (Status ${response.status}): ${errText}`);
        }

        const data = await response.json();
        handleUploadSuccess(data);
    } catch (err) {
        statusText.innerText = `Error: ${err.message}`;
        alert(`Upload Failed: ${err.message}`);
    } finally {
        webFileInput.value = '';
    }
});

// 1. Upload Button Handler (Supports both Electron and Web Browsers)
uploadBtn.addEventListener('click', async () => {
    if (window.electronAPI && typeof window.electronAPI.openFile === 'function') {
        statusText.innerText = 'Opening file selection dialog...';
        const filePath = await window.electronAPI.openFile();
        if (!filePath) {
            statusText.innerText = 'File selection cancelled.';
            return;
        }

        statusText.innerText = `Uploading ${filePath.split('\\').pop()} to MinIO...`;
        const res = await window.electronAPI.uploadFile(filePath);

        if (!res.success) {
            statusText.innerText = `Error: ${res.error}`;
            alert(`Upload Failed: ${res.error}`);
            return;
        }

        handleUploadSuccess(res.data);
    } else {
        // Fallback for regular web browser
        webFileInput.click();
    }
});

// 2. Demo Waveform Button Handler
demoBtn.addEventListener('click', async () => {
    statusText.innerText = 'Synthesizing live seismic station stream...';
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
        const data = await res.json();
        currentSignalId = data.signal_id;
        statusText.innerText = `Station CI.PAS stream loaded. Running ML picking...`;

        valSnr.innerText = `${data.quality.snr} dB`;
        valNoise.innerText = data.quality.noise_label.toUpperCase();
        valNoise.className = `pill ${data.quality.noise_label === 'clean' ? 'clean' : 'noise'}`;
        valClip.innerText = 'No (Clean)';
        valQuality.innerText = `${(data.quality.quality_score * 100).toFixed(0)}%`;

        await loadAndPlotWaveform(currentSignalId);
        runWebSocketInference(currentSignalId);
    } catch (err) {
        statusText.innerText = `Demo stream error: ${err.message}`;
    }
});

// 3. Fetch from SCEDC / EIDA Modal Handlers
fetchBtn.addEventListener('click', () => {
    fetchModal.style.display = 'flex';
});

modalCancelBtn.addEventListener('click', () => {
    fetchModal.style.display = 'none';
});

modalSubmitBtn.addEventListener('click', async () => {
    const payload = {
        source: document.getElementById('fetch-source').value,
        network: document.getElementById('fetch-network').value,
        station: document.getElementById('fetch-station').value,
        channel: document.getElementById('fetch-channel').value,
        starttime: document.getElementById('fetch-start').value,
        endtime: document.getElementById('fetch-end').value
    };

    fetchModal.style.display = 'none';
    statusText.innerText = `Fetching ${payload.network}.${payload.station} from ${payload.source.toUpperCase()}...`;

    try {
        const res = await fetch(`${API_BASE}/fetch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("FDSN query failed");
        const data = await res.json();
        currentSignalId = data.signal_id;
        statusText.innerText = `Fetched ${data.network}.${data.station}. Evaluating...`;

        valSnr.innerText = `${data.quality.snr} dB`;
        valNoise.innerText = data.quality.noise_label.toUpperCase();
        valNoise.className = `pill ${data.quality.noise_label === 'clean' ? 'clean' : 'noise'}`;
        valClip.innerText = 'No (Clean)';
        valQuality.innerText = `${(data.quality.quality_score * 100).toFixed(0)}%`;

        await loadAndPlotWaveform(currentSignalId);
        runWebSocketInference(currentSignalId);
    } catch (err) {
        statusText.innerText = `Fetch error: ${err.message}`;
    }
});

// 4. Generate PDF Report Handler
reportBtn.addEventListener('click', async () => {
    if (!currentEventId) return;

    statusText.innerText = 'Generating official ReportLab PDF...';
    try {
        const reportUrl = `${API_BASE}/report/${currentEventId}`;
        // Trigger download via browser/Electron
        const res = await fetch(reportUrl);
        if (!res.ok) throw new Error("Report generation failed");
        
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `SeismoDetect_Report_${currentEventId.slice(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        statusText.innerText = 'PDF Report successfully generated and downloaded!';
    } catch (err) {
        statusText.innerText = `PDF error: ${err.message}`;
    }
});
