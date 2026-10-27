// Waveform plot: one lane per component (Z/N/E) on a shared time axis,
// P/S pick markers spanning all lanes and a shaded S–P window.
(function () {
    // Colours come from the active theme's CSS tokens; re-read on theme change
    let COLORS = {};
    function readColors() {
        const css = getComputedStyle(document.documentElement);
        const token = name => css.getPropertyValue(name).trim();
        COLORS = {
            Z: token('--comp-z'), N: token('--comp-n'), E: token('--comp-e'),
            P: token('--phase-p'), S: token('--phase-s'), ink: token('--pick-ink'),
            grid: token('--plot-grid'), zero: token('--plot-zero'), spike: token('--plot-spike'),
            window: token('--plot-window'),
            text: token('--text-3'), text2: token('--text-2'), textStrong: token('--text'),
            bg: token('--bg'), surface: token('--surface'), surface3: token('--surface-3'),
            line: token('--line'), lineStrong: token('--line-strong')
        };
    }
    readColors();
    const MONO = "'Cascadia Mono', 'Cascadia Code', Consolas, monospace";
    const ORDER = ['Z', 'N', 'E'];
    const LANE_GAP = 0.05;

    const state = { div: null, traces: [], picks: [], hidden: new Set(), revision: 0, span: [0, 60] };

    function componentOf(tr) {
        return (tr.component || (tr.channel || '').slice(-1) || 'Z').toUpperCase();
    }

    // ISO strings from the backend have no zone; treat both sides as UTC so the offset is exact
    function parseUtc(iso) {
        if (!iso) return NaN;
        const trimmed = iso.replace(/(\.\d{3})\d+/, '$1').replace(/Z$|[+-]\d\d:\d\d$/, '');
        return Date.parse(trimmed + 'Z');
    }

    // Seconds from stream start: offset_s, then sample_index, then pick_time vs trace start
    function pickOffset(p, traces) {
        if (p.offset_s !== undefined && p.offset_s !== null) return p.offset_s;
        const fs = (traces[0] && traces[0].sample_rate) || 100.0;
        if (p.sample_index !== undefined && p.sample_index !== null) return p.sample_index / fs;
        const start = traces[0] && parseUtc(traces[0].starttime);
        const t = parseUtc(p.pick_time);
        if (Number.isFinite(start) && Number.isFinite(t)) return (t - start) / 1000;
        return null;
    }

    function baseLayout() {
        return {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: COLORS.text, family: MONO, size: 11 },
            margin: { t: 34, l: 64, r: 20, b: 8 },
            showlegend: false,
            dragmode: 'zoom',
            hovermode: 'x unified',
            hoversubplots: 'axis',
            hoverdistance: -1,
            hoverlabel: {
                bgcolor: COLORS.surface3, bordercolor: COLORS.lineStrong,
                font: { family: MONO, size: 11, color: COLORS.textStrong }
            }
        };
    }

    function axisStyle() {
        return {
            gridcolor: COLORS.grid, zerolinecolor: COLORS.zero, zerolinewidth: 1,
            linecolor: 'rgba(0,0,0,0)', tickcolor: 'rgba(0,0,0,0)', ticklen: 4,
            tickfont: { family: MONO, size: 10.5, color: COLORS.text }
        };
    }

    function init(div) {
        state.div = div;
        Plotly.newPlot(div, [], baseLayout(), { responsive: true, displayModeBar: false, displaylogo: false });
    }

    function build() {
        const visible = state.traces
            .map(tr => ({ tr, comp: componentOf(tr) }))
            .filter(({ comp }) => !state.hidden.has(comp))
            .sort((a, b) => ORDER.indexOf(a.comp) - ORDER.indexOf(b.comp));

        const n = Math.max(visible.length, 1);
        const laneH = (1 - LANE_GAP * (n - 1)) / n;
        const layout = baseLayout();
        layout.uirevision = state.revision;

        layout.xaxis = Object.assign(axisStyle(), {
            // autorange + uirevision keeps the user's zoom across component toggles
            autorange: true,
            ticksuffix: ' s',
            showspikes: true, spikemode: 'across', spikesnap: 'cursor',
            spikecolor: COLORS.spike, spikethickness: 1, spikedash: 'solid',
            rangeslider: {
                visible: true, thickness: 0.09,
                bgcolor: COLORS.bg, bordercolor: COLORS.line, borderwidth: 1
            }
        });

        const data = [];
        const annotations = [];

        visible.forEach(({ tr, comp }, i) => {
            const axisId = i === 0 ? 'y' : `y${i + 1}`;
            const axisKey = i === 0 ? 'yaxis' : `yaxis${i + 1}`;
            // Lane 0 at the top
            const top = 1 - i * (laneH + LANE_GAP);
            layout[axisKey] = Object.assign(axisStyle(), {
                domain: [Math.max(0, top - laneH), top],
                fixedrange: true, nticks: 4, tickformat: '~s',
                showspikes: false
            });

            const color = COLORS[comp] || COLORS.Z;
            data.push({
                x: tr.times, y: tr.data,
                type: 'scatter', mode: 'lines',
                xaxis: 'x', yaxis: axisId,
                name: tr.channel || comp,
                line: { color, width: 1, simplify: true },
                hovertemplate: '%{y:.4~s}<extra></extra>'
            });

            annotations.push({
                xref: 'paper', yref: `${axisId} domain`, x: 0.004, y: 0.98,
                xanchor: 'left', yanchor: 'top', showarrow: false,
                text: `<b>${tr.channel || comp}</b>`,
                font: { family: MONO, size: 11, color },
                bgcolor: COLORS.surface, borderpad: 3
            });
        });

        // Time ticks sit under the bottom lane
        layout.xaxis.anchor = visible.length > 1 ? `y${visible.length}` : 'y';

        const shapes = [];
        const marks = state.picks
            .map(p => ({ phase: p.phase, t: pickOffset(p, state.traces), conf: p.confidence }))
            .filter(p => p.t !== null && Number.isFinite(p.t));

        const pMark = marks.find(m => m.phase === 'P');
        const sMark = marks.find(m => m.phase === 'S');

        if (pMark && sMark && sMark.t > pMark.t) {
            shapes.push({
                type: 'rect', xref: 'x', yref: 'paper',
                x0: pMark.t, x1: sMark.t, y0: 0, y1: 1,
                fillcolor: COLORS.window, line: { width: 0 }, layer: 'below'
            });
            annotations.push({
                xref: 'x', yref: 'paper', x: (pMark.t + sMark.t) / 2, y: 1, yanchor: 'bottom',
                showarrow: false,
                text: `S–P  ${(sMark.t - pMark.t).toFixed(2)} s`,
                font: { family: MONO, size: 10.5, color: COLORS.text2 }
            });
        }

        marks.forEach(m => {
            const color = m.phase === 'P' ? COLORS.P : COLORS.S;
            shapes.push({
                type: 'line', xref: 'x', yref: 'paper',
                x0: m.t, x1: m.t, y0: 0, y1: 1,
                line: { color, width: 1.5 }, layer: 'above'
            });
            annotations.push({
                xref: 'x', yref: 'paper', x: m.t, y: 1, yanchor: 'bottom',
                xanchor: m.phase === 'P' ? 'right' : 'left',
                showarrow: false,
                text: `<b>${m.phase}</b>  ${(m.conf * 100).toFixed(0)}%`,
                font: { family: MONO, size: 11, color: COLORS.ink },
                bgcolor: color, borderpad: 4
            });
        });

        layout.shapes = shapes;
        layout.annotations = annotations;
        return { data, layout, marks: { p: pMark, s: sMark } };
    }

    // options.reveal: 'traces' (new stream) or 'picks' (results arrived)
    function render(traces, picks = [], options = {}) {
        if (options.reveal === 'traces') state.revision += 1;
        state.traces = traces;
        state.picks = picks;

        const allTimes = traces.flatMap(tr => [tr.times[0], tr.times[tr.times.length - 1]]).filter(Number.isFinite);
        state.span = allTimes.length ? [Math.min(...allTimes), Math.max(...allTimes)] : [0, 60];

        const { data, layout, marks } = build();
        return Plotly.react(state.div, data, layout, { responsive: true, displayModeBar: false, displaylogo: false })
            .then(() => {
                if (options.reveal === 'traces') Motion.revealTraces(state.div);
                if (options.reveal === 'picks') Motion.revealPicks(state.div);
                return marks;
            });
    }

    function setVisible(comp, on) {
        on ? state.hidden.delete(comp) : state.hidden.add(comp);
        if (!state.traces.length) return;
        const { data, layout } = build();
        Plotly.react(state.div, data, layout, { responsive: true, displayModeBar: false, displaylogo: false });
    }

    function fitPicks() {
        const ts = state.picks.map(p => pickOffset(p, state.traces)).filter(t => t !== null && Number.isFinite(t));
        if (!ts.length) return;
        const lo = Math.min(...ts), hi = Math.max(...ts);
        const pad = Math.max(3, (hi - lo) * 0.6);
        Plotly.relayout(state.div, { 'xaxis.range': [Math.max(state.span[0], lo - pad), Math.min(state.span[1], hi + pad)] });
    }

    function resetZoom() {
        if (!state.traces.length) return;
        Plotly.relayout(state.div, { 'xaxis.range': state.span.slice() });
    }

    function exportPng(filename) {
        if (!state.traces.length) return;
        // Exported image gets a solid background so it reads outside the app
        const bg = { paper_bgcolor: COLORS.surface, plot_bgcolor: COLORS.surface };
        Plotly.relayout(state.div, bg)
            .then(() => Plotly.downloadImage(state.div, { format: 'png', filename, width: 1800, height: 900, scale: 1 }))
            .finally(() => Plotly.relayout(state.div, { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' }));
    }

    function resize() {
        if (state.div) Plotly.Plots.resize(state.div);
    }

    // Re-read theme tokens and redraw without touching the user's zoom
    function refreshTheme() {
        readColors();
        if (!state.div) return;
        if (!state.traces.length) { Plotly.relayout(state.div, baseLayout()); return; }
        const { data, layout } = build();
        Plotly.react(state.div, data, layout, { responsive: true, displayModeBar: false, displaylogo: false });
    }

    window.WavePlot = { init, render, setVisible, fitPicks, resetZoom, exportPng, resize, refreshTheme, pickOffset, hasData: () => state.traces.length > 0 };
})();
