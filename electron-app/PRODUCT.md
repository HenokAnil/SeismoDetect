# SeismoDetect Desktop — Product

## What it is
Desktop workbench for analysing a single seismic recording end to end: load a MiniSEED stream (upload, FDSN fetch from SCEDC/EIDA, or demo), check signal quality, run EQTransformer detection and P/S phase picking, find similar historical events via 128-d embeddings in Qdrant, and export a PDF report with an optional RAG-generated narrative.

## Who uses it
Seismology students, researchers and evaluators reviewing the project. Sessions are focused and short: load one stream, read the picks, judge the confidence, export. Often shown on a projector or shared screen during presentations, so it must read clearly from a distance.

## Surface mode
Operate. The waveform is the work object; everything else is a readout about it. Scanability and trust in the numbers outrank expression.

## Tone
Precise, calm, instrument-grade. Like a well-made lab instrument, not a marketing dashboard. No hype copy, no emoji.

## Product truths to keep
- Three components (Z/N/E), P and S picks with confidence and arrival time, origin time, detection probability, magnitude proxy.
- Signal quality: SNR (dB), noise classifier label, clipping flag, quality score.
- Similar events with similarity score and time.
- Live pipeline over WebSocket with HTTP fallback.
- PDF export; RAG narrative + GMPE forecast (PGA, MMI, damage band) when generated.

## Constraints
- Electron 28, vanilla JS, no bundler. Must work offline: all fonts and libraries load from node_modules.
- Backend at http://127.0.0.1:8000.
