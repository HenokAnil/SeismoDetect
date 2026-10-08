# SeismoDetect Machine Learning Models Documentation 🧠📊

This directory contains the machine learning models, database schemas, and detector pipelines utilized in the **SeismoDetect** platform.

---

## 📋 Overview of Models

| Model | Type | Architecture / Base | File / Module | Primary Task |
| :--- | :--- | :--- | :--- | :--- |
| **Noise Classifier** | Supervised Classifier | Random Forest (100 estimators, max depth 10) | [`noise_classifier_v1.pkl`](file:///c:/Users/henok/OneDrive/Documents/VsCode/Minor-miniproject/backend/models/noise_classifier_v1.pkl) | Classify waveform signal quality & noise categories |
| **EQTransformer** | Deep Learning Seq2Seq | CNN + Self-Attention (SeisBench) | [`detector.py`](file:///c:/Users/henok/OneDrive/Documents/VsCode/Minor-miniproject/backend/models/detector.py) | Earthquake detection & P/S phase arrival picking |
| **Waveform Encoder** | Deep Learning Feature Extractor | 1D-CNN + Global Pooling | [`backend/similarity/encoder.py`](file:///c:/Users/henok/OneDrive/Documents/VsCode/Minor-miniproject/backend/similarity/encoder.py) | 128-dimensional embedding generation for similarity search |
| **RAG Reporting Engine** | Natural Language Generation | FLAN-T5 / Deterministic Template Fallback | [`backend/reporting/rag_generator.py`](file:///c:/Users/henok/OneDrive/Documents/VsCode/Minor-miniproject/backend/reporting/rag_generator.py) | Factual automated PDF report generation |

---

## 1. Noise Classifier (`noise_classifier_v1.pkl`)

### Description
The Noise Classifier evaluates incoming 3-component seismic traces and determines whether the trace contains clean signal or non-earthquake noise artifacts.

### Input Features
The classifier receives 4 engineered features extracted by `backend/preprocessing/quality.py`:
1. **`avg_snr`**: Signal-to-Noise Ratio in dB.
2. **`avg_entropy`**: Spectral entropy from Welch power spectral density.
3. **`avg_kurtosis`**: Fourth standardized moment (detects sudden spikes/glitches).
4. **`is_clipped`**: Boolean flag (1.0/0.0) indicating 24-bit digitizer rail saturation.

### Target Classes
- `0: clean` (Clean seismic event candidate)
- `1: wind_noise` (High entropy, low SNR)
- `2: traffic_noise` (Moderate SNR, mid entropy)
- `3: instrument_glitch` (Extremely high kurtosis, clipping)
- `4: calibration_pulse` (High SNR, low entropy regular signal)

### Performance & Benchmark Metrics
- **Overall Training Accuracy:** **91.06%**
- **Quality Score Output:** Predicted probability score $P(\text{class} = \text{clean}) \in [0.0, 1.0]$
- **Model Size:** 3.74 MB

---

## 2. Earthquake Detection & Phase Picking (EQTransformer)

### Description
Implemented in [`detector.py`](file:///c:/Users/henok/OneDrive/Documents/VsCode/Minor-miniproject/backend/models/detector.py) via SeisBench. Uses a deep convolutional and self-attention transformer network to process 3-component waveforms ($3 \times 6000$ samples @ 100 Hz = 60s window) and produce 3 probability curves:
- **Detection Probability curve** (Presence of an earthquake)
- **P-phase arrival probability curve**
- **S-phase arrival probability curve**

### Evaluation Metrics (`backend/reports/evaluation_report.json`)

#### Earthquake Detection Metrics:
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **0.8988** (89.88%) | Ratio of true detected earthquakes to total detections |
| **Recall** | **0.9024** (90.24%) | Ratio of detected earthquakes to all actual events |
| **F1-Score** | **0.9006** (90.06%) | Harmonic mean of detection precision and recall |

#### Phase Picking Arrival Time Accuracy:
| Phase | Metric | Error | Description |
| :--- | :--- | :--- | :--- |
| **P-Wave Arrival** | **MAE (Mean Absolute Error)** | **~3.98 samples** (~0.040s @ 100Hz) | Average timing offset for primary phase picks |
| **S-Wave Arrival** | **MAE (Mean Absolute Error)** | **~8.07 samples** (~0.081s @ 100Hz) | Average timing offset for secondary phase picks |

*(Cumulative error distribution curves are generated at `backend/reports/pick_cdf.png`)*

---

## 3. Waveform Embedding Model (1D-CNN)

### Description
Implemented in [`backend/similarity/encoder.py`](file:///c:/Users/henok/OneDrive/Documents/VsCode/Minor-miniproject/backend/similarity/encoder.py). Maps raw 3-component seismic waveforms ($3 \times 6000$) into a normalized vector space.

### Architecture
- **Conv1D Layer 1:** 3 $\rightarrow$ 32 channels, kernel 7, stride 2, BatchNorm, ReLU, MaxPool
- **Conv1D Layer 2:** 32 $\rightarrow$ 64 channels, kernel 5, stride 2, BatchNorm, ReLU, MaxPool
- **Conv1D Layer 3:** 64 $\rightarrow$ 128 channels, kernel 3, stride 2, BatchNorm, ReLU, AdaptiveAvgPool1d(1)
- **Dense Linear Layer:** 128 $\rightarrow$ 128 dimensions
- **L2 Normalization:** Vector norm $\|\mathbf{v}\|_2 = 1.0$ (enables fast cosine similarity matching in Qdrant)

---

## 4. RAG Reporting Engine

### Description
Implemented in [`backend/reporting/rag_generator.py`](file:///c:/Users/henok/OneDrive/Documents/VsCode/Minor-miniproject/backend/reporting/rag_generator.py). Generates a factual and deterministically verifiable narrative report of the earthquake using Retrieval-Augmented Generation (RAG). 

### Architecture
- **Retrieval Engine:** SimilarityService queries the Qdrant vector database to fetch historically similar events.
- **Context Builder:** Retrieves supplementary metadata from the USGS API and normalizes historical context data.
- **Generation:** 
  - **LLM Mode:** Uses HuggingFace `google/flan-t5-base` (or `t5-small`) to generate a narrative based on the enriched prompt.
  - **Deterministic Fallback:** A template-based generation engine for use when GPU/LLM capabilities are disabled.
- **Factual Constraints:** The engine cross-checks all generated metrics against the exact quantitative forecast (e.g. Ground Motion Prediction Equations). It automatically appends an uneditable **"Verified Facts Box"** to ensure zero-hallucination factual integrity in the final PDF export.

---

## 🔄 Retraining Models

### Train Noise Classifier:
```bash
python ml/training/train_noise.py --output backend/models/noise_classifier_v1.pkl
```

### Train EQTransformer on SeisBench datasets (e.g. INSTANCE / SCEDC):
```bash
python ml/training/train_detector.py --train-data instance --test-data scedc --epochs 10 --output-dir backend/models/weights
```

### Run Model Evaluation & Generate Reports:
```bash
python ml/evaluation/eval_picker.py
```
