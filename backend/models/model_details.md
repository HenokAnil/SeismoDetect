# SeismoDetect Machine Learning Models Report

This document provides a comprehensive report on the machine learning models used in the **SeismoDetect** platform. It details the purpose, architecture, inputs, outputs, and training methodologies for each component in the system's pipeline.

---

## 1. Noise Classifier (Signal Quality Evaluation)

### Overview
The Noise Classifier is a supervised machine learning model that acts as an initial quality gate. It evaluates incoming 3-component seismic traces and determines whether the trace contains a clean earthquake signal or is corrupted by various non-earthquake noise artifacts.

### Model Architecture
- **Type**: Random Forest Classifier
- **Parameters**: 100 estimators, maximum depth of 10.
- **Implementation**: Scikit-Learn `RandomForestClassifier` saved via `joblib` (`noise_classifier_v1.pkl`).

### Inputs
The classifier takes a 4-dimensional feature vector extracted from the raw waveform data:
1. **`avg_snr`**: Signal-to-Noise Ratio (in dB).
2. **`avg_entropy`**: Spectral entropy calculated from the Welch power spectral density.
3. **`avg_kurtosis`**: Fourth standardized moment (used to detect sudden spikes or glitches).
4. **`is_clipped`**: Boolean flag (1.0 or 0.0) indicating whether the 24-bit digitizer has reached rail saturation.

### Outputs
- **Predicted Class**: One of the following categories:
  - `0: clean` (Clean seismic event candidate)
  - `1: wind_noise` (Characterized by high entropy, low SNR)
  - `2: traffic_noise` (Moderate SNR, mid entropy)
  - `3: instrument_glitch` (Extremely high kurtosis, possible clipping)
  - `4: calibration_pulse` (High SNR, low entropy regular signal)
- **Quality Score Output**: The probability score that the signal belongs to the `clean` class ($P(\text{class} = \text{clean}) \in [0.0, 1.0]$).

### Training Details
- **Script**: `ml/training/train_noise.py`
- **Data**: The model is trained on **synthetic data** (5,000 generated samples). The feature values for each noise class are simulated using random normal distributions that mimic the empirical characteristics of different seismic noise types (e.g., wind has low SNR/high entropy; glitches have high kurtosis).
- **Performance Metrics**: The Random Forest achieves an overall training accuracy of **91.06%** on the synthetic dataset.

---

## 2. EQTransformer (Earthquake Detection & Phase Picking)

### Overview
EQTransformer is the core deep learning model responsible for processing continuous waveforms, identifying the presence of an earthquake, and accurately determining the precise arrival times of primary (P) and secondary (S) seismic waves.

### Model Architecture
- **Type**: Deep Learning Sequence-to-Sequence Model
- **Base Architecture**: 1D Convolutional Neural Network (CNN) paired with Self-Attention blocks (Transformer), implemented via the SeisBench library.

### Inputs
- **Features**: 3-component raw seismic waveforms.
- **Dimensions**: $3 \times 6000$ samples (equivalent to a 60-second window at a 100 Hz sampling rate).

### Outputs
The model produces three independent probability curves across the same sequence length:
1. **Detection Probability curve**: Indicates the likelihood of an earthquake occurring at any point in the time window.
2. **P-phase arrival probability curve**: Spikes at the predicted P-wave onset.
3. **S-phase arrival probability curve**: Spikes at the predicted S-wave onset.

### Training Details
- **Script**: `ml/training/train_detector.py`
- **Data**: Models are trained and evaluated using massive seismic datasets provided through SeisBench.
  - **Training Dataset**: By default, it uses the **INSTANCE** dataset (an Italian seismic network), which acts as a robust proxy for European (EIDA) data. 10% of the training dataset is held out for validation during training.
  - **Test/Cross-Domain Dataset**: The **SCEDC** dataset (Southern California) is used to perform cross-domain generalization testing.
- **Data Augmentation & Preprocessing**: During training, data is heavily augmented to improve robustness. Steps include:
  - `WindowAroundSample`: Selects 6000-sample windows around known P or S phase arrivals (either random or centered).
  - `RandomWindow`: Adds zero-padding dynamically.
  - `Normalize`: Detrends the signal and applies peak amplitude normalization.
  - `ProbabilisticLabeller`: Converts exact arrival time points into Gaussian probability curves (with $\sigma=20$) to serve as continuous targets.
- **Training Process**: Trained using the Adam optimizer with Binary Cross Entropy (BCE) Loss calculated individually for the detection, P-phase, and S-phase prediction heads. The model is trained over several epochs (default 10) with checkpoints saved for the lowest validation loss.

### Performance Metrics
- **Earthquake Detection**: Precision: **89.88%**, Recall: **90.24%**, F1-Score: **90.06%**.
- **Phase Picking (Mean Absolute Error)**:
  - **P-Wave**: ~3.98 samples (~0.040 seconds @ 100Hz)
  - **S-Wave**: ~8.07 samples (~0.081 seconds @ 100Hz)

---

## 3. Waveform Encoder (Similarity Embedding)

### Overview
The Waveform Encoder acts as a feature extractor that maps variable raw 3-component seismic waveforms into a fixed-size, lower-dimensional dense vector space. These vectors (embeddings) are designed to be ingested into vector databases (like Qdrant) to perform rapid similarity matching between seismic events.

### Model Architecture
- **Type**: 1D Convolutional Neural Network (CNN) with Global Pooling.
- **Implementation**: Custom PyTorch module (`backend/similarity/encoder.py`).
- **Layers**:
  1. **Conv1D Layer 1**: 3 $\rightarrow$ 32 channels (Kernel 7, Stride 2) + BatchNorm, ReLU, MaxPool.
  2. **Conv1D Layer 2**: 32 $\rightarrow$ 64 channels (Kernel 5, Stride 2) + BatchNorm, ReLU, MaxPool.
  3. **Conv1D Layer 3**: 64 $\rightarrow$ 128 channels (Kernel 3, Stride 2) + BatchNorm, ReLU, AdaptiveAvgPool1d(1).
  4. **Dense Layer**: 128 $\rightarrow$ 128 output dimensions.
  5. **Normalization**: L2 vector normalization ($\|\mathbf{v}\|_2 = 1.0$) to enable fast cosine similarity computations.

### Inputs
- **Features**: 3-component seismic waveforms ($3 \times N$).

### Outputs
- **Output**: A 128-dimensional dense embedding vector, constrained to unit length (L2 normalized).

---

## 4. Magnitude Estimator (Proxy)

### Overview
Currently, true magnitude estimation regression is not fully deployed via machine learning. Instead, an interim heuristic approximation (proxy) is used to calculate the Local Magnitude (ML) of detected events.

### Methodology
- **Current Implementation**: Embedded as a heuristic formula within the application pipeline (`backend/api/main.py` and `backend/models/database.py`).
- **Formula**: `ML ≈ 2.0 + (Detection Probability * 1.5)`
- **Input**: The detection confidence score returned by the EQTransformer model.
- **Output**: Estimated Local Magnitude (ML).
## 5. RAG Reporting Engine (Automated Narrative Generation)

### Overview
The Retrieval-Augmented Generation (RAG) system is responsible for converting numerical inferences and historical context into an easily readable, factual PDF report. It avoids common hallucination pitfalls associated with large language models through strict grounding and a hybrid architecture.

### Model Architecture
- **Type**: Natural Language Generator (NLG) with Deterministic Constraints.
- **Components**:
  - **Retriever**: `SimilarityService` leveraging Qdrant Vector DB to find the most relevant historical earthquake waveforms.
  - **Context Builder**: Enriches the localized data with global earthquake catalogs via the USGS Earthquake Hazards API.
  - **Generator**: Uses HuggingFace `google/flan-t5-base` (an instruction-tuned language model) or falls back to a deterministic, zero-dependency templating engine.
  
### Inputs
The system constructs a comprehensive prompt (`context_json`) consisting of:
1. **Current Event Metrics**: Station ID, time, proxy magnitude, detection probability, and signal analysis (SNR/clipping).
2. **Historical Context**: Up to 5 closely matched waveforms and their verified real-world impacts (e.g. USGS PAGER alerts).
3. **Quantitative Forecast**: A Ground Motion Prediction Equation (GMPE) output estimating maximum local shaking intensity (MMI) and Peak Ground Acceleration (PGA).

### Outputs
- **Narrative**: A textual explanation summarizing the earthquake event.
- **Verified Facts Box**: A rigidly structured, uneditable facts table that acts as the absolute ground-truth, preventing the LLM from hallucinating metrics.

### Pipeline Details
- **Script**: `backend/reporting/rag_generator.py` and `backend/reporting/context_builder.py`.
- **Anti-Hallucination Framework**: Before generating the final document, the system explicitly cross-checks every number present in the generated narrative against the `context_json` database. Any discrepancies trigger a fallback to the safe, deterministic template, guaranteeing that no false magnitudes or times are exported in official PDFs.
