# SeismoDetect — Quickstart

## Quick demo (local, CPU)
Please read the comprehensive **[Guide to Run](guide%20to%20run.md)** for detailed, step-by-step instructions on setting up Docker, installing dependencies, and running the full system.

# SeismoDetect 🌍⚡

SeismoDetect is an end-to-end seismic event detection, phase picking, quality filtering, and similarity search platform. It pairs modern deep learning models (EQTransformer, 1D-CNN Waveform Embeddings) with distributed infrastructure (FastAPI, Celery, PostgreSQL, MinIO S3, Redis, and Qdrant Vector DB) and a responsive Electron desktop interface.

---

## 🏛️ Architecture Overview

```
                      +-----------------------------+
                      |   Electron Desktop App      |
                      |   (Waveform Plotly & UI)    |
                      +--------------+--------------+
                                     |  HTTP / WebSocket
                                     v
                      +-----------------------------+
                      |     FastAPI Backend         |
                      |   (API, Ingestion, Router)  |
                      +---+-----------+---------+---+
                          |           |         |
         +----------------+     +-----+-----+   +----------------+
         |                      |           |                    |
         v                      v           v                    v
+-----------------+   +-------------+ +------------+   +-----------------+
|   PostgreSQL    |   |  MinIO S3   | |   Redis    |   | Qdrant Vector DB|
| (Metadata, DB)  |   | (Raw mseed) | |  (Broker)  |   | (128-d Embeds)  |
+-----------------+   +-------------+ +------------+   +-----------------+
```

- **PostgreSQL 15**: Relational metadata storage for stations, signals, signal quality metrics, seismic events, picks (P/S arrival times), and processing runs.
- **MinIO**: High-performance S3-compatible object storage for raw and preprocessed MiniSEED (`.mseed`) waveform files.
- **Redis 7**: Fast in-memory message broker and caching layer for background task queues.
- **Qdrant**: Vector search engine indexing 128-dimensional seismic waveform embeddings for rapid similarity search across historical seismic events.
- **FastAPI**: Asynchronous Python backend serving ingestion endpoints, inference pipelines, PDF report exports, and real-time WebSockets.
- **PyTorch & SeisBench / Scikit-learn**: ML inference models for earthquake detection, P/S phase picking, signal SNR/clipping extraction, and noise classification.
- **RAG Reporting Engine**: Hybrid FLAN-T5 LLM & deterministic templating system for generating factual, hallucination-free PDF reports.
- **Electron UI**: Cross-platform desktop application visualizing 3-component seismic traces and ML picks.

---

## 🚀 How to Run
We have moved all execution instructions (Docker, Python Backend, Electron App, and synthetic data) into a dedicated setup file. 

👉 **[Read the Step-by-Step Guide to Run here](guide%20to%20run.md)**



## 🧪 Testing & ML Pipelines

### 1. Running Unit & Integration Tests
Run pytest to verify FDSN waveform fetchers (EIDA, SCEDC), signal preprocessor, and quality extraction:

```bash
pytest backend/tests -v
```

### 2. Train Noise Classifier (Baseline)
Trains a Random Forest classifier on synthetic SNR, spectral entropy, and kurtosis metrics:

```bash
python ml/training/train_noise.py --output backend/models/noise_classifier_v1.pkl
```

### 3. Evaluate Phase Picker & Detection
Evaluates P and S wave pick arrival Mean Absolute Error (MAE) and outputs CDF curves and JSON evaluation reports:

```bash
python ml/evaluation/eval_picker.py
# Outputs:
# - backend/reports/pick_cdf.png
# - backend/reports/evaluation_report.json
```

### 4. Vector Embedding & Qdrant Similarity Search
Test the 1D-CNN waveform encoder and Qdrant indexing:

```bash
python -c "import uuid, numpy as np; from backend.similarity.encoder import EmbeddingPipeline; from backend.similarity.index import SimilarityIndex; ep = EmbeddingPipeline(); idx = SimilarityIndex(); event_id = uuid.uuid4(); emb = ep.encode(np.random.randn(3, 6000)).tolist(); idx.add(event_id, emb, {'magnitude': 3.2}); res = idx.search(emb, limit=1); print('Similar event score:', res[0].score)"
```



## 📂 Project Structure

```
Minor-miniproject/
├── docker-compose.yml          # Postgres, MinIO, Redis, and Qdrant definitions
├── README.md                   # Complete documentation and Docker guide
├── backend/
│   ├── requirements.txt        # Python dependencies
│   ├── api/
│   │   └── main.py             # FastAPI application and WebSocket endpoints
│   ├── ingestion/
│   │   ├── clients.py          # Obspy FDSN clients for EIDA & SCEDC
│   │   └── tasks.py            # Database population for events and picks
│   ├── models/
│   │   ├── database.py         # SQLAlchemy models (Signals, Picks, Events, etc.)
│   │   ├── detector.py         # EQTransformer deep learning wrapper
│   │   └── noise_classifier_v1.pkl # Trained baseline noise model
│   ├── preprocessing/
│   │   ├── pipeline.py         # Filtering, detrending, resampling
│   │   ├── quality.py          # SNR, entropy, kurtosis, and clipping detector
│   │   └── noise_classifier.py # Feature extraction and classification interface
│   ├── similarity/
│   │   ├── encoder.py          # 1D-CNN waveform embedding generator (128-d)
│   │   └── index.py            # Qdrant vector database client wrapper
│   └── tests/
│       ├── test_ingestion.py   # Ingestion client mock tests
│       └── test_preprocessing.py # Filter and quality feature tests
├── ml/
│   ├── evaluation/
│   │   └── eval_picker.py      # MAE evaluation and CDF plot generator
│   └── training/
│       ├── train_detector.py   # SeisBench EQTransformer training script
│       └── train_noise.py      # Random forest noise classifier trainer
└── electron-app/
    ├── package.json            # Electron and Forge dependencies
    └── src/
        ├── main/               # Electron main and preload scripts
        └── renderer/           # HTML & JS UI with Plotly visualizations
```

---

## 🛠️ Troubleshooting

- **Container fails to start due to port collision**:
  Check if local services are already occupying ports `5432`, `6379`, `9000`, `9001`, or `6333`:
  ```bash
  netstat -ano | findstr 5432
  ```
- **Database connection error**:
  Ensure PostgreSQL is accepting connections:
  ```bash
  docker exec seismodetect_postgres pg_isready -U seismo_user -d seismodetect
  ```
- **MinIO bucket not found**:
  Verify bucket creation:
  ```bash
  python -c "import boto3; s3 = boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin123'); print(s3.list_buckets()['Buckets'])"
  ```
- **Clean slate restart**:
  ```bash
  docker compose down -v && docker compose up -d
  ```
