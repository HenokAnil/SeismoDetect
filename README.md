# SeismoDetect — Quickstart

## Quick demo (local, CPU)
1. Create venv and install:
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements_cpu.txt
   ```
2. Download models:
   ```bash
   ../scripts/download_models.sh
   ```
3. Start API:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
4. Run smoke test (in another terminal):
   ```bash
   ./scripts/smoke_test_api.sh
   ```

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
- **Electron UI**: Cross-platform desktop application visualizing 3-component seismic traces and ML picks.

---

## 🐳 Docker Services & Commands Quick Reference

All core storage and database infrastructure services are orchestrated via `docker-compose.yml`.

### 1. Starting Services

```bash
# Start all containers in detached (background) mode
docker compose up -d

# Start all containers and force recreate them
docker compose up -d --force-recreate

# Build and start (if building local images)
docker compose up -d --build
```

### 2. Checking Status and Health

```bash
# View all running project containers
docker compose ps

# View all Docker containers on the host
docker ps

# Check container resource utilization (CPU, memory, I/O)
docker stats
```

### 3. Container Ports, Credentials & Defaults

| Service | Container Name | Host Port | Internal Port | Default User | Default Password | Default DB / Bucket |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PostgreSQL** | `seismodetect_postgres` | `5432` | `5432` | `seismo_user` | `seismo_password` | `seismodetect` |
| **MinIO API** | `seismodetect_minio` | `9000` | `9000` | `minioadmin` | `minioadmin123` | `seismic-waveforms` |
| **MinIO Web Console**| `seismodetect_minio` | `9001` | `9001` | `minioadmin` | `minioadmin123` | Web UI: `http://localhost:9001` |
| **Redis** | `seismodetect_redis` | `6379` | `6379` | *None* | *None* | DB `0` |
| **Qdrant REST** | `seismodetect_qdrant` | `6333` | `6333` | *None* | *None* | Collection: `seismic_events` |
| **Qdrant gRPC** | `seismodetect_qdrant` | `6334` | `6334` | *None* | *None* | - |

### 4. Viewing Logs

```bash
# Follow logs for all containers in real-time
docker compose logs -f

# View logs for a specific service
docker compose logs -f postgres
docker compose logs -f minio
docker compose logs -f redis
docker compose logs -f qdrant

# View the last 100 log lines of a specific container
docker logs --tail 100 seismodetect_postgres
```

### 5. Stopping and Removing Services

```bash
# Stop all containers (preserves database volumes)
docker compose stop

# Stop and remove containers and networks
docker compose down

# Stop, remove containers, and DELETE all persistent volume data (Clean Reset)
docker compose down -v
```

### 6. Executing Commands Directly Inside Containers

#### PostgreSQL CLI (`psql`)
```bash
# Open interactive PostgreSQL shell
docker exec -it seismodetect_postgres psql -U seismo_user -d seismodetect

# Check database connection readiness
docker exec seismodetect_postgres pg_isready -U seismo_user -d seismodetect

# List tables from outside the container
docker exec -it seismodetect_postgres psql -U seismo_user -d seismodetect -c "\dt"
```

#### Redis CLI (`redis-cli`)
```bash
# Ping Redis
docker exec seismodetect_redis redis-cli ping

# Open interactive Redis CLI
docker exec -it seismodetect_redis redis-cli

# Monitor real-time Redis commands
docker exec -it seismodetect_redis redis-cli monitor
```

#### MinIO Storage & Client
Access the MinIO Web Console directly in your browser at:
👉 **[http://localhost:9001](http://localhost:9001)** (Username: `minioadmin`, Password: `minioadmin123`)

#### Qdrant Vector Engine
```bash
# Check health status via cURL
curl -s http://localhost:6333/healthz

# List vector collections
curl -s http://localhost:6333/collections
```

---

## 🐍 Backend Setup & Execution

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.11 - 3.13)
- Docker running `docker compose up -d`

### 2. Environment Setup

```bash
# Recommended: Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
```

### 3. Initialize Database Tables
To create all tables (`signals`, `stations`, `signal_quality`, `events`, `picks`, `processing_runs`) in PostgreSQL:

```bash
python -c "from sqlalchemy import create_engine; from backend.models.database import Base; engine = create_engine('postgresql://seismo_user:seismo_password@localhost:5432/seismodetect'); Base.metadata.create_all(engine); print('Database initialized!')"
```

### 4. Running the FastAPI Server

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: **[http://localhost:8000/docs](http://localhost:8000/docs)**
- ReDoc: **[http://localhost:8000/redoc](http://localhost:8000/redoc)**

---

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

---

## 💻 Electron Desktop Application

The desktop app provides waveform visualization using Plotly and triggers ingestion and detection pipelines.

### 1. Setup

```bash
cd electron-app
npm install
```

### 2. Run the Desktop App

```bash
npm start
```

### 3. Build Distribution Packages

```bash
npm run make
```

---

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
