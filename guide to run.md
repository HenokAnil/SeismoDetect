# SeismoDetect — Guide to Run 🚀

This guide provides step-by-step instructions on how to properly set up and run the entire SeismoDetect framework, including the databases, the backend API, and the Electron desktop application.

## 1. Prerequisites
- **Python 3.10+**
- **Docker** and **Docker Compose** installed
- **Node.js** and **npm** installed (for the Electron app)

## 2. Start the Infrastructure (Docker)
The core storage and database infrastructure services are orchestrated via Docker Compose.
This includes PostgreSQL, MinIO, Redis, and Qdrant.

Open a terminal in the root directory of the project and run:
```bash
docker-compose up -d
```
You can verify the services are running with `docker-compose ps`.

## 3. Setup the Python Environment
Create and activate a virtual environment for the backend:

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements_cpu.txt
```

**On Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements_cpu.txt
```

*(Note: Depending on your hardware, you can also use `requirements.txt` if you have GPU support configured).*

## 4. Environment Variables
Make sure to copy the `config.example.env` file to a `.env` file in the root directory.
```bash
cp config.example.env .env
```
Ensure `QDRANT_COLLECTION=seismic_events_v2` and `REPORT_MODEL=google/flan-t5-base` (or `t5-small` for faster, lower-quality generation on CPU).

## 5. Initialize the Database
Before running the app for the first time, initialize the PostgreSQL tables:
```bash
python -c "from sqlalchemy import create_engine; from backend.models.database import Base; engine = create_engine('postgresql://seismo_user:seismo_password@localhost:5432/seismodetect'); Base.metadata.create_all(engine); print('Database initialized!')"
```

## 6. Run the FastAPI Backend
Start the FastAPI server:
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend will be available at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`. You can also access the web dashboard directly at `http://localhost:8000/app/`.

## 7. Run the Electron Desktop App
Open a **new terminal**, navigate to the `electron-app` directory, and start the desktop client:
```bash
cd electron-app
npm install
npm start
```

## 8. Run Scripts & Synthetic Data (Optional)
To test the pipeline, you can generate synthetic test waveforms:
```bash
python generate_samples.py
```
This creates `.mseed` files in the `sample_data/` directory which you can upload in the app.

To test the system endpoints, run:
```bash
./scripts/smoke_test_api.sh
```
