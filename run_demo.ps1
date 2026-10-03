$ErrorActionPreference = "Stop"
if (-Not (Test-Path "backend\.venv")) {
    python -m venv backend\.venv
}
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements_cpu.txt
bash scripts/download_models.sh
$env:PYTHONPATH="."
backend\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
