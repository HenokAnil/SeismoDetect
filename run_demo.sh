#!/usr/bin/env bash
set -euo pipefail
if [ ! -d "backend/.venv" ]; then
    python3 -m venv backend/.venv
fi
source backend/.venv/bin/activate || source backend/.venv/Scripts/activate || true
pip install -r backend/requirements_cpu.txt
bash scripts/download_models.sh
export PYTHONPATH="."
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
