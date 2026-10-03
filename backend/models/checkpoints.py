import os
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

def ensure_checkpoints_exist():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    required = {
        "eqtransformer.pt": "eqtransformer",
        "encoder.pth": "encoder",
    }
    for fname in required:
        p = CHECKPOINT_DIR / fname
        if not p.exists():
            missing.append(fname)
    return missing

def require_or_raise():
    missing = ensure_checkpoints_exist()
    if missing:
        raise FileNotFoundError(
            f"Missing model checkpoints: {missing}. "
            "Run scripts/download_models.sh or set the MODEL_* env vars."
        )
