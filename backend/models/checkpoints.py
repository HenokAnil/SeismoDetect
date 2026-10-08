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

def validate_encoder_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    if path.stat().st_size < 1024:
        raise ValueError(
            f"Encoder checkpoint appears invalid or is a placeholder: "
            f"{path}\nRun: python ml/training/train_encoder.py"
        )
