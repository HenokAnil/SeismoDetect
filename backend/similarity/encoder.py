import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from backend.models.checkpoints import validate_encoder_checkpoint

DEFAULT_ENCODER_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "checkpoints"
    / "encoder_v1.pth"
)

class WaveformEncoder(nn.Module):
    """
    A simple 1D CNN encoder to embed 3-component seismic waveforms into a fixed-length vector.
    """
    def __init__(self, input_channels=3, embedding_dim=128):
        super(WaveformEncoder, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global average pooling
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        # x shape: (batch, channels, samples)
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        
        # L2 normalize embeddings for cosine similarity
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

class EmbeddingPipeline:
    def __init__(self, weights_path=None, device="cpu"):
        self.device = torch.device(device)
        
        checkpoint_path = Path(
            weights_path or
            os.getenv("ENCODER_CHECKPOINT", DEFAULT_ENCODER_CHECKPOINT)
        )
        
        validate_encoder_checkpoint(checkpoint_path)
        
        self.model = WaveformEncoder(embedding_dim=128)
        
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.model_version = os.getenv(
            "EMBEDDING_MODEL_VERSION",
            "seismo-cnn-v1"
        )
        
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """
        Encode a single waveform (3, N) into a 128-dim embedding.
        """
        x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(x)
        return embedding.cpu().numpy()[0]
