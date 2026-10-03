import torch
import torch.nn as nn
import numpy as np
from backend.models.checkpoints import require_or_raise

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
        require_or_raise()
        self.device = torch.device(device)
        self.model = WaveformEncoder(embedding_dim=128)
        
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """
        Encode a single waveform (3, N) into a 128-dim embedding.
        """
        x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(x)
        return embedding.cpu().numpy()[0]
