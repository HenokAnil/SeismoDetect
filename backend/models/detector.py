import torch
import numpy as np
import seisbench.models as sbm
import logging
from dataclasses import dataclass
from typing import List, Tuple
from backend.models.checkpoints import require_or_raise

logger = logging.getLogger(__name__)

@dataclass
class PickResult:
    phase: str
    sample_index: int
    confidence: float

@dataclass
class DetectionResult:
    event_start_sample: int
    event_end_sample: int
    confidence: float
    picks: List[PickResult]

class EQTransformerDetector:
    def __init__(self, weights_path=None, device="cpu"):
        require_or_raise()
        self.device = torch.device(device)
        logger.info(f"Loading EQTransformer model on {self.device}")
        
        if weights_path:
            logger.info(f"Loading custom weights from {weights_path}")
            self.model = sbm.EQTransformer(phases=["P", "S"])
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            logger.info("Loading pretrained 'original' weights")
            self.model = sbm.EQTransformer.from_pretrained("original")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, detection_threshold=0.5, p_threshold=0.3, s_threshold=0.3) -> List[DetectionResult]:
        """
        Takes a 3-component waveform of shape (3, N) and returns detections.
        Assumes sampling rate is 100 Hz and length is 6000 (60s).
        """
        # Ensure waveform is shape (3, 6000)
        channels, length = waveform.shape
        if channels != 3:
            # If 1 channel or >3, expand or take first 3
            if channels == 1:
                waveform = np.repeat(waveform, 3, axis=0)
            else:
                waveform = waveform[:3]
                
        if waveform.shape[1] < 6000:
            padded = np.zeros((3, 6000), dtype=np.float32)
            padded[:, :waveform.shape[1]] = waveform
            waveform = padded
        elif waveform.shape[1] > 6000:
            waveform = waveform[:, :6000]

        # Standardize (z-score normalize) per component
        for i in range(3):
            std = np.std(waveform[i])
            if std > 1e-6:
                waveform[i] = (waveform[i] - np.mean(waveform[i])) / std
            
        # Prepare tensor (batch_size=1, channels=3, samples=6000)
        x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(x)
            
            if isinstance(preds, tuple):
                # SeisBench implementation might return multiple outputs
                det_prob = preds[0].cpu().numpy().squeeze()
                p_prob = preds[1].cpu().numpy().squeeze()
                s_prob = preds[2].cpu().numpy().squeeze()
            else:
                probs = preds.cpu().numpy().squeeze()
                # Assuming shape (classes, samples) where classes are Noise, P, S, or similar.
                # If output is a single tensor, we extract based on assumed indices.
                if len(probs.shape) == 2 and probs.shape[0] >= 3:
                    det_prob = 1.0 - probs[0] # 1 - noise prob
                    p_prob = probs[1]
                    s_prob = probs[2]
                else:
                    raise ValueError("Unexpected model output shape")

        results = []
        
        # Simple thresholding logic for demonstration
        # Real implementation would use peak finding / connected components
        det_indices = np.where(det_prob > detection_threshold)[0]
        p_indices = np.where(p_prob > p_threshold)[0]
        s_indices = np.where(s_prob > s_threshold)[0]
        
        # Very naive clustering: just take the max peak if any exists
        if len(det_indices) > 0:
            start_idx = int(det_indices[0])
            end_idx = int(det_indices[-1])
            confidence = float(np.max(det_prob))
            
            picks = []
            if len(p_indices) > 0:
                p_peak = int(np.argmax(p_prob))
                picks.append(PickResult("P", p_peak, float(p_prob[p_peak])))
                
            if len(s_indices) > 0:
                s_peak = int(np.argmax(s_prob))
                picks.append(PickResult("S", s_peak, float(s_prob[s_peak])))
                
            results.append(DetectionResult(start_idx, end_idx, confidence, picks))
            
        return results
