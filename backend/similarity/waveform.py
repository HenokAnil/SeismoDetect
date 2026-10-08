from __future__ import annotations

import numpy as np
from obspy import Stream

from backend.preprocessing.pipeline import Preprocessor

TARGET_SAMPLES = 6000
TARGET_CHANNELS = 3

_preprocessor = Preprocessor(
    target_sampling_rate=100.0,
    freqmin=1.0,
    freqmax=20.0,
)

def prepare_embedding_waveform(st: Stream) -> np.ndarray:
    """
    Returns exactly:
        shape = (3, 6000)
        dtype = float32

    The exact same function is used for:
      - historical indexing
      - live inference
      - report similarity retrieval
    """
    st = _preprocessor.process(st)

    if len(st) == 0:
        raise ValueError("Empty waveform stream after preprocessing")

    # Stack to 3 channels
    if len(st) >= 3:
        data = np.stack([
            tr.data for tr in st[:3]
        ])
    else:
        data = np.repeat(
            st[0].data[np.newaxis, :],
            3,
            axis=0
        )

    data = np.asarray(data, dtype=np.float32)

    # Handle NaN / Inf
    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Pad / crop to 60 seconds @ 100 Hz
    output = np.zeros(
        (TARGET_CHANNELS, TARGET_SAMPLES),
        dtype=np.float32
    )

    length = min(data.shape[1], TARGET_SAMPLES)
    output[:, :length] = data[:, :length]

    # Per-component z-normalization
    for i in range(3):
        mean = np.mean(output[i])
        std = np.std(output[i])

        if std > 1e-6:
            output[i] = (output[i] - mean) / std

    return output
