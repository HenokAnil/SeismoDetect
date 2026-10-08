from __future__ import annotations

import numpy as np
from obspy import Stream

from backend.similarity.encoder import EmbeddingPipeline
from backend.similarity.index import SimilarityIndex
from backend.similarity.waveform import prepare_embedding_waveform

class SimilarityService:
    def __init__(self, encoder: EmbeddingPipeline, index: SimilarityIndex):
        self.encoder = encoder
        self.index = index

    def embed_stream(self, stream: Stream) -> np.ndarray:
        waveform = prepare_embedding_waveform(stream)
        return self.encoder.encode(waveform)

    def index_event(self, event_id: str, stream: Stream, metadata: dict) -> np.ndarray:
        embedding = self.embed_stream(stream)
        self.index.add(
            event_id=event_id,
            embedding=embedding.tolist(),
            metadata=metadata
        )
        return embedding

    def find_historical_matches(self, stream: Stream, limit: int = 5) -> list:
        """Returns only catalog events (source_type='catalog')."""
        embedding = self.embed_stream(stream)
        return self.index.search(
            embedding.tolist(),
            limit=limit,
            source_type="catalog"
        )
