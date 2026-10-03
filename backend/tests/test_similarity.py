import pytest
import numpy as np
import uuid
from backend.similarity.encoder import WaveformEncoder, EmbeddingPipeline
from backend.similarity.index import SimilarityIndex

def test_waveform_encoder_shape():
    encoder = WaveformEncoder(embedding_dim=128)
    import torch
    x = torch.randn(1, 3, 6000)
    out = encoder(x)
    assert out.shape == (1, 128)

def test_embedding_pipeline():
    pipeline = EmbeddingPipeline()
    # Random waveform shape (3, 6000)
    w = np.random.randn(3, 6000).astype(np.float32)
    vec = pipeline.encode(w)
    assert vec.shape == (128,)
    assert isinstance(vec, np.ndarray)

@pytest.mark.integration
def test_similarity_index_qdrant():
    try:
        index = SimilarityIndex(collection_name="test_collection")
    except Exception:
        pytest.skip("Qdrant not available")
        
    if not index.client:
        pytest.skip("Qdrant client failed to initialize")
        
    test_id = uuid.uuid4()
    test_vec = np.random.randn(128).tolist()
    
    success = index.add(test_id, test_vec, metadata={"test": True})
    assert success
    
    hits = index.search(test_vec, limit=2)
    assert len(hits) > 0
