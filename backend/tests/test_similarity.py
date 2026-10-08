import pytest
import numpy as np
import uuid
import torch
from pathlib import Path
from backend.similarity.encoder import WaveformEncoder, EmbeddingPipeline
from backend.similarity.index import SimilarityIndex
from backend.similarity.waveform import prepare_embedding_waveform
from obspy import Stream, Trace

def test_waveform_encoder_shape():
    encoder = WaveformEncoder(embedding_dim=128)
    x = torch.randn(1, 3, 6000)
    out = encoder(x)
    assert out.shape == (1, 128)

def test_preprocessing_shape():
    tr = Trace(data=np.random.randn(10000))
    tr.stats.sampling_rate = 100.0
    st = Stream([tr])
    w = prepare_embedding_waveform(st)
    assert w.shape == (3, 6000)
    assert w.dtype == np.float32

def test_encoder_checkpoint_required():
    with pytest.raises(FileNotFoundError):
        EmbeddingPipeline(weights_path="/nonexistent/path.pth")

def test_placeholder_checkpoint_rejected(tmp_path):
    fake = tmp_path / "encoder.pth"
    fake.write_text("# placeholder")
    with pytest.raises(ValueError, match="placeholder"):
        EmbeddingPipeline(weights_path=str(fake))

@pytest.fixture
def fake_checkpoint(tmp_path):
    model = WaveformEncoder(embedding_dim=128)
    ckpt_path = tmp_path / "fake_encoder.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return str(ckpt_path)

def test_embedding_is_deterministic(fake_checkpoint):
    pipeline = EmbeddingPipeline(weights_path=fake_checkpoint)
    w = np.random.randn(3, 6000).astype(np.float32)
    vec1 = pipeline.encode(w)
    vec2 = pipeline.encode(w)
    np.testing.assert_array_equal(vec1, vec2)

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
    
    success = index.add(test_id, test_vec, metadata={"source_type": "catalog"})
    assert success
    
    hits = index.search(test_vec, limit=2, source_type="catalog")
    assert len(hits) > 0
