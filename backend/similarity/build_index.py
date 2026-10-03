#!/usr/bin/env python3
"""
Usage:
  python build_index.py --data-dir ../sample_data --qdrant-host http://localhost:6333 --collection seismo_events
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np

# obspy for loading waveforms
import obspy

# Existing encoder
from backend.similarity.encoder import EmbeddingPipeline

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

def load_waveforms(data_dir):
    events = []
    for p in Path(data_dir).glob("*.*"):
        if p.suffix.lower() in [".mseed", ".sac", ".csv"]:
            events.append(str(p))
    return events

def build_and_upload(data_dir, qdrant_host=None, collection_name="seismo_events"):
    files = load_waveforms(data_dir)
    # The EmbeddingPipeline requires a device and weights_path (if any).
    # Since we added require_or_raise, we can just instantiate it.
    encoder = EmbeddingPipeline()
    vectors = []
    metas = []
    ids = []
    
    for i, f in enumerate(files):
        # Load using obspy and convert to a numpy array (3, N)
        st = obspy.read(f)
        # Ensure 3 channels and shape
        if len(st) >= 3:
            w_data = np.stack([st[0].data, st[1].data, st[2].data])
        else:
            w_data = np.repeat(st[0].data[np.newaxis, :], 3, axis=0)
            
        # Target length 6000? Assuming encode will handle or we just pass it
        if w_data.shape[1] < 6000:
            padded = np.zeros((3, 6000), dtype=np.float32)
            padded[:, :w_data.shape[1]] = w_data
            w_data = padded
        elif w_data.shape[1] > 6000:
            w_data = w_data[:, :6000]

        emb = encoder.encode(w_data)  # shape (128,)
        vectors.append(emb.astype(np.float32))
        ids.append(str(i))
        metas.append({"path": f})
        
    if not vectors:
        print("No valid waveforms found to encode.")
        return

    vectors = np.vstack(vectors)
    if qdrant_host and QDRANT_AVAILABLE:
        client = QdrantClient(url=qdrant_host)
        client.recreate_collection(collection_name, vector_size=vectors.shape[1], distance="Cosine")
        client.upsert(collection_name, points=[(ids[i], vectors[i].tolist(), metas[i]) for i in range(len(ids))])
        print(f"Uploaded {len(ids)} events to Qdrant: {collection_name}")
    else:
        # FAISS fallback
        import faiss
        out_dir = Path(os.path.dirname(__file__)) / "faiss_index"
        out_dir.mkdir(parents=True, exist_ok=True)
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        np.save(out_dir / "ids.npy", np.array(ids))
        np.save(out_dir / "metas.npy", np.array(metas, dtype=object))
        print(f"Saved FAISS index to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--qdrant-host", default=None)
    parser.add_argument("--collection", default="seismo_events")
    args = parser.parse_args()
    build_and_upload(args.data_dir, args.qdrant_host, args.collection)
