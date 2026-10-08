#!/usr/bin/env python3
"""
Usage:
  python build_index.py --data-dir ../data/historical_waveforms --qdrant-host http://localhost:6333 --collection seismic_events_v2
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import uuid
import logging

import obspy

from backend.similarity.waveform import prepare_embedding_waveform
from backend.similarity.encoder import EmbeddingPipeline
from backend.similarity.index import SimilarityIndex

logger = logging.getLogger(__name__)

def build_catalog_index(waveform_dir: str, qdrant_host: str, qdrant_collection: str):
    encoder = EmbeddingPipeline()
    # Assuming host format: http://localhost:6333
    host_only = qdrant_host.replace("http://", "").replace("https://", "").split(":")[0]
    port = int(qdrant_host.split(":")[-1]) if ":" in qdrant_host else 6333
    index = SimilarityIndex(host=host_only, port=port, collection_name=qdrant_collection)
    
    waveform_path = Path(waveform_dir)
    meta_files = list(waveform_path.glob("*.json"))
    
    if not meta_files:
        logger.warning(f"No json files found in {waveform_dir}")
        return
        
    for meta_file in meta_files:
        meta = json.loads(meta_file.read_text())
        mseed_file = meta_file.with_suffix(".mseed")
        
        if not mseed_file.exists():
            logger.warning("No waveform for %s — skipping", meta["usgs_id"])
            continue
        
        try:
            st = obspy.read(str(mseed_file))
            waveform = prepare_embedding_waveform(st)
            embedding = encoder.encode(waveform)
            
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"usgs:{meta['usgs_id']}"))
            
            index.add(
                event_id=point_id,
                embedding=embedding.tolist(),
                metadata={
                    "schema_version": "2",
                    "source_type": "catalog",
                    "usgs_id": meta["usgs_id"],
                    "magnitude": meta.get("magnitude"),
                    "place": meta.get("place"),
                    "latitude": meta.get("lat"),
                    "longitude": meta.get("lon"),
                    "depth_km": meta.get("depth_km"),
                    "origin_time": meta.get("origin_time"),
                    "felt": meta.get("felt"),
                    "pager_alert": meta.get("pager_alert"),
                    "network": meta.get("network"),
                    "station": meta.get("station"),
                    "embedding_model_version": encoder.model_version,
                    "preprocessing_version": "preprocess-v1",
                }
            )
            logger.info("Added %s to Qdrant index", meta["usgs_id"])
        except Exception as e:
            logger.error("Failed to process %s: %s", meta["usgs_id"], e)
            
    logger.info("Catalog index built in collection: %s", qdrant_collection)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical_waveforms"))
    parser.add_argument("--qdrant-host", default="http://localhost:6333")
    parser.add_argument("--collection", default="seismic_events_v2")
    args = parser.parse_args()
    build_catalog_index(args.data_dir, args.qdrant_host, args.collection)

