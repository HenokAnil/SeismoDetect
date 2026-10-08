"""
backend/reporting/context_builder.py
--------------------------------------
Assemble an enriched RAG context dict from:
  - current detected event (from Postgres)
  - Qdrant similar-event hits (already retrieved)
  - USGS metadata for each hit (via usgs_client)
  - numeric forecast (from numeric_forecast module)
"""

from __future__ import annotations
import logging
from typing import List
from sqlalchemy.orm import Session
from backend.reporting.usgs_client import fetch_and_cache
from backend.reporting.comparison import compare_events
import datetime

logger = logging.getLogger(__name__)

def build_context(
    current_event: dict,
    similar_hits: list,
    numeric_forecast: dict,
    db: Session,
) -> dict:
    enriched_matches = _enrich_hits(similar_hits, current_event, db)
    
    limitations = [
        "Waveform similarity does not confirm shared source region or mechanism."
    ]
    if current_event.get("depth_km") is None or current_event.get("lat") is None:
        limitations.append("Source location and depth unavailable from single-station processing.")
    if numeric_forecast.get("inputs", {}).get("depth_source") == "default_assumption":
        limitations.append("Forecast depth uses explicit default assumptions.")
    if numeric_forecast.get("inputs", {}).get("distance_source") == "default_assumption":
        limitations.append("Forecast distance uses explicit default assumptions.")
    if current_event.get("magnitude_proxy") is not None:
        limitations.append("Magnitude is a proxy estimate, not a verified catalog magnitude.")

    prompt = _build_prompt(current_event, enriched_matches, numeric_forecast)

    return {
        "current": current_event,
        "analysis": current_event.get("analysis", {}),
        "historical_matches": enriched_matches,
        "forecast": numeric_forecast,
        "limitations": limitations,
        "prompt": prompt,
        "metadata": {
            "pipeline_version": "report-v2",
            "embedding_model": "seismo-cnn-v1",
            "preprocessing_version": "preprocess-v1",
            "generated_at": datetime.datetime.utcnow().isoformat(),
        }
    }

def _enrich_hits(hits: list, current_event: dict, db: Session) -> List[dict]:
    results: List[dict] = []
    for rank, hit in enumerate(hits, start=1):
        payload = getattr(hit, "payload", {}) or {}
        usgs_id = payload.get("usgs_id")
        score = float(getattr(hit, "score", 0.0))

        usgs_meta = {}
        if usgs_id:
            try:
                usgs_meta = fetch_and_cache(usgs_id, db)
            except Exception as exc:
                logger.warning("USGS enrichment failed for %s: %s", usgs_id, exc)

        catalog_meta = {
            "magnitude": usgs_meta.get("magnitude") if usgs_meta.get("magnitude") is not None else payload.get("magnitude"),
            "place": usgs_meta.get("place") if usgs_meta.get("place") is not None else payload.get("place"),
            "lat": usgs_meta.get("lat") if usgs_meta.get("lat") is not None else payload.get("latitude"),
            "lon": usgs_meta.get("lon") if usgs_meta.get("lon") is not None else payload.get("longitude"),
            "depth_km": usgs_meta.get("depth_km") if usgs_meta.get("depth_km") is not None else payload.get("depth_km"),
            "origin_time": usgs_meta.get("origin_time_iso") or payload.get("origin_time"),
            "felt": usgs_meta.get("felt") if usgs_meta.get("felt") is not None else payload.get("felt"),
            "pager_alert": usgs_meta.get("pager_alert") or payload.get("pager_alert"),
        }

        historical_event = {
            "station": payload.get("station"),
            "network": payload.get("network"),
            "catalog": catalog_meta,
        }

        row = {
            "rank": rank,
            "id": str(hit.id),
            "similarity_score": round(score, 3),
            "usgs_id": usgs_id or "unknown",
            "catalog": catalog_meta,
            "comparison": compare_events(current_event, historical_event),
            "metadata_source": "usgs_cache" if usgs_meta else "qdrant_payload",
            "shakemap_url": usgs_meta.get("shakemap_url"),
        }
        results.append(row)

    results.sort(key=lambda r: r["similarity_score"], reverse=True)
    return results

def _build_prompt(current: dict, matches: List[dict], forecast: dict) -> str:
    return """You are a seismic event analyst writing a professional report.

Write a concise analysis using ONLY the evidence supplied below.

Rules:
1. Do not invent earthquake location, depth, magnitude, casualties, or damage.
2. Never describe the similarity score as a probability or percentage.
3. A waveform similarity match does NOT prove shared source, mechanism, or location.
4. Distinguish the current event from historical reference events clearly.
5. Call magnitude_proxy a "magnitude proxy", not a "confirmed magnitude".
6. Explicitly mention missing information (depth unknown, location unavailable).
7. Do not infer damage from historical events for the current event.
8. Do not use historical PAGER values as the current forecast.
9. Use professional but readable language.

Produce exactly these five sections:

## EXECUTIVE SUMMARY
## EVENT ANALYSIS
## HISTORICAL CONTEXT
## FORECAST & INTERPRETATION
## LIMITATIONS & RECOMMENDATION
"""
