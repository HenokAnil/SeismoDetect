"""
backend/reporting/orchestrator.py
-----------------------------------
End-to-end pipeline function called by the FastAPI background task.

Steps executed in order:
  1. Load current event from Postgres (Event + Picks + Signal rows)
  2. Re-encode waveform and retrieve similar events from Qdrant
  3. Build enriched RAG context (USGS enrichment + prompt assembly)
  4. Compute quantitative numeric forecast (GMPE)
  5. Generate narrative (T5 or template fallback)
  6. Upload artefacts to MinIO (JSON + HTML + optional PDF)
  7. Update the GeneratedReport row in Postgres (status → done)

The function is intentionally synchronous so it can be run in
FastAPI's BackgroundTasks (thread pool) without an async event loop.
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from typing import Optional

import numpy as np

from backend.api.db_session import SessionLocal
from backend.api.minio_client import minio_storage
from backend.models.database import Event, Pick, Signal, GeneratedReport, SignalQuality
from backend.reporting.context_builder import build_context
from backend.reporting.numeric_forecast import compute_forecast
from backend.reporting.rag_generator import generate_narrative
from backend.reporting.storage import upload_report_artifacts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_report_pipeline(event_id: str, report_id: str) -> None:
    """
    Full report generation pipeline.  Designed to be called from a
    BackgroundTasks worker thread.

    Parameters
    ----------
    event_id  : UUID string of the detected event (Postgres)
    report_id : UUID string of the pre-created GeneratedReport row
    """
    db = SessionLocal()
    try:
        _update_status(db, report_id, "running")
        result = _pipeline(event_id, report_id, db)
        _save_result(db, report_id, result)
        _update_status(db, report_id, "done")
        logger.info("Report %s completed for event %s", report_id, event_id)
    except Exception as exc:
        logger.exception("Report pipeline failed for event %s: %s", event_id, exc)
        _update_status(db, report_id, "error", error_msg=str(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Internal pipeline
# ---------------------------------------------------------------------------


def _pipeline(event_id: str, report_id: str, db) -> dict:
    # ---- 1. Load current event ----
    event: Optional[Event] = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise ValueError(f"Event {event_id} not found in Postgres")

    sig: Optional[Signal]  = db.query(Signal).filter(Signal.id == event.signal_id).first()
    quality = (
        db.query(SignalQuality).filter(SignalQuality.signal_id == event.signal_id).first()
        if sig else None
    )
    picks = db.query(Pick).filter(Pick.event_id == event.id).all()

    current_event = _build_current_dict(event, sig, quality, picks)

    # ---- 2. Retrieve similar events via Qdrant ----
    similar_hits = _retrieve_similar(event_id, sig, current_event)

    # ---- 3. Numeric forecast ----
    # Use a default epicentral distance of 50 km when we don't have station coords
    r_km = _estimate_distance(event, sig)
    forecast = compute_forecast(
        magnitude=event.magnitude_proxy,
        depth_km=event.depth,
        r_km=r_km,
        pager_alert=None,   # will be filled in from similar events if available
    )

    # Upgrade pager_alert confidence from top similar event if available
    if similar_hits:
        top_payload = getattr(similar_hits[0], "payload", {}) or {}
        pager_from_hit = top_payload.get("pager_alert")
        if pager_from_hit:
            forecast = compute_forecast(
                magnitude=event.magnitude_proxy,
                depth_km=event.depth,
                r_km=r_km,
                pager_alert=pager_from_hit,
            )

    # ---- 4. Build RAG context ----
    context = build_context(current_event, similar_hits, forecast, db)

    # ---- 5. Generate narrative ----
    narrative = generate_narrative(context)

    # ---- 6. Upload artefacts to MinIO ----
    art_keys = upload_report_artifacts(report_id, narrative, context, forecast)

    return {
        "narrative":     narrative,
        "forecast":      forecast,
        "similar_ids":   [str(h.id) for h in similar_hits],
        "model_version": os.getenv("REPORT_MODEL", "t5-small"),
        **art_keys,
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_current_dict(event, sig, quality, picks) -> dict:
    """Assemble a JSON-serialisable dict for the current event."""
    return {
        "event_id":        str(event.id),
        "signal_id":       str(event.signal_id) if event.signal_id else None,
        "origin_time":     event.origin_time.isoformat() if event.origin_time else None,
        "detection_prob":  event.detection_prob,
        "magnitude_proxy": event.magnitude_proxy,
        "model_version":   event.model_version,
        "lat":             event.lat if event.lat is not None else "N/A",
        "lon":             event.lon if event.lon is not None else "N/A",
        "depth_km":        event.depth if event.depth is not None else "N/A",
        "network":         sig.network if sig else "??",
        "station":         sig.station if sig else "??",
        "channel":         sig.channel if sig else "??",
        "snr":             quality.snr if quality else None,
        "noise_label":     quality.noise_label if quality else None,
        "picks": [
            {
                "phase":      p.phase,
                "pick_time":  p.pick_time.isoformat() if p.pick_time else None,
                "confidence": p.confidence,
            }
            for p in picks
        ],
    }


def _retrieve_similar(event_id: str, sig, current_event: dict) -> list:
    """Re-encode the waveform and search Qdrant for similar events."""
    try:
        # Lazy imports to avoid circular deps at module load time
        from backend.similarity.encoder import EmbeddingPipeline
        from backend.similarity.index import SimilarityIndex
        import obspy

        raw_bytes = minio_storage.download_bytes(sig.s3_key)
        st = obspy.read(io.BytesIO(raw_bytes))
        w = (
            np.stack([tr.data for tr in st[:3]])
            if len(st) >= 3
            else np.repeat(st[0].data[np.newaxis, :], 3, axis=0)
        )

        pipeline = EmbeddingPipeline()
        embedding = pipeline.encode(w).tolist()

        index = SimilarityIndex()
        hits  = index.search(embedding, limit=6)

        # Exclude the event itself from results
        return [h for h in hits if str(h.id) != event_id][:5]

    except Exception as exc:
        logger.warning("Qdrant retrieval failed: %s — returning empty list", exc)
        return []


def _estimate_distance(event, sig) -> float:
    """
    Very rough distance estimate.
    Falls back to 50 km default when coordinates are unavailable.
    """
    if event.lat and event.lon:
        # No station coords stored yet — use 50 km default
        return 50.0
    return 50.0


def _update_status(db, report_id: str, status: str, error_msg: str = None) -> None:
    """Update the GeneratedReport.status (and optionally error_msg) in Postgres."""
    try:
        row = db.query(GeneratedReport).filter(GeneratedReport.id == report_id).first()
        if row:
            row.status = status
            if error_msg:
                row.error_msg = error_msg
            db.commit()
    except Exception as exc:
        logger.error("Failed to update report status: %s", exc)
        db.rollback()


def _save_result(db, report_id: str, result: dict) -> None:
    """Persist the generated content into the GeneratedReport row."""
    try:
        row = db.query(GeneratedReport).filter(GeneratedReport.id == report_id).first()
        if row:
            row.narrative_text = result.get("narrative")
            row.forecast_json  = result.get("forecast")
            row.similar_ids    = result.get("similar_ids")
            row.model_version  = result.get("model_version")
            row.json_s3_key    = result.get("json_s3_key")
            row.html_s3_key    = result.get("html_s3_key")
            row.pdf_s3_key     = result.get("pdf_s3_key")
            db.commit()
    except Exception as exc:
        logger.error("Failed to save report result: %s", exc)
        db.rollback()
