"""
backend/reporting/usgs_client.py
---------------------------------
Fetch and cache USGS earthquake event GeoJSON metadata.

Strategy:
  1. Check the `events_metadata` Postgres table for a cached row (keyed by usgs_id).
  2. On cache miss, call the USGS eventpage GeoJSON API and persist to DB.
  3. Return a normalised dict of fields used downstream.

This ensures:
  - Generated reports are reproducible (always re-read from cache).
  - We never hammer the USGS API during batch report generation.
"""

from __future__ import annotations

import logging
import datetime
from typing import Optional

import requests
from sqlalchemy.orm import Session

from backend.models.database import EventMetadata

logger = logging.getLogger(__name__)

# USGS event detail endpoint — returns GeoJSON
_USGS_EVENT_URL = (
    "https://earthquake.usgs.gov/earthquakes/eventpage/{usgs_id}/geojson"
)

# Timeout for USGS HTTP calls (seconds)
_HTTP_TIMEOUT = 12


def fetch_and_cache(usgs_id: str, db: Session) -> dict:
    """
    Return normalised USGS event metadata for *usgs_id*.

    Checks Postgres cache first; falls back to live USGS API call.
    Always returns a dict (may be empty if USGS is unreachable and no cache hit).

    Returned dict keys:
        usgs_id, magnitude, depth_km, lat, lon, place,
        origin_time_iso, felt, shakemap_url, pager_alert, raw_json
    """
    # --- cache lookup ---
    cached: Optional[EventMetadata] = (
        db.query(EventMetadata).filter(EventMetadata.usgs_id == usgs_id).first()
    )
    if cached:
        logger.debug("USGS cache hit for %s", usgs_id)
        return _to_dict(cached)

    # --- live API fetch ---
    raw = _fetch_from_usgs(usgs_id)
    if not raw:
        logger.warning("USGS fetch returned nothing for %s — returning empty dict", usgs_id)
        return {}

    row = _parse_and_store(usgs_id, raw, db)
    return _to_dict(row)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_from_usgs(usgs_id: str) -> dict:
    """Call the USGS GeoJSON endpoint; return parsed dict or empty dict on error."""
    url = _USGS_EVENT_URL.format(usgs_id=usgs_id)
    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        logger.warning("USGS HTTP error for %s: %s", usgs_id, exc)
    except requests.exceptions.RequestException as exc:
        logger.warning("USGS request failed for %s: %s", usgs_id, exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error fetching USGS %s: %s", usgs_id, exc)
    return {}


def _parse_and_store(usgs_id: str, raw: dict, db: Session) -> EventMetadata:
    """Parse GeoJSON, upsert into Postgres, and return the ORM row."""
    props = raw.get("properties") or {}
    coords = (raw.get("geometry") or {}).get("coordinates") or [None, None, None]

    # USGS times are milliseconds since epoch
    time_ms = props.get("time")
    origin_time = None
    if time_ms:
        try:
            origin_time = datetime.datetime.fromtimestamp(
                time_ms / 1000.0, tz=datetime.timezone.utc
            )
        except Exception:
            pass

    row = EventMetadata(
        usgs_id=usgs_id,
        magnitude=props.get("mag"),
        depth_km=coords[2],
        lat=coords[1],
        lon=coords[0],
        place=props.get("place"),
        origin_time=origin_time,
        felt=props.get("felt"),
        shakemap_url=props.get("detail"),
        pager_alert=props.get("alert"),  # None / 'green' / 'yellow' / 'orange' / 'red'
        raw_json=raw,
    )

    try:
        db.merge(row)
        db.commit()
        logger.info("Cached USGS metadata for %s", usgs_id)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to cache USGS metadata for %s: %s", usgs_id, exc)

    return row


def _to_dict(row: EventMetadata) -> dict:
    """Convert an ORM EventMetadata row to the normalised dict downstream expects."""
    return {
        "usgs_id": row.usgs_id,
        "magnitude": row.magnitude,
        "depth_km": row.depth_km,
        "lat": row.lat,
        "lon": row.lon,
        "place": row.place,
        "origin_time_iso": row.origin_time.isoformat() if row.origin_time else None,
        "felt": row.felt,
        "shakemap_url": row.shakemap_url,
        "pager_alert": row.pager_alert,
        "raw_json": row.raw_json or {},
    }
