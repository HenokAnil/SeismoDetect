"""
backend/reporting/context_builder.py
--------------------------------------
Assemble an enriched RAG context dict from:
  - current detected event (from Postgres)
  - Qdrant similar-event hits (already retrieved)
  - USGS metadata for each hit (via usgs_client)
  - numeric forecast (from numeric_forecast module)

The returned dict is passed directly to:
  - rag_generator.generate_narrative()   — produces the text narrative
  - the PDF renderer                     — populates tables & facts box
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.reporting.usgs_client import fetch_and_cache

logger = logging.getLogger(__name__)


def build_context(
    current_event: dict,
    similar_hits: list,
    numeric_forecast: dict,
    db: Session,
) -> dict:
    """
    Build the full RAG context dict.

    Parameters
    ----------
    current_event   : dict from Postgres Event row + picks
    similar_hits    : list of Qdrant ScoredPoint objects
    numeric_forecast: output of numeric_forecast.compute_forecast()
    db              : active SQLAlchemy session

    Returns
    -------
    {
        "current":  {...},          # cleaned current event facts
        "similar":  [{...}, ...],   # enriched historical hits
        "forecast": {...},          # numeric predictions
        "prompt":   "...",          # ready-to-use LLM prompt string
        "n_similar": int,
    }
    """
    enriched_similar = _enrich_hits(similar_hits, db)
    prompt = _build_prompt(current_event, enriched_similar, numeric_forecast)

    return {
        "current":  current_event,
        "similar":  enriched_similar,
        "forecast": numeric_forecast,
        "prompt":   prompt,
        "n_similar": len(enriched_similar),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _enrich_hits(hits: list, db: Session) -> List[dict]:
    """
    For each Qdrant hit, extract payload and optionally fetch USGS metadata.
    Returns a list of dicts, sorted by similarity score (descending).
    """
    results: List[dict] = []
    for hit in hits:
        payload  = getattr(hit, "payload", {}) or {}
        usgs_id  = payload.get("usgs_id")
        score    = float(getattr(hit, "score", 0.0))

        # Attempt USGS enrichment when a usgs_id is stored in the Qdrant payload
        usgs_meta: dict = {}
        if usgs_id:
            try:
                usgs_meta = fetch_and_cache(usgs_id, db)
            except Exception as exc:
                logger.warning("USGS enrichment failed for %s: %s", usgs_id, exc)

        row = {
            "id":           str(hit.id),
            "similarity":   round(score, 3),
            "usgs_id":      usgs_id or "unknown",
            "magnitude":    usgs_meta.get("magnitude") or payload.get("magnitude_proxy") or "N/A",
            "place":        usgs_meta.get("place") or payload.get("station") or "Unknown location",
            "origin_time":  usgs_meta.get("origin_time_iso") or payload.get("origin_time") or "N/A",
            "depth_km":     usgs_meta.get("depth_km") or payload.get("depth_km") or "N/A",
            "felt":         usgs_meta.get("felt"),
            "pager_alert":  usgs_meta.get("pager_alert"),
            "shakemap_url": usgs_meta.get("shakemap_url"),
            "station":      payload.get("station"),
            "network":      payload.get("network"),
        }
        results.append(row)

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results


def _build_prompt(
    current: dict,
    similar: List[dict],
    forecast: dict,
) -> str:
    """
    Build the exact T5 prompt string following the plan specification.

    Safety constraints are embedded verbatim so the generator can never
    accidentally omit them even when the prompt is truncated.
    """
    # --- current event facts ---
    cur_facts = (
        f"magnitude_proxy={current.get('magnitude_proxy', '?')}, "
        f"detection_confidence={current.get('detection_prob', '?')}, "
        f"origin_time={current.get('origin_time', '?')}, "
        f"station={current.get('station', '?')}, "
        f"network={current.get('network', '?')}, "
        f"depth_km={current.get('depth_km', 'unknown')}"
    )

    # --- similar event bullets ---
    bullets: List[str] = []
    for i, s in enumerate(similar[:5], start=1):
        pager_str = f", PAGER={s['pager_alert']}" if s.get("pager_alert") else ""
        felt_str  = f", felt_reports={s['felt']}" if s.get("felt") is not None else ""
        usgs_url  = (
            f"  [USGS link: https://earthquake.usgs.gov/earthquakes/eventpage/{s['usgs_id']}]"
            if s.get("usgs_id") and s["usgs_id"] != "unknown"
            else ""
        )
        bullets.append(
            f"  {i}. Event {s['usgs_id']} — "
            f"mag={s.get('magnitude', '?')}, "
            f"place={s.get('place', '?')}, "
            f"depth_km={s.get('depth_km', '?')}, "
            f"time={s.get('origin_time', '?')}, "
            f"similarity_score={s['similarity']}"
            f"{pager_str}{felt_str}"
            f"{usgs_url}"
        )
    bullet_block = "\n".join(bullets) if bullets else "  (no similar events retrieved)"

    # --- numeric forecast block ---
    fc = forecast
    forecast_block = (
        f"  predicted_PGA_median={fc.get('pga_g_median', '?')} g, "
        f"95%_CI=[{fc.get('pga_g_lower', '?')}, {fc.get('pga_g_upper', '?')}] g, "
        f"MMI={fc.get('mmi', '?')}, "
        f"damage_band={fc.get('damage_band', '?')}, "
        f"confidence={fc.get('confidence', '?')}, "
        f"method={fc.get('method', '?')}"
    )

    # --- assemble prompt ---
    prompt = f"""FACTS:
- Current event: {cur_facts}
- Retrieved similar historical events:
{bullet_block}

NUMERIC_FORECAST:
{forecast_block}

INSTRUCTIONS TO WRITER:
- Only use facts from the FACTS and NUMERIC_FORECAST sections above.
- Do NOT invent casualty, fatality, or damage numbers not present above.
- If the text mentions damage or casualties, prefix with "(Estimated — verify)".
- Compare the current event with the retrieved historical events: state similarities and relevant outcomes.
- Provide an executive summary (1-2 lines), then a details section, then a short recommendation.
- Always include a Confidence line with the confidence level and numeric uncertainty band from NUMERIC_FORECAST.
- Use simple language suitable for emergency managers.
- Do not cite sources beyond event IDs and the USGS eventpage URLs listed above.
REPORT:"""

    return prompt
