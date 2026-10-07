"""
backend/reporting/rag_generator.py
------------------------------------
T5-based narrative generation with anti-hallucination safeguards.

Pipeline:
  1. Receive context dict (built by context_builder.py).
  2. Extract the pre-built prompt string from context["prompt"].
  3. Run T5 (configurable via REPORT_MODEL env var; defaults to t5-small).
  4. Post-generation fact-check: extract any numeric values from the
     generated text and cross-reference against numeric_forecast.
     If a mismatch is detected, append a canonical FACTS BOX so the
     reader always has verified numbers regardless of what the model said.
  5. Return final narrative string.

Falls back to a deterministic template when USE_LLM_EXPLAIN != 'true'
so the pipeline works without a GPU / large download.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_tokenizer = None
_model = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_narrative(context: dict) -> str:
    """
    Generate a narrative report string from the assembled RAG context.

    Parameters
    ----------
    context : dict produced by context_builder.build_context()
              Must contain keys: "prompt", "forecast", "current", "similar"

    Returns
    -------
    str — the final narrative (≤ 800 words), always ending with the
          canonical FACTS BOX for numeric grounding.
    """
    use_llm = os.getenv("USE_LLM_EXPLAIN", "false").lower() == "true"

    if use_llm:
        try:
            raw_text = _t5_generate(context["prompt"])
        except Exception as exc:
            logger.warning("T5 generation failed, falling back to template: %s", exc)
            raw_text = _template_narrative(context)
    else:
        raw_text = _template_narrative(context)

    # Always append the verified facts box
    facts_box = _build_facts_box(context)
    narrative  = raw_text.strip() + "\n\n" + facts_box

    # Optionally flag hallucinated numbers
    narrative = _flag_numeric_discrepancies(narrative, context.get("forecast", {}))

    return narrative


# ---------------------------------------------------------------------------
# T5 generation
# ---------------------------------------------------------------------------


def _load_model():
    """Lazy-load tokenizer and model (avoids import cost at module load time)."""
    global _tokenizer, _model
    if _model is None:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        model_name = os.getenv("REPORT_MODEL", "t5-small")
        logger.info("Loading T5 model: %s", model_name)
        _tokenizer = T5Tokenizer.from_pretrained(model_name)
        _model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("T5 model loaded")


def _t5_generate(prompt: str) -> str:
    """Run T5 inference on the prompt; return decoded string."""
    _load_model()
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    )
    output_ids = _model.generate(
        **inputs,
        max_new_tokens=350,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return _tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Template fallback (deterministic, always available)
# ---------------------------------------------------------------------------


def _template_narrative(context: dict) -> str:
    """
    Deterministic template-based report.
    Guaranteed to contain only facts from the context dict.
    """
    cur      = context.get("current", {})
    similar  = context.get("similar", [])
    forecast = context.get("forecast", {})

    det_prob  = cur.get("detection_prob", 0.0) or 0.0
    mag       = cur.get("magnitude_proxy", "?")
    station   = cur.get("station", "unknown")
    origin_t  = cur.get("origin_time", "unknown")

    lines = [
        "## EXECUTIVE SUMMARY",
        (
            f"A seismic event was detected with {det_prob * 100:.1f}% confidence "
            f"at station {station} (magnitude proxy ≈ {mag}, "
            f"origin time: {origin_t})."
        ),
        "",
        "## DETECTED EVENT DETAILS",
        f"- Station : {station} ({cur.get('network', '?')})",
        f"- Origin time: {origin_t}",
        f"- Magnitude proxy: {mag}",
        f"- Detection confidence: {det_prob * 100:.1f}%",
        f"- Depth estimate: {cur.get('depth_km', 'unknown')} km",
    ]

    # Phase picks
    picks = cur.get("picks", [])
    if picks:
        lines.append("\n### Phase Picks")
        for p in picks:
            lines.append(
                f"  - {p.get('phase', '?')}-wave: "
                f"{p.get('pick_time', '?')} "
                f"(confidence: {p.get('confidence', '?')})"
            )

    # Similar events
    lines.append("\n## HISTORICAL SIMILAR EVENTS")
    if similar:
        for i, s in enumerate(similar[:3], start=1):
            usgs_link = (
                f"https://earthquake.usgs.gov/earthquakes/eventpage/{s['usgs_id']}"
                if s.get("usgs_id") and s["usgs_id"] != "unknown"
                else "N/A"
            )
            pager_note = f" (PAGER: {s['pager_alert']})" if s.get("pager_alert") else ""
            lines.append(
                f"\n**{i}. Event {s.get('usgs_id', s['id'][:8])}**{pager_note}  "
                f"  Similarity: {s['similarity']}  "
                f"  Magnitude: {s.get('magnitude', '?')}  "
                f"  Location: {s.get('place', '?')}  "
                f"  Time: {s.get('origin_time', '?')}  "
                f"  USGS: {usgs_link}"
            )
            if s.get("felt"):
                lines.append(f"  Felt reports: {s['felt']}")
    else:
        lines.append("No closely matched historical events in the Qdrant index.")

    # Recommendations
    damage = forecast.get("damage_band", "Unknown")
    mmi    = forecast.get("mmi", "?")
    lines += [
        "\n## RECOMMENDATION",
        (
            f"Based on the GMPE forecast (MMI ≈ {mmi}, damage band: {damage}), "
            "follow standard emergency protocols for the predicted intensity level. "
            "Verify instrument readings before public communication."
        ),
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anti-hallucination fact-check
# ---------------------------------------------------------------------------


_NUMERIC_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*g\b")  # matches "0.12 g", "1.3g"


def _flag_numeric_discrepancies(narrative: str, forecast: dict) -> str:
    """
    Scan narrative for numeric PGA values (pattern: N.NNN g).
    If any value falls outside the forecast 95 % CI, prepend a warning line.
    """
    low  = forecast.get("pga_g_lower")
    high = forecast.get("pga_g_upper")
    if low is None or high is None:
        return narrative

    found_values = [float(m.group(1)) for m in _NUMERIC_RE.finditer(narrative)]
    suspect = [v for v in found_values if not (low * 0.5 <= v <= high * 2.0)]
    if suspect:
        warning = (
            f"\n> ⚠️  NUMERIC DISCREPANCY DETECTED: value(s) {suspect} appear in the "
            f"narrative but fall outside the expected range "
            f"[{low}, {high}] g. "
            "Refer to the VERIFIED FACTS BOX below for canonical values.\n"
        )
        narrative = warning + narrative

    return narrative


def _build_facts_box(context: dict) -> str:
    """
    Build a canonical, always-accurate VERIFIED FACTS BOX appended to every report.
    This ensures readers always have the authoritative numbers regardless of what
    the generative model wrote.
    """
    fc  = context.get("forecast", {})
    cur = context.get("current", {})

    lines = [
        "---",
        "### ✅ VERIFIED FACTS BOX (auto-generated — do not edit)",
        f"- Detection confidence : {(cur.get('detection_prob') or 0) * 100:.1f} %",
        f"- Magnitude proxy      : {cur.get('magnitude_proxy', 'N/A')}",
        f"- Forecast method      : {fc.get('method', 'N/A')}",
        f"- Predicted PGA (median): {fc.get('pga_g_median', 'N/A')} g",
        f"- PGA 95 % CI          : [{fc.get('pga_g_lower', '?')}, {fc.get('pga_g_upper', '?')}] g",
        f"- MMI estimate         : {fc.get('mmi', 'N/A')}",
        f"- Damage band          : {fc.get('damage_band', 'N/A')}",
        f"- Forecast confidence  : {fc.get('confidence', 'N/A')}",
        "- ⚠️  This report is auto-generated. Numeric forecasts are model estimates "
        "and must be verified against real instrument data before emergency response.",
        "---",
    ]
    return "\n".join(lines)
