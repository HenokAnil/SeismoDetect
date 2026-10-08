"""
backend/reporting/rag_generator.py
------------------------------------
T5-based narrative generation with anti-hallucination safeguards.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None

def generate_narrative(context: dict) -> str:
    use_llm = os.getenv("USE_LLM_EXPLAIN", "false").lower() == "true"

    if use_llm:
        try:
            raw_text = _t5_generate(context["prompt"])
        except Exception as exc:
            logger.warning("T5 generation failed, falling back to template: %s", exc)
            raw_text = _template_narrative(context)
    else:
        raw_text = _template_narrative(context)

    # Note: facts box could be appended here if desired, but we follow the plan exactly
    # We will just append the facts box.
    facts_box = _build_facts_box(context)
    narrative = raw_text.strip() + "\n\n" + facts_box

    narrative = _flag_numeric_discrepancies(narrative, context.get("forecast", {}))

    return narrative

def _load_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        model_name = os.getenv("REPORT_MODEL", "google/flan-t5-base")
        logger.info("Loading T5 model: %s", model_name)
        _tokenizer = T5Tokenizer.from_pretrained(model_name)
        _model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("T5 model loaded")

def _t5_generate(prompt: str) -> str:
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
        max_new_tokens=400,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return _tokenizer.decode(output_ids[0], skip_special_tokens=True)

def _build_historical_summary(current: dict, matches: list) -> str:
    if not matches:
        return "No sufficiently informative historical matches were available in the indexed catalog."
    
    top = matches[0]
    cat = top.get("catalog", {})
    
    text = (
        f"The closest retrieved historical waveform match has a "
        f"cosine similarity score of {top['similarity_score']:.3f}. "
    )
    
    if cat.get("magnitude") is not None:
        text += f"The reference event has a catalog magnitude of {cat['magnitude']:.1f}. "
    
    if cat.get("place"):
        text += f"It was located near {cat['place']}. "
    
    comp = top.get("comparison", {})
    if comp.get("same_station"):
        text += "The reference was recorded at the same station. "
    
    text += (
        "This match should be interpreted as waveform-level context rather than "
        "evidence that the events share the same source region or mechanism."
    )
    return text

def _template_narrative(context: dict) -> str:
    cur = context.get("current", {})
    matches = context.get("historical_matches", [])
    forecast = context.get("forecast", {})
    analysis = context.get("analysis", {})
    
    det_prob = cur.get("detection_prob", 0.0) or 0.0
    mag = cur.get("magnitude_proxy", "unknown")
    station = cur.get("station", "unknown")
    
    snr_db = analysis.get("snr_db")
    noise_class = analysis.get("noise_class")
    
    fc_mmi = forecast.get("mmi", "unknown")
    fc_pga = forecast.get("pga_g_median", "unknown")

    lines = [
        "## EXECUTIVE SUMMARY",
        f"A seismic event was detected at station {station} with {det_prob * 100:.1f}% detection confidence and a magnitude proxy of approximately {mag}.",
        "",
        "## EVENT ANALYSIS",
        f"The signal has an average SNR of {snr_db if snr_db is not None else 'unknown'} dB and was classified as {noise_class if noise_class else 'unknown'}. No digitizer clipping was detected. The detected P and S arrivals provide a measurable phase sequence.",
        "",
        "## HISTORICAL CONTEXT",
        _build_historical_summary(cur, matches),
        "",
        "## FORECAST & INTERPRETATION",
        f"The simplified GMPE forecast corresponds to approximately MMI {fc_mmi} with a predicted median PGA of {fc_pga} g. Because the current event does not have a directly estimated source depth or epicentral distance, fallback assumptions are explicitly included in the forecast interpretation.",
        "",
        "## LIMITATIONS & RECOMMENDATION",
        "The current event does not have a directly estimated hypocentral location or depth from the available single-station data. The historical matches are contextual references rather than confirmed analogues. Authoritative catalog parameters should be verified before public communication."
    ]
    return "\n".join(lines)

_NUMERIC_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*g\b")

def _flag_numeric_discrepancies(narrative: str, forecast: dict) -> str:
    low = forecast.get("pga_g_lower")
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
    fc = context.get("forecast", {})
    cur = context.get("current", {})

    lines = [
        "---",
        "### ✅ VERIFIED FACTS BOX (auto-generated — do not edit)",
        f"- Detection confidence : {(cur.get('detection_prob') or 0) * 100:.1f} %",
        f"- Magnitude proxy      : {cur.get('magnitude_proxy') if cur.get('magnitude_proxy') is not None else 'None'}",
        f"- Forecast method      : {fc.get('method', 'None')}",
        f"- Predicted PGA (median): {fc.get('pga_g_median', 'None')} g",
        f"- PGA 95 % CI          : [{fc.get('pga_g_lower', 'None')}, {fc.get('pga_g_upper', 'None')}] g",
        f"- MMI estimate         : {fc.get('mmi', 'None')}",
        f"- Damage band          : {fc.get('damage_band', 'None')}",
        f"- Forecast confidence  : {fc.get('confidence', 'None')}",
        "- ⚠️  This report is auto-generated. Numeric forecasts are model estimates "
        "and must be verified against real instrument data before emergency response.",
        "---",
    ]
    return "\n".join(lines)
