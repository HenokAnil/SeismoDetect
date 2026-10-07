"""
backend/llm/explain.py
------------------------
Legacy entry point kept for backwards compatibility with the existing
`run_ml_pipeline()` call in main.py.

For the full RAG pipeline use backend/reporting/rag_generator.py directly.
This module delegates to it when a reporting context dict is passed, and
falls back to a simple inline template for the legacy call signature.
"""

import os
import json


def generate_explanation(structured_data: dict) -> str:
    """
    Generate a textual explanation for a detected event.

    Accepts either:
      - Legacy dict (event detection output from run_ml_pipeline) — produces
        a short inline explanation via the template.
      - Full RAG context dict (contains 'prompt', 'forecast', 'similar') —
        delegates to rag_generator for a full narrative report.
    """
    # Full RAG context path
    if "prompt" in structured_data and "forecast" in structured_data:
        try:
            from backend.reporting.rag_generator import generate_narrative
            return generate_narrative(structured_data)
        except Exception as exc:
            return _template_based_explanation(structured_data) + f"\n\n(RAG error: {exc})"

    # Legacy template path (original behaviour preserved)
    return _template_based_explanation(structured_data)


def _template_based_explanation(data: dict) -> str:
    """
    Simple deterministic explanation for the inline (non-RAG) call.
    Guaranteed to contain only facts from the input dict.
    """
    det_prob = data.get("detection_prob", 0) or 0
    mag = data.get("magnitude_proxy", 0)

    explanation = (
        f"A seismic event was detected with {det_prob * 100:.1f}% confidence. "
        f"The estimated magnitude proxy is {mag}. "
    )

    picks = data.get("picks", [])
    if picks:
        explanation += f"A total of {len(picks)} phase pick(s) were identified. "

    similar = data.get("similar_events", [])
    if similar:
        top = similar[0]
        explanation += (
            f"The closest historical match scored {top.get('score', '?')} similarity. "
        )

    explanation += (
        "All numeric values are model estimates and must be verified against "
        "instrument data before any emergency response."
    )
    return explanation
