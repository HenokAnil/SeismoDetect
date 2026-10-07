"""
tests/reporting/test_rag_generator.py
---------------------------------------
Unit tests for the RAG narrative generator.
T5 model loading is mocked — tests run instantly without a GPU.
"""

import pytest
import re
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_context(det_prob=0.88, mag=5.1, similar=None, forecast=None):
    from backend.reporting.context_builder import build_context
    from unittest.mock import MagicMock, patch

    current = {
        "event_id":        "evt-abc",
        "signal_id":       "sig-xyz",
        "origin_time":     "2024-03-10T14:22:00Z",
        "detection_prob":  det_prob,
        "magnitude_proxy": mag,
        "station":         "PAS",
        "network":         "CI",
        "depth_km":        10.0,
        "picks": [],
    }
    forecast = forecast or {
        "pga_g_median": 0.10,
        "pga_g_lower":  0.06,
        "pga_g_upper":  0.18,
        "pga_95_ci":    [0.06, 0.18],
        "mmi":          "V",
        "damage_band":  "Low",
        "confidence":   "medium",
        "method":       "GMPE-BA08-simplified",
    }
    similar = similar or []
    db = MagicMock()
    with patch("backend.reporting.context_builder.fetch_and_cache", return_value={}):
        return build_context(current, similar, forecast, db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_template_narrative_contains_facts_box():
    """Template fallback always appends a verified facts box."""
    from backend.reporting.rag_generator import generate_narrative

    ctx = _sample_context()
    # Force template mode (USE_LLM_EXPLAIN is not set)
    result = generate_narrative(ctx)

    assert "VERIFIED FACTS BOX" in result
    assert "Predicted PGA" in result


def test_template_narrative_has_executive_summary():
    from backend.reporting.rag_generator import generate_narrative

    ctx = _sample_context(det_prob=0.92)
    result = generate_narrative(ctx)

    assert "88.0%" in result or "92.0%" in result  # depending on fixture
    assert "EXECUTIVE SUMMARY" in result


def test_facts_box_contains_canonical_pga():
    from backend.reporting.rag_generator import generate_narrative

    ctx = _sample_context()
    result = generate_narrative(ctx)

    # Python formats 0.10 as '0.1' — check the rendered string, not the literal
    assert "0.1 g" in result or "0.10 g" in result  # pga_g_median
    assert "0.06" in result   # pga_g_lower
    assert "0.18" in result   # pga_g_upper



def test_t5_fallback_on_model_error():
    """When T5 raises, the template fallback is used — no crash."""
    from backend.reporting.rag_generator import generate_narrative
    import os

    ctx = _sample_context()
    with patch.dict("os.environ", {"USE_LLM_EXPLAIN": "true"}):
        with patch("backend.reporting.rag_generator._t5_generate", side_effect=RuntimeError("GPU OOM")):
            result = generate_narrative(ctx)

    # Should still return something with the facts box
    assert "VERIFIED FACTS BOX" in result


def test_hallucination_flag_appended_for_out_of_range_values():
    """If the narrative contains a PGA number outside CI, a warning is prepended."""
    from backend.reporting.rag_generator import _flag_numeric_discrepancies

    narrative = "The predicted acceleration was 9.99 g which is extreme."
    forecast = {"pga_g_lower": 0.05, "pga_g_upper": 0.20}

    flagged = _flag_numeric_discrepancies(narrative, forecast)
    assert "NUMERIC DISCREPANCY" in flagged


def test_no_flag_when_value_in_range():
    from backend.reporting.rag_generator import _flag_numeric_discrepancies

    narrative = "The predicted acceleration was 0.12 g."
    forecast = {"pga_g_lower": 0.06, "pga_g_upper": 0.20}

    result = _flag_numeric_discrepancies(narrative, forecast)
    assert "NUMERIC DISCREPANCY" not in result


def test_narrative_length_under_800_words():
    from backend.reporting.rag_generator import generate_narrative

    ctx = _sample_context()
    result = generate_narrative(ctx)
    word_count = len(result.split())
    assert word_count <= 800, f"Narrative too long: {word_count} words"
