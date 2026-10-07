"""
tests/reporting/test_context_builder.py
-----------------------------------------
Unit tests for context_builder.build_context().
Qdrant hits and DB calls are fully mocked.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qdrant_hit(hit_id: str, score: float, payload: dict = None):
    hit = MagicMock()
    hit.id = hit_id
    hit.score = score
    hit.payload = payload or {}
    return hit


def _sample_current_event():
    return {
        "event_id":        "evt-123",
        "signal_id":       "sig-456",
        "origin_time":     "2024-01-15T10:22:05Z",
        "detection_prob":  0.92,
        "magnitude_proxy": 5.1,
        "station":         "PAS",
        "network":         "CI",
        "depth_km":        12.0,
        "picks": [
            {"phase": "P", "pick_time": "2024-01-15T10:22:23Z", "confidence": 0.89},
            {"phase": "S", "pick_time": "2024-01-15T10:22:35Z", "confidence": 0.91},
        ],
    }


def _sample_forecast():
    return {
        "pga_g_median": 0.12,
        "pga_g_lower":  0.07,
        "pga_g_upper":  0.22,
        "mmi":          "VI",
        "damage_band":  "Low-Moderate",
        "confidence":   "high",
        "method":       "GMPE-BA08-simplified",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_context_returns_required_keys():
    from backend.reporting.context_builder import build_context

    hits = [_make_qdrant_hit("hit-001", 0.93)]
    db   = MagicMock()

    with patch("backend.reporting.context_builder.fetch_and_cache", return_value={}):
        ctx = build_context(_sample_current_event(), hits, _sample_forecast(), db)

    assert "current" in ctx
    assert "similar" in ctx
    assert "forecast" in ctx
    assert "prompt" in ctx
    assert "n_similar" in ctx


def test_build_context_n_similar_matches_hits():
    from backend.reporting.context_builder import build_context

    hits = [
        _make_qdrant_hit("hit-001", 0.93),
        _make_qdrant_hit("hit-002", 0.85),
        _make_qdrant_hit("hit-003", 0.77),
    ]
    db = MagicMock()

    with patch("backend.reporting.context_builder.fetch_and_cache", return_value={}):
        ctx = build_context(_sample_current_event(), hits, _sample_forecast(), db)

    assert ctx["n_similar"] == 3


def test_build_context_similar_sorted_by_score_descending():
    from backend.reporting.context_builder import build_context

    hits = [
        _make_qdrant_hit("low",  0.60),
        _make_qdrant_hit("high", 0.95),
        _make_qdrant_hit("mid",  0.80),
    ]
    db = MagicMock()

    with patch("backend.reporting.context_builder.fetch_and_cache", return_value={}):
        ctx = build_context(_sample_current_event(), hits, _sample_forecast(), db)

    scores = [s["similarity"] for s in ctx["similar"]]
    assert scores == sorted(scores, reverse=True)


def test_prompt_contains_instruction_block():
    from backend.reporting.context_builder import build_context

    db = MagicMock()
    with patch("backend.reporting.context_builder.fetch_and_cache", return_value={}):
        ctx = build_context(_sample_current_event(), [], _sample_forecast(), db)

    prompt = ctx["prompt"]
    assert "INSTRUCTIONS TO WRITER" in prompt
    assert "Do NOT invent" in prompt
    assert "NUMERIC_FORECAST" in prompt
    assert "REPORT:" in prompt


def test_prompt_contains_current_event_facts():
    from backend.reporting.context_builder import build_context

    db = MagicMock()
    with patch("backend.reporting.context_builder.fetch_and_cache", return_value={}):
        ctx = build_context(_sample_current_event(), [], _sample_forecast(), db)

    assert "magnitude_proxy=5.1" in ctx["prompt"]
    assert "PAS" in ctx["prompt"]


def test_usgs_metadata_merged_into_similar():
    from backend.reporting.context_builder import build_context

    usgs_meta = {
        "magnitude":      6.4,
        "place":          "Ridgecrest, CA",
        "origin_time_iso": "2019-07-04T17:33:49Z",
        "depth_km":       8.0,
        "felt":           150,
        "pager_alert":    "orange",
        "shakemap_url":   "https://example.usgs.gov/shakemap",
    }
    hit = _make_qdrant_hit("ci38443183", 0.88, payload={"usgs_id": "ci38443183"})
    db  = MagicMock()

    with patch("backend.reporting.context_builder.fetch_and_cache", return_value=usgs_meta):
        ctx = build_context(_sample_current_event(), [hit], _sample_forecast(), db)

    sim = ctx["similar"][0]
    assert sim["magnitude"] == 6.4
    assert sim["place"] == "Ridgecrest, CA"
    assert sim["pager_alert"] == "orange"
    assert sim["felt"] == 150
