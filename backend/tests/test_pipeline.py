import pytest
from backend.reporting.rag_generator import generate_narrative

def test_rag_generator_deterministic_fallback():
    context = {
        "prompt": "fake prompt",
        "current": {
            "station": "PAS",
            "magnitude_proxy": 5.5,
            "detection_prob": 0.99,
            "origin_time": "2024-01-01T00:00:00Z"
        },
        "forecast": {
            "method": "GMPE",
            "mmi": "VI",
            "pga_g_median": 0.15,
            "damage_band": "Light"
        },
        "historical_matches": [],
        "analysis": {
            "snr_db": 10.0,
            "noise_class": "low"
        }
    }
    
    import os
    os.environ["USE_LLM_EXPLAIN"] = "false"
    
    narrative = generate_narrative(context)
    
    assert "EXECUTIVE SUMMARY" in narrative
    assert "station PAS" in narrative
    assert "magnitude proxy of approximately 5.5" in narrative
    assert "VERIFIED FACTS BOX" in narrative
    assert "0.15" in narrative
    assert "VI" in narrative
