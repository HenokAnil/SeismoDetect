"""
tests/reporting/test_numeric_forecast.py
------------------------------------------
Unit tests for the GMPE numeric forecast module.
No external dependencies — all pure math.
"""

import math
import pytest

from backend.reporting.numeric_forecast import (
    compute_forecast,
    _gmpe_forecast,
    _pga_to_mmi,
    _damage_from_mmi,
    _damage_from_pager,
    _confidence_level,
)


# ---------------------------------------------------------------------------
# GMPE output structure
# ---------------------------------------------------------------------------


def test_compute_forecast_returns_required_keys():
    result = compute_forecast(magnitude=5.0, depth_km=10.0, r_km=50.0)
    required = {"pga_g_median", "pga_g_lower", "pga_g_upper",
                "pga_95_ci", "mmi", "damage_band", "confidence", "method"}
    assert required.issubset(result.keys())


def test_compute_forecast_median_is_positive():
    result = compute_forecast(5.0, 10.0, 50.0)
    assert result["pga_g_median"] > 0.0


def test_compute_forecast_ci_ordering():
    """Lower bound must be below median, upper must be above."""
    result = compute_forecast(6.0, 15.0, 30.0)
    assert result["pga_g_lower"] < result["pga_g_median"] < result["pga_g_upper"]


def test_compute_forecast_handles_none_inputs():
    """All-None inputs should not raise — fallback defaults used."""
    result = compute_forecast(None, None, None)
    assert result["pga_g_median"] is not None
    assert result["confidence"] == "low"


def test_higher_magnitude_gives_higher_pga():
    small = compute_forecast(4.0, 10.0, 50.0)
    large = compute_forecast(7.0, 10.0, 50.0)
    assert large["pga_g_median"] > small["pga_g_median"]


def test_farther_distance_gives_lower_pga():
    near = compute_forecast(5.5, 10.0, 10.0)
    far  = compute_forecast(5.5, 10.0, 200.0)
    assert far["pga_g_median"] < near["pga_g_median"]


# ---------------------------------------------------------------------------
# MMI conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pga_g,expected_mmi", [
    (0.001, "I"),
    (0.002, "II"),
    (0.015, "III"),
    (0.05,  "IV"),
    (0.10,  "V"),
    (0.20,  "VI"),
    (0.40,  "VII"),
    (0.70,  "VIII"),
    (1.50,  "IX+"),
])
def test_pga_to_mmi_thresholds(pga_g, expected_mmi):
    assert _pga_to_mmi(pga_g) == expected_mmi


# ---------------------------------------------------------------------------
# Damage band
# ---------------------------------------------------------------------------


def test_damage_from_pager_green():
    assert _damage_from_pager("green") == "None/Minimal"


def test_damage_from_pager_red():
    assert _damage_from_pager("red") == "Severe"


def test_damage_from_pager_none():
    assert _damage_from_pager(None) is None


def test_damage_from_mmi_moderate():
    assert _damage_from_mmi("VII") == "Moderate"


# ---------------------------------------------------------------------------
# Confidence level
# ---------------------------------------------------------------------------


def test_confidence_all_present():
    assert _confidence_level(5.0, 10.0, 50.0) == "high"


def test_confidence_partial():
    assert _confidence_level(5.0, None, None) == "medium"


def test_confidence_none():
    assert _confidence_level(None, None, None) == "low"


# ---------------------------------------------------------------------------
# PAGER integration into forecast
# ---------------------------------------------------------------------------


def test_pager_alert_included_in_output():
    result = compute_forecast(5.0, 10.0, 50.0, pager_alert="orange")
    # Damage band should be upgraded from PAGER rather than MMI
    assert result["damage_band"] == "Moderate-High"
