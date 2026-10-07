"""
backend/reporting/numeric_forecast.py
---------------------------------------
Rule-based numeric forecasting for seismic events.

Component A — GMPE (Boore-Atkinson 2008, simplified):
  Given magnitude, hypocentral distance, and depth, estimate:
    - Median PGA (g) and 95 % CI
    - Modified Mercalli Intensity (MMI) band
    - Damage band label

Component B — Optional learned XGBoost regressor:
  When `models/forecast_xgb.pkl` exists it is loaded and used
  instead of the rule-based GMPE.  The regressor must be trained
  externally with `tools/train_regressor.py`.

All public functions return JSON-serialisable dicts so the output
can be stored in Postgres JSONB columns without conversion.
"""

from __future__ import annotations

import math
import logging
import os
import pickle
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "forecast_xgb.pkl"
)

# PAGER alert colour → qualitative damage label mapping
_PAGER_TO_DAMAGE = {
    "green": "None/Minimal",
    "yellow": "Low-Moderate",
    "orange": "Moderate-High",
    "red": "Severe",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_forecast(
    magnitude: Optional[float],
    depth_km: Optional[float],
    r_km: Optional[float],
    pager_alert: Optional[str] = None,
) -> dict:
    """
    Compute a numeric forecast for a seismic event.

    Parameters
    ----------
    magnitude   : event magnitude (Mw / local)
    depth_km    : hypocentral depth in kilometres
    r_km        : epicentral distance to nearest station / target site (km)
    pager_alert : USGS PAGER alert colour from historical similar event (optional)

    Returns
    -------
    dict with keys:
        pga_g_median, pga_g_lower, pga_g_upper,
        mmi, damage_band, confidence, method
    """
    # Prefer learned regressor when available
    regressor = _load_regressor()
    if regressor is not None:
        return _learned_forecast(regressor, magnitude, depth_km, r_km, pager_alert)
    return _gmpe_forecast(magnitude, depth_km, r_km, pager_alert)


# ---------------------------------------------------------------------------
# GMPE (rule-based)
# ---------------------------------------------------------------------------


def _gmpe_forecast(
    magnitude: Optional[float],
    depth_km: Optional[float],
    r_km: Optional[float],
    pager_alert: Optional[str],
) -> dict:
    """
    Simplified Boore-Atkinson (2008) GMPE for peak ground acceleration.

    Formula (log-linear, Vs30 ≈ 760 m/s reference rock site):
        ln(PGA) = c0 + c1*M + c2*ln(R_hypo) + c3*ln(max(M, 3))
        σ_total ≈ 0.60 (ln units, period-independent simplified)

    References:
        Boore, D. M. & Atkinson, G. M. (2008). Ground-Motion Prediction
        Equations for the Average Horizontal Component of PGA, PGV, and
        5%-Damped PSA at Spectral Periods between 0.01 s and 10.0 s.
        Earthquake Spectra, 24(1), 99-138.
    """
    # Graceful degradation — fill in sensible defaults for missing inputs
    mag   = float(magnitude) if magnitude is not None else 4.0
    depth = float(depth_km)  if depth_km  is not None else 10.0
    r_epi = float(r_km)      if r_km      is not None else 50.0

    # Hypocentral distance
    r_hypo = math.sqrt(r_epi ** 2 + depth ** 2)
    r_hypo = max(r_hypo, 1.0)  # avoid log(0)

    # BA08 simplified coefficients (PGA, horizontal, reference rock)
    c0, c1, c2, c3 = -2.991, 1.414, -1.696, 0.503
    sigma = 0.60  # total sigma in ln units

    ln_pga_median = c0 + c1 * mag + c2 * math.log(r_hypo) + c3 * math.log(max(mag, 3.0))

    pga_median = math.exp(ln_pga_median)
    pga_lower  = math.exp(ln_pga_median - 1.645 * sigma)  # 5th percentile
    pga_upper  = math.exp(ln_pga_median + 1.645 * sigma)  # 95th percentile

    mmi        = _pga_to_mmi(pga_median)
    damage     = _damage_from_pager(pager_alert) or _damage_from_mmi(mmi)

    confidence = _confidence_level(magnitude, depth_km, r_km)

    return {
        "pga_g_median": round(pga_median, 4),
        "pga_g_lower":  round(pga_lower, 4),
        "pga_g_upper":  round(pga_upper, 4),
        "pga_95_ci":    [round(pga_lower, 4), round(pga_upper, 4)],
        "mmi":          mmi,
        "damage_band":  damage,
        "confidence":   confidence,
        "method":       "GMPE-BA08-simplified",
        "inputs":       {"mag": mag, "depth_km": depth, "r_km": r_epi},
    }


# ---------------------------------------------------------------------------
# Learned regressor
# ---------------------------------------------------------------------------


def _load_regressor():
    """Return the XGBoost regressor if the model file exists, else None."""
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        with open(_MODEL_PATH, "rb") as fh:
            model = pickle.load(fh)
        logger.info("Loaded XGBoost forecast regressor from %s", _MODEL_PATH)
        return model
    except Exception as exc:
        logger.warning("Failed to load forecast regressor: %s", exc)
        return None


def _learned_forecast(
    model,
    magnitude: Optional[float],
    depth_km: Optional[float],
    r_km: Optional[float],
    pager_alert: Optional[str],
) -> dict:
    """Use the fitted XGBoost model to predict PGA."""
    mag   = float(magnitude) if magnitude is not None else 4.0
    depth = float(depth_km)  if depth_km  is not None else 10.0
    r_epi = float(r_km)      if r_km      is not None else 50.0

    features = np.array([[mag, depth, r_epi]])
    try:
        pga_pred = float(model.predict(features)[0])
    except Exception as exc:
        logger.warning("Regressor prediction failed, falling back to GMPE: %s", exc)
        return _gmpe_forecast(magnitude, depth_km, r_km, pager_alert)

    # Derive CI heuristically (±30 % of prediction)
    pga_lower = pga_pred * 0.70
    pga_upper = pga_pred * 1.30
    mmi       = _pga_to_mmi(pga_pred)
    damage    = _damage_from_pager(pager_alert) or _damage_from_mmi(mmi)

    return {
        "pga_g_median": round(pga_pred, 4),
        "pga_g_lower":  round(pga_lower, 4),
        "pga_g_upper":  round(pga_upper, 4),
        "pga_95_ci":    [round(pga_lower, 4), round(pga_upper, 4)],
        "mmi":          mmi,
        "damage_band":  damage,
        "confidence":   "medium",
        "method":       "XGBoost-regressor",
        "inputs":       {"mag": mag, "depth_km": depth, "r_km": r_epi},
    }


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def _pga_to_mmi(pga_g: float) -> str:
    """
    Wald et al. (1999) PGA → MMI conversion.
    Thresholds are in units of g.
    """
    thresholds = [
        (1.24, "IX+"),
        (0.65, "VIII"),
        (0.34, "VII"),
        (0.18, "VI"),
        (0.092, "V"),
        (0.039, "IV"),
        (0.014, "III"),
        (0.0017, "II"),
    ]
    for thr, label in thresholds:
        if pga_g >= thr:
            return label
    return "I"


def _damage_from_pager(pager_alert: Optional[str]) -> Optional[str]:
    """Map USGS PAGER alert colour to a damage band label."""
    if not pager_alert:
        return None
    return _PAGER_TO_DAMAGE.get(pager_alert.lower())


def _damage_from_mmi(mmi: str) -> str:
    """Derive a coarse damage band from MMI string."""
    mapping = {
        "I": "None",
        "II": "None",
        "III": "None",
        "IV": "None/Minimal",
        "V": "Low",
        "VI": "Low-Moderate",
        "VII": "Moderate",
        "VIII": "Moderate-High",
        "IX+": "High-Severe",
    }
    return mapping.get(mmi, "Unknown")


def _confidence_level(
    magnitude: Optional[float],
    depth_km: Optional[float],
    r_km: Optional[float],
) -> str:
    """
    Heuristic confidence: 'high' when all inputs are present and plausible,
    'medium' when some are missing, 'low' when defaults are used for all.
    """
    present = sum(v is not None for v in [magnitude, depth_km, r_km])
    if present == 3:
        return "high"
    if present >= 1:
        return "medium"
    return "low"
