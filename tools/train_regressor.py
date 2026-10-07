"""
tools/train_regressor.py
--------------------------
Train an XGBoost regressor to predict PGA from event features,
replacing the rule-based GMPE when labeled ShakeMap data is available.

Usage:
    python tools/train_regressor.py \
        --data data/usgs_raw/index.csv \
        --output backend/models/forecast_xgb.pkl

Input CSV columns required:
    magnitude, depth_km, felt (used as proxy for r_km)
    pga_g — target column (observed PGA from ShakeMap; must be added manually)

This script is OPTIONAL. The pipeline uses GMPE by default.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost PGA regressor")
    parser.add_argument("--data",   required=True, help="Path to labeled CSV")
    parser.add_argument("--output", default="backend/models/forecast_xgb.pkl")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth",    type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    # --- Load and validate data ---
    df = pd.read_csv(args.data)
    required_cols = {"magnitude", "depth_km", "pga_g"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            "Add 'pga_g' column with observed PGA values from ShakeMap."
        )

    # Drop rows with any NaN in key columns
    df = df.dropna(subset=list(required_cols))
    logger.info("Training on %d labelled events", len(df))

    # Derive r_km proxy from felt reports (rough heuristic when no station distance)
    if "r_km" not in df.columns:
        df["r_km"] = np.where(
            df["felt"].notna() & (df["felt"] > 0),
            np.clip(200 / np.sqrt(df["felt"].fillna(1) + 1), 5, 300),
            50.0,
        )

    X = df[["magnitude", "depth_km", "r_km"]].values
    y = df["pga_g"].values

    # --- Train XGBoost ---
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost>=2.0")

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    mae = -scores.mean()
    logger.info("5-fold CV MAE: %.4f g (std ± %.4f)", mae, scores.std())

    # Final fit on all data
    model.fit(X, y)

    # --- Save model ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s", args.output)
    logger.info("Feature importances: %s", dict(zip(["magnitude", "depth_km", "r_km"], model.feature_importances_)))


if __name__ == "__main__":
    main()
