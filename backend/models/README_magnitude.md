# Magnitude Estimator (Proxy)
Current implementation: `magnitude_proxy` in `backend/api/main.py` and `backend/models/database.py`.

Formula: `ML ≈ 2.0 + det_prob * 1.5`

Notes:
- This is a rough proxy and should be calibrated against local station ground-truth (actual events).
- To replace this with an ML regressor based on features (like PGA, S-P interval, etc.), see `backend/ml/train_magnitude.py` which provides a skeleton for training a RandomForest model.
