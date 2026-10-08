import os
import json
import numpy as np
from pathlib import Path
from obspy import Stream, Trace
from obspy.core import UTCDateTime

OUT_DIR = Path("data/historical_waveforms")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for i in range(3):
    uid = f"fake_{i}"
    
    # 1. Create a dummy json
    meta = {
        "usgs_id": uid,
        "magnitude": 5.0 + i,
        "place": f"Fake Place {i}",
        "lat": 34.0 + i,
        "lon": -118.0 - i,
        "depth_km": 10.0,
        "origin_time": "2023-01-01T00:00:00Z",
        "felt": 100 * i,
        "pager_alert": "green",
        "network": "CI",
        "station": "PAS",
    }
    with open(OUT_DIR / f"{uid}.json", "w") as f:
        json.dump(meta, f)
        
    # 2. Create a dummy mseed
    tr1 = Trace(data=np.random.randn(6000).astype(np.float32))
    tr1.stats.sampling_rate = 100.0
    tr2 = tr1.copy()
    tr3 = tr1.copy()
    
    st = Stream([tr1, tr2, tr3])
    st.write(str(OUT_DIR / f"{uid}.mseed"), format="MSEED")

print("Created fake historical data.")
