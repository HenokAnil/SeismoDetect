import argparse
import json
import logging
import os
import time
from pathlib import Path
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATION_NET = "CI"
STATION_CODE = "PAS"
CHANNELS = "HH?"

OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "data", "historical_waveforms"))
USGS_FDSNWS = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def fetch_event_list(limit=20):
    params = {
        "format": "geojson",
        "starttime": "2010-01-01",
        "endtime": "2024-01-01",
        "minmagnitude": 3.0,
        "limit": limit,
        "orderby": "magnitude",
        "latitude": 34.14,
        "longitude": -118.17,
        "maxradiuskm": 200,
    }
    resp = requests.get(USGS_FDSNWS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("features", [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Client("SCEDC")

    events = fetch_event_list(args.limit)
    logger.info(f"Fetched {len(events)} events from USGS")

    for feat in events:
        usgs_id = feat.get("id")
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None, None])

        if not usgs_id or not props.get("time"):
            continue

        origin = UTCDateTime(props["time"] / 1000.0)
        start = origin - 10
        end = origin + 50

        mseed_path = OUTPUT_DIR / f"{usgs_id}.mseed"
        meta_path = OUTPUT_DIR / f"{usgs_id}.json"

        if mseed_path.exists() and meta_path.exists():
            continue

        try:
            st = client.get_waveforms(
                network=STATION_NET,
                station=STATION_CODE,
                location="*",
                channel=CHANNELS,
                starttime=start,
                endtime=end,
            )
            st.write(str(mseed_path), format="MSEED")
            
            with open(meta_path, "w") as f:
                json.dump({
                    "usgs_id": usgs_id,
                    "magnitude": props.get("mag"),
                    "place": props.get("place"),
                    "lat": coords[1],
                    "lon": coords[0],
                    "depth_km": coords[2],
                    "origin_time": origin.isoformat(),
                    "felt": props.get("felt"),
                    "pager_alert": props.get("alert"),
                    "network": STATION_NET,
                    "station": STATION_CODE,
                }, f, indent=2)
            logger.info(f"Saved {usgs_id}")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Skipping {usgs_id} - {e}")

if __name__ == "__main__":
    main()
