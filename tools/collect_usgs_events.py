"""
tools/collect_usgs_events.py
------------------------------
Download a seed dataset of historical USGS earthquake events and save
raw GeoJSON files to data/usgs_raw/.

Usage:
    python tools/collect_usgs_events.py \
        --starttime 2015-01-01 --endtime 2023-12-31 \
        --minmagnitude 4.5 --maxmagnitude 8.5 \
        --limit 500

Outputs:
    data/usgs_raw/<usgs_id>.json   — one file per event
    data/usgs_raw/index.csv        — index of all downloaded events
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

USGS_FDSNWS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_EVENTPAGE = "https://earthquake.usgs.gov/earthquakes/eventpage/{id}/geojson"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "usgs_raw")


def fetch_event_list(
    starttime: str,
    endtime: str,
    minmagnitude: float,
    maxmagnitude: float,
    limit: int,
) -> list:
    """
    Query USGS FDSN event feed and return a list of event summary dicts.
    """
    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmagnitude,
        "maxmagnitude": maxmagnitude,
        "limit": limit,
        "orderby": "magnitude",
    }
    logger.info("Querying USGS FDSN: %s", params)
    resp = requests.get(USGS_FDSNWS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    logger.info("Retrieved %d events from USGS feed", len(features))
    return features


def fetch_event_detail(usgs_id: str) -> dict:
    """Fetch full GeoJSON detail for a single event."""
    url = USGS_EVENTPAGE.format(id=usgs_id)
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    logger.warning("Skipping %s — HTTP %d", usgs_id, resp.status_code)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Collect USGS earthquake events")
    parser.add_argument("--starttime",    default="2015-01-01")
    parser.add_argument("--endtime",      default="2023-12-31")
    parser.add_argument("--minmagnitude", type=float, default=4.5)
    parser.add_argument("--maxmagnitude", type=float, default=8.5)
    parser.add_argument("--limit",        type=int,   default=200)
    parser.add_argument("--delay",        type=float, default=0.5,
                        help="Seconds between API calls (be polite to USGS)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events = fetch_event_list(
        args.starttime, args.endtime,
        args.minmagnitude, args.maxmagnitude,
        args.limit,
    )

    index_rows = []
    for feat in events:
        usgs_id = feat.get("id")
        if not usgs_id:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{usgs_id}.json")
        if os.path.exists(out_path):
            logger.debug("Already downloaded %s", usgs_id)
        else:
            detail = fetch_event_detail(usgs_id)
            if detail:
                with open(out_path, "w") as fh:
                    json.dump(detail, fh, indent=2)
                logger.info("Saved %s", out_path)
            time.sleep(args.delay)

        props = feat.get("properties", {})
        coords = (feat.get("geometry") or {}).get("coordinates", [None, None, None])
        index_rows.append({
            "usgs_id":   usgs_id,
            "magnitude": props.get("mag"),
            "place":     props.get("place"),
            "depth_km":  coords[2],
            "lat":       coords[1],
            "lon":       coords[0],
            "time":      props.get("time"),
            "felt":      props.get("felt"),
            "alert":     props.get("alert"),
        })

    index_path = os.path.join(OUTPUT_DIR, "index.csv")
    with open(index_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)

    logger.info("Done. %d events saved. Index at %s", len(index_rows), index_path)


if __name__ == "__main__":
    main()
