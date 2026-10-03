import requests
import json
import sys

URL_UPLOAD = "http://localhost:8000/upload"
URL_INFER = "http://localhost:8000/infer"
SAMPLE = "sample_data/clean_event_BK_PKD.mseed"

print(f"Posting {SAMPLE} to {URL_UPLOAD}")
try:
    with open(SAMPLE, "rb") as f:
        files = {"file": f}
        resp = requests.post(URL_UPLOAD, files=files)
        resp.raise_for_status()
        data = resp.json()
        print(f"Upload Response: {data}")
        
        signal_id = data.get("signal_id")
        if not signal_id:
            print("Failed to get signal_id")
            sys.exit(1)
            
        print(f"Signal ID: {signal_id}")
        print(f"Posting to {URL_INFER}/{signal_id}")
        infer_resp = requests.post(f"{URL_INFER}/{signal_id}")
        infer_resp.raise_for_status()
        infer_data = infer_resp.json()
        print(f"Infer Response: {json.dumps(infer_data, indent=2)}")
        
        result = infer_data.get("result", {})
        keys = ["event_id", "origin_time", "magnitude_proxy", "similar_events"]
        missing = [k for k in keys if k not in result]
        if missing:
            print(f"Missing keys in result: {missing}")
            sys.exit(1)
            
        print("Smoke test passed (basic keys present).")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
