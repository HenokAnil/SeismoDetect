#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
SAMPLE="${1:-../sample_data/clean_event_BK_PKD.mseed}"
# Note: Path to sample is relative to scripts dir, but usually run from repo root.
# Let's handle path assuming we run it from repo root.
if [ ! -f "$SAMPLE" ]; then
    SAMPLE="sample_data/clean_event_BK_PKD.mseed"
fi

echo "Posting $SAMPLE to $API_URL/upload"
resp_upload=$(curl -s -X POST "$API_URL/upload" -F "file=@$SAMPLE")
echo "Response: $resp_upload"

# Extract signal_id
signal_id=$(echo "$resp_upload" | jq -r '.signal_id')

if [ "$signal_id" == "null" ] || [ -z "$signal_id" ]; then
    echo "Upload failed, no signal_id returned."
    exit 1
fi

echo "Signal ID: $signal_id"
echo "Posting to $API_URL/infer/$signal_id"
resp_infer=$(curl -s -X POST "$API_URL/infer/$signal_id")
echo "Response: $resp_infer"

# Basic JSON key checks using jq
echo "$resp_infer" | jq '.result.quality_score, .result.event_id, .result.origin_time, .result.magnitude_proxy, .result.similar_events' > /dev/null

echo "Smoke test passed (basic keys present)."
