from __future__ import annotations
import datetime

def compare_events(current: dict, historical: dict) -> dict:
    """
    Produce factual comparison between current detected event
    and one historical catalog event.
    """
    result = {
        "same_station": current.get("station") == historical.get("station"),
        "same_network": current.get("network") == historical.get("network"),
    }
    
    cur_mag = current.get("magnitude_proxy")
    hist_mag = historical.get("catalog", {}).get("magnitude")
    if isinstance(cur_mag, (int, float)) and isinstance(hist_mag, (int, float)):
        result["magnitude_difference"] = round(abs(cur_mag - hist_mag), 2)
    
    cur_time = current.get("origin_time")
    hist_time = historical.get("catalog", {}).get("origin_time")
    if cur_time and hist_time:
        try:
            if isinstance(cur_time, str):
                cur_dt = datetime.datetime.fromisoformat(cur_time.replace("Z", "+00:00"))
            else:
                cur_dt = cur_time
                
            if isinstance(hist_time, str):
                hist_dt = datetime.datetime.fromisoformat(hist_time.replace("Z", "+00:00"))
            else:
                hist_dt = hist_time
                
            result["days_apart"] = abs((cur_dt - hist_dt).days)
        except Exception:
            pass
            
    return result
