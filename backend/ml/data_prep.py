import pandas as pd

def extract_features(waveform, p_time, s_time):
    """
    Placeholder for extracting features from a waveform for magnitude estimation.
    Would typically extract:
    - pga (Peak Ground Acceleration)
    - sp_interval (S time - P time)
    - snr (Signal-to-Noise Ratio)
    - freq_band_energy
    """
    return {
        "pga": 0.0,
        "sp_interval": 0.0,
        "freq_band_energy": 0.0,
        "snr": 0.0,
        "ml": 0.0
    }

def prepare_dataset(events, out_csv):
    """
    Prepares a dataset of features for magnitude training.
    """
    data = []
    for ev in events:
        data.append(extract_features(ev.waveform, ev.p_time, ev.s_time))
        
    df = pd.DataFrame(data)
    df.to_csv(out_csv, index=False)
    print(f"Saved dataset to {out_csv}")
