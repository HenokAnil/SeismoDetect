# train_magnitude.py
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def load_features_labels(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['pga','sp_interval','freq_band_energy','snr']].values
    y = df['ml'].values
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--out", default="backend/models/magnitude_model.joblib")
    args = parser.parse_args()
    
    X, y = load_features_labels(args.data_csv)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200)
    model.fit(Xtr, ytr)
    
    print("Train score", model.score(Xtr, ytr))
    print("Test score", model.score(Xte, yte))
    
    joblib.dump(model, args.out)
    print("Saved model to", args.out)
