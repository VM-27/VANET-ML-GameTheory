import pandas as pd
import joblib
import numpy as np

DATA = "../data/vanet_dataset.csv"
MODEL = "../models/stability_rf.pkl"

# Utility weights (tune later)
W_STABILITY = 0.5
W_DEGREE = 0.3
W_SPEED = 0.2

def main():
    df = pd.read_csv(DATA)
    model = joblib.load(MODEL)

    features = ["speed", "degree", "avg_neighbor_dist", "rel_speed_avg", "keep_ratio"]

    # Predict probability
    df["p_stable"] = model.predict_proba(df[features])[:, 1]

    # Normalize degree and speed
    df["degree_norm"] = df["degree"] / df["degree"].max()
    df["speed_norm"] = df["speed"] / df["speed"].max()

    # Utility
    df["utility"] = (
        W_STABILITY * df["p_stable"] +
        W_DEGREE * df["degree_norm"] -
        W_SPEED * df["speed_norm"]
    )

    # Select CH per time (highest utility at each time)
    ch_df = df.loc[df.groupby("time")["utility"].idxmax()]

    print("=== Sample Cluster Heads ===")
    print(ch_df[["time", "vid", "utility", "p_stable"]].head())

    print("\nTotal time steps with CH:", ch_df["time"].nunique())

if __name__ == "__main__":
    main()
