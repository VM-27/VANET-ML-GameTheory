import xml.etree.ElementTree as ET
import pandas as pd
import joblib
import math
from collections import defaultdict

MOB_XML = "../data/mobility.xml"
DATASET = "../data/vanet_dataset.csv"
MODEL = "../models/stability_rf.pkl"

K = 5              # number of cluster heads per timestep
RANGE_M = 250.0    # cluster association range (meters)

W_STABILITY = 0.5
W_DEGREE = 0.3
W_SPEED = 0.2

def dist(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def numeric_part(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())

def load_positions():
    """
    positions[t] is a dict of id->(x,y)
    We store both:
      - original SUMO id (e.g. 'veh8' or '0')
      - numeric-only id if present (e.g. '8')
    """
    tree = ET.parse(MOB_XML)
    root = tree.getroot()
    positions = {}

    for ts in root.findall("timestep"):
        t = int(round(float(ts.get("time"))))
        frame = {}

        for v in ts.findall("vehicle"):
            vid = str(v.get("id"))
            x = float(v.get("x"))
            y = float(v.get("y"))

            frame[vid] = (x, y)

            np = numeric_part(vid)
            if np != "":
                frame.setdefault(np, (x, y))

        positions[t] = frame

    return positions

def main():
    df = pd.read_csv(DATASET)
    model = joblib.load(MODEL)

    df["time_int"] = df["time"].round().astype(int)
    df["vid_str"] = df["vid"].astype(str)

    features = ["speed", "degree", "avg_neighbor_dist", "rel_speed_avg", "keep_ratio"]

    df["p_stable"] = model.predict_proba(df[features])[:, 1]
    df["degree_norm"] = df["degree"] / df["degree"].max()
    df["speed_norm"] = df["speed"] / df["speed"].max()

    df["utility"] = (
        W_STABILITY * df["p_stable"] +
        W_DEGREE * df["degree_norm"] -
        W_SPEED * df["speed_norm"]
    )

    # Select top-K CH per time
    ch_candidates = (
        df.sort_values(["time_int", "utility"], ascending=[True, False])
          .groupby("time_int")
          .head(K)
    )

    ch_map = defaultdict(list)
    for _, row in ch_candidates.iterrows():
        ch_map[int(row["time_int"])].append(str(row["vid_str"]))

    positions = load_positions()

    assignments = []
    times_used = 0
    times_skipped_no_frame = 0
    times_skipped_no_ch = 0

    for t, ch_list in ch_map.items():
        frame = positions.get(t)
        if frame is None:
            times_skipped_no_frame += 1
            continue

        # CH ids can be '8' while SUMO ids could be 'veh8'; numeric mapping handles that
        valid_chs = [ch for ch in ch_list if ch in frame]
        if not valid_chs:
            times_skipped_no_ch += 1
            continue

        times_used += 1

        # To avoid duplicates (if both 'veh8' and '8' exist), keep only numeric keys when possible
        for vid, (x, y) in frame.items():
            if any(c.isalpha() for c in vid):
                np = numeric_part(vid)
                if np and np in frame:
                    continue

            best_ch = "NONE"
            best_d = float("inf")

            for ch in valid_chs:
                chx, chy = frame[ch]
                d = dist(x, y, chx, chy)
                if d < best_d:
                    best_d = d
                    best_ch = ch

            if best_d <= RANGE_M:
                assignments.append((t, vid, best_ch, best_d))
            else:
                assignments.append((t, vid, "NONE", best_d))

    out = pd.DataFrame(assignments, columns=["time", "vid", "cluster_head", "distance_to_ch"])
    out.to_csv("../data/cluster_assignments.csv", index=False)

    print("Saved: ../data/cluster_assignments.csv")
    print("Total rows:", len(out))
    clustered_ratio = (out["cluster_head"] != "NONE").mean() if len(out) > 0 else 0.0
    print("Clustered %:", round(clustered_ratio * 100, 2))
    print("Times used:", times_used)
    print("Times skipped (no frame):", times_skipped_no_frame)
    print("Times skipped (no CH match):", times_skipped_no_ch)

    if len(out) > 0:
        print(out[out["cluster_head"] != "NONE"].head())

if __name__ == "__main__":
    main()
