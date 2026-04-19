#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import glob

DATA_DIR = os.path.expanduser("~/VANET-ML-GameTheory/ml/data")
OUT_DIR = os.path.join(DATA_DIR, "dynamic_maps")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.2
MAX_RANGE = 250.0


def pick_score_column(df):
    for col in ["utility", "p_stable", "distance_to_ch"]:
        if col in df.columns:
            return col
    return None


def normalize(s):
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if abs(mx - mn) < 1e-12:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def select_chs(df):
    score_col = pick_score_column(df)

    if score_col is None:
        raise ValueError("No usable score column found: utility / p_stable / distance_to_ch")

    work = df.copy()

    if score_col == "distance_to_ch":
        vals = safe_numeric(work[score_col])
        work["score"] = 1.0 - normalize(vals.fillna(vals.max()))
    else:
        vals = safe_numeric(work[score_col])
        work["score"] = normalize(vals.fillna(0))

    # Safe self-CH bonus
    if "cluster_head" in work.columns:
        ch_numeric = safe_numeric(work["cluster_head"])
        vid_numeric = safe_numeric(work["vid"])
        work["is_self_ch"] = (vid_numeric == ch_numeric).fillna(False).astype(float)
        work["score"] = work["score"] + 0.3 * work["is_self_ch"]
    else:
        work["is_self_ch"] = 0.0

    # If x,y are not present, just use self-CH rows or top score rows
    if "x" not in work.columns or "y" not in work.columns:
        self_ch_rows = work[work["is_self_ch"] > 0].copy()
        if not self_ch_rows.empty:
            return pd.DataFrame({"vid": self_ch_rows["vid"].astype(int).unique()})
        top = work.sort_values("score", ascending=False).head(1)
        return pd.DataFrame({"vid": top["vid"].astype(int).tolist()})

    work = work.sort_values("score", ascending=False)

    selected = []

    for _, row in work.iterrows():
        vid = int(row["vid"])
        x = float(row["x"])
        y = float(row["y"])

        nearby = 0
        for ch in selected:
            d = np.hypot(x - ch["x"], y - ch["y"])
            if d < MAX_RANGE:
                nearby += 1

        payoff = float(row["score"]) - ALPHA * nearby

        if payoff > 0.3:
            selected.append({"vid": vid, "x": x, "y": y})

    if selected:
        return pd.DataFrame(selected)

    # fallback: self-CH nodes
    self_ch_rows = work[work["is_self_ch"] > 0].copy()
    if not self_ch_rows.empty:
        return pd.DataFrame({
            "vid": self_ch_rows["vid"].astype(int).tolist(),
            "x": self_ch_rows["x"].astype(float).tolist(),
            "y": self_ch_rows["y"].astype(float).tolist(),
        })

    # final fallback: top one
    top = work.iloc[0]
    return pd.DataFrame([{
        "vid": int(top["vid"]),
        "x": float(top["x"]),
        "y": float(top["y"])
    }])


def assign_nodes(nodes, chs):
    result = []

    ch_ids = set(pd.to_numeric(chs["vid"], errors="coerce").dropna().astype(int).tolist())

    has_xy_nodes = "x" in nodes.columns and "y" in nodes.columns
    has_xy_chs = "x" in chs.columns and "y" in chs.columns

    for _, n in nodes.iterrows():
        vid = int(n["vid"])

        if vid in ch_ids:
            result.append([vid, vid])
            continue

        # If x,y missing, use existing valid cluster_head if present
        if not (has_xy_nodes and has_xy_chs):
            if "cluster_head" in nodes.columns:
                ch_val = pd.to_numeric(pd.Series([n["cluster_head"]]), errors="coerce").iloc[0]
                if pd.notna(ch_val):
                    result.append([vid, int(ch_val)])
                    continue

            first_ch = int(pd.to_numeric(chs.iloc[0]["vid"], errors="coerce"))
            result.append([vid, first_ch])
            continue

        x = float(n["x"])
        y = float(n["y"])
        best_ch = None
        best_dist = 1e18

        for _, ch in chs.iterrows():
            ch_vid = pd.to_numeric(pd.Series([ch["vid"]]), errors="coerce").iloc[0]
            if pd.isna(ch_vid):
                continue

            # if CH x,y missing, skip
            if "x" not in ch or "y" not in ch:
                continue

            d = np.hypot(x - float(ch["x"]), y - float(ch["y"]))
            if d < best_dist:
                best_dist = d
                best_ch = int(ch_vid)

        if best_ch is None:
            # fallback to existing valid cluster_head
            if "cluster_head" in nodes.columns:
                ch_val = pd.to_numeric(pd.Series([n["cluster_head"]]), errors="coerce").iloc[0]
                if pd.notna(ch_val):
                    best_ch = int(ch_val)
                else:
                    best_ch = int(pd.to_numeric(chs.iloc[0]["vid"], errors="coerce"))
            else:
                best_ch = int(pd.to_numeric(chs.iloc[0]["vid"], errors="coerce"))

        result.append([vid, best_ch])

    return pd.DataFrame(result, columns=["vid", "cluster_head"])


def main():
    src = os.path.join(DATA_DIR, "cluster_assignments.csv")
    df = pd.read_csv(src)

    if "time" not in df.columns or "vid" not in df.columns:
        raise ValueError("cluster_assignments.csv must contain at least: time, vid")

    times = sorted(df["time"].dropna().unique())
    last_mapping = None

    for t in times:
        frame = df[df["time"] == t].copy()
        if frame.empty:
            continue

        chs = select_chs(frame)
        if chs.empty:
            continue

        mapping = assign_nodes(frame, chs)
        out_file = os.path.join(OUT_DIR, f"node_to_ch_map_t{int(t)}.csv")
        mapping.to_csv(out_file, index=False)
        last_mapping = mapping

    if last_mapping is not None:
        last_mapping.to_csv(os.path.join(OUT_DIR, "node_to_ch_map.csv"), index=False)

    files = sorted(glob.glob(os.path.join(OUT_DIR, "node_to_ch_map_t*.csv")))
    print("Dynamic map files created:", len(files))
    print("Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()
