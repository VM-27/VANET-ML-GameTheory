import xml.etree.ElementTree as ET
import math
import csv
from collections import defaultdict

IN_XML = "../data/mobility.xml"
OUT_CSV = "../data/vanet_dataset.csv"

# Radio range (meters) - tune later (this should match your ns-3 connectivity roughly)
RANGE_M = 150.0

# Prediction horizon (seconds) for stability label
HORIZON_S = 5.0

# Step size in SUMO output (usually 1s)
# We'll infer it from data.

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def main():
    tree = ET.parse(IN_XML)
    root = tree.getroot()

    # times -> {vid: (x,y,speed)}
    timeline = []
    times = []

    for ts in root.findall("timestep"):
        t = float(ts.get("time"))
        frame = {}
        for v in ts.findall("vehicle"):
            vid = v.get("id")
            x = float(v.get("x"))
            y = float(v.get("y"))
            sp = float(v.get("speed", "0"))
            frame[vid] = (x, y, sp)
        timeline.append(frame)
        times.append(t)

    if len(times) < 5:
        print("Not enough timesteps in mobility.xml")
        return

    # Infer dt
    dt = times[1] - times[0]
    if dt <= 0:
        dt = 1.0

    # Build index for quick horizon lookup
    time_to_index = {t: i for i, t in enumerate(times)}
    horizon_steps = int(round(HORIZON_S / dt))
    if horizon_steps < 1:
        horizon_steps = 1

    vids_all = sorted({vid for frame in timeline for vid in frame.keys()})
    print("Total timesteps:", len(times))
    print("Total vehicles:", len(vids_all))
    print("dt:", dt, "horizon_steps:", horizon_steps)

    rows = []
    for i, t in enumerate(times):
        frame = timeline[i]
        # skip frames where horizon not available
        j = i + horizon_steps
        if j >= len(times):
            break
        future = timeline[j]

        # Precompute neighbor stats for each node at time i
        vids = list(frame.keys())
        positions = {vid: (frame[vid][0], frame[vid][1]) for vid in vids}
        speeds = {vid: frame[vid][2] for vid in vids}

        # Build neighbor list within range
        neighbors = defaultdict(list)
        for a_idx in range(len(vids)):
            va = vids[a_idx]
            for b_idx in range(a_idx + 1, len(vids)):
                vb = vids[b_idx]
                d = dist(positions[va], positions[vb])
                if d <= RANGE_M:
                    neighbors[va].append((vb, d))
                    neighbors[vb].append((va, d))

        # Label: stable = 1 if node keeps at least 50% of its neighbors after horizon
        # (practical label; later we can refine)
        for vid in vids:
            deg = len(neighbors[vid])
            if deg == 0:
                continue  # skip isolated nodes, not useful for CH prediction

            avg_ndist = sum(d for _, d in neighbors[vid]) / deg
            neigh_ids = [nid for nid, _ in neighbors[vid]]

            # Future neighbors based on future positions
            if vid not in future:
                continue
            vpos_future = (future[vid][0], future[vid][1])

            kept = 0
            total = 0
            rel_speed_sum = 0.0

            for nid in neigh_ids:
                if nid not in future:
                    continue
                npos_future = (future[nid][0], future[nid][1])
                total += 1
                if dist(vpos_future, npos_future) <= RANGE_M:
                    kept += 1

                # relative speed (current)
                rel_speed_sum += abs(speeds[vid] - speeds.get(nid, 0.0))

            if total == 0:
                continue

            keep_ratio = kept / total
            rel_speed_avg = rel_speed_sum / total

            stable = 1 if keep_ratio >= 0.5 else 0

            rows.append({
                "time": t,
                "vid": vid,
                "speed": speeds[vid],
                "degree": deg,
                "avg_neighbor_dist": avg_ndist,
                "rel_speed_avg": rel_speed_avg,
                "keep_ratio": keep_ratio,
                "stable_label": stable
            })

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = ["time", "vid", "speed", "degree", "avg_neighbor_dist", "rel_speed_avg", "keep_ratio", "stable_label"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Saved dataset:", OUT_CSV)
    print("Rows:", len(rows))

if __name__ == "__main__":
    main()
