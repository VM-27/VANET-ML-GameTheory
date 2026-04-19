import xml.etree.ElementTree as ET
from collections import defaultdict

# Input SUMO Floating Car Data output
IN_XML = "sumo_traces/mobility.xml"
OUT_TCL = "sumo_traces/ns2mobility.tcl"

# SUMO coordinates are in meters; NS-3 uses meters too (good).
# We'll map SUMO vehicle IDs to node indices 0..N-1
def main():
    tree = ET.parse(IN_XML)
    root = tree.getroot()

    # Collect all vehicle IDs
    vehicle_ids = set()
    for ts in root.findall("timestep"):
        for v in ts.findall("vehicle"):
            vehicle_ids.add(v.get("id"))

    vehicle_ids = sorted(list(vehicle_ids))
    vid_to_node = {vid: i for i, vid in enumerate(vehicle_ids)}
    n = len(vehicle_ids)

    # For each node, store time->(x,y)
    positions = defaultdict(list)

    for ts in root.findall("timestep"):
        t = float(ts.get("time"))
        for v in ts.findall("vehicle"):
            vid = v.get("id")
            x = float(v.get("x"))
            y = float(v.get("y"))
            node = vid_to_node[vid]
            positions[node].append((t, x, y))

    # Write ns2 mobility trace
    with open(OUT_TCL, "w") as f:
        f.write(f"# NS2 mobility trace generated from SUMO FCD\n")
        f.write(f"# Nodes: {n}\n\n")

        # Initial positions at t=0 (or earliest)
        for node in range(n):
            if positions[node]:
                t0, x0, y0 = positions[node][0]
                f.write(f"$node_({node}) set X_ {x0}\n")
                f.write(f"$node_({node}) set Y_ {y0}\n")
                f.write(f"$node_({node}) set Z_ 0\n")

        f.write("\n# Movements\n")
        # Use "setdest" commands
        for node in range(n):
            seq = positions[node]
            for i in range(1, len(seq)):
                t, x, y = seq[i]
                # speed in ns2 trace can be approximated; we use 0 because ns-3 will still move
                # Better: compute speed from previous point
                t_prev, x_prev, y_prev = seq[i-1]
                dt = max(t - t_prev, 1e-6)
                dist = ((x - x_prev)**2 + (y - y_prev)**2) ** 0.5
                speed = dist / dt
                f.write(f"$ns_ at {t:.2f} \"$node_({node}) setdest {x:.2f} {y:.2f} {speed:.2f}\"\n")

    print(f"Generated: {OUT_TCL}")
    print(f"Total nodes: {n}")

if __name__ == "__main__":
    main()
