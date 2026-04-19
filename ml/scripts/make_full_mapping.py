import pandas as pd
from collections import Counter

N_NODES = 50
TOP_K_CH = 8

map_path = "../data/node_to_ch_map.csv"

df = pd.read_csv(map_path)

# Count which CH IDs appear most often in current mapping
ch_counts = Counter(df["cluster_head"].astype(int).tolist())
top_chs = [ch for ch, _ in ch_counts.most_common(TOP_K_CH)]

# Fallback CH = most common CH
fallback_ch = top_chs[0] if top_chs else int(df["cluster_head"].iloc[0])

# Build full mapping for nodes 0..N_NODES-1
current = dict(zip(df["vid"].astype(int), df["cluster_head"].astype(int)))

rows = []
for vid in range(N_NODES):
    ch = current.get(vid, fallback_ch)

    # compress CHs into TOP_K_CH set
    if ch not in top_chs:
        ch = fallback_ch

    rows.append((vid, ch))

out = pd.DataFrame(rows, columns=["vid", "cluster_head"])
out_path = "../data/node_to_ch_map_full.csv"
out.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Rows:", len(out))
print("Unique CHs:", out["cluster_head"].nunique())
print("CHs:", sorted(out["cluster_head"].unique().tolist()))
