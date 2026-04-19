import pandas as pd

DATA = "../data/cluster_assignments.csv"

df = pd.read_csv(DATA)

# Keep only clustered rows
df = df[df["cluster_head"] != "NONE"]

# For each node, find most frequent cluster head
summary = (
    df.groupby("vid")["cluster_head"]
      .agg(lambda x: x.value_counts().idxmax())
      .reset_index()
)

summary.to_csv("../data/node_to_ch_map.csv", index=False)

print("Saved: ../data/node_to_ch_map.csv")
print(summary.head())
