
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from datasets import load_dataset

RANDOM_SEED = 42
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Load the original 5,000 question pairs
dataset = load_dataset("SetFit/qqp", split="train[:5000]")
df = dataset.to_pandas()
df = df[["text1", "text2", "label"]].dropna().copy()

# Normalise questions for overlap detection
df["q1_norm"] = df["text1"].str.strip().str.lower()
df["q2_norm"] = df["text2"].str.strip().str.lower()

df = df[
    df["q1_norm"].ne("")
    & df["q2_norm"].ne("")
].copy()

# Remove repeated pairs regardless of question order
df["pair_key"] = [
    tuple(sorted((a, b)))
    for a, b in zip(df["q1_norm"], df["q2_norm"])
]

df = df.drop_duplicates(subset="pair_key").copy()

# Build a graph of questions
graph = nx.Graph()
graph.add_edges_from(df["pair_key"])

# Every connected component must stay in one split
components = list(nx.connected_components(graph))

question_to_group = {
    question: group_id
    for group_id, component in enumerate(components)
    for question in component
}

df["group"] = df["q1_norm"].map(question_to_group)

# Assign entire components to partitions
rng = np.random.default_rng(RANDOM_SEED)
group_ids = np.array(list(df["group"].unique()))
rng.shuffle(group_ids)

target_sizes = {
    "train": 0.70 * len(df),
    "val": 0.15 * len(df),
    "test": 0.15 * len(df),
}

group_sizes = df.groupby("group").size().to_dict()

assignments = {"train": [], "val": [], "test": []}
current_sizes = {"train": 0, "val": 0, "test": 0}

for group_id in group_ids:
    # Assign to the split furthest below its target
    split = max(
        target_sizes,
        key=lambda s: target_sizes[s] - current_sizes[s]
    )

    assignments[split].append(group_id)
    current_sizes[split] += group_sizes[group_id]

# Save splits without helper columns
for name, groups in assignments.items():
    split_df = df[df["group"].isin(groups)]
    split_df = split_df[["text1", "text2", "label"]]

    split_df.to_csv(DATA_DIR / f"{name}.csv", index=False)

    print(
        f"{name}: {len(split_df)} pairs | "
        f"Duplicate rate: {split_df['label'].mean():.2%}"
    )

print("\nQuestion-disjoint splits saved.")

