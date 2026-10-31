
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Load the same 5,000 examples used previously
dataset = load_dataset("SetFit/qqp", split="train[:5000]")
df = dataset.to_pandas()

# Keep only the columns needed for modelling
df = df[["text1", "text2", "label"]].copy()

# Remove missing or empty questions
df = df.dropna(subset=["text1", "text2", "label"])

df = df[
    df["text1"].str.strip().ne("")
    & df["text2"].str.strip().ne("")
]

# Remove exact duplicate question pairs
# Treat reversed pairs as equivalent
df["pair_key"] = df.apply(
    lambda row: tuple(sorted([
        row["text1"].strip().lower(),
        row["text2"].strip().lower()
    ])),
    axis=1
)

df = df.drop_duplicates(subset="pair_key")
df = df.drop(columns="pair_key")

# Split into training (70%), validation (15%),
# and testing (15%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["label"]
)

# Save reproducible splits
train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("\nDATASET PREPARATION COMPLETE")
print("----------------------------")

for name, split in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(
        f"{name}: {len(split)} pairs | "
        f"Duplicate rate: {split['label'].mean():.2%}"
    )

print("\nFiles saved in:", DATA_DIR.resolve())
