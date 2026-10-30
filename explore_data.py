
from datasets import load_dataset

# Load 5,000 question pairs from the Quora dataset
dataset = load_dataset(
    "SetFit/qqp",
    split="train[:5000]"
)

# Convert the dataset into a pandas DataFrame
df = dataset.to_pandas()

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five examples:")
print(df.head().to_string())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
