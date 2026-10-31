
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_bert import QuestionPairDataset


MODEL_DIR = Path("models/distilbert-best")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
THRESHOLD = 0.15

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Device: {device}")

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)

model.eval()

# Load the held-out test dataset
test_df = pd.read_csv("data/test.csv")

dataset = QuestionPairDataset(test_df, tokenizer)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

probabilities = []

# Generate duplicate probabilities
with torch.no_grad():
    for batch in loader:
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        outputs = model(**inputs)

        batch_probs = torch.softmax(
            outputs.logits, dim=1
        )[:, 1]

        probabilities.extend(
            batch_probs.cpu().tolist()
        )

# Add predictions to the dataset
test_df["probability"] = np.array(probabilities)

test_df["prediction"] = (
    test_df["probability"] >= THRESHOLD
).astype(int)

# Categorise predictions
conditions = [
    (test_df["label"] == 0) & (test_df["prediction"] == 0),
    (test_df["label"] == 0) & (test_df["prediction"] == 1),
    (test_df["label"] == 1) & (test_df["prediction"] == 0),
    (test_df["label"] == 1) & (test_df["prediction"] == 1),
]

categories = [
    "True Negative",
    "False Positive",
    "False Negative",
    "True Positive",
]

test_df["error_type"] = np.select(
    conditions,
    categories,
    default="Unknown",
)

# Save all predictions
test_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)

print("\nPREDICTION BREAKDOWN")
print("---------------------------")
print(test_df["error_type"].value_counts())

# Examine confident false positives
false_positives = test_df[
    test_df["error_type"] == "False Positive"
].sort_values("probability", ascending=False)

print("\nMOST CONFIDENT FALSE POSITIVES")
print("---------------------------")

for _, row in false_positives.head(10).iterrows():
    print(f"\nQuestion 1: {row['text1']}")
    print(f"Question 2: {row['text2']}")
    print(f"Duplicate probability: {row['probability']:.4f}")

# Examine confident false negatives
false_negatives = test_df[
    test_df["error_type"] == "False Negative"
].sort_values("probability", ascending=True)

print("\nMOST CONFIDENT FALSE NEGATIVES")
print("---------------------------")

for _, row in false_negatives.head(10).iterrows():
    print(f"\nQuestion 1: {row['text1']}")
    print(f"Question 2: {row['text2']}")
    print(f"Duplicate probability: {row['probability']:.4f}")

# Save error subsets
false_positives.to_csv(
    OUTPUT_DIR / "false_positives.csv",
    index=False,
)

false_negatives.to_csv(
    OUTPUT_DIR / "false_negatives.csv",
    index=False,
)

print("\nError analysis files saved to results/")
