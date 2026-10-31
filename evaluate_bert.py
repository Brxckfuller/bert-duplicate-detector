
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_bert import QuestionPairDataset


MODEL_DIR = Path("models/distilbert-best")
BATCH_SIZE = 8

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Evaluation device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)

model.eval()


def get_probabilities(dataframe):
    dataset = QuestionPairDataset(dataframe, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    probabilities = []

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
                batch_probs.cpu().numpy()
            )

    return np.array(probabilities)


# Select threshold using validation data
val_df = pd.read_csv("data/val.csv")
val_probabilities = get_probabilities(val_df)

thresholds = np.arange(0.10, 0.91, 0.05)

best_threshold = max(
    thresholds,
    key=lambda threshold: f1_score(
        val_df["label"],
        (val_probabilities >= threshold).astype(int),
        zero_division=0,
    ),
)

print(f"Selected threshold: {best_threshold:.2f}")

# Evaluate on the held-out test set
test_df = pd.read_csv("data/test.csv")
test_probabilities = get_probabilities(test_df)

predictions = (
    test_probabilities >= best_threshold
).astype(int)

labels = test_df["label"]

print("\nDISTILBERT TEST RESULTS")
print("----------------------------")
print("Accuracy:", round(accuracy_score(labels, predictions), 4))
print("Precision:", round(precision_score(labels, predictions), 4))
print("Recall:", round(recall_score(labels, predictions), 4))
print("F1 Score:", round(f1_score(labels, predictions), 4))

print("\nConfusion matrix:")
print(confusion_matrix(labels, predictions))
