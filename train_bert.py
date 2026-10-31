
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Configuration
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3

OUTPUT_DIR = Path("models/distilbert-best")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAINING_CONFIG = {
    "seed": SEED,
    "base_model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "optimizer": "AdamW",
    "checkpoint_selection": "validation_f1_at_0.5_threshold",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class QuestionPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.questions1 = dataframe["text1"].tolist()
        self.questions2 = dataframe["text2"].tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.questions1[index],
            self.questions2[index],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            total_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training device: {device}")

    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = QuestionPairDataset(train_df, tokenizer)
    val_dataset = QuestionPairDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_f1 = -1.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch_number, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        train_loss = total_train_loss / len(train_loader)
        metrics = evaluate(model, val_loader, device)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
        }
        history.append(epoch_result)

        print(f"\nEpoch {epoch + 1} complete")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation precision: {metrics['precision']:.4f}")
        print(f"Validation recall: {metrics['recall']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}\n")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]

            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

            checkpoint_metadata = {
                **TRAINING_CONFIG,
                "best_epoch": epoch + 1,
                "best_validation_f1": float(best_f1),
            }

            with open(OUTPUT_DIR / "training_config.json", "w") as file:
                json.dump(checkpoint_metadata, file, indent=2)

            print(f"Saved new best model (F1={best_f1:.4f})")

    Path("results").mkdir(exist_ok=True)

    with open("results/training_history.json", "w") as file:
        json.dump(history, file, indent=2)

    print("\nTraining finished.")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best checkpoint: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
