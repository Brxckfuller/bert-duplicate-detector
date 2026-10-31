
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIR = Path("models/distilbert-best")
THRESHOLD = 0.15
MAX_LENGTH = 128


class DuplicateQuestionDetector:
    def __init__(self, model_dir=MODEL_DIR, threshold=THRESHOLD):
        self.threshold = threshold

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def predict(self, question1, question2):
        if not isinstance(question1, str) or not question1.strip():
            raise ValueError("Question 1 must be a non-empty string.")

        if not isinstance(question2, str) or not question2.strip():
            raise ValueError("Question 2 must be a non-empty string.")

        inputs = self.tokenizer(
            question1,
            question2,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        probability = torch.softmax(
            outputs.logits, dim=1
        )[0, 1].item()

        return {
            "duplicate_probability": round(probability, 4),
            "is_duplicate": probability >= self.threshold,
            "threshold": self.threshold,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Detect semantically duplicate questions."
    )

    parser.add_argument("--q1", required=True, help="First question")
    parser.add_argument("--q2", required=True, help="Second question")

    args = parser.parse_args()

    detector = DuplicateQuestionDetector()
    result = detector.predict(args.q1, args.q2)

    print("\nDUPLICATE QUESTION DETECTOR")
    print("---------------------------")
    print(f"Question 1: {args.q1}")
    print(f"Question 2: {args.q2}")
    print(f"Duplicate score: {result['duplicate_probability']:.4f}")
    print(f"Threshold: {result['threshold']:.2f}")
    print(f"Prediction: {'Duplicate' if result['is_duplicate'] else 'Not duplicate'}")


if __name__ == "__main__":
    main()
