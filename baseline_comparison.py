
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import paired_cosine_distances


SEED = 42
THRESHOLDS = np.arange(0.10, 0.91, 0.05)


def evaluate_predictions(y_true, predictions, scores=None):
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }

    if scores is not None:
        metrics["average_precision"] = average_precision_score(
            y_true, scores
        )
        metrics["roc_auc"] = roc_auc_score(y_true, scores)

    return metrics


def select_threshold(y_true, scores):
    return float(
        max(
            THRESHOLDS,
            key=lambda threshold: f1_score(
                y_true,
                (scores >= threshold).astype(int),
                zero_division=0,
            ),
        )
    )


def lexical_features(df):
    features = []

    for q1, q2 in zip(df["text1"], df["text2"]):
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        jaccard = intersection / union if union else 0.0
        containment = (
            intersection / min(len(words1), len(words2))
            if words1 and words2 else 0.0
        )

        features.append([jaccard, containment])

    return csr_matrix(np.asarray(features, dtype=np.float64))


def main():
    train = pd.read_csv("data/train.csv")
    val = pd.read_csv("data/val.csv")
    test = pd.read_csv("data/test.csv")

    y_train = train["label"].to_numpy(dtype=int)
    y_val = val["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
    )

    vectorizer.fit(
        train["text1"].tolist() + train["text2"].tolist()
    )

    def transform(df):
        q1 = vectorizer.transform(df["text1"])
        q2 = vectorizer.transform(df["text2"])

        cosine = (
            1 - paired_cosine_distances(q1, q2)
        ).reshape(-1, 1)

        simple = cosine

        enhanced = hstack(
            [
                csr_matrix(cosine),
                abs(q1 - q2),
                q1.multiply(q2),
                lexical_features(df),
            ],
            format="csr",
        )

        return simple, enhanced

    simple_train, enhanced_train = transform(train)
    simple_val, enhanced_val = transform(val)
    simple_test, enhanced_test = transform(test)

    results = {}
    prediction_output = {}

    # Majority-class baseline
    majority_label = int(np.bincount(y_train).argmax())
    majority_predictions = np.full_like(
        y_test, majority_label
    )

    results["majority_class"] = evaluate_predictions(
        y_test, majority_predictions
    )

    # All-positive baseline
    positive_predictions = np.ones_like(y_test)

    results["all_positive"] = evaluate_predictions(
        y_test, positive_predictions
    )

    # Logistic regression baselines
    configurations = {
        "tfidf_cosine": (
            simple_train, simple_val, simple_test
        ),
        "tfidf_enhanced": (
            enhanced_train, enhanced_val, enhanced_test
        ),
    }

    for name, (X_train, X_val, X_test) in configurations.items():
        model = LogisticRegression(
            random_state=SEED,
            max_iter=2000,
        )

        model.fit(X_train, y_train)

        val_scores = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_scores)

        test_scores = model.predict_proba(X_test)[:, 1]
        predictions = (test_scores >= threshold).astype(int)

        metrics = evaluate_predictions(
            y_test, predictions, test_scores
        )

        metrics["threshold"] = threshold
        results[name] = metrics

        prediction_output[name] = {
            "scores": test_scores.tolist(),
            "predictions": predictions.tolist(),
        }

    # Save aggregate evaluation metrics
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "baseline_comparison.json"

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)

    # Save row-level predictions for paired statistical analysis
    prediction_df = pd.DataFrame({
        "text1": test["text1"],
        "text2": test["text2"],
        "label": y_test,
        "tfidf_cosine_score": prediction_output["tfidf_cosine"]["scores"],
        "tfidf_cosine_prediction": prediction_output["tfidf_cosine"]["predictions"],
        "tfidf_enhanced_score": prediction_output["tfidf_enhanced"]["scores"],
        "tfidf_enhanced_prediction": prediction_output["tfidf_enhanced"]["predictions"],
    })

    prediction_path = output_dir / "baseline_test_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    # Display results
    for name, metrics in results.items():
        print(f"\n{name.upper()}")
        print("-" * 35)

        for key, value in metrics.items():
            print(f"{key}: {value}")

    print(f"\nSaved results to {output_path}")
    print(f"Saved predictions to {prediction_path}")


if __name__ == "__main__":
    main()

