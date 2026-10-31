
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


SEED = 42
N_BOOTSTRAPS = 5000
CONFIDENCE_LEVEL = 0.95

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def load_predictions():
    baseline = pd.read_csv(
        RESULTS_DIR / "baseline_test_predictions.csv"
    )
    bert = pd.read_csv(
        RESULTS_DIR / "test_predictions.csv"
    )

    if len(baseline) != len(bert):
        raise ValueError("Prediction files have different row counts.")

    comparison_columns = ["text1", "text2", "label"]

    if not baseline[comparison_columns].equals(
        bert[comparison_columns]
    ):
        raise ValueError(
            "Question pairs or labels do not match between files."
        )

    print(f"Verified alignment of {len(bert)} test examples.")

    return baseline, bert


def bootstrap_f1(y_true, predictions, indices):
    return np.array([
        f1_score(
            y_true[sample],
            predictions[sample],
            zero_division=0,
        )
        for sample in indices
    ])


def confidence_interval(values):
    alpha = 1 - CONFIDENCE_LEVEL

    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    ]


def mcnemar_test(y_true, predictions_a, predictions_b):
    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))

    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            min(a_only, b_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "p_value": float(p_value),
    }


def main():
    baseline, bert = load_predictions()

    y_true = bert["label"].to_numpy(dtype=int)

    models = {
        "TF-IDF cosine": {
            "predictions": baseline[
                "tfidf_cosine_prediction"
            ].to_numpy(dtype=int),
            "scores": baseline[
                "tfidf_cosine_score"
            ].to_numpy(dtype=float),
        },
        "Enhanced TF-IDF": {
            "predictions": baseline[
                "tfidf_enhanced_prediction"
            ].to_numpy(dtype=int),
            "scores": baseline[
                "tfidf_enhanced_score"
            ].to_numpy(dtype=float),
        },
        "DistilBERT": {
            "predictions": bert[
                "prediction"
            ].to_numpy(dtype=int),
            "scores": bert[
                "probability"
            ].to_numpy(dtype=float),
        },
    }

    rng = np.random.default_rng(SEED)

    bootstrap_indices = rng.integers(
        0,
        len(y_true),
        size=(N_BOOTSTRAPS, len(y_true)),
    )

    results = {}
    bootstrap_samples = {}

    for name, data in models.items():
        predictions = data["predictions"]
        scores = data["scores"]

        samples = bootstrap_f1(
            y_true, predictions, bootstrap_indices
        )
        bootstrap_samples[name] = samples

        results[name] = {
            "f1": float(
                f1_score(y_true, predictions)
            ),
            "f1_ci_95": confidence_interval(samples),
            "average_precision": float(
                average_precision_score(y_true, scores)
            ),
            "roc_auc": float(
                roc_auc_score(y_true, scores)
            ),
        }

    difference = (
        bootstrap_samples["DistilBERT"]
        - bootstrap_samples["Enhanced TF-IDF"]
    )

    results["paired_comparison"] = {
        "f1_difference": (
            results["DistilBERT"]["f1"]
            - results["Enhanced TF-IDF"]["f1"]
        ),
        "f1_difference_ci_95": confidence_interval(
            difference
        ),
        "mcnemar": mcnemar_test(
            y_true,
            models["DistilBERT"]["predictions"],
            models["Enhanced TF-IDF"]["predictions"],
        ),
    }

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Precision-recall curves
    plt.figure(figsize=(8, 6))

    for name, data in models.items():
        precision, recall, _ = precision_recall_curve(
            y_true, data["scores"]
        )

        ap = results[name]["average_precision"]

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "precision_recall_curves.png",
        dpi=200,
    )
    plt.close()

    # ROC curves
    plt.figure(figsize=(8, 6))

    for name, data in models.items():
        fpr, tpr, _ = roc_curve(
            y_true, data["scores"]
        )

        auc = results[name]["roc_auc"]

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "roc_curves.png",
        dpi=200,
    )
    plt.close()

    output_path = RESULTS_DIR / "statistical_evaluation.json"

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)

    print("\nSTATISTICAL EVALUATION")
    print("=" * 45)

    for name in models:
        metrics = results[name]

        print(f"\n{name}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"95% CI: {metrics['f1_ci_95']}")
        print(
            f"Average precision: "
            f"{metrics['average_precision']:.4f}"
        )
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    comparison = results["paired_comparison"]

    print("\nDISTILBERT VS ENHANCED TF-IDF")
    print("-" * 45)
    print(
        f"F1 difference: "
        f"{comparison['f1_difference']:.4f}"
    )
    print(
        f"95% CI for difference: "
        f"{comparison['f1_difference_ci_95']}"
    )
    print(
        f"McNemar p-value: "
        f"{comparison['mcnemar']['p_value']:.6f}"
    )

    print(f"\nSaved results to {output_path}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
