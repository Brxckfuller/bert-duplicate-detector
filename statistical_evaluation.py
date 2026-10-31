
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
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

    columns = ["text1", "text2", "label"]

    if len(baseline) != len(bert):
        raise ValueError(
            "Prediction files have different row counts."
        )

    if not baseline[columns].equals(bert[columns]):
        raise ValueError(
            "Question pairs or labels do not match."
        )

    print(f"Verified alignment of {len(bert)} test examples.")

    return baseline, bert


def assign_component_ids(df):
    graph = nx.Graph()

    graph.add_edges_from(
        zip(df["text1"], df["text2"])
    )

    node_to_component = {}

    for component_id, nodes in enumerate(
        nx.connected_components(graph)
    ):
        for node in nodes:
            node_to_component[node] = component_id

    component_ids = np.array([
        node_to_component[q1]
        for q1 in df["text1"]
    ])

    for q1, q2, component_id in zip(
        df["text1"],
        df["text2"],
        component_ids,
    ):
        if node_to_component[q2] != component_id:
            raise ValueError(
                "Question pair crosses connected components."
            )

    return component_ids


def confidence_interval(values):
    alpha = 1 - CONFIDENCE_LEVEL

    return [
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    ]


def bootstrap_indices(component_ids, rng):
    unique_components = np.unique(component_ids)

    component_rows = [
        np.flatnonzero(component_ids == component)
        for component in unique_components
    ]

    n_components = len(component_rows)

    for _ in range(N_BOOTSTRAPS):
        sampled_components = rng.integers(
            0, n_components, size=n_components
        )

        yield np.concatenate([
            component_rows[index]
            for index in sampled_components
        ])


def calculate_metrics(y_true, predictions, scores):
    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, scores)
        ),
    }


def mcnemar_test(y_true, predictions_a, predictions_b):
    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true

    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))

    discordant = a_only + b_only

    p_value = (
        float(
            binomtest(
                min(a_only, b_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )

    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "p_value": p_value,
        "note": (
            "Conventional pair-level McNemar test; "
            "does not adjust for component clustering."
        ),
    }


def main():
    baseline, bert = load_predictions()

    y_true = bert["label"].to_numpy(dtype=int)
    component_ids = assign_component_ids(bert)

    unique_components, component_sizes = np.unique(
        component_ids,
        return_counts=True,
    )

    print(
        f"Connected components: {len(unique_components)}"
    )
    print(
        f"Largest component: {component_sizes.max()} pairs"
    )
    print(
        "Pairs in multi-pair components:",
        int(component_sizes[component_sizes > 1].sum()),
    )

    models = {
        "tfidf_cosine": {
            "predictions": baseline[
                "tfidf_cosine_prediction"
            ].to_numpy(dtype=int),
            "scores": baseline[
                "tfidf_cosine_score"
            ].to_numpy(dtype=float),
        },
        "tfidf_enhanced": {
            "predictions": baseline[
                "tfidf_enhanced_prediction"
            ].to_numpy(dtype=int),
            "scores": baseline[
                "tfidf_enhanced_score"
            ].to_numpy(dtype=float),
        },
        "distilbert": {
            "predictions": bert[
                "prediction"
            ].to_numpy(dtype=int),
            "scores": bert[
                "probability"
            ].to_numpy(dtype=float),
        },
    }

    metrics_to_bootstrap = [
        "f1",
        "average_precision",
        "roc_auc",
    ]

    results = {
        "bootstrap_method": "paired_component_bootstrap",
        "bootstrap_resamples": N_BOOTSTRAPS,
        "seed": SEED,
        "test_pairs": len(y_true),
        "test_components": len(unique_components),
        "component_summary": {
            "largest_component_pairs": int(
                component_sizes.max()
            ),
            "multi_pair_components": int(
                np.sum(component_sizes > 1)
            ),
            "pairs_in_multi_pair_components": int(
                component_sizes[component_sizes > 1].sum()
            ),
        },
        "models": {},
        "comparisons": {},
    }

    for name, data in models.items():
        results["models"][name] = calculate_metrics(
            y_true,
            data["predictions"],
            data["scores"],
        )

    bootstrap_results = {
        name: {
            metric: []
            for metric in metrics_to_bootstrap
        }
        for name in models
    }

    rng = np.random.default_rng(SEED)

    for indices in bootstrap_indices(component_ids, rng):
        sampled_y = y_true[indices]

        if len(np.unique(sampled_y)) < 2:
            continue

        for name, data in models.items():
            sampled_predictions = data[
                "predictions"
            ][indices]
            sampled_scores = data["scores"][indices]

            sampled_metrics = calculate_metrics(
                sampled_y,
                sampled_predictions,
                sampled_scores,
            )

            for metric in metrics_to_bootstrap:
                bootstrap_results[name][metric].append(
                    sampled_metrics[metric]
                )

    for name in models:
        for metric in metrics_to_bootstrap:
            samples = bootstrap_results[name][metric]

            results["models"][name][
                f"{metric}_ci_95"
            ] = confidence_interval(samples)

    comparisons = [
        ("distilbert", "tfidf_enhanced"),
        ("distilbert", "tfidf_cosine"),
    ]

    for model_a, model_b in comparisons:
        comparison_name = f"{model_a}_vs_{model_b}"

        comparison = {}

        for metric in metrics_to_bootstrap:
            observed_difference = (
                results["models"][model_a][metric]
                - results["models"][model_b][metric]
            )

            bootstrap_difference = (
                np.asarray(
                    bootstrap_results[model_a][metric]
                )
                - np.asarray(
                    bootstrap_results[model_b][metric]
                )
            )

            comparison[metric] = {
                "difference": float(observed_difference),
                "ci_95": confidence_interval(
                    bootstrap_difference
                ),
            }

        comparison["mcnemar"] = mcnemar_test(
            y_true,
            models[model_a]["predictions"],
            models[model_b]["predictions"],
        )

        results["comparisons"][
            comparison_name
        ] = comparison

    # Save text-free prediction records
    prediction_output = pd.DataFrame({
        "pair_index": np.arange(len(y_true)),
        "component_id": component_ids,
        "label": y_true,
        "tfidf_cosine_score": models[
            "tfidf_cosine"
        ]["scores"],
        "tfidf_cosine_prediction": models[
            "tfidf_cosine"
        ]["predictions"],
        "tfidf_enhanced_score": models[
            "tfidf_enhanced"
        ]["scores"],
        "tfidf_enhanced_prediction": models[
            "tfidf_enhanced"
        ]["predictions"],
        "distilbert_score": models[
            "distilbert"
        ]["scores"],
        "distilbert_prediction": models[
            "distilbert"
        ]["predictions"],
    })

    prediction_path = (
        RESULTS_DIR / "evaluation_predictions.csv"
    )
    prediction_output.to_csv(
        prediction_path,
        index=False,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Precision-recall curves
    plt.figure(figsize=(8, 6))

    for name, data in models.items():
        precision, recall, _ = precision_recall_curve(
            y_true,
            data["scores"],
        )

        ap = results["models"][name][
            "average_precision"
        ]

        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={ap:.3f})",
        )

    plt.axhline(
        y=y_true.mean(),
        linestyle="--",
        color="gray",
        label=f"Positive prevalence ({y_true.mean():.3f})",
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
            y_true,
            data["scores"],
        )

        auc = results["models"][name]["roc_auc"]

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )

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

    output_path = (
        RESULTS_DIR / "statistical_evaluation.json"
    )

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)

    print("\nCOMPONENT-AWARE STATISTICAL EVALUATION")
    print("=" * 50)

    for name, metrics in results["models"].items():
        print(f"\n{name.upper()}")

        for metric in metrics_to_bootstrap:
            print(
                f"{metric}: {metrics[metric]:.4f} "
                f"95% CI: {metrics[f'{metric}_ci_95']}"
            )

    for name, comparison in results[
        "comparisons"
    ].items():
        print(f"\n{name.upper()}")
        print("-" * 50)

        for metric in metrics_to_bootstrap:
            values = comparison[metric]

            print(
                f"{metric} difference: "
                f"{values['difference']:.4f}"
            )
            print(
                f"95% CI: {values['ci_95']}"
            )

        print(
            "McNemar p-value:",
            comparison["mcnemar"]["p_value"],
        )

    print(f"\nSaved results to {output_path}")
    print(f"Saved predictions to {prediction_path}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
