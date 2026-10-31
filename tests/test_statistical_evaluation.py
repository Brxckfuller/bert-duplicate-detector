
import numpy as np
import pandas as pd
import pytest

from statistical_evaluation import (
    assign_component_ids,
    bootstrap_indices,
    calculate_metrics,
    confidence_interval,
    mcnemar_test,
)


def test_connected_pairs_share_component():
    df = pd.DataFrame({
        "text1": ["A", "B", "D", "F"],
        "text2": ["B", "C", "E", "G"],
    })

    ids = assign_component_ids(df)

    assert ids[0] == ids[1]
    assert len(set(ids)) == 3


def test_component_bootstrap_preserves_clusters():
    component_ids = np.array([0, 0, 1, 2, 2, 2])
    rng = np.random.default_rng(42)

    for indices in bootstrap_indices(component_ids, rng):
        counts = np.bincount(indices, minlength=6)

        assert counts[0] == counts[1]
        assert counts[3] == counts[4] == counts[5]


def test_confidence_interval():
    values = np.arange(101)

    lower, upper = confidence_interval(values)

    assert lower == pytest.approx(2.5)
    assert upper == pytest.approx(97.5)


def test_metrics_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_metrics(
        labels,
        predictions,
        scores,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_mcnemar_identical_predictions():
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])

    result = mcnemar_test(
        labels,
        predictions,
        predictions,
    )

    assert result["a_correct_b_wrong"] == 0
    assert result["a_wrong_b_correct"] == 0
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_discordant_predictions():
    labels = np.array([0, 0, 0, 0])
    predictions_a = np.array([0, 0, 0, 0])
    predictions_b = np.array([1, 1, 1, 1])

    result = mcnemar_test(
        labels,
        predictions_a,
        predictions_b,
    )

    assert result["a_correct_b_wrong"] == 4
    assert result["a_wrong_b_correct"] == 0
    assert result["p_value"] == pytest.approx(0.125)
