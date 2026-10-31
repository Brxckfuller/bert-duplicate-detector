
import pytest
import json
from unittest.mock import patch

from predict import DuplicateQuestionDetector


@pytest.mark.parametrize(
    "question1, question2",
    [
        ("", "Valid question"),
        ("   ", "Valid question"),
        ("Valid question", ""),
        ("Valid question", "   "),
        (None, "Valid question"),
        ("Valid question", None),
        (123, "Valid question"),
    ],
)
def test_invalid_questions(detector, question1, question2):
    with pytest.raises(ValueError):
        detector.predict(question1, question2)


def test_prediction_output(detector):
    result = detector.predict(
        "How do neural networks learn?",
        "How are neural networks trained?",
    )

    assert set(result) == {
        "duplicate_probability",
        "is_duplicate",
        "threshold",
    }

    assert isinstance(result["duplicate_probability"], float)
    assert isinstance(result["is_duplicate"], bool)
    assert 0.0 <= result["duplicate_probability"] <= 1.0


def test_threshold_consistency(detector):
    result = detector.predict(
        "Question one",
        "Question two",
    )

    assert result["is_duplicate"] == (
        result["duplicate_probability"] >= result["threshold"]
    )


def test_repeated_predictions(detector):
    first = detector.predict("Question one", "Question two")
    second = detector.predict("Question one", "Question two")

    assert first == second


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(threshold):
    # Avoid loading a model by providing an invalid threshold
    with pytest.raises(ValueError):
        DuplicateQuestionDetector(threshold=threshold)


def test_loads_threshold_from_checkpoint(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {"inference_threshold": 0.35}

    with open(model_dir / "training_config.json", "w") as file:
        json.dump(metadata, file)

    with patch("predict.AutoTokenizer.from_pretrained"), \
         patch("predict.AutoModelForSequenceClassification.from_pretrained"):

        detector = DuplicateQuestionDetector(model_dir=model_dir)

    assert detector.threshold == 0.35
