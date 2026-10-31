
import pytest

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
