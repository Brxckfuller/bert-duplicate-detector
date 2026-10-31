import pytest

from contrastive_robustness import (
    CONTRASTIVE_GROUPS,
    evaluate_group,
)


class FakeDetector:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, question1, question2):
        return self.predictions[(question1, question2)]


def test_expected_number_of_groups():
    assert len(CONTRASTIVE_GROUPS) == 16


def test_category_balance():
    categories = [group["category"] for group in CONTRASTIVE_GROUPS]

    for category in {"numerical", "negation", "entity", "paraphrase"}:
        assert categories.count(category) == 4


def test_question_fields_are_nonempty():
    for group in CONTRASTIVE_GROUPS:
        for field in ("reference", "equivalent", "contrast"):
            assert isinstance(group[field], str)
            assert group[field].strip()


def test_no_identical_questions_within_group():
    for group in CONTRASTIVE_GROUPS:
        questions = [
            group["reference"],
            group["equivalent"],
            group["contrast"],
        ]

        assert len(set(questions)) == 3


def test_no_duplicate_question_pairs():
    pairs = []

    for group in CONTRASTIVE_GROUPS:
        pairs.append(
            (group["reference"], group["equivalent"])
        )
        pairs.append(
            (group["reference"], group["contrast"])
        )

    assert len(pairs) == len(set(pairs))


@pytest.mark.parametrize(
    "positive_score,negative_score,expected_both,expected_ordering",
    [
        (0.90, 0.05, True, True),
        (0.90, 0.80, False, True),
        (0.05, 0.01, False, True),
        (0.05, 0.90, False, False),
        (0.50, 0.50, False, False),
    ],
)
def test_evaluate_group(
    positive_score,
    negative_score,
    expected_both,
    expected_ordering,
):
    group = {
        "category": "test",
        "reference": "Reference question?",
        "equivalent": "Equivalent question?",
        "contrast": "Different question?",
    }

    threshold = 0.15

    predictions = {
        ("Reference question?", "Equivalent question?"): {
            "duplicate_probability": positive_score,
            "is_duplicate": positive_score >= threshold,
        },
        ("Reference question?", "Different question?"): {
            "duplicate_probability": negative_score,
            "is_duplicate": negative_score >= threshold,
        },
    }

    detector = FakeDetector(predictions)

    result = evaluate_group(detector, group)

    assert result["both_correct"] == expected_both
    assert result["correct_ordering"] == expected_ordering
    assert result["positive_correct"] == (positive_score >= threshold)
    assert result["negative_correct"] == (negative_score < threshold)

    assert result["score_margin"] == pytest.approx(
        round(positive_score - negative_score, 4)
    )