
import pytest

from config import MODEL_DIR
from predict import DuplicateQuestionDetector


@pytest.mark.integration
def test_real_model_inference():
    if not MODEL_DIR.exists():
        pytest.skip("Fine-tuned model checkpoint not available")

    detector = DuplicateQuestionDetector()

    result = detector.predict(
        "What is machine learning?",
        "How do I cook pasta?",
    )

    assert 0.0 <= result["duplicate_probability"] <= 1.0
    assert isinstance(result["is_duplicate"], bool)
    assert result["is_duplicate"] is False
