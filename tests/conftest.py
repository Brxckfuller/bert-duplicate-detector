
from unittest.mock import MagicMock

import pytest
import torch

from predict import DuplicateQuestionDetector


@pytest.fixture
def detector():
    # Bypass model loading for fast unit tests
    instance = DuplicateQuestionDetector.__new__(
        DuplicateQuestionDetector
    )

    instance.threshold = 0.15
    instance.device = torch.device("cpu")

    instance.tokenizer = MagicMock(
        return_value={
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
    )

    instance.model = MagicMock()
    instance.model.return_value.logits = torch.tensor(
        [[0.0, 2.0]]
    )

    return instance
