import numpy as np
import torch
import torch.nn as nn
from pokertv.recognizer import CardRecognizer, CARD_CLASSES, _preprocess_crop


def make_mock_model() -> nn.Module:
    class TinyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(x.shape[0], 53)
            logits[:, 0] = 10.0
            return logits
    return TinyModel()


def make_low_confidence_model() -> nn.Module:
    class LowModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 53)
    return LowModel()


def test_card_classes_has_53_entries():
    assert len(CARD_CLASSES) == 53
    assert "Ac" in CARD_CLASSES
    assert "back" in CARD_CLASSES


def test_preprocess_crop_returns_correct_shape():
    crop = np.zeros((80, 60, 3), dtype=np.uint8)
    tensor = _preprocess_crop(crop)
    assert tensor.shape == (1, 3, 224, 224)


def test_recognizer_returns_card_predictions():
    rec = CardRecognizer(model=make_mock_model(), device="cpu")
    crops = {
        "hole_card1": np.zeros((80, 60, 3), dtype=np.uint8),
        "hole_card2": np.zeros((80, 60, 3), dtype=np.uint8),
    }
    preds = rec.recognize(crops)
    assert len(preds) == 2
    assert all(p.region_name in crops for p in preds)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)


def test_recognizer_ignores_non_card_regions():
    rec = CardRecognizer(model=make_mock_model(), device="cpu")
    crops = {
        "pot": np.zeros((20, 80, 3), dtype=np.uint8),
        "seat_0_name": np.zeros((15, 60, 3), dtype=np.uint8),
        "hole_card1": np.zeros((80, 60, 3), dtype=np.uint8),
    }
    preds = rec.recognize(crops)
    assert len(preds) == 1
    assert preds[0].region_name == "hole_card1"


def test_recognizer_low_confidence_emits_unknown():
    rec = CardRecognizer(model=make_low_confidence_model(), device="cpu", confidence_threshold=0.99)
    crops = {"hole_card1": np.zeros((80, 60, 3), dtype=np.uint8)}
    preds = rec.recognize(crops)
    assert preds[0].card == "unknown"
