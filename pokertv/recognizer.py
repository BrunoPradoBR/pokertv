from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from pokertv.models import CardPrediction

SUITS = ["c", "d", "h", "s"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
CARD_CLASSES: List[str] = [f"{r}{s}" for r in RANKS for s in SUITS] + ["back"]

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_CARD_REGION_PREFIXES = ("hole_card", "community")


def _preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    return _TRANSFORM(crop).unsqueeze(0)


class CardRecognizer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        confidence_threshold: float = 0.85,
    ):
        self._model = model.to(device)
        self._model.train(False)
        self._device = device
        self._confidence_threshold = confidence_threshold

    @classmethod
    def from_path(
        cls,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.85,
    ) -> "CardRecognizer":
        model = torch.jit.load(model_path, map_location=device)
        return cls(model=model, device=device, confidence_threshold=confidence_threshold)

    def recognize(self, region_map: Dict[str, np.ndarray]) -> List[CardPrediction]:
        card_regions = {
            k: v for k, v in region_map.items()
            if any(k.startswith(p) for p in _CARD_REGION_PREFIXES)
        }
        if not card_regions:
            return []

        names = list(card_regions.keys())
        tensors = torch.cat(
            [_preprocess_crop(card_regions[n]) for n in names], dim=0
        ).to(self._device)

        with torch.no_grad():
            probs = F.softmax(self._model(tensors), dim=1)

        predictions: List[CardPrediction] = []
        for i, name in enumerate(names):
            conf, idx = probs[i].max(dim=0)
            conf_val = float(conf)
            card = CARD_CLASSES[int(idx)] if conf_val >= self._confidence_threshold else "unknown"
            predictions.append(CardPrediction(region_name=name, card=card, confidence=conf_val))

        return predictions
