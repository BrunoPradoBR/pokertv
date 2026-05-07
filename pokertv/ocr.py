from __future__ import annotations
import re
from typing import Dict
import numpy as np

from pokertv.models import TextData


def parse_amount(s: str) -> float:
    """Extract the last dollar amount from OCR strings like '$12.50', 'Call $4.00', '10BB'.

    Args:
        s: Raw OCR string that may contain currency symbols, commas, or text.

    Returns:
        The parsed float amount, or 0.0 if no number is found or parsing fails.
    """
    s = s.strip()
    matches = re.findall(r'\d[\d,]*\.?\d*', s)
    if not matches:
        return 0.0
    raw = matches[-1].replace(',', '')
    try:
        return float(raw)
    except ValueError:
        return 0.0


class OCREngine:
    """Wraps PaddleOCR for extracting text from PokerStars UI crops."""

    def __init__(self, use_gpu: bool = True):
        """Initialize the OCR engine.

        Args:
            use_gpu: Whether to use GPU acceleration for PaddleOCR.
        """
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, show_log=False)

    def read_text(self, crop: np.ndarray) -> str:
        """Return best-guess string from a single image crop.

        Args:
            crop: A numpy array representing an image crop (H x W x C).

        Returns:
            The OCR-detected text string, or empty string if detection fails.
        """
        result = self._ocr.ocr(crop, cls=False)
        if not result or not result[0]:
            return ""
        lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]
        return " ".join(lines).strip()

    def extract(self, region_map: Dict[str, np.ndarray]) -> TextData:
        """Extract all TextData fields from a RegionMap.

        Args:
            region_map: A dictionary mapping region names to image crops.

        Returns:
            A TextData instance with extracted pot, stacks, names, and action labels.
        """
        empty = np.zeros((10, 50, 3), dtype=np.uint8)
        pot = parse_amount(self.read_text(region_map.get("pot", empty)))

        stacks: Dict[int, float] = {}
        names: Dict[int, str] = {}
        for key, crop in region_map.items():
            if key.startswith("seat_") and key.endswith("_stack"):
                idx = int(key.split("_")[1])
                stacks[idx] = parse_amount(self.read_text(crop))
            elif key.startswith("seat_") and key.endswith("_name"):
                idx = int(key.split("_")[1])
                names[idx] = self.read_text(crop)

        action_labels: Dict[str, str] = {}
        for btn in ("fold", "call", "raise"):
            key = f"action_{btn}"
            if key in region_map:
                action_labels[btn] = self.read_text(region_map[key])

        return TextData(pot=pot, stacks=stacks, names=names, action_labels=action_labels, dealer_seat=0)
