from __future__ import annotations

from typing import Any, Dict

import numpy as np
import yaml


class Segmenter:
    def __init__(self, layout: Dict[str, Any]) -> None:
        self._layout = layout

    @classmethod
    def from_yaml(cls, path: str) -> Segmenter:
        with open(path) as f:
            return cls(layout=yaml.safe_load(f))

    def segment(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = frame.shape[:2]
        regions: Dict[str, np.ndarray] = {}

        for card_def in self._layout.get("hole_cards", []):
            regions[card_def["name"]] = self._crop(frame, card_def, w, h)

        for card_def in self._layout.get("community_cards", []):
            regions[card_def["name"]] = self._crop(frame, card_def, w, h)

        pot_def = self._layout.get("pot")
        if pot_def:
            regions["pot"] = self._crop(frame, pot_def, w, h)

        for seat in self._layout.get("seats", []):
            idx = seat["index"]
            regions[f"seat_{idx}_name"] = self._crop(frame, seat["name"], w, h)
            regions[f"seat_{idx}_stack"] = self._crop(frame, seat["stack"], w, h)

        for btn_name, btn_def in self._layout.get("action_buttons", {}).items():
            regions[f"action_{btn_name}"] = self._crop(frame, btn_def, w, h)

        return regions

    def _crop(
        self, frame: np.ndarray, region: Dict[str, float], w: int, h: int
    ) -> np.ndarray:
        x1 = max(0, int(region["x"] * w))
        y1 = max(0, int(region["y"] * h))
        x2 = min(w, x1 + int(region["w"] * w))
        y2 = min(h, y1 + int(region["h"] * h))
        return frame[y1:y2, x1:x2].copy()
