from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from pokertv.models import TableDetection, Rect


def _compute_edge_density(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    center = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum()) / (255.0 * edges.size)


def _compute_hsv_histogram(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    center = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def save_reference_histogram(screenshot_path: str, output_path: str) -> None:
    """Utility: compute and save a reference histogram from a real PokerStars screenshot."""
    frame = cv2.imread(screenshot_path)
    if frame is None:
        raise FileNotFoundError(screenshot_path)
    hist = _compute_hsv_histogram(frame)
    np.save(output_path, hist)
    print(f"Saved histogram to {output_path}")


class TableDetector:
    def __init__(
        self,
        logo_template: Optional[np.ndarray],
        reference_histograms: List[np.ndarray],
        thresholds: Dict[str, float],
    ):
        self._logo_template = logo_template
        self._reference_histograms = reference_histograms
        self._thresholds = thresholds

    @classmethod
    def from_config(cls, layout_config: Dict[str, Any]) -> "TableDetector":
        t = layout_config.get("detection", {})
        thresholds = {
            "logo_match_threshold": t.get("logo_match_threshold", 0.85),
            "histogram_similarity_threshold": t.get("histogram_similarity_threshold", 0.80),
            "edge_density_min": t.get("edge_density_min", 0.005),
            "edge_density_max": t.get("edge_density_max", 0.12),
            "confidence_threshold": t.get("confidence_threshold", 0.70),
        }
        logo_path = layout_config.get("logo_template", "")
        logo_template = cv2.imread(logo_path) if os.path.exists(logo_path) else None
        reference_histograms = [
            np.load(p) for p in layout_config.get("reference_histograms", [])
            if os.path.exists(p)
        ]
        return cls(logo_template=logo_template, reference_histograms=reference_histograms, thresholds=thresholds)

    def detect(self, frame: np.ndarray, window_id: int) -> Optional[TableDetection]:
        logo_score = self._match_logo(frame)
        hist_score = self._match_histogram(frame)
        edge_ok = self._check_edge_density(frame)
        confidence = logo_score * 0.5 + hist_score * 0.3 + (0.2 if edge_ok else 0.0)
        if confidence < self._thresholds["confidence_threshold"]:
            return None
        return TableDetection(
            window_id=window_id,
            client="pokerstars",
            confidence=float(confidence),
            bbox=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
        )

    def _match_logo(self, frame: np.ndarray) -> float:
        if self._logo_template is None:
            return 0.0
        th, tw = self._logo_template.shape[:2]
        if frame.shape[0] < th or frame.shape[1] < tw:
            return 0.0
        crop = frame[:th, :tw]
        result = cv2.matchTemplate(crop, self._logo_template, cv2.TM_CCOEFF_NORMED)
        return float(result.max())

    def _match_histogram(self, frame: np.ndarray) -> float:
        if not self._reference_histograms:
            return 0.0
        hist = _compute_hsv_histogram(frame)
        return max(float(cv2.compareHist(hist, ref, cv2.HISTCMP_CORREL)) for ref in self._reference_histograms)

    def _check_edge_density(self, frame: np.ndarray) -> bool:
        density = _compute_edge_density(frame)
        return self._thresholds["edge_density_min"] <= density <= self._thresholds["edge_density_max"]


class ScreenCapture:
    def __init__(self, fps: int = 2):
        import mss
        self._sct = mss.mss()
        self._interval = 1.0 / max(1, min(fps, 4))

    def capture_all(self) -> List[Tuple[int, np.ndarray]]:
        frames = []
        for i, monitor in enumerate(self._sct.monitors[1:], start=1):
            shot = self._sct.grab(monitor)
            frame = np.array(shot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frames.append((i, frame))
        return frames
