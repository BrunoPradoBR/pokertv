import numpy as np
import pytest
from datetime import datetime
from pokertv.models import (
    FrameData, TableDetection, Rect, CardPrediction, TextData
)


@pytest.fixture
def blank_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def green_felt_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[180:540, 320:960] = [0, 120, 0]
    return frame


@pytest.fixture
def minimal_detection():
    return TableDetection(
        window_id=1,
        client="pokerstars",
        confidence=0.90,
        bbox=Rect(x=0.0, y=0.0, w=1.0, h=1.0),
    )


@pytest.fixture
def idle_frame_data(minimal_detection):
    return FrameData(
        timestamp=datetime(2026, 5, 7, 14, 32, 0),
        window_id=1,
        detection=minimal_detection,
        cards=[],
        text=TextData(pot=0.0, stacks={}, names={}, action_labels={}, dealer_seat=0),
    )


@pytest.fixture
def preflop_frame_data(minimal_detection):
    return FrameData(
        timestamp=datetime(2026, 5, 7, 14, 32, 1),
        window_id=1,
        detection=minimal_detection,
        cards=[
            CardPrediction(region_name="hole_card1", card="Ah", confidence=0.99),
            CardPrediction(region_name="hole_card2", card="Kd", confidence=0.98),
        ],
        text=TextData(
            pot=0.15,
            stacks={0: 10.0, 1: 9.90},
            names={0: "Hero", 1: "Villain"},
            action_labels={"fold": "Fold", "call": "Call $0.10", "raise": "Raise"},
            dealer_seat=0,
        ),
    )


@pytest.fixture
def flop_frame_data(minimal_detection):
    return FrameData(
        timestamp=datetime(2026, 5, 7, 14, 32, 5),
        window_id=1,
        detection=minimal_detection,
        cards=[
            CardPrediction(region_name="hole_card1", card="Ah", confidence=0.99),
            CardPrediction(region_name="hole_card2", card="Kd", confidence=0.98),
            CardPrediction(region_name="community1", card="2h", confidence=0.97),
            CardPrediction(region_name="community2", card="7d", confidence=0.96),
            CardPrediction(region_name="community3", card="Jc", confidence=0.95),
        ],
        text=TextData(
            pot=0.30,
            stacks={0: 10.0, 1: 9.70},
            names={0: "Hero", 1: "Villain"},
            action_labels={"check": "Check", "bet": "Bet"},
            dealer_seat=0,
        ),
    )
