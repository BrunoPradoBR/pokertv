from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from dataclasses_json import dataclass_json


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float


@dataclass
class TableDetection:
    window_id: int
    client: str
    confidence: float
    bbox: Rect


@dataclass
class CardPrediction:
    region_name: str
    card: str
    confidence: float


@dataclass
class TextData:
    pot: float
    stacks: Dict[int, float]
    names: Dict[int, str]
    action_labels: Dict[str, str]
    dealer_seat: int


@dataclass
class FrameData:
    timestamp: object  # datetime, not serialized
    window_id: int
    detection: TableDetection
    cards: List[CardPrediction]
    text: TextData


@dataclass_json
@dataclass
class Action:
    player: str
    action_type: str
    amount: float


@dataclass_json
@dataclass
class Street:
    name: str
    board: List[str]
    pot: float
    actions: List[Action] = field(default_factory=list)


@dataclass_json
@dataclass
class Seat:
    index: int
    name: str
    stack: float
    position: str


@dataclass_json
@dataclass
class Result:
    winner: str
    amount: float
    shown_cards: Dict[str, List[str]]


@dataclass_json
@dataclass
class Hand:
    hand_id: str
    timestamp: str
    table_name: str
    game_type: str
    stakes: str
    seats: List[Seat]
    hole_cards: List[str]
    board: List[str]
    streets: List[Street]
    result: Optional[Result]
    status: str
