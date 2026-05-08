from __future__ import annotations
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional
from pokertv.models import FrameData, Hand, Street, Action, Seat


class FSMState(Enum):
    IDLE = auto()
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()


def _community_cards(frame: FrameData) -> List[str]:
    return [c.card for c in frame.cards
            if c.region_name.startswith("community") and c.card not in ("back", "unknown")]


def _hole_cards(frame: FrameData) -> List[str]:
    return [c.card for c in frame.cards
            if c.region_name.startswith("hole_card") and c.card not in ("back", "unknown")]


def _infer_target(frame: FrameData, current: FSMState) -> FSMState:
    holes = _hole_cards(frame)
    community = _community_cards(frame)
    n = len(community)
    pot = frame.text.pot

    if not holes and not pot:
        return FSMState.IDLE
    if len(holes) == 2 and pot > 0 and n == 0:
        return FSMState.PREFLOP
    if n == 3:
        return FSMState.FLOP
    if n == 4:
        return FSMState.TURN
    if n == 5:
        if current == FSMState.RIVER and not frame.text.action_labels:
            return FSMState.SHOWDOWN
        return FSMState.RIVER
    return current


class _HandBuilder:
    def __init__(self, frame: FrameData):
        self.hand_id = str(uuid.uuid4())
        self.timestamp = frame.timestamp.isoformat()
        self.seats: List[Seat] = [
            Seat(index=idx, name=name, stack=frame.text.stacks.get(idx, 0.0), position="")
            for idx, name in frame.text.names.items()
        ]
        self.hole_cards: List[str] = _hole_cards(frame)
        self.board: List[str] = []
        self.streets: List[Street] = [Street(name="preflop", board=[], pot=frame.text.pot, actions=[])]

    def start_street(self, name: str, frame: FrameData) -> None:
        self.board = _community_cards(frame)
        self.streets.append(Street(name=name, board=list(self.board), pot=frame.text.pot, actions=[]))

    def add_action(self, action: Action) -> None:
        if self.streets:
            self.streets[-1].actions.append(action)

    def build(self, status: str) -> Hand:
        return Hand(
            hand_id=self.hand_id,
            timestamp=self.timestamp,
            table_name="PokerStars Table",
            game_type="NLH",
            stakes="unknown",
            seats=self.seats,
            hole_cards=self.hole_cards,
            board=self.board,
            streets=self.streets,
            result=None,
            status=status,
        )


class HandStateMachine:
    def __init__(self):
        self._state = FSMState.IDLE
        self._pending: Optional[FSMState] = None
        self._pending_count: int = 0
        self._builder: Optional[_HandBuilder] = None
        self._prev_stacks: Dict[int, float] = {}

    @property
    def state(self) -> FSMState:
        return self._state

    def update(self, frame: FrameData) -> Optional[Hand]:
        target = _infer_target(frame, self._state)

        if target != self._state:
            if target == self._pending:
                self._pending_count += 1
                if self._pending_count >= 2:
                    return self._transition(target, frame)
            else:
                self._pending = target
                self._pending_count = 1
        else:
            self._pending = None
            self._pending_count = 0
            self._record_actions(frame)

        return None

    def flush_incomplete(self) -> Optional[Hand]:
        if self._builder and self._state != FSMState.IDLE:
            hand = self._builder.build(status="incomplete")
            self._builder = None
            self._state = FSMState.IDLE
            self._prev_stacks = {}
            return hand
        return None

    def _transition(self, new_state: FSMState, frame: FrameData) -> Optional[Hand]:
        self._pending = None
        self._pending_count = 0
        self._state = new_state

        if new_state == FSMState.PREFLOP:
            self._builder = _HandBuilder(frame)
            self._prev_stacks = dict(frame.text.stacks)
        elif new_state in (FSMState.FLOP, FSMState.TURN, FSMState.RIVER):
            if self._builder:
                self._builder.start_street(new_state.name.lower(), frame)
        elif new_state == FSMState.SHOWDOWN:
            if self._builder:
                self._builder.start_street("showdown", frame)
        elif new_state == FSMState.IDLE and self._builder:
            hand = self._builder.build(status="complete")
            self._builder = None
            self._prev_stacks = {}
            return hand

        return None

    def _record_actions(self, frame: FrameData) -> None:
        if not self._builder or self._state == FSMState.IDLE:
            return
        for seat_idx, current_stack in frame.text.stacks.items():
            prev = self._prev_stacks.get(seat_idx)
            if prev is not None:
                delta = prev - current_stack
                if delta > 0.001:
                    name = frame.text.names.get(seat_idx, f"Seat{seat_idx}")
                    self._builder.add_action(Action(player=name, action_type="bet", amount=round(delta, 2)))
        self._prev_stacks = dict(frame.text.stacks)
