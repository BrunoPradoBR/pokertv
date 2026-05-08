import pytest
from datetime import datetime
from pokertv.state_machine import HandStateMachine, FSMState
from pokertv.models import FrameData, TableDetection, Rect, CardPrediction, TextData, Hand


def make_detection():
    return TableDetection(window_id=1, client="pokerstars", confidence=0.9,
                          bbox=Rect(x=0.0, y=0.0, w=1.0, h=1.0))


def make_frame(hole_cards, community_cards, pot, stacks=None, names=None, action_labels=None):
    cards = [CardPrediction(region_name=f"hole_card{i+1}", card=c, confidence=0.99)
             for i, c in enumerate(hole_cards)]
    cards += [CardPrediction(region_name=f"community{i+1}", card=c, confidence=0.97)
              for i, c in enumerate(community_cards)]
    return FrameData(
        timestamp=datetime(2026, 5, 7, 14, 0, 0),
        window_id=1,
        detection=make_detection(),
        cards=cards,
        text=TextData(
            pot=pot,
            stacks=stacks or {},
            names=names or {0: "Hero"},
            action_labels=action_labels if action_labels is not None else {"fold": "Fold"},
            dealer_seat=0,
        ),
    )


def test_initial_state_is_idle():
    assert HandStateMachine().state == FSMState.IDLE


def test_idle_stays_idle_on_empty_frame():
    fsm = HandStateMachine()
    result = fsm.update(make_frame([], [], pot=0.0))
    assert fsm.state == FSMState.IDLE
    assert result is None


def test_preflop_requires_two_consecutive_frames():
    fsm = HandStateMachine()
    pf = make_frame(["Ah", "Kd"], [], pot=0.15)
    result1 = fsm.update(pf)
    assert fsm.state == FSMState.IDLE
    assert result1 is None
    result2 = fsm.update(pf)
    assert fsm.state == FSMState.PREFLOP
    assert result2 is None


def test_debounce_resets_when_observed_state_changes():
    fsm = HandStateMachine()
    pf = make_frame(["Ah", "Kd"], [], pot=0.15)
    idle = make_frame([], [], pot=0.0)
    fsm.update(pf)   # pending: PREFLOP count=1
    fsm.update(idle) # different state -- resets pending
    assert fsm.state == FSMState.IDLE
    fsm.update(pf)   # pending: PREFLOP count=1 again
    fsm.update(pf)   # count=2 -- transitions
    assert fsm.state == FSMState.PREFLOP


def test_flop_transition_after_preflop():
    fsm = HandStateMachine()
    pf = make_frame(["Ah", "Kd"], [], pot=0.15)
    fsm.update(pf); fsm.update(pf)

    flop = make_frame(["Ah", "Kd"], ["2h", "7d", "Jc"], pot=0.30)
    fsm.update(flop); fsm.update(flop)
    assert fsm.state == FSMState.FLOP


def test_full_hand_returns_completed_hand():
    fsm = HandStateMachine()

    def twice(frame):
        r = fsm.update(frame)
        return r or fsm.update(frame)

    twice(make_frame(["Ah", "Kd"], [], pot=0.15))
    twice(make_frame(["Ah", "Kd"], ["2h", "7d", "Jc"], pot=0.30))
    twice(make_frame(["Ah", "Kd"], ["2h", "7d", "Jc", "9s"], pot=0.60))
    twice(make_frame(["Ah", "Kd"], ["2h", "7d", "Jc", "9s", "As"], pot=1.20))
    twice(make_frame(["Ah", "Kd"], ["2h", "7d", "Jc", "9s", "As"], pot=1.20, action_labels={}))
    assert fsm.state == FSMState.SHOWDOWN

    result = twice(make_frame([], [], pot=0.0))
    assert result is not None
    assert isinstance(result, Hand)
    assert result.status == "complete"
    assert "Ah" in result.hole_cards
    assert "Kd" in result.hole_cards


def test_flush_incomplete_returns_partial_hand():
    fsm = HandStateMachine()
    pf = make_frame(["Ah", "Kd"], [], pot=0.15)
    fsm.update(pf); fsm.update(pf)
    assert fsm.state == FSMState.PREFLOP

    hand = fsm.flush_incomplete()
    assert hand is not None
    assert hand.status == "incomplete"
    assert fsm.state == FSMState.IDLE
