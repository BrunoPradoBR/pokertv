import json
from pokertv.models import Hand, Seat, Street, Action


def test_hand_serializes_to_json():
    hand = Hand(
        hand_id="test-123",
        timestamp="2026-05-07T14:32:00",
        table_name="Zoom NL10",
        game_type="NLH",
        stakes="0.05/0.10",
        seats=[Seat(index=0, name="Hero", stack=10.0, position="BTN")],
        hole_cards=["Ah", "Kd"],
        board=["2h", "7d", "Jc"],
        streets=[Street(name="preflop", board=[], pot=0.15, actions=[
            Action(player="Hero", action_type="raise", amount=0.25)
        ])],
        result=None,
        status="complete",
    )
    data = json.loads(hand.to_json())
    assert data["hand_id"] == "test-123"
    assert data["hole_cards"] == ["Ah", "Kd"]
    assert data["status"] == "complete"


def test_hand_roundtrip():
    hand = Hand(
        hand_id="abc-456",
        timestamp="2026-05-07T15:00:00",
        table_name="Table1",
        game_type="NLH",
        stakes="0.02/0.05",
        seats=[],
        hole_cards=["Qc", "Js"],
        board=[],
        streets=[],
        result=None,
        status="incomplete",
    )
    restored = Hand.from_json(hand.to_json())
    assert restored.hand_id == hand.hand_id
    assert restored.status == hand.status
    assert restored.hole_cards == hand.hole_cards
