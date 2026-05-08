"""Tests for the writer module (HandWriter and PokerStarsExporter)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokertv.models import Action, Hand, Result, Seat, Street
from pokertv.writer import HandWriter, PokerStarsExporter


def _make_hand(hand_id: str = "1", status: str = "complete") -> Hand:
    seats = [
        Seat(index=1, name="Alice", stack=10.0, position="BTN"),
        Seat(index=2, name="Bob", stack=8.0, position="BB"),
    ]
    preflop = Street(
        name="preflop",
        board=[],
        pot=0.03,
        actions=[Action(player="Alice", action_type="call", amount=0.02)],
    )
    return Hand(
        hand_id=hand_id,
        timestamp="2024-01-15T12:00:00",
        table_name="Test Table",
        game_type="NL Hold'em",
        stakes="$0.01/$0.02",
        seats=seats,
        hole_cards=["Ah", "Kd"],
        board=["2c", "3d", "4h"],
        streets=[preflop],
        result=Result(winner="Alice", amount=0.05, shown_cards={"Alice": ["Ah", "Kd"]}),
        status=status,
    )


# ---------------------------------------------------------------------------
# HandWriter tests
# ---------------------------------------------------------------------------


def test_hand_writer_creates_jsonl_file(tmp_path: Path) -> None:
    """write 1 hand with buffer_size=1 -> file is created and has 1 line."""
    output = tmp_path / "hands.jsonl"
    hand = _make_hand()

    with HandWriter(str(output), buffer_size=1) as writer:
        writer.write(hand)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_hand_writer_jsonl_content_is_valid_json(tmp_path: Path) -> None:
    """write 1 hand, read line back, json.loads succeeds, hand_id matches."""
    output = tmp_path / "hands.jsonl"
    hand = _make_hand(hand_id="42")

    with HandWriter(str(output), buffer_size=1) as writer:
        writer.write(hand)

    line = output.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["hand_id"] == "42"


def test_hand_writer_buffers_and_flushes(tmp_path: Path) -> None:
    """buffer_size=3, write 3 hands -> file has 3 lines after 3rd write (auto-flush)."""
    output = tmp_path / "hands.jsonl"

    with HandWriter(str(output), buffer_size=3) as writer:
        writer.write(_make_hand(hand_id="1"))
        # After 1st write, buffer not full -> nothing on disk yet
        assert not output.exists() or output.read_text(encoding="utf-8").strip() == ""

        writer.write(_make_hand(hand_id="2"))
        writer.write(_make_hand(hand_id="3"))
        # After 3rd write, buffer reaches buffer_size -> auto-flush

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_hand_writer_flush_on_close(tmp_path: Path) -> None:
    """buffer_size=10, write 1 hand, close -> file has 1 line (close flushes)."""
    output = tmp_path / "hands.jsonl"
    hand = _make_hand()

    writer = HandWriter(str(output), buffer_size=10)
    writer.write(hand)
    writer.close()

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# PokerStarsExporter tests
# ---------------------------------------------------------------------------


def test_pokerstars_exporter_skips_incomplete(tmp_path: Path) -> None:
    """hand with status='incomplete' -> file stays empty."""
    output = tmp_path / "export.txt"
    hand = _make_hand(status="incomplete")

    with PokerStarsExporter(str(output)) as exporter:
        exporter.export(hand)

    content = output.read_text(encoding="utf-8") if output.exists() else ""
    assert content.strip() == ""


def test_pokerstars_exporter_writes_complete_hand(tmp_path: Path) -> None:
    """complete hand -> file contains 'PokerStars Hand #' and '*** HOLE CARDS ***'."""
    output = tmp_path / "export.txt"
    hand = _make_hand(status="complete")

    with PokerStarsExporter(str(output)) as exporter:
        exporter.export(hand)

    content = output.read_text(encoding="utf-8")
    assert "PokerStars Hand #" in content
    assert "*** HOLE CARDS ***" in content


def test_pokerstars_exporter_contains_seat_info(tmp_path: Path) -> None:
    """complete hand with 2 seats -> both seat names appear in file."""
    output = tmp_path / "export.txt"
    hand = _make_hand(status="complete")

    with PokerStarsExporter(str(output)) as exporter:
        exporter.export(hand)

    content = output.read_text(encoding="utf-8")
    assert "Alice" in content
    assert "Bob" in content
