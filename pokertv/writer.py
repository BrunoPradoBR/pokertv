"""Writer module: persists Hand objects to JSONL and PokerStars text formats."""
from __future__ import annotations

import io
from typing import IO, List

from pokertv.models import Hand


class HandWriter:
    """Appends Hand objects to a JSONL file with optional write buffering."""

    def __init__(self, path: str, buffer_size: int = 10) -> None:
        self._path = path
        self._buffer_size = buffer_size
        self._buffer: List[Hand] = []
        self._file: IO[str] = open(path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, hand: Hand) -> None:
        """Buffer a hand; flush automatically when buffer reaches buffer_size."""
        self._buffer.append(hand)
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write all buffered hands to disk (one JSON per line) and clear buffer."""
        for hand in self._buffer:
            self._file.write(hand.to_json() + "\n")
        self._file.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining buffered hands then close the file."""
        self.flush()
        self._file.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "HandWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class PokerStarsExporter:
    """Writes Hand objects in PokerStars hand history (.txt) format."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._file: IO[str] = open(path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, hand: Hand) -> None:
        """Format and write a complete hand; skip incomplete hands silently."""
        if hand.status != "complete":
            return
        text = self._format(hand)
        self._file.write(text)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._file.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PokerStarsExporter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format(self, hand: Hand) -> str:
        buf = io.StringIO()

        # --- Header ---
        buf.write(
            f"PokerStars Hand #{hand.hand_id}: {hand.game_type} ({hand.stakes})"
            f" - {hand.timestamp} ET\n"
        )
        buf.write(f"Table '{hand.table_name}' 6-max Seat #1 is the button\n")

        # --- Seat list ---
        for seat in hand.seats:
            buf.write(f"Seat {seat.index}: {seat.name} (${seat.stack:.2f} in chips)\n")

        buf.write("\n")

        # --- Hole cards ---
        buf.write("*** HOLE CARDS ***\n")
        if hand.hole_cards:
            cards_str = " ".join(hand.hole_cards)
            buf.write(f"Dealt to Hero [{cards_str}]\n")

        # Preflop actions
        preflop_street = next(
            (s for s in hand.streets if s.name == "preflop"), None
        )
        if preflop_street:
            for action in preflop_street.actions:
                buf.write(self._format_action(action))

        buf.write("\n")

        # --- Flop ---
        if len(hand.board) >= 3:
            flop = hand.board[:3]
            buf.write(f"*** FLOP *** [{' '.join(flop)}]\n")
            flop_street = next(
                (s for s in hand.streets if s.name == "flop"), None
            )
            if flop_street:
                for action in flop_street.actions:
                    buf.write(self._format_action(action))
            buf.write("\n")

        # --- Turn ---
        if len(hand.board) >= 4:
            flop = hand.board[:3]
            turn = hand.board[3]
            buf.write(f"*** TURN *** [{' '.join(flop)}] [{turn}]\n")
            turn_street = next(
                (s for s in hand.streets if s.name == "turn"), None
            )
            if turn_street:
                for action in turn_street.actions:
                    buf.write(self._format_action(action))
            buf.write("\n")

        # --- River ---
        if len(hand.board) >= 5:
            flop_turn = hand.board[:4]
            river = hand.board[4]
            buf.write(f"*** RIVER *** [{' '.join(flop_turn)}] [{river}]\n")
            river_street = next(
                (s for s in hand.streets if s.name == "river"), None
            )
            if river_street:
                for action in river_street.actions:
                    buf.write(self._format_action(action))
            buf.write("\n")

        # --- Showdown ---
        if hand.result is not None:
            buf.write("*** SHOWDOWN ***\n")
            for player, cards in hand.result.shown_cards.items():
                buf.write(f"{player}: shows [{' '.join(cards)}]\n")
            buf.write("\n")

        # --- Summary ---
        buf.write("*** SUMMARY ***\n")
        if hand.result is not None:
            buf.write(f"Total pot ${hand.result.amount:.2f} | Rake $0\n")
        else:
            buf.write("Total pot $0.00 | Rake $0\n")

        if hand.board:
            buf.write(f"Board [{' '.join(hand.board)}]\n")

        if hand.result is not None:
            winner_seat = next(
                (s for s in hand.seats if s.name == hand.result.winner), None
            )
            if winner_seat:
                buf.write(
                    f"Seat {winner_seat.index}: {winner_seat.name}"
                    f" won (${hand.result.amount:.2f})\n"
                )

        buf.write("\n")
        return buf.getvalue()

    @staticmethod
    def _format_action(action) -> str:
        """Return a single formatted action line."""
        if action.amount == 0:
            return f"{action.player}: {action.action_type}\n"
        return f"{action.player}: {action.action_type} ${action.amount:.2f}\n"
