"""Test-only setup helpers for building Board scenarios.

Scenarios still run by stepping the Board; these helpers only arrange the starting
position. Production code must not import this module.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from cr_rl.game.board import HAND_SIZE, Board
from cr_rl.game.cards import PLAYABLE_CARDS, Troop


def new_board(seed: int = 42) -> Board:
    return Board(seed=seed)


def set_hand(board: Board, player: int, hand: list[str]) -> None:
    """Start `player`'s hand with these cards; the rest of the deck follows in deck order."""
    cards = list(hand) + [c for c in PLAYABLE_CARDS if c not in hand]
    board.hand[player] = cards[:HAND_SIZE]
    board.hand_queue[player] = deque(cards[HAND_SIZE:])


def place_unit(
    board: Board,
    name: str,
    owner: int,
    x: float,
    y: float,
    *,
    hp: Optional[float] = None,
) -> Troop:
    """Put a unit on the board at (x, y), for setups legal play cannot reach."""
    unit = Troop(name, owner)
    unit.place((x, y))
    if hp is not None:
        unit.hp = hp
    board.troops.append(unit)
    return unit
