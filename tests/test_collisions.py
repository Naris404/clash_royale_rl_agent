"""Unit collisions at the Board seam: units push each other, buildings never move."""

from __future__ import annotations

from math import hypot

from cr_rl.game.board import (
    BRIDGE_HALF_WIDTH,
    BRIDGE_LANE_X,
    RIVER_HALF_WIDTH,
    RIVER_Y,
    NUM_ACTIONS,
    TICK_DT,
    Board,
)
from cr_rl.game.cards import cards_dic
from scenario import new_board, place_unit


def _step(board: Board, seconds: float) -> None:
    for _ in range(round(seconds / TICK_DT)):
        board.step(0, action_p1=0)


_TOWER_SPOTS = [(t.x, t.y) for t in new_board().towers]


def _gap(a, b) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def _contact(card_a: str, card_b: str) -> float:
    return cards_dic[card_a]["collision_radius"] + cards_dic[card_b]["collision_radius"]


def test_two_units_placed_on_the_same_spot_separate():
    board = new_board()
    a = place_unit(board, "Knight", 0, 9.0, 10.0)
    b = place_unit(board, "Knight", 0, 9.0, 10.0)

    _step(board, 0.1)

    assert _gap(a, b) >= _contact("Knight", "Knight") - 1e-6


def test_heavier_unit_is_pushed_less_than_the_lighter_one():
    board = new_board()
    giant = place_unit(board, "Giant", 0, 14.0, 10.0)
    knight = place_unit(board, "Knight", 0, 14.4, 10.0)

    _step(board, 0.1)

    assert abs(giant.x - 14.0) < abs(knight.x - 14.4)
    assert giant.x < knight.x  # pushed apart, not swapped


def test_buildings_are_never_displaced():
    board = new_board()
    cannon = place_unit(board, "Cannon", 1, 9.0, 20.0)
    start = (cannon.x, cannon.y)
    place_unit(board, "Giant", 1, 9.0, 20.0)
    place_unit(board, "Knight", 1, 9.2, 20.0)

    _step(board, 1.0)

    assert (cannon.x, cannon.y) == start
    assert all((t.x, t.y) == t0 for t, t0 in zip(board.towers, _TOWER_SPOTS))


def test_unit_walks_around_its_own_tower_instead_of_through_it():
    board = new_board()
    giant = place_unit(board, "Giant", 0, 3.0, 4.5)
    tower = next(t for t in board.towers if (t.owner, t.x) == (0, 3.5) and t.y == 8.0)
    closest = float("inf")

    for _ in range(150):
        board.step(0, action_p1=0)
        closest = min(closest, _gap(giant, tower))

    assert closest >= _contact("Giant", "Tower") - 1e-6
    assert giant.y > 12.0  # went past the tower, not stuck behind it


def _off_bridge_in_water(unit) -> bool:
    in_water = abs(unit.y - RIVER_Y) < RIVER_HALF_WIDTH
    on_bridge = any(abs(unit.x - bx) <= BRIDGE_HALF_WIDTH for bx in BRIDGE_LANE_X)
    return in_water and not on_bridge


def test_pushing_never_moves_a_ground_unit_into_the_river_off_a_bridge():
    board = new_board()
    bank = RIVER_Y - RIVER_HALF_WIDTH
    knight = place_unit(board, "Knight", 0, 9.0, bank - 0.1)
    for dy in (0.3, 0.6, 0.9):
        place_unit(board, "Giant", 0, 9.0, bank - dy)

    for _ in range(100):
        board.step(0, action_p1=0)
        assert not _off_bridge_in_water(knight)


def test_hog_rider_may_still_be_pushed_over_the_river():
    board = new_board()
    bank = RIVER_Y - RIVER_HALF_WIDTH
    hog = place_unit(board, "Hog_Rider", 0, 9.0, bank - 0.1)
    place_unit(board, "Giant", 0, 9.0, bank - 0.3)

    _step(board, 0.1)

    assert hog.y > bank  # jumps the river, so the water is no barrier


def _play_random_match(seed: int, steps: int = 1200) -> list[tuple[float, float]]:
    """Every unit position of a random-vs-random match, so a divergence anywhere shows."""
    board = Board(seed=seed)
    trace: list[tuple[float, float]] = []
    for _ in range(steps):
        board.step(int(board.np_rng.integers(0, NUM_ACTIONS)))
        assert not any(_off_bridge_in_water(t) for t in board.troops if not t.jumps_river)
        trace += [(t.x, t.y) for t in board.troops]
    return trace


def test_same_seed_gives_the_same_crowded_match():
    first = _play_random_match(seed=7)
    second = _play_random_match(seed=7)

    assert len(first) > 1000  # the match really was crowded
    assert first == second
