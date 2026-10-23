"""The observation layout (version 2, ADR-0002).

A trained model is only valid for the exact layout it was trained on, so every segment offset
and encoding is pinned with literals. Changing any of it means a new environment version
(ADR-0003): update this file, docs/rl-agent.md and the model compatibility check together.
These tests do not depend on the action layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_rl.game.board import (
    OBS_DIM,
    OBS_GLOBAL_FEATURES,
    OBS_GRID_BASE,
    OBS_HAND_BASE,
    OBS_NEXT_BASE,
    OBS_PLAYS_BASE,
    OBS_TOWER_BASE,
)
from cr_rl.game.cards import CARD_TO_ID, cards_dic
from scenario import new_board, place_unit, set_hand


def test_segment_offsets_and_total_size():
    assert OBS_GLOBAL_FEATURES == 4  # own elixir, enemy elixir, elixir advantage, time
    assert OBS_TOWER_BASE == 4  # 6 tower HP
    assert OBS_HAND_BASE == 10  # 4 slots x 8 cards
    assert OBS_NEXT_BASE == 42  # 8 cards
    assert OBS_PLAYS_BASE == 50  # 4 plays x (8 cards + none)
    assert OBS_GRID_BASE == 86  # 2 owners x 2 lanes x 8 depth bins x 6 unit cards
    assert OBS_DIM == 278
    assert OBS_GRID_BASE + 2 * 2 * 8 * 6 == OBS_DIM


def test_global_features_and_tower_order():
    board = new_board()
    board.elixir[:] = [4.0, 7.0]
    board.time = 90.0
    own_left = next(t for t in board.towers if t.owner == 0 and t.x < 9.0 and t.name == "Tower")
    own_left.hp = own_left.max_hp / 2

    obs = board.get_observation(0)

    assert obs[:OBS_GLOBAL_FEATURES] == pytest.approx([0.4, 0.7, -0.3, 0.5])
    # own left princess, own right princess, own king, then the same for the enemy
    assert obs[OBS_TOWER_BASE:OBS_HAND_BASE] == pytest.approx([0.5, 1, 1, 1, 1, 1])


def test_hand_slots_are_positional_and_the_next_card_is_separate():
    board = new_board()
    hand = ["Hog_Rider", "Knight", "Fireball", "Baby_Dragon"]
    set_hand(board, 0, hand)
    obs = board.get_observation(0)
    for slot, card in enumerate(hand):
        block = obs[OBS_HAND_BASE + slot * 8 : OBS_HAND_BASE + (slot + 1) * 8]
        assert block.argmax() == CARD_TO_ID[card] and block.sum() == 1.0
    assert obs[OBS_NEXT_BASE : OBS_PLAYS_BASE].argmax() == CARD_TO_ID["Giant"]  # queue head


def test_a_unit_lights_exactly_its_lane_depth_cell_with_hp_over_max_hp():
    board = new_board()
    place_unit(board, "Giant", 0, 4.5, 12.0, hp=cards_dic["Giant"]["hp"] / 2)

    grid = board.get_observation(0)[OBS_GRID_BASE:]

    # own owner, left lane, depth bin 3 (y 12-16), unit card 1 (Giant): ((0*2+0)*8+3)*6+1 = 19
    assert np.flatnonzero(grid).tolist() == [19]
    assert grid[19] == pytest.approx(0.5)


def _grid_cell(owner: int, lane: int, depth: int, unit_card: int) -> int:
    """Index into the grid segment, written out independently of the implementation."""
    return ((owner * 2 + lane) * 8 + depth) * 6 + unit_card


def test_two_knights_in_one_cell_read_one_and_a_wounded_one_reads_less():
    board = new_board()
    place_unit(board, "Knight", 0, 4.0, 6.0)
    place_unit(board, "Knight", 0, 5.0, 7.0)
    place_unit(board, "Musketeer", 0, 4.0, 6.0, hp=cards_dic["Musketeer"]["hp"] / 4)

    grid = board.get_observation(0)[OBS_GRID_BASE:]

    assert grid[_grid_cell(0, 0, 1, 0)] == 1.0  # Knight: 1.0 + 1.0 capped at 1
    assert grid[_grid_cell(0, 0, 1, 3)] == pytest.approx(0.25)  # Musketeer


@pytest.mark.parametrize(
    "x, y, lane, depth",
    [
        (8.9, 0.0, 0, 0),
        (9.0, 3.9, 1, 0),  # the centre line belongs to the right lane
        (3.5, 4.0, 0, 1),
        (3.5, 15.9, 0, 3),
        (3.5, 16.0, 0, 4),  # the river is a bin edge: y = 16 starts the enemy half
        (14.5, 31.9, 1, 7),
    ],
)
def test_lane_and_depth_bins(x, y, lane, depth):
    board = new_board()
    place_unit(board, "Knight", 1, x, y)

    grid = board.get_observation(0)[OBS_GRID_BASE:]

    assert np.flatnonzero(grid).tolist() == [_grid_cell(1, lane, depth, 0)]  # enemy owner block


def test_dead_units_and_spells_have_no_cells():
    board = new_board()
    place_unit(board, "Knight", 1, 4.0, 20.0).alive = False
    board.elixir[0] = 10.0
    set_hand(board, 0, ["Fireball"])
    assert board.play_card_at(0, "Fireball", 9.0, 20.0)

    assert not board.get_observation(0)[OBS_GRID_BASE:].any()


def test_both_players_see_a_mirrored_position_alike():
    board = new_board()
    place_unit(board, "Giant", 0, 4.0, 10.0, hp=cards_dic["Giant"]["hp"] / 2)  # P0 left lane, own half
    place_unit(board, "Knight", 1, 4.0, 22.0)  # P1 left lane, mirrored depth (32 - 22 = 10)

    own, other = board.get_observation(0), board.get_observation(1)

    # each sees its own unit at depth 2 of the left lane, and the opponent's at depth 5 of the same lane
    assert own[OBS_GRID_BASE + _grid_cell(0, 0, 2, 1)] == pytest.approx(0.5)
    assert own[OBS_GRID_BASE + _grid_cell(1, 0, 5, 0)] == 1.0
    assert other[OBS_GRID_BASE + _grid_cell(0, 0, 2, 0)] == 1.0
    assert other[OBS_GRID_BASE + _grid_cell(1, 0, 5, 1)] == pytest.approx(0.5)
    # elixir and tower blocks swap, time is shared
    assert other[0] == own[1] and other[1] == own[0] and other[2] == -own[2] and other[3] == own[3]


def _observed_plays(obs) -> list[int]:
    """The opponent's last plays as card ids (8 = none), oldest first."""
    block = obs[OBS_PLAYS_BASE:OBS_GRID_BASE].reshape(4, 9)
    assert (block.sum(axis=1) == 1).all()
    return block.argmax(axis=1).tolist()


def _play_in_order(board, player: int, cards: list[str]) -> None:
    for card in cards:
        board.elixir[player] = 10.0
        zone = 8 if card in ("Fireball", "Arrows") else 2
        assert board.play_card(player, card, zone)


def test_missing_opponent_plays_read_none_and_the_latest_is_always_last():
    board = new_board()
    assert _observed_plays(board.get_observation(0)) == [8, 8, 8, 8]

    set_hand(board, 1, ["Knight", "Fireball", "Giant", "Cannon"])
    _play_in_order(board, 1, ["Fireball", "Knight"])  # spells are plays too

    assert _observed_plays(board.get_observation(0)) == [8, 8, CARD_TO_ID["Fireball"], CARD_TO_ID["Knight"]]
    assert _observed_plays(board.get_observation(1)) == [8, 8, 8, 8]  # own plays are not part of the feature


def test_four_plays_reveal_the_opponents_hand_and_next_card():
    board = new_board(seed=7)
    for _ in range(3):  # plays well past the first 4 so the window has to slide
        _play_in_order(board, 1, board.get_hand(1))
        plays = _observed_plays(board.get_observation(0))
        assert 8 not in plays
        queue_ids = set(plays)
        assert {CARD_TO_ID[c] for c in board.get_hand(1)} == set(range(8)) - queue_ids
        assert plays[0] == CARD_TO_ID[board.get_next_card(1)]  # oldest play is next to be drawn


def test_a_reset_forgets_the_plays():
    board = new_board()
    _play_in_order(board, 1, [board.get_hand(1)[0]])
    board.reset()
    assert _observed_plays(board.get_observation(0)) == [8, 8, 8, 8]
