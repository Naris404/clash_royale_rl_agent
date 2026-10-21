"""The env-v1.1 observation and action contract.

A trained model is only valid for the exact layout it was trained on, so every index and
encoding here is pinned. Changing any of it means a new environment version (ADR-0003):
update this file, docs/rl-agent.md and the model compatibility check together.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_rl.game.board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    HAND_SIZE,
    MAX_UNITS_OBS,
    NUM_ACTIONS,
    NUM_ZONES,
    OBS_DIM,
    OBS_GLOBAL_FEATURES,
    OBS_HAND_BASE,
    OBS_NEXT_BASE,
    OBS_TOWER_FEATURES,
    OBS_UNIT_BASE,
)
from cr_rl.game.cards import CARD_TO_ID, DECK, cards_dic
from scenario import new_board, place_unit, set_hand

UNIT_FEATURES = 5


def test_segment_offsets_and_sizes():
    assert (OBS_GLOBAL_FEATURES, OBS_TOWER_FEATURES) == (3, 6)
    assert (OBS_HAND_BASE, OBS_NEXT_BASE, OBS_UNIT_BASE) == (9, 41, 49)
    assert (HAND_SIZE, MAX_UNITS_OBS, OBS_DIM) == (4, 24, 169)
    assert OBS_UNIT_BASE + MAX_UNITS_OBS * UNIT_FEATURES == OBS_DIM
    assert (NUM_ZONES, NUM_ACTIONS) == (11, 45)


def test_card_ids_follow_deck_order():
    assert DECK == ("Knight", "Giant", "Hog_Rider", "Musketeer", "Cannon", "Fireball", "Arrows", "Baby_Dragon")
    assert CARD_TO_ID == {card: i for i, card in enumerate(DECK)}


def test_global_features_and_tower_order():
    board = new_board()
    board.elixir[:] = [4.0, 7.0]
    board.time = 90.0
    own_left = next(t for t in board.towers if t.owner == 0 and t.x < ARENA_WIDTH / 2 and t.name == "Tower")
    own_left.hp = own_left.max_hp / 2

    obs = board.get_observation(0)

    assert obs[:3] == pytest.approx([0.4, 0.7, 0.5])  # own elixir, enemy elixir, time
    # own left princess, own right princess, own king, then the same for the enemy
    assert obs[3:9] == pytest.approx([0.5, 1, 1, 1, 1, 1])


def test_hand_slots_are_positional_and_the_next_card_is_separate():
    board = new_board()
    set_hand(board, 0, ["Hog_Rider", "Knight", "Fireball", "Baby_Dragon"])
    obs = board.get_observation(0)
    for slot, card in enumerate(["Hog_Rider", "Knight", "Fireball", "Baby_Dragon"]):
        block = obs[OBS_HAND_BASE + slot * 8 : OBS_HAND_BASE + (slot + 1) * 8]
        assert block.argmax() == CARD_TO_ID[card] and block.sum() == 1.0
    assert obs[OBS_NEXT_BASE : OBS_NEXT_BASE + 8].argmax() == CARD_TO_ID["Giant"]  # queue head


def test_unit_slot_encoding():
    board = new_board()
    place_unit(board, "Giant", 0, 4.5, 12.0, hp=cards_dic["Giant"]["hp"] / 2)
    place_unit(board, "Baby_Dragon", 1, 13.5, 20.0)

    obs = board.get_observation(0)
    slots = obs[OBS_UNIT_BASE:].reshape(MAX_UNITS_OBS, UNIT_FEATURES)

    # nearest to the enemy king first: the Baby Dragon at (13.5, 20) is closer than the Giant
    assert slots[0] == pytest.approx([-1, 1.0, 13.5 / ARENA_WIDTH, 20.0 / ARENA_LENGTH, 1.0])
    assert slots[1] == pytest.approx([1, 1 / 7, 4.5 / ARENA_WIDTH, 12.0 / ARENA_LENGTH, 0.5])
    assert not slots[2:].any()  # empty slots are all zero


def test_units_beyond_24_are_dropped_farthest_first():
    board = new_board()
    for i in range(MAX_UNITS_OBS + 3):  # one lane, so a larger y is closer to the enemy king
        place_unit(board, "Knight", 0, 9.0, 3.0 + i)

    slots = board.get_observation(0)[OBS_UNIT_BASE:].reshape(MAX_UNITS_OBS, UNIT_FEATURES)

    kept_y = slots[:, 3] * ARENA_LENGTH
    assert kept_y == pytest.approx([3.0 + i for i in range(MAX_UNITS_OBS + 2, 2, -1)])


@pytest.mark.parametrize("player", [0, 1])
def test_mask_agrees_with_legal_plays_for_every_action(player):
    board = new_board()
    set_hand(board, player, ["Knight", "Cannon", "Fireball", "Baby_Dragon"])
    for elixir in (0.0, 3.0, 4.0, 10.0):
        board.elixir[player] = elixir
        mask = board.valid_action_mask(player)
        assert mask[0]
        for action in range(1, NUM_ACTIONS):
            card, zone = board.action_to_card_zone(action, player)
            legal = elixir >= cards_dic[card]["elisir"] and board.zone_allows(player, card, zone)
            assert mask[action] == legal, (elixir, action, card, zone)
