"""The action contract and the Deck's card ids; the observation layout is pinned in test_observation.py."""

from __future__ import annotations

import pytest

from cr_rl.game.board import NUM_ACTIONS, NUM_ZONES
from cr_rl.game.cards import CARD_TO_ID, DECK, cards_dic
from scenario import new_board, set_hand


def test_action_layout():
    assert (NUM_ZONES, NUM_ACTIONS) == (13, 53)


def test_card_ids_follow_deck_order():
    assert DECK == ("Knight", "Giant", "Hog_Rider", "Musketeer", "Cannon", "Fireball", "Arrows", "Baby_Dragon")
    assert CARD_TO_ID == {card: i for i, card in enumerate(DECK)}


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
