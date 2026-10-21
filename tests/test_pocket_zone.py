"""Pocket deploy zone (#50): a fallen princess tower opens a 4 x 9 pocket on the enemy side of its lane."""

from __future__ import annotations

import pytest

from cr_rl.agents.logic import _knight_on_unit
from cr_rl.coach.inference import PolicyInspector
from cr_rl.game.board import (
    ARENA_WIDTH,
    DEPLOY_ZONES,
    RIVER_HALF_WIDTH,
    RIVER_Y,
    TOWER_LAYOUT,
    Board,
)
from cr_rl.server.sessions import MODE_VS_BOT, GameSession
from scenario import new_board, place_unit, set_hand

LEFT_LANE_X = TOWER_LAYOUT[1]["Tower_L"][0]
RIGHT_LANE_X = TOWER_LAYOUT[1]["Tower_R"][0]
POCKET_DEPTH = 4.0
FAR_BANK = RIVER_Y + RIVER_HALF_WIDTH  # first dry tile on player 1's side


def _destroy_princess(board: Board, owner: int, lane_x: float) -> None:
    tower = next(t for t in board.towers if t.owner == owner and t.name == "Tower" and t.x == lane_x)
    tower.hp = 0.0
    tower.alive = False


def _ready(board: Board, player: int, card: str = "Knight") -> None:
    set_hand(board, player, [card])
    board.elixir[player] = 10.0


def test_destroying_a_princess_opens_a_pocket_in_its_lane_only():
    board = new_board()
    _ready(board, 0)
    for x in (LEFT_LANE_X, RIGHT_LANE_X):
        assert not board.can_play_card_at(0, "Knight", x, FAR_BANK + 1.0)

    _destroy_princess(board, 1, LEFT_LANE_X)

    assert board.can_play_card_at(0, "Knight", LEFT_LANE_X, FAR_BANK + 1.0)
    assert not board.can_play_card_at(0, "Knight", RIGHT_LANE_X, FAR_BANK + 1.0)


@pytest.mark.parametrize("player", [0, 1])
def test_pocket_is_4_deep_and_one_lane_half_wide_and_never_in_the_water(player):
    board = new_board()
    _ready(board, player)
    enemy = 1 - player
    _destroy_princess(board, enemy, LEFT_LANE_X)
    direction = 1.0 if player == 0 else -1.0
    bank = RIVER_Y + direction * RIVER_HALF_WIDTH

    assert board.can_play_card_at(player, "Knight", 5.0, bank)
    assert board.can_play_card_at(player, "Knight", 5.0, bank + direction * POCKET_DEPTH)
    assert not board.can_play_card_at(player, "Knight", 5.0, bank + direction * (POCKET_DEPTH + 0.1))
    assert not board.can_play_card_at(player, "Knight", 5.0, RIVER_Y)  # the water
    assert board.can_play_card_at(player, "Knight", ARENA_WIDTH / 2, bank + direction)  # lane half is 9 wide
    assert not board.can_play_card_at(player, "Knight", ARENA_WIDTH / 2 + 0.1, bank + direction)


@pytest.mark.parametrize("player", [0, 1])
def test_agents_get_no_pocket_zones(player):
    board = new_board()
    _ready(board, player)
    before = board.valid_action_mask(player).copy()

    _destroy_princess(board, 1 - player, LEFT_LANE_X)
    _destroy_princess(board, 1 - player, RIGHT_LANE_X)

    assert (board.valid_action_mask(player) == before).all()
    enemy_half = (lambda y: y > RIVER_Y) if player == 0 else (lambda y: y < RIVER_Y)
    assert not any(enemy_half(zone.y) for zone in DEPLOY_ZONES[player] if zone.allows("ground"))


def test_snapshot_sends_the_humans_pockets():
    session = GameSession("pocket", MODE_VS_BOT, PolicyInspector(model_path=None), seed=1)
    assert session.build_snapshot().pockets == []

    _destroy_princess(session.board, 1, RIGHT_LANE_X)

    (pocket,) = session.build_snapshot().pockets
    assert (pocket.x_min, pocket.x_max) == (ARENA_WIDTH / 2, ARENA_WIDTH - 1.0)
    assert (pocket.y_min, pocket.y_max) == (FAR_BANK, FAR_BANK + POCKET_DEPTH)


def test_server_accepts_a_pocket_play_only_after_the_tower_falls():
    session = GameSession("pocket-play", MODE_VS_BOT, PolicyInspector(model_path=None), seed=1)
    _ready(session.board, 0)
    y = FAR_BANK + 1.0

    assert session.handle_play("Knight", LEFT_LANE_X, y)[0].type == "error"

    _destroy_princess(session.board, 1, LEFT_LANE_X)

    assert session.handle_play("Knight", LEFT_LANE_X, y) == []
    assert session.handle_play("Knight", RIGHT_LANE_X, y)[0].type == "error"


def test_a_rematch_closes_the_pocket():
    board = new_board()
    _destroy_princess(board, 1, LEFT_LANE_X)
    assert board.pocket_zones(0)

    board.reset(seed=1)

    assert board.pocket_zones(0) == []


def test_the_rule_based_bot_never_deploys_into_a_pocket():
    board = new_board()
    _destroy_princess(board, 0, LEFT_LANE_X)  # opens a pocket for player 1 on player 0's side
    human_unit = place_unit(board, "Knight", 0, 5.0, RIVER_Y - RIVER_HALF_WIDTH - 1.0)
    assert board.pocket_zones(1)

    _, x, y = _knight_on_unit(board, 1, human_unit)

    assert y >= FAR_BANK
