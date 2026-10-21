"""env-v1.1 arena geometry (ADR-0005): king towers, back-edge deploy, spell reach, bridges."""

from __future__ import annotations

import pytest

from cr_rl.agents.logic import LogicAgent
from cr_rl.game.board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    BRIDGE_LANE_X,
    RIVER_Y,
    TICK_DT,
    TOWER_LAYOUT,
    Board,
)
from cr_rl.game.cards import cards_dic
from scenario import new_board, set_hand

CENTER_X = ARENA_WIDTH / 2


def _hand_with(board: Board, player: int, card: str) -> None:
    set_hand(board, player, [card])
    board.elixir[player] = 10.0


def test_king_towers_are_4x4_with_a_free_tile_behind():
    assert TOWER_LAYOUT[0]["King_Tower"] == (9.0, 3.0)
    assert TOWER_LAYOUT[1]["King_Tower"] == (9.0, 29.0)
    assert cards_dic["King_Tower"]["collision_radius"] == 2.0
    for player, back_edge in ((0, 0.0), (1, ARENA_LENGTH)):
        king_y = TOWER_LAYOUT[player]["King_Tower"][1]
        assert abs(king_y - back_edge) - cards_dic["King_Tower"]["collision_radius"] == 1.0


def test_bridge_lanes_line_up_with_the_princess_towers():
    assert BRIDGE_LANE_X == (3.5, 14.5)
    assert BRIDGE_LANE_X == (TOWER_LAYOUT[0]["Tower_L"][0], TOWER_LAYOUT[0]["Tower_R"][0])
    assert BRIDGE_LANE_X[0] + BRIDGE_LANE_X[1] == ARENA_WIDTH


@pytest.mark.parametrize("card", ["Knight", "Cannon", "Baby_Dragon"])
def test_troops_deploy_up_to_the_back_edge(card):
    for player, back, beyond in ((0, 0.5, 0.4), (1, ARENA_LENGTH - 0.5, ARENA_LENGTH - 0.4)):
        board = new_board()
        _hand_with(board, player, card)
        assert board.can_play_card_at(player, card, 9.0, back)
        assert board.can_play_card_at(player, card, CENTER_X, TOWER_LAYOUT[player]["King_Tower"][1])
        assert not board.can_play_card_at(player, card, 9.0, beyond)


@pytest.mark.parametrize("spell", ["Fireball", "Arrows"])
def test_spells_reach_the_full_arena_length(spell):
    for player in (0, 1):
        board = new_board()
        _hand_with(board, player, spell)
        assert board.can_play_card_at(player, spell, 9.0, 0.5)
        assert board.can_play_card_at(player, spell, 9.0, ARENA_LENGTH - 0.5)
        assert not board.can_play_card_at(player, spell, 9.0, ARENA_LENGTH - 0.4)


def _lone_unit_trace(card: str, x: float, y: float, seconds: float) -> list[tuple[float, float]]:
    board = new_board()
    _hand_with(board, 0, card)
    assert board.play_card_at(0, card, x, y)
    unit = board.troops[-1]
    trace = []
    for _ in range(round(seconds / TICK_DT)):
        board.step(0, action_p1=0)
        trace.append((unit.x, unit.y))
    return trace


@pytest.mark.parametrize("card, x, y", [("Knight", 2.0, 5.0), ("Giant", 6.5, 9.0), ("Hog_Rider", 4.0, 12.0)])
def test_mirrored_placements_behave_as_mirror_images(card, x, y):
    left = _lone_unit_trace(card, x, y, seconds=20.0)
    right = _lone_unit_trace(card, ARENA_WIDTH - x, y, seconds=20.0)

    mirrored = [coord for lx, ly in left for coord in (ARENA_WIDTH - lx, ly)]
    assert mirrored == pytest.approx([coord for point in right for coord in point])


@pytest.mark.parametrize("x", [7.0, 8.5, 9.5, 11.0])
def test_a_troop_behind_the_king_takes_the_bridge_on_its_side(x):
    trace = _lone_unit_trace("Knight", x, 0.5, seconds=20.0)

    crossing_x = next(tx for tx, ty in trace if ty >= RIVER_Y - 1.0)
    assert crossing_x == pytest.approx(BRIDGE_LANE_X[0] if x < CENTER_X else BRIDGE_LANE_X[1], abs=0.3)


@pytest.mark.parametrize("player, beyond, edge", [(0, -5.0, 0.5), (1, ARENA_LENGTH + 5.0, ARENA_LENGTH - 0.5)])
def test_logic_agent_clamps_to_the_back_edge(player, beyond, edge):
    board = new_board()
    _hand_with(board, player, "Knight")

    _, x, y = LogicAgent()._clamp_play(player, ("Knight", 9.0, beyond))

    assert y == edge
    assert board.can_play_card_at(player, "Knight", x, y)
