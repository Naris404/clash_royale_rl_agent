"""Air units, splash damage and area spells, checked at the Board seam."""

from __future__ import annotations

import pytest

from cr_rl.game.board import (
    BRIDGE_HALF_WIDTH,
    BRIDGE_LANE_X,
    DEPLOY_ZONES,
    RIVER_HALF_WIDTH,
    RIVER_Y,
    TICK_DT,
    Board,
)
from cr_rl.game.cards import Troop, cards_dic
from scenario import new_board, place_unit, set_hand

ZONE_NAMES = [z.name for z in DEPLOY_ZONES[0]]
DRAGON_DAMAGE = cards_dic["Baby_Dragon"]["damage"]


def _step_seconds(board: Board, seconds: float) -> None:
    for _ in range(round(seconds / TICK_DT)):
        board.step(0, action_p1=0)


def _step_until_hit_by(board: Board, unit: Troop, seconds: float) -> None:
    for _ in range(round(seconds / TICK_DT)):
        board.step(0, action_p1=0)
        if any((e.from_x, e.from_y) == (unit.x, unit.y) for e in board.hit_events):
            return
    pytest.fail(f"{unit.name} never attacked within {seconds} s")


@pytest.mark.parametrize("attacker", ["Knight", "Giant", "Hog_Rider", "Cannon"])
def test_ground_only_attackers_never_target_baby_dragon(attacker):
    board = new_board()
    dragon = place_unit(board, "Baby_Dragon", 0, 9.0, 15.0)
    place_unit(board, attacker, 1, 9.0, 14.5)

    _step_seconds(board, 6.0)

    assert dragon.alive and dragon.hp == dragon.max_hp


def test_musketeer_shoots_baby_dragon():
    board = new_board()
    dragon = place_unit(board, "Baby_Dragon", 0, 9.0, 12.0)
    place_unit(board, "Musketeer", 1, 9.0, 15.0)

    _step_seconds(board, 4.0)

    assert dragon.hp < dragon.max_hp


def test_tower_shoots_baby_dragon():
    board = new_board()
    dragon = place_unit(board, "Baby_Dragon", 1, 3.5, 13.0)

    _step_seconds(board, 4.0)

    assert dragon.hp < dragon.max_hp


def test_baby_dragon_flies_over_the_river_away_from_the_bridges():
    board = new_board()
    set_hand(board, 0, ["Baby_Dragon"])
    board.elixir[0] = 10.0
    assert board.play_card_at(0, "Baby_Dragon", 9.0, 12.0)
    dragon = board.troops[0]
    off_bridge_in_river = False

    for _ in range(round(8.0 / TICK_DT)):
        board.step(0, action_p1=0)
        in_river = abs(dragon.y - RIVER_Y) < RIVER_HALF_WIDTH
        off_bridge = all(abs(dragon.x - bx) > BRIDGE_HALF_WIDTH for bx in BRIDGE_LANE_X)
        off_bridge_in_river = off_bridge_in_river or (in_river and off_bridge)

    assert off_bridge_in_river
    assert dragon.y > RIVER_Y + RIVER_HALF_WIDTH


def test_baby_dragon_splash_hits_every_enemy_unit_in_radius():
    board = new_board()
    dragon = place_unit(board, "Baby_Dragon", 0, 9.0, 15.0)
    target = place_unit(board, "Knight", 1, 9.0, 18.0)
    beside = place_unit(board, "Knight", 1, 9.6, 18.8)
    far = place_unit(board, "Knight", 1, 9.0, 24.0)

    _step_until_hit_by(board, dragon, 6.0)

    assert target.hp == target.max_hp - DRAGON_DAMAGE
    assert beside.hp == beside.max_hp - DRAGON_DAMAGE
    assert far.hp == far.max_hp


def test_baby_dragon_splash_reaches_air_units_it_can_target():
    board = new_board()
    dragon = place_unit(board, "Baby_Dragon", 0, 9.0, 15.0)
    target = place_unit(board, "Knight", 1, 9.0, 18.0)
    flyer = place_unit(board, "Baby_Dragon", 1, 9.5, 18.5)

    _step_until_hit_by(board, dragon, 6.0)

    assert target.hp < target.max_hp
    assert flyer.hp == flyer.max_hp - DRAGON_DAMAGE


def test_arrows_hit_every_enemy_unit_in_radius_including_air():
    board = new_board()
    set_hand(board, 0, ["Arrows"])
    board.elixir[0] = 10.0
    knight = place_unit(board, "Knight", 1, 9.0, 20.0)
    flyer = place_unit(board, "Baby_Dragon", 1, 11.0, 21.0)
    far = place_unit(board, "Knight", 1, 9.0, 25.0)
    friend = place_unit(board, "Knight", 0, 9.5, 20.0)
    damage = cards_dic["Arrows"]["damage"]

    assert board.play_card_at(0, "Arrows", 9.0, 20.0)

    assert knight.hp == knight.max_hp - damage
    assert flyer.hp == flyer.max_hp - damage
    assert far.hp == far.max_hp
    assert friend.hp == friend.max_hp
    assert board.elixir[0] == pytest.approx(10.0 - cards_dic["Arrows"]["elisir"])


def test_arrows_are_cast_in_spell_zones_only():
    board = new_board()
    set_hand(board, 0, ["Arrows"])
    board.elixir[0] = 10.0
    assert not board.play_card(0, "Arrows", ZONE_NAMES.index("mid-L"))
    assert board.play_card(0, "Arrows", ZONE_NAMES.index("spell-king"))


def test_baby_dragon_can_be_deployed_in_every_troop_zone():
    for zone in DEPLOY_ZONES[0]:
        if zone.name.startswith("spell"):
            continue
        board = new_board()
        set_hand(board, 0, ["Baby_Dragon"])
        board.elixir[0] = 10.0
        assert board.play_card(0, "Baby_Dragon", ZONE_NAMES.index(zone.name)), zone.name


def test_same_seed_air_and_splash_fight_is_deterministic():
    def run() -> list[tuple]:
        board = new_board(seed=3)
        place_unit(board, "Baby_Dragon", 0, 9.0, 15.0)
        place_unit(board, "Knight", 1, 9.0, 18.0)
        place_unit(board, "Musketeer", 1, 10.0, 19.0)
        trace = []
        for _ in range(100):
            result = board.step(0, action_p1=0)
            trace.append((result.observation.tobytes(), len(board.hit_events)))
        return trace

    assert run() == run()
