"""Real-game scenarios: the simulation checked against known Clash Royale outcomes.

Every scenario starts from legal plays and runs by stepping the Board. A scenario whose
outcome differs from the real game is marked xfail (strict) with the deviation, so it
starts failing loudly once a later ticket fixes the mechanic.
"""

from __future__ import annotations

from typing import Callable

import pytest

from cr_rl.game.board import DEPLOY_ZONES, TICK_DT, Board
from cr_rl.game.cards import Troop
from scenario import new_board, set_hand

ZONE_NAMES = [z.name for z in DEPLOY_ZONES[0]]


def _zone(name: str) -> int:
    return ZONE_NAMES.index(name)


def _play(board: Board, player: int, card: str, zone: str) -> Troop:
    set_hand(board, player, [card])
    board.elixir[player] = 10.0
    assert board.play_card(player, card, _zone(zone))
    return board.troops[-1]


def _run_until(board: Board, done: Callable[[], bool], seconds: float) -> None:
    for _ in range(round(seconds / TICK_DT)):
        board.step(0, action_p1=0)
        if done():
            return
    pytest.fail(f"scenario did not finish within {seconds} s")


def _princess(board: Board, owner: int, lane: str) -> Troop:
    return next(
        t for t in board.towers
        if t.owner == owner and t.name == "Tower" and (t.x < 9.0) == (lane == "L")
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Units act one after another within a tick, so the later-listed Knight starts "
        "its swing a tick earlier and wins the duel with ~222 HP left; mirror Knights "
        "should kill each other. Needs simultaneous attack resolution."
    ),
)
def test_mirror_knights_at_the_bridge_kill_each_other():
    board = new_board()
    knights = [_play(board, 0, "Knight", "bridge-L"), _play(board, 1, "Knight", "bridge-L")]

    _run_until(board, lambda: not all(k.alive for k in knights), seconds=30.0)

    assert not any(k.alive for k in knights)


@pytest.mark.parametrize("lane", ["L", "R"])
def test_cannon_in_pull_zone_pulls_hog_away_from_tower(lane):
    board = new_board()
    cannon = _play(board, 0, "Cannon", f"pull-{lane}")
    hog = _play(board, 1, "Hog_Rider", f"bridge-{lane}")

    _run_until(board, lambda: not hog.alive, seconds=20.0)

    assert cannon.hp < cannon.max_hp  # the Hog locked onto the Cannon
    tower = _princess(board, 0, lane)
    assert tower.hp == tower.max_hp  # and never reached the tower


def test_tower_kills_unsupported_hog_rider():
    board = new_board()
    hog = _play(board, 1, "Hog_Rider", "bridge-L")

    _run_until(board, lambda: not hog.alive, seconds=20.0)

    tower = _princess(board, 0, "L")
    assert tower.alive
    assert tower.hp < tower.max_hp  # the Hog connected before dying


def _fireball_fresh_musketeer(board: Board) -> Troop:
    musketeer = _play(board, 1, "Musketeer", "bridge-L")
    set_hand(board, 0, ["Fireball"])
    board.elixir[0] = 10.0
    assert board.play_card_at(0, "Fireball", musketeer.x, musketeer.y)
    return musketeer


def test_fireball_leaves_full_hp_musketeer_alive():
    board = new_board()
    musketeer = _fireball_fresh_musketeer(board)

    assert musketeer.alive
    assert musketeer.hp < musketeer.max_hp


@pytest.mark.xfail(
    strict=True,
    reason=(
        "At tournament standard a Fireball leaves the Musketeer on a sliver of HP that "
        "one princess-tower shot removes; with the current stats (Fireball 575, "
        "Musketeer 760, tower 158) it takes two. Revisit with the card data file."
    ),
)
def test_fireballed_musketeer_falls_to_one_tower_shot():
    board = new_board()
    musketeer = _fireball_fresh_musketeer(board)
    tower = _princess(board, 0, "L")
    tower_shots = 0

    def dead() -> bool:
        nonlocal tower_shots
        tower_shots += sum(
            1 for e in board.hit_events if (e.from_x, e.from_y) == (tower.x, tower.y)
        )
        return not musketeer.alive

    _run_until(board, dead, seconds=20.0)

    assert tower_shots == 1
