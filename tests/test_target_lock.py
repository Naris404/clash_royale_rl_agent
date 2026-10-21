"""Target locking: an attacker keeps its target until it dies or leaves range."""

from __future__ import annotations

from typing import Callable

from cr_rl.game.board import TICK_DT, Board
from cr_rl.game.cards import Troop
from scenario import new_board, place_unit

BIG_HP = 5000.0


def _run(board: Board, seconds: float, stop: Callable[[], bool] = lambda: False) -> None:
    for _ in range(round(seconds / TICK_DT)):
        board.step(0, action_p1=0)
        if stop():
            return


def _run_until_hit(board: Board, victim: Troop) -> None:
    _run(board, 5.0, stop=lambda: victim.hp < BIG_HP)
    assert victim.hp < BIG_HP


def _musketeer_locked_on_giant(board: Board) -> tuple[Troop, Troop]:
    for tower in board.towers:  # keep the P1 princess towers from shooting the P0 units
        tower.alive = tower.alive and not (tower.owner == 1 and tower.name == "Tower")
    musketeer = place_unit(board, "Musketeer", 1, 9.0, 22.0)
    giant = place_unit(board, "Giant", 0, 9.0, 17.0, hp=BIG_HP)
    _run_until_hit(board, giant)
    return musketeer, giant


def _tower_locked_on_knight(board: Board) -> tuple[Troop, Troop]:
    tower = next(t for t in board.towers if t.owner == 0 and t.name == "Tower" and t.x < 9)
    locked = place_unit(board, "Knight", 1, tower.x, 14.0, hp=BIG_HP)
    _run_until_hit(board, locked)
    return tower, locked


def test_musketeer_keeps_shooting_giant_when_a_knight_appears_closer():
    board = new_board()
    musketeer, giant = _musketeer_locked_on_giant(board)
    giant_hp = giant.hp
    knight = place_unit(board, "Knight", 0, 9.0, 18.5, hp=BIG_HP)

    _run(board, 2.0)

    assert musketeer.alive
    assert giant.hp < giant_hp
    assert knight.hp == BIG_HP


def test_musketeer_picks_the_closest_enemy_once_the_giant_leaves_range():
    board = new_board()
    musketeer, giant = _musketeer_locked_on_giant(board)
    knight = place_unit(board, "Knight", 0, 9.0, 18.5, hp=BIG_HP)
    giant.place((9.0, 5.0))

    _run(board, 2.0)

    assert knight.hp < BIG_HP


def test_musketeer_picks_the_closest_enemy_once_the_giant_dies():
    board = new_board()
    musketeer, giant = _musketeer_locked_on_giant(board)
    knight = place_unit(board, "Knight", 0, 9.0, 18.5, hp=BIG_HP)
    giant.hp = 1.0

    _run(board, 3.0)

    assert not giant.alive
    assert knight.hp < BIG_HP


def test_tower_keeps_shooting_its_unit_when_a_closer_enemy_enters_range():
    board = new_board()
    tower, locked = _tower_locked_on_knight(board)
    locked_hp = locked.hp
    closer = place_unit(board, "Knight", 1, tower.x, 11.0, hp=BIG_HP)

    _run(board, 2.0)

    assert locked.hp < locked_hp
    assert closer.hp == BIG_HP


def test_tower_switches_to_the_closest_enemy_once_its_unit_leaves_range():
    board = new_board()
    tower, locked = _tower_locked_on_knight(board)
    closer = place_unit(board, "Knight", 1, tower.x, 11.0, hp=BIG_HP)
    locked.place((tower.x, 30.0))

    _run(board, 2.0)

    assert closer.hp < BIG_HP
