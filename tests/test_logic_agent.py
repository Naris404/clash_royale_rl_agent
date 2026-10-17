"""LogicAgent tests — play legality, defense against Hog, combo queue."""

from __future__ import annotations

import logging

from cr_rl.game.board import RIVER_Y, Board
from cr_rl.agents.logic import LogicAgent
from cr_rl.agents.rl import RandomAgent
from scenario import new_board, place_unit, set_hand


def _decide_and_step(agent: LogicAgent, board: Board) -> list[str]:
    """One agent decision plus one Board step; returns the P0 cards that entered play."""
    before = board.get_hand(0)
    agent.choose_action(board, player=0)
    board.step(0, action_p1=0)
    return [card for card in before if card not in board.get_hand(0)]


class TestLegality:
    def test_every_decided_play_is_executed_where_chosen(self, caplog):
        board = new_board()
        agent = LogicAgent()
        with caplog.at_level(logging.WARNING, logger="cr_rl.game.board"):
            for _ in range(600):
                before = board.get_hand(0)
                agent.choose_action(board, player=0)
                board.step(0)  # P1: the Board's seeded random bot
                if agent.last_decision != "pass":
                    assert board.get_hand(0) != before, agent.last_decision
                if board.done:
                    break
        assert not caplog.records  # no illegal play fell back to the Board's default spot

    def test_a_decided_play_is_executed_once(self):
        board = new_board()
        agent = LogicAgent()
        board.elixir[0] = 10.0
        assert agent.choose_action(board, player=0) == 0  # the play goes through the Board
        board.step(0, action_p1=0)
        troops_after_play = len(board.troops)
        hand_after_play = board.get_hand(0)
        board.step(0, action_p1=0)
        assert len(board.troops) == troops_after_play
        assert board.get_hand(0) == hand_after_play


class TestDefenseRules:
    def test_reacts_to_hog_on_own_side(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Knight", "Cannon", "Musketeer", "Giant"])
        place_unit(board, "Hog_Rider", 1, 9.0, RIVER_Y - 1.5)
        assert _decide_and_step(LogicAgent(), board)[0] in ("Knight", "Cannon")

    def test_fireballs_valuable_musketeer(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"])
        musketeer = place_unit(board, "Musketeer", 1, 9.0, RIVER_Y + 1.0, hp=100.0)
        assert _decide_and_step(LogicAgent(), board) == ["Fireball"]
        assert not musketeer.alive

    def test_passes_when_nothing_legal(self):
        board = new_board()
        board.elixir[0] = 0.0  # cannot afford anything
        agent = LogicAgent()
        assert _decide_and_step(agent, board) == []
        assert agent.last_decision == "pass"


class TestComboQueue:
    @staticmethod
    def _giant_ready_board() -> Board:
        board = new_board()
        set_hand(board, 0, ["Giant", "Musketeer", "Hog_Rider", "Knight"])
        return board

    def test_giant_push_is_followed_by_musketeer_and_hog(self):
        board = self._giant_ready_board()
        agent = LogicAgent()
        played = []
        for _ in range(3):
            board.elixir[0] = 10.0
            played += _decide_and_step(agent, board)
        assert played == ["Giant", "Musketeer", "Hog_Rider"]

    def test_reset_drops_the_unfinished_combo(self):
        agent = LogicAgent()
        board = self._giant_ready_board()
        board.elixir[0] = 10.0
        assert _decide_and_step(agent, board) == ["Giant"]

        agent.reset()
        fresh = self._giant_ready_board()
        fresh.elixir[0] = 10.0
        assert _decide_and_step(agent, fresh) == ["Giant"]  # not the queued Musketeer


class TestIntegration:
    def test_full_match_vs_random_terminates(self):
        board = new_board(seed=3)
        logic = LogicAgent()
        random_bot = RandomAgent(seed=3, play_chance=0.42)
        steps = 0
        for _ in range(2500):
            logic.choose_action(board, 0)
            random_bot.choose_action(board, 1)
            result = board.step(0, 0)
            steps += 1
            if result.terminated or result.truncated:
                break
        assert board.done
        assert steps <= 1801  # limit 180 s / 0.1 s + margin
