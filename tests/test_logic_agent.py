"""Testy LogicAgent — legalność zagrań, obrona przed Hogiem, kolejka combo."""

from __future__ import annotations

from collections import deque

from cr_rl.game.board import ARENA_WIDTH, RIVER_Y, Board
from cr_rl.game.cards import Troop, cards_dic
from cr_rl.agents.logic import LogicAgent
from cr_rl.agents.rl import RandomAgent


def _board(seed: int = 42) -> Board:
    env = Board(seed=seed)
    env.reset(seed=seed)
    return env


def _force_hand(board: Board, player: int, hand: list[str], queue: list[str]) -> None:
    board.hand[player] = list(hand)
    board.hand_queue[player] = deque(queue)


def _add_enemy(board: Board, name: str, x: float, y: float) -> Troop:
    troop = Troop(name, 1)
    troop.place((x, y))
    board.troops.append(troop)
    return troop


class TestLegality:
    def test_pending_play_is_legal_and_affordable(self):
        board = _board()
        agent = LogicAgent()
        for _ in range(50):
            agent.choose_action(board, player=0)
            pending = board._pending_play[0]
            if pending is not None:
                card, x, y = pending
                assert board.card_in_hand(0, card)
                assert board.elixir[0] >= cards_dic[card]["elisir"]
                assert 1.0 <= x <= ARENA_WIDTH - 1.0
                if card != "Fireball":
                    assert 2.0 <= y <= RIVER_Y - 0.5
            board.step(0, action_p1=0)
            if board.done:
                break

    def test_choose_action_returns_noop_and_clears_on_step(self):
        board = _board()
        agent = LogicAgent()
        board.elixir[0] = 10.0
        assert agent.choose_action(board, player=0) == 0
        board.step(0, action_p1=0)
        assert board._pending_play[0] is None


class TestDefenseRules:
    def test_reacts_to_hog_on_own_side(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Knight", "Cannon", "Musketeer", "Giant"], ["Hog_Rider", "Fireball"])
        _add_enemy(board, "Hog_Rider", 9.0, RIVER_Y - 1.5)
        agent = LogicAgent()
        agent.choose_action(board, player=0)
        pending = board._pending_play[0]
        assert pending is not None
        assert pending[0] in ("Knight", "Cannon")

    def test_fireballs_valuable_musketeer(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Knight"])
        musk = _add_enemy(board, "Musketeer", 9.0, RIVER_Y + 1.0)
        musk.hp = 100.0  # Fireball zabija → wysoki score
        agent = LogicAgent()
        agent.choose_action(board, player=0)
        pending = board._pending_play[0]
        assert pending is not None and pending[0] == "Fireball"

    def test_passes_when_nothing_legal(self):
        board = _board()
        board.elixir[0] = 0.0  # nie stać na nic
        agent = LogicAgent()
        agent.choose_action(board, player=0)
        assert board._pending_play[0] is None
        assert agent.last_decision == "pass"


class TestComboQueue:
    def test_giant_push_schedules_followups(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Giant", "Musketeer", "Hog_Rider", "Knight"], ["Cannon", "Fireball"])
        agent = LogicAgent()
        agent.choose_action(board, player=0)
        pending = board._pending_play[0]
        assert pending is not None and pending[0] == "Giant"
        assert len(agent._combo_queue) == 2  # Musketeer + Hog zaplanowane

    def test_reset_clears_combo_queue(self):
        agent = LogicAgent()
        agent._combo_queue.append(("Knight", 9.0, 10.0))
        agent.last_decision = "x"
        agent.reset()
        assert not agent._combo_queue
        assert agent.last_decision == ""


class TestIntegration:
    def test_full_match_vs_random_terminates(self):
        board = _board(seed=3)
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
        assert steps <= 1801  # limit 180 s / 0.1 s + margines
