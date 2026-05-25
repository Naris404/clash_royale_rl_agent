"""
Agent RL — na razie losowe akcje z ręki (4 sloty × 3 strefy).
W przyszłości: sieć neuronowa ucząca się na obserwacji z board.get_observation().
"""

from __future__ import annotations

import random

from board import Board, DEPLOY_ZONES, NUM_ACTIONS
from cards import cards_dic


class RLAgent:
    """
    Gracz RL: wybiera akcję 0..NUM_ACTIONS-1 (noop lub karta ze slotu ręki + strefa).

    Nie używa set_pending_play — tylko standardowe akcje w board.step().
    """

    def __init__(self, *, seed: int | None = None, play_chance: float = 0.5):
        self._rng = random.Random(seed)
        self.play_chance = play_chance

    def choose_action(self, board: Board, player: int = 0) -> int:
        board.set_pending_play(player, None)

        hand = board.get_hand(player)
        zones = len(DEPLOY_ZONES[player])
        affordable: list[int] = []

        for slot, name in enumerate(hand):
            if board.elixir[player] >= cards_dic[name]["elisir"]:
                for zone_idx in range(zones):
                    affordable.append(1 + slot * zones + zone_idx)

        if affordable and self._rng.random() < self.play_chance:
            return self._rng.choice(affordable)
        return 0

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
